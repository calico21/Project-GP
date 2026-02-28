# ═══════════════════════════════════════════════════════════════════════════════
# PATCH for vehicle_dynamics.py — BUG 3 FIX (h_scale de-normalisation)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM
# ───────
# H_net is trained on targets divided by h_scale ≈ 103.31 J, so its output is
# in dimensionless normalised units (~1.0), not physical Joules.
# vehicle_dynamics.py adds H_net's output directly to the Hamiltonian without
# multiplying by h_scale, making the neural correction ~100× too small.
# This is why "Speed: 10.000 → 10.002" persists even after the passivity fix —
# H_net is having negligible effect in both directions.
#
# TWO CHANGES REQUIRED
# ─────────────────────
# Change 1: Load h_scale from disk in __init__
# Change 2: Multiply H_net output by self.h_scale in _compute_derivatives (or __call__)
#
# ───────────────────────────────────────────────────────────────────────────────
# CHANGE 1 — In DifferentiableMultiBodyVehicle.__init__
# ───────────────────────────────────────────────────────────────────────────────
# Find the block that loads h_net.bytes (looks roughly like this):
#
#   h_net_path = os.path.join(current_dir, 'h_net.bytes')
#   if os.path.exists(h_net_path):
#       with open(h_net_path, 'rb') as f:
#           self.h_params = flax.serialization.from_bytes(self.h_params, f.read())
#
# Immediately AFTER that block, add:

    h_scale_path = os.path.join(current_dir, 'h_net_scale.txt')
    if os.path.exists(h_scale_path):
        with open(h_scale_path) as f:
            self.h_scale = float(f.read().strip())
        print(f"[VehicleDynamics] H_net scale loaded: {self.h_scale:.2f} J")
    else:
        self.h_scale = 1.0
        print("[VehicleDynamics] WARNING: h_net_scale.txt not found — "
              "H_net output will not be de-normalised. "
              "Run `python residual_fitting.py` or `python main.py --mode pretrain` first.")

# ───────────────────────────────────────────────────────────────────────────────
# CHANGE 2 — In NeuralEnergyLandscape.__call__ (or wherever H_net is applied)
# ───────────────────────────────────────────────────────────────────────────────
# Find the line that adds H_residual to the Hamiltonian.  It will look like one
# of these patterns:
#
#   Pattern A (in __call__):
#       return T_prior + V_structural + H_residual
#
#   Pattern B (in _compute_derivatives, inlined):
#       H_total = h_net.apply(self.h_params, q, p, setup)
#       ... uses H_total directly ...
#
# Replace the H_residual term with H_residual * self.h_scale:
#
#   Pattern A fix:
#       H_residual_physical = H_residual * self.h_scale
#       return T_prior + V_structural + H_residual_physical
#
#   Pattern B fix:
#       H_total = h_net.apply(self.h_params, q, p, setup) * self.h_scale
#
# If NeuralEnergyLandscape is a standalone Flax module (not a method of
# DifferentiableMultiBodyVehicle), pass h_scale as a constructor argument
# or multiply at the call site:
#
#   h_out = self.h_net.apply(self.h_params, q, p, setup) * self.h_scale
#
# ───────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ───────────────────────────────────────────────────────────────────────────────
# After applying both changes, re-run sanity_checks.py.
# Test 2 should print something like:
#   > Speed changed: 10.000 m/s -> 9.998 m/s   (deceleration — energy is passive)
# instead of the current:
#   > Speed changed: 10.000 m/s -> 10.002 m/s  (injection — passivity not applied)
#
# If you still see injection after the fix, confirm h_net_scale.txt exists in
# the models/ directory and contains "103.31" (approximately).