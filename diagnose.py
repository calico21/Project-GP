"""
diagnose.py — Run this BEFORE the full test suite to isolate exactly which
component is failing. Each step is independent; a failure in one won't block
the others.

Usage:
    python diagnose.py
"""

import traceback
import sys

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def run(label, fn):
    try:
        result = fn()
        print(f"{PASS} {label}" + (f" → {result}" if result is not None else ""))
        return True
    except Exception as e:
        print(f"{FAIL} {label}")
        traceback.print_exc()
        return False


print("\n=== Project-GP Diagnostic ===\n")

# ─── Step 1: Imports ─────────────────────────────────────────────────────────
ok_jax = run("import jax + jnp", lambda: __import__('jax'))
ok_tc  = run("import tire_coeffs",
             lambda: __import__('data.configs.tire_coeffs', fromlist=['tire_coeffs']))
ok_vp  = run("import vehicle_params",
             lambda: __import__('data.configs.vehicle_params', fromlist=['vehicle_params']))
ok_tm  = run("import PacejkaTire",
             lambda: __import__('models.tire_model', fromlist=['PacejkaTire']))

if not all([ok_jax, ok_tc, ok_vp, ok_tm]):
    print("\nFix imports before continuing.")
    sys.exit(1)

import jax
import jax.numpy as jnp
from data.configs.tire_coeffs import tire_coeffs
from data.configs.vehicle_params import vehicle_params as VP
from models.tire_model import PacejkaTire

# ─── Step 2: PacejkaTire construction ────────────────────────────────────────
run("PacejkaTire.__init__", lambda: PacejkaTire(tire_coeffs, rng_seed=0))

# ─── Step 3: jnp.clip API (a_min/a_max vs min/max) ───────────────────────────
def check_clip():
    x = jnp.array(400.0)
    # Old API: a_min / a_max (deprecated in NumPy 2.0 / newer JAX)
    try:
        jnp.clip(x, a_min=0.0, a_max=350.0)
        return "a_min/a_max kwargs: OK"
    except TypeError:
        # New API
        jnp.clip(x, 0.0, 350.0)
        return "a_min/a_max kwargs BROKEN — positional args work (fix needed in tire_model.py)"

run("jnp.clip API", check_clip)

# ─── Step 4: compute_flash_temperature ───────────────────────────────────────
tire = PacejkaTire(tire_coeffs, rng_seed=0)
run("compute_flash_temperature",
    lambda: tire.compute_flash_temperature(1.5, 1000.0, 5.0))

# ─── Step 5: compute_force (pure lateral, zero kappa) ────────────────────────
T_r = jnp.array([90., 90., 90.])
run("compute_force (α=5°, κ=0)",
    lambda: tire.compute_force(jnp.deg2rad(5.), 0.0, 1000., 0., T_r, 90., 15.))

# ─── Step 6: compute_force (combined slip) ───────────────────────────────────
run("compute_force (α=8°, κ=-0.15)",
    lambda: tire.compute_force(jnp.deg2rad(8.), -0.15, 1000., 0., T_r, 90., 15.))

# ─── Step 7: compute_aligning_torque ─────────────────────────────────────────
def check_mz():
    Fx, Fy = tire.compute_force(jnp.deg2rad(5.), 0.0, 1000., 0., T_r, 90., 15.)
    Mz = tire.compute_aligning_torque(jnp.deg2rad(5.), 0.0, 1000., 0., Fy)
    return f"Mz = {float(Mz):.2f} N·m"
run("compute_aligning_torque", check_mz)

# ─── Step 8: compute_transient_slip_derivatives ───────────────────────────────
run("compute_transient_slip_derivatives",
    lambda: tire.compute_transient_slip_derivatives(
        jnp.deg2rad(5.), 0.0, jnp.deg2rad(3.), 0.0, 1000., 15.))

# ─── Step 9: compute_thermal_derivatives ─────────────────────────────────────
def check_thermal():
    T_state = jnp.array([90., 90., 90., 85.])   # 4-node: 3 ribs + core
    return tire.compute_thermal_derivatives(T_state, 90.0, 500.0, 3.0)
run("compute_thermal_derivatives", check_thermal)

# ─── Step 10: PINN/GP operator directly ──────────────────────────────────────
def check_pinn():
    state = jnp.array([jnp.deg2rad(5.), 0.0, 0.0, 1000., 15.])
    mods, sigma = tire.operator.apply(state, stochastic_key=None)
    mods_clipped = jnp.clip(mods, -0.25, 0.25)
    return f"mods={[float(m) for m in mods_clipped]}, sigma={float(sigma):.4f}"
run("PINN/GP operator + clip", check_pinn)

# ─── Step 11: Friction circle sanity values ───────────────────────────────────
def check_fc():
    import math
    Fx_p, Fy_p = tire.compute_force(jnp.deg2rad(8.), 0.0,   1000., 0., T_r, 90., 15.)
    Fx_c, Fy_c = tire.compute_force(jnp.deg2rad(8.), -0.15, 1000., 0., T_r, 90., 15.)
    F_res_p = math.sqrt(float(Fx_p)**2 + float(Fy_p)**2)
    F_res_c = math.sqrt(float(Fx_c)**2 + float(Fy_c)**2)
    mu_est  = tire_coeffs.get('PDY1', 2.218) * 1.05
    ratio_c = F_res_c / (mu_est * 1000.0 + 1e-6)
    reduction = (1.0 - float(Fy_c)/float(Fy_p)) * 100.0
    return (f"Fy_pure={float(Fy_p):.0f}N, Fy_comb={float(Fy_c):.0f}N, "
            f"reduction={reduction:.1f}%, friction circle ratio={ratio_c:.3f}")
run("Friction circle values", check_fc)

# ─── Step 12: vehicle_dynamics import ────────────────────────────────────────
ok_vd = run("import DifferentiableMultiBodyVehicle",
            lambda: __import__('models.vehicle_dynamics',
                               fromlist=['DifferentiableMultiBodyVehicle']))

# ─── Step 13: DifferentiableMultiBodyVehicle construction ────────────────────
if ok_vd:
    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    run("DifferentiableMultiBodyVehicle.__init__",
        lambda: DifferentiableMultiBodyVehicle(VP, tire_coeffs, rng_seed=0))

# ─── Step 14: objectives import ──────────────────────────────────────────────
run("import compute_skidpad_objective",
    lambda: __import__('optimization.objectives', fromlist=['compute_skidpad_objective']))

print("\n=== Diagnostic complete — fix any FAIL items above ===\n")