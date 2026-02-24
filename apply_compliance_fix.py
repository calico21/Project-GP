"""
fix_ordering_bug.py
Moves the alpha_t / kappa_t state extraction to BEFORE the compliance steer
block that uses those variables.

Usage:
    python fix_ordering_bug.py models/vehicle_dynamics.py
"""
import sys

# ── The compliance steer block (section 6) — must come AFTER alpha_t is defined
COMPLIANCE_BLOCK = \
"""        # ── Compliance Steer (§2.6) ──────────────────────────────────────────────
        # Previous code approximated Fy with a hardcoded -50000 N/rad stiffness
        # which ignored load sensitivity and had incorrect sign convention.
        # Fix: estimate Fy from the previous timestep's transient slip angle
        # (alpha_t is already in the state vector) with load-dependent Ky from MF6.2.
        C_cs_f = jnp.deg2rad(self.vp.get('compliance_steer_f', -0.15)) / 1000.0  # rad/N
        C_cs_r = jnp.deg2rad(self.vp.get('compliance_steer_r', -0.10)) / 1000.0

        # Load-dependent cornering stiffness: Ky(Fz) = PKY1·Fz0·sin(PKY4·arctan(Fz/(PKY2·Fz0)))
        _Fz0_tc = self.tire.coeffs.get('FNOMIN', 1000.0)
        _PKY1   = self.tire.coeffs.get('PKY1',  15.324)
        _PKY2   = self.tire.coeffs.get('PKY2',   1.715)
        _PKY4   = self.tire.coeffs.get('PKY4',   2.0  )

        def _Ky(Fz_c):
            return _PKY1 * _Fz0_tc * jnp.sin(
                _PKY4 * jnp.arctan(Fz_c / jnp.maximum(_PKY2 * _Fz0_tc, 1e-6))
            )

        # Fy estimate from previous step's transient slip angles (already in state)
        Fy_prev_fl = _Ky(Fz_fl) * alpha_t_fl
        Fy_prev_fr = _Ky(Fz_fr) * alpha_t_fr
        Fy_prev_rl = _Ky(Fz_rl) * alpha_t_rl
        Fy_prev_rr = _Ky(Fz_rr) * alpha_t_rr

        alpha_ss_fl = jnp.clip(alpha_kin_fl + C_cs_f * Fy_prev_fl, -1.5, 1.5)
        alpha_ss_fr = jnp.clip(alpha_kin_fr + C_cs_f * Fy_prev_fr, -1.5, 1.5)
        alpha_ss_rl = jnp.clip(alpha_kin_rl + C_cs_r * Fy_prev_rl, -1.5, 1.5)
        alpha_ss_rr = jnp.clip(alpha_kin_rr + C_cs_r * Fy_prev_rr, -1.5, 1.5)
        kappa_ss_f = 0.0"""

# ── Section 7 header — alpha_t extraction lives here currently
OLD_SECTION_7 = \
"""        # 7. Transient Slip Updates
        alpha_t_fl, kappa_t_fl, alpha_t_fr, kappa_t_fr = x[38], x[39], x[40], x[41]
        alpha_t_rl, kappa_t_rl, alpha_t_rr, kappa_t_rr = x[42], x[43], x[44], x[45]"""

# ── What we replace section 7 with: extraction first, THEN compliance, THEN rest
NEW_SECTION_7 = \
"""        # 7. Transient Slip Updates
        # Extract alpha_t / kappa_t BEFORE the compliance steer block below uses them.
        alpha_t_fl, kappa_t_fl, alpha_t_fr, kappa_t_fr = x[38], x[39], x[40], x[41]
        alpha_t_rl, kappa_t_rl, alpha_t_rr, kappa_t_rr = x[42], x[43], x[44], x[45]"""

# ── The FULL corrected section 6→7 splice ──────────────────────────────────────
# We want the final file to read:
#   [alpha_kin lines]
#   [alpha_t extraction]      ← moved up
#   [compliance steer block]  ← uses alpha_t
#   kappa_ss_f = 0.0
#   [transient derivative calls]

OLD_SPLICE = COMPLIANCE_BLOCK + "\n\n" + OLD_SECTION_7

NEW_SPLICE = \
"""        # 7. Transient Slip state extraction
        # Moved ABOVE compliance steer so alpha_t_* are defined before use.
        alpha_t_fl, kappa_t_fl, alpha_t_fr, kappa_t_fr = x[38], x[39], x[40], x[41]
        alpha_t_rl, kappa_t_rl, alpha_t_rr, kappa_t_rr = x[42], x[43], x[44], x[45]

        # ── Compliance Steer (§2.6) ──────────────────────────────────────────────
        # Uses alpha_t from the state vector (previous timestep) to avoid
        # circular dependency. Ky is load-dependent via MF6.2 PKY coefficients.
        C_cs_f = jnp.deg2rad(self.vp.get('compliance_steer_f', -0.15)) / 1000.0  # rad/N
        C_cs_r = jnp.deg2rad(self.vp.get('compliance_steer_r', -0.10)) / 1000.0

        _Fz0_tc = self.tire.coeffs.get('FNOMIN', 1000.0)
        _PKY1   = self.tire.coeffs.get('PKY1',  15.324)
        _PKY2   = self.tire.coeffs.get('PKY2',   1.715)
        _PKY4   = self.tire.coeffs.get('PKY4',   2.0  )

        def _Ky(Fz_c):
            return _PKY1 * _Fz0_tc * jnp.sin(
                _PKY4 * jnp.arctan(Fz_c / jnp.maximum(_PKY2 * _Fz0_tc, 1e-6))
            )

        Fy_prev_fl = _Ky(Fz_fl) * alpha_t_fl
        Fy_prev_fr = _Ky(Fz_fr) * alpha_t_fr
        Fy_prev_rl = _Ky(Fz_rl) * alpha_t_rl
        Fy_prev_rr = _Ky(Fz_rr) * alpha_t_rr

        alpha_ss_fl = jnp.clip(alpha_kin_fl + C_cs_f * Fy_prev_fl, -1.5, 1.5)
        alpha_ss_fr = jnp.clip(alpha_kin_fr + C_cs_f * Fy_prev_fr, -1.5, 1.5)
        alpha_ss_rl = jnp.clip(alpha_kin_rl + C_cs_r * Fy_prev_rl, -1.5, 1.5)
        alpha_ss_rr = jnp.clip(alpha_kin_rr + C_cs_r * Fy_prev_rr, -1.5, 1.5)
        kappa_ss_f = 0.0"""


def patch(path):
    with open(path, 'r') as f:
        src = f.read()

    if OLD_SPLICE not in src:
        print("ERROR: expected splice block not found.")
        print("Check that vehicle_dynamics.py has the compliance fix from the previous step.")
        print("The pattern must be the compliance block immediately followed by section 7.")
        return False

    patched = src.replace(OLD_SPLICE, NEW_SPLICE, 1)

    with open(path, 'w') as f:
        f.write(patched)

    print(f"[OK] Ordering bug fixed in {path}")
    print("     alpha_t extraction now precedes compliance steer block.")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_ordering_bug.py models/vehicle_dynamics.py")
        sys.exit(0)
    ok = patch(sys.argv[1])
    sys.exit(0 if ok else 1)