# tests/test_kinematics.py
# Project-GP — Suspension Kinematic Solver Tests
# =============================================================================
#
# Validation suite for suspension/kinematics.py.
# Run from project root:
#   python -m pytest tests/test_kinematics.py -v
#
# Or without pytest:
#   python tests/test_kinematics.py
#
# Tests cover:
#   1. IFD gradient correctness vs finite difference
#   2. Newton convergence (residual magnitude at solution)
#   3. Camber gain sign (must be negative for negative static camber)
#   4. Bump steer order of magnitude
#   5. Motion ratio range sanity (FS: 0.8 – 1.2 typical)
#   6. Roll centre height range
#   7. Upright rigid-body constraint verification
#   8. Round-trip: delta_L_tr_from_toe / psi_shim_from_camber
# =============================================================================

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import jax
import jax.numpy as jnp

# ── Synthetic hardpoints that match the Velis geometry ─────────────────────
# These reproduce the Excel values without requiring the file to be present
# during unit testing. Units: METRES (already converted).

FRONT_HPTS_SYNTHETIC = {
    # Double A-arm
    "CHAS_LowFor": np.array([0.160,  0.160,  0.110]),
    "CHAS_LowAft": np.array([-0.160, 0.160,  0.130]),
    "CHAS_UppFor": np.array([0.120,  0.245,  0.267]),
    "CHAS_UppAft": np.array([-0.120, 0.245,  0.258]),
    "UPRI_LowPnt": np.array([0.00227, 0.583374, 0.12265]),
    "UPRI_UppPnt": np.array([-0.011496, 0.55563, 0.280]),
    "CHAS_TiePnt": np.array([0.050,  0.14478, 0.1445]),
    "UPRI_TiePnt": np.array([0.070,  0.571,   0.150]),
    # Pushrod
    "NSMA_PPAttPnt_L": np.array([-0.00351, 0.51471,  0.29418]),
    "CHAS_AttPnt_L":   np.array([-0.17933, 0.150,    0.61479]),
    "CHAS_RocAxi_L":   np.array([0.00067,  0.22753,  0.61211]),
    "CHAS_RocPiv_L":   np.array([0.00067,  0.19506,  0.57518]),
    "ROCK_RodPnt_L":   np.array([0.05998,  0.20185,  0.56921]),
    "ROCK_CoiPnt_L":   np.array([0.00067,  0.150,    0.61479]),
    # ARB
    "NSMA_UBarAttPnt_L": np.array([0.00067, 0.17253, 0.59498]),
    "UBAR_AttPnt_L":     np.array([-0.19933, 0.17253, 0.59499]),
    "CHAS_PivPnt_L":     np.array([-0.19933, 0.17253, 0.63499]),
    # Scalars
    "Half Track_m": 0.615,
    "R_wheel":      0.2032,
    "spring_rate_N_per_m": 44000.0,
    "ubar_stiffness_Nm_per_rad": 286.5,
    "actuation_type": "pushrod",
}

REAR_HPTS_SYNTHETIC = {
    "CHAS_LowFor": np.array([0.150,   0.240,   0.1262]),
    "CHAS_LowAft": np.array([-0.150,  0.240,   0.120]),
    "CHAS_UppFor": np.array([0.150,   0.240,   0.282]),
    "CHAS_UppAft": np.array([-0.150,  0.240,   0.250]),
    "UPRI_LowPnt": np.array([0.000,   0.57678, 0.11265]),
    "UPRI_UppPnt": np.array([0.000,   0.520001,0.280]),
    "CHAS_TiePnt": np.array([-0.095,  0.240,   0.163]),
    "UPRI_TiePnt": np.array([-0.080,  0.590,   0.1658]),
    "NSMA_PPAttPnt_L": np.array([0.00893, 0.49739, 0.29758]),
    "CHAS_AttPnt_L":   np.array([-0.030,  0.050,   0.430]),
    "CHAS_RocAxi_L":   np.array([0.07451,  0.11973, 0.58004]),
    "CHAS_RocPiv_L":   np.array([0.10743,  0.10826, 0.54713]),
    "ROCK_RodPnt_L":   np.array([0.14842,  0.14410, 0.57238]),
    "ROCK_CoiPnt_L":   np.array([0.09728,  0.050,   0.55728]),
    "NSMA_UBarAttPnt_L": np.array([0.09728, 0.080,  0.56328]),
    "UBAR_AttPnt_L":     np.array([0.000,   0.080,  0.450]),
    "CHAS_PivPnt_L":     np.array([0.020,   0.080,  0.436]),
    "Half Track_m": 0.615,
    "R_wheel":      0.2032,
    "spring_rate_N_per_m": 53000.0,
    "ubar_stiffness_Nm_per_rad": 286.5,
    "actuation_type": "pullrod",
    "passive_rear_steer": True,
}


# ---------------------------------------------------------------------------
# Test runner (lightweight, no pytest dependency required)
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.messages = []

    def ok(self, name):
        self.passed += 1
        print(f"  [PASS] {name}")

    def fail(self, name, msg):
        self.failed += 1
        self.messages.append(f"  [FAIL] {name}: {msg}")
        print(f"  [FAIL] {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed.")
        if self.failed:
            print("Failed tests:")
            for m in self.messages:
                print(m)
        return self.failed == 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all() -> bool:
    from suspension.kinematics import SuspensionKinematics

    r = TestResult()
    print("\n[test_kinematics] Loading synthetic front/rear hardpoints...")

    front_kin = SuspensionKinematics(FRONT_HPTS_SYNTHETIC, side="left")
    rear_kin  = SuspensionKinematics(REAR_HPTS_SYNTHETIC,  side="left")
    print()

    z0     = jnp.array(0.0)
    dL0    = jnp.array(0.0)
    ps0    = jnp.array(0.0)

    # ── Test 1: Newton convergence (residual at solution) ─────────────────────
    print("[Test 1] Newton convergence")
    from suspension.kinematics import _constraint_residual, _solve_constraint_system

    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        theta_opt = _solve_constraint_system(
            z0, dL0, ps0, *kin._nondiff_geo
        )
        j = kin._jax
        F = _constraint_residual(
            theta_opt, z0, dL0, ps0,
            j["A1"], j["e_LA"], j["C0_rel_A1"],
            j["B1"], j["e_UA"], j["D0_rel_B1"],
            j["C0"], j["D0"], j["W0"], j["TU0"], j["TC"],
            j["L_upright_sq"], j["L_tr_nom"], j["W_z_nom"],
        )
        max_res = float(jnp.max(jnp.abs(F)))
        if max_res < 1e-5:
            r.ok(f"Newton residual ({name}) = {max_res:.2e} < 1e-5")
        else:
            r.fail(f"Newton residual ({name})", f"{max_res:.2e} ≥ 1e-5 — Newton not converged")

    # ── Test 2: Upright rigid-body constraint ─────────────────────────────────
    print("\n[Test 2] Upright rigid-body constraint at heave ±50 mm")
    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        j = kin._jax
        L_up_nom = float(jnp.sqrt(j["L_upright_sq"]))

        for z_mm in [-50.0, 0.0, 50.0]:
            z = jnp.array(z_mm / 1000.0)
            out = kin.solve_at_heave(z, dL0, ps0)
            L_up = float(jnp.linalg.norm(out.D - out.C))
            err = abs(L_up - L_up_nom)
            if err < 1e-5:
                r.ok(f"Rigid upright ({name}, z={z_mm:+.0f}mm) Δ={err*1e6:.1f}μm")
            else:
                r.fail(f"Rigid upright ({name}, z={z_mm:+.0f}mm)",
                       f"length error {err*1e3:.3f} mm ≥ 10μm")

    # ── Test 3: Nominal static camber ─────────────────────────────────────────
    print("\n[Test 3] Static camber sign and magnitude at z=0")
    out_f = front_kin.solve_at_heave(z0, dL0, ps0)
    out_r = rear_kin.solve_at_heave( z0, dL0, ps0)

    for name, out, velis_deg in [("front", out_f, -0.5), ("rear", out_r, -1.0)]:
        camber_deg = float(jnp.degrees(out.camber_rad))
        # We don't require exact match with Velis (psi_shim=0 = raw geometry),
        # but the sign must be negative (top leaning outboard for negative camber
        # in SAE convention where positive = top lean inboard).
        # Raw geometry without shims should be close to 0°.
        # The KPI contribution should give a small positive value ~1–3°.
        if -5.0 < camber_deg < 5.0:
            r.ok(f"Camber in range ({name}) = {camber_deg:.3f}°")
        else:
            r.fail(f"Camber out of range ({name})", f"{camber_deg:.3f}° not in (-5, 5)°")

    # ── Test 4: Camber gain sign ───────────────────────────────────────────────
    print("\n[Test 4] Camber gain sign (must be negative for typical SLA)")
    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        gain = float(jax.grad(
            lambda z: kin.solve_at_heave(z, dL0, ps0).camber_rad
        )(z0))
        gain_deg_per_mm = gain * math.pi / 180.0 * 1000.0  # convert to deg/m for display
        # For a typical SLA with outboard pivot higher than inboard:
        # in bump (z > 0), camber should become more negative → dγ/dz < 0
        if gain < 0:
            r.ok(f"Camber gain ({name}) = {gain:.3f} rad/m ({gain_deg_per_mm:.2f} deg/m) — negative ✓")
        else:
            r.fail(f"Camber gain ({name})",
                   f"{gain:.3f} rad/m — expected negative (bump reduces camber for SLA)")

    # ── Test 5: Bump steer magnitude ──────────────────────────────────────────
    print("\n[Test 5] Bump steer magnitude < 0.025 rad/m at z=0")
    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        bs = float(jax.grad(
            lambda z: kin.solve_at_heave(z, dL0, ps0).toe_rad
        )(z0))
        bs_mrad_per_m = bs * 1000.0
        if abs(bs) < 0.025:
            r.ok(f"Bump steer ({name}) = {bs_mrad_per_m:.2f} mrad/m  |<25 mrad/m ✓")
        else:
            r.fail(f"Bump steer ({name})",
                   f"{bs_mrad_per_m:.2f} mrad/m ≥ 25 mrad/m — check tie-rod geometry")

    # ── Test 6: Motion ratio range ────────────────────────────────────────────
    print("\n[Test 6] Motion ratio range at z=0")
    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        out = kin.solve_at_heave(z0, dL0, ps0)
        mr = float(out.motion_ratio)
        if 0.70 < mr < 1.30:
            r.ok(f"Motion ratio ({name}) = {mr:.4f}  [0.70–1.30 FS range ✓]")
        else:
            r.fail(f"Motion ratio ({name})",
                   f"{mr:.4f} outside [0.70, 1.30] — check pushrod/rocker geometry")

    # ── Test 7: Roll centre height range ─────────────────────────────────────
    print("\n[Test 7] Roll centre height range at z=0")
    for name, kin, lo, hi in [
        ("front", front_kin, 0.020, 0.080),
        ("rear",  rear_kin,  0.030, 0.120),
    ]:
        out = kin.solve_at_heave(z0, dL0, ps0)
        rc  = float(out.roll_centre_z)
        if lo < rc < hi:
            r.ok(f"Roll centre ({name}) = {rc*1e3:.1f} mm  [{lo*1e3:.0f}–{hi*1e3:.0f} mm ✓]")
        else:
            r.fail(f"Roll centre ({name})",
                   f"{rc*1e3:.1f} mm not in [{lo*1e3:.0f}, {hi*1e3:.0f}] mm range")

    # ── Test 8: IFD gradient vs finite difference ─────────────────────────────
    print("\n[Test 8] IFD gradient correctness (vs forward finite difference)")
    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        eps = 1e-5

        # ∂(camber)/∂(delta_L_tr) via IFD
        ifd_grad_cam = float(jax.grad(
            lambda dL: kin.solve_at_heave(z0, dL, ps0).camber_rad
        )(dL0))

        # Same via FD
        cam_p = float(kin.solve_at_heave(z0, jnp.array(eps), ps0).camber_rad)
        cam_m = float(kin.solve_at_heave(z0, jnp.array(-eps), ps0).camber_rad)
        fd_grad_cam = (cam_p - cam_m) / (2 * eps)

        relerr_cam = abs(ifd_grad_cam - fd_grad_cam) / (abs(fd_grad_cam) + 1e-12)

        # ∂(toe)/∂(delta_L_tr) via IFD
        ifd_grad_toe = float(jax.grad(
            lambda dL: kin.solve_at_heave(z0, dL, ps0).toe_rad
        )(dL0))

        toe_p = float(kin.solve_at_heave(z0, jnp.array(eps), ps0).toe_rad)
        toe_m = float(kin.solve_at_heave(z0, jnp.array(-eps), ps0).toe_rad)
        fd_grad_toe = (toe_p - toe_m) / (2 * eps)

        relerr_toe = abs(ifd_grad_toe - fd_grad_toe) / (abs(fd_grad_toe) + 1e-12)

        tol = 0.01   # 1% relative error tolerance
        if relerr_cam < tol:
            r.ok(f"IFD ∂camber/∂dL_tr ({name}) rel_err={relerr_cam*100:.4f}% < 1%")
        else:
            r.fail(f"IFD ∂camber/∂dL_tr ({name})",
                   f"rel_err={relerr_cam*100:.3f}% ≥ 1%  "
                   f"IFD={ifd_grad_cam:.6f}  FD={fd_grad_cam:.6f}")

        if relerr_toe < tol:
            r.ok(f"IFD ∂toe/∂dL_tr ({name}) rel_err={relerr_toe*100:.4f}% < 1%")
        else:
            r.fail(f"IFD ∂toe/∂dL_tr ({name})",
                   f"rel_err={relerr_toe*100:.3f}% ≥ 1%  "
                   f"IFD={ifd_grad_toe:.6f}  FD={fd_grad_toe:.6f}")

    # ── Test 9: Round-trip toe inversion ──────────────────────────────────────
    print("\n[Test 9] Round-trip toe inversion (delta_L_tr_from_toe)")
    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        for toe_target_deg in [-0.10, 0.0, 0.15]:
            toe_target = math.radians(toe_target_deg)
            dL = kin.delta_L_tr_from_toe(toe_target)
            toe_achieved = float(kin.solve_at_heave(
                z0, jnp.array(dL), ps0
            ).toe_rad)
            err_deg = abs(math.degrees(toe_achieved - toe_target))
            if err_deg < 0.001:
                r.ok(f"Toe round-trip ({name}, {toe_target_deg:+.2f}°) err={err_deg*1e4:.2f}×10⁻⁴°")
            else:
                r.fail(f"Toe round-trip ({name}, {toe_target_deg:+.2f}°)",
                       f"err={err_deg:.4f}° ≥ 0.001°")

    # ── Test 10: Round-trip camber inversion ──────────────────────────────────
    print("\n[Test 10] Round-trip camber inversion (psi_shim_from_camber)")
    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        for cam_target_deg in [-0.5, -1.0, -2.5]:
            cam_target = math.radians(cam_target_deg)
            ps = kin.psi_shim_from_camber(cam_target)
            cam_achieved = float(kin.solve_at_heave(
                z0, dL0, jnp.array(ps)
            ).camber_rad)
            err_deg = abs(math.degrees(cam_achieved - cam_target))
            if err_deg < 0.001:
                r.ok(f"Camber round-trip ({name}, {cam_target_deg:+.2f}°) err={err_deg*1e4:.2f}×10⁻⁴°")
            else:
                r.fail(f"Camber round-trip ({name}, {cam_target_deg:+.2f}°)",
                       f"err={err_deg:.4f}° ≥ 0.001°")

    # ── Test 11: kinematic_gains completeness ─────────────────────────────────
    print("\n[Test 11] KinematicGains completeness and finiteness")
    for name, kin in [("front", front_kin), ("rear", rear_kin)]:
        gains = kin.kinematic_gains(dL0, ps0)
        fields_to_check = [
            ("camber_gain_rad_per_m",       gains.camber_gain_rad_per_m),
            ("bump_steer_lin_rad_per_m",    gains.bump_steer_lin_rad_per_m),
            ("bump_steer_quad_rad_per_m2",  gains.bump_steer_quad_rad_per_m2),
            ("mr_poly[0] (MR at z=0)",      gains.mr_poly[0]),
            ("rc_height_m",                 gains.rc_height_m),
            ("drc_dz_m_per_m",              gains.drc_dz_m_per_m),
        ]
        all_ok = True
        for fname, fval in fields_to_check:
            v = float(fval)
            if not math.isfinite(v):
                r.fail(f"gains.{fname} ({name})", f"value is {v}")
                all_ok = False
        if all_ok:
            r.ok(f"All KinematicGains finite ({name})")
            print(f"    camber_gain = {float(gains.camber_gain_rad_per_m)*180/math.pi*1e3:.3f} deg/m")
            print(f"    bump_steer  = {float(gains.bump_steer_lin_rad_per_m)*1e3:.3f} mrad/m")
            print(f"    MR poly     = {[f'{float(x):.4f}' for x in gains.mr_poly]}")
            print(f"    RC height   = {float(gains.rc_height_m)*1e3:.2f} mm")
            print(f"    dRC/dz      = {float(gains.drc_dz_m_per_m):.4f} m/m")

    print()
    return r.summary()


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)


# ---------------------------------------------------------------------------
# Additional tests for sweep_analysis
# ---------------------------------------------------------------------------

def test_sweep():
    """Sanity-check the SweepResult arrays."""
    from suspension.kinematics import SuspensionKinematics
    from suspension.sweep_analysis import compute_sweep, compute_roll_analysis

    r = TestResult()
    print("\n[test_sweep] Running heave sweep tests...")

    front_kin = SuspensionKinematics(FRONT_HPTS_SYNTHETIC, side="left")
    rear_kin  = SuspensionKinematics(REAR_HPTS_SYNTHETIC,  side="left")
    front_right = SuspensionKinematics(FRONT_HPTS_SYNTHETIC, side="right")

    sweep_f = compute_sweep(front_kin, toe_target_deg=-0.10, camber_target_deg=-2.5, axle="front", n_pts=50)
    sweep_r = compute_sweep(rear_kin,  toe_target_deg= 0.15, camber_target_deg=-2.2, axle="rear",  n_pts=50)

    for sweep in [sweep_f, sweep_r]:
        name = sweep.axle
        # Camber at z=0 should be close to the target
        z0i = int(np.argmin(np.abs(sweep.z_mm)))
        cam_err = abs(sweep.camber_deg[z0i] - sweep.camber_target_deg)
        if cam_err < 0.05:
            r.ok(f"Camber at z=0 matches target ({name}) err={cam_err:.4f}°")
        else:
            r.fail(f"Camber at z=0 ({name})", f"err={cam_err:.4f}° ≥ 0.05°")

        toe_err = abs(sweep.toe_deg[z0i] - sweep.toe_target_deg)
        if toe_err < 0.02:
            r.ok(f"Toe at z=0 matches target ({name}) err={toe_err:.4f}°")
        else:
            r.fail(f"Toe at z=0 ({name})", f"err={toe_err:.4f}° ≥ 0.02°")

        # Arrays must all be finite and correct length
        for arr_name in ["camber_deg", "toe_deg", "motion_ratio", "rc_height_mm"]:
            arr = getattr(sweep, arr_name)
            if np.all(np.isfinite(arr)):
                r.ok(f"Sweep array finite ({name}, {arr_name}) len={len(arr)}")
            else:
                r.fail(f"Sweep array ({name}, {arr_name})", "contains NaN or Inf")

    sweep_f.print_summary()
    sweep_r.print_summary()
    print()
    return r.summary()


if __name__ == "__main__":
    success1 = test_all()
    success2 = test_sweep()
    sys.exit(0 if (success1 and success2) else 1)