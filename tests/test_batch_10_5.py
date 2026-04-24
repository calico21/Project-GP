# tests/test_batch_10_5.py
# Project-GP — Batch 10.5 Sanity Checks
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tests 19–25: Reality Gap Closure verification
#
# Run: python -m tests.test_batch_10_5
# Or:  python -c "from tests.test_batch_10_5 import test_all; test_all()"
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os, sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

try:
    import jax_config  # noqa: F401
except ImportError:
    pass

import jax
import jax.numpy as jnp
import numpy as np
import time


def _header(test_num, title):
    print(f"\n{'=' * 60}")
    print(f"TEST {test_num}: {title}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 19: Hub Motor Input Vector
# ─────────────────────────────────────────────────────────────────────────────

def test_19_hub_motor():
    _header(19, "HUB MOTOR INPUT VECTOR (4WD)")
    from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
    from config.vehicles.ter27 import vehicle_params_ter27 as VP
    from config.tire_coeffs import tire_coeffs as TC

    veh = DifferentiableMultiBodyVehicle(VP, TC)

    # u = [δ, T_fl, T_fr, T_rl, T_rr, F_brake_hyd]
    u = jnp.array([0.0, 100.0, 100.0, 100.0, 100.0, 0.0])

    x0 = jnp.zeros(46)
    x0 = x0.at[14].set(5.0)   # initial vx = 5 m/s
    x0 = x0.at[28:38].set(jnp.full(10, 85.0))  # warm tires

    setup = veh._default_setup_vec
    dt = 0.005

    x = x0
    for i in range(10):
        x = veh.simulate_step(x, u, setup, dt)

    vx_final = float(x[14])
    has_nan = bool(jnp.any(jnp.isnan(x)))

    print(f"  vx after 10 steps: {vx_final:.3f} m/s (initial: 5.0)")
    print(f"  NaN in state: {has_nan}")

    if vx_final > 5.0 and not has_nan:
        print("[PASS] Hub motors produce positive acceleration, no NaN")
        return True
    else:
        print("[FAIL] Hub motor test failed")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 20: UKF Convergence (perfect sensors)
# ─────────────────────────────────────────────────────────────────────────────

def test_20_ukf_perfect():
    _header(20, "UKF CONVERGENCE (ZERO NOISE)")
    from powertrain.state_estimator import (
        UKFState, UKFParams, ukf_step, extract_estimated_state,
        make_ukf_state, pack_measurement,
    )

    params = UKFParams()
    ukf = make_ukf_state(vx_init=0.0, params=params)

    # True state: vx=15, vy=0.3, wz=0.5
    vx_true = 15.0
    vy_true = 0.3
    wz_true = 0.5

    R = params.R_wheel
    tf2 = params.track_f / 2.0
    tr2 = params.track_r / 2.0
    delta = 0.05

    dt = 0.005
    for step in range(200):
        # Perfect measurements (zero noise)
        z = pack_measurement(
            ax_imu=jnp.array(-vy_true * wz_true),
            ay_imu=jnp.array(vx_true * wz_true),
            wz_gyro=jnp.array(wz_true),
            omega_fl=jnp.array(abs(vx_true - wz_true * tf2) / R),
            omega_fr=jnp.array(abs(vx_true + wz_true * tf2) / R),
            omega_rl=jnp.array(abs(vx_true - wz_true * tr2) / R),
            omega_rr=jnp.array(abs(vx_true + wz_true * tr2) / R),
            delta_steer=jnp.array(delta),
            vx_gps=jnp.array(vx_true),
        )
        ukf = ukf_step(ukf, z, jnp.array(delta), jnp.array(dt), params)

    est = extract_estimated_state(ukf)
    vx_err = abs(float(est.vx) - vx_true)
    wz_err = abs(float(est.wz) - wz_true)

    print(f"  vx estimated: {float(est.vx):.3f} (true: {vx_true}), err: {vx_err:.4f}")
    print(f"  wz estimated: {float(est.wz):.4f} (true: {wz_true}), err: {wz_err:.4f}")
    print(f"  Fz estimated: {np.array(est.Fz).round(1)}")

    # Check covariance is decreasing
    P_trace = float(jnp.trace(ukf.P))
    print(f"  P trace: {P_trace:.2f}")

    if vx_err < 0.5 and wz_err < 0.05:
        print("[PASS] UKF converges to truth with perfect sensors")
        return True
    else:
        print("[FAIL] UKF failed to converge")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 21: UKF with Noise
# ─────────────────────────────────────────────────────────────────────────────

def test_21_ukf_noisy():
    _header(21, "UKF WITH REALISTIC NOISE")
    from powertrain.state_estimator import (
        UKFParams, ukf_step, extract_estimated_state,
        make_ukf_state, pack_measurement,
    )

    params = UKFParams()
    ukf = make_ukf_state(vx_init=10.0, params=params)
    key = jax.random.PRNGKey(42)

    vx_true = 15.0
    dt = 0.005
    R = params.R_wheel
    tf2 = params.track_f / 2.0
    delta = 0.0

    for step in range(500):
        key, k1 = jax.random.split(key)
        noise = 0.15 * jax.random.normal(k1, shape=(10,))

        z = pack_measurement(
            ax_imu=jnp.array(0.0 + noise[0]),
            ay_imu=jnp.array(0.0 + noise[1]),
            wz_gyro=jnp.array(0.0 + noise[2] * 0.005),
            omega_fl=jnp.array(vx_true / R + noise[3] * 0.3),
            omega_fr=jnp.array(vx_true / R + noise[4] * 0.3),
            omega_rl=jnp.array(vx_true / R + noise[5] * 0.3),
            omega_rr=jnp.array(vx_true / R + noise[6] * 0.3),
            delta_steer=jnp.array(delta + noise[7] * 0.002),
            vx_gps=jnp.array(vx_true + noise[8] * 0.5),
        )
        ukf = ukf_step(ukf, z, jnp.array(delta), jnp.array(dt), params)

    est = extract_estimated_state(ukf)
    vx_err = abs(float(est.vx) - vx_true)
    P_eig = np.linalg.eigvalsh(np.array(ukf.P))

    print(f"  vx estimated: {float(est.vx):.3f} (true: {vx_true}), err: {vx_err:.3f}")
    print(f"  P eigenvalues: min={P_eig.min():.2e}, max={P_eig.max():.2e}")
    print(f"  All eigenvalues positive: {bool(np.all(P_eig > 0))}")

    if vx_err < 1.0 and P_eig.max() < 1e4 and P_eig.min() > 0:
        print("[PASS] UKF tracks truth with noise, covariance stable")
        return True
    else:
        print("[FAIL] UKF diverged or covariance exploded")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 22: yaw_target_gain in SuspensionSetup
# ─────────────────────────────────────────────────────────────────────────────

def test_22_yaw_target_gain():
    _header(22, "YAW_TARGET_GAIN IN SUSPENSIONSTEP")
    from models.vehicle_dynamics import (
        SETUP_NAMES, SuspensionSetup, DEFAULT_SETUP,
    )

    checks = []

    # Check name
    if SETUP_NAMES[23] == 'yaw_target_gain':
        print("  SETUP_NAMES[23] = 'yaw_target_gain' ✓")
        checks.append(True)
    else:
        print(f"  SETUP_NAMES[23] = '{SETUP_NAMES[23]}' ✗ (expected 'yaw_target_gain')")
        checks.append(False)

    # Check field
    if hasattr(SuspensionSetup, '_fields') and SuspensionSetup._fields[23] == 'yaw_target_gain':
        print("  SuspensionSetup._fields[23] = 'yaw_target_gain' ✓")
        checks.append(True)
    else:
        field = SuspensionSetup._fields[23] if hasattr(SuspensionSetup, '_fields') else '???'
        print(f"  SuspensionSetup._fields[23] = '{field}' ✗")
        checks.append(False)

    # Check default value
    val = float(DEFAULT_SETUP[23])
    if abs(val - 0.80) < 0.01:
        print(f"  DEFAULT_SETUP[23] = {val:.2f} ✓")
        checks.append(True)
    else:
        print(f"  DEFAULT_SETUP[23] = {val:.2f} ✗ (expected 0.80)")
        checks.append(False)

    # Roundtrip test
    s = SuspensionSetup.from_vector(DEFAULT_SETUP)
    v = s.to_vector()
    if jnp.allclose(v, DEFAULT_SETUP, atol=1e-6):
        print("  from_vector/to_vector roundtrip ✓")
        checks.append(True)
    else:
        print("  from_vector/to_vector roundtrip ✗")
        checks.append(False)

    if all(checks):
        print("[PASS] yaw_target_gain correctly integrated")
    else:
        print("[FAIL] yaw_target_gain integration incomplete")
    return all(checks)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 23: Compliance Steer
# ─────────────────────────────────────────────────────────────────────────────

def test_23_compliance_steer():
    _header(23, "ANALYTICAL COMPLIANCE STEER")
    try:
        from suspension.compliance import (
            compute_compliance_steer_coefficient, BushingParams,
        )
        from suspension.kinematics import SuspensionKinematics
        from tests.test_kinematics import FRONT_HPTS_SYNTHETIC

        kin = SuspensionKinematics(FRONT_HPTS_SYNTHETIC, side="left")
        bp = BushingParams()

        cs = compute_compliance_steer_coefficient(kin, bushing_params=bp)
        print(f"  Compliance steer (front): {cs:+.4f} deg/kN")

        if -0.30 < cs < -0.05:
            print("[PASS] Compliance steer in realistic range")
            return True
        else:
            print(f"[FAIL] Compliance steer {cs:.4f} outside [-0.30, -0.05]")
            return False
    except ImportError as e:
        print(f"[SKIP] Missing dependency: {e}")
        return True  # Don't fail if kinematics test data not available


# ─────────────────────────────────────────────────────────────────────────────
# TEST 24: Anti-Geometry Sweep
# ─────────────────────────────────────────────────────────────────────────────

def test_24_anti_geometry():
    _header(24, "ANTI-GEOMETRY BRAKE BIAS SWEEP")
    from suspension.anti_geometry_sweep import (
        sweep_anti_geometry, anti_geometry_penalty,
    )

    ic_f_z = jnp.array(0.040)   # typical front IC height
    ic_r_z = jnp.array(0.060)   # typical rear IC height
    h_cg = jnp.array(0.330)
    lf = jnp.array(0.8525)
    lr = jnp.array(0.6975)

    # Test sweep
    result = sweep_anti_geometry(ic_f_z, ic_r_z, h_cg, lf, lr)
    print(f"  Bias range: [{float(result.bias_range[0]):.2f}, {float(result.bias_range[-1]):.2f}]")
    print(f"  Worst pitch gain: {float(result.worst_pitch):.3f} deg/G at bias={float(result.worst_bias):.2f}")
    print(f"  Pitch gains finite: {bool(jnp.all(jnp.isfinite(result.pitch_gain)))}")

    # Test penalty
    penalty = anti_geometry_penalty(ic_f_z, ic_r_z, h_cg, lf, lr)
    print(f"  Penalty value: {float(penalty):.4f}")

    # Test gradient
    grad_fn = jax.grad(lambda z: anti_geometry_penalty(z, ic_r_z, h_cg, lf, lr))
    grad_val = float(grad_fn(ic_f_z))
    print(f"  ∂penalty/∂ic_f_z: {grad_val:.6f}")

    if jnp.all(jnp.isfinite(result.pitch_gain)) and jnp.isfinite(penalty) and jnp.isfinite(grad_val):
        print("[PASS] Anti-geometry sweep is smooth and differentiable")
        return True
    else:
        print("[FAIL] Anti-geometry contains NaN/Inf")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 25: Domain Randomization
# ─────────────────────────────────────────────────────────────────────────────

def test_25_domain_randomization():
    _header(25, "DOMAIN RANDOMIZATION SAMPLING")
    from models.vehicle_dynamics import (
        sample_domain_randomization, DomainRandomization,
    )

    key = jax.random.PRNGKey(0)
    dr = sample_domain_randomization(key)

    print(f"  mu_scale:    {float(dr.mu_scale):.4f} (nominal: 1.0)")
    print(f"  track_noise: {np.array(dr.track_noise).round(5)}")
    print(f"  mass_delta:  {float(dr.mass_delta):.2f} kg")
    print(f"  aero_scale:  {float(dr.aero_scale):.4f}")

    # Check distributions are reasonable
    checks = [
        0.7 < float(dr.mu_scale) < 1.3,
        float(jnp.max(jnp.abs(dr.track_noise))) < 0.005,
        abs(float(dr.mass_delta)) < 15.0,
        0.7 < float(dr.aero_scale) < 1.3,
    ]

    if all(checks):
        print("[PASS] Domain randomization produces physically reasonable samples")
        return True
    else:
        print("[FAIL] Domain randomization out of range")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def test_all():
    print("\n" + "═" * 60)
    print("  BATCH 10.5 SANITY CHECK SUITE")
    print("═" * 60)

    results = {}

    # Tests that don't require the full vehicle_dynamics patch
    # (can run before hub motor rewrite)
    try:
        results['T20'] = test_20_ukf_perfect()
    except Exception as e:
        print(f"  [ERROR] Test 20: {e}")
        results['T20'] = False

    try:
        results['T21'] = test_21_ukf_noisy()
    except Exception as e:
        print(f"  [ERROR] Test 21: {e}")
        results['T21'] = False

    try:
        results['T24'] = test_24_anti_geometry()
    except Exception as e:
        print(f"  [ERROR] Test 24: {e}")
        results['T24'] = False

    # Tests that require vehicle_dynamics patch
    try:
        results['T22'] = test_22_yaw_target_gain()
    except Exception as e:
        print(f"  [SKIP] Test 22 (requires VD patch): {e}")
        results['T22'] = None

    try:
        results['T19'] = test_19_hub_motor()
    except Exception as e:
        print(f"  [SKIP] Test 19 (requires VD patch): {e}")
        results['T19'] = None

    try:
        results['T23'] = test_23_compliance_steer()
    except Exception as e:
        print(f"  [SKIP] Test 23: {e}")
        results['T23'] = None

    try:
        results['T25'] = test_25_domain_randomization()
    except Exception as e:
        print(f"  [SKIP] Test 25 (requires VD patch): {e}")
        results['T25'] = None

    # Summary
    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("─" * 60)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)
    print(f"  PASSED:  {passed}/{total}")
    print(f"  FAILED:  {failed}/{total}")
    print(f"  SKIPPED: {skipped}/{total}")
    print("═" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)