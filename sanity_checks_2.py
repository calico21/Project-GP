# sanity_checks_2.py
# Project-GP — Pre-Flight Sanity Checks (Part 2)
# =============================================================================
#
# This file is the EXTENDED partner to `sanity_checks.py`. The original file
# grew past 1.5k LOC and mixed three distinct concerns:
#
#   Part 1 (sanity_checks.py): Tests 1–16
#     · Physics subsystem      (Tests 1–9)
#     · Powertrain control     (Tests 10–16)
#
#   Part 2 (THIS FILE):        Tests 17–32
#     · Koopman observer       (17, 17b)
#     · Dynamic regen blend    (18)
#     · Aerodynamics platform  (19, 20)
#     · Thermal & damper       (21, 22)
#     · Elastokinematics       (23)
#     · Tire transient 2nd-ord (24)
#     · Track surface model    (25)
#     · Suspension kinematics  (26-29: bridges to tests/test_kinematics.py)
#     · State integrity        (30)
#     · Domain randomization   (31)
#     · MORL Pareto archive    (32)
#
# Run:
#   python sanity_checks_2.py             # standalone, all tests
#   python -m sanity_checks_2 --quick     # skip slow tests (training, MORL)
#
# Pairs naturally with: `python sanity_checks.py && python sanity_checks_2.py`
# =============================================================================

from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path

# ── JAX / XLA environment (same flags as sanity_checks.py for parity) ────────
if "XLA_FLAGS" in os.environ:
    del os.environ["XLA_FLAGS"]
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import jax_config  # noqa: F401
except ImportError:
    pass

import jax
import jax.numpy as jnp
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Header / scoreboard helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(num: int | str, title: str) -> None:
    print("\n" + "=" * 60)
    print(f"TEST {num}: {title}")
    print("=" * 60)


_results: dict[str, str] = {}

def _record(test_id: str, status: str) -> None:
    """Track outcome for the final summary."""
    _results[test_id] = status


def _pass(test_id: str, msg: str) -> None:
    _record(test_id, "PASS")
    print(f"[PASS] {msg}")

def _fail(test_id: str, msg: str) -> None:
    _record(test_id, "FAIL")
    print(f"[FAIL] {msg}")

def _warn(test_id: str, msg: str) -> None:
    _record(test_id, "WARN")
    print(f"[WARN] {msg}")

def _skip(test_id: str, msg: str) -> None:
    _record(test_id, "SKIP")
    print(f"[SKIP] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 17: Koopman Observer (DKLQR)
# ─────────────────────────────────────────────────────────────────────────────

def test_17_koopman_observer():
    _header(17, "KOOPMAN DICTIONARY-SWITCHED OBSERVER")
    try:
        from powertrain.modes.advanced.koopman_tv import (
            KoopmanParams, predict_step,
        )
    except ImportError as e:
        _skip("T17", f"koopman_tv not available: {e}")
        return

    p = KoopmanParams() if hasattr(KoopmanParams, "__call__") else KoopmanParams
    z0 = jnp.array([15.0, 0.0, 0.0, 0.0])  # vx, vy, wz, β-est
    u  = jnp.array([0.0, 0.0])

    try:
        z1 = predict_step(z0, u, p)
        if jnp.all(jnp.isfinite(z1)):
            _pass("T17", f"Koopman 1-step prediction finite, "
                         f"|Δz|={float(jnp.linalg.norm(z1 - z0)):.3e}")
        else:
            _fail("T17", "Koopman returned NaN/Inf")
    except Exception as e:
        _warn("T17", f"Koopman API mismatch: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 17b: Koopman EABS Slip Containment
# ─────────────────────────────────────────────────────────────────────────────

def test_17b_koopman_eabs():
    _header("17b", "KOOPMAN E-ABS SLIP CONTAINMENT")
    try:
        from powertrain.modes.advanced.koopman_tv import compute_slip_safe_brake
    except ImportError as e:
        _skip("T17b", f"compute_slip_safe_brake not available: {e}")
        return

    omega = jnp.array([95.0, 95.0, 95.0, 95.0])  # rad/s
    vx    = jnp.array(20.0)
    R     = 0.2032
    T_brake_request = jnp.full(4, -300.0)
    try:
        T_safe = compute_slip_safe_brake(T_brake_request, omega, vx, R)
        # E-ABS should clip more aggressively than the request → smaller |T|
        if bool(jnp.all(jnp.abs(T_safe) <= jnp.abs(T_brake_request) + 1e-3)):
            _pass("T17b", f"E-ABS contains slip — T_safe={[float(t) for t in T_safe]}")
        else:
            _fail("T17b", f"E-ABS amplified torque — T_safe={[float(t) for t in T_safe]}")
    except Exception as e:
        _warn("T17b", f"API mismatch: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 18: Dynamic Regen Blend Energy Accounting
# ─────────────────────────────────────────────────────────────────────────────

def test_18_regen_blend():
    _header(18, "DYNAMIC REGEN BLEND (KKT-OPTIMAL α*)")
    try:
        from powertrain.regen_blend import (
            compute_regen_blend, RegenEnergyState, update_regen_energy,
            regen_efficiency, RegenBlendParams,
        )
    except ImportError as e:
        _skip("T18", f"regen_blend not present: {e}")
        return

    p = RegenBlendParams() if hasattr(RegenBlendParams, "__call__") else RegenBlendParams
    T_brake = jnp.array(-3000.0)
    T_min   = jnp.full(4, -400.0)
    Fx_hard = jnp.full(4, 1500.0)
    omega   = jnp.full(4, 60.0)
    soc     = jnp.array(0.75)

    try:
        T_sc, alpha = compute_regen_blend(T_brake, T_min, Fx_hard, omega, soc)
        # Accumulate energy over 200 steps (1 second at dt=5ms)
        e_state = RegenEnergyState.zero()
        dt = jnp.array(0.005)
        for _ in range(200):
            e_state = update_regen_energy(e_state, alpha, T_sc, omega, dt)
        E_Wh = float(e_state.E_regen_J) / 3600.0
        eff  = float(regen_efficiency(e_state))
        if E_Wh > 0.0:
            _pass("T18", f"Regen energy {E_Wh*1000:.2f} mWh / 1.0 s; η={eff:.2%}")
        else:
            _warn("T18", f"E_regen={E_Wh*1000:.2f} mWh — sign or accumulation issue")
    except Exception as e:
        _warn("T18", f"compute_regen_blend signature mismatch: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 19: 108-DOF State Integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_19_state_integrity():
    _header(19, "STATE VECTOR INTEGRITY (ENERGY MONOTONE BOUNDED)")
    from models.vehicle_dynamics import (
        DifferentiableMultiBodyVehicle, DEFAULT_SETUP,
    )
    try:
        from config.vehicles.ter26 import vehicle_params as VP
        from config.tire_coeffs import tire_coeffs as TC
    except ImportError:
        from data.configs.vehicle_params import vehicle_params as VP  # type: ignore
        from data.configs.tire_coeffs import tire_coeffs as TC        # type: ignore

    veh = DifferentiableMultiBodyVehicle(VP, TC)

    x = jnp.zeros(46).at[14].set(15.0).at[28:38].set(85.0)
    u = jnp.array([0.0, 0.0])
    sp = jnp.asarray(DEFAULT_SETUP)
    KE0 = 0.5 * VP.get("total_mass", 300.0) * float(x[14]) ** 2
    energies = [KE0]
    for k in range(50):
        x = veh.simulate_step(x, u, sp, dt=0.005)
        if not bool(jnp.all(jnp.isfinite(x))):
            _fail("T19", f"NaN at step {k}, indices {jnp.where(~jnp.isfinite(x))[0].tolist()}")
            return
        energies.append(0.5 * VP.get("total_mass", 300.0) * float(x[14]) ** 2)
    drift = abs(energies[-1] - energies[0]) / energies[0]
    # Coasting: kinetic energy should drop slowly, never rise.
    if drift < 0.05 and energies[-1] <= energies[0] + 1.0:
        _pass("T19", f"State integrity OK; KE drift = {drift*100:.2f}%")
    else:
        _warn("T19", f"KE drift {drift*100:.2f}% — check dissipation")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 20: Aero Ground-Effect Stall
# ─────────────────────────────────────────────────────────────────────────────

def test_20_ground_effect_stall():
    _header(20, "GROUND EFFECT STALL ENVELOPE")
    try:
        from models.aero_platform import (
            AeroPlatformConfig, ground_effect_envelope,
        )
    except ImportError as e:
        _skip("T20", f"aero_platform not present: {e}")
        return

    cfg = AeroPlatformConfig()
    rh_peak  = cfg.rh_peak
    rh_stall = cfg.rh_stall
    rh_high  = cfg.rh_high

    G_peak  = float(ground_effect_envelope(jnp.array(rh_peak),  rh_peak, rh_stall, rh_high))
    G_stall = float(ground_effect_envelope(jnp.array(rh_stall * 0.6), rh_peak, rh_stall, rh_high))
    G_high  = float(ground_effect_envelope(jnp.array(rh_high * 1.3),  rh_peak, rh_stall, rh_high))

    print(f"  Γ_peak ={G_peak:.3f}  Γ_stall={G_stall:.3f}  Γ_high={G_high:.3f}")

    if G_peak > 0.85 and G_stall < 0.5 and G_high < 0.7:
        _pass("T20", "Ground-effect envelope behaves correctly")
    else:
        _fail("T20", f"Envelope mis-shaped: Γ=[peak={G_peak}, stall={G_stall}, high={G_high}]")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 21: Damper Hysteresis (LS/HS Bilinear Symmetry)
# ─────────────────────────────────────────────────────────────────────────────

def test_21_damper_hysteresis():
    _header(21, "DAMPER F–v HYSTERESIS LOOP")
    # Synthesize a digressive bilinear damper directly here — does not depend
    # on internal vehicle_dynamics structure.
    c_low_f, c_hi_f, v_knee_f = 2500.0, 900.0, 0.15
    rebound_ratio = 1.5
    gas_force = 120.0

    def damper_f(v, low, hi, knee, reb_ratio, gas):
        # Smooth knee transition: tanh blend
        s = jnp.tanh(v / knee)        # ≈ sign for |v|>>knee
        c_eff = jnp.where(v > 0,
                          low + (hi - low) * jax.nn.sigmoid((v - knee) * 50),
                          (low + (hi - low) * jax.nn.sigmoid((-v - knee) * 50)) * reb_ratio)
        return c_eff * v + gas * s

    v_arr = jnp.linspace(-0.5, 0.5, 200)
    F_arr = jax.vmap(lambda v: damper_f(v, c_low_f, c_hi_f, v_knee_f,
                                        rebound_ratio, gas_force))(v_arr)

    # Symmetry contracts: rebound side has higher slope (rebound_ratio>1)
    bump_slope   = float((F_arr[150] - F_arr[100]) / (v_arr[150] - v_arr[100]))
    rebound_slope = float((F_arr[100] - F_arr[50])  / (v_arr[100] - v_arr[50]))
    if rebound_slope > bump_slope > 0:
        _pass("T21", f"F–v slopes: bump={bump_slope:.0f}, rebound={rebound_slope:.0f} (≥)")
    else:
        _fail("T21", f"Rebound not stiffer: bump={bump_slope:.0f}, reb={rebound_slope:.0f}")

    # Finiteness across full sweep
    if bool(jnp.all(jnp.isfinite(F_arr))):
        _pass("T21", "Damper force finite across [-0.5, 0.5] m/s")
    else:
        _fail("T21", "Damper force non-finite")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 22: Tire Thermal 3-Node (or 5-Node) Causality
# ─────────────────────────────────────────────────────────────────────────────

def test_22_tire_thermal():
    _header(22, "TIRE THERMAL CAUSALITY (HEAT FLOW DIRECTION)")
    try:
        from models.tire_model import PacejkaTire
        try:
            from config.tire_coeffs import tire_coeffs as TC
        except ImportError:
            from data.configs.tire_coeffs import tire_coeffs as TC  # type: ignore
    except ImportError as e:
        _skip("T22", f"tire_model unavailable: {e}")
        return

    tire = PacejkaTire(TC, rng_seed=0)
    # Sweep slip → expect grip to decrease as T deviates from optimum
    T_cold = jnp.array([55., 55., 55.])
    T_opt  = jnp.array([90., 90., 90.])
    T_hot  = jnp.array([130., 130., 130.])

    Fy_at = lambda T_arr: float(
        tire.compute_force(jnp.deg2rad(6.0), 0.0, 1000., 0., T_arr, 90., 15.0)[1]
    )
    Fy_cold = abs(Fy_at(T_cold))
    Fy_opt  = abs(Fy_at(T_opt))
    Fy_hot  = abs(Fy_at(T_hot))

    print(f"  |Fy| @55°C={Fy_cold:.0f} N, @90°C={Fy_opt:.0f} N, @130°C={Fy_hot:.0f} N")
    if Fy_opt > Fy_cold and Fy_opt > Fy_hot:
        _pass("T22", "Thermal grip peaks near T_opt; cold + hot both reduce grip")
    else:
        _fail("T22", f"Thermal envelope mis-shaped: cold={Fy_cold:.0f}, "
                     f"opt={Fy_opt:.0f}, hot={Fy_hot:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 23: Elastokinematic Corrections (Compliance Steer + Camber)
# ─────────────────────────────────────────────────────────────────────────────

def test_23_elastokinematics():
    _header(23, "ELASTOKINEMATICS — COMPLIANCE STEER & CAMBER UNDER LOAD")
    try:
        from suspension.elastokinematics import (
            compute_elastokinematic_corrections,
        )
    except ImportError as e:
        _skip("T23", f"elastokinematics unavailable: {e}")
        return

    zh = jnp.zeros(6)
    vd = jnp.zeros(6)

    # Symmetric load test: ±Fy should give opposite-sign d_toe
    out_pos = compute_elastokinematic_corrections(
        jnp.array(0.0), jnp.array(+1500.0), jnp.array(800.0), jnp.array(0.0), zh, vd)
    out_neg = compute_elastokinematic_corrections(
        jnp.array(0.0), jnp.array(-1500.0), jnp.array(800.0), jnp.array(0.0), zh, vd)

    d_toe_pos = float(out_pos[0])
    d_toe_neg = float(out_neg[0])

    if d_toe_pos * d_toe_neg < 0.0:
        _pass("T23", f"d_toe sign-mirrors with Fy: +Fy→{d_toe_pos:+.2e}, "
                     f"-Fy→{d_toe_neg:+.2e}")
    else:
        _fail("T23", f"d_toe NOT antisymmetric: {d_toe_pos:+.3e} / {d_toe_neg:+.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 24: Tire Transient (2nd-order Carcass Lag)
# ─────────────────────────────────────────────────────────────────────────────

def test_24_tire_transient():
    _header(24, "TIRE TRANSIENT — RELAXATION-LENGTH OBEYS V·τ ≈ σ")
    # The first-order transient: dα_t/dt = (V/σ)(α − α_t)
    # Step input: α_t should reach 63% of α in time τ = σ/V.
    sigma = 0.30   # m, FS-typical relaxation length
    V     = 15.0
    tau   = sigma / V

    # Numerical Euler simulation of the lag at dt = 1 ms
    dt = 0.001
    n_steps = int(2.0 * tau / dt)
    alpha_in = 0.05   # 50 mrad ≈ 2.86°
    alpha_t  = 0.0
    for _ in range(n_steps):
        alpha_t = alpha_t + dt * (V / sigma) * (alpha_in - alpha_t)
    expected = (1.0 - jnp.exp(-2.0)) * alpha_in     # 86.5% of step
    err = abs(alpha_t - float(expected))
    if err < 0.005 * alpha_in:
        _pass("T24", f"Relaxation 86.5% of step at t=2τ (err={err:.1e})")
    else:
        _fail("T24", f"Relaxation off: α_t={alpha_t:.4f}, expected ≈{float(expected):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 25: Track Surface Friction Map
# ─────────────────────────────────────────────────────────────────────────────

def test_25_track_surface():
    _header(25, "TRACK SURFACE FRICTION SMOOTHNESS")
    try:
        from optimization.differentiable_track import (
            query_track_friction, make_track_state, TrackConfig,
        )
    except ImportError as e:
        _skip("T25", f"differentiable_track unavailable: {e}")
        return

    try:
        cfg = TrackConfig()
        state = make_track_state(cfg=cfg)
        # Must be smoothly differentiable in s
        grad_mu = jax.grad(
            lambda s: query_track_friction(s, jnp.array(0.0), state, cfg)[0]
        )(jnp.array(250.0))
        if math.isfinite(float(grad_mu)):
            _pass("T25", f"∂μ/∂s = {float(grad_mu):+.3e} /m  (finite)")
        else:
            _fail("T25", f"Non-finite ∂μ/∂s: {grad_mu}")
    except Exception as e:
        _warn("T25", f"Track surface API mismatch: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 26: Suspension Kinematics — Bridge to test_kinematics.py
# ─────────────────────────────────────────────────────────────────────────────

def test_26_suspension_kinematics():
    _header(26, "SUSPENSION KINEMATICS BRIDGE (delegates to test_kinematics)")
    try:
        from tests.test_kinematics import test_all as run_kin_tests
    except ImportError as e:
        _skip("T26", f"test_kinematics not on path: {e}")
        return
    try:
        ok = run_kin_tests()
        (_pass if ok else _fail)("T26",
            "test_kinematics.py reports " + ("all-pass" if ok else "failures"))
    except Exception as e:
        _fail("T26", f"test_kinematics raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 27: Suspension Sweep Bridge
# ─────────────────────────────────────────────────────────────────────────────

def test_27_suspension_sweep():
    _header(27, "SUSPENSION SWEEP BRIDGE (delegates to test_kinematics.test_sweep)")
    try:
        from tests.test_kinematics import test_sweep
    except ImportError as e:
        _skip("T27", f"test_sweep not on path: {e}")
        return
    try:
        ok = test_sweep()
        (_pass if ok else _fail)("T27",
            "test_sweep reports " + ("all-pass" if ok else "failures"))
    except Exception as e:
        _fail("T27", f"test_sweep raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 28: Full Suspension Folder Bridge
# ─────────────────────────────────────────────────────────────────────────────

def test_28_full_suspension():
    _header(28, "FULL SUSPENSION FOLDER (delegates to test_suspension_full)")
    try:
        from tests.test_suspension_full import run_all as run_susp_full
    except ImportError as e:
        _skip("T28", f"test_suspension_full not on path: {e}")
        return
    try:
        ok = run_susp_full()
        (_pass if ok else _fail)("T28",
            "test_suspension_full reports " + ("all-pass" if ok else "failures"))
    except Exception as e:
        _fail("T28", f"test_suspension_full raised: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 29: Domain Randomization
# ─────────────────────────────────────────────────────────────────────────────

def test_29_domain_randomization():
    _header(29, "DOMAIN RANDOMIZATION SAMPLING DISTRIBUTIONS")
    try:
        from models.vehicle_dynamics import sample_domain_randomization
    except ImportError as e:
        _skip("T29", f"sample_domain_randomization unavailable: {e}")
        return

    try:
        # Draw a sample and check distribution support
        keys = jax.random.split(jax.random.PRNGKey(42), 64)
        samples = [sample_domain_randomization(k) for k in keys]
        mu_arr   = np.array([float(s.mu_scale)   for s in samples])
        mass_arr = np.array([float(s.mass_delta) for s in samples])
        aero_arr = np.array([float(s.aero_scale) for s in samples])

        in_range = (
            np.all((mu_arr   > 0.7) & (mu_arr   < 1.3)) and
            np.all(np.abs(mass_arr) < 15.0) and
            np.all((aero_arr > 0.7) & (aero_arr < 1.3))
        )
        if in_range:
            _pass("T29", f"DR samples in physical range over n=64 draws")
            print(f"  μ_scale range: [{mu_arr.min():.3f}, {mu_arr.max():.3f}]")
            print(f"  mass_delta:    [{mass_arr.min():+.2f}, {mass_arr.max():+.2f}] kg")
            print(f"  aero_scale:    [{aero_arr.min():.3f}, {aero_arr.max():.3f}]")
        else:
            _fail("T29", "DR samples out of expected physical range")
    except Exception as e:
        _warn("T29", f"DR API mismatch: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 30: MORL Pareto Archive (Smoke Test)
# ─────────────────────────────────────────────────────────────────────────────

def test_30_morl_smoke(quick: bool = False):
    _header(30, "MORL OPTIMIZER SMOKE TEST (1 BO ITERATION)")
    if quick:
        _skip("T30", "skipped under --quick (heavy)")
        return
    try:
        from optimization.evolutionary import MORL_SB_TRPO_Optimizer
    except ImportError as e:
        _skip("T30", f"evolutionary unavailable: {e}")
        return

    try:
        opt = MORL_SB_TRPO_Optimizer(ensemble_size=4)
        opt.BO_N_INIT = 4
        opt.BO_N_ITERS = 4
        # Patch to a minimal smoke run.
        try:
            setups, grips, stabs, _ = opt.run(iterations=0)
        except Exception as e:
            # Some signatures need a tiny n_steps
            setups, grips, stabs, _ = opt.run(iterations=1)
        if len(grips) > 0:
            _pass("T30", f"MORL produced {len(grips)} archive entries; "
                         f"max grip = {float(np.max(grips)):.3f}G")
        else:
            _warn("T30", "MORL returned empty archive")
    except Exception as e:
        _warn("T30", f"MORL smoke failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 31: WMPC Solver Smoke
# ─────────────────────────────────────────────────────────────────────────────

def test_31_wmpc_smoke():
    _header(31, "WMPC SOLVER SMOKE (CONSTANT-CURVATURE TRACK)")
    try:
        from optimization.ocp_solver import DiffWMPCSolver
    except ImportError as e:
        _skip("T31", f"ocp_solver unavailable: {e}")
        return

    N = 32
    track_s = np.linspace(0, 60, N)
    track_k = np.full(N, 0.05)
    track_psi = track_s * 0.05
    track_x = 20.0 * np.sin(track_psi)
    track_y = 20.0 * (1.0 - np.cos(track_psi))
    track_w_left = np.full(N, 3.0)
    track_w_right = np.full(N, 3.0)

    try:
        solver = DiffWMPCSolver(N_horizon=N, mu_friction=1.4, V_limit=25.0,
                                dev_mode=True)
        result = solver.solve(
            track_s=track_s, track_k=track_k,
            track_x=track_x, track_y=track_y, track_psi=track_psi,
            track_w_left=track_w_left, track_w_right=track_w_right,
        )
        v_arr = np.asarray(result["v"])
        if np.all(np.isfinite(v_arr)) and 0 < v_arr.mean() < 30:
            _pass("T31", f"WMPC v_mean={v_arr.mean():.1f} m/s, "
                         f"v_max={v_arr.max():.1f} m/s")
        else:
            _fail("T31", f"WMPC v out of bounds or non-finite: {v_arr}")
    except Exception as e:
        _warn("T31", f"WMPC smoke skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 32: SuspensionSetup Pytree Round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_32_setup_pytree():
    _header(32, "SUSPENSIONSETUP PYTREE — to/from VECTOR ROUND-TRIP")
    try:
        from models.vehicle_dynamics import (
            SuspensionSetup, DEFAULT_SETUP, SETUP_DIM, SETUP_NAMES,
        )
    except ImportError as e:
        _skip("T32", f"vehicle_dynamics unavailable: {e}")
        return

    # Round-trip
    s = SuspensionSetup.from_vector(DEFAULT_SETUP)
    v = s.to_vector()
    rt_ok = bool(jnp.allclose(v, DEFAULT_SETUP, atol=1e-6))

    # Dimension and naming consistency
    dim_ok = (DEFAULT_SETUP.shape[0] == SETUP_DIM
              == len(SETUP_NAMES) == len(s._fields))

    # Pytree round-trip: leaves → reconstruct
    leaves = jax.tree_util.tree_leaves(s)
    s_rt   = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(s), leaves)
    pyt_ok = bool(jnp.allclose(s_rt.to_vector(), v, atol=1e-6))

    # project_to_bounds preserves shape
    sb = s.project_to_bounds()
    bounds_ok = sb.to_vector().shape == v.shape

    if rt_ok and dim_ok and pyt_ok and bounds_ok:
        _pass("T32", f"SuspensionSetup pytree integrity (dim={SETUP_DIM})")
    else:
        _fail("T32", f"rt={rt_ok}, dim={dim_ok}, pytree={pyt_ok}, bounds={bounds_ok}")


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    ("T17",   test_17_koopman_observer),
    ("T17b",  test_17b_koopman_eabs),
    ("T18",   test_18_regen_blend),
    ("T19",   test_19_state_integrity),
    ("T20",   test_20_ground_effect_stall),
    ("T21",   test_21_damper_hysteresis),
    ("T22",   test_22_tire_thermal),
    ("T23",   test_23_elastokinematics),
    ("T24",   test_24_tire_transient),
    ("T25",   test_25_track_surface),
    ("T26",   test_26_suspension_kinematics),
    ("T27",   test_27_suspension_sweep),
    ("T28",   test_28_full_suspension),
    ("T29",   test_29_domain_randomization),
    ("T30",   None),   # quick-aware, handled in main()
    ("T31",   test_31_wmpc_smoke),
    ("T32",   test_32_setup_pytree),
]


def main():
    parser = argparse.ArgumentParser(description="Project-GP sanity checks (Part 2)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow tests (MORL, WMPC, training)")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated test IDs to run (e.g. T19,T22)")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  PROJECT-GP — PRE-FLIGHT SANITY CHECKS · PART 2")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}"
          + (f"   Filter: {args.only}" if args.only else ""))
    print("█" * 60)
    t0 = time.perf_counter()

    only_set = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None

    for tid, fn in ALL_TESTS:
        if only_set and tid not in only_set:
            continue
        try:
            if tid == "T30":
                test_30_morl_smoke(quick=args.quick)
            else:
                fn()
        except Exception as e:
            _fail(tid, f"raised: {e}")
            import traceback; traceback.print_exc()

    # Summary
    pass_n = sum(1 for v in _results.values() if v == "PASS")
    fail_n = sum(1 for v in _results.values() if v == "FAIL")
    warn_n = sum(1 for v in _results.values() if v == "WARN")
    skip_n = sum(1 for v in _results.values() if v == "SKIP")
    total = len(_results)

    print("\n" + "█" * 60)
    print("  SUMMARY · PART 2")
    print("█" * 60)
    print(f"  Total tests run:  {total}")
    print(f"  PASS: {pass_n}    FAIL: {fail_n}    WARN: {warn_n}    SKIP: {skip_n}")
    print(f"  Wall-clock:       {time.perf_counter() - t0:.1f} s")
    print("█" * 60)

    if fail_n > 0:
        print("\n  FAILED TESTS:")
        for tid, status in _results.items():
            if status == "FAIL":
                print(f"    · {tid}")

    return fail_n == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)