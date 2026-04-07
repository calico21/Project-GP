# powertrain/powertrain_sanity_checks.py
# Project-GP — Powertrain Control Stack Verification
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tests 10–16 for the 4WD powertrain control modules.
# Run standalone:  python -m powertrain.powertrain_sanity_checks
# Or append to main sanity_checks.py:  import powertrain.powertrain_sanity_checks
#
# Each test follows the existing PASS/FAIL/WARN format from sanity_checks.py.
# ═══════════════════════════════════════════════════════════════════════════════

import time
import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10: Motor Torque Envelope Continuity
# ─────────────────────────────────────────────────────────────────────────────

def test_motor_torque_envelope():
    print("\n" + "=" * 60)
    print("TEST 10: MOTOR TORQUE ENVELOPE (FIELD-WEAKENING CONTINUITY)")
    print("=" * 60)
    from powertrain.motor_model import (
        MotorParams, motor_torque_envelope, motor_torque_limits_at_wheel,
        BatteryParams,
    )

    mp = MotorParams()
    V_bus = jnp.array(600.0)
    T_motor = jnp.array(60.0)
    T_inv = jnp.array(50.0)

    # Sweep motor speed from 0 to 1200 rad/s
    omegas = jnp.linspace(0.1, 1200.0, 200)
    T_max = jax.vmap(lambda w: motor_torque_envelope(w, V_bus, T_motor, T_inv, mp))(omegas)

    # Check 1: T_max at low speed should be close to T_peak
    T_at_low = float(T_max[0])
    if abs(T_at_low - mp.T_peak) / mp.T_peak < 0.05:
        print(f"[PASS] Low-speed torque: {T_at_low:.1f} Nm "
              f"(expected ≈{mp.T_peak:.0f} Nm, error <5%)")
    else:
        print(f"[FAIL] Low-speed torque: {T_at_low:.1f} Nm "
              f"(expected ≈{mp.T_peak:.0f} Nm)")

    # Check 2: T_max should decrease monotonically in field-weakening region
    # Find index where power limiting kicks in: P_peak / omega < T_peak
    omega_base = mp.P_peak / mp.T_peak  # ~166 rad/s
    fw_start = int(jnp.searchsorted(omegas, omega_base * 1.2))
    T_fw = T_max[fw_start:]
    diffs = jnp.diff(T_fw)
    mono_violations = int(jnp.sum(diffs > 0.5))  # allow 0.5 Nm numerical tolerance

    if mono_violations == 0:
        print(f"[PASS] Field-weakening region is monotonically decreasing "
              f"({len(T_fw)} points checked)")
    else:
        print(f"[FAIL] {mono_violations} monotonicity violations in field-weakening region")

    # Check 3: T_max should be strictly positive everywhere (no negative torque capability)
    if float(jnp.min(T_max)) >= 0.0:
        print(f"[PASS] Torque envelope is non-negative everywhere "
              f"(min={float(jnp.min(T_max)):.4f} Nm)")
    else:
        print(f"[FAIL] Negative torque in envelope: min={float(jnp.min(T_max)):.4f} Nm")

    # Check 4: Per-wheel limits are finite and correctly ordered
    omega_wheel = jnp.full(4, 60.0)
    T_motors = jnp.full(4, 60.0)
    T_invs = jnp.full(4, 50.0)
    T_min_w, T_max_w = motor_torque_limits_at_wheel(
        omega_wheel, V_bus, T_motors, T_invs, jnp.array(80.0), mp, BatteryParams(),
    )
    all_finite = bool(jnp.all(jnp.isfinite(T_min_w)) and jnp.all(jnp.isfinite(T_max_w)))
    all_ordered = bool(jnp.all(T_min_w <= T_max_w))

    if all_finite and all_ordered:
        print(f"[PASS] Per-wheel limits: T_min=[{float(T_min_w[0]):.1f}] "
              f"≤ T_max=[{float(T_max_w[0]):.1f}] Nm (all finite, correctly ordered)")
    else:
        print(f"[FAIL] Per-wheel limits invalid: finite={all_finite}, ordered={all_ordered}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 11: SOCP Allocator Produces Feasible Torques
# ─────────────────────────────────────────────────────────────────────────────

def test_socp_allocator():
    print("\n" + "=" * 60)
    print("TEST 11: SOCP TORQUE ALLOCATOR (FRICTION FEASIBILITY)")
    print("=" * 60)
    from powertrain.torque_vectoring import (
        solve_torque_allocation, TVGeometry, AllocatorWeights,
    )

    geo = TVGeometry()
    w = AllocatorWeights()

    # Scenario: 15 m/s straight-line acceleration
    T_warmstart = jnp.zeros(4)
    T_prev = jnp.zeros(4)
    Fx_target = jnp.array(3000.0)   # moderate acceleration
    Mz_target = jnp.array(0.0)      # straight-line
    delta = jnp.array(0.0)
    Fz = jnp.array([700.0, 700.0, 800.0, 800.0])  # rear-biased from accel
    Fy = jnp.zeros(4)
    mu = jnp.full(4, 1.4)
    omega_w = jnp.full(4, 15.0 / 0.2032)
    T_min = jnp.full(4, -400.0)
    T_max = jnp.full(4, 450.0)
    P_max = jnp.array(80000.0)

    t0 = time.perf_counter()
    T_alloc = solve_torque_allocation(
        T_warmstart, T_prev, Fx_target, Mz_target, delta,
        Fz, Fy, mu, omega_w, T_min, T_max, P_max, geo, w,
    )
    _ = float(T_alloc[0])  # force evaluation
    t_solve = (time.perf_counter() - t0) * 1000

    # Check 1: All torques within motor limits
    within_limits = bool(jnp.all(T_alloc >= T_min - 1.0) and jnp.all(T_alloc <= T_max + 1.0))
    if within_limits:
        print(f"[PASS] All torques within motor limits "
              f"(T=[{float(T_alloc[0]):.1f}, {float(T_alloc[1]):.1f}, "
              f"{float(T_alloc[2]):.1f}, {float(T_alloc[3]):.1f}] Nm)")
    else:
        print(f"[FAIL] Torque limit violation detected")

    # Check 2: Friction circle feasibility (Fx² + Fy² ≤ (μ·Fz)²)
    Fx_wheel = T_alloc / geo.r_w
    force_sq = Fx_wheel ** 2 + Fy ** 2
    limit_sq = (mu * Fz) ** 2
    utilization = jnp.sqrt(force_sq / limit_sq)
    max_util = float(jnp.max(utilization))

    if max_util <= 1.05:  # 5% tolerance for barrier relaxation
        print(f"[PASS] Friction circles feasible (max utilization: {max_util:.3f})")
    else:
        print(f"[FAIL] Friction circle violated (max utilization: {max_util:.3f} > 1.05)")

    # Check 3: Straight-line → near-zero yaw moment
    from powertrain.torque_vectoring import yaw_moment_arms
    arms = yaw_moment_arms(delta, geo)
    Mz_actual = float(jnp.sum(T_alloc * arms))
    if abs(Mz_actual) < 50.0:  # < 50 Nm residual yaw moment
        print(f"[PASS] Straight-line yaw moment ≈ 0 (actual: {Mz_actual:.1f} Nm)")
    else:
        print(f"[WARN] Non-zero yaw moment on straight: {Mz_actual:.1f} Nm")

    # Check 4: Solve time
    if t_solve < 50.0:
        print(f"[PASS] SOCP solve time: {t_solve:.1f}ms (budget: <50ms)")
    else:
        print(f"[WARN] SOCP solve time: {t_solve:.1f}ms (exceeds 50ms budget)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 12: CBF Safety Filter Clips Unsafe Commands
# ─────────────────────────────────────────────────────────────────────────────

def test_cbf_safety():
    print("\n" + "=" * 60)
    print("TEST 12: CBF SAFETY FILTER (SIDESLIP INVARIANCE)")
    print("=" * 60)
    from powertrain.torque_vectoring import (
        cbf_safety_filter, TVGeometry, CBFParams,
    )

    geo = TVGeometry()
    cbf = CBFParams(beta_max=0.15, wz_max=1.5)

    # Scenario: large sideslip angle, aggressive torque command tries to
    # push yaw even further → CBF should intervene
    vx = jnp.array(15.0)
    vy = jnp.array(2.0)      # β ≈ 0.133 rad — near limit
    wz = jnp.array(1.3)      # near wz_max
    Fz = jnp.array([700., 700., 800., 800.])
    Fy_total = jnp.array(4000.0)
    mu_est = jnp.array(1.4)
    omega_w = jnp.full(4, 15.0 / 0.2032)
    T_min = jnp.full(4, -400.0)
    T_max = jnp.full(4, 450.0)

    # Aggressive allocator output: tries to increase yaw moment
    T_alloc = jnp.array([-100.0, 300.0, -50.0, 350.0])  # strong rightward yaw
    T_prev = jnp.zeros(4)

    T_safe = cbf_safety_filter(
        T_alloc, T_prev, vx, vy, wz, Fz, Fy_total, mu_est,
        omega_w, T_min, T_max, gp_sigma, geo, cbf,
    )

    intervention = float(jnp.linalg.norm(T_safe - T_alloc))
    if intervention > 5.0:
        print(f"[PASS] CBF intervened: modification magnitude = {intervention:.1f} Nm "
              f"(β={float(vy/vx):.3f} rad, near limit {cbf.beta_max:.2f})")
    else:
        print(f"[FAIL] CBF did NOT intervene despite near-limit sideslip "
              f"(β={float(vy/vx):.3f}, intervention={intervention:.1f} Nm)")

    # Check: safe torques should produce less yaw-amplifying moment
    from powertrain.torque_vectoring import yaw_moment_arms
    arms = yaw_moment_arms(jnp.array(0.0), geo)
    Mz_alloc = float(jnp.sum(T_alloc * arms))
    Mz_safe = float(jnp.sum(T_safe * arms))

    if abs(Mz_safe) < abs(Mz_alloc):
        print(f"[PASS] CBF reduced yaw moment: {Mz_alloc:.1f} → {Mz_safe:.1f} Nm")
    else:
        print(f"[WARN] CBF did not reduce yaw moment: {Mz_alloc:.1f} → {Mz_safe:.1f} Nm")

    # All outputs must be finite
    if bool(jnp.all(jnp.isfinite(T_safe))):
        print(f"[PASS] CBF output is finite: [{', '.join(f'{float(t):.1f}' for t in T_safe)}] Nm")
    else:
        print(f"[FAIL] CBF output contains NaN/Inf")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 13: DESC Gradient Convergence Direction
# ─────────────────────────────────────────────────────────────────────────────

def test_desc_convergence():
    print("\n" + "=" * 60)
    print("TEST 13: DESC EXTREMUM-SEEKING (GRADIENT CONVERGENCE)")
    print("=" * 60)
    from powertrain.traction_control import (
        DESCState, DESCParams, desc_step, kappa_star_model,
    )

    params = DESCParams()
    state = DESCState.default(params)
    dt = jnp.array(0.005)

    # Simulate 200 steps (1 second) with synthetic Fx that peaks at κ ≈ 0.12
    kappa_peak = 0.12
    vx = jnp.array(15.0)

    for i in range(200):
        # Synthetic Fx response: Pacejka-like curve with peak at kappa_peak
        kappa_current = state.kappa_base
        Fx_synth = 1500.0 * jnp.sin(1.579 * jnp.arctan(18.5 * kappa_current))
        omega_w = jnp.full(4, 15.0 / 0.2032)
        state, kappa_ref = desc_step(state, Fx_synth, omega_w, vx, dt, params)

    kappa_final = float(jnp.mean(state.kappa_base))
    error = abs(kappa_final - kappa_peak)

    if error < 0.05:
        print(f"[PASS] DESC converged: κ_base = {kappa_final:.4f} "
              f"(target ≈ {kappa_peak:.2f}, error = {error:.4f})")
    elif error < 0.10:
        print(f"[WARN] DESC partially converged: κ_base = {kappa_final:.4f} "
              f"(target ≈ {kappa_peak:.2f}, error = {error:.4f})")
    else:
        print(f"[FAIL] DESC did not converge: κ_base = {kappa_final:.4f} "
              f"(target ≈ {kappa_peak:.2f}, error = {error:.4f})")

    # Check model-based κ* is reasonable
    Fz = jnp.array([700., 700., 800., 800.])
    T_tire = jnp.full(4, 85.0)
    kappa_model = kappa_star_model(Fz, jnp.array(1.4), T_tire)
    k_model_mean = float(jnp.mean(kappa_model))

    if 0.05 < k_model_mean < 0.20:
        print(f"[PASS] Model-based κ* = {k_model_mean:.4f} (physically reasonable range)")
    else:
        print(f"[FAIL] Model-based κ* = {k_model_mean:.4f} (outside 0.05–0.20 range)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 14: Launch State Machine Phase Progression
# ─────────────────────────────────────────────────────────────────────────────

def test_launch_state_machine():
    print("\n" + "=" * 60)
    print("TEST 14: LAUNCH CONTROL STATE MACHINE")
    print("=" * 60)
    from powertrain.launch_control import (
        LaunchParams, LaunchState, launch_step,
        PHASE_IDLE, PHASE_ARMED, PHASE_LAUNCH, PHASE_TC,
    )

    params = LaunchParams()
    state = LaunchState.default(params)
    dt = jnp.array(0.005)
    T_tc = jnp.full(4, 200.0)

    # Phase 1: IDLE (throttle = 0, brake = 0)
    out, state = launch_step(
        jnp.array(0.0), jnp.array(0.0), jnp.array(0.0),
        jnp.zeros(4), T_tc, state, dt, params,
    )
    phase_idle = float(out.phase)

    # Phase 2: ARM (throttle > 0.95, brake > 0.5)
    for _ in range(20):
        out, state = launch_step(
            jnp.array(0.98), jnp.array(0.7), jnp.array(0.0),
            jnp.zeros(4), T_tc, state, dt, params,
        )
    phase_armed = float(out.phase)

    # Phase 3: LAUNCH (brake released while throttle held)
    # Run enough steps for probe to complete
    for _ in range(50):
        out, state = launch_step(
            jnp.array(0.98), jnp.array(0.0), jnp.array(0.0),
            jnp.full(4, 5.0), T_tc, state, dt, params,
        )
    phase_launch = float(out.phase)

    # Phase 4: Simulate acceleration to handoff speed
    for _ in range(400):
        out, state = launch_step(
            jnp.array(0.98), jnp.array(0.0), jnp.array(8.0),
            jnp.full(4, 40.0), T_tc, state, dt, params,
        )
    phase_final = float(out.phase)

    phases_ok = phase_idle < phase_armed <= phase_launch
    print(f"  Phase progression: IDLE({phase_idle:.0f}) → ARMED({phase_armed:.0f}) "
          f"→ LAUNCH({phase_launch:.0f}) → FINAL({phase_final:.0f})")

    if phases_ok:
        print(f"[PASS] State machine progresses monotonically through phases")
    else:
        print(f"[FAIL] Non-monotonic phase progression detected")

    # Check launch torque is non-zero during LAUNCH phase
    T_mag = float(jnp.sum(jnp.abs(out.T_command)))
    if T_mag > 10.0 or phase_final >= PHASE_TC:
        print(f"[PASS] Launch produces torque or has handed off to TC "
              f"(|T| = {T_mag:.1f} Nm, phase = {phase_final:.0f})")
    else:
        print(f"[FAIL] Zero torque during launch phase (|T| = {T_mag:.1f} Nm)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 15: Virtual Impedance PIO Suppression
# ─────────────────────────────────────────────────────────────────────────────

def test_virtual_impedance():
    print("\n" + "=" * 60)
    print("TEST 15: VIRTUAL IMPEDANCE (PIO SUPPRESSION)")
    print("=" * 60)
    from powertrain.virtual_impedance import (
        ImpedanceParams, ImpedanceState, impedance_step,
        impedance_frequency_response,
    )

    params = ImpedanceParams()
    state = ImpedanceState.default()
    dt = jnp.array(0.005)

    # Test 1: Step response — should settle within 100ms
    step_outputs = []
    for i in range(60):  # 300ms
        state, th, br = impedance_step(state, jnp.array(1.0), jnp.array(0.0), dt, params)
        step_outputs.append(float(th))

    # Find 90% rise time
    target_90 = 0.90
    rise_idx = next((i for i, v in enumerate(step_outputs) if v >= target_90), len(step_outputs))
    rise_time_ms = rise_idx * 5  # ms

    if rise_time_ms < 150:
        print(f"[PASS] Step response 90% rise time: {rise_time_ms}ms (budget: <150ms)")
    else:
        print(f"[FAIL] Step response too slow: {rise_time_ms}ms (budget: <150ms)")

    # Test 2: Steady-state accuracy — should reach 1.0 eventually
    final_val = step_outputs[-1]
    if abs(final_val - 1.0) < 0.05:
        print(f"[PASS] Steady-state value: {final_val:.4f} (target: 1.0, error <5%)")
    else:
        print(f"[FAIL] Steady-state error: {final_val:.4f} (target: 1.0)")

    # Test 3: Frequency response at 3 Hz (PIO frequency)
    resp = impedance_frequency_response(
        [3.0], J=params.J_throttle, C=params.C_throttle, K=params.K_throttle,
    )
    phase_3hz = abs(resp['phase_deg'][0])
    mag_3hz = resp['magnitude'][0]

    if phase_3hz > 30.0:
        print(f"[PASS] Phase lag at 3 Hz: {phase_3hz:.1f}° "
              f"(>30° breaks PIO loop, attenuation: {(1-mag_3hz)*100:.1f}%)")
    else:
        print(f"[FAIL] Insufficient phase lag at 3 Hz: {phase_3hz:.1f}° (need >30°)")

    print(f"  Natural freq: {resp['f_n']:.1f} Hz, damping ratio: {resp['zeta']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 16: Full Pipeline JIT Compilation + Timing
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline():
    print("\n" + "=" * 60)
    print("TEST 16: FULL POWERTRAIN PIPELINE (JIT + TIMING)")
    print("=" * 60)
    from powertrain.powertrain_manager import (
        make_powertrain_manager, powertrain_step, PowertrainConfig,
    )

    config, state = make_powertrain_manager()
    dt = jnp.array(0.005)

    # Vehicle state: 15 m/s gentle right turn
    vx = jnp.array(15.0)
    vy = jnp.array(0.3)
    wz = jnp.array(0.2)
    delta = jnp.array(0.08)
    Fz = jnp.array([650., 850., 600., 900.])
    Fy = jnp.array([-200., -300., -150., -250.])
    omega_w = jnp.full(4, 15.0 / 0.2032)
    alpha_t = jnp.array([0.04, 0.05, 0.03, 0.04])
    T_tire = jnp.full(4, 88.0)
    mu_est = jnp.array(1.4)
    gp_sigma = jnp.array(0.05)
    curvature = jnp.array(0.03)

    # JIT compile
    print("  Compiling full pipeline (first call)...")
    t0 = time.perf_counter()
    diag, state = powertrain_step(
        jnp.array(0.6), jnp.array(0.0), delta,
        vx, vy, wz, Fz, Fy, omega_w, alpha_t, T_tire,
        mu_est, gp_sigma, curvature, state, dt, config,
    )
    _ = float(diag.T_wheel[0])
    t_compile = time.perf_counter() - t0
    print(f"  Compile time: {t_compile:.2f}s")

    # Timing: 50 warm steps
    print("  Running 50 warm steps for timing...")
    t0 = time.perf_counter()
    for i in range(50):
        th = jnp.array(0.4 + 0.2 * jnp.sin(i * 0.15))
        diag, state = powertrain_step(
            th, jnp.array(0.0), delta,
            vx, vy, wz, Fz, Fy, omega_w, alpha_t, T_tire,
            mu_est, gp_sigma, curvature, state, dt, config,
        )
    _ = float(diag.T_wheel[0])
    t_run = time.perf_counter() - t0
    ms_per_step = t_run / 50 * 1000

    if ms_per_step < 5.0:
        print(f"[PASS] Per-step time: {ms_per_step:.3f}ms (budget: 5.0ms at 200 Hz)")
    else:
        print(f"[WARN] Per-step time: {ms_per_step:.3f}ms (exceeds 5.0ms budget — "
              f"expected on CPU, OK if <1ms on target SBC)")

    # Check outputs are physically sane
    T_w = [float(diag.T_wheel[i]) for i in range(4)]
    all_finite = all(abs(t) < 1e6 for t in T_w)
    if all_finite:
        print(f"[PASS] Output torques finite: [{T_w[0]:.1f}, {T_w[1]:.1f}, "
              f"{T_w[2]:.1f}, {T_w[3]:.1f}] Nm")
    else:
        print(f"[FAIL] Non-finite output torques detected")

    # Thermal state should have changed slightly
    dT = float(diag.T_motors[0]) - 55.0
    if abs(dT) > 0.0 and abs(dT) < 50.0:
        print(f"[PASS] Thermal dynamics active: ΔT_motor = {dT:+.4f}°C after 50 steps")
    else:
        print(f"[WARN] Thermal state unchanged or excessive: ΔT = {dT:+.4f}°C")

    # SoC should have decreased slightly (driving)
    dSoC = float(diag.SoC) - 95.0
    if dSoC < 0:
        print(f"[PASS] SoC decreasing under load: {float(diag.SoC):.4f}% "
              f"(Δ = {dSoC:+.6f}%)")
    else:
        print(f"[WARN] SoC not decreasing: {float(diag.SoC):.4f}%")

    # Diagnostics completeness
    diag_fields = len(diag._fields)
    print(f"  Diagnostics: {diag_fields} fields available for telemetry/dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "#" * 60)
    print(" PROJECT-GP POWERTRAIN CONTROL: SUBSYSTEM VERIFICATION")
    print("#" * 60)

    test_motor_torque_envelope()
    test_socp_allocator()
    test_cbf_safety()
    test_desc_convergence()
    test_launch_state_machine()
    test_virtual_impedance()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("✅ POWERTRAIN CONTROL STACK VERIFICATION COMPLETE.")
    print("7 tests covering: motor envelope, SOCP feasibility, CBF safety,")
    print("DESC convergence, launch sequencing, PIO suppression, full pipeline.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_all()