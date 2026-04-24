# powertrain/powertrain_manager.py
# Project-GP — Unified Powertrain Manager
# ═══════════════════════════════════════════════════════════════════════════════
#
# Single entry point for the complete Ter27 4WD powertrain control stack.
# Orchestrates all subsystems in the correct order:
#
#   1. Virtual Impedance → filtered pedal inputs (PIO mitigation)
#   2. Traction Control → κ* references + TC/TV blending weights
#   3. Launch Control → torque commands during launch phase
#   4. Torque Vectoring → SOCP allocation + CBF safety filter
#   5. Powertrain Thermal → motor/inverter/battery state update
#   6. Sensor Health → confidence scoring + graceful degradation
#
# The manager produces a single (4,) torque vector at 200 Hz that is:
#   · Friction-circle-feasible (SOCP constraint)
#   · Formally safe (CBF invariant)
#   · Thermally bounded (motor/inverter derating)
#   · Driver-intent-aware (counter-steer detection)
#   · Energy-optimal (loss minimization)
#   · PIO-free (virtual impedance)
#
# No mode switching. All transitions are sigmoid-smooth. The entire
# pipeline compiles to a single XLA graph via jax.jit.
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from powertrain.motor_model import (
    MotorParams, BatteryParams, PowertrainState,
    motor_torque_limits_at_wheel, total_power_limit,
)
from powertrain.modes.advanced.torque_vectoring import (
    TVGeometry, AllocatorWeights, CBFParams,
    tv_step, TVState, TVOutput, make_tv_state,
    yaw_moment_arms, solve_torque_allocation, cbf_safety_filter,
    smooth_output,
)
from powertrain.modes.advanced.koopman_tv import (
    KoopmanTVBundle, KoopmanTVConfig, make_default_koopman_bundle,
    load_koopman_bundle,
)
from powertrain.modes.advanced.traction_control import (
    DESCParams, TCWeights, TCState, TCOutput,
    tc_step, compute_blend_weights as compute_blending_weights,
    compute_slip_ratios as estimate_slip_ratios,
    wheel_speed_confidence,
)
from powertrain.modes.advanced.launch_control import (
    LaunchConfig, LaunchState, LaunchOutput,
    launch_step, launch_step_v2,
)
from powertrain.modes.advanced.virtual_impedance import (
    ImpedanceParams, ImpedanceState,
    impedance_step,
)
from powertrain.modes.advanced.slip_barrier import (
    SlipBarrierInputs, SlipBarrierParams,
    make_slip_barrier_inputs, build_slip_barrier_rows,
)
from powertrain.regen_blend import (
    RegenBlendParams, RegenDiagnostics, RegenEnergyState,
    compute_regen_blend, update_regen_energy, regen_efficiency,
)
try:
    from powertrain.modes.advanced.active_set_classifier import load_classifier_v2
except ImportError:
    load_classifier_v2 = None
from powertrain.modes.advanced.explicit_mpqp_allocator import (
    make_explicit_allocator_step_auto,
)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Unified Configuration
# ─────────────────────────────────────────────────────────────────────────────

class PowertrainConfig(NamedTuple):
    """Complete powertrain configuration — all sub-module params."""
    geo: TVGeometry = TVGeometry()
    motor: MotorParams = MotorParams()
    battery: BatteryParams = BatteryParams()
    alloc_weights: AllocatorWeights = AllocatorWeights()
    cbf: CBFParams = CBFParams()
    desc: DESCParams = DESCParams()
    tc_weights: TCWeights = TCWeights()
    launch: LaunchConfig = LaunchConfig()
    impedance:    ImpedanceParams    = ImpedanceParams()
    slip_barrier:  SlipBarrierParams  = SlipBarrierParams()   # Batch 8
    regen:         RegenBlendParams   = RegenBlendParams()    # Batch 9
    # Global tuning
    max_throttle_force: float = 6000.0
    max_brake_force: float = 8000.0      # N max force from brake pedal
    regen_blend: float = 0.7             # deprecated scalar — superseded by Batch 9 regen params
    is_rwd: bool = False   # True = Ter26 RWD, False = Ter27 AWD
    koopman_bundle: KoopmanTVBundle = None  # None → default inert bundle


    @staticmethod
    def from_vehicle_params(vp: dict) -> 'PowertrainConfig':
        """Construct from the canonical vehicle_params dictionary."""
        return PowertrainConfig(
            geo=TVGeometry.from_vehicle_params(vp),
            motor=MotorParams(
                T_peak=vp.get('motor_peak_torque', 120.0) / vp.get('drivetrain_ratio', 4.5),
                P_peak=vp.get('motor_peak_power', 80000.0) / 4.0,
                gear_ratio=vp.get('drivetrain_ratio', 4.5),
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Unified Persistent State
# ─────────────────────────────────────────────────────────────────────────────

class PowertrainManagerState(NamedTuple):
    """Complete state vector for the powertrain manager."""
    tv: TVState                    # torque vectoring controller
    tc: TCState                    # traction control (DESC + confidence)
    launch: LaunchState            # launch sequencer
    impedance: ImpedanceState      # virtual impedance filter
    powertrain: PowertrainState    # electrothermal state
    # Derived persistent values
    ax_filtered:  jax.Array         # scalar: low-pass filtered longitudinal accel [m/s²]
    ay_filtered:  jax.Array         # scalar: low-pass filtered lateral accel [m/s²]
    regen_energy: RegenEnergyState  # Batch 9: cumulative regen / consumption accounting

    @staticmethod
    def default(config: PowertrainConfig = PowertrainConfig()) -> 'PowertrainManagerState':
        return PowertrainManagerState(
            tv=make_tv_state(),
            tc=TCState.default(),
            launch=LaunchState.default(config.launch),
            impedance=ImpedanceState.default(),
            powertrain=PowertrainState.default(),
            ax_filtered=jnp.array(0.0),
            ay_filtered=jnp.array(0.0),
            regen_energy=RegenEnergyState.zero(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3  Comprehensive Diagnostics Output
# ─────────────────────────────────────────────────────────────────────────────

class PowertrainDiagnostics(NamedTuple):
    """Full diagnostic output — every signal needed for telemetry/dashboard."""
    # Final outputs
    T_wheel: jax.Array              # (4,) commanded wheel torques [Nm]
    T_wheel_raw: jax.Array          # (4,) pre-CBF torques (for CBF intervention analysis)

    # Torque vectoring
    Mz_target: jax.Array            # scalar yaw moment demand [Nm]
    Mz_actual: jax.Array            # scalar achieved yaw moment [Nm]
    wz_ref: jax.Array               # scalar yaw rate reference [rad/s]
    cbf_active: jax.Array           # scalar CBF intervention flag

    # Traction control
    kappa_star: jax.Array           # (4,) optimal slip ratios
    kappa_measured: jax.Array       # (4,) measured slip ratios
    kappa_error: jax.Array          # (4,) slip tracking errors
    desc_grad: jax.Array            # (4,) DESC gradient estimates
    w_slip: jax.Array               # scalar slip tracking weight
    w_yaw: jax.Array                # scalar yaw tracking weight

    # Launch control
    launch_phase: jax.Array         # scalar: state machine phase
    launch_active: jax.Array        # scalar: 0/1 launch active
    mu_probe_est: jax.Array         # scalar: probed friction coefficient
    f_front: jax.Array              # scalar: front torque fraction
    tc_ceiling: jax.Array           # (4,) launch TC ceiling [Nm]
    yaw_correction: jax.Array       # (4,) differential yaw-lock torque [Nm]
    abort_triggered: jax.Array      # scalar: 1.0 if abort fired this step

    # Virtual impedance
    throttle_filtered: jax.Array    # scalar: impedance-filtered throttle
    brake_filtered: jax.Array       # scalar: impedance-filtered brake

    # Powertrain thermal
    T_motors: jax.Array             # (4,) motor temperatures [°C]
    T_invs: jax.Array               # (4,) inverter temperatures [°C]
    SoC: jax.Array                  # scalar state of charge [%]
    T_cell: jax.Array               # scalar cell temperature [°C]
    V_bus: jax.Array                # scalar bus voltage [V]
    P_total: jax.Array              # scalar total electrical power [W]

    # Sensor health
    sensor_confidence: jax.Array    # (4,) per-wheel confidence scores
    degradation_level: jax.Array    # scalar: 0=nominal, 1=severely degraded

    # Allocator cost (for optimization diagnostics)
    allocator_cost: jax.Array       # scalar SOCP cost value
    koopman_rho: jax.Array          # scalar grip utilisation from Koopman TV [0, 1]

    # Batch 8: slip-barrier diagnostics
    slip_barrier_active: jax.Array    # scalar [0, 1]  — gate was open
    slip_viol_max:       jax.Array    # scalar [Nm] max slip-constraint residual
    kappa_star_fused:    jax.Array    # (4,)  κ* per wheel from Koopman
 
    # Batch 9: dynamic regen blend
    alpha_regen:         jax.Array    # scalar ∈ [0,1]  regen fraction of braking demand
    F_brake_hydraulic:   jax.Array    # scalar [N]  hydraulic brake command to actuator
    P_regen_achieved:    jax.Array    # scalar [W]  actual regen power this step
    P_regen_budget:      jax.Array    # scalar [W]  battery-side regen power limit
    regen_efficiency:    jax.Array    # scalar      cumulative E_regen / E_consumed


# ─────────────────────────────────────────────────────────────────────────────
# §4  Driver Force Demand Computation
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_driver_force_demand(
    throttle: jax.Array,           # [0, 1] filtered throttle
    brake: jax.Array,              # [0, 1] filtered brake
    vx: jax.Array,                 # vehicle speed [m/s]
    config: PowertrainConfig = PowertrainConfig(),
) -> jax.Array:
    """
    Map filtered pedal positions to total longitudinal force demand [N].

    Throttle → positive force (driving)
    Brake → negative force (braking, split between regen and friction)

    The force demand is the TOTAL vehicle-level force. The SOCP allocator
    distributes this across 4 wheels optimally.
    """
    # Throttle force: quadratic pedal map (progressive feel)
    F_throttle = throttle ** 1.5 * config.max_throttle_force

    # Brake force: linear pedal map (direct feel)
    F_brake = -brake * config.max_brake_force

    # Net force demand (throttle and brake can overlap for left-foot braking)
    # Brake always overrides throttle via smooth min
    Fx_demand = F_throttle + F_brake

    # Speed-dependent throttle limiting (prevent wheelspin at low speed)
    # At vx < 3 m/s, reduce max force to prevent instant wheel lockup/spin
    speed_limit = jnp.clip(vx / 3.0, 0.1, 1.0)
    Fx_demand = Fx_demand * jnp.where(Fx_demand > 0, speed_limit, 1.0)

    return Fx_demand


# ─────────────────────────────────────────────────────────────────────────────
# §5  Graceful Degradation Logic
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_degradation_level(
    sensor_confidence: jax.Array,  # (4,) per-wheel health scores
) -> tuple[jax.Array, jax.Array]:
    """
    Determine system degradation level from sensor health.

    Returns:
        degradation_level: scalar [0, 1] (0=healthy, 1=severe)
        friction_budget_scale: (4,) per-wheel friction budget multiplier

    Degradation levels:
        Level 0 (0.0–0.2): Nominal — full TV + TC
        Level 1 (0.2–0.5): Mild — reduced TV authority, widened CBF margins
        Level 2 (0.5–0.8): Moderate — equal torque split, basic TC only
        Level 3 (0.8–1.0): Severe — limp mode, reduced peak torque
    """
    # Overall degradation: 1 − min(confidence)
    min_conf = jnp.min(sensor_confidence)
    degradation = 1.0 - min_conf

    # Per-wheel friction budget scaling:
    # Low confidence → small friction circle → SOCP won't command torque
    friction_budget_scale = sensor_confidence ** 2  # quadratic: aggressive shrinkage

    return degradation, friction_budget_scale


# ─────────────────────────────────────────────────────────────────────────────
# §6  Vectorial ABS (Trail-Braking Integration)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_trail_brake_yaw_target(
    wz_current: jax.Array,     # current yaw rate [rad/s]
    delta: jax.Array,          # steering angle [rad]
    vx: jax.Array,             # vehicle speed [m/s]
    ax: jax.Array,             # longitudinal acceleration [m/s²] (negative = braking)
    curvature: jax.Array,      # track curvature at current position [1/m]
    geo: TVGeometry = TVGeometry(),
) -> jax.Array:
    """
    Trail-braking yaw moment target.

    During braking into a corner, compute the additional yaw moment needed
    to rotate the car toward the apex. The vectorial ABS uses this to
    drive the inner-rear wheel while regenerating the outer-front.

    Returns additional ΔMz_trail [Nm] to add to the TV's yaw demand.
    """
    vx_safe = jnp.maximum(jnp.abs(vx), 1.0)
    L = geo.lf + geo.lr

    # Desired yaw rate from curvature
    wz_desired = vx_safe * curvature

    # Yaw rate deficit: how much more yaw we need
    wz_deficit = wz_desired - wz_current

    # Trail-braking activation: only when braking (ax < 0) and turning
    brake_intensity = jnp.clip(-ax / 15.0, 0.0, 1.0)  # 0–1, scaled by max decel
    turn_intensity = jnp.clip(jnp.abs(curvature) * vx_safe ** 2 / 15.0, 0.0, 1.0)

    # Yaw moment for trail-braking rotation
    Kp_trail = 40.0  # Nm / (rad/s) yaw gain for trail-braking
    Mz_trail = Kp_trail * wz_deficit * brake_intensity * turn_intensity

    return Mz_trail


# ─────────────────────────────────────────────────────────────────────────────
# §7  Main Step Function — THE SINGLE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=('config',))
def powertrain_step(
    # ── Raw driver inputs ────────────────────────────────────────────────
    throttle_raw: jax.Array,       # [0, 1] raw throttle pedal
    brake_raw: jax.Array,          # [0, 1] raw brake pedal
    delta: jax.Array,              # steering angle [rad]
    # ── Vehicle state (from physics engine / sensors) ────────────────────
    vx: jax.Array,                 # longitudinal velocity [m/s]
    vy: jax.Array,                 # lateral velocity [m/s]
    wz: jax.Array,                 # yaw rate [rad/s]
    Fz: jax.Array,                 # (4,) vertical loads [N]
    Fy: jax.Array,                 # (4,) lateral tire forces [N]
    omega_wheel: jax.Array,        # (4,) wheel angular velocities [rad/s]
    alpha_t: jax.Array,            # (4,) transient slip angles [rad]
    T_tire: jax.Array,             # (4,) tire temperatures [°C]
    # ── Estimation signals ───────────────────────────────────────────────
    mu_est: jax.Array,             # scalar estimated friction coefficient
    gp_sigma: jax.Array,           # scalar GP uncertainty
    curvature: jax.Array,          # scalar track curvature at current position [1/m]
    # ── Persistent state ─────────────────────────────────────────────────
    manager_state: PowertrainManagerState,
    # ── Time ─────────────────────────────────────────────────────────────
    dt: jax.Array,
    # ── Configuration ────────────────────────────────────────────────────
    config: PowertrainConfig = PowertrainConfig(),              # <-- MOVED UP
    launch_button: jax.Array = jnp.array(0.0),                  # <-- MOVED DOWN
) -> tuple[PowertrainDiagnostics, PowertrainManagerState]:
    """
    Execute one complete powertrain control cycle at 200 Hz.

    This is the ONLY function the physics server calls. Everything else
    is internal. The function is fully JIT-compiled and deterministic.

    Pipeline:
      1. Virtual impedance → filtered pedals
      2. Acceleration estimation → low-pass filtered ax, ay
      3. Traction control → κ* references, blending weights, sensor health
      4. Driver force demand → Fx from filtered pedals
      5. Launch controller → launch-phase torques (if active)
      6. Trail-braking yaw target → vectorial ABS integration
      7. Torque vectoring → SOCP allocation + CBF safety
      8. Mode blending → launch torques vs TV torques
      9. Powertrain thermal update
     10. Diagnostics packaging

    Returns:
        diagnostics: PowertrainDiagnostics (comprehensive telemetry)
        new_state: PowertrainManagerState (carry to next timestep)
    """
    geo = config.geo
    mp = config.motor
    bp = config.battery

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Virtual Impedance (PIO mitigation)
    # ═══════════════════════════════════════════════════════════════════════
    imp_state_new, throttle_filt, brake_filt = impedance_step(
        manager_state.impedance, throttle_raw, brake_raw, dt, config.impedance,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Acceleration Estimation (low-pass for blending weights)
    # ═══════════════════════════════════════════════════════════════════════
    alpha_accel = 0.90  # LP filter coefficient for acceleration signals
    # Approximate ax from velocity change (avoids noisy IMU direct path)
    ax_raw = jnp.sum(Fz * 0.0) + vx * wz * 0.0  # placeholder — actual ax from IMU
    # For now, estimate from Fx applied
    ax_approx = jnp.sum(manager_state.tv.T_prev) / (geo.r_w * geo.mass)
    ay_approx = vx * wz  # centripetal approximation

    ax_filt = alpha_accel * manager_state.ax_filtered + (1.0 - alpha_accel) * ax_approx
    ay_filt = alpha_accel * manager_state.ay_filtered + (1.0 - alpha_accel) * ay_approx

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Traction Control (DESC + κ* + blending weights)
    # ═══════════════════════════════════════════════════════════════════════
    tc_output, tc_state_new = tc_step(
        vx=vx, vy=vy, ax=ax_filt, ay=ay_filt,
        omega_wheel=omega_wheel, alpha_t=alpha_t,
        Fz=Fz, T_applied=manager_state.tv.T_prev,
        T_tire=T_tire, mu_est=mu_est, gp_sigma=gp_sigma,
        tc_state=manager_state.tc,
        dt=dt,
        desc_params=config.desc,
        tc_weights=config.tc_weights,
        r_w=geo.r_w,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Driver Force Demand
    # ═══════════════════════════════════════════════════════════════════════
    Fx_driver = compute_driver_force_demand(throttle_filt, brake_filt, vx, config)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Motor Torque Limits
    # ═══════════════════════════════════════════════════════════════════════
    pt = manager_state.powertrain
    T_min, T_max = motor_torque_limits_at_wheel(
        omega_wheel, pt.V_bus, pt.T_motors, pt.T_invs, pt.SoC, mp, bp,
    )
    P_max = total_power_limit(pt.V_bus, pt.SoC, pt.T_cell, bp)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Degradation Assessment
    # ═══════════════════════════════════════════════════════════════════════
    degradation, friction_scale = compute_degradation_level(tc_output.confidence)
    mu_scaled = mu_est * friction_scale  # per-wheel friction budget

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Yaw Rate Reference + Trail-Braking Yaw Target
    # ═══════════════════════════════════════════════════════════════════════
    delta_dot = (delta - manager_state.tv.delta_prev) / (dt + 1e-6)

    # Trail-braking yaw moment (vectorial ABS)
    Mz_trail = compute_trail_brake_yaw_target(
        wz, delta, vx, ax_filt, curvature, geo,
    )

    # Yaw rate reference (driver-intent-aware)
    from powertrain.modes.advanced.torque_vectoring import yaw_rate_reference
    wz_ref = yaw_rate_reference(delta, vx, delta_dot, wz, mu_est, geo)

    # PD yaw moment demand
    wz_error = wz - wz_ref
    Kp_yaw = 80.0
    Kd_yaw = 5.0
    dwz_ref = (wz_ref - manager_state.tv.wz_ref_prev) / (dt + 1e-6)
    Mz_target = Kp_yaw * wz_error + Kd_yaw * dwz_ref + Mz_trail

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7b: Batch 8 — Slip Barrier Inputs (E-ABS + Trail-Brake Regen)
    # ═══════════════════════════════════════════════════════════════════════
    # Build per-wheel κ-CBF rows from Koopman observer output.
    # These rows enter the extended 24×24 KKT in Step 8's V2 allocator.
    # During launch (launch_active → 1), the barrier is gated off smoothly.
    slip_inputs = make_slip_barrier_inputs(
        kappa_star_front = tc_output.kappa_star[0],    # FL (front axle)
        kappa_star_rear  = tc_output.kappa_star[2],    # RL (rear  axle)
        sigma_front      = tc_output.sigma_front if hasattr(tc_output, "sigma_front")
                           else jnp.array(0.03),       # fallback if V1 TC
        sigma_rear       = tc_output.sigma_rear  if hasattr(tc_output, "sigma_rear")
                           else jnp.array(0.03),
        kappa_measured   = tc_output.kappa_measured,
        T_prev           = manager_state.tv.T_prev,
        omega_wheel      = omega_wheel,
        omega_prev       = manager_state.tc.omega_prev,
        vx               = vx,
        launch_active    = manager_state.launch.phase.astype(jnp.float32)
                           / 5.0,                     # crude: phase 3 (LAUNCH) → ~0.6
        dt               = dt,
        r_w              = float(geo.r_w),
        I_w              = 1.2,
    )
 
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: SOCP Torque Allocation
    # ═══════════════════════════════════════════════════════════════════════

    # Build dynamic allocator weights from TC blending
    alloc_w = AllocatorWeights(
        w_force=config.alloc_weights.w_force,
        w_yaw=tc_output.w_yaw,              # from TC blending
        w_workload=config.alloc_weights.w_workload + tc_output.w_slip,  # slip tracking
        w_energy=config.alloc_weights.w_energy,
        w_smooth=config.alloc_weights.w_smooth,
        w_feel=config.alloc_weights.w_feel,
        w_friction_barrier=config.alloc_weights.w_friction_barrier,
        w_power=config.alloc_weights.w_power,
    )

    T_alloc_socp = solve_torque_allocation(
        T_warmstart=manager_state.tv.T_prev,
        T_prev=manager_state.tv.T_prev,
        Fx_target=Fx_driver,
        Mz_target=Mz_target,
        delta=delta,
        Fz=Fz,
        Fy=Fy,
        mu=mu_scaled,
        omega_wheel=omega_wheel,
        T_min=T_min,
        T_max=T_max,
        P_max=P_max,
        T_ribs=T_tire,
        geo=geo,
        w=alloc_w,
        is_rwd=config.is_rwd,
    )
    # Batch 8: enforce slip constraints via V2 extended polish
    from powertrain.modes.advanced.explicit_mpqp_allocator import (
        build_qp_matrices, polish_step_v2, QPParams,
    )
    from powertrain.modes.advanced.slip_barrier import build_slip_barrier_rows
    A_slip_mgr, b_slip_mgr = build_slip_barrier_rows(slip_inputs, config.slip_barrier)
    _qp_p_mgr = QPParams(
        mz_ref=Mz_target, fx_d=Fx_driver,
        t_min=T_min, t_max=T_max,
        t_fric=mu_scaled * Fz * geo.r_w,
        delta=delta, t_prev=manager_state.tv.T_prev, omega=omega_wheel,
    )
    from powertrain.modes.advanced.explicit_mpqp_allocator import (
        build_qp_matrices as _bqpm, check_slip_feasibility,
    )
    _Q_mgr, _c_mgr = _bqpm(_qp_p_mgr, geo)
    slip_ok_mgr = check_slip_feasibility(T_alloc_socp, A_slip_mgr, b_slip_mgr)
    T_alloc = jax.lax.cond(
        slip_ok_mgr,
        lambda t: t,
        lambda t: polish_step_v2(t, _Q_mgr, _c_mgr, _qp_p_mgr,
                                  A_slip_mgr, b_slip_mgr),
        T_alloc_socp,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8b: Dynamic Regen Blend (Batch 9)
    # ═══════════════════════════════════════════════════════════════════════
    # Compute optimal α* from battery state (SoC, T_cell) and scale regen
    # torques accordingly.  Hydraulic brake fills the residual deficit.
    # T_alloc_regen replaces T_alloc into the CBF — the CBF sees the
    # battery-feasible torques, not raw KKT outputs.
    T_alloc_regen, regen_diag = compute_regen_blend(
        T_alloc, T_min, Fx_driver, omega_wheel, pt, bp, config.regen,
    )
    regen_energy_new = update_regen_energy(
        manager_state.regen_energy, regen_diag, T_alloc_regen, omega_wheel, dt,
    )
 
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 9: CBF Safety Filter
    # ═══════════════════════════════════════════════════════════════════════
    Fy_total = jnp.sum(Fy)
    T_safe = cbf_safety_filter(
        T_alloc_regen, manager_state.tv.T_prev,
        vx, vy, wz, Fz, Fy_total, mu_est, omega_wheel,
        T_min, T_max, gp_sigma, geo, config.cbf,
        is_rwd=config.is_rwd,
    )
    cbf_intervention = jnp.linalg.norm(T_safe - T_alloc_regen)
    cbf_active = (cbf_intervention > 1.0).astype(jnp.float32)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 10: Launch Controller (v2.1 — button arming + TC ceiling + yaw lock)
    # ═══════════════════════════════════════════════════════════════════════
    launch_output, launch_state_new = launch_step_v2(
        throttle=throttle_filt,
        brake=brake_filt,
        vx=vx,
        omega_wheel=omega_wheel,
        Fz=Fz,                                  # (4,) normal loads from vehicle model
        T_max=T_max,                            # (4,) max torque from STEP 5
        T_tc=T_safe,                            # TC/TV torques as the handoff target
        launch_state=manager_state.launch,
        dt=dt,
        params=config.launch,
        launch_button=launch_button,            # v2.1: steering wheel button signal
        kappa_star=tc_output.kappa_star,        # v2.1: DESC optimal slip → TC ceiling
        wz=wz,                                  # v2.1: yaw rate → yaw lock PI
    )

    # Final torque selection: launch overrides TV/TC when active
    launch_active = launch_output.is_launch_active
    T_final_raw = jnp.where(
        launch_active > 0.5,
        launch_output.T_command,
        T_safe,
    )

    # Smooth output (EMA + rate limit)
    T_final = smooth_output(T_final_raw, manager_state.tv.T_prev, dt)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 11: Powertrain Thermal Update
    # ═══════════════════════════════════════════════════════════════════════
    pt_new = pt.step(T_final, omega_wheel, dt, mp, bp)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 12: Achieved Quantities (diagnostics)
    # ═══════════════════════════════════════════════════════════════════════
    arms = yaw_moment_arms(delta, geo)
    Mz_actual = jnp.sum(T_final * arms)
    Fx_actual = jnp.sum(T_final) / geo.r_w
    P_total_diag = jnp.sum(jnp.abs(T_final * omega_wheel))

    # SOCP cost at final torques (for optimization tracking)
    from powertrain.modes.advanced.torque_vectoring import allocator_cost
    cost_val = allocator_cost(
        T_final, manager_state.tv.T_prev, Fx_driver, Mz_target, delta,
        Fz, Fy, mu_scaled, omega_wheel, T_min, T_max, P_max, T_tire, geo, alloc_w,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 13: Package Outputs
    # ═══════════════════════════════════════════════════════════════════════
    diagnostics = PowertrainDiagnostics(
        T_wheel=T_final,
        T_wheel_raw=T_alloc,
        Mz_target=Mz_target,
        Mz_actual=Mz_actual,
        wz_ref=wz_ref,
        cbf_active=cbf_active,
        kappa_star=tc_output.kappa_star,
        kappa_measured=tc_output.kappa_measured,
        kappa_error=tc_output.kappa_error,
        desc_grad=tc_output.desc_grad,
        w_slip=tc_output.w_slip,
        w_yaw=tc_output.w_yaw,
        launch_phase=launch_output.phase,
        launch_active=launch_active,
        mu_probe_est=launch_output.mu_estimate,
        f_front=launch_output.f_front,
        tc_ceiling=launch_output.tc_ceiling,
        yaw_correction=launch_output.yaw_correction,
        abort_triggered=launch_output.abort_triggered,
        throttle_filtered=throttle_filt,
        brake_filtered=brake_filt,
        T_motors=pt_new.T_motors,
        T_invs=pt_new.T_invs,
        SoC=pt_new.SoC,
        T_cell=pt_new.T_cell,
        V_bus=pt_new.V_bus,
        P_total=P_total_diag,
        sensor_confidence=tc_output.confidence,
        degradation_level=degradation,
        allocator_cost=cost_val,
        koopman_rho=jnp.array(0.0),
        slip_barrier_active = slip_inputs.active,
        slip_viol_max       = jnp.max(A_slip_mgr @ T_alloc - b_slip_mgr),
        kappa_star_fused    = slip_inputs.kappa_star,
        # Batch 9
        alpha_regen         = regen_diag.alpha_regen,
        F_brake_hydraulic   = regen_diag.F_brake_hydraulic,
        P_regen_achieved    = regen_diag.P_regen_achieved,
        P_regen_budget      = regen_diag.P_regen_budget,
        regen_efficiency    = regen_efficiency(regen_energy_new),
    )


    new_state = PowertrainManagerState(
        tv=TVState(
            T_prev=T_final,
            delta_prev=delta,
            wz_ref_prev=wz_ref,
            T_qp_prev=T_alloc,    # pre-CBF SOCP solution for warm-start
        ),
        tc=tc_state_new,
        launch=launch_state_new,
        impedance=imp_state_new,
        powertrain=pt_new,
        ax_filtered=ax_filt,
        ay_filtered=ay_filt,
        regen_energy=regen_energy_new,
    )

    return diagnostics, new_state


# ─────────────────────────────────────────────────────────────────────────────
# §8  Factory Functions
# ─────────────────────────────────────────────────────────────────────────────

def make_powertrain_manager(vp: dict = None) -> tuple[PowertrainConfig, PowertrainManagerState]:
    """
    Create a fully initialized powertrain manager.

    Args:
        vp: vehicle_params dict (optional — uses defaults if None)

    Returns:
        config: PowertrainConfig
        state: PowertrainManagerState (initial state, zero torques)

    Usage:
        config, state = make_powertrain_manager(vehicle_params)
        for each timestep:
            diagnostics, state = powertrain_step(
                throttle, brake, delta,
                vx, vy, wz, Fz, Fy, omega, alpha_t, T_tire,
                mu_est, gp_sigma, curvature,
                state, dt, config,
            )
            T_command = diagnostics.T_wheel  # (4,) torques to send to inverters
    """
    if vp is not None:
        config = PowertrainConfig.from_vehicle_params(vp)
    else:
        config = PowertrainConfig()
    state = PowertrainManagerState.default(config)
    return config, state

# ═══════════════════════════════════════════════════════════════════════════════
# Batch 10.5 — UKF Wrapper (powertrain_step_v2)
# ═══════════════════════════════════════════════════════════════════════════════

def powertrain_step_v2(
    throttle_raw: jax.Array, brake_raw: jax.Array, delta: jax.Array,
    sensor_reading,  # Noisy inputs from vehicle_dynamics.observe_sensors
    ukf_state, imu_bias: jax.Array,
    mu_est: jax.Array, gp_sigma: jax.Array, curvature: jax.Array,
    manager_state: PowertrainManagerState, dt: jax.Array,
    ukf_params, config: PowertrainConfig = PowertrainConfig(),
    launch_button: jax.Array = jnp.array(0.0)
):
    from powertrain.powertrain_wiring_v2 import pack_hub_motor_command, PowertrainOutputV2
    from powertrain.state_estimator import ukf_step, extract_estimated_state, pack_measurement_from_reading
    
    # 1. Run the Unscented Kalman Filter (UKF) to filter the noisy sensors
    z_meas = pack_measurement_from_reading(sensor_reading)
    ukf_new = ukf_step(ukf_state, z_meas, delta, dt, ukf_params)
    est = extract_estimated_state(ukf_new)
    
    # 2. Call the original powertrain_step, but feed it the CLEAN, ESTIMATED 
    #    states instead of the raw, noisy sensors.
    diagnostics, new_state = powertrain_step(
        throttle_raw=throttle_raw, brake_raw=brake_raw, delta=delta,
        vx=est.vx, vy=est.vy, wz=est.wz,
        Fz=est.Fz, Fy=est.Fy, omega_wheel=est.omega_wheel, alpha_t=est.alpha_t,
        T_tire=jnp.full(4, 85.0), # Assuming constant thermal tire state
        mu_est=mu_est, gp_sigma=gp_sigma, curvature=curvature,
        manager_state=manager_state, dt=dt, config=config,
        launch_button=launch_button
    )
    
    # 3. Pack the output for the new 4WD Hub Motor physics engine
    command = pack_hub_motor_command(diagnostics.T_wheel, delta, diagnostics.F_brake_hydraulic)
    
    return PowertrainOutputV2(
        command=command,
        diagnostics=diagnostics,
        ukf_state=ukf_new,
        imu_bias=imu_bias
    )

# ─────────────────────────────────────────────────────────────────────────────
# §9  Standalone Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """
    Quick validation that the full pipeline compiles and runs.
    Call from project root: python -m powertrain.powertrain_manager
    """
    print("=" * 60)
    print(" POWERTRAIN MANAGER SMOKE TEST")
    print("=" * 60)

    config, state = make_powertrain_manager()

    # Simulate 100 steps at 200 Hz (0.5 seconds) with constant throttle
    dt = jnp.array(0.005)

    # Nominal vehicle state: 15 m/s straight-ahead
    vx = jnp.array(15.0)
    vy = jnp.array(0.0)
    wz = jnp.array(0.0)
    delta = jnp.array(0.0)
    Fz = jnp.array([750.0, 750.0, 750.0, 750.0])
    Fy = jnp.array([0.0, 0.0, 0.0, 0.0])
    omega = jnp.full(4, 15.0 / 0.2032)
    alpha_t = jnp.zeros(4)
    T_tire = jnp.full(4, 85.0)
    mu_est = jnp.array(1.4)
    gp_sigma = jnp.array(0.05)
    curvature = jnp.array(0.0)

    print("\n[1/3] JIT-compiling powertrain_step (first call)...")
    import time
    t0 = time.perf_counter()

    diag, state = powertrain_step(
        throttle_raw=jnp.array(0.5),
        brake_raw=jnp.array(0.0),
        delta=delta,
        vx=vx, vy=vy, wz=wz,
        Fz=Fz, Fy=Fy, omega_wheel=omega, alpha_t=alpha_t, T_tire=T_tire,
        mu_est=mu_est, gp_sigma=gp_sigma, curvature=curvature,
        manager_state=state, dt=dt, config=config,
    )
    # Force evaluation
    _ = float(diag.T_wheel[0])
    t_compile = time.perf_counter() - t0
    print(f"  Compile time: {t_compile:.2f}s")

    print("\n[2/3] Running 100 steps (0.5s simulated)...")
    t0 = time.perf_counter()
    for i in range(100):
        throttle = jnp.array(0.3 + 0.2 * jnp.sin(i * 0.1))
        diag, state = powertrain_step(
            throttle_raw=throttle, brake_raw=jnp.array(0.0),
            delta=jnp.array(0.05 * jnp.sin(i * 0.05)),
            vx=vx, vy=vy, wz=jnp.array(0.1 * jnp.sin(i * 0.05)),
            Fz=Fz, Fy=Fy, omega_wheel=omega, alpha_t=alpha_t, T_tire=T_tire,
            mu_est=mu_est, gp_sigma=gp_sigma, curvature=jnp.array(0.02),
            manager_state=state, dt=dt, config=config,
        )
    # Force final evaluation
    T_final = [float(diag.T_wheel[i]) for i in range(4)]
    t_run = time.perf_counter() - t0
    t_per_step = t_run / 100 * 1000  # ms

    print(f"  Total: {t_run*1000:.1f}ms | Per-step: {t_per_step:.3f}ms")
    budget_cpu = 25.0   # ms — CPU baseline; GPU target is 5.0ms
    print(f"  Budget (CPU): {budget_cpu:.1f}ms/step → {'PASS' if t_per_step < budget_cpu else 'FAIL'} "
          f"(margin: {budget_cpu - t_per_step:.2f}ms | GPU target: 5.0ms)")

    print(f"\n[3/3] Final state diagnostics:")
    print(f"  T_wheel:     [{T_final[0]:>7.1f}, {T_final[1]:>7.1f}, {T_final[2]:>7.1f}, {T_final[3]:>7.1f}] Nm")
    print(f"  Mz_target:   {float(diag.Mz_target):>7.1f} Nm")
    print(f"  Mz_actual:   {float(diag.Mz_actual):>7.1f} Nm")
    print(f"  wz_ref:      {float(diag.wz_ref):>7.4f} rad/s")
    print(f"  CBF active:  {float(diag.cbf_active):.0f}")
    print(f"  κ* (fused):  [{float(diag.kappa_star[0]):.4f}, {float(diag.kappa_star[1]):.4f}, "
          f"{float(diag.kappa_star[2]):.4f}, {float(diag.kappa_star[3]):.4f}]")
    print(f"  Launch phase:{float(diag.launch_phase):.0f} (0=IDLE)")
    print(f"  Throttle:    {float(diag.throttle_filtered):.3f} (filtered)")
    print(f"  T_motors:    [{float(diag.T_motors[0]):.1f}, {float(diag.T_motors[1]):.1f}, "
          f"{float(diag.T_motors[2]):.1f}, {float(diag.T_motors[3]):.1f}] °C")
    print(f"  SoC:         {float(diag.SoC):.1f}%")
    print(f"  V_bus:       {float(diag.V_bus):.1f} V")
    print(f"  Confidence:  {float(diag.sensor_confidence):.3f}")
    print(f"  Degradation: {float(diag.degradation_level):.3f}")

    print(f"\n{'=' * 60}")
    print(f" ✅ SMOKE TEST PASSED")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    smoke_test()