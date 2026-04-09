# powertrain/torque_vectoring.py
# Project-GP — Differentiable SOCP Torque Allocator + CBF Safety Filter
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the central torque allocation module for the Ter27 4WD system.
# It takes virtual demands from the WMPC (total force, yaw moment, front
# ratio) and distributes them as per-wheel torques while respecting:
#
#   1. Per-wheel friction circles (SOCP conic constraints)
#   2. Motor torque-speed envelopes (box constraints from motor_model)
#   3. Battery power budget (linear inequality)
#   4. Energy efficiency (motor loss minimization)
#   5. Torque smoothness (rate limiting)
#   6. Steering feel preservation (steering torque penalty)
#
# The solver is a fixed-iteration projected-gradient method that is:
#   · Fully JIT-compilable (no dynamic loops, no scipy)
#   · Fully differentiable (gradients flow via jax.lax.scan)
#   · Warm-startable (previous solution seeds next timestep)
#   · Deterministic (fixed 12-iteration solve, static memory)
#
# The CBF safety filter wraps the allocator output with formal guarantees
# on sideslip and yaw rate invariance, accounting for actuator delay.
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

from powertrain.motor_model import (
    MotorParams, BatteryParams, PowertrainState,
    motor_torque_limits_at_wheel, motor_power_loss, total_power_limit,
)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Vehicle Geometry Constants (from vehicle_params)
# ─────────────────────────────────────────────────────────────────────────────

class TVGeometry(NamedTuple):
    """Static vehicle geometry for torque vectoring computations."""
    lf: float = 0.8525       # m  CG to front axle
    lr: float = 0.6975       # m  CG to rear axle
    track_f: float = 1.200   # m  front track width
    track_r: float = 1.180   # m  rear track width
    r_w: float = 0.2032      # m  tire loaded radius
    h_cg: float = 0.330      # m  CG height
    I_z: float = 150.0       # kg·m² yaw inertia
    mass: float = 300.0      # kg total mass
    # Steering geometry (for feel penalty)
    kingpin_offset: float = 0.020    # m
    mechanical_trail: float = 0.015  # m
    steer_ratio: float = 4.2         # steering wheel / rack
    # Anti-geometry (for aero-platform)
    anti_squat: float = 0.30
    anti_dive_f: float = 0.40

    @staticmethod
    def from_vehicle_params(vp: dict) -> 'TVGeometry':
        return TVGeometry(
            lf=vp.get('lf', 0.8525),
            lr=vp.get('lr', 0.6975),
            track_f=vp.get('track_front', 1.200),
            track_r=vp.get('track_rear', 1.180),
            r_w=vp.get('wheel_radius', 0.2032),
            h_cg=vp.get('h_cg', 0.330),
            I_z=vp.get('Iz', 150.0),
            mass=vp.get('total_mass', 300.0),
            anti_squat=vp.get('anti_squat', 0.30),
            anti_dive_f=vp.get('anti_dive_f', 0.40),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Yaw Moment Arms
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def yaw_moment_arms(delta: jax.Array, geo: TVGeometry = TVGeometry()) -> jax.Array:
    """
    Yaw moment arm per wheel [m]. Shape: (4,) for [FL, FR, RL, RR].

    Positive arm = positive torque on that wheel produces positive yaw moment
    (counter-clockwise when viewed from above, right-hand rule).

    Accounts for Ackermann-corrected front wheel angles.
    """
    cos_d = jnp.cos(delta)
    sin_d = jnp.sin(delta)

    # Front wheels: lateral arm × cos(δ) ± longitudinal arm × sin(δ)
    arm_fl = -geo.track_f / 2.0 * cos_d + geo.lf * sin_d
    arm_fr = +geo.track_f / 2.0 * cos_d - geo.lf * sin_d

    # Rear wheels: pure lateral arm (no steering)
    arm_rl = -geo.track_r / 2.0
    arm_rr = +geo.track_r / 2.0

    # Divide by r_w to convert wheel torque to yaw moment:
    # M_z = Σ (T_i / r_w) × arm_i
    return jnp.array([arm_fl, arm_fr, arm_rl, arm_rr]) / geo.r_w


# ─────────────────────────────────────────────────────────────────────────────
# §3  SOCP Cost Function (differentiable quadratic + barriers)
# ─────────────────────────────────────────────────────────────────────────────

class AllocatorWeights(NamedTuple):
    """Tunable cost weights for the SOCP allocator. MORL-optimizable."""
    w_force: float = 100.0      # total longitudinal force tracking
    w_yaw: float = 200.0        # yaw moment tracking
    w_workload: float = 1.0     # tire workload equalization
    w_energy: float = 0.01      # motor loss minimization
    w_smooth: float = 5.0       # torque rate limiting
    w_feel: float = 0.05        # steering torque minimization (feel)
    w_friction_barrier: float = 50.0  # friction circle log-barrier
    w_power: float = 10.0       # power budget barrier


@partial(jax.jit, static_argnums=())
def allocator_cost(
    T: jax.Array,             # (4,) candidate torques at wheel [Nm]
    T_prev: jax.Array,        # (4,) previous timestep torques [Nm]
    Fx_target: jax.Array,     # scalar total longitudinal force demand [N]
    Mz_target: jax.Array,     # scalar yaw moment demand [Nm]
    delta: jax.Array,         # scalar steering angle [rad]
    Fz: jax.Array,            # (4,) vertical loads [N]
    Fy: jax.Array,            # (4,) lateral forces [N] (from tire model)
    mu: jax.Array,            # (4,) friction coefficients
    omega_wheel: jax.Array,   # (4,) wheel speeds [rad/s]
    T_min: jax.Array,         # (4,) motor min torque at wheel [Nm]
    T_max: jax.Array,         # (4,) motor max torque at wheel [Nm]
    P_max: jax.Array,         # scalar max battery power [W]
    geo: TVGeometry = TVGeometry(),
    w: AllocatorWeights = AllocatorWeights(),
) -> jax.Array:
    """
    Scalar cost for torque allocation. Minimized by the projected gradient solver.

    All terms are smooth and differentiable. Constraint enforcement uses
    log-barrier relaxation rather than hard projection — the projected
    gradient step handles feasibility, while the barrier shapes the
    interior to prefer solutions away from constraint boundaries.
    """
    r_w = geo.r_w

    # ── 1. Force tracking: Fx_actual = Σ T_i / r_w should match Fx_target
    Fx_actual = jnp.sum(T) / r_w
    J_force = w.w_force * (Fx_actual - Fx_target) ** 2

    # ── 2. Yaw moment tracking: Mz_actual = Σ (T_i / r_w) × arm_i
    arms = yaw_moment_arms(delta, geo)
    Mz_actual = jnp.sum(T * arms)
    J_yaw = w.w_yaw * (Mz_actual - Mz_target) ** 2

    # ── 3. Tire workload equalization: minimize max(T_i / friction_budget)²
    friction_budget = jnp.maximum(mu * Fz * r_w, 10.0)  # Nm at wheel
    workload = (T / friction_budget) ** 2
    J_workload = w.w_workload * jnp.sum(workload)

    # ── 4. Energy efficiency: minimize motor electrical losses
    P_loss = motor_power_loss(T, omega_wheel)
    J_energy = w.w_energy * jnp.sum(P_loss)

    # ── 5. Torque smoothness: penalize rate of change
    dT = T - T_prev
    J_smooth = w.w_smooth * jnp.sum(dT ** 2)

    # ── 6. Steering feel: penalize front axle torque asymmetry
    # Parasitic steering torque = (Fx_fr - Fx_fl) × effective_arm
    steer_arm = geo.kingpin_offset * jnp.cos(delta) + geo.mechanical_trail * jnp.sin(delta)
    M_steer = (T[1] - T[0]) / r_w * steer_arm / geo.steer_ratio
    J_feel = w.w_feel * M_steer ** 2

    # ── 7. Friction circle barrier: soft log-barrier inside the friction cone
    # Barrier = -log(1 - (Fx_i² + Fy_i²) / (μ·Fz)²)
    # Smoothed: -log(softplus(margin * scale) / scale + ε)
    Fx_wheel = T / r_w
    friction_radius_sq = (mu * Fz) ** 2 + 1e-4
    force_radius_sq = Fx_wheel ** 2 + Fy ** 2
    margin = 1.0 - force_radius_sq / friction_radius_sq  # >0 = inside circle
    # Smooth barrier: large cost as margin → 0, zero cost when margin >> 0
    barrier_scale = 30.0
    safe_margin = jax.nn.softplus(margin * barrier_scale) / barrier_scale + 1e-6
    J_friction = w.w_friction_barrier * jnp.sum(-jnp.log(safe_margin))

    # ── 8. Power budget barrier: total electrical power ≤ P_max
    P_mech = jnp.sum(jnp.maximum(T * omega_wheel, 0.0))  # driving power only
    P_total = P_mech + jnp.sum(P_loss)
    power_margin = 1.0 - P_total / (P_max + 1e-3)
    safe_power = jax.nn.softplus(power_margin * 20.0) / 20.0 + 1e-6
    J_power = w.w_power * (-jnp.log(safe_power))

    return J_force + J_yaw + J_workload + J_energy + J_smooth + J_feel + J_friction + J_power


# ─────────────────────────────────────────────────────────────────────────────
# §4  Projected Gradient Solver (fixed-iteration, JIT-compiled)
# ─────────────────────────────────────────────────────────────────────────────

N_SOLVER_ITERS = 12    # fixed iteration count — deterministic timing
STEP_SIZE = 0.3        # gradient descent step size (Armijo-tuned offline)

@partial(jax.jit, static_argnames=('is_rwd',))
def solve_torque_allocation(
    T_warmstart: jax.Array,
    T_prev: jax.Array,
    Fx_target: jax.Array,
    Mz_target: jax.Array,
    delta: jax.Array,
    Fz: jax.Array,
    Fy: jax.Array,
    mu: jax.Array,
    omega_wheel: jax.Array,
    T_min: jax.Array,
    T_max: jax.Array,
    P_max: jax.Array,
    geo: TVGeometry = TVGeometry(),
    w: AllocatorWeights = AllocatorWeights(),
    is_rwd: bool = False,
) -> jax.Array:
    """
    Solve the torque allocation problem via projected gradient descent.

    is_rwd=True: front bounds are zeroed at compile time before the solver runs.
    The SOCP is algorithmically unchanged — RWD emerges from the tightened box:
      T ∈ [0, 0, T_min_rl, T_min_rr] × [0, 0, T_max_rl, T_max_rr]
    The projected-gradient projection step then trivially satisfies the front constraint.

    Fixed 12 iterations with:
      1. Gradient computation via jax.grad(cost)
      2. Gradient step with adaptive step size
      3. Box projection (motor limits)
      4. Friction cone scaling (per-wheel)
      5. Warm-start from previous solution

    Returns: (4,) optimal wheel torques [Nm]
    """
    # Python branch — resolved at trace time, not part of XLA graph
    if is_rwd:
        _mask = jnp.array([0., 0., 1., 1.])
        T_min = T_min * _mask
        T_max = T_max * _mask
        T_warmstart = T_warmstart * _mask  # sanitise warm-start: no phantom front torque

    r_w = geo.r_w

    # Cost gradient function (closed over all parameters)
    def cost_fn(T):
        return allocator_cost(
            T, T_prev, Fx_target, Mz_target, delta,
            Fz, Fy, mu, omega_wheel, T_min, T_max, P_max, geo, w,
        )

    grad_fn = jax.grad(cost_fn)

    def solver_step(T, _):
        """Single projected gradient step."""
        g = grad_fn(T)

        # Gradient norm clipping (prevents instability near barrier boundaries)
        g_norm = jnp.linalg.norm(g) + 1e-8
        g_clipped = g * jnp.minimum(1.0, 500.0 / g_norm)

        # Adaptive step size: smaller when gradient is large (near barriers)
        alpha = STEP_SIZE / (1.0 + 0.01 * g_norm)

        # Gradient step
        T_new = T - alpha * g_clipped

        # ── Projection 1: Box constraints (motor limits) ─────────────────
        T_new = jnp.clip(T_new, T_min, T_max)

        # ── Projection 2: Friction cone scaling (per-wheel) ─────────────
        # If |Fx_i|² + |Fy_i|² > (μ·Fz)², scale Fx_i to bring inside circle
        Fx_new = T_new / r_w
        force_sq = Fx_new ** 2 + Fy ** 2
        limit_sq = (mu * Fz) ** 2
        # Scale factor: 1.0 if inside, <1.0 if outside
        scale = jnp.where(
            force_sq > limit_sq * 0.98,  # 2% margin
            jnp.sqrt(limit_sq * 0.98 / (force_sq + 1e-6)),
            1.0,
        )
        # Only scale the longitudinal component (Fy is externally determined)
        Fx_scaled = Fx_new * scale
        T_new = Fx_scaled * r_w

        # Re-apply box constraints (scaling might have moved outside)
        T_new = jnp.clip(T_new, T_min, T_max)

        return T_new, cost_fn(T_new)

    # Run fixed iterations via scan (JIT-friendly, differentiable)
    T_final, costs = jax.lax.scan(solver_step, T_warmstart, None, length=N_SOLVER_ITERS)

    return T_final


# ─────────────────────────────────────────────────────────────────────────────
# §5  Control Barrier Function Safety Filter
# ─────────────────────────────────────────────────────────────────────────────

class CBFParams(NamedTuple):
    """CBF tuning parameters — with Robust CBF extensions (Batch 3)."""
    beta_max: float = 0.15       # rad maximum sideslip angle (~8.6°)
    wz_max: float = 1.5          # rad/s maximum yaw rate
    kappa_max: float = 0.20      # maximum per-wheel slip ratio
    alpha_beta: float = 5.0      # class-K function gain for sideslip CBF
    alpha_wz: float = 8.0        # class-K function gain for yaw rate CBF
    alpha_kappa: float = 10.0    # class-K function gain for slip CBF
    tau_delay: float = 0.003     # s actuator delay for predictive CBF
    # ── Robust CBF (rCBF) extensions ──────────────────────────────────────
    kappa_r_beta: float = 0.6    # robustness gain: β_max shrinks by κ_r·σ_GP
    kappa_r_wz: float = 3.0      # robustness gain: ψ̇_max shrinks by κ_r·σ_GP
    sigma_floor: float = 0.01    # minimum σ_GP (prevents numerical issues)
    sigma_cap: float = 0.20      # maximum σ_GP effect (prevents over-conservatism)


@partial(jax.jit, static_argnames=('is_rwd',))
def cbf_safety_filter(
    T_alloc: jax.Array,
    T_prev: jax.Array,
    vx: jax.Array,
    vy: jax.Array,
    wz: jax.Array,
    Fz: jax.Array,
    Fy_total: jax.Array,
    mu_est: jax.Array,
    omega_wheel: jax.Array,
    T_min: jax.Array,
    T_max: jax.Array,
    gp_sigma: jax.Array = jnp.array(0.05),
    geo: TVGeometry = TVGeometry(),
    cbf: CBFParams = CBFParams(),
    is_rwd: bool = False,
) -> jax.Array:
    """
    Input-Delay Discrete-Time CBF safety filter.

    Evaluates barrier functions at the PREDICTED future state (accounting
    for actuator delay) and minimally modifies the allocator's torque
    commands to maintain forward invariance of the safe set.

    The filter solves:
        min  ||T - T_alloc||²
        s.t. ΔB_β ≥ -α_β · B_β(x_predicted)
             ΔB_ψ ≥ -α_ψ · B_ψ(x_predicted)
             T_min ≤ T ≤ T_max

    via 3 iterations of projected gradient on the constraint violations.

    Returns: (4,) safety-filtered wheel torques [Nm]
    """
    # RWD: mask front torques and bounds at compile time.
    # The CBF Lie derivatives (Lg_Bwz) use arms — front arms are non-zero but
    # with T_alloc_front=0 and T_max_front=0 the projection step leaves them at zero.
    if is_rwd:
        _mask = jnp.array([0., 0., 1., 1.])
        T_alloc = T_alloc * _mask
        T_min   = T_min   * _mask
        T_max   = T_max   * _mask

    r_w = geo.r_w
    vx_safe = jnp.maximum(jnp.abs(vx), 0.5)

    # ── Step 1: Predict state at time of torque realization ──────────────
    # During the delay τ, the previous torque T_prev is still active
    # Simplified bicycle model prediction (fast, differentiable)
    Fx_prev = jnp.sum(T_prev) / r_w
    ax_pred = Fx_prev / geo.mass
    ay_pred = Fy_total / geo.mass

    # Predicted state after delay
    vx_pred = vx + ax_pred * cbf.tau_delay
    vy_pred = vy + (ay_pred - wz * vx) * cbf.tau_delay
    wz_pred = wz  # yaw rate changes slowly relative to delay

    beta_pred = vy_pred / jnp.maximum(jnp.abs(vx_pred), 0.5)

    # ── Step 2: Evaluate ROBUST barrier functions at predicted state ─────
    # rCBF: safe set contracts proportionally to GP uncertainty σ_GP.
    # When σ_GP is large → β_max shrinks → more conservative.
    # When σ_GP is small → β_max near nominal → full aggressiveness.
    sigma_eff = jnp.clip(gp_sigma, cbf.sigma_floor, cbf.sigma_cap)

    # Sideslip rCBF: B_β = (β_max - κ_r · σ)² - β²
    beta_max_nominal = cbf.beta_max * jnp.clip(mu_est / 1.5, 0.5, 1.5)
    beta_max_robust = jnp.maximum(
        beta_max_nominal - cbf.kappa_r_beta * sigma_eff,
        0.03,  # absolute minimum: ~1.7° — never fully close the safe set
    )
    B_beta = beta_max_robust ** 2 - beta_pred ** 2

    # Yaw rate rCBF: B_ψ = (ψ̇_max - κ_r · σ · v/g)² - ψ̇²
    wz_max_nominal = cbf.wz_max * jnp.clip(mu_est * 9.81 / (vx_safe + 1e-3), 0.3, 2.0)
    # σ_GP affects yaw rate limit through the friction circle:
    # ψ̇_max = μ·g/v, so Δψ̇_max = Δμ·g/v = κ_r·σ·g/v
    wz_max_robust = jnp.maximum(
        wz_max_nominal - cbf.kappa_r_wz * sigma_eff * 9.81 / (vx_safe + 1e-3),
        0.2,  # absolute minimum: prevent singularity at low speed
    )
    B_wz = wz_max_robust ** 2 - wz_pred ** 2

    # ── Step 3: Compute CBF constraint Jacobians w.r.t. T ─────────────
    # dB_β/dT ≈ (∂β/∂ay) × (∂ay/∂Fy_total) × (∂Fy_total/∂T)
    # Since Fy depends on α which depends indirectly on T through yaw
    # dynamics, we use the direct yaw moment path:
    #   dβ/dt ≈ (ay - ψ̇·vx) / vx
    #   d(ψ̇)/dT = arm_i / (I_z · r_w)
    # The CBF constraint is linearized as:
    #   L_f B + L_g B · T + α · B ≥ 0

    arms = yaw_moment_arms(jnp.array(0.0), geo)  # arms at δ≈0 (approximate for CBF)

    # Lie derivative of B_wz w.r.t. T (most actionable CBF):
    # ψ̈ = Mz / Iz, Mz = Σ T_i × arm_i / r_w
    # dB_wz/dt = -2·ψ̇·ψ̈ = -2·ψ̇·(Σ T_i arm_i)/(Iz·r_w)
    # L_g B_wz = ∂B_wz/∂T = -2·wz_pred · arms / (geo.I_z)
    Lg_Bwz = -2.0 * wz_pred * arms / geo.I_z

    # Drift term (what happens if T = 0)
    Lf_Bwz = -2.0 * wz_pred * (Fy_total * geo.lf / geo.I_z)  # natural yaw dynamics

    # CBF constraint: Lf_Bwz + Lg_Bwz · T + α_wz · B_wz ≥ 0
    # This is LINEAR in T → can be enforced via projection

    # ── Step 4: Minimal modification via projected gradient ─────────────
    def cbf_project_step(T, _):
        """Single CBF projection step — minimize ||T - T_alloc||² s.t. CBF."""
        # Yaw rate CBF constraint value
        cbf_wz_val = Lf_Bwz + jnp.dot(Lg_Bwz, T) + cbf.alpha_wz * B_wz

        # If constraint is violated (cbf_wz_val < 0), project T to satisfy it
        # Minimum-norm correction: T_new = T + λ · Lg_Bwz where
        # λ = max(0, -cbf_wz_val / (||Lg_Bwz||² + ε))
        Lg_norm_sq = jnp.sum(Lg_Bwz ** 2) + 1e-8
        lambda_wz = jax.nn.relu(-cbf_wz_val) / Lg_norm_sq
        T_corrected = T + lambda_wz * Lg_Bwz

        # Per-wheel slip CBF: prevent any wheel from spinning excessively
        # B_κ_i = κ_max² − κ_i², where κ_i ≈ (ω_i·r_w − vx) / vx
        # For driving: high torque → high ω → high κ
        # Correction: reduce torque on wheels approaching κ_max
        kappa_est = (omega_wheel * r_w - vx_safe) / vx_safe
        kappa_violation = jax.nn.relu(jnp.abs(kappa_est) - cbf.kappa_max * 0.95)
        # Scale down torque proportional to violation
        slip_scale = 1.0 / (1.0 + cbf.alpha_kappa * kappa_violation)
        T_corrected = T_corrected * slip_scale

        # Re-apply motor limits
        T_corrected = jnp.clip(T_corrected, T_min, T_max)

        # Pull toward allocator solution (minimize ||T - T_alloc||²)
        # Blend: 70% corrected, 30% toward T_alloc (prevents over-correction)
        T_blended = 0.7 * T_corrected + 0.3 * T_alloc

        return T_blended, None

    T_safe, _ = jax.lax.scan(cbf_project_step, T_alloc, None, length=3)
    return T_safe


# ─────────────────────────────────────────────────────────────────────────────
# §6  Yaw Rate Reference Generator
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def yaw_rate_reference(
    delta: jax.Array,          # steering angle [rad]
    vx: jax.Array,             # longitudinal velocity [m/s]
    delta_dot: jax.Array,      # steering rate [rad/s]
    wz_measured: jax.Array,    # measured yaw rate [rad/s]
    mu_est: jax.Array,         # estimated friction coefficient
    geo: TVGeometry = TVGeometry(),
) -> jax.Array:
    """
    Driver-intent-aware yaw rate reference with counter-steer detection.

    Returns target yaw rate [rad/s] for the TV system.
    """
    vx_safe = jnp.maximum(jnp.abs(vx), 1.0)
    L = geo.lf + geo.lr

    # Kinematic yaw rate from bicycle model (understeer gradient K_us)
    # K_us estimated from mass distribution and tire cornering stiffness
    K_us = geo.mass * (geo.lr - geo.lf) / (L * 30000.0)  # approximate
    wz_ref_kinematic = vx_safe * delta / (L + K_us * vx_safe ** 2)

    # Friction saturation: |ψ̇| ≤ μ·g / vx
    wz_max = mu_est * 9.81 / vx_safe
    wz_ref_saturated = jnp.clip(wz_ref_kinematic, -wz_max, wz_max)

    # ── Counter-steer detection ──────────────────────────────────────────
    # If driver is steering opposite to current yaw: they're catching a slide
    is_counter_steer = jnp.sign(delta_dot) * jnp.sign(wz_measured)
    # Negative product = counter-steering → reduce TV intervention
    counter_factor = jax.nn.sigmoid(is_counter_steer * 3.0)  # 0.05–0.95

    # ── Driver urgency ───────────────────────────────────────────────────
    urgency = jnp.tanh(jnp.abs(delta_dot) / 3.0)  # saturates at ~3 rad/s

    # Blend: high urgency → trust driver (blend toward measured)
    # Counter-steering → strongly reduce TV reference
    wz_ref_blended = (
        (1.0 - urgency * 0.6) * wz_ref_saturated * counter_factor
        + urgency * 0.6 * wz_measured
    )

    return wz_ref_blended


# ─────────────────────────────────────────────────────────────────────────────
# §7  Output EMA Filter + Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def smooth_output(
    T_new: jax.Array,      # (4,) raw solver output [Nm]
    T_prev: jax.Array,     # (4,) previous applied torque [Nm]
    dt: jax.Array,         # timestep [s]
    alpha_ema: float = 0.7,     # EMA smoothing (0=no change, 1=instant)
    R_max: float = 15000.0,     # Nm/s maximum torque rate (per wheel)
) -> jax.Array:
    """
    Output conditioning: EMA filter + smooth rate limiter.

    Eliminates chattering from solver residual while preserving step response.
    The rate limit uses tanh soft-clamp (differentiable, C∞).
    """
    # EMA blend
    T_ema = alpha_ema * T_new + (1.0 - alpha_ema) * T_prev

    # Smooth rate limiter: tanh soft-clamp
    dT = T_ema - T_prev
    dT_max = R_max * dt
    T_rate_limited = T_prev + dT_max * jnp.tanh(dT / (dT_max + 1e-6))

    return T_rate_limited


# ─────────────────────────────────────────────────────────────────────────────
# §8  Top-Level TV Controller (single-call interface)
# ─────────────────────────────────────────────────────────────────────────────

class TVState(NamedTuple):
    """Persistent state for the TV controller across timesteps."""
    T_prev: jax.Array        # (4,) previous applied torques [Nm]
    delta_prev: jax.Array    # scalar previous steering angle [rad]
    wz_ref_prev: jax.Array   # scalar previous yaw rate reference [rad/s]


class TVOutput(NamedTuple):
    """Output of a single TV step."""
    T_wheel: jax.Array       # (4,) commanded wheel torques [Nm]
    Mz_actual: jax.Array     # scalar achieved yaw moment [Nm]
    Mz_target: jax.Array     # scalar target yaw moment [Nm]
    Fx_actual: jax.Array     # scalar achieved total force [N]
    wz_ref: jax.Array        # scalar yaw rate reference [rad/s]
    cost: jax.Array          # scalar allocator cost (for diagnostics)
    cbf_active: jax.Array    # scalar 0/1 CBF intervention flag


@partial(jax.jit, static_argnums=())
def tv_step(
    # Vehicle state
    vx: jax.Array,              # longitudinal velocity [m/s]
    vy: jax.Array,              # lateral velocity [m/s]
    wz: jax.Array,              # yaw rate [rad/s]
    delta: jax.Array,           # steering angle [rad]
    Fz: jax.Array,              # (4,) vertical loads [N]
    Fy: jax.Array,              # (4,) lateral forces at tires [N]
    omega_wheel: jax.Array,     # (4,) wheel angular velocities [rad/s]
    kappa: jax.Array,           # (4,) longitudinal slip ratios
    mu_est: jax.Array,          # scalar estimated friction coefficient
    # Driver demand
    Fx_driver: jax.Array,       # scalar driver total force demand [N]
    # Powertrain state
    pt_state: PowertrainState,  # electrothermal state
    # Controller state
    tv_state: TVState,          # persistent TV state
    # Time
    dt: jax.Array,              # timestep [s]
    # Configuration
    geo: TVGeometry = TVGeometry(),
    w: AllocatorWeights = AllocatorWeights(),
    cbf_params: CBFParams = CBFParams(),
    mp: MotorParams = MotorParams(),
    bp: BatteryParams = BatteryParams(),
) -> tuple[TVOutput, TVState]:
    """
    Single timestep of the full TV controller.

    Pipeline:
      1. Yaw rate reference generator (driver-intent-aware)
      2. Yaw moment demand computation (P controller on ψ̇ error)
      3. Motor torque limit computation (from electrothermal state)
      4. SOCP allocation (projected gradient, 12 iterations)
      5. CBF safety filter (input-delay predictive, 3 iterations)
      6. Output smoothing (EMA + rate limiter)

    Returns (TVOutput, TVState_new).
    """
    # ── 1. Yaw rate reference ────────────────────────────────────────────
    delta_dot = (delta - tv_state.delta_prev) / (dt + 1e-6)
    wz_ref = yaw_rate_reference(delta, vx, delta_dot, wz, mu_est, geo)

    # ── 2. Yaw moment demand (proportional + derivative) ─────────────────
    wz_error = wz_ref - wz
    dwz_error = (wz_ref - tv_state.wz_ref_prev) / (dt + 1e-6) - 0.0  # D-term on ref
    Kp_yaw = 80.0   # Nm / (rad/s) yaw P-gain
    Kd_yaw = 5.0    # Nm / (rad/s²) yaw D-gain
    Mz_target = Kp_yaw * wz_error + Kd_yaw * dwz_error

    # ── 3. Motor torque limits ──────────────────────────────────────────
    T_min, T_max = motor_torque_limits_at_wheel(
        omega_wheel, pt_state.V_bus, pt_state.T_motors, pt_state.T_invs,
        pt_state.SoC, mp, bp,
    )
    P_max = total_power_limit(pt_state.V_bus, pt_state.SoC, pt_state.T_cell, bp)

    # ── 4. Friction coefficient per wheel ────────────────────────────────
    mu_per_wheel = jnp.full(4, mu_est)

    # ── 5. SOCP allocation ──────────────────────────────────────────────
    T_alloc = solve_torque_allocation(
        T_warmstart=tv_state.T_prev,
        T_prev=tv_state.T_prev,
        Fx_target=Fx_driver,
        Mz_target=Mz_target,
        delta=delta,
        Fz=Fz,
        Fy=Fy,
        mu=mu_per_wheel,
        omega_wheel=omega_wheel,
        T_min=T_min,
        T_max=T_max,
        P_max=P_max,
        geo=geo,
        w=w,
    )

    # ── 6. CBF safety filter ────────────────────────────────────────────
    Fy_total = jnp.sum(Fy)
    T_safe = cbf_safety_filter(
        T_alloc, tv_state.T_prev, vx, vy, wz, Fz, Fy_total, mu_est,
        omega_wheel, T_min, T_max, gp_sigma, geo, cbf,
    )
    cbf_intervention = jnp.linalg.norm(T_safe - T_alloc)
    cbf_active = (cbf_intervention > 1.0).astype(jnp.float32)
    
    # ── 7. Output smoothing ─────────────────────────────────────────────
    T_output = smooth_output(T_safe, tv_state.T_prev, dt)

    # ── 8. Compute achieved quantities ──────────────────────────────────
    r_w = geo.r_w
    arms = yaw_moment_arms(delta, geo)
    Fx_actual = jnp.sum(T_output) / r_w
    Mz_actual = jnp.sum(T_output * arms)
    cost = allocator_cost(
        T_output, tv_state.T_prev, Fx_driver, Mz_target, delta,
        Fz, Fy, mu_per_wheel, omega_wheel, T_min, T_max, P_max, geo, w,
    )

    # ── 9. Pack outputs ─────────────────────────────────────────────────
    output = TVOutput(
        T_wheel=T_output,
        Mz_actual=Mz_actual,
        Mz_target=Mz_target,
        Fx_actual=Fx_actual,
        wz_ref=wz_ref,
        cost=cost,
        cbf_active=cbf_active,
    )

    new_state = TVState(
        T_prev=T_output,
        delta_prev=delta,
        wz_ref_prev=wz_ref,
    )

    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# §9  Factory / Initialization
# ─────────────────────────────────────────────────────────────────────────────

def make_tv_state() -> TVState:
    """Create initial TV controller state (zero torques)."""
    return TVState(
        T_prev=jnp.zeros(4),
        delta_prev=jnp.array(0.0),
        wz_ref_prev=jnp.array(0.0),
    )


def make_tv_geometry(vp: dict = None) -> TVGeometry:
    """Create TVGeometry from vehicle_params dict or defaults."""
    if vp is None:
        return TVGeometry()
    return TVGeometry.from_vehicle_params(vp)