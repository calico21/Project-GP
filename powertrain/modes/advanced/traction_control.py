# powertrain/traction_control.py
# Project-GP — Differentiable Extremum-Seeking Traction Controller (DESC)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Dual-path optimal slip ratio estimation:
#   Path 1 — Model-based: analytical kappa* from Pacejka MF6.2 dFx/dkappa = 0
#   Path 2 — Model-free:  DESC extremum seeking via 15 Hz dither on Fx
#   Fusion:  GP uncertainty-weighted blend
#
# Combined-slip awareness:
#   kappa*_combined = kappa*_pure * sqrt(1 - (alpha_t / alpha_peak)^2)
#
# Mode-free TC/TV integration:
#   Continuous sigmoid-blended weights for the unified SOCP allocator.
#
# All functions are pure JAX — safe inside jit/grad/vmap/scan.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────────────────
# S1  DESC Configuration + State
# ─────────────────────────────────────────────────────────────────────────────

class DESCParams(NamedTuple):
    """
    DESC hyperparameters — calibrated for Hoosier R20 on FS vehicle.
 
    Tuning rationale (GP-vX2 Batch 1 fix):
      eta:      5e-4 → each step moves κ_base by ~3.75e-4 (200 steps = 0.075 range).
                Sufficient to traverse kappa_min→kappa_max in ~400 steps = 2s.
      alpha_hp: 0.65 → HPF cutoff ≈ 11 Hz. The 15 Hz dither passes with <5% attenuation.
                Previous 0.85 → cutoff ≈ 4.8 Hz → 25% signal loss at 15 Hz.
      alpha_lp: 0.85 → slightly faster LP tracking. Previous 0.90 was overdamped.
      A_dither: 0.008 → increased from 0.005 for better SNR. Still small enough
                that the torque perturbation ΔT ≈ dFx/dκ × A × r_w ≈ 10³ × 0.008 × 0.2 ≈ 1.6 Nm
                is imperceptible to the driver.
    """
    omega_es:   float = 94.25       # rad/s dither frequency (15 Hz)
    A_dither:   float = 0.008       # dither amplitude on kappa_ref  [was 0.005]
    eta:        float = 5e-4        # gradient ascent learning rate   [was 1e-4, 5× increase]
    alpha_hp:   float = 0.65        # high-pass filter coefficient    [was 0.85]
    alpha_lp:   float = 0.85        # low-pass filter coefficient     [was 0.90]
    kappa_init: float = 0.10        # initial kappa_base estimate
    kappa_min:  float = 0.03        # minimum kappa_base
    kappa_max:  float = 0.25        # maximum kappa_base

# ── PATCH 1a: add to DESCState class body ────────────────────────────────────
class DESCState(NamedTuple):
    kappa_base: jax.Array
    integrator: jax.Array
    hpf_state: jax.Array
    lpf_state: jax.Array
    t_acc: jax.Array          # ← NEW: accumulated time for dither phase

    @classmethod
    def default(cls, params=None):
        if params is None:
            params = DESCParams()
        return cls(
            kappa_base=jnp.array(params.kappa_init),
            integrator=jnp.array(params.kappa_init),
            hpf_state=jnp.array(0.0),
            lpf_state=jnp.array(0.0),
            t_acc=jnp.array(0.0),       # ← NEW
        )

    @classmethod
    def default(cls, params: "DESCParams") -> "DESCState":
        """Convenience constructor matching the make_desc_state factory."""
        return cls(
            kappa_base=jnp.array(params.kappa_init),
            integrator=jnp.array(params.kappa_init),
            hpf_state=jnp.array(0.0),
            lpf_state=jnp.array(0.0),
            t_acc=jnp.array(0.0),
        )

def make_desc_state(params: DESCParams = DESCParams()) -> DESCState:
    return DESCState(
        kappa_base=jnp.array(params.kappa_init),
        integrator=jnp.array(params.kappa_init),
        hpf_state=jnp.array(0.0),
        lpf_state=jnp.array(0.0),
        t_acc=jnp.array(0.0),
    )

# ─────────────────────────────────────────────────────────────────────────────
# S2  DESC Step (single timestep, fully differentiable)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def desc_step(
    state: DESCState,
    Fx_measured: jax.Array,
    omega_wheel: jax.Array,     # (4,) — unused here but keeps API consistent with TC
    vx: jax.Array,
    dt: jax.Array,              # ← NEW: timestep, NOT accumulated time
    params: DESCParams = DESCParams(),
) -> tuple[DESCState, jax.Array]:
    """
    Single DESC update via lock-in demodulation on motor-side Fx.
    Uses Fx (from motor torque accounting) not a_x (from IMU) to bypass
    chassis vibration. Returns (new_state, kappa_ref_with_dither).
    """
    kappa_base, integrator, hpf_state, lpf_state, _ = state

    # Dither signal
    t_now = state.t_acc + dt
    dither = params.A_dither * jnp.sin(params.omega_es * t_now)
    # High-pass: remove DC + low-freq vehicle dynamics
    hpf_new = params.alpha_hp * hpf_state + (1.0 - params.alpha_hp) * Fx_measured
    Fx_hp = Fx_measured - hpf_new

    # Lock-in correlation: extract Fx component at exactly omega_es
    grad_raw = Fx_hp * jnp.sin(params.omega_es * t_now) * (2.0 / (params.A_dither + 1e-8))

    # Low-pass: smooth noisy gradient estimate
    lpf_new = params.alpha_lp * lpf_state + (1.0 - params.alpha_lp) * grad_raw

    # Speed gate: DESC meaningless below 3 m/s (tau -> inf)
    speed_gate = jax.nn.sigmoid((vx - 3.0) * 2.0)

    # Gradient ascent on kappa_base
    integrator_new = integrator + params.eta * lpf_new * speed_gate * dt
    kappa_base_new = jnp.clip(integrator_new, params.kappa_min, params.kappa_max)
    integrator_new = kappa_base_new

    kappa_ref = kappa_base_new + dither * speed_gate
    return DESCState(kappa_base_new, integrator_new, hpf_new, lpf_new, t_now), kappa_ref

# ─────────────────────────────────────────────────────────────────────────────
# S3  Model-Based kappa* from Pacejka Coefficients
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def kappa_star_pacejka(
    Fz: jax.Array,
    gamma: jax.Array,
    mu_thermal: jax.Array,
    # Default Hoosier R20 coefficients
    Fz0: float = 654.0,
    PCX1: float = 1.579, PDX1: float = 1.0, PDX2: float = -0.10, PDX3: float = 0.0,
    PKX1: float = 18.5, PKX2: float = 0.0, PKX3: float = 0.20,
    PEX1: float = -0.20, PEX2: float = 0.10,
) -> jax.Array:
    """
    Analytical optimal slip ratio from Pacejka MF6.2 pure longitudinal.
    Solves dFx/dkappa = 0 via 3 Newton iterations.
    """
    Fz_safe = jnp.maximum(Fz, 10.0)
    dfz = (Fz_safe - Fz0) / (Fz0 + 1e-6)

    Cx = PCX1
    Dx = PDX1 * (1.0 + PDX2 * dfz) * (1.0 - PDX3 * gamma ** 2) * Fz_safe * mu_thermal
    Kx = PKX1 * Fz_safe * jnp.exp(PKX3 * dfz) * (1.0 + PKX2 * dfz)
    Bx = Kx / jnp.maximum(Cx * Dx, 1e-6)
    Ex = jnp.clip(PEX1 + PEX2 * dfz, -10.0, 1.0)

    # Initial guess: peak of simplified magic formula
    kappa_init = jnp.tan(jnp.pi / (2.0 * Cx + 1e-6)) / (Bx + 1e-6)
    kappa_init = jnp.clip(kappa_init, 0.02, 0.25)

    # 3 Newton iterations (unrolled for JIT, accounts for E curvature)
    def newton_step(kappa, _):
        Bk = Bx * kappa
        inner = Bk - Ex * (Bk - jnp.arctan(Bk))

        # dFx/dkappa
        dBk = Bx
        d_inner = dBk * (1.0 - Ex * (1.0 - 1.0 / (1.0 + Bk ** 2)))
        d_atan_inner = d_inner / (1.0 + inner ** 2)
        dFx = Dx * jnp.cos(Cx * jnp.arctan(inner)) * Cx * d_atan_inner

        # d2Fx/dkappa2 via finite difference
        eps_fd = 1e-4
        Bk_p = Bx * (kappa + eps_fd)
        inner_p = Bk_p - Ex * (Bk_p - jnp.arctan(Bk_p))
        d_inner_p = Bx * (1.0 - Ex * (1.0 - 1.0 / (1.0 + Bk_p ** 2)))
        d_atan_inner_p = d_inner_p / (1.0 + inner_p ** 2)
        dFx_p = Dx * jnp.cos(Cx * jnp.arctan(inner_p)) * Cx * d_atan_inner_p
        d2Fx = (dFx_p - dFx) / eps_fd

        d2Fx_safe = jnp.where(jnp.abs(d2Fx) > 1e-6, d2Fx, -1e-6)
        kappa_new = kappa - dFx / d2Fx_safe
        return jnp.clip(kappa_new, 0.02, 0.30), None

    kappa_star, _ = jax.lax.scan(newton_step, kappa_init, None, length=3)
    return kappa_star
# ── PATCH 1b: add immediately after kappa_star_pacejka definition ─────────────
@jax.jit
def kappa_star_model(
    Fz: jax.Array,            # (4,) or scalar — vertical load [N]
    mu_scale: jax.Array,      # scalar friction scale (e.g. 1.4 for dry)
    T_tire: jax.Array,        # (4,) or scalar — tire surface temp [°C]
    gamma: float = 0.0,
    T_opt: float = 85.0,      # °C optimal operating temperature
    T_range: float = 30.0,    # °C half-width of thermal μ window
) -> jax.Array:
    """
    Public-facing kappa* API used by sanity checks and external callers.
    Converts (mu_scale, T_tire) → mu_thermal then delegates to kappa_star_pacejka.
    Thermal derating: mu_thermal = mu_scale * exp(-((T - T_opt)/T_range)^2)
    Maps onto a per-wheel vmapped Pacejka solve.
    """
    # Gaussian thermal window: peak at T_opt, smooth derating outside
    mu_thermal = mu_scale * jnp.exp(-((T_tire - T_opt) / T_range) ** 2)

    # vmap over wheel axis — handles both (4,) and scalar Fz/T_tire
    Fz_arr = jnp.broadcast_to(jnp.atleast_1d(Fz), (4,))
    mu_arr = jnp.broadcast_to(jnp.atleast_1d(mu_thermal), (4,))
    gamma_arr = jnp.full(4, gamma)

    return jax.vmap(kappa_star_pacejka)(Fz_arr, gamma_arr, mu_arr)
# ─────────────────────────────────────────────────────────────────────────────
# S4  Combined-Slip kappa* Reduction
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def kappa_star_combined(
    kappa_star_pure: jax.Array,  # (4,)
    alpha_t: jax.Array,          # (4,) transient slip angles [rad]
    alpha_peak: jax.Array,       # scalar peak lateral slip [rad]
) -> jax.Array:
    """Reduce kappa* when tire is cornering (friction ellipse)."""
    alpha_ratio_sq = (alpha_t / (jnp.abs(alpha_peak) + 1e-3)) ** 2
    alpha_ratio_clamped = jnp.clip(alpha_ratio_sq, 0.0, 0.95)
    reduction = jnp.sqrt(
        jax.nn.softplus((1.0 - alpha_ratio_clamped) * 10.0) / 10.0 + 1e-6
    )
    return kappa_star_pure * reduction

# ─────────────────────────────────────────────────────────────────────────────
# S5  Dual-Path Fusion
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def fuse_kappa_star(
    kappa_model: jax.Array,
    kappa_esc: jax.Array,
    gp_sigma: jax.Array,
    sigma_base: float = 0.05,
) -> jax.Array:
    """GP-uncertainty-weighted fusion. High sigma -> trust ESC, low -> trust model."""
    alpha = jnp.clip(gp_sigma / (gp_sigma + sigma_base + 1e-8), 0.05, 0.95)
    return alpha * kappa_esc + (1.0 - alpha) * kappa_model

# ─────────────────────────────────────────────────────────────────────────────
# S6  Mode-Free TC/TV Weight Blending
# ─────────────────────────────────────────────────────────────────────────────

class TCWeights(NamedTuple):
    """TC/TV blending weight config — stored in PowertrainConfig.tc_weights."""
    w_slip_base: float = 1.0
    w_slip_launch_boost: float = 5.0
    w_yaw_base: float = 200.0
    w_energy_base: float = 0.01


class BlendWeights(NamedTuple):
    w_slip: jax.Array
    w_yaw: jax.Array
    w_energy: jax.Array

@jax.jit
def compute_blend_weights(
    vx: jax.Array, ax: jax.Array, ay: jax.Array, is_launch: jax.Array,
    w_slip_base: float = 1.0, w_slip_launch_boost: float = 5.0,
    w_yaw_base: float = 200.0, w_energy_base: float = 0.01,
) -> BlendWeights:
    """Continuous sigmoid-blended TC/TV weights. No mode switching."""
    ax_abs = jnp.abs(ax)
    ay_abs = jnp.abs(ay)
    lon_ratio = ax_abs / (ax_abs + ay_abs + 0.1)
    low_speed_boost = jax.nn.softplus(5.0 - vx) / 5.0

    w_slip = w_slip_base * (1.0 + lon_ratio * 2.0 + low_speed_boost
                            + is_launch * w_slip_launch_boost)
    w_yaw = w_yaw_base * (1.0 - lon_ratio * 0.5) * (1.0 - is_launch * 0.9)
    w_energy = w_energy_base * jax.nn.sigmoid(vx - 10.0)

    return BlendWeights(w_slip=w_slip, w_yaw=w_yaw, w_energy=w_energy)

# ─────────────────────────────────────────────────────────────────────────────
# S7  Slip Ratio Computation
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_slip_ratios(omega_wheel: jax.Array, vx: jax.Array, r_w: float = 0.2032):
    """Per-wheel kappa = (omega*r - vx) / max(|vx|, 0.5)."""
    vx_safe = jnp.maximum(jnp.abs(vx), 0.5)
    return jnp.clip((omega_wheel * r_w - vx) / vx_safe, -0.8, 0.8)

# ─────────────────────────────────────────────────────────────────────────────
# S8  Motor-Side Fx Estimator
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def estimate_fx_from_motors(
    T_wheel: jax.Array, omega_wheel: jax.Array,
    Iw: float = 1.2, r_w: float = 0.2032, dt: float = 0.005,
    omega_prev: jax.Array = None,
) -> jax.Array:
    """Fx_tire = (T_motor - Iw*omega_dot*r_w) / r_w. Bypasses IMU vibration."""
    if omega_prev is None:
        omega_prev = omega_wheel
    omega_dot = (omega_wheel - omega_prev) / (dt + 1e-6)
    return (T_wheel - Iw * omega_dot * r_w) / r_w

# ─────────────────────────────────────────────────────────────────────────────
# S9  Top-Level TC Controller
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def wheel_speed_confidence(
    omega_wheel: jax.Array,  # (4,) wheel angular speeds [rad/s]
    vx: jax.Array,           # vehicle longitudinal speed [m/s]
    r_w: float = 0.2032,
    omega_max: float = 1200.0,
) -> jax.Array:
    """
    Scalar sensor confidence for wheel speed measurements [0, 1].
    Degrades under: out-of-range speeds, negative speeds, or large
    front-rear slip disagreement (diagnostic of sensor fault / spinout).
    All ops are smooth — safe for grad().
    """
    # Per-wheel range gate: softplus sigmoid on [0, omega_max]
    in_range = jnp.prod(
        jax.nn.sigmoid((omega_max - omega_wheel) * 0.01)
        * jax.nn.sigmoid(omega_wheel * 10.0)
    )
    # Axle consistency: large front–rear slip delta → confidence degrades
    kappa = compute_slip_ratios(omega_wheel, vx, r_w)
    axle_delta = jnp.abs(jnp.mean(kappa[:2]) - jnp.mean(kappa[2:]))
    consistency = jax.nn.sigmoid(0.6 - axle_delta * 5.0)
    return jnp.clip(in_range * consistency, 0.0, 1.0)


class TCState(NamedTuple):
    desc_front: DESCState
    desc_rear: DESCState
    omega_prev: jax.Array     # (4,) previous wheel speeds
    kappa_star: jax.Array     # (4,) current targets
    t_current: jax.Array      # scalar: accumulated sim time [s] for DESC dither phase

    @classmethod
    def default(cls, params: DESCParams = DESCParams()) -> "TCState":
        return make_tc_state(params)


def make_tc_state(params: DESCParams = DESCParams()) -> TCState:
    return TCState(
        desc_front=make_desc_state(params),
        desc_rear=make_desc_state(params),
        omega_prev=jnp.zeros(4),
        kappa_star=jnp.full(4, params.kappa_init),
        t_current=jnp.array(0.0),
    )

class TCOutput(NamedTuple):
    kappa_star: jax.Array
    kappa_measured: jax.Array
    kappa_error: jax.Array
    desc_grad_front: jax.Array
    desc_grad_rear: jax.Array
    blend_weights: BlendWeights
    # Flat aliases for direct manager / diagnostics access
    desc_grad: jax.Array      # = (desc_grad_front + desc_grad_rear) / 2
    w_slip: jax.Array         # = blend_weights.w_slip
    w_yaw: jax.Array          # = blend_weights.w_yaw
    confidence: jax.Array     # wheel speed confidence score [0, 1]

@partial(jax.jit, static_argnums=())
def tc_step(
    vx: jax.Array,
    vy: jax.Array,
    ax: jax.Array,
    ay: jax.Array,
    omega_wheel: jax.Array,
    alpha_t: jax.Array,
    Fz: jax.Array,
    T_applied: jax.Array,    # (4,) previous wheel torques [Nm]  ← was T_wheel
    T_tire: jax.Array,       # (4,) tire surface temps [°C]      ← replaces mu_thermal
    mu_est: jax.Array,       # scalar base friction estimate
    gp_sigma: jax.Array,
    tc_state: TCState,
    dt: jax.Array,
    desc_params: DESCParams = DESCParams(),
    tc_weights: TCWeights = TCWeights(),
    r_w: float = 0.2032,
    alpha_peak: float = 0.12,
    T_opt: float = 85.0,     # °C optimal tire temp for peak μ
    T_range: float = 30.0,   # °C Gaussian derating half-width
) -> tuple[TCOutput, TCState]:
    """
    Single TC timestep — manager-facing API.

    Thermal μ derating:  mu_i = mu_est × exp(-((T_tire_i − T_opt)/T_range)²)
    DESC dither phase:   t accumulated via tc_state.t_current
    gamma / is_launch:   defaulted to 0 — handled at higher layers
    """
    t          = tc_state.t_current
    gamma      = jnp.zeros(4)
    is_launch  = jnp.array(0.0)

    # Per-wheel thermal friction derating; scalar mean for Pacejka solve
    mu_per_wheel     = mu_est * jnp.exp(-((T_tire - T_opt) / T_range) ** 2)
    mu_thermal_mean  = jnp.mean(mu_per_wheel)

    kappa_measured = compute_slip_ratios(omega_wheel, vx, r_w)

    # Motor-side Fx — uses T_applied (previous step torques) as proxy
    Fx_est       = estimate_fx_from_motors(T_applied, omega_wheel,
                                           omega_prev=tc_state.omega_prev)
    Fx_front_avg = (Fx_est[0] + Fx_est[1]) * 0.5
    Fx_rear_avg  = (Fx_est[2] + Fx_est[3]) * 0.5

    desc_f_new, kappa_ref_f = desc_step(tc_state.desc_front, Fx_front_avg, omega_wheel, vx, dt, desc_params)
    desc_r_new, kappa_ref_r = desc_step(tc_state.desc_rear,  Fx_rear_avg,  omega_wheel, vx, dt, desc_params)
    kappa_esc = jnp.array([kappa_ref_f, kappa_ref_f, kappa_ref_r, kappa_ref_r])

    kappa_model_vals = jax.vmap(
        lambda fz, gam: kappa_star_pacejka(fz, gam, mu_thermal_mean)
    )(Fz, gamma)

    kappa_fused = jax.vmap(
        lambda km, ke: fuse_kappa_star(km, ke, gp_sigma)
    )(kappa_model_vals, kappa_esc)

    kappa_star = kappa_star_combined(kappa_fused, alpha_t, jnp.array(alpha_peak))
    blend = compute_blend_weights(
        vx, ax, ay, is_launch,
        w_slip_base=tc_weights.w_slip_base,
        w_slip_launch_boost=tc_weights.w_slip_launch_boost,
        w_yaw_base=tc_weights.w_yaw_base,
        w_energy_base=tc_weights.w_energy_base,
    )

    conf = wheel_speed_confidence(omega_wheel, vx, r_w)
    output = TCOutput(
        kappa_star=kappa_star, kappa_measured=kappa_measured,
        kappa_error=kappa_star - kappa_measured,
        desc_grad_front=desc_f_new.lpf_state,
        desc_grad_rear=desc_r_new.lpf_state,
        blend_weights=blend,
        desc_grad=(desc_f_new.lpf_state + desc_r_new.lpf_state) * 0.5,
        w_slip=blend.w_slip,
        w_yaw=blend.w_yaw,
        confidence=conf,
    )
    new_state = TCState(
        desc_front=desc_f_new,
        desc_rear=desc_r_new,
        omega_prev=omega_wheel,
        kappa_star=kappa_star,
        t_current=t + dt,
    )
    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# S10  Public Aliases (manager / external API surface)
# ─────────────────────────────────────────────────────────────────────────────

# powertrain_manager imports these names — aliases keep internal names stable
# while the public surface matches the architecture doc.
compute_blending_weights = compute_blend_weights
estimate_slip_ratios     = compute_slip_ratios