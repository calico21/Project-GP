"""
control_mapper.py — Project-GP 4WD torque allocation → MirenaSim CarControl.msg

MirenaSim contract (CarControl.msg):
  gas        ∈ [-1, 1]   where -1=full brake, +1=full throttle
  steer_angle (rad, CCW+) — single equivalent front steering angle

Project-GP output:
  torques      [4]  (Nm) per-wheel  [fl, fr, rl, rr]
  steer_angles [2]  (rad) front wheels [fl, fr]  — FS rear wheels don't steer

Control projection:
  gas   = tanh(F_x_net / C_M)
          F_x_net = Σ T_i / R_WHEEL  (signed: positive=drive, negative=regen/brake)
          C_M     = 325 N             (from system_dynamics.py: 1.3 × m=250)
          tanh:   smooth, differentiable, bounded — avoids hard clip gradient kill

  steer = mean(δ_fl, δ_fr) — Ackermann-equivalent single front angle.
          For more accurate Ackermann: use inner/outer geometry from wheelbase.

All functions are pure JAX, JIT-compiled, gradient-safe.
"""

import jax
import jax.numpy as jnp

# ─── Vehicle Parameters (system_dynamics.py Defaults + mirena_const.h) ────────
_C_M       = 1.3 * 250.0   # = 325.0 N — MirenaSim motor map gain coefficient
_R_WHEEL   = 0.2023         # m — wheel radius
_T_FRONT   = 1.185          # m — front track width  (mirena_const.h)
_WB        = 0.806 + 0.744  # m — wheelbase Lf + Lr  (mirena_const.h)

# Physical steer limits for FS (typical Ter27 rack limits)
_STEER_MAX  = 0.40   # rad ≈ 23° — hard physical limit
_GAS_MIN    = -1.0
_GAS_MAX    =  1.0


@jax.jit
def project_to_car_control(
    torques:      jnp.ndarray,  # [4] per-wheel torques (Nm) [fl, fr, rl, rr]
    steer_angles: jnp.ndarray,  # [2] front steer angles (rad) [fl, fr] CCW+
    u_vel:        jnp.ndarray,  # () longitudinal velocity (m/s) — for context
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Projects 4WD torque allocation → scalar (gas, steer_angle).

    Gas uses tanh rather than hard clip: gradient flows through the saturation
    region (important for online sysid and any end-to-end gradient chain).

    Steer uses front-axle mean as Ackermann-equivalent. For proper Ackermann:
      δ_ack = arctan(WB / (WB/tan(δ_inner) - T_front/2))
    but mean is adequate for FS steering angles (<25°) where sin≈tan.

    Returns: (gas ∈ [-1,1], steer_angle in rad)
    """
    # Net longitudinal wheel force from all four motors
    F_x_net = jnp.sum(torques) / _R_WHEEL   # (N)

    # tanh-normalised gas command — smooth, bounded, fully differentiable
    gas   = jnp.tanh(F_x_net / _C_M)

    # Ackermann-equivalent front steer
    steer = jnp.mean(steer_angles)

    return gas, steer


@jax.jit
def project_with_ackermann(
    torques:      jnp.ndarray,  # [4]
    steer_fl:     jnp.ndarray,  # () inner/outer FL steer (rad)
    steer_fr:     jnp.ndarray,  # () inner/outer FR steer (rad)
    u_vel:        jnp.ndarray,  # ()
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Higher-fidelity Ackermann equivalent for larger steer angles.
    Computes the geometrically correct single-track equivalent steer angle.
    Differentiable through arctan.
    """
    F_x_net = jnp.sum(torques) / _R_WHEEL
    gas     = jnp.tanh(F_x_net / _C_M)

    # Ackermann geometry: δ_ack satisfies cot(δ_fl) - cot(δ_fr) = T_front/WB
    # Equivalent center angle via the exact formula
    eps      = 1e-6
    cot_fl   = jnp.cos(steer_fl) / (jnp.sin(steer_fl) + eps)
    cot_fr   = jnp.cos(steer_fr) / (jnp.sin(steer_fr) + eps)
    cot_ack  = 0.5 * (cot_fl + cot_fr)  # mean cotangent → center angle
    steer    = jnp.arctan(1.0 / (cot_ack + eps))

    return gas, steer


@jax.jit
def inverse_gas_to_torque_target(
    gas:   jnp.ndarray,  # () scalar gas command (from MirenaSim or reference)
    u_vel: jnp.ndarray,  # () longitudinal velocity (m/s)
) -> jnp.ndarray:
    """
    Inverse map: gas → target net longitudinal force → target per-wheel torque.

    Useful for:
      - Traction control: compare wheel speed targets vs actuals.
      - Sysid: recover implied torque from MirenaSim state transitions.

    Assumes uniform torque split (4WD equally shared). The SOCP layer refines
    the distribution given tire load, but this gives a valid nominal point.
    """
    # Invert tanh: F_x = C_M * atanh(gas)
    gas_clamped  = jnp.clip(gas, -0.999, 0.999)   # avoid atanh(±1) = ±inf
    F_x_target   = _C_M * jnp.arctanh(gas_clamped)
    T_per_wheel  = F_x_target * _R_WHEEL / 4.0     # equal split
    return jnp.full(4, T_per_wheel)                # [4]


@jax.jit
def compute_traction_reserve(
    torques:   jnp.ndarray,  # [4] commanded torques
    gp_sigma:  jnp.ndarray,  # [4] GP uncertainty σ per wheel (from tire PINN)
    mu_est:    jnp.ndarray,  # [4] estimated friction coefficient per wheel
    Fz:        jnp.ndarray,  # [4] normal loads (N)
) -> jnp.ndarray:
    """
    Smooth traction reserve per wheel: margin to friction circle saturation.
    Used by the CBF safety filter as an input feature for slip prediction.

    Returns [4] reserve ∈ [0, 1]:  1=no load, 0=at friction limit.
    """
    F_x_wheel  = torques / _R_WHEEL           # [4] longitudinal wheel forces
    F_fric_max = mu_est * Fz                  # [4] friction capacity (N)
    # Soft normalisation: use sigmoid so gradient stays alive at saturation
    utilisation = jax.nn.sigmoid(
        4.0 * (jnp.abs(F_x_wheel) - F_fric_max) / (F_fric_max + 1.0)
    )
    return 1.0 - utilisation                  # [4]
