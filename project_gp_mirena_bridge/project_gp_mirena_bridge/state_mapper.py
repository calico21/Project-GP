"""
state_mapper.py — JAX-native bidirectional state mapper
MirenaSim 6-DOF (Car.msg + WheelSpeeds) ↔ Project-GP 46-DOF Port-Hamiltonian state.

Key design:
  - expand_to_46dof() fuses observable MirenaSim state with prev_state for
    thermal/suspension continuity. Never cold-initialises thermal on every frame.
  - Slip estimation is fully differentiable (tanh-bounded κ, atan-based α).
  - covariance_to_precision() converts Car.msg's 6×6 covariance to per-DOF
    precision weights for the online sysid loss weighting.
"""

import jax
import jax.numpy as jnp
import numpy as np

# ─── State dimension ─────────────────────────────────────────────────────────
STATE_DIM = 46  # must be defined before _NOMINAL_STATE_NP uses it

# ─── 46-DOF Index Layout (from simulator/sim_config.py _StateIndices) ────────
_M_X      = 0    # world X        (m)
_M_Y      = 1    # world Y        (m)
_M_Z      = 2    # world Z        (m)
_M_ROLL   = 3    # roll φ         (rad)
_M_PITCH  = 4    # pitch θ        (rad)
_M_PSI    = 5    # yaw ψ          (rad)
_M_ZS_FL  = 6    # suspension heave FL  (m)
_M_ZS_FR  = 7    # suspension heave FR  (m)
_M_ZS_RL  = 8    # suspension heave RL  (m)
_M_ZS_RR  = 9    # suspension heave RR  (m)
# x[10:14] — wheel rotation angles (not needed for bridge)

_M_U      = 14   # vx longitudinal  (m/s)
_M_V      = 15   # vy lateral       (m/s)
_M_VZ     = 16   # vz vertical      (m/s)
_M_WX     = 17   # roll rate        (rad/s)
_M_WY     = 18   # pitch rate       (rad/s)
_M_OMEGA  = 19   # wz yaw rate      (rad/s)
# x[20:24] — suspension rates (carried from prev_state)
_M_W_FL   = 24   # wheel spin rate FL  (rad/s)
_M_W_FR   = 25   # wheel spin rate FR  (rad/s)
_M_W_RL   = 26   # wheel spin rate RL  (rad/s)
_M_W_RR   = 27   # wheel spin rate RR  (rad/s)

# Thermal nodes (28:38) — °C, 5-node front+rear
_T_ALL_START = 28
_T_ALL_END   = 38

# Transient slip (38:46) — α_t first, then κ_t per wheel
_S_FL_A = 38;  _S_FL_K = 39   # front-left:  slip angle, slip ratio
_S_FR_A = 40;  _S_FR_K = 41   # front-right
_S_RL_A = 42;  _S_RL_K = 43   # rear-left
_S_RR_A = 44;  _S_RR_K = 45   # rear-right

# Observable DOF indices: what Car.msg gives us in the 46-DOF space
OBS_INDICES: jnp.ndarray = jnp.array([0, 1, 5, 14, 15, 19])  # X, Y, ψ, vx, vy, wz

# ─── Physical constants ───────────────────────────────────────────────────────
R_WHEEL = 0.2032   # m — from sim_config.py VC.tire_radius (mirena_const.h: 0.2023)

# ─── Nominal cold-start state ─────────────────────────────────────────────────
_NOMINAL_STATE_NP = np.zeros(STATE_DIM, dtype=np.float32)
_NOMINAL_STATE_NP[_T_ALL_START:_T_ALL_END] = 25.0   # °C ambient tire temperature
NOMINAL_STATE: jnp.ndarray = jnp.array(_NOMINAL_STATE_NP)


# ─── Core JIT functions ───────────────────────────────────────────────────────

@jax.jit
def expand_to_46dof(
    car_6d:       jnp.ndarray,   # [6]  [x, y, ψ, vx, vy, wz]  — Car.msg order
    wheel_speeds: jnp.ndarray,   # [4]  [ω_fl, ω_fr, ω_rl, ω_rr] rad/s — WheelSpeeds
    prev_state:   jnp.ndarray,   # [46] previous Project-GP state (thermal continuity)
) -> jnp.ndarray:
    """
    Fuses MirenaSim observables with previous Project-GP state into full 46-DOF.

    Update strategy:
      - Directly observed (Car.msg):      x, y, ψ, vx, vy, wz
      - Directly observed (WheelSpeeds):  ω_fl..rr → also used for slip estimation
      - Kinematically estimated:          slip κ (tanh-bounded), slip angle α
      - Carried from prev_state:          thermal, suspension travel, suspension rates
        (unobservable from Godot — carrying prev values is physically correct)

    Note on Car.msg ordering: Car.msg fields are [x, y, psi, u, v, omega],
    so car_6d[2]=ψ, car_6d[3]=vx, car_6d[4]=vy, car_6d[5]=wz.
    """
    vx  = car_6d[3]   # longitudinal velocity
    vy  = car_6d[4]   # lateral velocity
    eps = 1e-3        # velocity floor — keeps slip differentiable through standstill

    # Longitudinal slip ratio κ = tanh((v_wheel - vx) / |vx|)
    # tanh bounds to (-1, 1) — physically correct, smooth gradient everywhere
    v_wheel = wheel_speeds * R_WHEEL                              # [4] m/s
    kappa   = jnp.tanh((v_wheel - vx) / (jnp.abs(vx) + eps))    # [4]

    # Slip angle α ≈ -atan(vy / vx) — scalar (symmetric 2D approximation for FS)
    alpha_scalar = -jnp.arctan(vy / (jnp.abs(vx) + eps))
    alpha_vec    = jnp.full(4, alpha_scalar)                      # [4]

    return (
        prev_state
        # ── Observed positions ──────────────────────────────────────────────
        .at[_M_X    ].set(car_6d[0])   # world X
        .at[_M_Y    ].set(car_6d[1])   # world Y
        .at[_M_PSI  ].set(car_6d[2])   # yaw ψ
        # ── Observed velocities ─────────────────────────────────────────────
        .at[_M_U    ].set(car_6d[3])   # vx
        .at[_M_V    ].set(car_6d[4])   # vy
        .at[_M_OMEGA].set(car_6d[5])   # wz
        # ── Wheel spin rates ────────────────────────────────────────────────
        .at[_M_W_FL ].set(wheel_speeds[0])
        .at[_M_W_FR ].set(wheel_speeds[1])
        .at[_M_W_RL ].set(wheel_speeds[2])
        .at[_M_W_RR ].set(wheel_speeds[3])
        # ── Estimated slip states ───────────────────────────────────────────
        .at[_S_FL_A ].set(alpha_vec[0])
        .at[_S_FL_K ].set(kappa[0])
        .at[_S_FR_A ].set(alpha_vec[1])
        .at[_S_FR_K ].set(kappa[1])
        .at[_S_RL_A ].set(alpha_vec[2])
        .at[_S_RL_K ].set(kappa[2])
        .at[_S_RR_A ].set(alpha_vec[3])
        .at[_S_RR_K ].set(kappa[3])
        # ── Thermal, suspension, Z, roll, pitch: inherited from prev_state ──
    )


@jax.jit
def collapse_to_6dof(state_46d: jnp.ndarray) -> jnp.ndarray:
    """Extract the 6 MirenaSim-observable DOFs from the 46-DOF state."""
    return state_46d[OBS_INDICES]


@jax.jit
def covariance_to_precision(
    cov36:         jnp.ndarray,  # [36] row-major 6×6 from Car.msg
    min_precision: float = 1e-2,
    max_precision: float = 1e4,
) -> jnp.ndarray:
    """
    Converts Car.msg covariance to per-DOF precision weights (1/σ²) for sysid loss.

    High precision (low covariance) → stronger gradient pull in online adaptation.
    Low precision (high covariance) → attenuated — don't chase uncertain measurements.
    """
    cov6x6    = cov36.reshape(6, 6)
    variances  = jnp.diag(cov6x6)                          # [6]
    precisions = 1.0 / (variances + 1.0 / max_precision)   # regularised
    return jnp.clip(precisions, min_precision, max_precision)


@jax.jit
def state_residual_on_obs(
    pred_46d:  jnp.ndarray,  # [46] Project-GP predicted next state
    true_6d:   jnp.ndarray,  # [6]  MirenaSim ground truth (Car.msg)
    precision: jnp.ndarray,  # [6]  per-DOF precisions from covariance_to_precision
) -> jnp.ndarray:
    """
    Precision-weighted squared residual on observable DOFs only.
    Scalar loss term for online sysid — gradient flows through pred_46d only.
    """
    pred_obs = pred_46d[OBS_INDICES]          # [6]
    residual = pred_obs - true_6d             # [6]
    return jnp.sum(precision * residual ** 2) # scalar


def make_nominal_state() -> jnp.ndarray:
    """Cold-start 46-DOF state: zeros everywhere, tire thermals at 25 °C."""
    return NOMINAL_STATE