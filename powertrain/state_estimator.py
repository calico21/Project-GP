# powertrain/state_estimator.py
# Project-GP — Batch 10.5: JAX-Native Unscented Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════
#
# Estimates the following unmeasured states from noisy sensor data:
#   · Lateral velocity (vy)           — critical for slip angle computation
#   · Normal loads (Fz_fl..Fz_rr)     — critical for friction budgets
#   · Transient slip angles (α_t × 4) — critical for TC/TV
#   · IMU bias states (ax,ay,az,wx,wy,wz)  — for drift compensation
#
# Sensor inputs (from observe_sensors in vehicle_dynamics.py):
#   · IMU accelerometer  (ax, ay, az) @ 200 Hz
#   · IMU gyroscope      (wx, wy, wz) @ 200 Hz
#   · Wheel speed encoders (ω_fl..ω_rr) @ 200 Hz
#   · Steering angle sensor (δ) @ 200 Hz
#   · GPS longitudinal speed (vx) @ 10 Hz (used as aiding)
#
# The UKF is chosen over EKF because:
#   1. The tire force model is highly nonlinear (Pacejka)
#   2. σ-point propagation captures second-order statistics
#   3. No Jacobian computation needed (unlike EKF)
#   4. Naturally handles the multiplicative noise in IMU bias
#
# JAX CONTRACT:
#   · All functions are pure JAX, JIT-safe, vmap-safe
#   · No Python conditionals inside traced code
#   · Symmetric positive-definite P maintained via Cholesky + reconstitution
#   · Fully differentiable — gradients flow through the filter for
#     end-to-end optimization of sensor placement / noise parameters
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# §1  State and Measurement Definitions
# ─────────────────────────────────────────────────────────────────────────────

# UKF state vector (15-dim):
#   [0]   vx      — longitudinal velocity  [m/s]
#   [1]   vy      — lateral velocity        [m/s]
#   [2]   wz      — yaw rate                [rad/s]
#   [3:7] Fz      — normal loads [FL,FR,RL,RR]  [N]
#   [7:11] alpha_t — transient slip angles [FL,FR,RL,RR]  [rad]
#   [11:14] bias   — IMU accel bias [ax,ay,az]  [m/s²]
#   NOTE: gyro bias is small enough to omit from state (reduces dim)

N_STATE = 14

# Measurement vector (10-dim):
#   [0]   ax_imu
#   [1]   ay_imu
#   [2]   wz_gyro
#   [3:7] omega_wheel [FL,FR,RL,RR]
#   [4]   delta_steer
#   [5]   vx_gps  (may be NaN when GPS update not available)

N_MEAS = 10


class UKFState(NamedTuple):
    """Unscented Kalman Filter state."""
    x:   jax.Array   # (N_STATE,) state estimate
    P:   jax.Array   # (N_STATE, N_STATE) covariance
    t:   jax.Array   # scalar: time since last GPS update [s]


class UKFParams(NamedTuple):
    """UKF tuning parameters."""
    # Process noise standard deviations
    q_vx:       float = 0.5      # m/s² velocity process noise
    q_vy:       float = 1.0      # m/s² lateral velocity (higher: less model trust)
    q_wz:       float = 0.2      # rad/s² yaw rate process noise
    q_Fz:       float = 50.0     # N normal load process noise
    q_alpha:    float = 0.05     # rad slip angle process noise
    q_bias:     float = 0.001    # m/s² IMU bias random walk

    # Measurement noise standard deviations
    r_ax:       float = 0.15     # m/s² IMU accel
    r_ay:       float = 0.15
    r_wz:       float = 0.005    # rad/s gyro
    r_omega:    float = 0.3      # rad/s wheel speed
    r_delta:    float = 0.002    # rad steering
    r_vx_gps:   float = 0.5      # m/s GPS speed

    # UKF scaling parameters (Wan-van der Merwe defaults)
    alpha_ukf:  float = 1.0      # MUST be 1.0 for float32 numerical stability
    beta_ukf:   float = 2.0      # prior: Gaussian
    kappa_ukf:  float = 0.0      # secondary scaling

    # Vehicle geometry (needed for process model)
    lf:         float = 0.8525   # m
    lr:         float = 0.6975   # m
    track_f:    float = 1.200    # m
    track_r:    float = 1.180    # m
    R_wheel:    float = 0.2032   # m
    mass:       float = 320.0    # kg
    h_cg:       float = 0.330    # m
    rl:         float = 0.35     # m relaxation length

    @staticmethod
    def from_vehicle_params(vp: dict) -> 'UKFParams':
        return UKFParams(
            lf=vp.get('lf', 0.8525),
            lr=vp.get('lr', 0.6975),
            track_f=vp.get('track_front', 1.200),
            track_r=vp.get('track_rear', 1.180),
            R_wheel=vp.get('wheel_radius', 0.2032),
            mass=vp.get('total_mass', 320.0),
            h_cg=vp.get('h_cg', 0.330),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §2  UKF Sigma Point Machinery
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sigma_weights(n: int, params: UKFParams):
    """
    Compute Wan-van der Merwe sigma point weights.

    Returns:
        W_m: (2n+1,) mean weights
        W_c: (2n+1,) covariance weights
        lam: scaling parameter
    """
    alpha = params.alpha_ukf
    beta  = params.beta_ukf
    kappa = params.kappa_ukf
    lam   = alpha ** 2 * (n + kappa) - n

    W_m = jnp.full(2 * n + 1, 0.5 / (n + lam))
    W_m = W_m.at[0].set(lam / (n + lam))

    W_c = jnp.full(2 * n + 1, 0.5 / (n + lam))
    W_c = W_c.at[0].set(lam / (n + lam) + (1.0 - alpha ** 2 + beta))

    return W_m, W_c, lam


def _generate_sigma_points(
    x: jax.Array,
    P: jax.Array,
    lam: float,
) -> jax.Array:
    """
    Generate 2n+1 sigma points via Cholesky decomposition.

    Args:
        x: (n,) state mean
        P: (n,n) state covariance
        lam: scaling parameter

    Returns:
        (2n+1, n) sigma point matrix
    """
    n = x.shape[0]
    scale = jnp.sqrt(n + lam)

    # Symmetrize P and add jitter for numerical stability
    P_sym = 0.5 * (P + P.T)
    L = jnp.linalg.cholesky(P_sym + 1e-6 * jnp.eye(n))
    S = scale * L

    # sigma_points[0] = x (mean)
    # sigma_points[1:n+1] = x + S columns
    # sigma_points[n+1:2n+1] = x - S columns
    sigma_plus  = x[None, :] + S.T     # (n, n) → each row is x + s_i
    sigma_minus = x[None, :] - S.T
    return jnp.concatenate([x[None, :], sigma_plus, sigma_minus], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Process Model (bicycle + load transfer + slip dynamics)
# ─────────────────────────────────────────────────────────────────────────────

def _process_model(
    x_sigma: jax.Array,
    u_delta: jax.Array,
    dt: float,
    params: UKFParams,
) -> jax.Array:
    """
    Propagate a single sigma point through the nonlinear process model.

    Simplified bicycle model augmented with quasi-static load transfer
    and first-order slip dynamics. NOT the full 46-DOF model — that would
    be too expensive for 2n+1 sigma points at 200 Hz.

    Args:
        x_sigma: (N_STATE,) single sigma point
        u_delta: scalar steering angle [rad]
        dt: timestep [s]
        params: UKF parameters

    Returns:
        (N_STATE,) propagated sigma point
    """
    vx   = x_sigma[0]
    vy   = x_sigma[1]
    wz   = x_sigma[2]
    Fz   = x_sigma[3:7]
    a_t  = x_sigma[7:11]
    bias = x_sigma[11:14]

    # Clamp velocities for numerical safety
    vx_s = jnp.maximum(jnp.abs(vx), 0.5)

    # ── Bicycle model kinematics ────────────────────────────────────────
    # dvx/dt ≈ ax + vy·wz  (Coriolis in body frame)
    # dvy/dt ≈ ay - vx·wz
    # dwz/dt ≈ (Fy_f·lf - Fy_r·lr) / Iz  (simplified, not propagated fully)

    # We don't propagate vx/vy/wz with full dynamics — the IMU measurement
    # update handles this. Process model just applies kinematic consistency.
    vx_new = vx    # held constant between IMU updates
    vy_new = vy
    wz_new = wz

    # ── Quasi-static load transfer ──────────────────────────────────────
    m  = params.mass
    g  = 9.81
    lf = params.lf
    lr = params.lr
    L  = lf + lr
    tf = params.track_f
    tr = params.track_r
    h  = params.h_cg

    ay_est = vx * wz  # centripetal (body frame)

    # Static + longitudinal + lateral load transfer
    F_grav_f = m * g * lr / L
    F_grav_r = m * g * lf / L
    dFz_lat_f = m * jnp.clip(ay_est, -50.0, 50.0) * h / tf
    dFz_lat_r = m * jnp.clip(ay_est, -50.0, 50.0) * h / tr

    Fz_new = jnp.array([
        jnp.maximum(F_grav_f * 0.5 - dFz_lat_f * 0.5, 10.0),
        jnp.maximum(F_grav_f * 0.5 + dFz_lat_f * 0.5, 10.0),
        jnp.maximum(F_grav_r * 0.5 - dFz_lat_r * 0.5, 10.0),
        jnp.maximum(F_grav_r * 0.5 + dFz_lat_r * 0.5, 10.0),
    ])
    # Blend toward quasi-static (low-pass filter on Fz)
    tau_Fz = 0.02   # 50 Hz bandwidth on load estimate
    alpha_Fz = jnp.clip(dt / (tau_Fz + dt), 0.0, 1.0)
    Fz_out = Fz + alpha_Fz * (Fz_new - Fz)

    # ── Transient slip angle dynamics ───────────────────────────────────
    # α_kin = δ - arctan(vy + wz·l / vx) for each corner
    tf2 = tf / 2.0
    tr2 = tr / 2.0
    eps_v = 0.5

    alpha_kin_fl = u_delta - jnp.arctan2(vy + wz * lf, jnp.maximum(jnp.abs(vx - wz * tf2), eps_v))
    alpha_kin_fr = u_delta - jnp.arctan2(vy + wz * lf, jnp.maximum(jnp.abs(vx + wz * tf2), eps_v))
    alpha_kin_rl = -jnp.arctan2(vy - wz * lr, jnp.maximum(jnp.abs(vx - wz * tr2), eps_v))
    alpha_kin_rr = -jnp.arctan2(vy - wz * lr, jnp.maximum(jnp.abs(vx + wz * tr2), eps_v))

    alpha_kin = jnp.array([alpha_kin_fl, alpha_kin_fr, alpha_kin_rl, alpha_kin_rr])

    tau_slip = params.rl / vx_s
    a_t_new = a_t + dt * (alpha_kin - a_t) / tau_slip

    # ── IMU bias: random walk (no explicit update in process model) ─────
    bias_new = bias  # updated via measurement residuals

    return jnp.concatenate([
        jnp.array([vx_new, vy_new, wz_new]),
        Fz_out,
        a_t_new,
        bias_new,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# §4  Measurement Model
# ─────────────────────────────────────────────────────────────────────────────

def _measurement_model(
    x_sigma: jax.Array,
    u_delta: jax.Array,
    params: UKFParams,
) -> jax.Array:
    """
    Predicted measurement from a single sigma point.

    Maps state → expected sensor readings.

    Returns:
        (N_MEAS,) predicted measurement vector
    """
    vx   = x_sigma[0]
    vy   = x_sigma[1]
    wz   = x_sigma[2]
    Fz   = x_sigma[3:7]
    a_t  = x_sigma[7:11]
    bias = x_sigma[11:14]

    # IMU accelerometer prediction (body frame)
    # ax_body = dvx/dt - vy·wz ≈ -vy·wz (steady-state approx) + bias
    # ay_body = dvy/dt + vx·wz ≈ vx·wz + bias
    ax_pred = -vy * wz + bias[0]   # Coriolis in body frame
    ay_pred = vx * wz + bias[1]

    # Gyroscope prediction
    wz_pred = wz

    # Wheel speed prediction: ω = vx_wheel / R
    R = params.R_wheel
    tf2 = params.track_f / 2.0
    tr2 = params.track_r / 2.0
    omega_fl = jnp.maximum(jnp.abs(vx - wz * tf2), 0.1) / R
    omega_fr = jnp.maximum(jnp.abs(vx + wz * tf2), 0.1) / R
    omega_rl = jnp.maximum(jnp.abs(vx - wz * tr2), 0.1) / R
    omega_rr = jnp.maximum(jnp.abs(vx + wz * tr2), 0.1) / R

    # Steering angle (direct pass-through)
    delta_pred = u_delta

    # GPS speed (direct observation of vx)
    vx_gps_pred = vx

    return jnp.array([
        ax_pred, ay_pred, wz_pred,
        omega_fl, omega_fr, omega_rl, omega_rr,
        delta_pred, vx_gps_pred, 0.0,   # padding to N_MEAS=10
    ])


# ─────────────────────────────────────────────────────────────────────────────
# §5  UKF Predict + Update
# ─────────────────────────────────────────────────────────────────────────────

def _build_Q(params: UKFParams, dt: float) -> jax.Array:
    """Process noise covariance (diagonal)."""
    q = jnp.array([
        params.q_vx,      # vx
        params.q_vy,      # vy
        params.q_wz,      # wz
        params.q_Fz, params.q_Fz, params.q_Fz, params.q_Fz,  # Fz × 4
        params.q_alpha, params.q_alpha, params.q_alpha, params.q_alpha,  # α_t × 4
        params.q_bias, params.q_bias, params.q_bias,   # bias × 3
    ]) ** 2 * dt
    return jnp.diag(q)


def _build_R(params: UKFParams) -> jax.Array:
    """Measurement noise covariance (diagonal)."""
    r = jnp.array([
        params.r_ax,      # ax IMU
        params.r_ay,      # ay IMU
        params.r_wz,      # wz gyro
        params.r_omega, params.r_omega, params.r_omega, params.r_omega,  # ω × 4
        params.r_delta,   # δ
        params.r_vx_gps,  # GPS vx
        1e6,              # padding (large noise = ignored)
    ]) ** 2
    return jnp.diag(r)


@partial(jax.jit, static_argnums=(4,))
def ukf_predict(
    ukf: UKFState,
    u_delta: jax.Array,
    dt: jax.Array,
    params: UKFParams,
    n_state: int = N_STATE,
) -> UKFState:
    """
    UKF prediction step: propagate sigma points through process model.
    """
    W_m, W_c, lam = _compute_sigma_weights(n_state, params)
    Q = _build_Q(params, dt)

    # Generate sigma points
    sigmas = _generate_sigma_points(ukf.x, ukf.P, lam)   # (2n+1, n)

    # Propagate each sigma point through process model
    sigmas_pred = jax.vmap(
        lambda s: _process_model(s, u_delta, dt, params)
    )(sigmas)   # (2n+1, n)

    # Recover predicted mean and covariance
    x_pred = jnp.sum(W_m[:, None] * sigmas_pred, axis=0)

    dx = sigmas_pred - x_pred[None, :]
    P_pred = jnp.sum(W_c[:, None, None] * (dx[:, :, None] * dx[:, None, :]), axis=0) + Q

    # Symmetrize for numerical stability
    P_pred = 0.5 * (P_pred + P_pred.T)

    return UKFState(x=x_pred, P=P_pred, t=ukf.t + dt)


@partial(jax.jit, static_argnums=(4,))
def ukf_update(
    ukf: UKFState,
    z_meas: jax.Array,
    u_delta: jax.Array,
    params: UKFParams,
    n_state: int = N_STATE,
) -> UKFState:
    W_m, W_c, lam = _compute_sigma_weights(n_state, params)
    R = _build_R(params)

    sigmas = _generate_sigma_points(ukf.x, ukf.P, lam)

    z_sigmas = jax.vmap(
        lambda s: _measurement_model(s, u_delta, params)
    )(sigmas)

    z_pred = jnp.sum(W_m[:, None] * z_sigmas, axis=0)

    dz = z_sigmas - z_pred[None, :]
    dx = sigmas - ukf.x[None, :]

    S_raw = jnp.sum(W_c[:, None, None] * (dz[:, :, None] * dz[:, None, :]), axis=0) + R
    T     = jnp.sum(W_c[:, None, None] * (dx[:, :, None] * dz[:, None, :]), axis=0)

    S_stable = 0.5 * (S_raw + S_raw.T) + 1e-6 * jnp.eye(S_raw.shape[0])

    K = jnp.linalg.solve(S_stable.T, T.T).T   # (n_state, n_meas)

    x_new = ukf.x + K @ (z_meas - z_pred)

    # ── Covariance update: standard UKF form  P -= K S K^T  ────────────
    # Avoids the second matrix inversion in the Joseph form (P^{-1} T),
    # which compounds float32 error over O(500) steps and drives P non-PSD.
    P_new = ukf.P - K @ S_stable @ K.T

    # Symmetrize + small diagonal floor to prevent float32 drift escaping
    P_new = 0.5 * (P_new + P_new.T) + 1e-7 * jnp.eye(n_state)

    return UKFState(x=x_new, P=P_new, t=ukf.t)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Combined Predict+Update Step (200 Hz main loop call)
# ─────────────────────────────────────────────────────────────────────────────

def ukf_step(
    ukf: UKFState,
    z_meas: jax.Array,
    u_delta: jax.Array,
    dt: jax.Array,
    params: UKFParams = UKFParams(),
) -> UKFState:
    """
    Single UKF step: predict + update.

    Call at 200 Hz in the powertrain_manager loop.

    Args:
        ukf: current UKF state
        z_meas: (N_MEAS,) sensor readings from observe_sensors()
        u_delta: steering angle [rad]
        dt: timestep [s]
        params: UKF parameters

    Returns:
        Updated UKF state
    """
    ukf_pred = ukf_predict(ukf, u_delta, dt, params)
    return ukf_update(ukf_pred, z_meas, u_delta, params)


# ─────────────────────────────────────────────────────────────────────────────
# §7  State Extraction (interface to powertrain_manager)
# ─────────────────────────────────────────────────────────────────────────────

class EstimatedVehicleState(NamedTuple):
    """Extracted estimated states for powertrain_manager consumption."""
    vx:     jax.Array   # [m/s]
    vy:     jax.Array   # [m/s]
    wz:     jax.Array   # [rad/s]
    Fz:     jax.Array   # (4,) [N]
    alpha_t: jax.Array  # (4,) [rad]


def extract_estimated_state(ukf: UKFState) -> EstimatedVehicleState:
    """
    Extract estimated vehicle state from UKF for powertrain_manager.

    This is the ONLY interface between state_estimator and powertrain_manager.
    powertrain_step should call this instead of reading truth from simulation.
    """
    return EstimatedVehicleState(
        vx=ukf.x[0],
        vy=ukf.x[1],
        wz=ukf.x[2],
        Fz=ukf.x[3:7],
        alpha_t=ukf.x[7:11],
    )


# ─────────────────────────────────────────────────────────────────────────────
# §8  Initialization
# ─────────────────────────────────────────────────────────────────────────────

def make_ukf_state(
    vx_init: float = 0.0,
    params: UKFParams = UKFParams(),
) -> UKFState:
    """
    Initialize UKF with conservative priors.

    Large initial covariance on vy and alpha_t (we know nothing).
    Small initial covariance on vx (from GPS) and Fz (from weight).
    """
    m = params.mass
    g = 9.81
    lf = params.lf
    lr = params.lr
    L = lf + lr

    Fz_f_static = m * g * lr / (2.0 * L)
    Fz_r_static = m * g * lf / (2.0 * L)

    x0 = jnp.array([
        vx_init,           # vx
        0.0,               # vy (unknown)
        0.0,               # wz
        Fz_f_static, Fz_f_static,   # Fz FL, FR
        Fz_r_static, Fz_r_static,   # Fz RL, RR
        0.0, 0.0, 0.0, 0.0,         # alpha_t
        0.0, 0.0, 0.0,              # IMU bias
    ])

    P0_diag = jnp.array([
        1.0,       # vx uncertainty [m²/s²]
        5.0,       # vy uncertainty (large — no direct measurement)
        0.1,       # wz uncertainty
        200.0, 200.0, 200.0, 200.0,   # Fz uncertainty [N²]
        0.1, 0.1, 0.1, 0.1,           # alpha_t uncertainty [rad²]
        0.01, 0.01, 0.01,             # bias uncertainty
    ]) ** 2

    return UKFState(
        x=x0,
        P=jnp.diag(P0_diag),
        t=jnp.array(0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §9  Sensor Reading → Measurement Vector Packing
# ─────────────────────────────────────────────────────────────────────────────

def pack_measurement(
    ax_imu: jax.Array,
    ay_imu: jax.Array,
    wz_gyro: jax.Array,
    omega_fl: jax.Array,
    omega_fr: jax.Array,
    omega_rl: jax.Array,
    omega_rr: jax.Array,
    delta_steer: jax.Array,
    vx_gps: jax.Array,
) -> jax.Array:
    """Pack individual sensor readings into the measurement vector."""
    return jnp.array([
        ax_imu, ay_imu, wz_gyro,
        omega_fl, omega_fr, omega_rl, omega_rr,
        delta_steer, vx_gps, 0.0,
    ])


def pack_measurement_from_reading(reading) -> jax.Array:
    """Pack a SensorReading NamedTuple into measurement vector."""
    return jnp.array([
        reading.ax_imu, reading.ay_imu, reading.wz_gyro,
        reading.omega_fl, reading.omega_fr,
        reading.omega_rl, reading.omega_rr,
        reading.delta_steer, reading.vx_gps, 0.0,
    ])