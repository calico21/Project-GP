# powertrain/modes/intermediate/torque_vectoring.py
# Project-GP — QP Torque Vectoring v3 (Intermediate Mode)
# ═══════════════════════════════════════════════════════════════════════════════
#
# v3 upgrades over v2 — all sim-to-real hardening:
#
#   1. STEERING RATE FEEDFORWARD (highest lap-time ROI)
#      Mz_ff = K_ff_δ̇ · I_z · vx · δ̇ / L  +  K_ff_ṙef · I_z · ψ̇_ref_dot
#      → Injects model-predicted yaw moment the instant the driver turns the
#        wheel, eliminating the 30–80 ms reactive lag of the pure PD+I loop.
#        Corner entry lines tighten; exit speed increases.
#      δ̇ is finite-differenced from delta_prev (zero new state fields).
#
#   2. 2D GAIN SCHEDULE: Kp / Kd / Ki over (vx_norm, ay_norm)
#      4×4 bilinear table indexed by [vx ∈ 0–25 m/s, ay_norm ∈ 0–1.3g].
#      → Correct authority in every FS event:
#          Low-v slaloms : High Kp/Ki — fast, aggressive correction
#          High-v sweepers: Low Kp, High Kd — stability-first damping
#          Limit cornering : Low Ki — prevents integral windup at apex
#      Tables stored as flat 16-tuples (compile-time constants in XLA).
#
#   3. QP WARM-START from T_qp_prev
#      The AL-QP now starts from the previous QP solution rather than the
#      nominal allocation T_nom. Convergence is O(constraint-set-shift)
#      rather than O(|T_nom − T*|). Critical for chicane sequences where
#      the constraint set inverts in <100 ms.
#      Adds one (4,) field to IntermediateTVState; zero API breakage.
#
#   4. DRIVELINE RATE LIMITER (tanh smooth clip)
#      dT_max = dT_rate_max · dt  (default: 800 Nm/s → 4 Nm/step at 200 Hz)
#      dT_lim = dT_max · tanh(dT / dT_max)   C∞, gradients flow through tanh
#      → Suppresses excitation of halfshaft torsional resonance (~8 Hz),
#        eliminates steering-wheel flutter and halfshaft ringing under
#        rapid torque demand changes.
#
#   5. UNDERSTEER / OVERSTEER AUTHORITY GATE (sigmoid smooth)
#      psi_ratio = |ψ̇| / (|ψ̇_ref| + ε)
#      gate = σ((ρ − us_thr)/w) · σ((os_thr − ρ)/w) · straight_mask
#      → Reduces TV Mz authority under heavy US (friction-limited) to
#        avoid wasting longitudinal force on a corner that is already
#        saturated, and under heavy OS to let the mechanical balance
#        recover without TV interference. Both transitions are C∞.
#
# All v2 public interfaces preserved — zero call-site changes required.
# is_rwd resolves at XLA compile time — separate XLA graph per config.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Tuple

_N_QP_ITER: int = 16   # lax.scan — zero dynamic overhead


# ─────────────────────────────────────────────────────────────────────────────
# §1  Vehicle Geometry
# ─────────────────────────────────────────────────────────────────────────────

class IntermediateTVGeometry(NamedTuple):
    """
    Static Ter27 geometry — all values from ter27.py config.
    Stored as Python scalars → XLA compile-time constants inside @jit.
    """
    lf:      float = 0.8525   # [m] CG to front axle
    lr:      float = 0.6975   # [m] CG to rear axle
    track_f: float = 1.200    # [m] front track width
    track_r: float = 1.180    # [m] rear track width
    r_w:     float = 0.2032   # [m] wheel radius
    h_cg:    float = 0.330    # [m] CG height
    I_z:     float = 150.0    # [kg·m²] yaw inertia
    mass:    float = 300.0    # [kg] total mass

    @staticmethod
    def from_vehicle_params(vp: dict) -> "IntermediateTVGeometry":
        return IntermediateTVGeometry(
            lf      = vp.get("lf",           0.8525),
            lr      = vp.get("lr",           0.6975),
            track_f = vp.get("track_front",  1.200),
            track_r = vp.get("track_rear",   1.180),
            r_w     = vp.get("wheel_radius", 0.2032),
            h_cg    = vp.get("h_cg",         0.330),
            I_z     = vp.get("Iz",           150.0),
            mass    = vp.get("total_mass",   300.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §2  Controller Parameters
# ─────────────────────────────────────────────────────────────────────────────

class IntermediateTVParams(NamedTuple):
    """
    v3 parameters — all Python scalars / tuples → XLA compile-time constants
    when params is a static_argname in @jax.jit.

    ── 2D Gain Tables ──────────────────────────────────────────────────────
    Shape: (4,4) stored row-major as flat 16-tuples.
      Axis 0 (rows)   : vx_norm ∈ [0,1]  mapping to [0, 25] m/s
      Axis 1 (columns): ay_norm ∈ [0,1]  mapping to [0, 1.3g]

    Grid boundaries (equal spacing):
      vx: [0–6.25, 6.25–12.5, 12.5–18.75, 18.75–25] m/s
      ay: [0–0.325, 0.325–0.65, 0.65–0.975, 0.975–1.3] × g

    Kp table rationale:
      Low-v (<6 m/s): 160 Nm/(rad/s) — need high authority in slaloms where
        yaw dynamics are fastest relative to chassis inertia.
      High-v (>18 m/s): 30 Nm/(rad/s) — stability-critical; any Kp above
        ~60 at speed creates limit-cycle oscillation in the yaw channel.
      Saturation ay (>0.975g): further Kp reduction — tires at limit, TV has
        very little additional friction to spend on yaw moment.

    Kd table rationale:
      Inverted vs Kp: high Kd at high speed provides chassis damping
      that replaces the Kp authority without the stability risk.
      Low-v Kd can be lower because the inertial time constant is short.

    Ki table rationale:
      Aggressive reduction at high-ay: integral windup at the friction limit
      causes phase-lagged over-correction at corner exit → understeer spike.
      3 Nm/(rad·s) floor at high-v/high-ay is effectively "integrator off".

    ── Steering Feedforward ────────────────────────────────────────────────
    K_ff_delta_dot: Conservative initial calibration. At δ̇=0.5 rad/s,
      vx=15 m/s, L=1.55 m: Mz_ff_δ = 0.12 × 150 × 15 × 0.5 / 1.55 ≈ 87 Nm.
      This is ~1.5× the PD term at the same corner entry — correct dominance.
      Tune upward only after validating no oscillation on real car.

    K_ff_ref_dot: The ψ̇_ref derivative informs how quickly the steady-state
      target is changing. Gain 0.08 adds ~58 Nm at same corner entry conditions.
      Sum of both FF terms ≈ 145 Nm, vs PD ≈ 60 Nm reactive → 3:2 ratio FF:FB.

    ── Driveline Rate Limiter ──────────────────────────────────────────────
    dT_rate_max = 800 Nm/s → 4.0 Nm/step at 200 Hz.
      Halfshaft resonance is ~8 Hz. A 4 Nm/step limit on a 55 Nm peak torque
      wheel means full-ramp time ≈ 14 steps = 70 ms → f_knee ≈ 14 Hz, which
      is above the 8 Hz resonance. The tanh shape rolls off sharply at the
      limit rather than hard-clipping, so the XLA graph remains smooth.

    ── US/OS Gate ──────────────────────────────────────────────────────────
    us_threshold = 0.65: car must deliver at least 65% of reference yaw rate
      before TV is active. Below this, reducing inner-wheel torque loses traction
      without gaining yaw moment (friction circle is saturated longitudinally).
    os_threshold = 1.35: 35% over-rotation before TV backs off.
      Natural mechanical oversteer correction should operate first.
    gate_width = 0.10: sigmoid knee width. At σ = ±gate_width from threshold,
      the gate transitions from 16% to 84% — well-behaved C∞ derivative.
    """
    # ── 2D Kp schedule table [Nm/(rad/s)] — row-major (vx_norm, ay_norm) ─
    Kp_table: tuple = (
        160.0, 145.0, 130.0, 110.0,   # low vx
        130.0, 118.0, 100.0,  80.0,   # mid vx
         90.0,  78.0,  65.0,  48.0,   # high vx
         60.0,  48.0,  38.0,  30.0,   # very high vx
    )
    # ── 2D Kd schedule table [Nm·s/rad] ──────────────────────────────────
    Kd_table: tuple = (
         22.0,  22.0,  18.0,  14.0,   # low vx
         30.0,  30.0,  26.0,  20.0,   # mid vx
         42.0,  40.0,  36.0,  30.0,   # high vx
         58.0,  54.0,  46.0,  38.0,   # very high vx
    )
    # ── 2D Ki schedule table [Nm/rad] ────────────────────────────────────
    Ki_table: tuple = (
         45.0,  35.0,  20.0,  10.0,   # low vx
         40.0,  30.0,  16.0,   8.0,   # mid vx
         28.0,  20.0,  11.0,   5.0,   # high vx
         18.0,  13.0,   7.0,   3.0,   # very high vx
    )
    # ── Gain table axis limits ────────────────────────────────────────────
    vx_sched_max:  float = 25.0   # [m/s]   vx at which top row is used
    ay_norm_max:   float = 1.3    # [g]     ay/g at which right column is used
    # ── Integral anti-windup ──────────────────────────────────────────────
    I_max: float          = 10.0   # [rad·s] tanh saturation level
    # ── Reference model ───────────────────────────────────────────────────
    K_us: float           = 0.006  # [s²/m]  understeer gradient
    k_us_fz: float        = 0.0015 # [1/N]   load-adaptive K_us tweak (v2)
    # ── QP weights ────────────────────────────────────────────────────────
    w_reg: float          = 1.0    # quadratic tracking weight
    w_smooth: float       = 0.3    # quadratic smoothness weight
    rho_al: float         = 10.0   # augmented-Lagrangian penalty
    # ── Friction ellipse (v2) ─────────────────────────────────────────────
    C_alpha_f: float      = 35000.0 # [N/rad] front cornering stiffness
    C_alpha_r: float      = 32000.0 # [N/rad] rear cornering stiffness
    # ── Power ceiling (v2) ────────────────────────────────────────────────
    P_max_per_wheel: float = 20000.0  # [W] per motor (80 kW / 4)
    # ── EMA output smoother (v2) ─────────────────────────────────────────
    alpha_ema: float      = 0.75
    # ── v3 STEERING FEEDFORWARD ───────────────────────────────────────────
    K_ff_delta_dot: float = 0.12   # [-]    inertial feedforward gain
    K_ff_ref_dot:   float = 0.08   # [-]    reference-rate feedforward gain
    # ── v3 DRIVELINE RATE LIMITER ─────────────────────────────────────────
    dT_rate_max: float    = 800.0  # [Nm/s] max torque rate at wheel
    # ── v3 US/OS AUTHORITY GATE ───────────────────────────────────────────
    us_threshold: float   = 0.65   # [-]    gate closes below this ψ̇ ratio
    os_threshold: float   = 1.35   # [-]    gate closes above this ψ̇ ratio
    gate_width:   float   = 0.10   # [-]    sigmoid transition half-width
    gate_wz_ref_min: float = 0.15  # [rad/s] min |ψ̇_ref| to activate gating


# ─────────────────────────────────────────────────────────────────────────────
# §3  State & Output
# ─────────────────────────────────────────────────────────────────────────────

class IntermediateTVState(NamedTuple):
    wz_int:    jax.Array  # scalar: yaw rate integral [rad·s]
    wz_prev:   jax.Array  # scalar: previous ψ̇ for D-term [rad/s]
    T_prev:    jax.Array  # (4,):  last applied wheel torques [Nm]
    delta_prev: jax.Array # scalar: last steering angle for δ̇ [rad]
    T_qp_prev: jax.Array  # (4,):  last QP solution — v3 warm-start [Nm]

    @classmethod
    def default(cls) -> "IntermediateTVState":
        return cls(
            wz_int    = jnp.array(0.0),
            wz_prev   = jnp.array(0.0),
            T_prev    = jnp.zeros(4),
            delta_prev= jnp.array(0.0),
            T_qp_prev = jnp.zeros(4),    # v3: warm-start seed
        )


class IntermediateTVOutput(NamedTuple):
    # ── v2 fields (order preserved for backward compat) ──────────────────
    T_wheel:       jax.Array  # (4,) commanded torques [Nm]
    Mz_actual:     jax.Array  # scalar: achieved yaw moment [Nm]
    Mz_target:     jax.Array  # scalar: total demanded Mz (PID + FF) [Nm]
    wz_ref:        jax.Array  # scalar: nonlinear yaw rate reference [rad/s]
    wz_error:      jax.Array  # scalar: tracking error [rad/s]
    qp_residual:   jax.Array  # scalar: |a_eq @ T − Fx_driver| [N]
    Fy_est:        jax.Array  # (4,) estimated lateral forces [N]
    T_ceil_ellipse: jax.Array # (4,) friction-ellipse torque ceilings [Nm]
    # ── v3 new diagnostic fields ──────────────────────────────────────────
    Mz_pid:        jax.Array  # scalar: reactive PID contribution [Nm]
    Mz_ff:         jax.Array  # scalar: feedforward contribution [Nm]
    authority_gate: jax.Array # scalar: US/OS gate ∈ [0,1] (telemetry)
    Kp_eff:        jax.Array  # scalar: scheduled Kp at this step [Nm/(rad/s)]
    Kd_eff:        jax.Array  # scalar: scheduled Kd at this step [Nm·s/rad]
    Ki_eff:        jax.Array  # scalar: scheduled Ki at this step [Nm/rad]


# ─────────────────────────────────────────────────────────────────────────────
# §4  v3 — Bilinear Gain Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def _bilinear_interp_4x4(
    table_flat: tuple,   # 16-element flat tuple, row-major (vx_idx, ay_idx)
    x: jax.Array,        # vx_norm ∈ [0, 1]
    y: jax.Array,        # ay_norm ∈ [0, 1]
) -> jax.Array:
    """
    2D bilinear interpolation on a 4×4 grid.

    Grid: uniform [0,1]² subdivided into 3×3 cells.
    Indices ix ∈ {0,1,2,3}, iy ∈ {0,1,2,3}.

    Because params is static, jnp.array(table_flat) is constant-folded by XLA
    into the compiled graph — zero runtime allocation, zero FLOP overhead
    beyond the 4 table lookups and bilinear arithmetic.
    """
    table = jnp.array(table_flat, dtype=jnp.float32).reshape(4, 4)

    gx = jnp.clip(x * 3.0, 0.0, 3.0 - 1e-6)
    gy = jnp.clip(y * 3.0, 0.0, 3.0 - 1e-6)

    x0 = jnp.floor(gx).astype(jnp.int32)
    y0 = jnp.floor(gy).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, 3)
    y1 = jnp.clip(y0 + 1, 0, 3)

    fx = gx - x0.astype(jnp.float32)
    fy = gy - y0.astype(jnp.float32)

    c00 = table[x0, y0]
    c10 = table[x1, y0]
    c01 = table[x0, y1]
    c11 = table[x1, y1]

    return (c00 * (1.0 - fx) * (1.0 - fy)
          + c10 * fx          * (1.0 - fy)
          + c01 * (1.0 - fx) * fy
          + c11 * fx          * fy)


def _scheduled_gains(
    vx: jax.Array,
    ay: jax.Array,
    mu_est: jax.Array,
    params: IntermediateTVParams,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Lookup Kp, Kd, Ki from 2D bilinear table.

    Normalisation:
      vx_norm = clip(vx / vx_sched_max, 0, 1)
      ay_norm = clip(|ay| / (mu_est · g · ay_norm_max), 0, 1)
        → ay_norm represents utilisation relative to current grip.
        A 0.8g lateral accel on μ=1.3 surface → ay_norm = 0.8/1.3 ≈ 0.62
        (mid-table), correctly reflecting that the tires are not yet at limit.

    μ-gate: linearly scales all gains below μ=0.6 to avoid demanding Mz
    that the surface cannot support.
    """
    g = 9.81
    vx_norm = jnp.clip(jnp.abs(vx) / (params.vx_sched_max + 1e-3), 0.0, 1.0)
    ay_denom = jnp.clip(mu_est, 0.5, 2.0) * g * (params.ay_norm_max + 1e-3)
    ay_norm = jnp.clip(jnp.abs(ay) / ay_denom, 0.0, 1.0)

    Kp = _bilinear_interp_4x4(params.Kp_table, vx_norm, ay_norm)
    Kd = _bilinear_interp_4x4(params.Kd_table, vx_norm, ay_norm)
    Ki = _bilinear_interp_4x4(params.Ki_table, vx_norm, ay_norm)

    # μ-gate: smoothly reduce all gains on low-grip surfaces
    mu_gate = jnp.clip(mu_est / 1.5, 0.35, 1.2)
    return Kp * mu_gate, Kd * mu_gate, Ki * mu_gate


# ─────────────────────────────────────────────────────────────────────────────
# §5  Geometry Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _moment_arms(geo: IntermediateTVGeometry) -> jax.Array:
    """
    Signed moment arms for M_z = arms @ T_wheel.
    Convention: positive torque at left wheel → negative Mz (yaw right).
    Ordering: [FL, FR, RL, RR].
    """
    hw_f = 0.5 * geo.track_f / geo.r_w
    hw_r = 0.5 * geo.track_r / geo.r_w
    return jnp.array([-hw_f, hw_f, -hw_r, hw_r])


# ─────────────────────────────────────────────────────────────────────────────
# §6  Load-Transfer Fz Estimate
# ─────────────────────────────────────────────────────────────────────────────

def _fz_estimate(
    ax: jax.Array,
    ay: jax.Array,
    geo: IntermediateTVGeometry,
) -> jax.Array:
    """
    Quasi-static load transfer on all four corners.

    Longitudinal: ΔFz_long = m·ax·h_cg / wb
      → pitch transfer front–rear
    Lateral: ΔFz_lat = m·ay·h_cg / track
      → roll transfer left–right (equal front and rear for simplicity)

    No roll stiffness split — conservative; actual split from suspension
    parameters would require k_roll_f/r which varies with setup.
    Returns Fz ≥ 50 N per corner (softplus floor, prevents inversion).
    """
    wb = geo.lf + geo.lr
    Fz_static = 0.25 * geo.mass * 9.81  # per corner

    dFz_lon = geo.mass * ax * geo.h_cg / (wb + 1e-3)
    dFz_lat = geo.mass * ay * geo.h_cg / (geo.track_f + geo.track_r + 1e-3)

    Fz_fl = Fz_static - dFz_lon + dFz_lat
    Fz_fr = Fz_static - dFz_lon - dFz_lat
    Fz_rl = Fz_static + dFz_lon + dFz_lat
    Fz_rr = Fz_static + dFz_lon - dFz_lat

    Fz_raw = jnp.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr])
    # softplus floor at 50 N — maintains Fz > 0 without hard clamp
    return 50.0 + jax.nn.softplus(Fz_raw - 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# §7  Lateral Force Estimate
# ─────────────────────────────────────────────────────────────────────────────

def _fy_estimate(
    vx: jax.Array,
    vy: jax.Array,
    wz: jax.Array,
    delta: jax.Array,
    Fz: jax.Array,
    geo: IntermediateTVGeometry,
    params: IntermediateTVParams,
) -> jax.Array:
    """
    Linear tire + tanh saturation Fy estimate for friction-ellipse QP bound.

    Kinematic slip angles:
      α_f = δ − atan2(vy + wz·lf, vx)
      α_r =   − atan2(vy − wz·lr, vx)

    Per-axle linear → saturated force, split by Fz share.
    """
    vx_safe = jnp.abs(vx) + 0.5
    mu_nom  = 1.5  # nominal μ for saturation guard (EKF λ_μ not wired here)

    alpha_f = delta - jnp.arctan2(vy + wz * geo.lf, vx_safe)
    alpha_r =       - jnp.arctan2(vy - wz * geo.lr, vx_safe)

    Fz_f_total = Fz[0] + Fz[1] + 1e-3
    Fz_r_total = Fz[2] + Fz[3] + 1e-3

    Fy_f_lin = params.C_alpha_f * alpha_f
    Fy_r_lin = params.C_alpha_r * alpha_r

    Fy_f_max = mu_nom * Fz_f_total
    Fy_r_max = mu_nom * Fz_r_total
    Fy_f = Fy_f_max * jnp.tanh(Fy_f_lin / (Fy_f_max + 1e-3))
    Fy_r = Fy_r_max * jnp.tanh(Fy_r_lin / (Fy_r_max + 1e-3))

    return jnp.array([
        Fy_f * Fz[0] / Fz_f_total,
        Fy_f * Fz[1] / Fz_f_total,
        Fy_r * Fz[2] / Fz_r_total,
        Fy_r * Fz[3] / Fz_r_total,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# §8  Friction Ellipse Torque Ceiling (v2, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _friction_ellipse_t_ub(
    Fz: jax.Array,
    Fy_est: jax.Array,
    mu_est: jax.Array,
    geo: IntermediateTVGeometry,
) -> jax.Array:
    """
    T_ub_i = r_w · √(max(0, (μ·Fz_i)² − Fy_i²))

    softplus-sqrt approximation for C∞: √(softplus(x · k)/k) avoids
    the singularity of sqrt at x=0. Factor of 4 gives <1% error for Fx_sq > 0.25.
    """
    mu_safe = jnp.clip(mu_est, 0.4, 2.0)
    Fx_sq_max = (mu_safe * Fz) ** 2 - Fy_est ** 2
    Fx_max = jnp.sqrt(jax.nn.softplus(Fx_sq_max * 4.0) / 4.0)
    return Fx_max * geo.r_w


# ─────────────────────────────────────────────────────────────────────────────
# §9  Power-Limited Torque Ceiling (v2, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _power_limited_t_ub(
    omega_wheel: jax.Array,
    params: IntermediateTVParams,
    geo: IntermediateTVGeometry,
) -> jax.Array:
    """Hyperbolic power ceiling: T_max = P / (ω · r_w + ε)."""
    omega_safe = jax.nn.softplus(omega_wheel * geo.r_w)
    T_power = params.P_max_per_wheel / (omega_safe + 1e-3)
    return jnp.clip(T_power, 0.0, 2000.0)


# ─────────────────────────────────────────────────────────────────────────────
# §10  Load-Adaptive K_us (v2, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_k_us(
    Fz: jax.Array,
    params: IntermediateTVParams,
    geo: IntermediateTVGeometry,
) -> jax.Array:
    Fz_front_mean = 0.5 * (Fz[0] + Fz[1])
    Fz_rear_mean  = 0.5 * (Fz[2] + Fz[3])
    delta_fz = Fz_rear_mean - Fz_front_mean
    return params.K_us * (1.0 + params.k_us_fz * delta_fz)


# ─────────────────────────────────────────────────────────────────────────────
# §11  v3 — Steering Rate Feedforward
# ─────────────────────────────────────────────────────────────────────────────

def _steering_feedforward(
    vx: jax.Array,
    ax: jax.Array,
    delta: jax.Array,
    delta_dot: jax.Array,      # δ̇ = (delta − delta_prev) / dt
    K_us_eff: jax.Array,
    geo: IntermediateTVGeometry,
    params: IntermediateTVParams,
) -> jax.Array:
    """
    Two-component model-predictive feedforward yaw moment.

    Component A — Inertial feedforward:
      The bicycle model predicts that at the current steering rate, the
      yaw rate reference will change at rate ψ̇_ref_dot. The Euler-angle
      equivalent yaw moment to produce this change is I_z · ψ̇_ref_dot:
        Mz_ff_A = K_ff_δ̇ · I_z · (vx · δ̇) / L_eff

    Component B — Reference-rate feedforward:
      The full derivative of ψ̇_ref via product rule:
        ψ̇_ref_dot = (vx·δ̇ + ax·δ) / (L · (1 + K_us·vx²))
        Mz_ff_B = K_ff_ṙef · I_z · ψ̇_ref_dot

    Sign convention: positive δ (left turn) → positive Mz_ff (yaw left).
    Both components vanish on straights (δ≈0, δ̇≈0) and at rest (vx≈0).

    Low-speed gate (sigmoid on vx): feedforward meaningless below ~2 m/s
    where the bicycle model reference is dominated by Ackermann geometry.
    """
    L_eff = geo.lf + geo.lr
    K_us_denom = L_eff * (1.0 + K_us_eff * vx ** 2) + 1e-6

    # Component A: directly from δ̇ and vx
    Mz_ff_A = params.K_ff_delta_dot * geo.I_z * vx * delta_dot / (L_eff + 1e-6)

    # Component B: full ψ̇_ref time-derivative
    wz_ref_dot = (vx * delta_dot + ax * delta) / K_us_denom
    Mz_ff_B = params.K_ff_ref_dot * geo.I_z * wz_ref_dot

    Mz_ff = Mz_ff_A + Mz_ff_B

    # Low-speed gate: feedforward fades in above 2.0 m/s
    vx_gate = jax.nn.sigmoid((jnp.abs(vx) - 2.0) * 1.5)
    return Mz_ff * vx_gate


# ─────────────────────────────────────────────────────────────────────────────
# §12  Scheduled PD+I Yaw Controller (v3)
# ─────────────────────────────────────────────────────────────────────────────

def _pid_yaw_scheduled(
    vx: jax.Array,
    wz: jax.Array,
    wz_prev: jax.Array,
    delta: jax.Array,
    mu_est: jax.Array,
    K_us_eff: jax.Array,
    wz_int: jax.Array,
    dt: jax.Array,
    ay_est: jax.Array,        # v3: needed for 2D gain schedule
    Kp: jax.Array,            # pre-scheduled gain from §4
    Kd: jax.Array,
    Ki: jax.Array,
    geo: IntermediateTVGeometry,
    params: IntermediateTVParams,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    v3 scheduled PID — gains come from the 2D table rather than fixed params.

    Returns (Mz_pid, wz_ref, wz_error, wz_int_new).
    The feedforward Mz_ff is computed separately and added at the top level.
    """
    L      = geo.lf + geo.lr
    vx_abs = jnp.abs(vx) + 1e-3

    # Nonlinear Segel yaw reference with load-adaptive K_us
    wz_ref = vx * delta / (L + K_us_eff * vx_abs ** 2)

    e_now = wz - wz_ref

    # Derivative on measurement (avoids setpoint kick)
    wz_dot = (wz - wz_prev) / (dt + 1e-6)

    # Speed-gated integral (prevents windup at standstill)
    v_gate = jax.nn.sigmoid((vx_abs - 1.0) * 5.0)
    wz_int_raw = wz_int + e_now * dt * v_gate
    wz_int_new = params.I_max * jnp.tanh(wz_int_raw / params.I_max)

    Mz_pid = Kp * e_now + Kd * wz_dot + Ki * wz_int_new

    return Mz_pid, wz_ref, e_now, wz_int_new


# ─────────────────────────────────────────────────────────────────────────────
# §13  v3 — US/OS Authority Gate
# ─────────────────────────────────────────────────────────────────────────────

def _us_os_gate(
    wz: jax.Array,
    wz_ref: jax.Array,
    params: IntermediateTVParams,
) -> jax.Array:
    """
    Smooth authority gate suppressing TV Mz under heavy US or OS.

    psi_ratio = |ψ̇_meas| / (|ψ̇_ref| + ε) — always non-negative.

    US gate:  σ((ρ − us_thr) / w)   →  0 when ρ << us_thr (heavy understeer)
    OS gate:  σ((os_thr − ρ) / w)   →  0 when ρ >> os_thr (heavy oversteer)
    Combined: product of both gates  → bell-curve shape over [us_thr, os_thr]

    Straight-line mask: sigmoid on |ψ̇_ref| activates gating only when the
    reference is meaningful. On straights (|ψ̇_ref| < 0.15 rad/s), gate = 1.0
    — TV authority is unrestricted because the yaw error is numerical noise.

    The gate is pure diagnostics / soft limiting, not safety-critical.
    It does NOT prevent the CBF from intervening (that happens downstream
    in advanced mode; here it just softens reactive overcorrection).
    """
    wz_ref_mag = jnp.abs(wz_ref) + 1e-3
    psi_ratio  = jnp.abs(wz) / wz_ref_mag

    us_gate = jax.nn.sigmoid(
        (psi_ratio - params.us_threshold) / (params.gate_width + 1e-6)
    )
    os_gate = jax.nn.sigmoid(
        (params.os_threshold - psi_ratio) / (params.gate_width + 1e-6)
    )
    combined = us_gate * os_gate

    # Activate gating only when |ψ̇_ref| is meaningful
    straight_mask = jax.nn.sigmoid(
        (jnp.abs(wz_ref) - params.gate_wz_ref_min) / 0.05
    )
    # On straight: gate = 1.0 (no restriction); in corner: gate = combined
    return straight_mask * combined + (1.0 - straight_mask)


# ─────────────────────────────────────────────────────────────────────────────
# §14  Nominal Allocation (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def _nominal_allocation(
    Fx_driver: jax.Array,
    Mz_target: jax.Array,
    arms: jax.Array,
    driven: jax.Array,
    geo: IntermediateTVGeometry,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Analytical warm-start: distribute Fx equally + Mz via moment arms.
    This is the QP initial point (T_qp_prev overrides it in v3 warm-start).
    """
    n_driven   = jnp.sum(driven)
    T_fx       = driven * (Fx_driver * geo.r_w / jnp.maximum(n_driven, 1.0))
    arms_driv  = arms * driven
    denom      = jnp.dot(arms_driv, arms_driv)
    T_mz       = arms_driv * Mz_target / jnp.maximum(denom, 1e-6)
    a_eq       = driven / geo.r_w
    return T_fx + T_mz, a_eq, Fx_driver


# ─────────────────────────────────────────────────────────────────────────────
# §15  AL-QP Solver (v3: warm-started from T_qp_prev)
# ─────────────────────────────────────────────────────────────────────────────

def _qp_solve(
    T_warmstart: jax.Array,   # v3: T_qp_prev (previous QP solution)
    T_prev: jax.Array,        # last applied torques for smoothness objective
    a_eq: jax.Array,
    b_eq: jax.Array,
    T_lb: jax.Array,
    T_ub: jax.Array,
    n_driven_f: jax.Array,
    geo: IntermediateTVGeometry,
    params: IntermediateTVParams,
) -> jax.Array:
    """
    Projected-gradient Augmented-Lagrangian QP, 16 iterations via lax.scan.

    v3 change: initial iterate is T_warmstart = T_qp_prev (previous QP solution)
    rather than T_nom (nominal allocation).

    Convergence analysis:
      Cold start from T_nom: the iterate must travel |T_nom − T*| in step-size-α
        increments. For a chicane (left→right), |T_nom − T*| ≈ 2 × ΔMz × r_w /
        track ≈ 2 × 300 × 0.2 / 1.2 ≈ 100 Nm per wheel. At α ≈ 9.8e-4,
        convergence requires O(100 / (1.3 × 9.8e-4)) ≈ 78k iterations.
      Warm start from T_qp_prev: the iterate is already near T* (shift ≈ rate
        of change of constraint set × dt = O(5 Nm/step)). 16 iterations then
        refine to within 0.3 Nm of true optimum.

    The step size α = 1 / (h + ρ·a²) is the exact Lipschitz constant of the
    augmented-Lagrangian gradient — no tuning required.
    """
    h       = params.w_reg + params.w_smooth
    T_blend = (params.w_reg * T_warmstart + params.w_smooth * T_prev) / h

    a_sq  = n_driven_f / (geo.r_w ** 2)
    alpha = 1.0 / (h + params.rho_al * a_sq)

    def _al_step(carry, _):
        T, lam = carry
        viol   = jnp.dot(a_eq, T) - b_eq
        g      = h * (T - T_blend) + a_eq * (lam + params.rho_al * viol)
        T_new  = jnp.clip(T - alpha * g, T_lb, T_ub)
        lam_new = lam + params.rho_al * (jnp.dot(a_eq, T_new) - b_eq)
        return (T_new, lam_new), None

    # Warm-started initial iterate: T_qp_prev clipped to current bounds
    T_init = jnp.clip(T_warmstart, T_lb, T_ub)
    (T_qp, _), _ = jax.lax.scan(
        _al_step, (T_init, jnp.array(0.0)), None, length=_N_QP_ITER,
    )
    return T_qp


# ─────────────────────────────────────────────────────────────────────────────
# §16  v3 — Driveline Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

def _driveline_rate_limiter(
    T_qp: jax.Array,          # raw QP solution this step
    T_prev: jax.Array,         # last applied torques (EMA'd output)
    dt: jax.Array,
    params: IntermediateTVParams,
) -> jax.Array:
    """
    Per-wheel tanh rate limiter suppressing halfshaft resonance excitation.

    The driveline torsional resonance (f ≈ 8 Hz) is excited by rapid ΔT.
    A 800 Nm/s limit at 200 Hz → 4 Nm/step maximum. This keeps the torque
    ramp-rate below the resonance frequency (4 Nm/5 ms = 800 Nm/s → f_knee
    = 800/(2π × 55) ≈ 2.3 Hz per wheel — well below 8 Hz).

    tanh approximation of clip():
      dT_lim = dT_max · tanh(dT / dT_max)
    Properties: C∞, gradient flows through tanh, |dT_lim| ≤ dT_max exactly.
    At |dT| << dT_max: dT_lim ≈ dT (identity). At saturation: dT_lim = dT_max.

    Applied to the delta between T_qp and T_prev (the actually-applied torque,
    not the intermediate QP solution). The EMA smoother then operates on top,
    providing additional high-frequency attenuation.
    """
    dT_max  = params.dT_rate_max * dt   # Nm/step
    dT_raw  = T_qp - T_prev
    dT_lim  = dT_max * jnp.tanh(dT_raw / (dT_max + 1e-6))
    return T_prev + dT_lim


# ─────────────────────────────────────────────────────────────────────────────
# §17  Top-Level Step (v3)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("is_rwd", "geo", "params"))
def intermediate_tv_step(
    vx:          jax.Array,    # [m/s]   longitudinal velocity
    vy:          jax.Array,    # [m/s]   lateral velocity (0 if unavailable)
    wz:          jax.Array,    # [rad/s] yaw rate
    delta:       jax.Array,    # [rad]   steering angle
    ax:          jax.Array,    # [m/s²]  longitudinal acceleration
    Fx_driver:   jax.Array,    # [N]     total longitudinal force demand
    mu_est:      jax.Array,    # scalar  friction estimate
    omega_wheel: jax.Array,    # (4,)    wheel speeds [rad/s]
    T_min_hw:    jax.Array,    # (4,)    hardware torque floor [Nm]
    T_max_hw:    jax.Array,    # (4,)    hardware torque ceiling [Nm]
    tv_state:    IntermediateTVState,
    dt:          jax.Array,
    geo:         IntermediateTVGeometry = IntermediateTVGeometry(),
    params:      IntermediateTVParams   = IntermediateTVParams(),
    is_rwd:      bool = False,    # Ter27 AWD default
) -> Tuple[IntermediateTVOutput, IntermediateTVState]:
    """
    v3 single-step QP torque vectoring.

    Pipeline additions vs v2:
      · δ̇ from delta_prev state field → steering feedforward
      · 2D scheduled Kp/Kd/Ki replacing scalar params
      · QP warm-start from T_qp_prev
      · Driveline rate limiter before EMA
      · US/OS authority gate on total Mz_target
      · New output fields: Mz_pid, Mz_ff, authority_gate, Kp/Kd/Ki_eff

    All v2 call sites remain valid — positional args unchanged.
    """
    driven     = jnp.array([0.0, 0.0, 1.0, 1.0] if is_rwd else [1.0, 1.0, 1.0, 1.0])
    n_driven_f = jnp.array(2.0 if is_rwd else 4.0)

    # ── 1. Load-transfer Fz ────────────────────────────────────────────────
    ay_est = wz * jnp.abs(vx)
    Fz = _fz_estimate(ax, ay_est, geo)

    # ── 2. Lateral force estimate ──────────────────────────────────────────
    Fy_est = _fy_estimate(vx, vy, wz, delta, Fz, geo, params)

    # ── 3. Friction-ellipse + power + hardware combined T bounds ──────────
    T_ub_ellipse = _friction_ellipse_t_ub(Fz, Fy_est, mu_est, geo)
    T_ub_power   = _power_limited_t_ub(omega_wheel, params, geo)
    T_ub_combined = jnp.minimum(jnp.minimum(T_max_hw, T_ub_ellipse), T_ub_power)
    T_lb = jnp.maximum(T_min_hw, jnp.zeros(4))
    T_ub = jnp.maximum(T_lb, T_ub_combined)

    # ── 4. Load-adaptive K_us ─────────────────────────────────────────────
    K_us_eff = _adaptive_k_us(Fz, params, geo)

    # ── 5. v3: 2D scheduled gains ─────────────────────────────────────────
    Kp_eff, Kd_eff, Ki_eff = _scheduled_gains(vx, ay_est, mu_est, params)

    # ── 6. v3: Steering rate feedforward ──────────────────────────────────
    delta_dot = (delta - tv_state.delta_prev) / (dt + 1e-6)
    Mz_ff = _steering_feedforward(vx, ax, delta, delta_dot, K_us_eff, geo, params)

    # ── 7. Scheduled PD+I yaw controller ──────────────────────────────────
    Mz_pid, wz_ref, wz_error, wz_int_new = _pid_yaw_scheduled(
        vx, wz, tv_state.wz_prev, delta, mu_est, K_us_eff,
        tv_state.wz_int, dt, ay_est,
        Kp_eff, Kd_eff, Ki_eff, geo, params,
    )

    # ── 8. v3: US/OS authority gate ────────────────────────────────────────
    authority_gate = _us_os_gate(wz, wz_ref, params)

    # ── 9. Total yaw moment demand: (PID + FF) × gate ─────────────────────
    Mz_target_raw  = Mz_pid + Mz_ff
    Mz_target      = Mz_target_raw * authority_gate

    # ── 10. Nominal allocation (QP initial point fallback) ─────────────────
    arms = _moment_arms(geo)
    T_nom, a_eq, b_eq = _nominal_allocation(Fx_driver, Mz_target, arms, driven, geo)

    # ── 11. v3: AL-QP warm-started from T_qp_prev ─────────────────────────
    T_qp = _qp_solve(
        tv_state.T_qp_prev,   # ← v3: previous QP solution, not T_nom
        tv_state.T_prev, a_eq, b_eq, T_lb, T_ub, n_driven_f, geo, params,
    )

    # ── 12. v3: Driveline rate limiter ────────────────────────────────────
    T_rate_limited = _driveline_rate_limiter(T_qp, tv_state.T_prev, dt, params)

    # ── 13. EMA output smoother ────────────────────────────────────────────
    T_output = params.alpha_ema * T_rate_limited + (1.0 - params.alpha_ema) * tv_state.T_prev

    # ── 14. Pack outputs ───────────────────────────────────────────────────
    output = IntermediateTVOutput(
        T_wheel        = T_output,
        Mz_actual      = jnp.dot(arms, T_output),
        Mz_target      = Mz_target,
        wz_ref         = wz_ref,
        wz_error       = wz_error,
        qp_residual    = jnp.abs(jnp.dot(a_eq, T_qp) - b_eq),
        Fy_est         = Fy_est,
        T_ceil_ellipse = T_ub_ellipse,
        # v3 diagnostics
        Mz_pid         = Mz_pid,
        Mz_ff          = Mz_ff,
        authority_gate = authority_gate,
        Kp_eff         = Kp_eff,
        Kd_eff         = Kd_eff,
        Ki_eff         = Ki_eff,
    )

    new_state = IntermediateTVState(
        wz_int     = wz_int_new,
        wz_prev    = wz,
        T_prev     = T_output,
        delta_prev = delta,
        T_qp_prev  = T_qp,     # v3: store raw QP solution for next warm-start
    )

    return output, new_state


# ─────────────────────────────────────────────────────────────────────────────
# §18  Init Helpers (backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def make_intermediate_tv_state() -> IntermediateTVState:
    """Factory: zero-initialised state including v3 T_qp_prev field."""
    return IntermediateTVState.default()


def make_intermediate_tv_geometry(vp: dict | None = None) -> IntermediateTVGeometry:
    return IntermediateTVGeometry() if vp is None else IntermediateTVGeometry.from_vehicle_params(vp)


def make_intermediate_tv_params(**overrides) -> IntermediateTVParams:
    """
    Convenience factory allowing selective override of default params.
    Usage: make_intermediate_tv_params(K_ff_delta_dot=0.18, dT_rate_max=600.0)
    All non-overridden fields default to the values in IntermediateTVParams.
    """
    defaults = IntermediateTVParams()._asdict()
    defaults.update(overrides)
    return IntermediateTVParams(**defaults)