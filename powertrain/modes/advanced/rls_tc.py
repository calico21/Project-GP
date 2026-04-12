# powertrain/modes/advanced/rls_tc.py
# Project-GP — Recursive Least-Squares Slip-Slope Observer (Primary TC Path)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces DESC as the PRIMARY κ* estimator for Traction Control on real track.
# DESC is retained as the SECONDARY path and survives as a model-free backup
# whenever RLS excitation is insufficient (low slip variance, constant throttle).
#
# ── Motivation ────────────────────────────────────────────────────────────────
# DESC extremum-seeking requires a 15 Hz dither signal to be visible at the
# tire contact patch. Three mechanisms corrupt it in the physical car:
#   1. Halfshaft resonance (~8.2 Hz) attenuates the dither by ~61% at 15 Hz.
#   2. Convergence time (~0.8–2 s) is longer than FS grip transitions (0.1–0.5 s).
#   3. Road roughness in 10–20 Hz band aliases into the lock-in demodulator.
#
# The RLS Slip-Slope Observer avoids all three by extracting dFx/dκ directly
# from natural torque/slip variation — the car's own acceleration profile
# provides sufficient excitation on any dynamic FS track section without
# injecting any deliberate perturbation.
#
# ── Algorithm ─────────────────────────────────────────────────────────────────
#
# Per-axle scalar forgetting-factor RLS:
#   Measurement model: ΔFx_k = θ_k × Δκ_k + ε_k
#     θ_k = dFx/dκ  (the slip stiffness / slope we want)
#   Standard scalar update with forgetting factor λ:
#     k_k  = P_{k-1} × Δκ / (λ + Δκ² × P_{k-1})
#     P_k  = (P_{k-1} - k_k × Δκ × P_{k-1}) / λ
#     θ_k  = θ_{k-1} + k_k × (ΔFx_k − θ_{k-1} × Δκ_k)
#   Update gated on |Δκ| > δ_κ_min (sigmoid-smooth gate).
#
# κ* estimation from slope:
#   Two complementary methods, blended by secant conditioning:
#   A) Two-point secant interpolation of slope zero-crossing:
#       κ*_s = (κ_{k-1}·θ_k - κ_k·θ_{k-1}) / (θ_k - θ_{k-1})
#      Valid when θ sign has recently crossed zero (past the peak).
#   B) Gradient ascent step in direction of positive slope:
#       κ*_g = κ_k + η_step · tanh(θ_k / slope_scale)
#      Valid when still below peak (θ > 0), fast response to μ step.
#   Blend weight: secant_ok = σ((|Δθ| / Δθ_min) − 2)
#   → Secant dominates when Δθ is large (good two-point resolution).
#   → Gradient dominates when slope barely changed (post-peak plateau).
#   Final κ* EMA-smoothed with τ ≈ 25 ms.
#
# DESC fusion gate (continuous, no mode switch):
#   SNR_rls  = |θ| / (√P + ε)   [t-statistic of slope estimate]
#   SNR_desc = |lpf_state| / noise_floor
#   w_rls    = σ((SNR_rls − SNR_desc) / τ_snr)   ∈ (0, 1)
#   κ*_fused = w_rls · κ*_rls + (1 − w_rls) · κ*_desc
#
# ── JAX contract ─────────────────────────────────────────────────────────────
# All functions: pure, no Python-level conditionals inside traced code, C∞,
# safe inside jit/grad/vmap/scan. No host-device syncing.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class RLSParams(NamedTuple):
    """
    RLS observer hyperparameters — all scalars → XLA compile-time constants.

    ── Forgetting factor λ = 0.985 ──────────────────────────────────────────
    Effective data window: 1/(1−λ) = 66.7 steps ≈ 333 ms at 200 Hz.
    At vx = 15 m/s, the car covers ~5 m in this window — about 1 rubber patch
    width. The observer "forgets" the old grip regime just fast enough to track
    μ step changes (0.1–0.5 s) without being too noisy.
    Rule of thumb: λ = 1 − f_event / (2 × f_sample)
      μ transition at ~3 Hz → λ = 1 − 3/400 = 0.9925. Use 0.985 (slightly
      more memory) for additional noise rejection on bumpy tracks.

    ── Initial covariance P0 = 1e6 ──────────────────────────────────────────
    Very large initial uncertainty → first few data points dominate.
    After ~1/(1-λ)/5 ≈ 13 steps with |Δκ| > δ_κ_min, P converges to
    σ²_Fx / (σ²_Δκ) — the noise-to-signal ratio. Convergence is fast.

    ── Minimum Δκ gate: δ_κ_min = 0.005 ────────────────────────────────────
    Below 0.5% slip change, the regressor is near-zero and the gain k ≈ P/ε
    would amplify noise. The sigmoid gate (sharpness 50) transitions from 5%
    to 95% active over the range [δ/2, 3δ/2] = [0.0025, 0.0075].
    Typical slip variation on corner exit: Δκ ≈ 0.01–0.05 per step.
    Typical slip variation at cruise: Δκ ≈ 0.001 (well below gate threshold).

    ── slope_nom = 17500 N/unit_κ ───────────────────────────────────────────
    Initial slope estimate: Hoosier R25 at nominal load 700 N.
    BCD ≈ 12 × 1.65 × 900 ≈ 17820 N. Warm-starting near the physical value
    halves the convergence time vs starting at 0.

    ── slope_scale = 5000 N/unit_κ ──────────────────────────────────────────
    tanh normalisation for gradient step. At slope = slope_scale:
      tanh(1) ≈ 0.76 → 76% of full η_step applied.
    At slope >> slope_scale: full η_step (moving confidently toward κ*).
    At slope << slope_scale: partial step (near κ*, slow approach).
    5000 ≈ slope at κ ≈ 0.08, half-way to peak — correct inflection scale.

    ── eta_step = 0.0020 ────────────────────────────────────────────────────
    Gradient step per timestep. At 200 Hz, full step rate = 0.4 κ/s.
    κ range ≈ 0–0.20 → full traversal in 0.5 s. Responsive to μ transitions.

    ── alpha_kappa_star = 0.80 (τ = 25 ms at 200 Hz) ───────────────────────
    EMA smoothing of κ* output. Prevents chattering from slope estimate noise.
    τ = 1/(f_sample × (1−α)) = 1/(200 × 0.20) = 25 ms — preserves κ* response
    to μ transitions while rejecting sensor noise above ~6 Hz.

    ── dtheta_min = 500 N/unit_κ ────────────────────────────────────────────
    Secant method requires the slope to have changed meaningfully between steps.
    500 = ~3% of slope_nom — a slope change detectable above noise.
    Below this threshold, gradient step is preferred over secant.

    ── SNR fusion: snr_tau = 2.0 ────────────────────────────────────────────
    Sigmoid temperature for RLS/DESC fusion gate.
    At SNR_rls − SNR_desc = ±snr_tau → gate = sigmoid(±1) ≈ {0.73, 0.27}.
    Transition is smooth over a ±4-unit SNR range.
    """
    # ── RLS update ────────────────────────────────────────────────────────
    lambda_f:         float = 0.985      # forgetting factor
    P0:               float = 1e6        # initial covariance
    P_max:            float = 1e8        # covariance upper bound (anti-windup)
    delta_kappa_min:  float = 0.005      # min |Δκ| for RLS update gate
    gate_sharpness:   float = 50.0       # sigmoid sharpness for excitation gate
    slope_nom:        float = 17500.0    # initial slope prior [N/unit_κ]
    # ── κ* estimation ─────────────────────────────────────────────────────
    slope_scale:      float = 5000.0     # tanh scale for gradient step [N/unit_κ]
    eta_step:         float = 0.0020     # gradient step size [κ/step]
    alpha_kappa_star: float = 0.80       # EMA smoothing factor for κ* output
    dtheta_min:       float = 500.0      # min |Δθ| for secant method [N/unit_κ]
    kappa_min:        float = 0.02       # lower κ* bound
    kappa_max:        float = 0.25       # upper κ* bound
    # ── DESC fusion ───────────────────────────────────────────────────────
    snr_tau:          float = 2.0        # sigmoid temperature for SNR gate
    desc_noise_floor: float = 50.0       # DESC lpf noise floor [N]
    # ── Wheel inertia (for Fx estimation, shared with motor model) ────────
    I_wheel:          float = 1.2        # [kg·m²] wheel + motor rotor inertia
    r_w:              float = 0.2032     # [m]     wheel radius


# ─────────────────────────────────────────────────────────────────────────────
# §2  State Types
# ─────────────────────────────────────────────────────────────────────────────

class RLSAxleState(NamedTuple):
    """
    Per-axle RLS slip-slope observer state.
    Front and rear axles run identical, independent observers.
    All fields are scalars — mean over the two wheels on each axle.
    """
    slope:       jax.Array   # dFx/dκ estimate [N/unit_κ] — positive below κ*
    P:           jax.Array   # RLS covariance — shrinks with excitation
    kappa_prev:  jax.Array   # previous mean axle κ (for Δκ computation)
    Fx_prev:     jax.Array   # previous mean axle Fx [N]
    slope_prev:  jax.Array   # previous slope (for secant κ* extrapolation)
    kappa_star:  jax.Array   # current κ* estimate for this axle

    @classmethod
    def default(cls, params: RLSParams = RLSParams()) -> "RLSAxleState":
        return cls(
            slope      = jnp.array(params.slope_nom),
            P          = jnp.array(params.P0),
            kappa_prev = jnp.array(params.kappa_min),
            Fx_prev    = jnp.array(0.0),
            slope_prev = jnp.array(params.slope_nom),
            kappa_star = jnp.array(0.5 * (params.kappa_min + params.kappa_max)),
        )


class RLSState(NamedTuple):
    """Combined front + rear axle RLS states."""
    front: RLSAxleState
    rear:  RLSAxleState

    @classmethod
    def default(cls, params: RLSParams = RLSParams()) -> "RLSState":
        return cls(
            front = RLSAxleState.default(params),
            rear  = RLSAxleState.default(params),
        )


class RLSOutput(NamedTuple):
    """Per-step RLS diagnostics — wired to dashboard and telemetry."""
    kappa_star_rls:  jax.Array   # (4,) per-wheel RLS-derived κ* [after combined-slip]
    kappa_star_fused: jax.Array  # (4,) fused RLS+DESC κ* (the actual output)
    w_rls:           jax.Array   # (2,) per-axle RLS fusion weight ∈ (0,1)
    slope_front:     jax.Array   # scalar: front axle slope estimate [N/unit_κ]
    slope_rear:      jax.Array   # scalar: rear axle slope estimate [N/unit_κ]
    snr_rls_front:   jax.Array   # scalar: front RLS SNR
    snr_rls_rear:    jax.Array   # scalar: rear RLS SNR
    snr_desc_front:  jax.Array   # scalar: front DESC SNR (for comparison)
    snr_desc_rear:   jax.Array   # scalar: rear DESC SNR
    P_front:         jax.Array   # scalar: front RLS covariance (convergence indicator)
    P_rear:          jax.Array   # scalar: rear RLS covariance


# ─────────────────────────────────────────────────────────────────────────────
# §3  Motor-Side Fx Estimator
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _estimate_axle_fx(
    T_wheel:     jax.Array,   # (2,) wheel torques this axle [Nm]
    omega_wheel: jax.Array,   # (2,) wheel angular speeds [rad/s]
    omega_prev:  jax.Array,   # (2,) previous wheel angular speeds
    dt:          jax.Array,   # timestep [s]
    params:      RLSParams,
) -> jax.Array:
    """
    Per-wheel Fx from motor torque accounting: Fx = (T − I_w × α_w) / r_w.

    Subtracting wheel angular acceleration × I_w removes the rotational
    inertia contribution — this is the NET tire longitudinal force, not the
    motor torque divided by radius. Critical for accurate slope estimation
    because the motor torque contains the inertial contribution that does NOT
    act on the road surface.

    Returns mean Fx over the two wheels on the axle (scalar).
    """
    omega_dot = (omega_wheel - omega_prev) / (dt + 1e-6)
    Fx_per_wheel = (T_wheel - params.I_wheel * omega_dot * params.r_w) / (params.r_w + 1e-6)
    return jnp.mean(Fx_per_wheel)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Single Axle RLS Update
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def rls_axle_update(
    state:    RLSAxleState,
    Fx_meas:  jax.Array,    # scalar: measured mean axle Fx [N]
    kappa_meas: jax.Array,  # scalar: measured mean axle κ
    params:   RLSParams,
) -> RLSAxleState:
    """
    One-step per-axle scalar RLS update.

    Standard forgetting-factor RLS for the model ΔFx = θ × Δκ + ε:
      Kalman gain:    k = P × Δκ / (λ + Δκ² × P)
      Covariance:     P_new = (P − k × Δκ × P) / λ
      Slope estimate: θ_new = θ + k × (ΔFx − θ × Δκ)

    Excitation gate (sigmoid-smooth):
      gate = σ(sharpness × (|Δκ| − δ_κ_min))
    When |Δκ| < δ_κ_min: gate ≈ 0 → θ and P unchanged (no update).
    When |Δκ| > δ_κ_min: gate ≈ 1 → standard RLS update.

    The gate prevents covariance explosion (P → ∞) and slope drift
    during low-excitation phases (constant speed, constant torque).

    Covariance clipping at P_max provides upper-bound anti-windup:
    even during extended no-excitation phases, P ≤ P_max → bounded gain.
    """
    dkappa = kappa_meas - state.kappa_prev
    dFx    = Fx_meas   - state.Fx_prev

    # Smooth excitation gate: active when |Δκ| is meaningful
    gate = jax.nn.sigmoid(
        params.gate_sharpness * (jnp.abs(dkappa) - params.delta_kappa_min)
    )

    # ── Scalar RLS gain ───────────────────────────────────────────────────
    Pdk    = state.P * dkappa
    denom  = params.lambda_f + dkappa * Pdk + 1e-6   # always positive
    k_gain = Pdk / denom

    # ── Covariance update ─────────────────────────────────────────────────
    P_raw  = (state.P - k_gain * dkappa * state.P) / (params.lambda_f + 1e-10)
    P_new  = jnp.clip(jnp.abs(P_raw), 1.0, params.P_max)   # abs: keep positive

    # ── Slope estimate update ─────────────────────────────────────────────
    innovation = dFx - state.slope * dkappa
    slope_raw  = state.slope + k_gain * innovation
    slope_new  = gate * slope_raw + (1.0 - gate) * state.slope

    # ── κ* estimation from updated slope ─────────────────────────────────
    kappa_star_new = _kappa_star_from_slope(
        slope_new, state.slope,
        kappa_meas, state.kappa_prev,
        state.kappa_star, params,
    )

    return RLSAxleState(
        slope      = slope_new,
        P          = gate * P_new + (1.0 - gate) * state.P,
        kappa_prev = kappa_meas,
        Fx_prev    = Fx_meas,
        slope_prev = slope_new,
        kappa_star = kappa_star_new,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §5  κ* Extraction from Slope
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _kappa_star_from_slope(
    slope:       jax.Array,   # current slope estimate
    slope_prev:  jax.Array,   # previous slope estimate
    kappa:       jax.Array,   # current axle κ
    kappa_prev:  jax.Array,   # previous axle κ
    kappa_star_prev: jax.Array,  # previous κ* estimate (for EMA)
    params:      RLSParams,
) -> jax.Array:
    """
    Estimate κ* (Pacejka peak slip) from the current slope observation.

    Two methods blended continuously by conditioning quality:

    Method A — Secant interpolation (preferred when Δθ is large):
      Finds κ* as the zero-crossing of the slope function using two
      consecutive (κ, θ) observations:
        κ*_A = (κ_{k-1}·θ_k − κ_k·θ_{k-1}) / (θ_k − θ_{k-1})

      This is the secant method for solving θ(κ*) = 0.
      When the slope has crossed zero (θ changed sign), this directly
      interpolates the crossing point — typical 0.3–0.5 s convergence.
      When the slope has not yet crossed zero, it extrapolates toward the
      expected zero — still useful but less precise.

      Conditioning: secant is reliable when |θ_k − θ_{k-1}| > Δθ_min.

    Method B — Gradient ascent step (preferred near peak / low Δθ):
      Moves κ* in the direction of positive slope with tanh-limited step:
        κ*_B = κ_k + η_step · tanh(θ_k / slope_scale)
      At θ >> slope_scale: full step (aggressive convergence far from peak).
      At θ ≈ 0: near-zero step (correctly stalls at peak).
      At θ < 0: negative step (retreats from over-slip condition).

    Blend: w_A = σ((|Δθ| / Δθ_min) − 2) → 0 when Δθ small, 1 when large.
    Combined: κ*_raw = w_A · κ*_A + (1−w_A) · κ*_B
    EMA:       κ*    = α · κ*_prev + (1−α) · clip(κ*_raw, κ_min, κ_max)
    """
    dtheta = slope - slope_prev

    # ── Method A: Secant cross-ratio ──────────────────────────────────────
    # Safe denominator: adds regularisation term to avoid 0/0 at Δθ→0
    dtheta_safe = dtheta + jnp.sign(dtheta + 1e-12) * params.dtheta_min * 0.01
    kappa_star_secant = (kappa_prev * slope - kappa * slope_prev) / dtheta_safe
    kappa_star_secant = jnp.clip(kappa_star_secant, params.kappa_min, params.kappa_max)

    # ── Method B: Gradient ascent ─────────────────────────────────────────
    kappa_star_gradient = kappa + params.eta_step * jnp.tanh(
        slope / (params.slope_scale + 1e-6)
    )
    kappa_star_gradient = jnp.clip(kappa_star_gradient, params.kappa_min, params.kappa_max)

    # ── Blend by secant conditioning ──────────────────────────────────────
    # secant_ok → 1 when |Δθ| >> Δθ_min; → 0 when |Δθ| << Δθ_min
    secant_ok = jax.nn.sigmoid(jnp.abs(dtheta) / (params.dtheta_min + 1e-6) - 2.0)
    kappa_star_raw = secant_ok * kappa_star_secant + (1.0 - secant_ok) * kappa_star_gradient

    # ── EMA smoothing ─────────────────────────────────────────────────────
    return (params.alpha_kappa_star * kappa_star_prev
            + (1.0 - params.alpha_kappa_star) * kappa_star_raw)


# ─────────────────────────────────────────────────────────────────────────────
# §6  SNR Computation & DESC Fusion
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _rls_snr(state: RLSAxleState, params: RLSParams) -> jax.Array:
    """
    RLS estimation SNR — t-statistic of the slope estimate.

    SNR = |θ| / √P
    High |θ|: the slope is far from zero (we have a clear gradient signal).
    Low √P: high estimation precision (many data points, low noise).
    High SNR → RLS κ* estimate is reliable → w_rls → 1.
    Low SNR (near peak where θ≈0, or low excitation where P is large) → w_rls → 0.

    Adding 1.0 to the denominator prevents singularity at P→0.
    """
    return jnp.abs(state.slope) / (jnp.sqrt(state.P + 1e-3) + 1.0)


@jax.jit
def _desc_snr(lpf_state: jax.Array, params: RLSParams) -> jax.Array:
    """DESC SNR from lock-in signal strength vs noise floor."""
    return jnp.abs(lpf_state) / (params.desc_noise_floor + 1e-3)


@jax.jit
def fuse_rls_desc(
    kappa_star_rls:  jax.Array,   # scalar: RLS κ* for this axle
    kappa_star_desc: jax.Array,   # scalar: DESC κ* for this axle (from desc_step)
    snr_rls:         jax.Array,   # scalar: RLS SNR
    snr_desc:        jax.Array,   # scalar: DESC SNR
    params:          RLSParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Continuous SNR-weighted fusion of RLS and DESC κ* estimates.

    w_rls = σ((SNR_rls − SNR_desc) / τ_snr)

    Properties:
    · w_rls → 1 when RLS is more confident (clear gradient, high excitation)
    · w_rls → 0 when DESC is more confident (strong dither correlation)
    · w_rls ≈ 0.5 when both are equally uncertain (low speed, const torque)
      → in this regime the output is a safe blend, not a hard switch

    The τ_snr = 2.0 temperature means a 4-unit SNR difference gives ~86%
    weight to the dominant method — decisive but not discontinuous.

    Returns: (kappa_star_fused, w_rls)
    """
    w_rls = jax.nn.sigmoid((snr_rls - snr_desc) / (params.snr_tau + 1e-6))
    kappa_star_fused = w_rls * kappa_star_rls + (1.0 - w_rls) * kappa_star_desc
    return kappa_star_fused, w_rls


# ─────────────────────────────────────────────────────────────────────────────
# §7  Full 4-Wheel RLS TC Step
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def rls_tc_step(
    T_applied:      jax.Array,    # (4,) wheel torques from previous step [Nm]
    omega_wheel:    jax.Array,    # (4,) wheel angular speeds [rad/s]
    omega_prev:     jax.Array,    # (4,) previous wheel angular speeds [rad/s]
    vx:             jax.Array,    # scalar longitudinal velocity [m/s]
    Fz:             jax.Array,    # (4,) normal loads [N]
    alpha_t:        jax.Array,    # (4,) transient slip angles [rad]
    alpha_peak:     jax.Array,    # scalar peak slip angle [rad]
    mu_thermal:     jax.Array,    # (4,) thermally-derated friction per wheel
    # DESC parallel path (kept as secondary signal source)
    desc_kappa_ref_f: jax.Array,  # scalar: DESC κ* front axle
    desc_kappa_ref_r: jax.Array,  # scalar: DESC κ* rear axle
    desc_lpf_front:   jax.Array,  # scalar: DESC lock-in front lpf state
    desc_lpf_rear:    jax.Array,  # scalar: DESC lock-in rear lpf state
    rls_state:      RLSState,
    dt:             jax.Array,
    params:         RLSParams = RLSParams(),
) -> tuple[RLSOutput, RLSState]:
    """
    Full 4-wheel RLS TC observer step.

    Pipeline per axle (front and rear independently):
      1. Estimate axle Fx from motor torque accounting (inertia-corrected)
      2. Compute mean axle κ from wheel speeds + vx
      3. RLS slope update (scalar, forgetting-factor)
      4. Extract κ*_rls from slope via secant+gradient blend
      5. DESC SNR vs RLS SNR → continuous fusion weight
      6. Fuse: κ*_fused = w_rls × κ*_rls + (1−w_rls) × κ*_desc
      7. Apply combined-slip correction (friction ellipse κ* reduction)
      8. Apply thermal μ derating on κ*

    Returns per-wheel κ*_fused and full diagnostics.
    """
    # ── 1. Per-axle Fx from motor accounting ──────────────────────────────
    Fx_front = _estimate_axle_fx(
        T_applied[:2], omega_wheel[:2], omega_prev[:2], dt, params,
    )
    Fx_rear = _estimate_axle_fx(
        T_applied[2:], omega_wheel[2:], omega_prev[2:], dt, params,
    )

    # ── 2. Mean axle κ ────────────────────────────────────────────────────
    vx_safe = jnp.maximum(jnp.abs(vx), 0.5)
    kappa_per_wheel = jnp.clip((omega_wheel * params.r_w - vx) / vx_safe, -0.8, 0.8)
    kappa_front_mean = 0.5 * (kappa_per_wheel[0] + kappa_per_wheel[1])
    kappa_rear_mean  = 0.5 * (kappa_per_wheel[2] + kappa_per_wheel[3])

    # ── 3–4. RLS update + κ* extraction (front and rear) ─────────────────
    front_new = rls_axle_update(rls_state.front, Fx_front, kappa_front_mean, params)
    rear_new  = rls_axle_update(rls_state.rear,  Fx_rear,  kappa_rear_mean,  params)

    # ── 5. SNR computation ────────────────────────────────────────────────
    snr_rls_f  = _rls_snr(front_new, params)
    snr_rls_r  = _rls_snr(rear_new,  params)
    snr_desc_f = _desc_snr(desc_lpf_front, params)
    snr_desc_r = _desc_snr(desc_lpf_rear,  params)

    # ── 6. Fuse RLS + DESC per axle ───────────────────────────────────────
    kappa_star_fused_f, w_rls_f = fuse_rls_desc(
        front_new.kappa_star, desc_kappa_ref_f, snr_rls_f, snr_desc_f, params,
    )
    kappa_star_fused_r, w_rls_r = fuse_rls_desc(
        rear_new.kappa_star,  desc_kappa_ref_r, snr_rls_r, snr_desc_r, params,
    )

    # ── 7. Combined-slip correction (friction ellipse) ────────────────────
    # Mirrors kappa_star_combined() in traction_control.py — reduces κ* when
    # cornering to respect the friction ellipse constraint.
    alpha_ratio_sq_f = ((alpha_t[0] + alpha_t[1]) * 0.5 / (jnp.abs(alpha_peak) + 1e-3)) ** 2
    alpha_ratio_sq_r = ((alpha_t[2] + alpha_t[3]) * 0.5 / (jnp.abs(alpha_peak) + 1e-3)) ** 2
    cs_factor_f = jnp.sqrt(jax.nn.softplus((1.0 - jnp.clip(alpha_ratio_sq_f, 0.0, 0.95)) * 10.0) / 10.0 + 1e-6)
    cs_factor_r = jnp.sqrt(jax.nn.softplus((1.0 - jnp.clip(alpha_ratio_sq_r, 0.0, 0.95)) * 10.0) / 10.0 + 1e-6)

    kappa_star_cs_f = kappa_star_fused_f * cs_factor_f
    kappa_star_cs_r = kappa_star_fused_r * cs_factor_r

    # ── 8. Thermal derating of κ* ─────────────────────────────────────────
    # mu_thermal reflects reduced grip at sub/super-optimal tire temp.
    # κ* shrinks proportionally (peak slip shifts left on low-μ surface).
    # Conservative factor: κ*_thermal = κ*_cs × clip(mu_thermal/mu_nom, 0.7, 1.2)
    mu_nom = 1.5
    kappa_star_fl = kappa_star_cs_f * jnp.clip(mu_thermal[0] / mu_nom, 0.70, 1.20)
    kappa_star_fr = kappa_star_cs_f * jnp.clip(mu_thermal[1] / mu_nom, 0.70, 1.20)
    kappa_star_rl = kappa_star_cs_r * jnp.clip(mu_thermal[2] / mu_nom, 0.70, 1.20)
    kappa_star_rr = kappa_star_cs_r * jnp.clip(mu_thermal[3] / mu_nom, 0.70, 1.20)

    kappa_star_fused_4 = jnp.array([kappa_star_fl, kappa_star_fr,
                                     kappa_star_rl, kappa_star_rr])
    kappa_star_rls_4   = jnp.array([front_new.kappa_star, front_new.kappa_star,
                                     rear_new.kappa_star,  rear_new.kappa_star])

    # ── Outputs ───────────────────────────────────────────────────────────
    output = RLSOutput(
        kappa_star_rls   = kappa_star_rls_4,
        kappa_star_fused = kappa_star_fused_4,
        w_rls            = jnp.array([w_rls_f, w_rls_r]),
        slope_front      = front_new.slope,
        slope_rear       = rear_new.slope,
        snr_rls_front    = snr_rls_f,
        snr_rls_rear     = snr_rls_r,
        snr_desc_front   = snr_desc_f,
        snr_desc_rear    = snr_desc_r,
        P_front          = front_new.P,
        P_rear           = rear_new.P,
    )

    new_rls_state = RLSState(front=front_new, rear=rear_new)
    return output, new_rls_state


# ─────────────────────────────────────────────────────────────────────────────
# §8  Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_rls_state(params: RLSParams = RLSParams()) -> RLSState:
    """Zero-initialised RLS state. Convergence in ~13 steps with excitation."""
    return RLSState.default(params)


def make_rls_params(**overrides) -> RLSParams:
    """Convenience factory for selective override of RLS hyperparameters."""
    defaults = RLSParams()._asdict()
    defaults.update(overrides)
    return RLSParams(**defaults)