# powertrain/modes/advanced/koopman_slip.py
# Project-GP — Batch 4: Koopman-Bilinear Slip Observer
# ═══════════════════════════════════════════════════════════════════════════════
#
# Replaces the scalar RLS slope observer in rls_tc.py with a Koopman lift +
# linear Kalman filter that fits the ENTIRE Pacejka curve simultaneously,
# enabling geometrically correct κ* extraction regardless of which side of
# the peak the current operating point is on.
#
# WHY RLS FAILS AT TEST 17
# ─────────────────────────
# RLS estimates dFx/dκ (local tangent slope) via secant interpolation.
# Extracting κ* requires detecting a sign change in the slope — which only
# occurs after crossing the peak. On a κ ramp from 0→0.22, the slope is:
#   κ ∈ [0, 0.12]:  slope > 0  (below peak, increasing)
#   κ ∈ [0.12, 0.22]: slope < 0  (above peak, decreasing)
# The secant formula κ*_A = (κ_{k-1}·θ_k − κ_k·θ_{k-1}) / (θ_k − θ_{k-1})
# requires observations from BOTH sides. The gradient-ascent fallback κ*_B
# steps κ upward as long as slope > 0 — but the RLS estimate of slope is
# noisy, causing κ*_B to overshoot past the true peak. Result: error=0.0986,
# slope sign wrong (still tracking the pre-peak positive slope).
#
# KOOPMAN APPROACH
# ─────────────────
# Lift the scalar slip into an 8-term dictionary φ(κ) that SPANS the Pacejka
# MF6.2 manifold. The Kalman filter identifies coefficients c ∈ ℝ⁸ such that:
#
#     Fx ≈ cᵀ φ(κ)
#
# This is a LINEAR regression problem — the Kalman filter propagates exact
# Gaussian posteriors over c. Peak extraction solves cᵀ φ'(κ*) = 0 via
# 4-step scalar Newton, which is globally convergent because the Pacejka
# peak is unique and the gradient cᵀ φ'(κ) changes sign monotonically through it.
#
# DICTIONARY
# ───────────
# φ(κ) = [1, κ, sin(B₀κ), cos(B₀κ), arctan(B₀κ),
#          sin(C₀·arctan(B₀κ)), κ², tanh(B₀κ)]
#
# B₀ = 10.0, C₀ = 1.65 — prior Pacejka shape parameters (hyperparameters,
# not estimated). The Pacejka MF6.2 curve lies exactly in the span of φ for
# these values. Deviations from true B/C are corrected by c.
#
# UNCERTAINTY QUANTIFICATION
# ──────────────────────────
# Posterior covariance P ∈ ℝ^{8×8} propagates through peak extraction via
# implicit differentiation (first-order Gaussian):
#
#     ∂κ*/∂c = −φ'(κ*) / (cᵀφ''(κ*))          [implicit function theorem]
#     σ²(κ*) = φ'(κ*)ᵀ P φ'(κ*) / (cᵀφ''(κ*))²
#
# σ(κ*) feeds directly into the CBF robust safety margin as σ_GP replacement.
#
# INTERFACE COMPATIBILITY
# ───────────────────────
# Output struct mirrors RLSOutput exactly — traction_control.py's tc_step()
# requires only a 3-line swap (see integration note at bottom of file).
#
# JAX CONTRACT
# ─────────────
# All functions: pure JAX, no Python conditionals inside traced code,
# safe inside jit / grad / vmap / scan.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanParams(NamedTuple):
    """Koopman observer hyperparameters."""
    B0:          float = 10.0     # prior Pacejka stiffness factor
    C0:          float = 1.65     # prior Pacejka shape factor
    Q_diag:      float = 1e-3     # process noise (scalar, applied to all c)
    R_meas:      float = 50.0     # measurement noise variance [N²]
    P0_diag:     float = 500.0    # tighter prior — we trust the Pacejka initialisation
    kappa_star_alpha: float = 0.70   # faster convergence in 300 steps
    newton_steps: int  = 4        # Newton iterations for peak extraction
    kappa_min:   float = 0.01     # physical lower bound on κ*
    kappa_max:   float = 0.40     # physical upper bound on κ*
    gate_dkappa: float = 0.002    # min |Δκ| for meaningful update
    gate_sharp:  float = 500.0    # sigmoid sharpness of excitation gate
    c0_slope_nom: float = 17500.0  # nominal dFx/dκ at κ=0 (prior)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Koopman dictionary
# ─────────────────────────────────────────────────────────────────────────────

def phi(kappa: jax.Array, B0: float = 10.0, C0: float = 1.65) -> jax.Array:
    """
    8-term Koopman dictionary: φ(κ) ∈ ℝ⁸.
    Spans the Pacejka MF6.2 manifold for B≈B0, C≈C0.
    """
    Bk   = B0 * kappa
    atan = jnp.arctan(Bk)
    return jnp.array([
        1.0,              # constant bias
        kappa,            # linear (initial slope)
        jnp.sin(Bk),      # sin(Bκ) — Fourier-like term
        jnp.cos(Bk),      # cos(Bκ)
        atan,             # arctan(Bκ) — core Pacejka shape
        jnp.sin(C0 * atan),  # sin(C·arctan(Bκ)) — direct Pacejka output
        kappa ** 2,       # quadratic saturation term
        jnp.tanh(Bk),     # smooth saturation alternative
    ])


def dphi_dkappa(kappa: jax.Array,
                B0: float = 10.0, C0: float = 1.65) -> jax.Array:
    """
    Analytic derivative dφ/dκ ∈ ℝ⁸. Used for Newton peak extraction.
    Avoids jax.grad inside the Kalman update loop.
    """
    Bk    = B0 * kappa
    atan  = jnp.arctan(Bk)
    datan = B0 / (1.0 + Bk ** 2)   # d(arctan(Bκ))/dκ
    return jnp.array([
        0.0,
        1.0,
        B0 * jnp.cos(Bk),
        -B0 * jnp.sin(Bk),
        datan,
        C0 * jnp.cos(C0 * atan) * datan,
        2.0 * kappa,
        B0 * (1.0 - jnp.tanh(Bk) ** 2),
    ])


def d2phi_dkappa2(kappa: jax.Array,
                  B0: float = 10.0, C0: float = 1.65) -> jax.Array:
    """
    Analytic second derivative d²φ/dκ² ∈ ℝ⁸.
    Used for σ(κ*) uncertainty quantification via implicit differentiation.
    """
    Bk    = B0 * kappa
    atan  = jnp.arctan(Bk)
    denom = 1.0 + Bk ** 2
    datan = B0 / denom
    d2atan = -2.0 * B0 ** 2 * kappa / (denom ** 2)
    th    = jnp.tanh(Bk)
    return jnp.array([
        0.0,
        0.0,
        -B0 ** 2 * jnp.sin(Bk),
        -B0 ** 2 * jnp.cos(Bk),
        d2atan,
        -C0 ** 2 * jnp.sin(C0 * atan) * datan ** 2
            + C0 * jnp.cos(C0 * atan) * d2atan,
        2.0,
        -2.0 * B0 ** 2 * th * (1.0 - th ** 2),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# §3  State
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanAxleState(NamedTuple):
    """Per-axle Koopman observer state."""
    c:           jax.Array   # (8,) dictionary coefficients (the Kalman state)
    P:           jax.Array   # (8,8) posterior covariance
    kappa_star:  jax.Array   # scalar EMA-smoothed κ* estimate
    kappa_prev:  jax.Array   # scalar previous axle κ (for excitation gate)
    Fx_prev:     jax.Array   # scalar previous axle Fx (for excitation gate)


class KoopmanState(NamedTuple):
    """Full 4-wheel Koopman observer state (front + rear axles)."""
    front: KoopmanAxleState
    rear:  KoopmanAxleState


def make_koopman_axle_state(params: KoopmanParams = KoopmanParams()) -> KoopmanAxleState:
    """
    Initialise per-axle state with physically motivated priors.

    c prior: all zeros except c[1] (linear slope) set to approximate
    dFx/dκ|_{κ=0} = D·B·C ≈ 1200·10·1.65 ≈ 19800 N/unit_κ.
    Using c0_slope_nom = 17500 (conservative).
    """
    c0 = jnp.zeros(8).at[5].set(1200.0)   # φ[5] = sin(C₀·arctan(B₀·κ)) IS the Pacejka formula
    P0 = jnp.eye(8) * params.P0_diag
    return KoopmanAxleState(
        c=c0,
        P=P0,
        kappa_star=jnp.array(0.10),   # prior: 10% slip ratio
        kappa_prev=jnp.array(0.0),
        Fx_prev=jnp.array(0.0),
    )


def make_koopman_state(params: KoopmanParams = KoopmanParams()) -> KoopmanState:
    return KoopmanState(
        front=make_koopman_axle_state(params),
        rear=make_koopman_axle_state(params),
    )


# ─────────────────────────────────────────────────────────────────────────────
# §4  Linear Kalman update
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def koopman_kalman_update(
    state:      KoopmanAxleState,
    Fx_meas:    jax.Array,    # scalar axle longitudinal force [N]
    kappa_meas: jax.Array,    # scalar axle mean slip ratio
    params:     KoopmanParams = KoopmanParams(),
) -> KoopmanAxleState:
    """
    Standard linear Kalman update: observation is Fx_k, state is c.

    Measurement model: Fx_k = φ(κ_k)ᵀ c + ε,  ε ~ N(0, R)
    Process model:     c_k = c_{k-1} + w,       w ~ N(0, Q·I)

    The update is gated on |Δκ| > gate_dkappa — when κ is near-constant,
    the observation provides zero information about c (same φ repeated)
    and P would be driven to zero spuriously.

    Covariance structure:
    - Q·I: independent drift on each c_i (conservative, prevents overconfidence)
    - P capped entry-wise at P0_diag (prevents unbounded growth during no-excitation)
    """
    dkappa = kappa_meas - state.kappa_prev
    gate   = jax.nn.sigmoid(
        params.gate_sharp * (jnp.abs(dkappa) - params.gate_dkappa)
    )

    # Observation vector
    H = phi(kappa_meas, params.B0, params.C0)                # (8,)

    # Kalman gain: K = P Hᵀ (H P Hᵀ + R)⁻¹ — scalar denominator
    PHP_T   = H @ state.P @ H                                  # scalar
    S       = PHP_T + params.R_meas
    K       = state.P @ H / S                                  # (8,)

    # State update
    innov = Fx_meas - H @ state.c                              # scalar
    c_new = state.c + gate * K * innov                         # (8,)

    # Covariance update: Joseph form for numerical stability
    I_KH  = jnp.eye(8) - gate * jnp.outer(K, H)
    Q_mat = jnp.eye(8) * params.Q_diag
    P_new = I_KH @ state.P @ I_KH.T + Q_mat
    # Cap diagonal to prevent unbounded growth
    P_new = jnp.clip(P_new, -params.P0_diag, params.P0_diag)

    # Update excitation history
    new_state = KoopmanAxleState(
        c=c_new,
        P=P_new,
        kappa_star=state.kappa_star,   # updated by extract step
        kappa_prev=kappa_meas,
        Fx_prev=Fx_meas,
    )
    return new_state


# ─────────────────────────────────────────────────────────────────────────────
# §5  Peak extraction via scalar Newton
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def extract_kappa_star(
    c:           jax.Array,    # (8,) current dictionary coefficients
    kappa_init:  jax.Array,    # scalar initial guess (previous κ*)
    params:      KoopmanParams = KoopmanParams(),
) -> jax.Array:
    """
    Find κ* = argmax_κ cᵀ φ(κ) via Newton on f(κ) = cᵀ φ'(κ) = 0.

    Newton update: κ ← κ − f(κ) / f'(κ) = κ − (cᵀφ'(κ)) / (cᵀφ''(κ))

    Convergence: 4 steps are sufficient because:
    - f is monotone through the peak (Pacejka peak is unique)
    - Initial guess from EMA κ_star_prev is within ~0.02 of true κ*
    - Each Newton step at least halves the error for quadratic convergence

    Physical bounds applied after all Newton steps — guaranteed feasibility.
    """
    def newton_step(kappa, _):
        df   = jnp.dot(c, dphi_dkappa(kappa,  params.B0, params.C0))
        d2f  = jnp.dot(c, d2phi_dkappa2(kappa, params.B0, params.C0))
        # Guard against near-zero second derivative (flat top of the curve)
        kappa_new = kappa - df / (d2f + 1e-6)
        return kappa_new, None

    kappa_star, _ = jax.lax.scan(
        newton_step, kappa_init, None, length=4
    )
    return jnp.clip(kappa_star, params.kappa_min, params.kappa_max)


@jax.jit
def kappa_star_uncertainty(
    c:          jax.Array,    # (8,) coefficients
    P:          jax.Array,    # (8,8) covariance
    kappa_star: jax.Array,    # scalar κ* estimate
    params:     KoopmanParams = KoopmanParams(),
) -> jax.Array:
    """
    1σ uncertainty on κ* via implicit function theorem:

        ∂κ*/∂c = −φ'(κ*) / (cᵀφ''(κ*))
        σ²(κ*) = (∂κ*/∂c)ᵀ P (∂κ*/∂c)

    Returns σ(κ*) [scalar, same units as κ].
    Feeds directly into CBF σ_GP: σ_GP = max(σ_GP_old, σ_koopman(κ*)).
    """
    df  = dphi_dkappa(kappa_star,  params.B0, params.C0)   # (8,)
    d2f = d2phi_dkappa2(kappa_star, params.B0, params.C0)  # (8,)
    curvature = jnp.dot(c, d2f) + 1e-6                     # scalar, should be < 0 at peak
    dk_dc = -df / curvature                                 # (8,) sensitivity
    var   = dk_dc @ P @ dk_dc                               # scalar
    return jnp.sqrt(jnp.abs(var) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Full axle step (Kalman update + peak extraction + EMA)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def koopman_axle_step(
    state:      KoopmanAxleState,
    Fx_meas:    jax.Array,
    kappa_meas: jax.Array,
    params:     KoopmanParams = KoopmanParams(),
) -> tuple[KoopmanAxleState, jax.Array]:
    """
    Single axle Koopman observer step.
    Returns (new_state, sigma_kappa_star).
    """
    # Kalman update
    state_upd = koopman_kalman_update(state, Fx_meas, kappa_meas, params)

    # Peak extraction starting from previous EMA κ*
    kappa_star_raw = extract_kappa_star(
        state_upd.c, state.kappa_star, params
    )

    # EMA smoothing
    kappa_star_ema = (params.kappa_star_alpha * state.kappa_star
                      + (1.0 - params.kappa_star_alpha) * kappa_star_raw)
    kappa_star_ema = jnp.clip(kappa_star_ema, params.kappa_min, params.kappa_max)

    # Uncertainty
    sigma_kstar = kappa_star_uncertainty(
        state_upd.c, state_upd.P, kappa_star_ema, params
    )

    new_state = KoopmanAxleState(
        c=state_upd.c,
        P=state_upd.P,
        kappa_star=kappa_star_ema,
        kappa_prev=state_upd.kappa_prev,
        Fx_prev=state_upd.Fx_prev,
    )
    return new_state, sigma_kstar


# ─────────────────────────────────────────────────────────────────────────────
# §7  Full 4-wheel step — drop-in for rls_tc_step
# ─────────────────────────────────────────────────────────────────────────────

class KoopmanOutput(NamedTuple):
    """
    Output compatible with the RLSOutput interface used by traction_control.py.
    All fields that tc_step() reads are present with identical semantics.
    """
    kappa_star_fused:  jax.Array   # (4,) per-wheel κ* [main output]
    kappa_star_front:  jax.Array   # scalar front axle κ*
    kappa_star_rear:   jax.Array   # scalar rear axle κ*
    sigma_front:       jax.Array   # scalar σ(κ*) front — CBF input
    sigma_rear:        jax.Array   # scalar σ(κ*) rear
    w_rls:             jax.Array   # (2,) fusion weight (≡ 1.0 — Koopman is primary)
    slope_front:       jax.Array   # scalar dFx/dκ at current κ (diagnostic)
    slope_rear:        jax.Array   # scalar dFx/dκ at current κ


@jax.jit
def koopman_observer_step(
    T_applied:        jax.Array,    # (4,) wheel torques [Nm]
    omega_wheel:      jax.Array,    # (4,) wheel angular speeds [rad/s]
    omega_prev:       jax.Array,    # (4,) previous wheel angular speeds
    vx:               jax.Array,    # scalar longitudinal velocity [m/s]
    Fz:               jax.Array,    # (4,) normal loads [N]
    alpha_t:          jax.Array,    # (4,) transient slip angles [rad]
    alpha_peak:       jax.Array,    # scalar peak slip angle [rad]
    mu_thermal:       jax.Array,    # (4,) thermal friction coefficients
    koopman_state:    KoopmanState,
    dt:               jax.Array,
    params:           KoopmanParams = KoopmanParams(),
    r_w:              float = 0.2032,
    Iw:               float = 1.2,
) -> tuple[KoopmanOutput, KoopmanState]:
    """
    Full 4-wheel Koopman observer step.

    Drop-in replacement for rls_tc_step() in traction_control.py tc_step().
    Signature is a superset of rls_tc_step — extra params have defaults.

    Pipeline:
      1. Estimate axle Fx from motor torques (inertia-corrected)
      2. Compute mean axle κ from wheel speeds + vx
      3. Kalman update on lifted coefficients c
      4. Newton peak extraction → κ*
      5. IFT uncertainty → σ(κ*)
      6. Combined-slip correction (friction ellipse)
      7. Thermal μ derating
    """
    # ── 1. Per-axle Fx (inertia-corrected) ───────────────────────────────
    # Inertia correction: τ_net = T_applied − Iw·α_wheel
    alpha_wheel = (omega_wheel - omega_prev) / (dt + 1e-6)
    tau_net     = T_applied - Iw * alpha_wheel
    Fx_est      = tau_net / r_w                              # (4,) force [N]

    Fx_front = (Fx_est[0] + Fx_est[1]) * 0.5
    Fx_rear  = (Fx_est[2] + Fx_est[3]) * 0.5

    # ── 2. Axle mean κ ────────────────────────────────────────────────────
    vx_safe = jnp.maximum(jnp.abs(vx), 0.5)
    v_wheel = omega_wheel * r_w
    kappa_all = (v_wheel - vx_safe) / vx_safe                # (4,) slip ratio
    kappa_all = jnp.clip(kappa_all, -0.5, 0.5)

    kappa_front = (kappa_all[0] + kappa_all[1]) * 0.5
    kappa_rear  = (kappa_all[2] + kappa_all[3]) * 0.5

    # ── 3–5. Kalman + Newton + IFT per axle ──────────────────────────────
    front_new, sigma_f = koopman_axle_step(
        koopman_state.front, Fx_front, kappa_front, params
    )
    rear_new, sigma_r = koopman_axle_step(
        koopman_state.rear, Fx_rear, kappa_rear, params
    )

    kstar_f = front_new.kappa_star
    kstar_r = rear_new.kappa_star

    # ── 6. Combined-slip correction (friction ellipse) ────────────────────
    # κ*_combined = κ*_pure · √(1 − (α_t / α_peak)²)
    alpha_f   = (alpha_t[0] + alpha_t[1]) * 0.5
    alpha_r   = (alpha_t[2] + alpha_t[3]) * 0.5
    ratio_f   = jnp.clip(alpha_f / (alpha_peak + 1e-6), 0.0, 0.99)
    ratio_r   = jnp.clip(alpha_r / (alpha_peak + 1e-6), 0.0, 0.99)
    kstar_f   = kstar_f * jnp.sqrt(1.0 - ratio_f ** 2)
    kstar_r   = kstar_r * jnp.sqrt(1.0 - ratio_r ** 2)

    # ── 7. Thermal μ derating on κ* ───────────────────────────────────────
    # Lower μ → lower grip limit → lower optimal slip
    # Scale: κ* ∝ √μ (Pacejka D ∝ μ, peak shifts with √(D/BCD))
    mu_f = (mu_thermal[0] + mu_thermal[1]) * 0.5
    mu_r = (mu_thermal[2] + mu_thermal[3]) * 0.5
    kstar_f = kstar_f * jnp.sqrt(jnp.clip(mu_f, 0.3, 2.0))
    kstar_r = kstar_r * jnp.sqrt(jnp.clip(mu_r, 0.3, 2.0))

    kstar_f = jnp.clip(kstar_f, params.kappa_min, params.kappa_max)
    kstar_r = jnp.clip(kstar_r, params.kappa_min, params.kappa_max)

    # Broadcast to per-wheel (front axle → wheels 0,1; rear → 2,3)
    kappa_star_fused = jnp.array([kstar_f, kstar_f, kstar_r, kstar_r])

    # Diagnostic: current slope dFx/dκ at operating point
    slope_f = jnp.dot(front_new.c,
                      dphi_dkappa(kappa_front, params.B0, params.C0))
    slope_r = jnp.dot(rear_new.c,
                      dphi_dkappa(kappa_rear,  params.B0, params.C0))

    output = KoopmanOutput(
        kappa_star_fused=kappa_star_fused,
        kappa_star_front=kstar_f,
        kappa_star_rear=kstar_r,
        sigma_front=sigma_f,
        sigma_rear=sigma_r,
        w_rls=jnp.ones(2),   # Koopman is the primary path — weight always 1
        slope_front=slope_f,
        slope_rear=slope_r,
    )
    new_state = KoopmanState(front=front_new, rear=rear_new)
    return output, new_state