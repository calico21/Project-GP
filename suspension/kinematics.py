# suspension/kinematics.py
# Project-GP — Exact Double A-Arm Kinematic Solver (IFD / Custom VJP)
# =============================================================================
#
# ARCHITECTURAL DECISIONS (see planning report critique):
#
# 1. IMPLICIT FUNCTION DIFFERENTIATION (not backprop through Newton)
#    The outer gradient ∂(camber,toe,...)/∂(delta_L_tr, psi_shim) is computed
#    via the Implicit Function Theorem:
#       dθ*/dp = −(∂F/∂θ)⁻¹ · (∂F/∂p)   at the converged point θ*
#    This is registered as a jax.custom_vjp. The Newton-Raphson iterations
#    run inside the forward pass WITHOUT gradient tracking. JAX never unrolls
#    the Newton graph — memory is O(1) in the number of iterations.
#
# 2. BUMP STEER IS NOT AN INDEPENDENT OPTIMIZER VARIABLE
#    With frozen hardpoints (CHAS_TiePnt fixed), the bump-steer curve is a
#    deterministic function of delta_L_tr (tie-rod length change). The MORL
#    optimizer sets toe_static; this module computes the forced bump-steer
#    consequence. The result is written back into VP, never into SuspensionSetup
#    as a free parameter.
#
# 3. ANTI-SQUAT SPLIT (4WD) — handled upstream in VP, not here
#    anti_squat_f and anti_squat_r are computed in compute_4wd_anti_geometry()
#    and injected into vehicle_params before vehicle model instantiation.
#    SuspensionSetup's scalar anti_squat is left at its nominal value and
#    DEPRECATED for Ter27.
#
# MATHEMATICAL REFERENCE:
#   Constraint system (3 equations, 3 unknowns: θ_LA, θ_UA, ψ):
#     F1 = ‖D(θ_UA) − C(θ_LA)‖² − L_upright²  = 0   [rigid upright]
#     F2 = W_z(θ_LA, θ_UA, ψ)   − (W_z0 + z)   = 0   [heave]
#     F3 = ‖T_U(θ_LA, θ_UA, ψ) − T_C‖² − L_tr² = 0   [tie-rod / locked steer]
#
#   where:
#     C(θ_LA)          = A1 + R(e_LA, θ_LA) @ (C0 − A1)       [lower ball joint]
#     D(θ_UA)          = B1 + R(e_UA, θ_UA) @ (D0 − B1)       [upper ball joint]
#     R_up(θ_LA,θ_UA,ψ) is the 3D rotation of the upright rigid body:
#       1. R_align maps e_kp_nom → e_kp = (D−C)/‖D−C‖  (aligns kingpin axis)
#       2. R_ψ is an additional rotation by ψ about e_kp  (sets camber/toe)
#       Combined: R_up = R_ψ(e_kp) @ R_align(e_kp_nom→e_kp)
#     W(θ_LA, θ_UA, ψ) = C + R_up @ (W0 − C0)            [wheel centre]
#     T_U(θ_LA, θ_UA, ψ) = C + R_up @ (TU0 − C0)         [tie-rod upright pt]
#     L_tr = ‖T_U_nom − T_C‖ + delta_L_tr                [tie-rod length]
#     psi_shim is added to ψ as a camber shim offset
#
#   Rodrigues rotation: R(e, α) @ v = v cos α + (e×v) sin α + e(e·v)(1−cos α)
#   This is C∞ in α, e for all ‖e‖ = 1.
# =============================================================================

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Dict, NamedTuple, Tuple, Any

import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# §0  Pure-JAX geometry primitives
# ---------------------------------------------------------------------------

def rodrigues(e: jax.Array, alpha: jax.Array, v: jax.Array) -> jax.Array:
    """
    Rodrigues rotation formula: rotate vector v by angle alpha about UNIT
    vector e.

    R(e, α) @ v = v·cos(α) + (e × v)·sin(α) + e·(e·v)·(1 − cos(α))

    Properties:
    · Exact for any α (no small-angle approximation).
    · C∞ in both α and e (when ‖e‖ = 1).
    · Gradient w.r.t. α: dR/dα @ v = −v·sin(α) + (e × v)·cos(α) + e·(e·v)·sin(α)

    Args:
        e    : (3,) unit vector — rotation axis
        alpha: scalar — rotation angle [rad]
        v    : (3,) vector to rotate

    Returns: (3,) rotated vector
    """
    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)
    return v * cos_a + jnp.cross(e, v) * sin_a + e * jnp.dot(e, v) * (1.0 - cos_a)


def rotation_align(a: jax.Array, b: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Return (axis, angle) of the rotation that maps unit vector a → unit vector b.
    The rotation axis is cross(a, b) / ‖cross(a, b)‖.

    Handles the degenerate case a = b (zero rotation) via softplus-like
    regularisation of ‖cross‖: the axis is arbitrary but the angle is zero,
    so the result is always the identity in the limit.

    Returns:
        axis  : (3,) unit rotation axis
        angle : scalar rotation angle [rad]
    """
    cross = jnp.cross(a, b)
    sin_angle = jnp.linalg.norm(cross)      # = sin(angle) for unit vectors
    cos_angle  = jnp.dot(a, b)              # = cos(angle)
    angle = jnp.arctan2(sin_angle, cos_angle)   # ∈ [0, π]

    # Safe normalisation: if ‖cross‖ ≈ 0, axis direction is irrelevant
    # because angle ≈ 0 → rotation is identity regardless of axis.
    axis = cross / (sin_angle + 1e-12)
    return axis, angle


def upright_rotation(
    C0: jax.Array, D0: jax.Array, W0: jax.Array,
    C:  jax.Array, D:  jax.Array, psi: jax.Array,
) -> jax.Array:
    """
    Compute R_upright: the rotation matrix of the upright rigid body.

    Step 1 — Align the kingpin axis e_kp_nom → e_kp:
        e_kp_nom = (D0 − C0) / ‖D0 − C0‖
        e_kp     = (D  − C ) / ‖D  − C ‖
        R_align  : R_align @ e_kp_nom = e_kp  (minimal-angle rotation)

    Step 2 — Apply camber-shim / constraint rotation ψ about the NEW e_kp:
        R_up = R_ψ(e_kp) ∘ R_align

    Args:
        C0, D0, W0 : nominal positions (lower ball joint, upper ball joint, wheel centre)
        C,  D      : deformed positions of ball joints
        psi        : rotation about new kingpin axis [rad]

    Returns:
        R_up as a Callable (jax.Array → jax.Array): v_body → v_deformed
        (implemented as the composed Rodrigues rotations, not as a 3×3 matrix,
        to keep the gradient graph sparse)

    Note: We return a FUNCTION, not a matrix. Callers should apply it as
    R_up_fn(v). This avoids materialising a 3×3 matrix in the JIT graph.
    """
    e_kp_nom = (D0 - C0) / jnp.linalg.norm(D0 - C0)
    e_kp     = (D  - C ) / (jnp.linalg.norm(D - C) + 1e-12)

    axis_align, angle_align = rotation_align(e_kp_nom, e_kp)

    def R_up(v: jax.Array) -> jax.Array:
        # Step 1: align kingpin
        v1 = rodrigues(axis_align, angle_align, v)
        # Step 2: rotate by psi about new kingpin
        v2 = rodrigues(e_kp, psi, v1)
        return v2

    return R_up


# ---------------------------------------------------------------------------
# §1  Constraint residual  F(θ) = 0
# ---------------------------------------------------------------------------

def _constraint_residual(
    theta: jax.Array,           # (3,): [theta_LA, theta_UA, psi]
    z: jax.Array,               # scalar: heave [m]
    delta_L_tr: jax.Array,      # scalar: tie-rod length adjustment [m]
    psi_shim: jax.Array,        # scalar: camber shim [rad] — added to psi
    # ── Frozen numpy constants converted to JAX (passed as closed-over) ──
    # These are JAX arrays pre-computed in __init__, not traced through IFD.
    A1: jax.Array, e_LA: jax.Array, C0_rel_A1: jax.Array,
    B1: jax.Array, e_UA: jax.Array, D0_rel_B1: jax.Array,
    C0: jax.Array, D0: jax.Array, W0: jax.Array, TU0: jax.Array, TC: jax.Array,
    L_upright_sq: jax.Array, L_tr_nom: jax.Array, W_z_nom: jax.Array,
) -> jax.Array:
    """
    Evaluates the 3-equation constraint system F(θ_LA, θ_UA, ψ) at the given
    heave z and tie-rod adjustment delta_L_tr.

    Returns: (3,) residual vector — zero at the kinematic equilibrium.
    """
    theta_LA = theta[0]
    theta_UA = theta[1]
    psi      = theta[2]

    # ── Ball joint positions ─────────────────────────────────────────────────
    C = A1 + rodrigues(e_LA, theta_LA, C0_rel_A1)    # lower ball joint
    D = B1 + rodrigues(e_UA, theta_UA, D0_rel_B1)    # upper ball joint

    # ── Upright rotation (combined alignment + psi + psi_shim) ──────────────
    R_up = upright_rotation(C0, D0, W0, C, D, psi + psi_shim)

    # ── Wheel centre and tie-rod upright point ───────────────────────────────
    W  = C + R_up(W0  - C0)
    TU = C + R_up(TU0 - C0)

    # ── Modified tie-rod length ──────────────────────────────────────────────
    L_tr_sq = (L_tr_nom + delta_L_tr) ** 2

    # ── Constraint equations ─────────────────────────────────────────────────
    # F1: rigid upright (ball-joint distance constant)
    CD_sq = jnp.dot(D - C, D - C)
    F1 = CD_sq - L_upright_sq

    # F2: heave (wheel-centre Z follows commanded z)
    F2 = W[2] - (W_z_nom + z)

    # F3: tie-rod (locked steering — tie-rod ball-joint distance constant)
    dT  = TU - TC
    F3  = jnp.dot(dT, dT) - L_tr_sq

    return jnp.array([F1, F2, F3])


# ---------------------------------------------------------------------------
# §2  IFD solver — Newton + custom_vjp
# ---------------------------------------------------------------------------

# We use nondiff_argnums to mark all the frozen geometric constants so that
# JAX's custom_vjp machinery does not try to trace or differentiate through them.
# Only (z, delta_L_tr, psi_shim) are differentiated w.r.t.
#
# Signature: _solve(z, delta_L_tr, psi_shim,  <nondiff geometry args...>)
#
# nondiff_argnums = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
# (indices of A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1, C0, D0, W0, TU0, TC,
#  L_upright_sq, L_tr_nom, W_z_nom, N_NEWTON)

_NONDIFF = tuple(range(3, 18))   # args 3..17 inclusive (18 total args, last is N_NEWTON)

@partial(jax.custom_vjp, nondiff_argnums=_NONDIFF)
def _solve_constraint_system(
    z: jax.Array,
    delta_L_tr: jax.Array,
    psi_shim: jax.Array,
    # ── nondiff args ────
    A1, e_LA, C0_rel_A1,
    B1, e_UA, D0_rel_B1,
    C0, D0, W0, TU0, TC,
    L_upright_sq, L_tr_nom, W_z_nom,
    N_NEWTON: int,
) -> jax.Array:
    """
    Solve F(θ; z, delta_L_tr, psi_shim) = 0 for θ = (θ_LA, θ_UA, ψ).

    Uses N_NEWTON Newton-Raphson steps inside a jax.lax.scan.
    The scan runs WITHOUT gradient tracking (gradients come from the
    custom_vjp backward below).

    Returns: (3,) converged solution [theta_LA, theta_UA, psi].
    """
    geo = (A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1,
           C0, D0, W0, TU0, TC, L_upright_sq, L_tr_nom, W_z_nom)

    def residual(th):
        return _constraint_residual(th, z, delta_L_tr, psi_shim, *geo)

    def nr_step(theta_carry, _):
        F = residual(theta_carry)
        # Jacobian of residual w.r.t. theta (3×3 system — cheap)
        J = jax.jacfwd(residual)(theta_carry)
        # Newton update: θ_new = θ − J⁻¹ F
        dtheta = jnp.linalg.solve(J, F)
        return theta_carry - dtheta, None

    # Initial guess: nominal (zero rotation, zero psi)
    theta0 = jnp.zeros(3)
    theta_opt, _ = jax.lax.scan(nr_step, theta0, None, length=N_NEWTON)
    return theta_opt


def _solve_fwd(
    z, delta_L_tr, psi_shim,
    A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1,
    C0, D0, W0, TU0, TC, L_upright_sq, L_tr_nom, W_z_nom,
    N_NEWTON,
):
    theta_opt = _solve_constraint_system(
        z, delta_L_tr, psi_shim,
        A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1,
        C0, D0, W0, TU0, TC, L_upright_sq, L_tr_nom, W_z_nom,
        N_NEWTON,
    )
    # Save for backward: the converged solution and the differentiated inputs
    residuals = (theta_opt, z, delta_L_tr, psi_shim)
    return theta_opt, residuals


def _solve_bwd(
    # nondiff args come first (positional), then res, then g
    A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1,
    C0, D0, W0, TU0, TC, L_upright_sq, L_tr_nom, W_z_nom,
    N_NEWTON,
    res,   # saved residuals from forward
    g,     # upstream gradient (3,)
):
    """
    Implicit Function Theorem backward pass.

    At the converged point θ* satisfying F(θ*; p) = 0, the total derivative is:
        (∂F/∂θ) dθ*/dp + (∂F/∂p) = 0
        ⟹  dθ*/dp = −(∂F/∂θ)⁻¹ (∂F/∂p)

    For the VJP:
        v = (∂F/∂θ)⁻ᵀ g          [adjoint equation — one 3×3 linear solve]
        ∂L/∂p = −(∂F/∂p)ᵀ v      [vector-Jacobian product]

    where p = (z, delta_L_tr, psi_shim) are the differentiable inputs.

    Memory cost: O(1) — no unrolled Newton graph.
    FLOPs: one 3×3 jacfwd of residual × 3 + one linalg.solve.
    """
    theta_opt, z, delta_L_tr, psi_shim = res

    geo = (A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1,
           C0, D0, W0, TU0, TC, L_upright_sq, L_tr_nom, W_z_nom)

    def residual(th, zz, dL, ps):
        return _constraint_residual(th, zz, dL, ps, *geo)

    # ── Jacobian of F w.r.t. theta (3×3) at the solution ────────────────────
    J_theta  = jax.jacfwd(lambda t: residual(t, z, delta_L_tr, psi_shim))(theta_opt)

    # ── Jacobian of F w.r.t. differentiated inputs ───────────────────────────
    # ∂F/∂z         (3,) — scalar input
    J_z      = jax.jacfwd(lambda zz: residual(theta_opt, zz, delta_L_tr, psi_shim))(z)
    # ∂F/∂delta_L_tr (3,) — scalar input
    J_dL     = jax.jacfwd(lambda dL: residual(theta_opt, z, dL, psi_shim))(delta_L_tr)
    # ∂F/∂psi_shim   (3,) — scalar input
    J_ps     = jax.jacfwd(lambda ps: residual(theta_opt, z, delta_L_tr, ps))(psi_shim)

    # ── Adjoint: solve J_theta^T v = g ───────────────────────────────────────
    v = jnp.linalg.solve(J_theta.T, g)

    # ── Gradients w.r.t. each differentiated input ───────────────────────────
    # dL/dp_i = -J_p_i^T v   (p_i scalar → J_p_i is (3,) → dot product)
    grad_z      = -jnp.dot(J_z,  v)
    grad_dL     = -jnp.dot(J_dL, v)
    grad_ps     = -jnp.dot(J_ps, v)

    return grad_z, grad_dL, grad_ps


_solve_constraint_system.defvjp(_solve_fwd, _solve_bwd)


# ---------------------------------------------------------------------------
# §3  Derived outputs from the solved theta
# ---------------------------------------------------------------------------

def _compute_outputs(
    theta: jax.Array,
    delta_L_tr: jax.Array,
    psi_shim: jax.Array,
    # geometry constants
    A1, e_LA, C0_rel_A1,
    B1, e_UA, D0_rel_B1,
    C0, D0, W0, TU0, TC,
    e_kp_nom: jax.Array,
) -> Dict[str, jax.Array]:
    """
    From converged theta = (θ_LA, θ_UA, ψ), compute all physical outputs.

    Returns dict with:
        'camber_rad'    : wheel camber angle [rad] (SAE: positive = top lean inboard)
        'toe_rad'       : wheel toe angle [rad] (positive = toe-in)
        'wheel_pos'     : (3,) wheel centre in body frame [m]
        'e_kp'          : (3,) kingpin unit vector in body frame
        'C'             : (3,) lower ball joint position [m]
        'D'             : (3,) upper ball joint position [m]
        'TU'            : (3,) tie-rod upright attachment [m]
        'e_spin_axis'   : (3,) wheel spin axis unit vector
    """
    theta_LA = theta[0]
    theta_UA = theta[1]
    psi      = theta[2]

    C = A1 + rodrigues(e_LA, theta_LA, C0_rel_A1)
    D = B1 + rodrigues(e_UA, theta_UA, D0_rel_B1)

    R_up = upright_rotation(C0, D0, W0, C, D, psi + psi_shim)

    W  = C + R_up(W0  - C0)
    TU = C + R_up(TU0 - C0)

    e_kp = (D - C) / (jnp.linalg.norm(D - C) + 1e-12)

    # Camber: angle of kingpin axis projected onto YZ plane from vertical
    # SAE convention: positive = top of kingpin leaning inboard (Y-axis)
    # camber = arctan2(e_kp[Y], e_kp[Z])  [in YZ plane, from Z-axis]
    camber_rad = jnp.arctan2(e_kp[1], e_kp[2])

    # Toe: angular position of wheel spin axis in XY plane
    # The wheel spin axis is perpendicular to the kingpin axis and lies in the
    # wheel plane. After applying R_up, the nominal spin axis (initially Y-axis)
    # rotates with the upright.
    e_spin_nom = jnp.array([0.0, 1.0, 0.0])   # nominal spin axis = Y
    e_spin = R_up(e_spin_nom)

    # Toe angle: angle of e_spin projected onto XY plane from Y-axis
    # Positive = toe-in (front of wheel rotated inward, i.e. toward vehicle centreline)
    # For the left wheel: toe-in means the front of the wheel is closer to CL
    # → e_spin has a small negative X component → toe = arctan2(-e_spin[X], e_spin[Y])
    toe_rad = jnp.arctan2(-e_spin[0], e_spin[1])

    return {
        "camber_rad":  camber_rad,
        "toe_rad":     toe_rad,
        "wheel_pos":   W,
        "e_kp":        e_kp,
        "C":           C,
        "D":           D,
        "TU":          TU,
        "e_spin_axis": e_spin,
    }


# ---------------------------------------------------------------------------
# §4  Motion ratio — pushrod/pullrod-rocker-coilover
# ---------------------------------------------------------------------------

def _motion_ratio_at_theta_LA(
    theta_LA: jax.Array,
    A1: jax.Array, e_LA: jax.Array, C0_rel_A1: jax.Array,
    # pushrod attachment on upright side (moves with lower A-arm in pushrod config)
    PP0_rel_A1: jax.Array,      # NSMA_PPAttPnt − A1 in local A-arm frame
    CHAS_PP: jax.Array,         # CHAS_AttPnt_L (chassis end of pushrod — fixed)
    # Rocker geometry (all fixed in chassis frame)
    ROC_AXI: jax.Array,         # CHAS_RocAxi_L
    ROC_PIV: jax.Array,         # CHAS_RocPiv_L
    ROC_ROD: jax.Array,         # ROCK_RodPnt_L (nominal)
    ROC_COI: jax.Array,         # ROCK_CoiPnt_L (nominal)
    L_rod: jax.Array,           # pushrod length (constant)
    L_coi_nom: jax.Array,       # nominal coilover length (between ROC_COI and chassis coil point)
    CHAS_COI: jax.Array,        # chassis attachment of coilover (fixed)
    actuation_sign: float,      # +1 for pushrod (lower arm), -1 for pullrod (upper arm)
) -> jax.Array:
    """
    Compute the instantaneous motion ratio dL_spring / dz at angle theta_LA.

    MR(z) = dL_spring / dz  [dimensionless, ideally in (0.8, 1.2) for FS]

    The rocker pivot axis is e_roc = (ROC_AXI − ROC_PIV) / ‖...‖.
    The rocker rotates by angle φ about this axis through ROC_PIV.
    φ is found from the pushrod length constraint:
        ‖ROC_ROD(φ) − PP(θ_LA)‖ = L_rod

    Then the coilover travel is:
        L_coi(φ) = ‖ROC_COI(φ) − CHAS_COI‖
        MR = dL_coi/dz ≈ (dL_coi/dφ) / (dz/dφ)  [via chain rule]

    NOTE: This function returns MR at a given θ_LA; the motion ratio at a given
    heave z is obtained by first solving the A-arm constraint to get θ_LA(z),
    then calling this function.

    For the 3-level polynomial fit used in vehicle_dynamics, call this function
    at many z values via SuspensionKinematics.compute_mr_polynomial().
    """
    # ── Pushrod lower attachment point (moves with lower A-arm) ──────────────
    PP = A1 + rodrigues(e_LA, theta_LA, PP0_rel_A1)

    # ── Rocker axis ───────────────────────────────────────────────────────────
    e_roc = (ROC_AXI - ROC_PIV) / (jnp.linalg.norm(ROC_AXI - ROC_PIV) + 1e-12)

    # ── Solve rocker angle φ from pushrod-length constraint ───────────────────
    ROD0_rel_PIV = ROC_ROD - ROC_PIV   # nominal rod attachment in local rocker frame

    def rod_constraint(phi):
        ROD = ROC_PIV + rodrigues(e_roc, phi, ROD0_rel_PIV)
        return jnp.dot(ROD - PP, ROD - PP) - L_rod ** 2

    # Newton-Raphson for φ (scalar, 1D)
    def nr_phi(phi_carry, _):
        F  = rod_constraint(phi_carry)
        dF = jax.grad(rod_constraint)(phi_carry)
        return phi_carry - F / (dF + 1e-12), None

    phi_opt, _ = jax.lax.scan(nr_phi, jnp.array(0.0), None, length=8)

    # ── Coilover length at phi_opt ────────────────────────────────────────────
    COI0_rel_PIV = ROC_COI - ROC_PIV
    COI = ROC_PIV + rodrigues(e_roc, phi_opt, COI0_rel_PIV)
    L_coi = jnp.linalg.norm(COI - CHAS_COI)

    # MR = dL_coi / dθ_LA
    # Computed as the ratio of Jacobians via jax.grad (forward mode available)
    dL_coi_dphi  = jax.grad(
        lambda phi: jnp.linalg.norm(
            ROC_PIV + rodrigues(e_roc, phi, COI0_rel_PIV) - CHAS_COI
        )
    )(phi_opt)

    # dφ/dθ_LA from the rod constraint: ∂C/∂θ_LA
    dF_dphi    = jax.grad(rod_constraint)(phi_opt)
    dF_dtheta  = jax.grad(
        lambda th: rod_constraint(phi_opt) - (
            _rod_constraint_at_theta(th, phi_opt, A1, e_LA, PP0_rel_A1,
                                     ROC_PIV, e_roc, ROD0_rel_PIV, L_rod)
            - rod_constraint(phi_opt)
        )
    )(theta_LA)

    # Implicit: dφ/dθ_LA = −(∂F/∂φ)⁻¹ (∂F/∂θ_LA)
    dphi_dtheta = -jax.grad(
        lambda th: _rod_constraint_at_theta(th, phi_opt, A1, e_LA, PP0_rel_A1,
                                             ROC_PIV, e_roc, ROD0_rel_PIV, L_rod)
    )(theta_LA) / (dF_dphi + 1e-12)

    # MR = dL_coi/dθ_LA × sign (actuation_sign flips for pullrod)
    MR = dL_coi_dphi * dphi_dtheta * actuation_sign
    return MR


def _rod_constraint_at_theta(theta_LA, phi, A1, e_LA, PP0_rel_A1,
                               ROC_PIV, e_roc, ROD0_rel_PIV, L_rod):
    """Auxiliary: rod constraint as a function of theta_LA at fixed phi."""
    PP  = A1 + rodrigues(e_LA, theta_LA, PP0_rel_A1)
    ROD = ROC_PIV + rodrigues(e_roc, phi, ROD0_rel_PIV)
    return jnp.dot(ROD - PP, ROD - PP) - L_rod ** 2


# ---------------------------------------------------------------------------
# §5  Roll centre — force-line intersection method
# ---------------------------------------------------------------------------

def _roll_centre_height(
    C: jax.Array, D: jax.Array,
    A1: jax.Array, A2: jax.Array,
    B1: jax.Array, B2: jax.Array,
    half_track: jax.Array,
) -> jax.Array:
    """
    Compute the instantaneous roll centre height using the force-based method.

    The lateral force on each wheel acts along a line from the contact patch
    to the Virtual Pivot Point (VPP). The VPP is the intersection of the two
    A-arm planes extended into the YZ transverse plane.

    Steps:
    1. Lower A-arm plane normal: n_L = (A2−A1) × (C−A1)  (normalised)
    2. Upper A-arm plane normal: n_U = (B2−B1) × (D−B1)  (normalised)
    3. The A-arm EXTENDS as a plane through its pivot axis and the ball joint.
       In the YZ plane (X=0), the A-arm plane intersects at a line.
       The VPP is where the two intersection lines cross in the YZ plane.
    4. Draw a line from contact patch (Y=±half_track, Z=0) through VPP.
    5. Roll centre = where this line crosses Y=0 (vehicle centreline).

    This is the exact (non-approximated) force-based roll centre method.
    Valid for all suspension travel ranges within the geometric constraints.

    Returns: scalar Z-coordinate of the roll centre [m].
    """
    # ── Lower A-arm plane ────────────────────────────────────────────────────
    nL = jnp.cross(A2 - A1, C - A1)
    nL = nL / (jnp.linalg.norm(nL) + 1e-12)

    # ── Upper A-arm plane ────────────────────────────────────────────────────
    nU = jnp.cross(B2 - B1, D - B1)
    nU = nU / (jnp.linalg.norm(nU) + 1e-12)

    # ── VPP: intersection of the two planes in the YZ transverse section ─────
    # In the YZ plane (X=0), each A-arm plane intersects as a line.
    # A point P on the lower plane satisfies: nL · (P − A1) = 0, with P_x = 0.
    # The line direction in YZ: d_L = e_X × nL (perpendicular to both nL and X)
    e_X = jnp.array([1.0, 0.0, 0.0])
    dL  = jnp.cross(e_X, nL)    # direction of lower arm projection in YZ
    dU  = jnp.cross(e_X, nU)    # direction of upper arm projection in YZ

    # Anchor points (set X=0 in each plane): solve nL · (P − A1) = 0 at P_x=0
    # P = A1 − (A1_x / nL_x) * nL  (if nL_x ≠ 0), or project A1 onto X=0 along nL
    def plane_anchor(n, anchor):
        """Find the point in the YZ plane (X=0) closest to anchor within the plane."""
        # Project anchor onto plane then project result onto X=0:
        # p_plane = anchor − (n · anchor) * n  is on the plane but not at X=0.
        # Move along n direction until X=0: anchor + t*(-e_X + ...) is complex.
        # Simpler: parametrize the intersection line of the plane with X=0 directly.
        # The plane n · (P − anchor) = 0 at X=0 is: n[1]*y + n[2]*z = n · anchor
        # This is a line in the YZ plane. Return its intersection with Z=0:
        # n[1]*y = n · anchor  → y = (n · anchor) / n[1]  (if n[1] ≠ 0)
        # Or with Y=0: z = (n · anchor) / n[2]
        # For robustness, return the point on this intersection line at the
        # parametric position closest to anchor (projected onto YZ plane).
        anchor_yz = anchor.at[0].set(0.0)
        # Parametric line: p(t) = anchor_yz + t * d  where d = cross(e_X, n)
        d = jnp.cross(e_X, n)
        d = d / (jnp.linalg.norm(d) + 1e-12)
        # Closest point on line to anchor_yz is anchor_yz itself (anchor_yz IS
        # a point satisfying n·(p−anchor_yz)=0 approximately, since anchor_x=0)
        # We need a reference point on the line. Use the plane intersection with X=Z=0:
        # n[1]*y = n·anchor → y0 = (n·anchor) / n[1]
        n_dot_anchor = jnp.dot(n, anchor)
        y0 = jnp.where(jnp.abs(n[1]) > 1e-10,
                        n_dot_anchor / (n[1] + 1e-12),
                        0.0)
        z0 = jnp.where(jnp.abs(n[2]) > 1e-10,
                        (n_dot_anchor - n[1] * y0) / (n[2] + 1e-12),
                        0.0)
        return jnp.array([0.0, y0, z0])

    P_L = plane_anchor(nL, A1)    # reference point on lower arm projection
    P_U = plane_anchor(nU, B1)    # reference point on upper arm projection

    # ── Intersection of the two lines in YZ plane ─────────────────────────────
    # Line 1: P_L + s * dL
    # Line 2: P_U + t * dU
    # Solve for s: P_L + s*dL = P_U + t*dU  → [dL, -dU] [s, t]^T = P_U - P_L
    # Using Y and Z components:
    A_mat = jnp.array([[dL[1], -dU[1]],
                        [dL[2], -dU[2]]])
    b_vec = jnp.array([P_U[1] - P_L[1],
                        P_U[2] - P_L[2]])

    # Solve 2×2 system (det may be near zero for near-parallel A-arms)
    det = A_mat[0, 0] * A_mat[1, 1] - A_mat[0, 1] * A_mat[1, 0]
    s   = (b_vec[0] * A_mat[1, 1] - b_vec[1] * A_mat[0, 1]) / (det + 1e-10)
    VPP = P_L + s * dL   # (3,) — Virtual Pivot Point

    # ── Roll centre: intersection of force line with centreline ───────────────
    # Contact patch: (0, half_track, 0)  [left wheel]
    cp = jnp.array([0.0, half_track, 0.0])

    # Force line direction: cp → VPP
    d_force = VPP - cp
    d_force = d_force / (jnp.linalg.norm(d_force) + 1e-12)

    # Intersect with Y=0: cp + t * d_force, set Y-component = 0
    # cp[1] + t * d_force[1] = 0  →  t = -cp[1] / d_force[1]
    t_rc   = -cp[1] / (d_force[1] + 1e-10)
    RC_pt  = cp + t_rc * d_force
    return RC_pt[2]   # Z-coordinate = roll centre height


# ---------------------------------------------------------------------------
# §6  SuspensionKinematics class
# ---------------------------------------------------------------------------

class KinematicOutputs(NamedTuple):
    """All kinematic outputs at a single heave position."""
    camber_rad:    jax.Array    # wheel camber [rad]
    toe_rad:       jax.Array    # static toe [rad] (positive = toe-in)
    wheel_pos:     jax.Array    # wheel centre (3,) [m]
    motion_ratio:  jax.Array    # spring/damper MR [-]
    roll_centre_z: jax.Array    # roll centre height [m]
    C:             jax.Array    # lower ball joint (3,) [m]
    D:             jax.Array    # upper ball joint (3,) [m]
    e_kp:          jax.Array    # kingpin unit vector (3,)


class KinematicGains(NamedTuple):
    """Kinematic coefficients at z=0, used to populate VP."""
    camber_gain_rad_per_m:   jax.Array    # dγ/dz at z=0  [rad/m]
    bump_steer_lin_rad_per_m: jax.Array   # dδ/dz at z=0  [rad/m]
    bump_steer_quad_rad_per_m2: jax.Array # d²δ/dz² /2    [rad/m²]
    mr_poly:                 jax.Array    # (3,): [a0, a1, a2] quadratic MR fit
    rc_height_m:             jax.Array    # roll centre Z at z=0  [m]
    drc_dz_m_per_m:          jax.Array    # dRC/dz at z=0
    kpi_rad:                 float        # kingpin inclination (static, scalar)
    caster_rad:              float        # caster angle (static, scalar)
    mech_trail_m:            float        # mechanical trail at z=0 [m]


class SuspensionKinematics:
    """
    Full Double A-Arm kinematic solver built on frozen Optimum K hardpoints.

    Free JAX parameters (differentiable via IFD):
        delta_L_tr  [m]   : tie-rod length adjustment → sets static toe
        psi_shim    [rad] : upright rotation about kingpin → sets static camber

    All hardpoint coordinates are stored as numpy constants and never traced
    by JAX. This keeps the IFD Jacobian at a fixed 3×3 size regardless of
    the number of optimizer parameters.

    Usage:
        front_kin = SuspensionKinematics(front_hpts, side='left')
        gains = front_kin.kinematic_gains(
            delta_L_tr=jnp.array(0.0),
            psi_shim=jnp.array(0.0)
        )
        # gains.bump_steer_lin_rad_per_m, gains.camber_gain_rad_per_m, etc.

        # Sweep over heave (differentiable w.r.t. delta_L_tr, psi_shim):
        z_arr = jnp.linspace(-0.08, 0.15, 100)
        outputs = front_kin.sweep(z_arr, delta_L_tr, psi_shim)
    """

    N_NEWTON: int = 8   # Newton-Raphson iterations (converges in < 5 for FS geometry)

    def __init__(self, hpts: Dict[str, Any], side: str = "left"):
        """
        Args:
            hpts : dict from hardpoints.load_front/rear_hardpoints()
            side : 'left' or 'right'  (kinematics are mirrored for right side)
        """
        assert side in ("left", "right"), f"side must be 'left' or 'right', got '{side}'"

        sign = 1.0 if side == "left" else -1.0   # mirror Y for right side
        def pt(key: str) -> np.ndarray:
            """Return hardpoint, mirrored if right side."""
            p = hpts[key].copy()
            p[1] *= sign
            return p

        self.side     = side
        self.hpts     = hpts
        self.act_sign = 1.0 if hpts.get("actuation_type", "pushrod") == "pushrod" else -1.0

        # ── A-arm chassis pivots ──────────────────────────────────────────────
        A1_np = pt("CHAS_LowFor");  A2_np = pt("CHAS_LowAft")
        B1_np = pt("CHAS_UppFor");  B2_np = pt("CHAS_UppAft")
        C0_np = pt("UPRI_LowPnt")
        D0_np = pt("UPRI_UppPnt")

        # ── Wheel centre (nominal) ────────────────────────────────────────────
        R_w   = float(hpts.get("R_wheel", 0.2032))
        HT    = float(hpts.get("Half Track_m", hpts.get("Half Track", 615) / 1000.0))
        W0_np = np.array([0.0, sign * HT, R_w])

        # ── Tie-rod ───────────────────────────────────────────────────────────
        TC_np  = pt("CHAS_TiePnt")
        TU0_np = pt("UPRI_TiePnt")

        # ── Pushrod/pullrod ───────────────────────────────────────────────────
        PP0_np     = pt("NSMA_PPAttPnt_L")
        CHAS_PP_np = pt("CHAS_AttPnt_L")

        # ── Rocker ───────────────────────────────────────────────────────────
        ROC_AXI_np = pt("CHAS_RocAxi_L")
        ROC_PIV_np = pt("CHAS_RocPiv_L")
        ROC_ROD_np = pt("ROCK_RodPnt_L")
        ROC_COI_np = pt("ROCK_CoiPnt_L")
        # Assume coilover chassis attachment is at the same Z as CHAS_AttPnt_L
        # but at the rocker coil side — approximated here as CHAS_AttPnt_L for
        # the body-side anchor. Adjust if a separate CHAS_COI hardpoint exists.
        CHAS_COI_np = CHAS_PP_np.copy()

        # ── Precomputed static constants ──────────────────────────────────────
        e_LA_np = A2_np - A1_np;  e_LA_np /= (np.linalg.norm(e_LA_np) + 1e-12)
        e_UA_np = B2_np - B1_np;  e_UA_np /= (np.linalg.norm(e_UA_np) + 1e-12)

        C0_rel_A1_np = C0_np - A1_np
        D0_rel_B1_np = D0_np - B1_np

        L_upright_sq_np = np.dot(D0_np - C0_np, D0_np - C0_np)
        L_tr_nom_np     = np.linalg.norm(TU0_np - TC_np)
        W_z_nom_np      = W0_np[2]

        # Pushrod length
        L_rod_np        = np.linalg.norm(PP0_np - CHAS_PP_np)

        # Coilover nominal length
        L_coi_nom_np    = np.linalg.norm(ROC_COI_np - CHAS_COI_np)

        # PP attachment relative to A1 (stays in local A-arm frame as it rotates)
        # NOTE: for pushrod, PP attaches to the lower A-arm at NSMA_PPAttPnt,
        # which is on the upright side. In the kinematic model, this point follows
        # the lower A-arm rotation (it's rigidly connected to the lower A-arm body).
        PP0_rel_A1_np = PP0_np - A1_np

        # Nominal kingpin unit vector
        e_kp_nom_np = D0_np - C0_np
        e_kp_nom_np /= (np.linalg.norm(e_kp_nom_np) + 1e-12)

        # Static KPI and Caster from nominal kingpin vector
        self.kpi_rad    = float(np.arctan2(e_kp_nom_np[1], e_kp_nom_np[2]))
        self.caster_rad = float(np.arctan2(-e_kp_nom_np[0], e_kp_nom_np[2]))
        self.mech_trail_m = float(R_w * np.tan(abs(self.caster_rad)))
        self.R_wheel    = R_w
        self.half_track = HT

        # ── Convert all to JAX arrays (stored, not re-traced per call) ────────
        to_jax = lambda x: jnp.array(x, dtype=jnp.float32)
        self._jax = dict(
            A1=to_jax(A1_np), e_LA=to_jax(e_LA_np), C0_rel_A1=to_jax(C0_rel_A1_np),
            B1=to_jax(B1_np), e_UA=to_jax(e_UA_np), D0_rel_B1=to_jax(D0_rel_B1_np),
            C0=to_jax(C0_np), D0=to_jax(D0_np), W0=to_jax(W0_np),
            TU0=to_jax(TU0_np), TC=to_jax(TC_np),
            L_upright_sq=to_jax(float(L_upright_sq_np)),
            L_tr_nom=to_jax(float(L_tr_nom_np)),
            W_z_nom=to_jax(float(W_z_nom_np)),
            e_kp_nom=to_jax(e_kp_nom_np),
            # MR geometry
            PP0_rel_A1=to_jax(PP0_rel_A1_np),
            CHAS_PP=to_jax(CHAS_PP_np),
            ROC_AXI=to_jax(ROC_AXI_np), ROC_PIV=to_jax(ROC_PIV_np),
            ROC_ROD=to_jax(ROC_ROD_np), ROC_COI=to_jax(ROC_COI_np),
            CHAS_COI=to_jax(CHAS_COI_np),
            L_rod=to_jax(float(L_rod_np)),
            L_coi_nom=to_jax(float(L_coi_nom_np)),
            # RC geometry
            A1_rc=to_jax(A1_np), A2_rc=to_jax(A2_np),
            B1_rc=to_jax(B1_np), B2_rc=to_jax(B2_np),
            half_track=to_jax(HT),
        )
        # Tuple of nondiff args in order expected by _solve_constraint_system
        self._nondiff_geo = (
            self._jax["A1"], self._jax["e_LA"], self._jax["C0_rel_A1"],
            self._jax["B1"], self._jax["e_UA"], self._jax["D0_rel_B1"],
            self._jax["C0"], self._jax["D0"], self._jax["W0"],
            self._jax["TU0"], self._jax["TC"],
            self._jax["L_upright_sq"], self._jax["L_tr_nom"], self._jax["W_z_nom"],
            self.N_NEWTON,
        )

        print(f"[SuspensionKinematics/{side}] Initialised. "
              f"KPI={np.degrees(self.kpi_rad):.2f}°, "
              f"Caster={np.degrees(self.caster_rad):.2f}°, "
              f"trail={self.mech_trail_m*1e3:.1f} mm, "
              f"L_upright={np.sqrt(float(L_upright_sq_np))*1e3:.1f} mm, "
              f"L_tr={float(L_tr_nom_np)*1e3:.1f} mm")

    # ── Public interface ──────────────────────────────────────────────────────

    def solve_at_heave(
        self,
        z:          jax.Array,     # scalar heave [m]
        delta_L_tr: jax.Array,     # scalar tie-rod length change [m]
        psi_shim:   jax.Array,     # scalar camber shim [rad]
    ) -> KinematicOutputs:
        """
        Solve the full kinematic system at one heave position.
        Differentiable w.r.t. (z, delta_L_tr, psi_shim) via IFD.
        """
        j = self._jax
        theta = _solve_constraint_system(
            z, delta_L_tr, psi_shim, *self._nondiff_geo
        )
        out = _compute_outputs(
            theta, delta_L_tr, psi_shim,
            j["A1"], j["e_LA"], j["C0_rel_A1"],
            j["B1"], j["e_UA"], j["D0_rel_B1"],
            j["C0"], j["D0"], j["W0"], j["TU0"], j["TC"],
            j["e_kp_nom"],
        )
        # Motion ratio
        mr = _motion_ratio_at_theta_LA(
            theta[0],
            j["A1"], j["e_LA"], j["C0_rel_A1"],
            j["PP0_rel_A1"], j["CHAS_PP"],
            j["ROC_AXI"], j["ROC_PIV"], j["ROC_ROD"], j["ROC_COI"],
            j["L_rod"], j["L_coi_nom"], j["CHAS_COI"],
            self.act_sign,
        )
        # Roll centre
        rc_z = _roll_centre_height(
            out["C"], out["D"],
            j["A1_rc"], j["A2_rc"],
            j["B1_rc"], j["B2_rc"],
            j["half_track"],
        )
        return KinematicOutputs(
            camber_rad=out["camber_rad"],
            toe_rad=out["toe_rad"],
            wheel_pos=out["wheel_pos"],
            motion_ratio=mr,
            roll_centre_z=rc_z,
            C=out["C"], D=out["D"], e_kp=out["e_kp"],
        )

    def sweep(
        self,
        z_array:    jax.Array,    # (N,) heave positions [m]
        delta_L_tr: jax.Array,    # scalar
        psi_shim:   jax.Array,    # scalar
    ) -> KinematicOutputs:
        """
        Vectorised kinematic sweep over all heave positions.
        Returns a KinematicOutputs where each field has shape (N, ...).
        """
        return jax.vmap(
            lambda z: self.solve_at_heave(z, delta_L_tr, psi_shim)
        )(z_array)

    def kinematic_gains(
        self,
        delta_L_tr: jax.Array,
        psi_shim:   jax.Array,
        dz:         float = 1e-4,     # finite difference step for gains [m]
        n_mr_pts:   int   = 50,
    ) -> KinematicGains:
        """
        Compute all gains at z=0 by differentiation of solve_at_heave.

        NOTE: camber_gain and bump_steer are computed here via jax.grad
        (which itself uses IFD for the inner Newton), NOT finite differences.
        dz is kept as a fallback parameter only.

        The motion ratio polynomial is fit to MR(z) over the full travel range
        using n_mr_pts points.
        """
        z0 = jnp.array(0.0)

        # ── Gains via jax.grad of the solve (leverages IFD custom_vjp) ───────
        camber_gain = jax.grad(
            lambda z: self.solve_at_heave(z, delta_L_tr, psi_shim).camber_rad
        )(z0)

        bs_fn       = lambda z: self.solve_at_heave(z, delta_L_tr, psi_shim).toe_rad
        bs_lin      = jax.grad(bs_fn)(z0)
        bs_quad     = jax.grad(jax.grad(bs_fn))(z0) * 0.5   # half-coefficient

        # ── Roll centre and its gain ───────────────────────────────────────────
        rc_fn = lambda z: self.solve_at_heave(z, delta_L_tr, psi_shim).roll_centre_z
        rc0   = rc_fn(z0)
        drc_dz = jax.grad(rc_fn)(z0)

        # ── Motion ratio polynomial fit ───────────────────────────────────────
        z_arr  = jnp.linspace(-0.08, 0.15, n_mr_pts)
        outputs = self.sweep(z_arr, delta_L_tr, psi_shim)
        mr_arr  = outputs.motion_ratio    # (n_mr_pts,)

        # Fit quadratic: MR(z) ≈ a0 + a1*z + a2*z²
        # Using Vandermonde matrix (differentiable via jnp.linalg.lstsq)
        Vander = jnp.column_stack([
            jnp.ones(n_mr_pts),
            z_arr,
            z_arr ** 2,
        ])
        mr_poly, _, _, _ = jnp.linalg.lstsq(Vander, mr_arr)

        # ── Static outputs ────────────────────────────────────────────────────
        static = self.solve_at_heave(z0, delta_L_tr, psi_shim)

        return KinematicGains(
            camber_gain_rad_per_m=camber_gain,
            bump_steer_lin_rad_per_m=bs_lin,
            bump_steer_quad_rad_per_m2=bs_quad,
            mr_poly=mr_poly,
            rc_height_m=rc0,
            drc_dz_m_per_m=drc_dz,
            kpi_rad=self.kpi_rad,
            caster_rad=self.caster_rad,
            mech_trail_m=self.mech_trail_m,
        )

    def delta_L_tr_from_toe(self, toe_target_rad: float) -> float:
        """
        Invert the toe-vs-tie-rod relationship to find the tie-rod length
        adjustment that achieves a target static toe.

        Solves: toe(delta_L_tr) = toe_target_rad  (scalar Newton, Python-level).

        Returns: delta_L_tr [m] (positive = longer tie rod → toe-out for left wheel).
        """
        def toe_at(dL: float) -> float:
            out = self.solve_at_heave(
                jnp.array(0.0),
                jnp.array(dL),
                jnp.array(0.0)
            )
            return float(out.toe_rad)

        # Newton-Raphson on the scalar Python level (not in JAX graph)
        dL = 0.0
        for _ in range(20):
            f  = toe_at(dL) - toe_target_rad
            df = float(jax.grad(
                lambda d: self.solve_at_heave(
                    jnp.array(0.0), d, jnp.array(0.0)
                ).toe_rad
            )(jnp.array(dL)))
            if abs(df) < 1e-12:
                break
            dL -= f / df
            if abs(f) < 1e-8:
                break
        return dL

    def psi_shim_from_camber(self, camber_target_rad: float) -> float:
        """
        Find the shim rotation psi_shim that achieves a target static camber.
        At z=0, camber ≈ kpi + psi_shim (for small psi_shim), so the initial
        guess is psi_shim = camber_target − camber_0.  Newton converges in 1–3 steps.
        """
        def camber_at(ps: float) -> float:
            out = self.solve_at_heave(
                jnp.array(0.0),
                jnp.array(0.0),
                jnp.array(ps)
            )
            return float(out.camber_rad)

        # Initial residual
        camber_0 = camber_at(0.0)
        ps = camber_target_rad - camber_0   # good first guess

        for _ in range(10):
            f  = camber_at(ps) - camber_target_rad
            df = float(jax.grad(
                lambda p: self.solve_at_heave(
                    jnp.array(0.0), jnp.array(0.0), p
                ).camber_rad
            )(jnp.array(ps)))
            if abs(df) < 1e-12:
                break
            ps -= f / df
            if abs(f) < 1e-8:
                break
        return ps


# ---------------------------------------------------------------------------
# §7  4WD Anti-Geometry (not in SuspensionSetup — injected into VP)
# ---------------------------------------------------------------------------

def compute_4wd_anti_geometry(
    front_kin: SuspensionKinematics,
    rear_kin:  SuspensionKinematics,
    VP: Dict[str, Any],
) -> Dict[str, float]:
    """
    Compute front and rear anti-squat / anti-dive / anti-lift fractions
    from the kinematic instant centres and inject them into vehicle_params.

    For a 4WD car with torque vectoring, anti-squat is split between axles:
        AS_f = h_ic_long_f / (h_cg × lf / L)   [front axle contribution]
        AS_r = h_ic_long_r / (h_cg × lr / L)   [rear axle contribution]

    These replace the single scalar 'anti_squat' in SuspensionSetup for the
    Ter27. The scalar anti_squat entry in SuspensionSetup is DEPRECATED and
    emits a warning when the Ter27 VP is loaded.

    The instant centre height (side view) for each axle is computed from the
    side-view projections of the upper and lower A-arms:
        IC_long_z = (h_ua × l_la − h_la × l_ua) / (l_la − l_ua)
    where h, l are the Z and X coordinates of each A-arm pivot in the XZ plane.

    Returns a dict of new VP entries to merge into vehicle_params:
        'anti_squat_f', 'anti_squat_r'
        'anti_dive_f',  'anti_dive_r'
        'anti_lift_r'   (rear anti-lift under braking, 4WD specific)
    """
    h_cg = VP.get("h_cg", 0.285)
    lf   = VP.get("lf",   0.8525)
    lr   = VP.get("lr",   0.6975)
    L    = lf + lr
    brake_bias_f = VP.get("brake_bias_f", 0.60)

    def _instant_centre_z(kin: SuspensionKinematics) -> Tuple[float, float]:
        """
        Side-view instant centre (X, Z) from the lower and upper A-arm lines.

        Each A-arm line in XZ is defined by its two chassis pivot points projected
        onto the XZ plane. The side-view IC is the intersection of these two lines.
        """
        j = kin._jax
        A1 = np.array(j["A1"]); A2 = np.array(j["A1"]) + np.array(j["e_LA"]) * 0.3
        # Use actual CHAS_LowAft/For for direction
        hpts = kin.hpts
        A1_np = hpts["CHAS_LowFor"].copy(); A1_np[1] = 0.0
        A2_np = hpts["CHAS_LowAft"].copy(); A2_np[1] = 0.0
        B1_np = hpts["CHAS_UppFor"].copy(); B1_np[1] = 0.0
        B2_np = hpts["CHAS_UppAft"].copy(); B2_np[1] = 0.0

        # Lower A-arm direction in XZ (Y=0 projection)
        dLA_x = A2_np[0] - A1_np[0]; dLA_z = A2_np[2] - A1_np[2]
        # Upper A-arm direction in XZ
        dUA_x = B2_np[0] - B1_np[0]; dUA_z = B2_np[2] - B1_np[2]

        # Intersection: A1 + s*dLA = B1 + t*dUA in XZ
        # [dLA_x, -dUA_x] [s]   [B1_x - A1_x]
        # [dLA_z, -dUA_z] [t] = [B1_z - A1_z]
        M = np.array([[dLA_x, -dUA_x], [dLA_z, -dUA_z]])
        b = np.array([B1_np[0] - A1_np[0], B1_np[2] - A1_np[2]])
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) < 1e-8:
            # Parallel A-arms (unlikely for FS): IC at infinity → anti = 0
            return 0.0, 0.0
        s = (b[0] * M[1, 1] - b[1] * M[0, 1]) / det
        ic_x = A1_np[0] + s * dLA_x
        ic_z = A1_np[2] + s * dLA_z
        return float(ic_x), float(ic_z)

    # ── Front anti-geometry ────────────────────────────────────────────────────
    ic_f_x, ic_f_z = _instant_centre_z(front_kin)
    # Anti-dive (front braking): ratio of IC height to the pitch height at brake force
    # Under deceleration with brake_bias_f at front:
    # Anti-dive_f = ic_f_z / (h_cg × brake_bias_f × L / lf)
    anti_dive_f = ic_f_z / (h_cg * brake_bias_f * L / (lf + 1e-6) + 1e-6)
    # For 4WD front drive: front anti-squat
    anti_squat_f = ic_f_z / (h_cg * lf / (L + 1e-6) + 1e-6)

    # ── Rear anti-geometry ────────────────────────────────────────────────────
    ic_r_x, ic_r_z = _instant_centre_z(rear_kin)
    # Anti-squat (rear drive): dominant term for 4WD
    anti_squat_r = ic_r_z / (h_cg * lr / (L + 1e-6) + 1e-6)
    # Anti-dive (rear braking)
    anti_dive_r  = ic_r_z / (h_cg * (1.0 - brake_bias_f) * L / (lr + 1e-6) + 1e-6)
    # Anti-lift (4WD: rear driven under braking opposite to brake geometry)
    anti_lift_r  = anti_squat_r   # same IC, different force interpretation

    result = {
        "anti_squat_f": float(np.clip(anti_squat_f, 0.0, 1.0)),
        "anti_squat_r": float(np.clip(anti_squat_r, 0.0, 1.0)),
        "anti_dive_f":  float(np.clip(anti_dive_f,  0.0, 0.9)),
        "anti_dive_r":  float(np.clip(anti_dive_r,  0.0, 0.9)),
        "anti_lift":    float(np.clip(anti_lift_r,  0.0, 0.9)),
        "_ic_front_xz": (ic_f_x, ic_f_z),
        "_ic_rear_xz":  (ic_r_x, ic_r_z),
    }

    print(f"[4WD Anti-Geometry]")
    print(f"  Front IC: X={ic_f_x*1e3:.1f} mm, Z={ic_f_z*1e3:.1f} mm")
    print(f"  Rear  IC: X={ic_r_x*1e3:.1f} mm, Z={ic_r_z*1e3:.1f} mm")
    print(f"  anti_squat_f={result['anti_squat_f']:.3f}  "
          f"anti_squat_r={result['anti_squat_r']:.3f}")
    print(f"  anti_dive_f ={result['anti_dive_f']:.3f}  "
          f"anti_dive_r ={result['anti_dive_r']:.3f}")
    print(f"  anti_lift   ={result['anti_lift']:.3f}")
    print(f"  NOTE: SuspensionSetup.anti_squat is DEPRECATED for Ter27 4WD. "
          f"These values are injected into vehicle_params directly.")

    return result


# ---------------------------------------------------------------------------
# §8  VP injection helper
# ---------------------------------------------------------------------------

def build_ter27_vp_from_kinematics(
    front_kin:  SuspensionKinematics,
    rear_kin:   SuspensionKinematics,
    base_vp:    Dict[str, Any],
    toe_f_rad:  float = 0.0,     # target front static toe [rad]
    toe_r_rad:  float = 0.0,     # target rear static toe [rad]
    camber_f_rad: float = -0.5 * math.pi / 180.0,  # target front camber [rad]
    camber_r_rad: float = -1.0 * math.pi / 180.0,  # target rear camber [rad]
) -> Dict[str, Any]:
    """
    Build a complete vehicle_params dict for the Ter27 Alex configuration.

    This function:
    1. Inverts the kinematic model to find (delta_L_tr, psi_shim) for each axle
       that achieve the target (toe, camber) angles.
    2. Computes all derived kinematic gains (camber_gain, bump_steer, MR poly,
       roll centre height and gradient, anti-geometry).
    3. Merges everything into a copy of base_vp.

    The returned VP is ready for use with DifferentiableMultiBodyVehicle.
    The SuspensionSetup that pairs with this VP should set:
       toe_f     = toe_f_rad * 180/π
       toe_r     = toe_r_rad * 180/π
       camber_f  = camber_f_rad * 180/π
       camber_r  = camber_r_rad * 180/π
       bump_steer_f  = gains_f.bump_steer_lin_rad_per_m  ← DERIVED, not optimized
       bump_steer_r  = gains_r.bump_steer_lin_rad_per_m  ← DERIVED, not optimized
    """
    vp = dict(base_vp)   # shallow copy

    # ── Find adjustable parameters for target toe/camber ──────────────────────
    dL_f = front_kin.delta_L_tr_from_toe(toe_f_rad)
    dL_r = rear_kin.delta_L_tr_from_toe(toe_r_rad)
    ps_f = front_kin.psi_shim_from_camber(camber_f_rad)
    ps_r = rear_kin.psi_shim_from_camber(camber_r_rad)

    # ── Compute kinematic gains ────────────────────────────────────────────────
    gains_f = front_kin.kinematic_gains(jnp.array(dL_f), jnp.array(ps_f))
    gains_r = rear_kin.kinematic_gains(jnp.array(dL_r), jnp.array(ps_r))

    # ── 4WD anti-geometry ─────────────────────────────────────────────────────
    anti = compute_4wd_anti_geometry(front_kin, rear_kin, vp)

    # ── Update VP ─────────────────────────────────────────────────────────────
    # Kinematic gains — derived from hardpoints + toe/camber targets
    vp["motion_ratio_f_poly"]    = list(np.array(gains_f.mr_poly))
    vp["motion_ratio_r_poly"]    = list(np.array(gains_r.mr_poly))
    vp["bump_steer_quad_f"]      = float(gains_f.bump_steer_quad_rad_per_m2)
    vp["bump_steer_quad_r"]      = float(gains_r.bump_steer_quad_rad_per_m2)
    vp["camber_per_m_travel_f"]  = float(gains_f.camber_gain_rad_per_m) * (180.0 / math.pi)
    vp["camber_per_m_travel_r"]  = float(gains_r.camber_gain_rad_per_m) * (180.0 / math.pi)
    vp["h_rc_f"]                 = float(gains_f.rc_height_m)
    vp["h_rc_r"]                 = float(gains_r.rc_height_m)
    vp["dh_rc_dz_f"]             = float(gains_f.drc_dz_m_per_m)
    vp["dh_rc_dz_r"]             = float(gains_r.drc_dz_m_per_m)

    # Anti-geometry (4WD split — separate from SuspensionSetup scalar)
    vp.update(anti)

    # Wheel / geometry
    vp["wheel_radius"] = front_kin.R_wheel
    vp["track_front"]  = 2.0 * front_kin.half_track
    vp["track_rear"]   = 2.0 * rear_kin.half_track

    print(f"[build_ter27_vp_from_kinematics]")
    print(f"  Front: toe={math.degrees(toe_f_rad):.3f}°, "
          f"camber={math.degrees(camber_f_rad):.2f}°, "
          f"dL_tr={dL_f*1e3:.2f} mm, psi_shim={math.degrees(ps_f):.3f}°")
    print(f"  Rear:  toe={math.degrees(toe_r_rad):.3f}°, "
          f"camber={math.degrees(camber_r_rad):.2f}°, "
          f"dL_tr={dL_r*1e3:.2f} mm, psi_shim={math.degrees(ps_r):.3f}°")
    print(f"  Bump steer: F={math.degrees(float(gains_f.bump_steer_lin_rad_per_m))*1e3:.3f} deg/m, "
          f"R={math.degrees(float(gains_r.bump_steer_lin_rad_per_m))*1e3:.3f} deg/m")
    print(f"  MR polys: F={[f'{x:.4f}' for x in vp['motion_ratio_f_poly']]}, "
          f"R={[f'{x:.4f}' for x in vp['motion_ratio_r_poly']]}")
    print(f"  RC heights: F={float(gains_f.rc_height_m)*1e3:.1f} mm, "
          f"R={float(gains_r.rc_height_m)*1e3:.1f} mm")

    return vp