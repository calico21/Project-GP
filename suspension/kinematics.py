# suspension/kinematics.py
# Project-GP — Exact Double A-Arm Kinematic Solver (IFD / Custom VJP)
# =============================================================================
# PATCH LOG
# ---------
# FIX-1  rotation_align: replace jnp.linalg.norm(cross) with safe dot-product
#        sqrt.  jnp.linalg.norm has grad x/|x| → NaN at x=0; the safe form
#        sqrt(dot(x,x)+eps) has grad x/sqrt(dot+eps) → 0 at x=0.
#        ROOT CAUSE of rear Newton NaN: rear ball joints all have X=0 exactly
#        in float32, so cross=0 exactly at theta=0 and jacfwd returns NaN.
#
# FIX-2  L_rod: was np.linalg.norm(PP0 − CHAS_AttPnt_L) ≈ 0.52 m (wrong).
#        CHAS_AttPnt_L is the COILOVER chassis anchor, not the pushrod chassis end.
#        The pushrod/pullrod spans NSMA_PPAttPnt_L → ROCK_RodPnt_L (the rocker).
#        Correct: L_rod = np.linalg.norm(PP0 − ROC_ROD).  The old value exceeded
#        the rocker's reach, making the phi Newton unsolvable → MR NaN/garbage.
#
# FIX-3  MR formula: was dL_coi/dθ_arm [m/rad] (dimensionless only by accident).
#        True motion ratio = dL_spring/dz [dimensionless] = (dL_coi/dθ_arm) /
#        (dz/dθ_arm).  dz/dθ ≈ (e_arm × arm_ball_rel_piv)[Z] (ball-joint Z vel).
#        Added the divisor and a leading minus to get positive MR on bump.
#
# FIX-4  Pullrod arm selection: for pullrod (act_sign=-1) the rod attachment
#        rides with the UPPER A-arm (theta_UA, B1, e_UA, D0_rel_B1, PP0_rel_B1).
#        solve_at_heave now branches on act_sign at Python level (safe because
#        act_sign is a compile-time constant on self).
# =============================================================================

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Dict, NamedTuple, Tuple, Any

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# §0  Pure-JAX geometry primitives
# ---------------------------------------------------------------------------

def rodrigues(e: jax.Array, alpha: jax.Array, v: jax.Array) -> jax.Array:
    """
    Rodrigues rotation formula: rotate vector v by angle alpha about UNIT
    vector e.

    R(e, α) @ v = v·cos(α) + (e × v)·sin(α) + e·(e·v)·(1 − cos(α))
    """
    cos_a = jnp.cos(alpha)
    sin_a = jnp.sin(alpha)
    return v * cos_a + jnp.cross(e, v) * sin_a + e * jnp.dot(e, v) * (1.0 - cos_a)


def rotation_align(a: jax.Array, b: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Return (axis, angle) of the rotation that maps unit vector a → unit vector b.

    FIX-1: sin_angle uses sqrt(dot(cross,cross)+eps) instead of linalg.norm(cross).
    jnp.linalg.norm has gradient x/|x|, which is NaN at x=0.
    sqrt(dot(x,x)+eps) has gradient x/sqrt(dot(x,x)+eps), which is 0 at x=0. ✓
    This prevents jacfwd from returning NaN when a≈b (rear suspension at theta=0).
    """
    cross = jnp.cross(a, b)
    # Safe norm — gradient is well-defined (=0) at cross=0
    sin_angle = jnp.sqrt(jnp.dot(cross, cross) + 1e-24)
    cos_angle  = jnp.dot(a, b)
    angle = jnp.arctan2(sin_angle, cos_angle)   # ∈ [0, π]
    # axis direction is irrelevant when angle≈0 (rotation → identity anyway)
    axis = cross / (sin_angle + 1e-12)
    return axis, angle


def upright_rotation(
    C0: jax.Array, D0: jax.Array, W0: jax.Array,
    C:  jax.Array, D:  jax.Array, psi: jax.Array,
) -> jax.Array:
    """
    Compute R_upright as a function v_body → v_deformed.
    Step 1: R_align maps e_kp_nom → e_kp  (minimal-angle rotation).
    Step 2: R_ψ rotates by ψ about the new e_kp.
    """
    e_kp_nom = (D0 - C0) / jnp.linalg.norm(D0 - C0)
    e_kp     = (D  - C ) / (jnp.linalg.norm(D - C) + 1e-12)

    axis_align, angle_align = rotation_align(e_kp_nom, e_kp)

    def R_up(v: jax.Array) -> jax.Array:
        v1 = rodrigues(axis_align, angle_align, v)   # align kingpin
        v2 = rodrigues(e_kp, psi, v1)                # psi + psi_shim rotation
        return v2

    return R_up


# ---------------------------------------------------------------------------
# §1  Constraint residual  F(θ) = 0
# ---------------------------------------------------------------------------

def _constraint_residual(
    theta: jax.Array,
    z: jax.Array,
    delta_L_tr: jax.Array,
    psi_shim: jax.Array,
    A1: jax.Array, e_LA: jax.Array, C0_rel_A1: jax.Array,
    B1: jax.Array, e_UA: jax.Array, D0_rel_B1: jax.Array,
    C0: jax.Array, D0: jax.Array, W0: jax.Array, TU0: jax.Array, TC: jax.Array,
    L_upright_sq: jax.Array, L_tr_nom: jax.Array, W_z_nom: jax.Array,
) -> jax.Array:
    theta_LA = theta[0]
    theta_UA = theta[1]
    psi      = theta[2]

    C = A1 + rodrigues(e_LA, theta_LA, C0_rel_A1)
    D = B1 + rodrigues(e_UA, theta_UA, D0_rel_B1)

    R_up = upright_rotation(C0, D0, W0, C, D, psi)

    W  = C + R_up(W0  - C0)
    TU = C + R_up(TU0 - C0)

    L_tr_sq = (L_tr_nom + delta_L_tr) ** 2

    CD_sq  = jnp.dot(D - C, D - C)
    L_up   = jnp.sqrt(L_upright_sq + 1e-24)
    F1     = (CD_sq - L_upright_sq) / (2.0 * L_up)

    F2 = W[2] - (W_z_nom + z)

    dT     = TU - TC
    L_tr   = L_tr_nom + delta_L_tr
    F3     = (jnp.dot(dT, dT) - (L_tr_nom + delta_L_tr) ** 2) / (2.0 * L_tr_nom)

    return jnp.array([F1, F2, F3])


# ---------------------------------------------------------------------------
# §2  IFD solver — Newton + custom_vjp
# ---------------------------------------------------------------------------

_NONDIFF = tuple(range(3, 18))

@partial(jax.custom_vjp, nondiff_argnums=_NONDIFF)
def _solve_constraint_system(
    z: jax.Array,
    delta_L_tr: jax.Array,
    psi_shim: jax.Array,
    A1, e_LA, C0_rel_A1,
    B1, e_UA, D0_rel_B1,
    C0, D0, W0, TU0, TC,
    L_upright_sq, L_tr_nom, W_z_nom,
    N_NEWTON: int,
) -> jax.Array:
    geo = (A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1,
           C0, D0, W0, TU0, TC, L_upright_sq, L_tr_nom, W_z_nom)

    def residual(th):
        return _constraint_residual(th, z, delta_L_tr, psi_shim, *geo)

    def nr_step(theta_carry, _):
        F = residual(theta_carry)
        J = jax.jacfwd(residual)(theta_carry)
        dtheta = jnp.linalg.solve(J, F)
        return theta_carry - dtheta, None

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
    residuals = (theta_opt, z, delta_L_tr, psi_shim)
    return theta_opt, residuals


def _solve_bwd(
    A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1,
    C0, D0, W0, TU0, TC, L_upright_sq, L_tr_nom, W_z_nom,
    N_NEWTON,
    res,
    g,
):
    theta_opt, z, delta_L_tr, psi_shim = res

    geo = (A1, e_LA, C0_rel_A1, B1, e_UA, D0_rel_B1,
           C0, D0, W0, TU0, TC, L_upright_sq, L_tr_nom, W_z_nom)

    def residual(th, zz, dL, ps):
        return _constraint_residual(th, zz, dL, ps, *geo)

    J_theta  = jax.jacfwd(lambda t: residual(t, z, delta_L_tr, psi_shim))(theta_opt)
    J_z      = jax.jacfwd(lambda zz: residual(theta_opt, zz, delta_L_tr, psi_shim))(z)
    J_dL     = jax.jacfwd(lambda dL: residual(theta_opt, z, dL, psi_shim))(delta_L_tr)
    J_ps     = jax.jacfwd(lambda ps: residual(theta_opt, z, delta_L_tr, ps))(psi_shim)

    v = jnp.linalg.solve(J_theta.T, g)

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
    A1, e_LA, C0_rel_A1,
    B1, e_UA, D0_rel_B1,
    C0, D0, W0, TU0, TC,
    e_kp_nom: jax.Array,
) -> Dict[str, jax.Array]:
    theta_LA = theta[0]
    theta_UA = theta[1]
    psi      = theta[2]

    C = A1 + rodrigues(e_LA, theta_LA, C0_rel_A1)
    D = B1 + rodrigues(e_UA, theta_UA, D0_rel_B1)

    R_up = upright_rotation(C0, D0, W0, C, D, psi)

    W  = C + R_up(W0  - C0)
    TU = C + R_up(TU0 - C0)

    e_kp = (D - C) / (jnp.linalg.norm(D - C) + 1e-12)

    # Wheel spin axis: nominal [0,1,0] rotated with the upright, then shimmed
    # about the kingpin axis.  Shim must affect camber/toe but not the pose solve.
    e_spin_nom = jnp.array([0.0, 1.0, 0.0])
    e_spin = rodrigues(e_kp, psi_shim, R_up(e_spin_nom))

    # Camber convention: negative = top leans outboard.
    camber_rad = -jnp.arctan2(e_spin[2], e_spin[1])

    # Toe: left wheel toe-in positive.
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
    theta_arm: jax.Array,
    # Generic arm geometry (lower arm for pushrod, upper arm for pullrod):
    A_piv: jax.Array,           # arm chassis pivot (A1 or B1)
    e_arm: jax.Array,           # arm rotation axis (e_LA or e_UA)
    ball_rel_piv: jax.Array,    # ball joint relative to arm pivot  [for dWz/dθ]
    PP0_rel_Apiv: jax.Array,    # rod upright attachment relative to arm pivot
    CHAS_PP: jax.Array,         # (unused — kept for API compatibility)
    # Rocker geometry (fixed in chassis frame):
    ROC_AXI: jax.Array,
    ROC_PIV: jax.Array,
    ROC_ROD: jax.Array,         # rocker rod attachment (nominal)
    ROC_COI: jax.Array,         # rocker coilover attachment (nominal)
    L_rod: jax.Array,           # FIX-2: |PP0 − ROC_ROD| (upright→rocker, NOT →chassis)
    L_coi_nom: jax.Array,
    CHAS_COI: jax.Array,
    actuation_sign: float,      # +1 pushrod, -1 pullrod
) -> jax.Array:
    """
    Compute instantaneous motion ratio  MR = dL_spring / dz  [dimensionless].

    FIX-2: L_rod must be |PP0 − ROC_ROD|. Old code used |PP0 − CHAS_AttPnt|
    which is ~0.52 m — beyond the rocker's reach — making the phi Newton
    unsolvable and producing NaN/garbage MR.

    FIX-3: MR formula is (dL_coi/dθ_arm) / (dz/dθ_arm), dimensionless.
    Old code returned dL_coi/dθ [m/rad] — dimensionally wrong, giving MR~2.89.
    dz/dθ_arm ≈ (e_arm × ball_rel_piv_rotated)[Z]  (ball-joint Z velocity).
    Leading minus converts spring-length-decrease-on-bump to positive MR.

    FIX-4 (caller): For pullrod, caller passes theta_UA, B1, e_UA, D0_rel_B1,
    PP0_rel_B1 instead of the lower-arm equivalents.
    """
    # ── Pushrod/pullrod upright attachment (moves with arm) ───────────────────
    PP = A_piv + rodrigues(e_arm, theta_arm, PP0_rel_Apiv)

    # ── Rocker axis ───────────────────────────────────────────────────────────
    e_roc = (ROC_AXI - ROC_PIV) / (jnp.linalg.norm(ROC_AXI - ROC_PIV) + 1e-12)
    ROD0_rel_PIV = ROC_ROD - ROC_PIV

    # ── Newton for rocker angle φ from |ROD(φ) − PP|² = L_rod² ──────────────
    def rod_constraint(phi):
        ROD = ROC_PIV + rodrigues(e_roc, phi, ROD0_rel_PIV)
        return jnp.dot(ROD - PP, ROD - PP) - L_rod ** 2

    def nr_phi(phi_carry, _):
        F  = rod_constraint(phi_carry)
        dF = jax.grad(rod_constraint)(phi_carry)
        return phi_carry - F / (dF + 1e-12), None

    phi_opt, _ = jax.lax.scan(nr_phi, jnp.array(0.0), None, length=8)

    # ── Coilover length at phi_opt ────────────────────────────────────────────
    COI0_rel_PIV = ROC_COI - ROC_PIV
    COI = ROC_PIV + rodrigues(e_roc, phi_opt, COI0_rel_PIV)
    L_coi = jnp.linalg.norm(COI - CHAS_COI)

    dL_coi_dphi = jax.grad(
        lambda phi: jnp.linalg.norm(
            ROC_PIV + rodrigues(e_roc, phi, COI0_rel_PIV) - CHAS_COI
        )
    )(phi_opt)

    # ── dφ/dθ_arm via implicit function theorem on the rod constraint ─────────
    dF_dphi   = jax.grad(rod_constraint)(phi_opt)
    dphi_dtheta = -jax.grad(
        lambda th: _rod_constraint_at_theta(
            th, phi_opt, A_piv, e_arm, PP0_rel_Apiv,
            ROC_PIV, e_roc, ROD0_rel_PIV, L_rod)
    )(theta_arm) / (dF_dphi + 1e-12)

    # ── FIX-3: dWz/dθ_arm — wheel-centre Z sensitivity to arm angle ──────────
    # Dominant term = Z-component of the arm ball-joint velocity.
    # ball_rel_piv_rotated = rodrigues(e_arm, theta_arm, ball_rel_piv) is the
    # current relative position; cross product gives the instantaneous velocity.
    ball_rotated = rodrigues(e_arm, theta_arm, ball_rel_piv)
    arm_ball_vel = jnp.cross(e_arm, ball_rotated)
    dWz_dtheta   = arm_ball_vel[2]

    # MR = -(dL_coi/dθ) / (dz/dθ) × actuation_sign
    # Negative: spring shortens when wheel bumps → positive MR by convention.
    MR = jnp.abs((dL_coi_dphi * dphi_dtheta) / (dWz_dtheta + 1e-12))
    return MR


def _rod_constraint_at_theta(theta_arm, phi, A_piv, e_arm, PP0_rel_Apiv,
                               ROC_PIV, e_roc, ROD0_rel_PIV, L_rod):
    """Auxiliary: rod constraint as function of theta_arm at fixed phi."""
    PP  = A_piv + rodrigues(e_arm, theta_arm, PP0_rel_Apiv)
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
    Instantaneous roll centre height via the force-based method.
    VPP = intersection of the two A-arm planes in the YZ transverse section.
    RC  = force line (contact-patch → VPP) intersects vehicle centreline.
    """
    nL = jnp.cross(A2 - A1, C - A1)
    nL = nL / (jnp.linalg.norm(nL) + 1e-12)

    nU = jnp.cross(B2 - B1, D - B1)
    nU = nU / (jnp.linalg.norm(nU) + 1e-12)

    e_X = jnp.array([1.0, 0.0, 0.0])
    dL  = jnp.cross(e_X, nL)
    dU  = jnp.cross(e_X, nU)

    def plane_anchor(n, anchor):
        anchor_yz = anchor.at[0].set(0.0)
        d = jnp.cross(e_X, n)
        d = d / (jnp.linalg.norm(d) + 1e-12)
        n_dot_anchor = jnp.dot(n, anchor)
        y0 = jnp.where(jnp.abs(n[1]) > 1e-10, n_dot_anchor / (n[1] + 1e-12), 0.0)
        z0 = jnp.where(jnp.abs(n[2]) > 1e-10,
                        (n_dot_anchor - n[1] * y0) / (n[2] + 1e-12), 0.0)
        return jnp.array([0.0, y0, z0])

    P_L = plane_anchor(nL, A1)
    P_U = plane_anchor(nU, B1)

    A_mat = jnp.array([[dL[1], -dU[1]],
                        [dL[2], -dU[2]]])
    b_vec = jnp.array([P_U[1] - P_L[1],
                        P_U[2] - P_L[2]])

    det = A_mat[0, 0] * A_mat[1, 1] - A_mat[0, 1] * A_mat[1, 0]
    s   = (b_vec[0] * A_mat[1, 1] - b_vec[1] * A_mat[0, 1]) / (det + 1e-10)
    VPP = P_L + s * dL

    cp = jnp.array([0.0, half_track, 0.0])
    d_force = VPP - cp
    d_force = d_force / (jnp.linalg.norm(d_force) + 1e-12)

    t_rc   = -cp[1] / (d_force[1] + 1e-10)
    RC_pt  = cp + t_rc * d_force
    return RC_pt[2]


# ---------------------------------------------------------------------------
# §6  SuspensionKinematics class
# ---------------------------------------------------------------------------

class KinematicOutputs(NamedTuple):
    camber_rad:    jax.Array
    toe_rad:       jax.Array
    wheel_pos:     jax.Array
    motion_ratio:  jax.Array
    roll_centre_z: jax.Array
    C:             jax.Array
    D:             jax.Array
    e_kp:          jax.Array


class KinematicGains(NamedTuple):
    camber_gain_rad_per_m:      jax.Array
    bump_steer_lin_rad_per_m:   jax.Array
    bump_steer_quad_rad_per_m2: jax.Array
    mr_poly:                    jax.Array
    rc_height_m:                jax.Array
    drc_dz_m_per_m:             jax.Array
    kpi_rad:                    float
    caster_rad:                 float
    mech_trail_m:               float


class SuspensionKinematics:
    """
    Full Double A-Arm kinematic solver with IFD (Implicit Function Diff).

    Differentiable parameters: delta_L_tr [m], psi_shim [rad].
    All hardpoints frozen as numpy → converted to JAX float32 in __init__.
    """

    N_NEWTON: int = 8

    def __init__(self, hpts: Dict[str, Any], side: str = "left"):
        assert side in ("left", "right")

        sign = 1.0 if side == "left" else -1.0
        def pt(key: str) -> np.ndarray:
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
        CHAS_PP_np = pt("CHAS_AttPnt_L")       # = coilover chassis anchor

        # ── Rocker ───────────────────────────────────────────────────────────
        ROC_AXI_np = pt("CHAS_RocAxi_L")
        ROC_PIV_np = pt("CHAS_RocPiv_L")
        ROC_ROD_np = pt("ROCK_RodPnt_L")
        ROC_COI_np = pt("ROCK_CoiPnt_L")
        # Coilover chassis anchor = CHAS_AttPnt_L (Optimum K convention).
        CHAS_COI_np = CHAS_PP_np.copy()

        # ── Axis unit vectors ─────────────────────────────────────────────────
        e_LA_np = A2_np - A1_np;  e_LA_np /= (np.linalg.norm(e_LA_np) + 1e-12)
        e_UA_np = B2_np - B1_np;  e_UA_np /= (np.linalg.norm(e_UA_np) + 1e-12)

        C0_rel_A1_np = C0_np - A1_np
        D0_rel_B1_np = D0_np - B1_np

        L_upright_sq_np = np.dot(D0_np - C0_np, D0_np - C0_np)
        L_tr_nom_np     = np.linalg.norm(TU0_np - TC_np)
        W_z_nom_np      = W0_np[2]

        # FIX-2: Correct rod length = upright attachment → rocker rod point.
        # CHAS_AttPnt_L is the COILOVER anchor, not the pushrod chassis end.
        # The pushrod/pullrod spans: NSMA_PPAttPnt_L  →  ROCK_RodPnt_L.
        L_rod_np     = np.linalg.norm(PP0_np - ROC_ROD_np)

        L_coi_nom_np = np.linalg.norm(ROC_COI_np - CHAS_COI_np)

        # Rod attachment relative to arm pivot.
        # FIX-4: pushrod → relative to lower arm pivot A1
        #         pullrod → relative to upper arm pivot B1
        PP0_rel_A1_np = PP0_np - A1_np   # pushrod (lower arm)
        PP0_rel_B1_np = PP0_np - B1_np   # pullrod (upper arm)

        e_kp_nom_np = D0_np - C0_np
        e_kp_nom_np /= (np.linalg.norm(e_kp_nom_np) + 1e-12)

        self.kpi_rad    = float(np.arctan2(e_kp_nom_np[1], e_kp_nom_np[2]))
        self.caster_rad = float(np.arctan2(-e_kp_nom_np[0], e_kp_nom_np[2]))
        self.mech_trail_m = float(R_w * np.tan(abs(self.caster_rad)))
        self.R_wheel    = R_w
        self.half_track = HT

        # ── JAX arrays ───────────────────────────────────────────────────────
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
            PP0_rel_A1=to_jax(PP0_rel_A1_np),   # pushrod
            PP0_rel_B1=to_jax(PP0_rel_B1_np),   # pullrod  (FIX-4)
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
              f"L_tr={float(L_tr_nom_np)*1e3:.1f} mm, "
              f"L_rod={float(L_rod_np)*1e3:.1f} mm  "
              f"({'pushrod' if self.act_sign > 0 else 'pullrod'})")

    # ── Public interface ──────────────────────────────────────────────────────

    def solve_at_heave(
        self,
        z:          jax.Array,
        delta_L_tr: jax.Array,
        psi_shim:   jax.Array,
    ) -> KinematicOutputs:
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

        # FIX-4: Select arm geometry based on actuation type.
        # act_sign is a Python float → branch evaluated at trace time → safe.
        if self.act_sign > 0:   # pushrod: PP rides with lower arm
            mr = _motion_ratio_at_theta_LA(
                theta[0],                                   # theta_LA
                j["A1"], j["e_LA"], j["C0_rel_A1"],        # lower arm geom
                j["PP0_rel_A1"], j["CHAS_PP"],
                j["ROC_AXI"], j["ROC_PIV"], j["ROC_ROD"], j["ROC_COI"],
                j["L_rod"], j["L_coi_nom"], j["CHAS_COI"],
                self.act_sign,
            )
        else:                   # pullrod: PP rides with upper arm
            mr = _motion_ratio_at_theta_LA(
                theta[1],                                   # theta_UA
                j["B1"], j["e_UA"], j["D0_rel_B1"],        # upper arm geom
                j["PP0_rel_B1"], j["CHAS_PP"],
                j["ROC_AXI"], j["ROC_PIV"], j["ROC_ROD"], j["ROC_COI"],
                j["L_rod"], j["L_coi_nom"], j["CHAS_COI"],
                self.act_sign,
            )

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
        z_array:    jax.Array,
        delta_L_tr: jax.Array,
        psi_shim:   jax.Array,
    ) -> KinematicOutputs:
        return jax.vmap(
            lambda z: self.solve_at_heave(z, delta_L_tr, psi_shim)
        )(z_array)

    def kinematic_gains(
        self,
        delta_L_tr: jax.Array,
        psi_shim:   jax.Array,
        dz:         float = 1e-4,
        n_mr_pts:   int   = 50,
    ) -> KinematicGains:
        z0 = jnp.array(0.0)

        camber_gain = jax.grad(
            lambda z: self.solve_at_heave(z, delta_L_tr, psi_shim).camber_rad
        )(z0)

        bs_fn   = lambda z: self.solve_at_heave(z, delta_L_tr, psi_shim).toe_rad
        bs_lin  = jax.grad(bs_fn)(z0)
        bs_quad = jax.grad(jax.grad(bs_fn))(z0) * 0.5

        rc_fn  = lambda z: self.solve_at_heave(z, delta_L_tr, psi_shim).roll_centre_z
        rc0    = rc_fn(z0)
        drc_dz = jax.grad(rc_fn)(z0)

        z_arr   = jnp.linspace(-0.08, 0.15, n_mr_pts)
        outputs = self.sweep(z_arr, delta_L_tr, psi_shim)
        mr_arr  = outputs.motion_ratio

        Vander = jnp.column_stack([
            jnp.ones(n_mr_pts),
            z_arr,
            z_arr ** 2,
        ])
        mr_poly, _, _, _ = jnp.linalg.lstsq(Vander, mr_arr)

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
        def toe_at(dL: float) -> float:
            out = self.solve_at_heave(jnp.array(0.0), jnp.array(dL), jnp.array(0.0))
            return float(out.toe_rad)

        dL = 0.0
        for _ in range(20):
            f  = toe_at(dL) - toe_target_rad
            df = float(jax.grad(
                lambda d: self.solve_at_heave(jnp.array(0.0), d, jnp.array(0.0)).toe_rad
            )(jnp.array(dL)))
            if abs(df) < 1e-12:
                break
            dL -= f / df
            if abs(f) < 1e-8:
                break
        return dL

    def psi_shim_from_camber(self, camber_target_rad: float) -> float:
        def camber_at(ps: float) -> float:
            out = self.solve_at_heave(jnp.array(0.0), jnp.array(0.0), jnp.array(ps))
            return float(out.camber_rad)

        camber_0 = camber_at(0.0)
        ps = camber_target_rad - camber_0

        # INCREASED ITERATIONS to 50 for extreme geometries
        for _ in range(50):
            f  = camber_at(ps) - camber_target_rad
            df = float(jax.grad(
                lambda p: self.solve_at_heave(jnp.array(0.0), jnp.array(0.0), p).camber_rad
            )(jnp.array(ps)))
            
            if abs(df) < 1e-12:
                break
                
            # RELAXATION FACTOR: Instead of hard-clipping, we take a 50% step.
            # This prevents violent divergence while still allowing large movements if needed.
            step = f / df
            ps -= 0.5 * step  
            
            if abs(f) < 1e-8:
                break
        return ps


# ---------------------------------------------------------------------------
# §7  4WD Anti-Geometry
# ---------------------------------------------------------------------------

def compute_4wd_anti_geometry(
    front_kin: SuspensionKinematics,
    rear_kin:  SuspensionKinematics,
    VP: Dict[str, Any],
) -> Dict[str, float]:
    h_cg = VP.get("h_cg", 0.285)
    lf   = VP.get("lf",   0.8525)
    lr   = VP.get("lr",   0.6975)
    L    = lf + lr
    brake_bias_f = VP.get("brake_bias_f", 0.60)

    def _instant_centre_z(kin: SuspensionKinematics) -> Tuple[float, float]:
        hpts = kin.hpts
        A1_np = hpts["CHAS_LowFor"].copy(); A1_np[1] = 0.0
        A2_np = hpts["CHAS_LowAft"].copy(); A2_np[1] = 0.0
        B1_np = hpts["CHAS_UppFor"].copy(); B1_np[1] = 0.0
        B2_np = hpts["CHAS_UppAft"].copy(); B2_np[1] = 0.0

        dLA_x = A2_np[0] - A1_np[0]; dLA_z = A2_np[2] - A1_np[2]
        dUA_x = B2_np[0] - B1_np[0]; dUA_z = B2_np[2] - B1_np[2]

        M = np.array([[dLA_x, -dUA_x], [dLA_z, -dUA_z]])
        b = np.array([B1_np[0] - A1_np[0], B1_np[2] - A1_np[2]])
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) < 1e-8:
            return 0.0, 0.0
        s = (b[0] * M[1, 1] - b[1] * M[0, 1]) / det
        ic_x = A1_np[0] + s * dLA_x
        ic_z = A1_np[2] + s * dLA_z
        return float(ic_x), float(ic_z)

    ic_f_x, ic_f_z = _instant_centre_z(front_kin)
    anti_dive_f  = ic_f_z / (h_cg * brake_bias_f * L / (lf + 1e-6) + 1e-6)
    anti_squat_f = ic_f_z / (h_cg * lf / (L + 1e-6) + 1e-6)

    ic_r_x, ic_r_z = _instant_centre_z(rear_kin)
    anti_squat_r = ic_r_z / (h_cg * lr / (L + 1e-6) + 1e-6)
    anti_dive_r  = ic_r_z / (h_cg * (1.0 - brake_bias_f) * L / (lr + 1e-6) + 1e-6)
    anti_lift_r  = anti_squat_r

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
    return result


# ---------------------------------------------------------------------------
# §8  VP injection helper
# ---------------------------------------------------------------------------

def build_ter27_vp_from_kinematics(
    front_kin:    SuspensionKinematics,
    rear_kin:     SuspensionKinematics,
    base_vp:      Dict[str, Any],
    toe_f_rad:    float = 0.0,
    toe_r_rad:    float = 0.0,
    camber_f_rad: float = -0.5 * math.pi / 180.0,
    camber_r_rad: float = -1.0 * math.pi / 180.0,
) -> Dict[str, Any]:
    vp = dict(base_vp)

    dL_f = front_kin.delta_L_tr_from_toe(toe_f_rad)
    dL_r = rear_kin.delta_L_tr_from_toe(toe_r_rad)
    ps_f = front_kin.psi_shim_from_camber(camber_f_rad)
    ps_r = rear_kin.psi_shim_from_camber(camber_r_rad)

    gains_f = front_kin.kinematic_gains(jnp.array(dL_f), jnp.array(ps_f))
    gains_r = rear_kin.kinematic_gains(jnp.array(dL_r), jnp.array(ps_r))

    anti = compute_4wd_anti_geometry(front_kin, rear_kin, vp)

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
    vp.update(anti)
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
    return vp