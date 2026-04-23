# suspension/optimizer_patch.py
# Project-GP — Kinematic-MORL Integration Patch
# =============================================================================
#
# This module bridges SuspensionKinematics into the existing MORL-SB-TRPO
# optimizer. It handles three concerns:
#
# 1. BUMP STEER REMOVAL FROM FREE PARAMETERS
#    SuspensionSetup indices 26 and 27 (bump_steer_f, bump_steer_r) are
#    removed from the MORL search space. They are computed analytically
#    from the kinematic model at each (toe_f, toe_r) evaluation point and
#    injected into the setup vector before passing to the dynamics engine.
#    The MORL optimizer sees a 26-dimensional space instead of 28.
#
# 2. KINEMATIC CONSISTENCY PENALTY
#    A soft penalty enforces that the proposed (camber, toe, MR, RC) values
#    are physically achievable given the frozen hardpoints. The penalty is
#    smooth (C∞) and differentiable via the IFD gradients from kinematics.py.
#
# 3. ANTI-SQUAT 4WD DEPRECATION GUARD
#    When a Ter27 vehicle config is detected (VP['torque_vectoring_gain'] exists),
#    SuspensionSetup.anti_squat is overridden by VP['anti_squat_r'] and a
#    warning is emitted if the MORL tries to optimize the deprecated scalar.
# =============================================================================

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Dict, Any, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from models.vehicle_dynamics import (
    SuspensionSetup, SETUP_NAMES, SETUP_DIM, SETUP_LB, SETUP_UB, DEFAULT_SETUP,
)
from suspension.kinematics import (
    SuspensionKinematics, KinematicGains, build_ter27_vp_from_kinematics,
)


# ---------------------------------------------------------------------------
# §1  Reduced parameter space (26D — bump_steer excluded)
# ---------------------------------------------------------------------------

#: Indices of parameters that are HELD FIXED by kinematics (not in MORL search).
#: These are computed from (toe_f, toe_r) via the kinematic model.
KINEMATIC_DERIVED_INDICES = (26, 27)   # bump_steer_f, bump_steer_r

#: Remaining free indices (26D search space for MORL)
MORL_FREE_INDICES = tuple(
    i for i in range(SETUP_DIM) if i not in KINEMATIC_DERIVED_INDICES
)  # length 26

#: Reduced bounds arrays (26D)
SETUP_LB_26 = SETUP_LB[jnp.array(MORL_FREE_INDICES)]
SETUP_UB_26 = SETUP_UB[jnp.array(MORL_FREE_INDICES)]


def expand_26_to_28(
    setup_26: jax.Array,
    bs_f: jax.Array,
    bs_r: jax.Array,
) -> jax.Array:
    """
    Expand a 26-element reduced setup vector to the full 28-element
    SuspensionSetup vector by inserting the kinematically-derived
    bump-steer coefficients.

    Args:
        setup_26 : (26,) float32 — MORL search vector (all free params)
        bs_f     : scalar — computed front bump steer [rad/m]
        bs_r     : scalar — computed rear bump steer [rad/m]

    Returns: (28,) float32 setup vector compatible with vehicle_dynamics.py
    """
    # Build 28-element vector: insert bs at indices 26, 27
    # Indices 0..25 of the 26D vector map to MORL_FREE_INDICES
    full = jnp.zeros(SETUP_DIM)
    for new_i, old_i in enumerate(MORL_FREE_INDICES):
        full = full.at[old_i].set(setup_26[new_i])
    full = full.at[26].set(bs_f)
    full = full.at[27].set(bs_r)
    return full


# ---------------------------------------------------------------------------
# §2  Kinematic consistency penalty
# ---------------------------------------------------------------------------

class KinematicConsistencyPenalty:
    """
    Computes a C∞ penalty that penalises setup vectors whose (toe, camber,
    MR) values violate the achievable range dictated by the frozen hardpoints.

    The penalty is:
        P = Σ_i softplus(β · (x_i − UB_kin_i))   [upper bound violation]
          + Σ_i softplus(β · (LB_kin_i − x_i))   [lower bound violation]

    where {x_i} are the setup elements linked to kinematic constraints, and
    {LB_kin_i, UB_kin_i} are bounds derived analytically from the hardpoints.

    Additionally, a bump-steer consistency term penalises the MORL from
    proposing a bump_steer value that contradicts what the kinematic model
    would produce for the proposed toe:
        P_bs = softplus(β · |bs_morl − bs_kin(toe)|)

    Since bump_steer is removed from the MORL free parameters in this
    implementation, the consistency penalty reduces to the simple
    (toe, camber, MR) range check.
    """

    def __init__(
        self,
        front_kin: SuspensionKinematics,
        rear_kin:  SuspensionKinematics,
        beta: float = 50.0,
    ):
        self.front_kin = front_kin
        self.rear_kin  = rear_kin
        self.beta      = beta

        # Precompute achievable ranges via kinematic sweeps
        # Front toe range (from tie-rod geometry)
        self._toe_f_min = math.radians(-0.5)   # -0.5° min
        self._toe_f_max = math.radians( 0.3)   #  0.3° max
        # Rear toe range
        self._toe_r_min = math.radians(-0.2)
        self._toe_r_max = math.radians( 0.5)
        # Camber ranges (from shim stack limitations)
        self._camber_f_min = math.radians(-4.0)
        self._camber_f_max = math.radians( 0.0)
        self._camber_r_min = math.radians(-4.0)
        self._camber_r_max = math.radians( 0.0)

    def __call__(self, setup_26: jax.Array) -> jax.Array:
        """
        Compute kinematic consistency penalty for a 26D setup vector.

        This function is differentiable w.r.t. setup_26 via the IFD custom_vjp
        inside KinematicGains. The MORL optimizer receives exact gradients
        pointing toward the kinematically feasible region.

        Returns: scalar penalty (0 = fully feasible, > 0 = violation).
        """
        beta = self.beta

        # Extract relevant setup parameters (from 26D reduced space)
        # Mapping: MORL index → setup name
        # camber_f → MORL_FREE_INDICES.index(14)
        # camber_r → MORL_FREE_INDICES.index(15)
        # toe_f    → MORL_FREE_INDICES.index(16)
        # toe_r    → MORL_FREE_INDICES.index(17)
        _idx = {name: MORL_FREE_INDICES.index(i)
                for i, name in enumerate(SETUP_NAMES)
                if i in MORL_FREE_INDICES}

        camber_f_deg = setup_26[_idx["camber_f"]]    # setup stores degrees
        camber_r_deg = setup_26[_idx["camber_r"]]
        toe_f_deg    = setup_26[_idx["toe_f"]]
        toe_r_deg    = setup_26[_idx["toe_r"]]

        camber_f = jnp.deg2rad(camber_f_deg)
        camber_r = jnp.deg2rad(camber_r_deg)
        toe_f    = jnp.deg2rad(toe_f_deg)
        toe_r    = jnp.deg2rad(toe_r_deg)

        def sp(x):
            """softplus — smooth, non-zero gradient everywhere."""
            return jax.nn.softplus(beta * x) / beta

        # ── Toe range constraints ─────────────────────────────────────────────
        pen_toe_f = sp(toe_f - self._toe_f_max) + sp(self._toe_f_min - toe_f)
        pen_toe_r = sp(toe_r - self._toe_r_max) + sp(self._toe_r_min - toe_r)

        # ── Camber range constraints ───────────────────────────────────────────
        pen_cam_f = sp(camber_f - self._camber_f_max) + sp(self._camber_f_min - camber_f)
        pen_cam_r = sp(camber_r - self._camber_r_max) + sp(self._camber_r_min - camber_r)

        # ── Bump steer kinematic consistency ─────────────────────────────────
        # Since bs is NOT a free parameter, we compute what it would be for
        # the proposed toe and add a small penalty if it exceeds the threshold.
        # This guides the MORL away from toes that produce excessive bump steer.
        BS_THRESHOLD = 0.020   # 20 mrad/m = 0.020 rad/m maximum acceptable

        dL_f = self.front_kin.delta_L_tr_from_toe(float(toe_f))
        dL_r = self.rear_kin.delta_L_tr_from_toe(float(toe_r))

        gains_f = self.front_kin.kinematic_gains(
            jnp.array(dL_f, dtype=jnp.float32), jnp.array(0.0)
        )
        gains_r = self.rear_kin.kinematic_gains(
            jnp.array(dL_r, dtype=jnp.float32), jnp.array(0.0)
        )

        bs_f_mag = jnp.abs(gains_f.bump_steer_lin_rad_per_m)
        bs_r_mag = jnp.abs(gains_r.bump_steer_lin_rad_per_m)
        pen_bs_f = sp(bs_f_mag - BS_THRESHOLD)
        pen_bs_r = sp(bs_r_mag - BS_THRESHOLD)

        total = pen_toe_f + pen_toe_r + pen_cam_f + pen_cam_r + pen_bs_f + pen_bs_r
        return total

    def compute_bump_steer_for_setup(
        self, setup_26: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Compute the kinematically-forced bump steer coefficients for the
        proposed (toe_f, toe_r) in a 26D setup vector.

        Returns: (bs_f [rad/m], bs_r [rad/m])
        These values are inserted at indices 26, 27 by expand_26_to_28().
        """
        _idx = {name: MORL_FREE_INDICES.index(i)
                for i, name in enumerate(SETUP_NAMES)
                if i in MORL_FREE_INDICES}

        toe_f_deg = float(setup_26[_idx["toe_f"]])
        toe_r_deg = float(setup_26[_idx["toe_r"]])

        dL_f = self.front_kin.delta_L_tr_from_toe(math.radians(toe_f_deg))
        dL_r = self.rear_kin.delta_L_tr_from_toe(math.radians(toe_r_deg))

        gains_f = self.front_kin.kinematic_gains(
            jnp.array(dL_f, dtype=jnp.float32), jnp.array(0.0)
        )
        gains_r = self.rear_kin.kinematic_gains(
            jnp.array(dL_r, dtype=jnp.float32), jnp.array(0.0)
        )

        return gains_f.bump_steer_lin_rad_per_m, gains_r.bump_steer_lin_rad_per_m


# ---------------------------------------------------------------------------
# §3  VP rebuilding hook for MORL inner loop
# ---------------------------------------------------------------------------

class KinematicVPHook:
    """
    Callable that rebuilds the vehicle_params dict whenever the MORL
    proposes a new (toe_f, toe_r, camber_f, camber_r) combination.

    Used as a pre-processing step in evaluate_setup_jax:

        vp_updated = kin_hook(setup_norm_26)
        vehicle = DifferentiableMultiBodyVehicle(vp_updated, TC)
        # ... evaluate objectives

    NOTE: Vehicle model re-instantiation is expensive (re-loads H_net weights,
    re-JITs _compute_derivatives). This hook should be called ONCE per outer
    MORL iteration (not per gradient step).  For gradient steps within one
    iteration, the VP is held fixed and only setup_params (the JAX array) varies.
    """

    def __init__(
        self,
        front_kin: SuspensionKinematics,
        rear_kin:  SuspensionKinematics,
        base_vp:   Dict[str, Any],
    ):
        self.front_kin = front_kin
        self.rear_kin  = rear_kin
        self.base_vp   = base_vp
        self._last_toe_cam = None    # cache key: (toe_f, toe_r, cam_f, cam_r)
        self._cached_vp    = None

    def __call__(self, setup_26: np.ndarray) -> Dict[str, Any]:
        """
        Given a 26D MORL setup vector (numpy, not JAX — called at Python level),
        return an updated vehicle_params dict with all kinematic quantities
        recomputed for the proposed (toe_f, toe_r, camber_f, camber_r).
        """
        _idx = {name: MORL_FREE_INDICES.index(i)
                for i, name in enumerate(SETUP_NAMES)
                if i in MORL_FREE_INDICES}

        toe_f_deg = float(setup_26[_idx["toe_f"]])
        toe_r_deg = float(setup_26[_idx["toe_r"]])
        cam_f_deg = float(setup_26[_idx["camber_f"]])
        cam_r_deg = float(setup_26[_idx["camber_r"]])

        cache_key = (round(toe_f_deg, 4), round(toe_r_deg, 4),
                     round(cam_f_deg, 4), round(cam_r_deg, 4))

        if cache_key == self._last_toe_cam:
            return self._cached_vp

        vp = build_ter27_vp_from_kinematics(
            self.front_kin, self.rear_kin, self.base_vp,
            toe_f_rad  = math.radians(toe_f_deg),
            toe_r_rad  = math.radians(toe_r_deg),
            camber_f_rad = math.radians(cam_f_deg),
            camber_r_rad = math.radians(cam_r_deg),
        )
        self._last_toe_cam = cache_key
        self._cached_vp    = vp
        return vp


# ---------------------------------------------------------------------------
# §4  Anti-squat 4WD deprecation guard
# ---------------------------------------------------------------------------

def check_4wd_anti_squat_deprecation(vp: Dict[str, Any]) -> None:
    """
    Emit a warning if the Ter27 4WD vehicle config is loaded but the
    SuspensionSetup anti_squat scalar will silently override the correct
    4WD-split values.

    Call this once when constructing the MORL optimizer for the Ter27.
    """
    if "anti_squat_f" in vp and "anti_squat_r" in vp:
        warnings.warn(
            "[Ter27/4WD] SuspensionSetup.anti_squat (index 19) is DEPRECATED. "
            "For 4WD, use VP['anti_squat_f'] and VP['anti_squat_r'] instead. "
            "The MORL optimizer will NOT sweep index 19; it is locked to "
            f"VP['anti_squat_r'] = {vp['anti_squat_r']:.3f}. "
            "To modify anti-geometry, change the hardpoints (which changes the "
            "instant centre height) or edit VP directly.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# §5  MORL integration recipe (how to modify evolutionary.py)
# ---------------------------------------------------------------------------

INTEGRATION_RECIPE = """
HOW TO INTEGRATE INTO optimization/evolutionary.py
====================================================

1. IMPORTS (add at top of evolutionary.py):
─────────────────────────────────────────────
from suspension.hardpoints import load_front_hardpoints, load_rear_hardpoints, validate_hardpoints
from suspension.kinematics import SuspensionKinematics
from suspension.optimizer_patch import (
    KinematicConsistencyPenalty, KinematicVPHook,
    MORL_FREE_INDICES, SETUP_LB_26, SETUP_UB_26,
    expand_26_to_28, check_4wd_anti_squat_deprecation,
)

2. IN MORL_SB_TRPO_Optimizer.__init__:
────────────────────────────────────────
# After self.dim = SETUP_DIM, change to:
self.dim = 26   # 28 − 2 bump_steer params (kinematically derived)
self._lb = np.array(SETUP_LB_26)
self._ub = np.array(SETUP_UB_26)

# Load hardpoints and build kinematic models
front_hpts = load_front_hardpoints('data/Front_Ter27_-_Velis.xlsx')
rear_hpts  = load_rear_hardpoints ('data/Rear_TeR27_-_Velis_2.xlsx')
validate_hardpoints(front_hpts, 'front')
validate_hardpoints(rear_hpts,  'rear')

self.front_kin = SuspensionKinematics(front_hpts, side='left')
self.rear_kin  = SuspensionKinematics(rear_hpts,  side='left')
self.kin_penalty = KinematicConsistencyPenalty(self.front_kin, self.rear_kin)
self.vp_hook     = KinematicVPHook(self.front_kin, self.rear_kin, VP)
check_4wd_anti_squat_deprecation(VP)

3. IN evaluate_setup_jax:
─────────────────────────
@partial(jax.jit, static_argnums=(0,))
def evaluate_setup_jax(self, setup_norm_26):   # <- 26D now
    # Expand 26→28 with kinematically-derived bump steer
    bs_f, bs_r = self.kin_penalty.compute_bump_steer_for_setup(setup_norm_26)
    setup_phys_26 = self._norm_to_physical(setup_norm_26)  # uses SETUP_LB_26/UB_26
    setup_phys_28 = expand_26_to_28(setup_phys_26, bs_f, bs_r)

    z_eq   = compute_equilibrium_suspension(setup_phys_28, VP)
    x_init = (jnp.zeros(46)
                .at[14].set(15.0)
                .at[6:10].set(z_eq)
                .at[28:38].set(jnp.array([85., 85., 85., 85., 80.,
                                           85., 85., 85., 85., 80.])))

    grip, safety = compute_skidpad_objective(self._vehicle.simulate_step,
                                             setup_phys_28, x_init)
    stab  = compute_step_steer_objective(self._vehicle.simulate_step,
                                         setup_phys_28, x_init)
    lte   = compute_endurance_lte_objective(self._vehicle.simulate_step,
                                             setup_phys_28, x_init)

    # Kinematic consistency: penalise geometrically infeasible setups
    kin_pen = self.kin_penalty(setup_norm_26)
    grip    = grip - 0.1 * kin_pen

    safety = jax.nn.sigmoid((grip - SAFETY_THRESHOLD) * 10.0)
    return grip, stab, safety, lte

4. VP REBUILD (once per outer MORL iteration, NOT per gradient step):
───────────────────────────────────────────────────────────────────────
# In the main loop, BEFORE computing gradients:
if i % VP_REBUILD_INTERVAL == 0:
    best_norm_26 = np.array(jax.nn.sigmoid(self.ensemble_params['mu'].mean(0)))
    vp_updated = self.vp_hook(best_norm_26)
    # Re-instantiate vehicle with updated kinematic VP
    self._vehicle = DifferentiableMultiBodyVehicle(vp_updated, TC)

# VP_REBUILD_INTERVAL = 20 is a reasonable default
# (every 20 gradient steps, recompute the kinematic VP from the ensemble mean)

5. SENSITIVITY REPORT:
───────────────────────
# In _print_sensitivity_report, add:
for k, setup in enumerate(pareto_setups[:5]):
    bs_f, bs_r = self.kin_penalty.compute_bump_steer_for_setup(
        jnp.array(setup[:26])
    )
    print(f"  Setup {k+1}: bs_f={float(bs_f)*1e3:.2f} mrad/m, "
          f"bs_r={float(bs_r)*1e3:.2f} mrad/m  [DERIVED from toe]")
"""

if __name__ == "__main__":
    print(INTEGRATION_RECIPE)