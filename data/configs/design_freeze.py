# data/configs/design_freeze.py
# Project-GP  —  Parameter Freeze/Unfreeze Controller
# ═══════════════════════════════════════════════════════════════════════════════
#
# Progressive lockdown for suspension geometry design:
#   Phase 1 (concept):   all 28 params FREE → optimizer explores full space
#   Phase 2 (CAD lock):  hardpoints committed → freeze caster, anti-geometry
#   Phase 3 (prototype): springs/dampers TBD → free only tune-at-track params
#
# MECHANISM:
#   Frozen params get LB[i] = UB[i] = fixed_value. The existing tanh
#   projection in SuspensionSetup.project_to_bounds() handles this:
#     mid = val, half = 0 → output = val (constant, zero gradient).
#   Belt-and-suspenders: a gradient mask zeros frozen dims after each step.
#
# USAGE:
#   from data.configs.design_freeze import DesignFreeze
#
#   freeze = DesignFreeze.for_ter27_phase2()   # preset
#   # — or —
#   freeze = DesignFreeze({
#       'castor_f':    5.5,     # locked from CAD
#       'anti_dive_f': 0.35,    # locked from hardpoint geometry
#       'h_cg':        0.310,   # locked from packaging study
#   })
#
#   lb, ub = freeze.apply_bounds(design_lb, design_ub)
#   mask   = freeze.gradient_mask()    # (28,) float32: 1.0=free, 0.0=frozen
#   freeze.summary()                   # prints human-readable table
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Dict, Optional, List
import jax.numpy as jnp

from models.vehicle_dynamics import SETUP_NAMES, SETUP_DIM


class DesignFreeze:
    """
    Manages which suspension parameters are frozen (CAD-committed) vs free
    (still being explored by the optimizer).

    Frozen parameters are pinned at their fixed values by clamping LB = UB.
    Free parameters retain their design-mode bounds.
    """

    def __init__(self, frozen: Dict[str, float] = None):
        """
        Args:
            frozen: Dict mapping parameter name → fixed value.
                    Only parameters listed here are frozen.
                    All others remain free for optimization.

        Example:
            DesignFreeze({
                'castor_f': 5.5,
                'anti_dive_f': 0.35,
                'anti_dive_r': 0.15,
            })
        """
        self._frozen: Dict[str, float] = {}
        if frozen:
            for name, val in frozen.items():
                if name not in SETUP_NAMES:
                    raise ValueError(
                        f"Unknown parameter '{name}'. "
                        f"Valid: {SETUP_NAMES}"
                    )
                self._frozen[name] = float(val)

    # ─────────────────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def frozen_names(self) -> List[str]:
        return list(self._frozen.keys())

    @property
    def free_names(self) -> List[str]:
        return [n for n in SETUP_NAMES if n not in self._frozen]

    @property
    def n_free(self) -> int:
        return SETUP_DIM - len(self._frozen)

    @property
    def n_frozen(self) -> int:
        return len(self._frozen)

    def is_frozen(self, name: str) -> bool:
        return name in self._frozen

    def frozen_value(self, name: str) -> float:
        return self._frozen[name]

    # ─────────────────────────────────────────────────────────────────────────
    # Bound clamping
    # ─────────────────────────────────────────────────────────────────────────

    def apply_bounds(
        self,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns (new_lb, new_ub) with frozen params clamped to LB=UB=val.

        The existing project_to_bounds() tanh projection handles this:
            mid  = 0.5*(UB+LB) = val
            half = 0.5*(UB-LB) = 0
            output = mid + 0 * tanh(...) = val   ← constant, zero grad
        """
        new_lb = lb.copy()
        new_ub = ub.copy()

        for name, val in self._frozen.items():
            idx = SETUP_NAMES.index(name)
            new_lb = new_lb.at[idx].set(val)
            new_ub = new_ub.at[idx].set(val)

        return new_lb, new_ub

    # ─────────────────────────────────────────────────────────────────────────
    # Gradient mask (belt-and-suspenders safety)
    # ─────────────────────────────────────────────────────────────────────────

    def gradient_mask(self) -> jnp.ndarray:
        """
        Returns (28,) float32 mask: 1.0 for free params, 0.0 for frozen.

        Apply after computing gradients:
            grads['mu'] = grads['mu'] * freeze.gradient_mask()

        This is technically redundant with bound clamping (the tanh
        projection already zeroes the gradient), but protects against
        floating-point drift accumulating over thousands of Adam steps.
        """
        mask = jnp.ones(SETUP_DIM, dtype=jnp.float32)
        for name in self._frozen:
            idx = SETUP_NAMES.index(name)
            mask = mask.at[idx].set(0.0)
        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Reset vector (force frozen params back to their fixed values)
    # ─────────────────────────────────────────────────────────────────────────

    def enforce(self, setup_vec: jnp.ndarray) -> jnp.ndarray:
        """
        Forces frozen params back to their fixed values in the given
        setup vector. Call after each optimizer step for absolute safety.

        Differentiable: uses straight-through (lax.stop_gradient on delta).
        """
        for name, val in self._frozen.items():
            idx = SETUP_NAMES.index(name)
            # Straight-through: replace value but don't backprop through it
            delta = val - setup_vec[idx]
            setup_vec = setup_vec.at[idx].add(
                jax.lax.stop_gradient(delta)
            )
        return setup_vec

    # ─────────────────────────────────────────────────────────────────────────
    # Presets for common design phases
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def all_free(cls) -> "DesignFreeze":
        """Phase 1: concept exploration — everything is optimizable."""
        return cls({})

    @classmethod
    def for_ter27_phase2(cls) -> "DesignFreeze":
        """
        Phase 2: hardpoints locked from CAD, alignment still free.

        Frozen (set by physical geometry, can't change at track):
          - castor_f:    determined by UCA fore/aft offset
          - anti_dive_f/r: determined by wishbone pickup heights
          - anti_squat:  determined by rear SVSA geometry
          - anti_lift:   determined by rear side-view under decel
          - bump_steer_f/r: determined by rack height vs tie rod arc
          - h_cg:        determined by packaging

        Still free (adjustable at track or via shims):
          - camber_f/r:  shimmed at upper wishbone
          - toe_f/r:     tie rod length
          - springs, dampers, ARBs, ride height, brake bias
        """
        return cls({
            'castor_f':     5.5,     # deg — from CAD
            'anti_squat':   0.35,    # frac — from rear hardpoints
            'anti_dive_f':  0.35,    # frac — from front hardpoints
            'anti_dive_r':  0.15,    # frac — from rear hardpoints
            'anti_lift':    0.20,    # frac — from rear side-view
            'bump_steer_f': 0.000,   # rad/m — rack height set
            'bump_steer_r': 0.000,   # rad/m — toe link mount set
            'h_cg':         0.310,   # m — packaging locked
        })

    @classmethod
    def for_ter27_phase3(cls) -> "DesignFreeze":
        """
        Phase 3: prototype built, only track-tuning params free.

        Everything from Phase 2 frozen PLUS camber and toe locked
        (alignment set on the car). Only springs, dampers, ARBs,
        ride height, and brake bias remain free.
        """
        return cls({
            # Phase 2 locks
            'castor_f':     5.5,
            'anti_squat':   0.35,
            'anti_dive_f':  0.35,
            'anti_dive_r':  0.15,
            'anti_lift':    0.20,
            'bump_steer_f': 0.000,
            'bump_steer_r': 0.000,
            'h_cg':         0.310,
            # Phase 3 additional locks
            'camber_f':    -2.5,     # deg — alignment set
            'camber_r':    -1.8,     # deg
            'toe_f':       -0.08,    # deg — tie rods set
            'toe_r':        0.00,    # deg
            'diff_lock_ratio': 0.0,  # 4WD: pure electronic TV
        })

    @classmethod
    def custom(cls, freeze_names: List[str], values: Dict[str, float]) -> "DesignFreeze":
        """
        Convenience: freeze only the named params, pulling values from
        a provided dict (e.g., from a previous optimizer run's output).
        """
        frozen = {n: values[n] for n in freeze_names if n in values}
        return cls(frozen)

    # ─────────────────────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable table of frozen vs free parameters."""
        lines = []
        lines.append(f"{'═' * 62}")
        lines.append(f"  DESIGN FREEZE  ·  {self.n_frozen} frozen  ·  "
                      f"{self.n_free} free  ·  {SETUP_DIM} total")
        lines.append(f"{'═' * 62}")
        lines.append(f"  {'idx':<4s} {'Parameter':<22s} {'Status':<10s} {'Value':>10s}")
        lines.append(f"  {'─' * 56}")

        for i, name in enumerate(SETUP_NAMES):
            if name in self._frozen:
                val_str = f"{self._frozen[name]:>10.4f}"
                status = "🔒 FROZEN"
            else:
                val_str = "     ─    "
                status = "🔓 FREE  "
            lines.append(f"  {i:<4d} {name:<22s} {status:<10s} {val_str}")

        lines.append(f"{'═' * 62}")
        lines.append(f"  Effective optimization dimensionality: {self.n_free}D "
                      f"(of {SETUP_DIM}D)")
        lines.append(f"{'═' * 62}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"DesignFreeze(n_frozen={self.n_frozen}, "
                f"n_free={self.n_free}, "
                f"frozen={self.frozen_names})")


# ─────────────────────────────────────────────────────────────────────────────
# §2  Integration helper: install freeze into optimizer
# ─────────────────────────────────────────────────────────────────────────────

import jax  # needed for lax.stop_gradient in enforce()


def install_freeze(freeze: DesignFreeze, car_id: str = 'ter27'):
    """
    Installs the DesignFreeze into the vehicle_dynamics module namespace.

    After calling this:
      - SETUP_LB/SETUP_UB have frozen params clamped
      - project_to_bounds() automatically pins frozen params
      - Optimizer gradients for frozen params are zero

    Call BEFORE importing evolutionary.py.

    Args:
        freeze: DesignFreeze instance
        car_id: Car to get design bounds from
    """
    import models.vehicle_dynamics as vd
    from data.configs.car_config import get_design_bounds

    # Get design-mode bounds, then apply freeze
    design_lb, design_ub = get_design_bounds(car_id)
    frozen_lb, frozen_ub = freeze.apply_bounds(design_lb, design_ub)

    vd.SETUP_LB = frozen_lb
    vd.SETUP_UB = frozen_ub

    print(freeze.summary())
    print(f"\n[DesignFreeze] Installed into vehicle_dynamics module.")
    print(f"  Optimizer will search in {freeze.n_free}D subspace.")

    return freeze.gradient_mask()