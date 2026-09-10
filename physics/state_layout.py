# physics/state_layout.py
# Project-GP — Canonical 108-state layout
# ═══════════════════════════════════════════════════════════════════════════════
#
# Single source of truth for the full vehicle state vector.
#
# IMPORTANT:
#   The stored mechanical state is [q, v], NOT [q, p].
#   Generalized momentum is derived in the dynamics through:
#
#       p = M(q/setup) · v
#
#   Therefore:
#       Q_SLICE = stored generalized coordinates
#       V_SLICE = stored generalized velocities
#       P_SLICE is intentionally NOT used as a storage slice.
#
# Full state:
#
#   x[0:14]     q                  generalized coordinates
#   x[14:28]    v                  generalized velocities
#   x[28:56]    thermal             4 × 7 tire thermal states
#   x[56:72]    tire transient      4 × 4 second-order tire states
#   x[72:84]    damper              4 × 3 damper states
#   x[84:108]   elastokinematic      4 × 6 Bouc-Wen states
#
# Total:
#   14 + 14 + 28 + 16 + 12 + 24 = 108
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# Dimensions
# ─────────────────────────────────────────────────────────────────────────────

Q_DIM = 14
V_DIM = 14
THERMAL_DIM = 28
SLIP_DIM = 16
DAMPER_DIM = 12
ELASTOKIN_DIM = 24

STATE_DIM = (
    Q_DIM
    + V_DIM
    + THERMAL_DIM
    + SLIP_DIM
    + DAMPER_DIM
    + ELASTOKIN_DIM
)

assert STATE_DIM == 108


# ─────────────────────────────────────────────────────────────────────────────
# Canonical slices
# ─────────────────────────────────────────────────────────────────────────────

Q_SLICE = slice(0, 14)
V_SLICE = slice(14, 28)
THERMAL_SLICE = slice(28, 56)
SLIP_SLICE = slice(56, 72)
DAMPER_SLICE = slice(72, 84)
ELASTOKIN_SLICE = slice(84, 108)


# Compatibility aliases for code that needs explicit storage terminology.
MECHANICAL_SLICE = slice(0, 28)
AUX_SLICE = slice(28, 108)


# ─────────────────────────────────────────────────────────────────────────────
# Per-block shapes
# ─────────────────────────────────────────────────────────────────────────────

THERMAL_SHAPE = (4, 7)
SLIP_SHAPE = (4, 4)
DAMPER_SHAPE = (4, 3)
ELASTOKIN_SHAPE = (4, 6)


# ─────────────────────────────────────────────────────────────────────────────
# State component names
# ─────────────────────────────────────────────────────────────────────────────

Q_NAMES = (
    "X",
    "Y",
    "Z",
    "roll",
    "pitch",
    "yaw",
    "z_fl",
    "z_fr",
    "z_rl",
    "z_rr",
    "theta_fl",
    "theta_fr",
    "theta_rl",
    "theta_rr",
)

V_NAMES = (
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "dz_fl",
    "dz_fr",
    "dz_rl",
    "dz_rr",
    "omega_fl",
    "omega_fr",
    "omega_rl",
    "omega_rr",
)

THERMAL_NAMES = tuple(
    f"{corner}_{node}"
    for corner in ("FL", "FR", "RL", "RR")
    for node in (
        "T_inner",
        "T_mid",
        "T_outer",
        "T_bulk",
        "T_carcass",
        "T_gas",
        "T_contact",
    )
)

SLIP_NAMES = tuple(
    f"{corner}_{state}"
    for corner in ("FL", "FR", "RL", "RR")
    for state in (
        "alpha_t",
        "alpha_dot",
        "kappa_t",
        "kappa_dot",
    )
)

DAMPER_NAMES = tuple(
    f"{corner}_{state}"
    for corner in ("FL", "FR", "RL", "RR")
    for state in (
        "F_branch_1",
        "F_branch_2",
        "T_oil",
    )
)

ELASTOKIN_NAMES = tuple(
    f"{corner}_bw_{i}"
    for corner in ("FL", "FR", "RL", "RR")
    for i in range(6)
)

STATE_NAMES = (
    Q_NAMES
    + V_NAMES
    + THERMAL_NAMES
    + SLIP_NAMES
    + DAMPER_NAMES
    + ELASTOKIN_NAMES
)

assert len(Q_NAMES) == Q_DIM
assert len(V_NAMES) == V_DIM
assert len(THERMAL_NAMES) == THERMAL_DIM
assert len(SLIP_NAMES) == SLIP_DIM
assert len(DAMPER_NAMES) == DAMPER_DIM
assert len(ELASTOKIN_NAMES) == ELASTOKIN_DIM
assert len(STATE_NAMES) == STATE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# State container
# ─────────────────────────────────────────────────────────────────────────────

class StateBlocks(NamedTuple):
    q: jax.Array
    v: jax.Array
    thermal: jax.Array
    slip: jax.Array
    damper: jax.Array
    elastokin: jax.Array


def split_state(x: jax.Array) -> StateBlocks:
    """
    Split the canonical 108-state vector.

    Returns:
        q:
            shape (14,)
        v:
            shape (14,)
        thermal:
            shape (28,)
        slip:
            shape (16,)
        damper:
            shape (12,)
        elastokin:
            shape (24,)
    """
    if x.ndim != 1:
        raise ValueError(
            f"split_state expects a 1-D state vector, got shape {x.shape}"
        )

    if x.shape[0] != STATE_DIM:
        raise ValueError(
            f"split_state expects state dimension {STATE_DIM}, "
            f"got {x.shape[0]}"
        )

    return StateBlocks(
        q=x[Q_SLICE],
        v=x[V_SLICE],
        thermal=x[THERMAL_SLICE],
        slip=x[SLIP_SLICE],
        damper=x[DAMPER_SLICE],
        elastokin=x[ELASTOKIN_SLICE],
    )


def pack_state(
    q: jax.Array,
    v: jax.Array,
    thermal: jax.Array,
    slip: jax.Array,
    damper: jax.Array,
    elastokin: jax.Array,
) -> jax.Array:
    """
    Pack canonical state blocks into the 108-state vector.
    """
    arrays = (
        q,
        v,
        thermal,
        slip,
        damper,
        elastokin,
    )

    expected = (
        Q_DIM,
        V_DIM,
        THERMAL_DIM,
        SLIP_DIM,
        DAMPER_DIM,
        ELASTOKIN_DIM,
    )

    for name, arr, dim in zip(
        (
            "q",
            "v",
            "thermal",
            "slip",
            "damper",
            "elastokin",
        ),
        arrays,
        expected,
    ):
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D, got shape {arr.shape}"
            )

        if arr.shape[0] != dim:
            raise ValueError(
                f"{name} must have dimension {dim}, "
                f"got {arr.shape[0]}"
            )

    return jnp.concatenate(arrays, axis=0)


def state_name(index: int) -> str:
    """
    Return the canonical human-readable name of a state component.
    """
    if not 0 <= index < STATE_DIM:
        raise IndexError(
            f"State index {index} outside [0, {STATE_DIM - 1}]"
        )
    return STATE_NAMES[index]


def state_index(name: str) -> int:
    """
    Return the canonical integer index corresponding to a state name.
    """
    try:
        return STATE_NAMES.index(name)
    except ValueError as exc:
        raise KeyError(f"Unknown state name: {name}") from exc


def validate_state(x: jax.Array) -> None:
    """
    Validate the basic structural contract of a state vector.

    This deliberately checks only structural properties.
    Physical validity is handled by the physics/audit layer.
    """
    if x.ndim != 1:
        raise ValueError(
            f"State must be 1-D, got shape {x.shape}"
        )

    if x.shape[0] != STATE_DIM:
        raise ValueError(
            f"State must have dimension {STATE_DIM}, "
            f"got {x.shape[0]}"
        )


__all__ = [
    "Q_DIM",
    "V_DIM",
    "THERMAL_DIM",
    "SLIP_DIM",
    "DAMPER_DIM",
    "ELASTOKIN_DIM",
    "STATE_DIM",
    "Q_SLICE",
    "V_SLICE",
    "THERMAL_SLICE",
    "SLIP_SLICE",
    "DAMPER_SLICE",
    "ELASTOKIN_SLICE",
    "MECHANICAL_SLICE",
    "AUX_SLICE",
    "THERMAL_SHAPE",
    "SLIP_SHAPE",
    "DAMPER_SHAPE",
    "ELASTOKIN_SHAPE",
    "Q_NAMES",
    "V_NAMES",
    "THERMAL_NAMES",
    "SLIP_NAMES",
    "DAMPER_NAMES",
    "ELASTOKIN_NAMES",
    "STATE_NAMES",
    "StateBlocks",
    "split_state",
    "pack_state",
    "state_name",
    "state_index",
    "validate_state",
]