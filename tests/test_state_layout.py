# tests/test_state_layout.py

import jax.numpy as jnp

from physics.state_layout import (
    STATE_DIM,
    Q_DIM,
    V_DIM,
    THERMAL_DIM,
    SLIP_DIM,
    DAMPER_DIM,
    ELASTOKIN_DIM,
    Q_SLICE,
    V_SLICE,
    THERMAL_SLICE,
    SLIP_SLICE,
    DAMPER_SLICE,
    ELASTOKIN_SLICE,
    split_state,
    pack_state,
    state_name,
    state_index,
    validate_state,
)


def test_dimensions():
    assert Q_DIM == 14
    assert V_DIM == 14
    assert THERMAL_DIM == 28
    assert SLIP_DIM == 16
    assert DAMPER_DIM == 12
    assert ELASTOKIN_DIM == 24

    assert (
        Q_DIM
        + V_DIM
        + THERMAL_DIM
        + SLIP_DIM
        + DAMPER_DIM
        + ELASTOKIN_DIM
        == STATE_DIM
    )
    assert STATE_DIM == 108


def test_slices_cover_entire_state_without_overlap():
    x = jnp.arange(STATE_DIM)

    blocks = [
        x[Q_SLICE],
        x[V_SLICE],
        x[THERMAL_SLICE],
        x[SLIP_SLICE],
        x[DAMPER_SLICE],
        x[ELASTOKIN_SLICE],
    ]

    reconstructed = jnp.concatenate(blocks)

    assert reconstructed.shape == (108,)
    assert jnp.array_equal(reconstructed, x)


def test_split_pack_roundtrip():
    x = jnp.arange(STATE_DIM, dtype=jnp.float32)

    blocks = split_state(x)

    x_reconstructed = pack_state(
        blocks.q,
        blocks.v,
        blocks.thermal,
        blocks.slip,
        blocks.damper,
        blocks.elastokin,
    )

    assert jnp.allclose(x_reconstructed, x)


def test_block_shapes():
    x = jnp.zeros(STATE_DIM)
    blocks = split_state(x)

    assert blocks.q.shape == (14,)
    assert blocks.v.shape == (14,)
    assert blocks.thermal.shape == (28,)
    assert blocks.slip.shape == (16,)
    assert blocks.damper.shape == (12,)
    assert blocks.elastokin.shape == (24,)


def test_known_state_names():
    assert state_name(0) == "X"
    assert state_name(5) == "yaw"
    assert state_name(6) == "z_fl"

    assert state_name(14) == "vx"
    assert state_name(19) == "wz"
    assert state_name(24) == "omega_fl"

    assert state_name(28) == "FL_T_inner"
    assert state_name(55) == "RR_T_contact"

    assert state_name(56) == "FL_alpha_t"
    assert state_name(71) == "RR_kappa_dot"

    assert state_name(72) == "FL_F_branch_1"
    assert state_name(83) == "RR_T_oil"

    assert state_name(84) == "FL_bw_0"
    assert state_name(107) == "RR_bw_5"


def test_name_index_roundtrip():
    for i in range(STATE_DIM):
        assert state_index(state_name(i)) == i


def test_validate_state():
    validate_state(jnp.zeros(108))


def test_invalid_state_dimension():
    try:
        validate_state(jnp.zeros(107))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for wrong state dimension")