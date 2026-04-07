#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# optimization/differentiable_track.py
# Project-GP — JAX-Native Differentiable Track Representation
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Provide a fully JAX-traced track representation that supports:
#     jax.grad(lap_time)(setup_vector)
#
#   The existing track_builder.py uses NumPy → not differentiable.
#   This module wraps track geometry as JAX arrays with differentiable
#   interpolation (cubic Hermite), enabling the full-lap sim to propagate
#   gradients from lap_time back through the track-following controller
#   to the setup parameters.
#
# DESIGN:
#   Track = {s, x, y, ψ, κ, w_left, w_right} sampled at N equidistant nodes.
#   Queries at arbitrary arc-length s* use cubic Hermite interpolation
#   (C¹ continuous, O(h⁴) accuracy, fully differentiable).
#
#   The track itself is NOT optimised — it's a fixed input. But the
#   interpolation must be differentiable because the vehicle's position
#   on the track depends on the state trajectory, which depends on setup.
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# §1  Track Data Container
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableTrack(NamedTuple):
    """
    Equidistant-sampled track geometry — all JAX arrays.

    Fields (all shape (N,)):
        s       : arc-length [m], monotonically increasing
        x       : centreline x [m]
        y       : centreline y [m]
        psi     : heading [rad], unwrapped
        kappa   : curvature [1/m], positive = left turn
        w_left  : distance to left boundary [m]
        w_right : distance to right boundary [m]
        total_length : scalar, total track length [m]
    """
    s:       jax.Array
    x:       jax.Array
    y:       jax.Array
    psi:     jax.Array
    kappa:   jax.Array
    w_left:  jax.Array
    w_right: jax.Array
    total_length: jax.Array


# ─────────────────────────────────────────────────────────────────────────────
# §2  Differentiable Interpolation
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def interp_track_at_s(track: DifferentiableTrack, s_query: jax.Array) -> dict:
    """
    Query track geometry at arbitrary arc-length via cubic Hermite interpolation.

    Fully differentiable w.r.t. s_query (which depends on vehicle state,
    which depends on setup → gradient flows).

    Returns dict with scalar values: x, y, psi, kappa, w_left, w_right.
    """
    s = track.s
    N = s.shape[0]
    ds = s[1] - s[0]  # uniform spacing

    # Wrap s_query to [0, total_length) for closed tracks
    s_q = jnp.mod(s_query, track.total_length)

    # Find bracketing index (differentiable via soft floor)
    idx_float = s_q / ds
    idx = jnp.clip(jnp.floor(idx_float).astype(jnp.int32), 0, N - 2)
    t = idx_float - idx.astype(jnp.float32)  # local parameter [0, 1)

    # Cubic Hermite basis functions (C¹, exact at nodes)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2

    def _hermite_interp(values):
        """Cubic Hermite interpolation with Catmull-Rom tangent estimation."""
        v0 = values[idx]
        v1 = values[jnp.minimum(idx + 1, N - 1)]

        # Catmull-Rom tangents: m_k = (v_{k+1} - v_{k-1}) / 2
        # Boundary handling via clipping
        im1 = jnp.maximum(idx - 1, 0)
        ip2 = jnp.minimum(idx + 2, N - 1)
        m0 = (values[jnp.minimum(idx + 1, N - 1)] - values[im1]) * 0.5
        m1 = (values[ip2] - values[idx]) * 0.5

        return h00 * v0 + h10 * ds * m0 + h01 * v1 + h11 * ds * m1

    return {
        'x':       _hermite_interp(track.x),
        'y':       _hermite_interp(track.y),
        'psi':     _hermite_interp(track.psi),
        'kappa':   _hermite_interp(track.kappa),
        'w_left':  _hermite_interp(track.w_left),
        'w_right': _hermite_interp(track.w_right),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §3  Track Construction from NumPy Data
# ─────────────────────────────────────────────────────────────────────────────

def make_differentiable_track(
    track_dict: dict = None,
    track_name: str = 'fsg_autocross',
) -> DifferentiableTrack:
    """
    Convert a NumPy track dictionary (from track_builder.py or track_generator.py)
    into a DifferentiableTrack with JAX arrays.

    If track_dict is None, builds a preset track.
    """
    if track_dict is not None:
        return DifferentiableTrack(
            s=jnp.array(track_dict['s'], dtype=jnp.float32),
            x=jnp.array(track_dict.get('x', track_dict.get('cx', np.zeros(1))), dtype=jnp.float32),
            y=jnp.array(track_dict.get('y', track_dict.get('cy', np.zeros(1))), dtype=jnp.float32),
            psi=jnp.array(track_dict.get('psi', track_dict.get('cpsi', np.zeros(1))), dtype=jnp.float32),
            kappa=jnp.array(track_dict.get('k', track_dict.get('ck', np.zeros(1))), dtype=jnp.float32),
            w_left=jnp.array(track_dict.get('w_left', track_dict.get('width_left', np.full(1, 3.5))), dtype=jnp.float32),
            w_right=jnp.array(track_dict.get('w_right', track_dict.get('width_right', np.full(1, 3.5))), dtype=jnp.float32),
            total_length=jnp.array(float(track_dict.get('total_length', track_dict['s'][-1])), dtype=jnp.float32),
        )

    # ── Preset tracks ────────────────────────────────────────────────────────
    if track_name == 'fsg_autocross':
        return _build_fsg_autocross_jax()
    elif track_name == 'skidpad':
        return _build_skidpad_jax()
    else:
        return _build_fsg_autocross_jax()


def _build_fsg_autocross_jax(N: int = 500) -> DifferentiableTrack:
    """
    Procedural FSG autocross: ~750m, 12 corners, 3 sectors.
    Curvature profile integrated to produce (x, y, ψ).
    """
    total_length = 750.0
    ds = total_length / N
    s = np.linspace(0, total_length, N)

    # Curvature profile: sequence of straights + corners
    kappa = np.zeros(N)
    segments = [
        (0, 30, 0.0),         # straight start
        (30, 55, 0.12),       # hairpin L (R≈8.3m)
        (55, 75, 0.0),        # straight
        (75, 95, -0.09),      # right sweeper (R≈11m)
        (95, 115, 0.0),       # straight
        (115, 135, 0.15),     # tight L hairpin (R≈6.7m)
        (135, 150, 0.0),      # straight
        (150, 170, -0.12),    # medium R (R≈8.3m)
        (170, 190, 0.0),      # straight
        (190, 210, 0.10),     # chicane L
        (210, 225, -0.10),    # chicane R
        (225, 260, 0.0),      # straight
        (260, 310, -0.04),    # fast sweeper R (R≈25m)
        (310, 370, 0.0),      # long straight
        (370, 395, 0.13),     # hairpin L
        (395, 420, 0.0),      # straight
        (420, 445, -0.11),    # medium R
        (445, 490, 0.0),      # straight
        (490, 520, 0.08),     # gentle L (R≈12.5m)
        (520, 560, 0.0),      # straight
        (560, 590, -0.14),    # tight R
        (590, 640, 0.0),      # straight
        (640, 670, 0.10),     # penultimate L
        (670, 750, 0.0),      # final straight to finish
    ]

    for s_start, s_end, k_val in segments:
        mask = (s >= s_start) & (s < s_end)
        kappa[mask] = k_val

    # Smooth transitions (Gaussian filter equivalent via rolling average)
    kernel = np.ones(7) / 7
    kappa = np.convolve(kappa, kernel, mode='same')

    # Integrate curvature → heading → position
    psi = np.cumsum(kappa * ds)
    x = np.cumsum(np.cos(psi) * ds)
    y = np.cumsum(np.sin(psi) * ds)

    # Track width: 3.5m half-width (7m total, typical FSG)
    w = np.full(N, 3.5)

    return DifferentiableTrack(
        s=jnp.array(s, dtype=jnp.float32),
        x=jnp.array(x, dtype=jnp.float32),
        y=jnp.array(y, dtype=jnp.float32),
        psi=jnp.array(psi, dtype=jnp.float32),
        kappa=jnp.array(kappa, dtype=jnp.float32),
        w_left=jnp.array(w, dtype=jnp.float32),
        w_right=jnp.array(w, dtype=jnp.float32),
        total_length=jnp.array(total_length, dtype=jnp.float32),
    )


def _build_skidpad_jax(N: int = 200) -> DifferentiableTrack:
    """Skidpad: two circles R=9.125m, entry/exit straights."""
    R = 9.125
    total_length = 2 * np.pi * R * 2 + 30  # two full circles + straights
    s = np.linspace(0, total_length, N)
    ds = s[1] - s[0]

    kappa = np.zeros(N)
    straight_len = 15.0
    circle_len = 2 * np.pi * R

    # Entry straight → left circle → left circle → exit straight
    s1 = straight_len
    s2 = s1 + circle_len
    s3 = s2 + circle_len

    kappa[(s >= s1) & (s < s2)] = 1.0 / R
    kappa[(s >= s2) & (s < s3)] = 1.0 / R

    psi = np.cumsum(kappa * ds)
    x = np.cumsum(np.cos(psi) * ds)
    y = np.cumsum(np.sin(psi) * ds)

    return DifferentiableTrack(
        s=jnp.array(s, dtype=jnp.float32),
        x=jnp.array(x, dtype=jnp.float32),
        y=jnp.array(y, dtype=jnp.float32),
        psi=jnp.array(psi, dtype=jnp.float32),
        kappa=jnp.array(kappa, dtype=jnp.float32),
        w_left=jnp.full(N, 1.5, dtype=jnp.float32),
        w_right=jnp.full(N, 1.5, dtype=jnp.float32),
        total_length=jnp.array(total_length, dtype=jnp.float32),
    )