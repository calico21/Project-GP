"""
track_ingestor.py — BezierCurve.msg → JAX arc-length-parameterised spline

Converts the driverless path planner's BezierCurve output (cubic Bézier
polycurve) into the arc-length-parameterised spline representation consumed
by Project-GP's Diff-WMPC track reference.

Two input paths are supported:
  1. BezierCurve.msg  — from driverless path planner (preferred, full curvature)
  2. Track.msg gates  — fallback when path planner hasn't published yet;
                        uses centripetal Catmull-Rom to estimate control points.

All JAX functions are JIT-compiled and fully differentiable:
  ∂(spline outputs) / ∂(anchor positions) is well-defined — enables
  end-to-end gradient flow from lap time back to perception/planning.

Static shapes: MAX_TRACK_POINTS must bound the track length for XLA compilation.
  FS tracks: typically 100-400 gates. Set MAX_TRACK_POINTS = 512 conservatively.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from functools import partial
from typing import NamedTuple

N_SAMPLES_PER_SEG = 24   # Bézier sampling resolution (24 pts/seg → smooth at FS scale)
MAX_TRACK_POINTS  = 512  # Static upper bound on track gates for XLA static shapes


class SplineData(NamedTuple):
    """
    Arc-length-parameterised spline output consumed by Diff-WMPC.
    All arrays share the same leading dimension N (total sample count).
    """
    points:    jnp.ndarray  # [N, 2]  XY positions (m, REP-103)
    arc_len:   jnp.ndarray  # [N]     cumulative arc length (m)
    tangents:  jnp.ndarray  # [N, 2]  unit tangent vectors
    normals:   jnp.ndarray  # [N, 2]  unit inward normals (left of travel)
    curvature: jnp.ndarray  # [N]     signed curvature κ (rad/m, left-positive)
    total_len: jnp.ndarray  # ()      total track length (m)
    n_valid:   int          # number of valid (non-padded) samples


# ─── Bézier evaluation ────────────────────────────────────────────────────────

@jax.jit
def _cubic_bezier_batch(
    p0: jnp.ndarray,  # [2]
    p1: jnp.ndarray,  # [2]  = anchor_i + out_cp_i
    p2: jnp.ndarray,  # [2]  = anchor_{i+1} + in_cp_{i+1}
    p3: jnp.ndarray,  # [2]
    t:  jnp.ndarray,  # [N]  ∈ [0, 1]
) -> jnp.ndarray:     # [N, 2]
    """
    Vectorised De Casteljau evaluation.
    Formula: B(t) = Σ C(3,k) (1-t)^{3-k} t^k P_k
    Numerically identical to mirena/bezier_curve.hpp _bezier_interpolate().
    """
    omt  = 1.0 - t                          # [N]
    b0   = omt ** 3                         # [N]
    b1   = 3.0 * omt ** 2 * t              # [N]
    b2   = 3.0 * omt * t ** 2              # [N]
    b3   = t ** 3                           # [N]
    return (
        b0[:, None] * p0[None, :]
        + b1[:, None] * p1[None, :]
        + b2[:, None] * p2[None, :]
        + b3[:, None] * p3[None, :]
    )  # [N, 2]


@jax.jit
def _cubic_bezier_tangent_batch(
    p0: jnp.ndarray, p1: jnp.ndarray,
    p2: jnp.ndarray, p3: jnp.ndarray,
    t:  jnp.ndarray,
) -> jnp.ndarray:  # [N, 2]
    """Analytic first derivative B'(t) — avoids finite differences."""
    omt  = 1.0 - t
    d0   = 3.0 * omt ** 2              # [N]
    d1   = 6.0 * omt * t              # [N]
    d2   = 3.0 * t ** 2               # [N]
    return (
        d0[:, None] * (p1 - p0)[None, :]
        + d1[:, None] * (p2 - p1)[None, :]
        + d2[:, None] * (p3 - p2)[None, :]
    )  # [N, 2]


@jax.jit
def _cubic_bezier_curvature_batch(
    p0: jnp.ndarray, p1: jnp.ndarray,
    p2: jnp.ndarray, p3: jnp.ndarray,
    t:  jnp.ndarray,
) -> jnp.ndarray:  # [N] signed curvature
    """
    Analytic signed curvature κ = (x'y'' - y'x'') / (x'² + y'²)^{3/2}
    Second derivative B''(t) computed analytically for gradient cleanliness.
    """
    omt  = 1.0 - t
    # B''(t) = 6[(1-t)(P2-2P1+P0) + t(P3-2P2+P1)]
    a    = p2 - 2.0 * p1 + p0     # [2]
    b    = p3 - 2.0 * p2 + p1     # [2]
    d2   = 6.0 * (omt[:, None] * a[None, :] + t[:, None] * b[None, :])  # [N, 2]

    d1   = _cubic_bezier_tangent_batch(p0, p1, p2, p3, t)  # [N, 2]

    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]    # [N]
    speed = jnp.sum(d1 ** 2, axis=1) ** 1.5 + 1e-12       # [N]
    return cross / speed


# ─── Host-side message parsing (runs outside JAX trace) ───────────────────────

def bezier_msg_to_arrays(
    points_list,  # list of mirena_common/msg/BezierCurvePoint ROS2 messages
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses BezierCurve.msg points into three NumPy arrays.
    Run ONCE per message on the ROS2 callback thread (not inside JAX trace).

    BezierCurvePoint convention (from msg definition):
      .point            — absolute anchor position (world frame)
      .in_control_point  — RELATIVE to .point (incoming tangent side)
      .out_control_point — RELATIVE to .point (outgoing tangent side)

    Returns:
      anchors  [M, 2]  absolute XY of each anchor
      out_cps  [M, 2]  out_control_point relative to its anchor
      in_cps   [M, 2]  in_control_point  relative to its anchor
    """
    M        = len(points_list)
    anchors  = np.zeros((M, 2), dtype=np.float32)
    out_cps  = np.zeros((M, 2), dtype=np.float32)
    in_cps   = np.zeros((M, 2), dtype=np.float32)

    for i, pt in enumerate(points_list):
        anchors[i] = [pt.point.x,             pt.point.y            ]
        out_cps[i] = [pt.out_control_point.x, pt.out_control_point.y]
        in_cps[i]  = [pt.in_control_point.x,  pt.in_control_point.y ]

    return anchors, out_cps, in_cps


def gates_to_arrays(
    gates,         # list of mirena_common/msg/Gate messages (x, y, psi)
    closed: bool,
    alpha:  float = 0.5,  # centripetal Catmull-Rom exponent
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates Bézier control points from Gate[] gate centrelines via
    centripetal Catmull-Rom → cubic Bézier conversion.

    Centripetal parameterisation (α=0.5) avoids cusps and self-intersections
    even for non-uniform gate spacing — common in FS acceleration/skidpad layouts.
    """
    M       = len(gates)
    anchors = np.array([[g.x, g.y] for g in gates], dtype=np.float32)  # [M, 2]
    out_cps = np.zeros((M, 2), dtype=np.float32)
    in_cps  = np.zeros((M, 2), dtype=np.float32)
    eps     = 1e-8

    def _get(i: int) -> np.ndarray:
        return anchors[i % M] if closed else anchors[np.clip(i, 0, M - 1)]

    for i in range(M):
        p0, p1, p2, p3 = _get(i-1), _get(i), _get(i+1), _get(i+2)

        d01 = np.linalg.norm(p1 - p0) ** alpha + eps
        d12 = np.linalg.norm(p2 - p1) ** alpha + eps
        d23 = np.linalg.norm(p3 - p2) ** alpha + eps

        # Catmull-Rom tangent at p1 and p2 (centripetal parametrisation)
        t1 = (
            (p1 - p0) / d01
            - (p2 - p0) / (d01 + d12)
            + (p2 - p1) / d12
        ) * d12

        t2 = (
            (p2 - p1) / d12
            - (p3 - p1) / (d12 + d23)
            + (p3 - p2) / d23
        ) * d12

        # Catmull-Rom → Bézier: CP1 = P1 + t1/3, CP2 = P2 - t2/3
        out_cps[i] = t1 / 3.0    # relative to anchors[i]
        in_cps[i]  = -t2 / 3.0   # relative to anchors[i] (note: stored at i, used at i for "incoming into i+1")

    # Fix: in_cps as defined above are the incoming CP for the NEXT segment,
    # stored relative to anchors[i+1]. Shift by one for correct indexing.
    in_cps_shifted       = np.roll(in_cps, 1, axis=0)
    in_cps_shifted[0]    = in_cps[-1] if closed else np.zeros(2)

    return anchors, out_cps, in_cps_shifted


# ─── JAX-native spline construction ──────────────────────────────────────────

@partial(jax.jit, static_argnames=("n_samples",))
def build_spline(
    anchors: jnp.ndarray,   # [M, 2]  absolute XY anchors
    out_cps: jnp.ndarray,   # [M, 2]  outgoing control points, relative
    in_cps:  jnp.ndarray,   # [M, 2]  incoming control points, relative
    n_samples: int = N_SAMPLES_PER_SEG,
) -> SplineData:
    """
    Builds an arc-length-parameterised SplineData from a polycubic Bézier curve.

    Uses jax.lax.scan over segments for XLA compatibility (static M required
    for compilation; pad to MAX_TRACK_POINTS on host before calling).

    Outputs positions, unit tangents, inward normals, and signed curvature —
    all differentiable w.r.t. input anchor positions. This enables:
        jax.grad(lap_time)(anchor_positions)
    once Project-GP's lap simulation is wired to this spline.
    """
    M    = anchors.shape[0]
    t_v  = jnp.linspace(0.0, 1.0, n_samples)  # [N_samp] per segment

    def _scan_segment(carry, i):
        # Absolute control points for segment i → i+1
        p0 = anchors[i]
        p3 = anchors[(i + 1) % M]
        p1 = p0 + out_cps[i]                # absolute CP1
        p2 = p3 + in_cps[(i + 1) % M]      # absolute CP2

        pts   = _cubic_bezier_batch(p0, p1, p2, p3, t_v)          # [N, 2]
        tans  = _cubic_bezier_tangent_batch(p0, p1, p2, p3, t_v)  # [N, 2]
        kaps  = _cubic_bezier_curvature_batch(p0, p1, p2, p3, t_v)  # [N]
        return carry, (pts, tans, kaps)

    _, (all_pts, all_tans, all_kaps) = lax.scan(
        _scan_segment, None, jnp.arange(M),
    )  # [M, N, 2], [M, N, 2], [M, N]

    pts_flat  = all_pts.reshape(-1, 2)    # [M*N, 2]
    tans_flat = all_tans.reshape(-1, 2)   # [M*N, 2]
    kaps_flat = all_kaps.reshape(-1)      # [M*N]

    # Arc-length via cumulative chord norm (piecewise linear approximation)
    diffs      = jnp.diff(pts_flat, axis=0)                          # [M*N-1, 2]
    chord_lens = jnp.linalg.norm(diffs, axis=1)                      # [M*N-1]
    arc_len    = jnp.concatenate([jnp.zeros(1), jnp.cumsum(chord_lens)])  # [M*N]

    # Normalise tangents (analytic tangents may not be unit due to parameterisation)
    tan_norms = jnp.linalg.norm(tans_flat, axis=1, keepdims=True) + 1e-8
    unit_tans = tans_flat / tan_norms                                # [M*N, 2]

    # Inward normal = left-perpendicular of tangent (CCW convention, REP-103)
    # n = R(+90°) t = [-ty, tx]
    unit_norms = jnp.stack([-unit_tans[:, 1], unit_tans[:, 0]], axis=1)  # [M*N, 2]

    return SplineData(
        points    = pts_flat,
        arc_len   = arc_len,
        tangents  = unit_tans,
        normals   = unit_norms,
        curvature = kaps_flat,
        total_len = arc_len[-1],
        n_valid   = M * n_samples,
    )


@jax.jit
def query_at_s(
    spline: SplineData,
    s:      jnp.ndarray,  # () arc-length query (m), wraps if closed
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Differentiable spline query at arc length s.

    Linear interpolation between samples: sufficient for WMPC with ≥24 samp/seg,
    and maintains gradient w.r.t. s (needed for optimal progress objective).

    Returns: (position [2], unit_tangent [2], unit_normal [2], curvature ())
    """
    s_clamped = jnp.clip(s % (spline.total_len + 1e-8), 0.0, spline.total_len)
    arc       = spline.arc_len

    # Differentiable searchsorted via bisect (no hard branch, gradient-safe)
    idx       = jnp.clip(
        jnp.searchsorted(arc, s_clamped, side="right") - 1,
        0,
        arc.shape[0] - 2,
    )
    dt        = (s_clamped - arc[idx]) / (arc[idx + 1] - arc[idx] + 1e-12)
    w1, w0    = dt, 1.0 - dt

    pos   = w0 * spline.points[idx]    + w1 * spline.points[idx + 1]
    tan   = w0 * spline.tangents[idx]  + w1 * spline.tangents[idx + 1]
    nor   = w0 * spline.normals[idx]   + w1 * spline.normals[idx + 1]
    kap   = w0 * spline.curvature[idx] + w1 * spline.curvature[idx + 1]

    # Re-normalise after interpolation (interpolated unit vectors aren't unit)
    tan = tan / (jnp.linalg.norm(tan) + 1e-8)
    nor = nor / (jnp.linalg.norm(nor) + 1e-8)

    return pos, tan, nor, kap


@jax.jit
def project_car_onto_spline(
    spline:      SplineData,
    car_xy:      jnp.ndarray,  # [2]  car XY position
    s_guess:     jnp.ndarray,  # ()   previous progress estimate (warm-start)
    search_half: float = 10.0, # m    search window half-width around s_guess
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Projects car position onto spline: finds closest point via windowed search.

    Returns: (s_closest, lateral_error, heading_error)
      s_closest:     arc-length of closest spline point (m)
      lateral_error: signed cross-track error (m, +left of centreline)
      heading_error: signed heading error (rad, +CCW)

    Differentiable w.r.t. car_xy — enables online error gradient for WMPC.
    """
    # Windowed candidates: only search ±search_half around s_guess
    # This avoids the global argmin which is non-differentiable
    n_cands  = 64
    s_lo     = jnp.clip(s_guess - search_half, 0.0, spline.total_len)
    s_hi     = jnp.clip(s_guess + search_half, 0.0, spline.total_len)
    s_cands  = jnp.linspace(s_lo, s_hi, n_cands)  # [64]

    # Vectorised query
    query_v  = jax.vmap(lambda s_q: query_at_s(spline, s_q))
    pos_c, tan_c, nor_c, kap_c = query_v(s_cands)  # [64, 2], [64, 2], ...

    # Squared distance to each candidate
    d_sq     = jnp.sum((pos_c - car_xy[None, :]) ** 2, axis=1)  # [64]

    # Soft argmin for differentiability (temperature τ — lower=sharper)
    tau      = 0.5
    weights  = jax.nn.softmax(-d_sq / tau)                       # [64]
    s_best   = jnp.dot(weights, s_cands)                         # ()

    pos_b, tan_b, nor_b, _ = query_at_s(spline, s_best)

    # Cross-track: signed projection onto inward normal
    delta_xy     = car_xy - pos_b
    lateral_err  = jnp.dot(delta_xy, nor_b)

    # Heading: atan2 of car heading relative to spline tangent (caller provides psi)
    # Returned as (pos_b, nor_b) so caller can compute heading_err = psi - atan2(tan)
    return s_best, lateral_err, pos_b
