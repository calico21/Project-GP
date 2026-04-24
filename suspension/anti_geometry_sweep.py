# suspension/anti_geometry_sweep.py
# Project-GP — Batch 10.5: Dynamic Brake-Bias Anti-Geometry Validation
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM:
#   sweep_analysis.py and the MORL optimizer evaluate anti-dive/anti-squat
#   at a FIXED brake bias (typically 60/40 F/R). But regen_blend.py
#   dynamically shifts the effective brake bias from 100% front to 100% rear
#   depending on motor temperature, SoC, and recovery targets.
#
#   If the optimizer finds suspension hardpoints optimized for 60/40, the car
#   will violently squat or dive when the regen bias shifts.
#
# SOLUTION:
#   Evaluate anti-geometry over a swept range of brake biases (0% → 100% front).
#   The MORL fitness includes a WORST-CASE anti-geometry penalty across the
#   entire bias range, forcing the optimizer to find hardpoints that maintain
#   aero platform stability regardless of what the Powertrain Manager does.
#
# INTEGRATION:
#   Called from optimizer_patch.py's kinematic consistency penalty.
#   Returns a scalar penalty that is smooth (C∞) and differentiable.
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial


class AntiGeometryResult(NamedTuple):
    """Anti-geometry metrics at a specific brake bias."""
    anti_dive_f:  jax.Array   # front anti-dive fraction [-]
    anti_dive_r:  jax.Array   # rear anti-dive fraction [-]
    anti_squat_f: jax.Array   # front anti-squat fraction [-] (4WD)
    anti_squat_r: jax.Array   # rear anti-squat fraction [-]
    anti_lift:    jax.Array   # rear anti-lift fraction [-]
    pitch_gain:   jax.Array   # pitch angle per G of decel [deg/G]


class AntiGeometrySweepResult(NamedTuple):
    """Swept anti-geometry across brake bias range."""
    bias_range:    jax.Array   # (N,) brake bias values [0,1]
    anti_dive_f:   jax.Array   # (N,) front anti-dive
    anti_dive_r:   jax.Array   # (N,) rear anti-dive
    anti_squat_f:  jax.Array   # (N,) front anti-squat
    anti_squat_r:  jax.Array   # (N,) rear anti-squat
    pitch_gain:    jax.Array   # (N,) pitch gain
    worst_pitch:   jax.Array   # scalar: max absolute pitch gain
    worst_bias:    jax.Array   # scalar: brake bias at worst pitch


def compute_anti_geometry_at_bias(
    ic_f_z: jax.Array,
    ic_r_z: jax.Array,
    h_cg: jax.Array,
    lf: jax.Array,
    lr: jax.Array,
    brake_bias_f: jax.Array,
    drive_bias_f: jax.Array = jnp.array(0.5),  # 4WD: 50/50 default
) -> AntiGeometryResult:
    """
    Compute anti-pitch geometry at a specific brake/drive bias.

    Anti-dive (braking):
      Front: anti_dive_f = IC_f_z / (h_cg · brake_bias_f · L/lf)
      Rear:  anti_dive_r = IC_r_z / (h_cg · (1-brake_bias_f) · L/lr)

    Anti-squat (acceleration, 4WD):
      Front: anti_squat_f = IC_f_z / (h_cg · drive_bias_f · L/lf)
      Rear:  anti_squat_r = IC_r_z / (h_cg · (1-drive_bias_f) · L/lr)

    Args:
        ic_f_z: front side-view instant centre height [m]
        ic_r_z: rear side-view instant centre height [m]
        h_cg: CG height [m]
        lf: CG to front axle [m]
        lr: CG to rear axle [m]
        brake_bias_f: fraction of braking force on front [0,1]
        drive_bias_f: fraction of drive force on front [0,1]

    Returns:
        AntiGeometryResult
    """
    L = lf + lr
    eps = 1e-6

    # Anti-dive under braking
    ad_f = ic_f_z / (h_cg * jnp.maximum(brake_bias_f, eps) * L / (lf + eps) + eps)
    ad_r = ic_r_z / (h_cg * jnp.maximum(1.0 - brake_bias_f, eps) * L / (lr + eps) + eps)

    # Anti-squat under acceleration (4WD)
    as_f = ic_f_z / (h_cg * jnp.maximum(drive_bias_f, eps) * L / (lf + eps) + eps)
    as_r = ic_r_z / (h_cg * jnp.maximum(1.0 - drive_bias_f, eps) * L / (lr + eps) + eps)

    # Anti-lift
    al = ic_r_z / (h_cg + eps)

    # Pitch gain: residual pitch after anti-geometry [deg/G]
    # At 1G decel: θ_pitch = (1 - anti_dive_total) · m·g·h_cg / K_pitch
    # Simplified: pitch_gain ∝ (1 - weighted_anti_dive)
    anti_dive_total = brake_bias_f * ad_f + (1.0 - brake_bias_f) * ad_r
    pitch_gain = (1.0 - jnp.clip(anti_dive_total, 0.0, 1.5)) * 2.0  # deg/G approx

    return AntiGeometryResult(
        anti_dive_f=ad_f,
        anti_dive_r=ad_r,
        anti_squat_f=as_f,
        anti_squat_r=as_r,
        anti_lift=al,
        pitch_gain=pitch_gain,
    )


def sweep_anti_geometry(
    ic_f_z: jax.Array,
    ic_r_z: jax.Array,
    h_cg: jax.Array,
    lf: jax.Array,
    lr: jax.Array,
    n_points: int = 21,
    drive_bias_f: jax.Array = jnp.array(0.5),
) -> AntiGeometrySweepResult:
    """
    Evaluate anti-geometry across the full brake bias range [0, 1].

    This is the core function that prevents the optimizer from building
    a car that only works at one brake bias. Called from the MORL
    kinematic consistency penalty.

    Args:
        ic_f_z: front side-view IC height [m]
        ic_r_z: rear side-view IC height [m]
        h_cg: CG height [m]
        lf: CG to front [m]
        lr: CG to rear [m]
        n_points: number of bias points to evaluate
        drive_bias_f: drive torque split (4WD)

    Returns:
        AntiGeometrySweepResult with worst-case metrics
    """
    bias_range = jnp.linspace(0.05, 0.95, n_points)  # avoid 0/1 singularities

    results = jax.vmap(
        lambda b: compute_anti_geometry_at_bias(
            ic_f_z, ic_r_z, h_cg, lf, lr, b, drive_bias_f
        )
    )(bias_range)

    # Worst-case pitch gain (maximum absolute pitch excursion)
    abs_pitch = jnp.abs(results.pitch_gain)
    worst_idx = jnp.argmax(abs_pitch)

    return AntiGeometrySweepResult(
        bias_range=bias_range,
        anti_dive_f=results.anti_dive_f,
        anti_dive_r=results.anti_dive_r,
        anti_squat_f=results.anti_squat_f,
        anti_squat_r=results.anti_squat_r,
        pitch_gain=results.pitch_gain,
        worst_pitch=abs_pitch[worst_idx],
        worst_bias=bias_range[worst_idx],
    )


def anti_geometry_penalty(
    ic_f_z: jax.Array,
    ic_r_z: jax.Array,
    h_cg: jax.Array,
    lf: jax.Array,
    lr: jax.Array,
    max_pitch_gain: float = 1.5,   # deg/G threshold
    penalty_weight: float = 10.0,
    drive_bias_f: jax.Array = jnp.array(0.5),
) -> jax.Array:
    """
    Smooth penalty for anti-geometry violation across all brake biases.

    Returns a scalar penalty ≥ 0 that is:
      0   if pitch gain ≤ max_pitch_gain at ALL bias points
      >0  if any bias point exceeds the threshold

    The penalty is C∞ smooth (softplus) — safe inside jit/grad.

    Use this in the MORL objective:
      total_cost = lap_time + anti_geometry_penalty(...)

    Args:
        max_pitch_gain: maximum acceptable pitch rate [deg/G]
        penalty_weight: Lagrangian multiplier

    Returns:
        Scalar penalty
    """
    sweep = sweep_anti_geometry(ic_f_z, ic_r_z, h_cg, lf, lr,
                                 drive_bias_f=drive_bias_f)

    # Soft max over all bias points: smooth approximation of max(pitch_gains)
    # Using log-sum-exp with temperature β=5
    beta = 5.0
    soft_max_pitch = jnp.log(
        jnp.sum(jnp.exp(beta * sweep.pitch_gain))
    ) / beta

    # Penalty: softplus(pitch - threshold) → 0 when safe, grows when violated
    violation = jax.nn.softplus(soft_max_pitch - max_pitch_gain)

    return penalty_weight * violation


# ─────────────────────────────────────────────────────────────────────────────
# §2  Integration with SuspensionKinematics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic_heights_from_kinematics(
    front_kin,   # SuspensionKinematics
    rear_kin,    # SuspensionKinematics
) -> Tuple[jax.Array, jax.Array]:
    """
    Extract side-view instant centre heights from kinematic solvers.

    The IC is computed from the A-arm pivot geometry in the XZ plane
    (side view). This gives the anti-geometry angles.

    Returns:
        (ic_f_z, ic_r_z) — instant centre heights [m]
    """
    def _ic_height(kin) -> float:
        hpts = kin.hpts
        # Side-view IC from lower and upper A-arm projections
        A1 = hpts["CHAS_LowFor"]
        A2 = hpts["CHAS_LowAft"]
        B1 = hpts["CHAS_UppFor"]
        B2 = hpts["CHAS_UppAft"]

        # Project to XZ plane (side view)
        dLA_x = A2[0] - A1[0]; dLA_z = A2[2] - A1[2]
        dUA_x = B2[0] - B1[0]; dUA_z = B2[2] - B1[2]

        import numpy as np
        M = np.array([[dLA_x, -dUA_x], [dLA_z, -dUA_z]])
        b = np.array([B1[0] - A1[0], B1[2] - A1[2]])
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) < 1e-8:
            return 0.0
        s = (b[0] * M[1, 1] - b[1] * M[0, 1]) / det
        ic_z = float(A1[2] + s * dLA_z)
        return ic_z

    return jnp.array(_ic_height(front_kin)), jnp.array(_ic_height(rear_kin))