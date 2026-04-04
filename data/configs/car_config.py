# data/configs/car_config.py
# Project-GP  —  Multi-Car Configuration Loader
# ═══════════════════════════════════════════════════════════════════════════════
# Single entry point for selecting the active car configuration.
# The physics engine, optimizer, and dashboard all consume parameters
# through this module — never import vehicle_params directly in new code.
#
# USAGE:
#   from data.configs.car_config import get_car_config, get_design_bounds
#
#   cfg = get_car_config('ter27')   # or 'ter26'
#   vp  = cfg['vehicle_params']
#   lb, ub = get_design_bounds('ter27')
#
# DESIGN MODE vs TUNING MODE:
#   Tuning mode  = existing car, narrow bounds (what to run at the track)
#   Design mode  = new car, wide bounds (what geometry to build into the car)
#   The optimizer doesn't change — only the bounds and objectives shift.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Dict, Tuple, Any
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# §1  Car Registry
# ─────────────────────────────────────────────────────────────────────────────

_CAR_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _register_ter26():
    from data.configs.vehicle_params import vehicle_params
    _CAR_REGISTRY['ter26'] = {
        'vehicle_params': vehicle_params,
        'drivetrain':     'rwd',
        'setup_dim':      28,              # Current production SuspensionSetup
        'season':         2026,
        'label':          'Ter26 RWD',
    }


def _register_ter27():
    from data.configs.vehicle_params_ter27 import vehicle_params_ter27
    _CAR_REGISTRY['ter27'] = {
        'vehicle_params': vehicle_params_ter27,
        'drivetrain':     'awd',
        'setup_dim':      28,              # Same 28-dim SuspensionSetup for now
        'season':         2027,
        'label':          'Ter27 4WD',
    }


def get_car_config(car_id: str = 'ter26') -> Dict[str, Any]:
    """
    Returns the full configuration dict for the requested car.

    Lazy-loads on first access to avoid circular imports.
    """
    car_id = car_id.lower().strip()
    if car_id not in _CAR_REGISTRY:
        # Lazy registration
        if car_id == 'ter26':
            _register_ter26()
        elif car_id == 'ter27':
            _register_ter27()
        else:
            raise ValueError(
                f"Unknown car_id '{car_id}'. Available: 'ter26', 'ter27'."
            )
    return _CAR_REGISTRY[car_id]


def list_cars() -> list[str]:
    """Returns all registered car IDs."""
    # Ensure both are loaded
    for cid in ('ter26', 'ter27'):
        if cid not in _CAR_REGISTRY:
            get_car_config(cid)
    return list(_CAR_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# §2  Design-Mode Bounds
# ─────────────────────────────────────────────────────────────────────────────
#
# The 28-dim SuspensionSetup already contains the key geometry parameters:
#
#   idx  name           TUNING mode (Ter26)    DESIGN mode (Ter27)
#   ───  ─────────────  ─────────────────────  ──────────────────────
#   14   camber_f       [-5.0, 0.0] deg        [-6.0, 0.0] deg
#   15   camber_r       [-5.0, 0.0] deg        [-5.0, 0.0] deg
#   16   toe_f          [-1.0, 1.0] deg        [-2.0, 2.0] deg  ← wider
#   17   toe_r          [-1.0, 1.0] deg        [-2.0, 2.0] deg  ← wider
#   18   castor_f       [0.0, 10.0] deg        [2.0, 12.0] deg  ← wider
#   19   anti_squat     [0.0, 0.9]             [0.0, 0.95]      ← wider
#   20   anti_dive_f    [0.0, 0.9]             [0.0, 0.95]
#   21   anti_dive_r    [0.0, 0.9]             [0.0, 0.95]
#   22   anti_lift      [0.0, 0.9]             [0.0, 0.95]
#   26   bump_steer_f   [-0.05, 0.05] rad/m    [-0.08, 0.08]    ← wider
#   27   bump_steer_r   [-0.05, 0.05] rad/m    [-0.08, 0.08]
#
# Spring/damper bounds also widen — 4WD changes optimal stiffness targets.
#
# Inner toe / outer toe: these are NOT separate parameters. They are a
# CONSEQUENCE of static_toe + ackermann_factor + steering angle.
#   inner_toe(δ) = static_toe + δ × (1 + ackermann_factor × f(geometry))
#   outer_toe(δ) = static_toe + δ × (1 - ackermann_factor × f(geometry))
# The optimizer finds optimal (static_toe, ackermann_factor) pairs — from
# which the team derives inner/outer toe at any steering angle.


def get_tuning_bounds() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Standard SETUP_LB / SETUP_UB for track-day tuning (narrow, existing car).
    These are the production bounds from vehicle_dynamics.py.
    """
    from models.vehicle_dynamics import SETUP_LB, SETUP_UB
    return SETUP_LB, SETUP_UB


def get_design_bounds(car_id: str = 'ter27') -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Widened bounds for suspension geometry DESIGN exploration.

    Returns (design_lb, design_ub) as float32 arrays of shape (28,).

    The bounds are wider than tuning mode because we're exploring what
    geometry the car SHOULD have, not what settings to run on an existing car.
    """
    # Start from production bounds
    from models.vehicle_dynamics import SETUP_LB, SETUP_UB
    lb = SETUP_LB.copy()
    ub = SETUP_UB.copy()

    if car_id == 'ter27':
        cfg = get_car_config('ter27')
        vp = cfg['vehicle_params']

        # ── Springs: 4WD shifts optimal front/rear balance ─────────────────
        # Front springs: need stiffer range (heavier unsprung mass)
        lb = lb.at[0].set(15000.0)     # k_f lower (explore softer too)
        ub = ub.at[0].set(140000.0)    # k_f upper
        lb = lb.at[1].set(15000.0)     # k_r lower
        ub = ub.at[1].set(130000.0)    # k_r upper (lighter rear now)

        # ── ARBs: wider range for load transfer distribution study ─────────
        ub = ub.at[2].set(6000.0)      # arb_f upper (stiffer option)
        ub = ub.at[3].set(5000.0)      # arb_r upper

        # ── Dampers: higher range for heavier front ────────────────────────
        ub = ub.at[4].set(10000.0)     # c_low_f upper
        ub = ub.at[5].set(9000.0)      # c_low_r upper
        ub = ub.at[6].set(5000.0)      # c_high_f upper
        ub = ub.at[7].set(4500.0)      # c_high_r upper

        # ── Camber: wider exploration for 4WD traction optimisation ────────
        lb = lb.at[14].set(-6.0)       # camber_f lower (more aggressive)
        # camber_r stays [-5.0, 0.0]

        # ── Toe: wider for geometry design ─────────────────────────────────
        # Design mode: we need to find the sweet spot, not just ±1°.
        lb = lb.at[16].set(-2.0)       # toe_f: -2.0° (heavy toe-in)
        ub = ub.at[16].set(2.0)        # toe_f: +2.0° (toe-out)
        lb = lb.at[17].set(-2.0)       # toe_r
        ub = ub.at[17].set(2.0)        # toe_r

        # ── Caster: wider for steering feel exploration ────────────────────
        lb = lb.at[18].set(2.0)        # castor_f lower
        ub = ub.at[18].set(12.0)       # castor_f upper

        # ── Anti-geometry: critical for 4WD — widen to near-maximum ────────
        ub = ub.at[19].set(0.95)       # anti_squat — explore up to 95%
        ub = ub.at[20].set(0.95)       # anti_dive_f
        ub = ub.at[21].set(0.95)       # anti_dive_r
        ub = ub.at[22].set(0.95)       # anti_lift

        # ── Bump steer: wider for rack height optimisation ─────────────────
        lb = lb.at[26].set(-0.08)      # bump_steer_f
        ub = ub.at[26].set(0.08)
        lb = lb.at[27].set(-0.08)      # bump_steer_r
        ub = ub.at[27].set(0.08)

        # ── Brake bias: more range with regen braking on all 4 wheels ──────
        lb = lb.at[24].set(0.40)       # brake_bias_f lower (more rear-biased)
        ub = ub.at[24].set(0.75)       # upper

        # ── CG height: explore packaging options ───────────────────────────
        lb = lb.at[25].set(0.25)       # h_cg lower (aggressive low CG)
        ub = ub.at[25].set(0.40)       # upper

    return lb, ub


# ─────────────────────────────────────────────────────────────────────────────
# §3  Geometry Parameter Map
# ─────────────────────────────────────────────────────────────────────────────
# Maps the optimizer output back to quantities the suspension team needs.
# This is a documentation + utility layer — call after optimizer converges.

GEOMETRY_PARAM_MAP = {
    # idx: (name, unit, description, how_to_implement)
    14: ('camber_f',     'deg',   'Front static camber',
         'Shim thickness at upper wishbone pickup → set on alignment rig'),
    15: ('camber_r',     'deg',   'Rear static camber',
         'Shim thickness at upper wishbone pickup'),
    16: ('toe_f',        'deg',   'Front static toe (per wheel)',
         'Tie rod length adjustment. Negative = toe-in. '
         'Combined with ackermann_factor for inner/outer toe split.'),
    17: ('toe_r',        'deg',   'Rear static toe (per wheel)',
         'Rear toe link length. Negative = toe-in.'),
    18: ('castor_f',     'deg',   'Front caster angle',
         'Upper wishbone fore/aft pickup offset. Affects mechanical trail, '
         'self-centering torque, and camber-in-turn gain.'),
    19: ('anti_squat',   'frac',  'Anti-squat percentage (rear)',
         'Side-view swing arm angle: SVSA = arctan(anti_squat × h_cg / wb). '
         'Set by lower wishbone pickup heights.'),
    20: ('anti_dive_f',  'frac',  'Front anti-dive percentage',
         'Front side-view geometry. Set by LCA/UCA fore-aft inclination. '
         'Higher = less nose dive under braking = more consistent aero platform.'),
    21: ('anti_dive_r',  'frac',  'Rear anti-dive percentage',
         'Rear side-view geometry under braking.'),
    22: ('anti_lift',    'frac',  'Anti-lift percentage (rear under decel)',
         'Controls rear squat/lift during regenerative braking — critical for '
         '4WD where front regen induces front lift.'),
    24: ('brake_bias_f', 'frac',  'Front brake bias',
         'Brake balance bar position. 4WD with regen: bias shifts dynamically.'),
    26: ('bump_steer_f', 'rad/m', 'Front bump steer coefficient',
         'Steering rack height relative to outer tie rod pickup. '
         'Target: ≈0 for zero bump steer. Non-zero = track rod arc mismatch.'),
    27: ('bump_steer_r', 'rad/m', 'Rear bump steer coefficient',
         'Rear toe link mount height. Non-zero indicates packaging compromise.'),
}


def format_optimizer_output(setup_vec, car_id: str = 'ter27') -> str:
    """
    Formats the optimizer's raw 28-element setup vector into a human-readable
    geometry recommendation for the suspension design team.

    Returns a formatted string suitable for a design review meeting.
    """
    from models.vehicle_dynamics import SETUP_NAMES

    lines = []
    lines.append(f"{'═' * 60}")
    lines.append(f"  PROJECT-GP  ·  GEOMETRY RECOMMENDATION  ·  {car_id.upper()}")
    lines.append(f"{'═' * 60}")
    lines.append("")

    # Geometry section (the parameters the suspension team cares about)
    lines.append("  SUSPENSION GEOMETRY TARGETS")
    lines.append(f"  {'─' * 50}")
    for idx, (name, unit, desc, impl) in GEOMETRY_PARAM_MAP.items():
        val = float(setup_vec[idx])
        if unit == 'deg':
            lines.append(f"  {desc:<35s}  {val:>+8.3f} {unit}")
        elif unit == 'frac':
            lines.append(f"  {desc:<35s}  {val:>8.1%}")
        else:
            lines.append(f"  {desc:<35s}  {val:>+8.5f} {unit}")
    lines.append("")

    # Derived quantities
    vp = get_car_config(car_id)['vehicle_params']
    wb = vp.get('wheelbase', vp.get('wb', 1.55))
    h_cg = float(setup_vec[25])

    anti_squat_val = float(setup_vec[19])
    svsa_r = jnp.degrees(jnp.arctan(anti_squat_val * h_cg / wb))

    anti_dive_f_val = float(setup_vec[20])
    svsa_f_brake = jnp.degrees(jnp.arctan(anti_dive_f_val * h_cg / wb))

    lines.append("  DERIVED QUANTITIES")
    lines.append(f"  {'─' * 50}")
    lines.append(f"  {'Rear SVSA (drive):':<35s}  {float(svsa_r):>+8.2f} deg")
    lines.append(f"  {'Front SVSA (brake):':<35s}  {float(svsa_f_brake):>+8.2f} deg")
    lines.append(f"  {'CG height:':<35s}  {h_cg * 1000:>8.1f} mm")
    lines.append("")

    # Implementation notes
    lines.append("  IMPLEMENTATION NOTES")
    lines.append(f"  {'─' * 50}")
    for idx, (name, unit, desc, impl) in GEOMETRY_PARAM_MAP.items():
        lines.append(f"  {desc}:")
        lines.append(f"    → {impl}")
    lines.append("")
    lines.append(f"{'═' * 60}")

    return "\n".join(lines)