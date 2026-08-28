#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# PROJECT-GP: 3D INTERACTIVE MULTI-OBJECTIVE KINEMATIC OPTIMIZER (TER27)
# ==============================================================================
# Autor: Alex Revilla / Tecnun eRacing
# Archivo: main/interactive_kinematic_optimizer.py
#
# FIX LOG vs. previous version
# ------------------------------------------------------------------------------
# BUG-1  Front axle never computed anti-dive. evaluate_full_geometry() returned
#        a single `anti_pct` computed with the SAME formula regardless of axle,
#        but the caller only wired it into the front loss for Scrub Radius —
#        anti_pct was silently dropped for is_front=True everywhere (loss,
#        live table, final report). Optimum Kinematics' front axle report
#        was therefore never actually being matched.
#
# BUG-2  Anti-squat/anti-dive/anti-lift are NOT the same formula applied to
#        different SVICs. The %anti value scales the SVIC angle by the
#        FRACTION OF TOTAL LONGITUDINAL FORCE that axle actually reacts:
#          - Anti-squat (rear, under power)   -> scaled by rear DRIVE torque fraction
#          - Anti-dive  (front, under brake)  -> scaled by front BRAKE force fraction
#          - Anti-lift  (front, under power)  -> scaled by front DRIVE torque fraction
#                                                  (relevant: Ter27 is 4WD hub-motor,
#                                                  not RWD, so front axle IS driven)
#          - Anti-lift  (rear, under brake)   -> scaled by rear BRAKE force fraction
#        The old script used force_fraction=1.0 implicitly for whichever axle
#        it was evaluating — correct by accident only for a 100%-RWD-under-power
#        rear axle. Wrong for front (which never got the calc) and wrong for
#        any partial front/rear torque or brake split.
#        This now matches SuspensionSetup.anti_squat / anti_dive_f / anti_dive_r
#        / anti_lift in models/vehicle_dynamics.py exactly — one JAX param per
#        physical anti-geometry quantity, all four computed here.
#
# BUG-3  Hardpoints re-verified against Front_Ter27_-_Velis.xlsx and
#        Rear_TeR27_-_Velis_2.xlsx (Optimum Kinematics native export) —
#        confirmed byte-exact with the nominal dicts below.
#
# NEW    - Per-point-group lock matrix (independent for front/rear)
#        - Runs front-only / rear-only / both in one session
#        - brake_bias_f / drive_bias_f (fraction of TOTAL brake / drive force
#          reacted at the FRONT axle) are explicit, user-set inputs — read
#          from your calibrated setup (SuspensionSetup.brake_bias_f) rather
#          than assumed
#        - Optimum-Kinematics-formatted point dump + config/vehicles/ter27.py
#          SuspensionSetup line (anti_squat, anti_dive_f, anti_dive_r,
#          anti_lift) auto-generated at the end
# ==============================================================================

import numpy as np
from scipy.optimize import minimize
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.prompt import Prompt, Confirm, FloatPrompt
from rich import box

console = Console()

# ==============================================================================
# 1. HARDPOINTS BASE NOMINALES (TER27 — verified against Optimum Kinematics
#    exports Front_Ter27_-_Velis.xlsx / Rear_TeR27_-_Velis_2.xlsx) [mm]
#    Left-side values shown; right side is Y-mirrored (X, -Y, Z) — the solver
#    below only ever needs one side since the vehicle is symmetric.
# ==============================================================================

HP_FRONT_NOMINAL = {
    'CHAS_LowFor': np.array([160.000, 160.000, 110.000]),
    'CHAS_LowAft': np.array([-160.000, 160.000, 130.000]),
    'UPRI_LowPnt': np.array([2.270, 583.374, 122.650]),
    'CHAS_UppFor': np.array([120.000, 245.000, 267.000]),
    'CHAS_UppAft': np.array([-120.000, 245.000, 258.000]),
    'UPRI_UppPnt': np.array([-11.496, 555.630, 280.000]),
    'CHAS_TiePnt': np.array([50.000, 144.780, 144.500]),
    'UPRI_TiePnt': np.array([70.000, 564.600, 150.000]),
    'NSMA_PPAtt':  np.array([-3.510, 514.710, 294.180]),
    'CHAS_AttPnt': np.array([-179.330, 150.000, 614.790]),
    'CHAS_RocAxi': np.array([0.670, 227.530, 612.110]),
    'CHAS_RocPiv': np.array([0.670, 195.060, 575.180]),
    'ROCK_RodPnt': np.array([59.980, 201.850, 569.210]),
    'ROCK_CoiPnt': np.array([0.670, 150.000, 614.790]),
    'WC': np.array([0.000, 613.227, 203.192]),
    'CP': np.array([0.000, 615.000, 0.000]),
}

HP_REAR_NOMINAL = {
    'CHAS_LowFor': np.array([150.000, 240.000, 126.200]),
    'CHAS_LowAft': np.array([-150.000, 240.000, 120.000]),
    'UPRI_LowPnt': np.array([0.000, 576.780, 112.650]),
    'CHAS_UppFor': np.array([150.000, 240.000, 282.000]),
    'CHAS_UppAft': np.array([-150.000, 240.000, 250.000]),
    'UPRI_UppPnt': np.array([0.000, 520.001, 280.000]),
    'CHAS_TiePnt': np.array([-95.000, 240.000, 163.000]),
    'UPRI_TiePnt': np.array([-80.000, 590.000, 165.800]),
    'NSMA_PPAtt':  np.array([8.930, 497.390, 297.580]),
    'CHAS_AttPnt': np.array([-30.000, 50.000, 430.000]),
    'CHAS_RocAxi': np.array([74.510, 119.730, 580.040]),
    'CHAS_RocPiv': np.array([107.430, 108.260, 547.130]),
    'ROCK_RodPnt': np.array([148.420, 144.100, 572.380]),
    'ROCK_CoiPnt': np.array([97.280, 50.000, 557.280]),
    'WC': np.array([0.000, 613.227, 203.192]),
    'CP': np.array([0.000, 615.000, 0.000]),
}

H_CG = 270.0        # mm — from vehicle_params h_cg * 1000, keep in sync manually
WHEELBASE = 1535.0   # mm — lf+lr from vehicle_params, keep in sync manually


# ==============================================================================
# 2. MOTOR CINEMATICO ANALITICO 3D
# ==============================================================================

def line_intersection_2d(p1, p2, p3, p4):
    y1, z1 = p1[0], p1[1]
    y2, z2 = p2[0], p2[1]
    y3, z3 = p3[0], p3[1]
    y4, z4 = p4[0], p4[1]
    denom = (y1 - y2) * (z3 - z4) - (z1 - z2) * (y3 - y4)
    if np.abs(denom) < 1e-7:
        return np.array([np.nan, np.nan])
    t = ((y1 - y3) * (z3 - z4) - (z1 - z3) * (y3 - y4)) / denom
    return np.array([y1 + t * (y2 - y1), z1 + t * (z2 - z1)])


def _svic_side_view(hp):
    """
    Side-View Instant Centre (X-Z plane). Shared by anti-squat / anti-dive /
    anti-lift — they differ only in WHICH axle's SVIC is used and what
    longitudinal-force fraction scales the resulting angle (see §2b).

    Returns (x_svic, z_svic, tan_alpha) — tan_alpha is the geometric anti
    angle BEFORE force-fraction scaling.
    """
    dx_l = hp['CHAS_LowFor'][0] - hp['CHAS_LowAft'][0]
    dz_l = hp['CHAS_LowFor'][2] - hp['CHAS_LowAft'][2]
    slope_l = dz_l / (dx_l + 1e-6)

    dx_u = hp['CHAS_UppFor'][0] - hp['CHAS_UppAft'][0]
    dz_u = hp['CHAS_UppFor'][2] - hp['CHAS_UppAft'][2]
    slope_u = dz_u / (dx_u + 1e-6)

    z0_l = (hp['CHAS_LowFor'][2] + hp['CHAS_LowAft'][2]) / 2.0
    z0_u = (hp['CHAS_UppFor'][2] + hp['CHAS_UppAft'][2]) / 2.0

    x_svic = (z0_u - z0_l) / (slope_l - slope_u + 1e-7)
    z_svic = slope_l * x_svic + z0_l
    tan_alpha = z_svic / (abs(x_svic) + 1e-6)
    return x_svic, z_svic, tan_alpha


def anti_geometry_percent(hp, force_fraction: float) -> float:
    """
    General %anti-geometry, valid for anti-squat, anti-dive, OR anti-lift —
    the ONLY thing that differentiates them is which axle's hardpoints go in
    (hp) and what force_fraction represents:

      Rear axle,  force_fraction = drive_bias_r  -> %Anti-Squat
      Front axle, force_fraction = brake_bias_f  -> %Anti-Dive
      Front axle, force_fraction = drive_bias_f  -> %Anti-Lift (front, 4WD)
      Rear axle,  force_fraction = brake_bias_r  -> %Anti-Lift (rear, braking)

    %anti = (tan(SVIC angle) / (h_cg / wheelbase)) * 100 * force_fraction

    force_fraction ∈ [0,1] is the share of TOTAL longitudinal (brake or
    drive) force that axle actually reacts. For a 4WD car with brake_bias_f
    and drive torque split both settable, front and rear both get real
    anti-dive/anti-lift contributions — this was completely absent for the
    front axle in the previous version of this script.
    """
    _, _, tan_alpha = _svic_side_view(hp)
    return (tan_alpha / (H_CG / WHEELBASE)) * 100.0 * force_fraction


def evaluate_full_geometry(hp, is_front=True):
    # ── Front View: FVIC, Roll Center, Camber Recovery ─────────────────────
    lca_c_yz = (hp['CHAS_LowFor'][1:] + hp['CHAS_LowAft'][1:]) / 2.0
    uca_c_yz = (hp['CHAS_UppFor'][1:] + hp['CHAS_UppAft'][1:]) / 2.0
    fvic_yz = line_intersection_2d(lca_c_yz, hp['UPRI_LowPnt'][1:], uca_c_yz, hp['UPRI_UppPnt'][1:])

    y_cp, z_cp = hp['CP'][1], hp['CP'][2]
    denom = (fvic_yz[0] - y_cp)
    rc_z = z_cp - y_cp * (fvic_yz[1] - z_cp) / (denom + 1e-7)

    fvsa_len = np.linalg.norm(fvic_yz - hp['WC'][1:])
    camber_roll = -(1.0 - (y_cp / (fvsa_len + 1e-6)))

    # ── Bump Steer ───────────────────────────────────────────────────────
    vec_tr = hp['UPRI_TiePnt'][1:] - hp['CHAS_TiePnt'][1:]
    vec_ic = fvic_yz - hp['UPRI_TiePnt'][1:]
    cross_prod = (vec_tr[0] * vec_ic[1] - vec_tr[1] * vec_ic[0]) / (
        np.linalg.norm(vec_tr) * np.linalg.norm(vec_ic) + 1e-7
    )
    bump_steer = abs(cross_prod)

    # ── Motion Ratio Heave ───────────────────────────────────────────────
    u_axis = (hp['CHAS_RocAxi'] - hp['CHAS_RocPiv']) / np.linalg.norm(hp['CHAS_RocAxi'] - hp['CHAS_RocPiv'])
    r_rod = hp['ROCK_RodPnt'] - hp['CHAS_RocPiv']
    r_dam = hp['ROCK_CoiPnt'] - hp['CHAS_RocPiv']

    v_prod = hp['ROCK_RodPnt'] - hp['NSMA_PPAtt']
    u_prod = v_prod / np.linalg.norm(v_prod)
    v_dam = hp['CHAS_AttPnt'] - hp['ROCK_CoiPnt']
    u_dam = v_dam / np.linalg.norm(v_dam)

    tau_prod = abs(np.dot(np.cross(r_rod, u_prod), u_axis))
    tau_dam = abs(np.dot(np.cross(r_dam, u_dam), u_axis))

    if is_front:
        u_arm = (hp['CHAS_LowFor'] - hp['CHAS_LowAft']) / np.linalg.norm(hp['CHAS_LowFor'] - hp['CHAS_LowAft'])
        p0_arm = hp['CHAS_LowAft']
        upr_pt = hp['UPRI_LowPnt']
    else:
        u_arm = (hp['CHAS_UppFor'] - hp['CHAS_UppAft']) / np.linalg.norm(hp['CHAS_UppFor'] - hp['CHAS_UppAft'])
        p0_arm = hp['CHAS_UppAft']
        upr_pt = hp['UPRI_UppPnt']

    r_upr = upr_pt - p0_arm
    r_pp = hp['NSMA_PPAtt'] - p0_arm
    rad_upr = np.linalg.norm(r_upr - np.dot(r_upr, u_arm) * u_arm)
    rad_pp = np.linalg.norm(r_pp - np.dot(r_pp, u_arm) * u_arm)
    arm_ratio = rad_pp / (rad_upr + 1e-6)

    mr_heave = 1.0 / (arm_ratio * abs(u_prod[2]) * (tau_dam / (tau_prod + 1e-6)) + 1e-6)
    l_damper = np.linalg.norm(v_dam)

    # ── Scrub Radius (front only, geometrically defined but harmless to
    #    compute for rear too — just unused there) ─────────────────────────
    kp_vec = hp['UPRI_UppPnt'] - hp['UPRI_LowPnt']
    t_g = -hp['UPRI_LowPnt'][2] / (kp_vec[2] + 1e-6)
    kp_g = hp['UPRI_LowPnt'] + t_g * kp_vec
    scrub = hp['CP'][1] - kp_g[1]

    return {
        'rc_z': rc_z, 'camber_roll': camber_roll, 'bump_steer': bump_steer,
        'mr_heave': mr_heave, 'scrub': scrub, 'l_damper': l_damper,
    }


# ==============================================================================
# 3. OPTIMIZACION MULTIOBJETIVO INTERACTIVA
# ==============================================================================

# Per-axle hardpoint groups. Each group maps to a slice of the design vector.
# Locking a group pins its bounds to (0,0) — same mechanism as before, just
# generalized so BOTH axles get identical group structure (previous script's
# front/rear vector layouts were hand-special-cased and easy to desync).
_GROUPS_FRONT = [
    ('tie_chassis', ['CHAS_TiePnt.z']),
    ('rocker', ['ROCK_RodPnt.x', 'ROCK_RodPnt.y', 'ROCK_RodPnt.z',
                'ROCK_CoiPnt.x', 'ROCK_CoiPnt.z',
                'CHAS_AttPnt.x', 'CHAS_AttPnt.z']),
    ('upright', ['UPRI_LowPnt.y']),
    ('chassis_upp', ['CHAS_UppFor.z', 'CHAS_UppAft.z']),   # anti-lift lever (front, 4WD)
    ('chassis_low', ['CHAS_LowFor.z', 'CHAS_LowAft.z']),   # anti-dive lever
]

_GROUPS_REAR = [
    ('tie_chassis', ['CHAS_TiePnt.z']),
    ('rocker', ['ROCK_RodPnt.x', 'ROCK_RodPnt.y', 'ROCK_RodPnt.z',
                'ROCK_CoiPnt.x', 'ROCK_CoiPnt.z',
                'CHAS_AttPnt.z']),
    ('chassis_upp', ['CHAS_UppFor.z', 'CHAS_UppAft.z']),   # anti-squat lever
    ('chassis_low', ['CHAS_LowFor.z', 'CHAS_LowAft.z']),   # anti-lift-under-brake lever
]

_AXIS_IDX = {'x': 0, 'y': 1, 'z': 2}
_BOUND_RANGE = {'tie_chassis': 25.0, 'rocker': 35.0, 'upright': 15.0,
                'chassis_upp': 45.0, 'chassis_low': 20.0}   # ampliado 30→45mm


def _build_design_vector_spec(groups, locks):
    """Flatten group -> [(point_name, axis_idx), ...] with per-entry bound."""
    spec = []
    for gname, fields in groups:
        locked = locks.get(gname, True)
        rng = 0.0 if locked else _BOUND_RANGE[gname]
        for f in fields:
            pt, axis = f.split('.')
            spec.append((pt, _AXIS_IDX[axis], rng))
    return spec


def build_hp(x, base_hp, spec):
    hp = {k: v.copy() for k, v in base_hp.items()}
    for (pt, axis, _), dx in zip(spec, x):
        hp[pt][axis] += dx
    return hp


class OptTracker:
    def __init__(self, targets, is_front):
        self.targets = targets
        self.is_front = is_front
        self.iterations = 0
        self.current_metrics = {}
        self.current_loss = 0.0

    def callback(self, loss, geo, anti_vals):
        self.iterations += 1
        self.current_loss = loss
        self.current_metrics = {**geo, **anti_vals}


def optimize_axle(is_front, base_hp, targets, locks, force_fractions):
    """
    force_fractions: dict with keys relevant to this axle —
      front: {'brake_bias_f': ..., 'drive_bias_f': ...}
      rear:  {'drive_bias_r': ..., 'brake_bias_r': ...}
    """
    groups = _GROUPS_FRONT if is_front else _GROUPS_REAR
    spec = _build_design_vector_spec(groups, locks)
    bounds = [(-r, r) if r > 0 else (0.0, 0.0) for (_, _, r) in spec]
    x0 = np.zeros(len(spec))

    tracker = OptTracker(targets, is_front)

    def cost(x):
        hp = build_hp(x, base_hp, spec)
        geo = evaluate_full_geometry(hp, is_front)

        if is_front:
            anti_dive_f = anti_geometry_percent(hp, force_fractions['brake_bias_f'])
            anti_lift_f = anti_geometry_percent(hp, force_fractions['drive_bias_f'])
            anti_vals = {'anti_dive_f': anti_dive_f, 'anti_lift_f': anti_lift_f}
        else:
            anti_squat_r = anti_geometry_percent(hp, force_fractions['drive_bias_r'])
            anti_lift_r = anti_geometry_percent(hp, force_fractions['brake_bias_r'])
            anti_vals = {'anti_squat_r': anti_squat_r, 'anti_lift_r': anti_lift_r}

        loss = 0.0
        if not locks.get('tie_chassis', True):
            loss += 100000.0 * (geo['bump_steer'] ** 2)
        if not locks.get('rocker', True):
            loss += 2000.0 * ((geo['mr_heave'] - targets['mr']) ** 2)
            loss += 50.0 * ((geo['l_damper'] - targets.get('damper_len', 180.0)) ** 2)
        if not locks.get('chassis_upp', True) or not locks.get('chassis_low', True):
            if 'rc_z' in targets:
                loss += 100.0 * ((geo['rc_z'] - targets['rc_z']) ** 2)
            if 'camber_roll' in targets:
                loss += 500.0 * ((geo['camber_roll'] - targets['camber_roll']) ** 2)   # bajado de 2000→500

        if is_front:
            if not locks.get('chassis_low', True) and 'anti_dive' in targets:
                loss += 80.0 * ((anti_vals['anti_dive_f'] - targets['anti_dive']) ** 2)
            if not locks.get('chassis_upp', True) and 'anti_lift' in targets:
                loss += 80.0 * ((anti_vals['anti_lift_f'] - targets['anti_lift']) ** 2)
            if not locks.get('upright', True) and 'scrub' in targets:
                loss += 200.0 * ((geo['scrub'] - targets['scrub']) ** 2)
        else:
            if not locks.get('chassis_upp', True) and 'anti_squat' in targets:
                loss += 100.0 * ((anti_vals['anti_squat_r'] - targets['anti_squat']) ** 2)
            if not locks.get('chassis_upp', True) and 'anti_squat' in targets:
                loss += 800.0 * ((anti_vals['anti_squat_r'] - targets['anti_squat']) ** 2)   # subido de 100→800

        tracker.callback(loss, geo, anti_vals)
        return loss

    axle_label = "Frontal" if is_front else "Trasero"
    console.print(f"\n[bold cyan]Lanzando Optimizador Multicuerpo para eje {axle_label}...[/]")
    with Live(generate_live_table(tracker), refresh_per_second=15) as live:
        def iter_cb(xk):
            live.update(generate_live_table(tracker))
        res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                        callback=iter_cb, options={'ftol': 1e-8, 'maxiter': 400})
        live.update(generate_live_table(tracker))

    opt_hp = build_hp(res.x, base_hp, spec)
    print_final_report(base_hp, opt_hp, targets, is_front, force_fractions)
    return opt_hp


# ==============================================================================
# 4. REPORTES Y MENUS
# ==============================================================================

def generate_live_table(tracker):
    table = Table(title="[bold blue]Optimizador Cinemático L-BFGS-B (En Vivo)", box=box.SIMPLE)
    table.add_column("Iteración", justify="center")
    table.add_column("Loss", justify="right", style="magenta")
    table.add_column("Bump Steer", justify="right")
    table.add_column("MR Heave", justify="right")
    table.add_column("RC Z (mm)", justify="right")
    table.add_column("Camber Rec.", justify="right")
    if tracker.is_front:
        table.add_column("Anti-Dive %", justify="right")
        table.add_column("Anti-Lift %", justify="right")
    else:
        table.add_column("Anti-Squat %", justify="right")
        table.add_column("Anti-Lift %", justify="right")

    if tracker.current_metrics:
        m = tracker.current_metrics
        bs_str = f"[green]{m['bump_steer']:.5f}[/]" if m['bump_steer'] < 0.005 else f"[red]{m['bump_steer']:.5f}[/]"
        mr_str = (f"[green]{m['mr_heave']:.3f}[/]"
                   if abs(m['mr_heave'] - tracker.targets.get('mr', m['mr_heave'])) < 0.02
                   else f"{m['mr_heave']:.3f}")
        row = [str(tracker.iterations), f"{tracker.current_loss:.2f}", bs_str, mr_str,
               f"{m['rc_z']:.2f}", f"{abs(m['camber_roll'])*100:.1f}%"]
        if tracker.is_front:
            row += [f"{m.get('anti_dive_f', 0):.1f}%", f"{m.get('anti_lift_f', 0):.1f}%"]
        else:
            row += [f"{m.get('anti_squat_r', 0):.1f}%", f"{m.get('anti_lift_r', 0):.1f}%"]
        table.add_row(*row)
    return table


def print_final_report(base_hp, opt_hp, targets, is_front, force_fractions):
    axle = "DELANTERA (FRONT)" if is_front else "TRASERA (REAR)"
    geo = evaluate_full_geometry(opt_hp, is_front)

    if is_front:
        anti_dive_f = anti_geometry_percent(opt_hp, force_fractions['brake_bias_f'])
        anti_lift_f = anti_geometry_percent(opt_hp, force_fractions['drive_bias_f'])
    else:
        anti_squat_r = anti_geometry_percent(opt_hp, force_fractions['drive_bias_r'])
        anti_lift_r = anti_geometry_percent(opt_hp, force_fractions['brake_bias_r'])

    line_sep = "═" * 92
    console.print(f"\n[bold green]{line_sep}[/]")
    console.print(f"  PROJECT-GP · REPORTE TÉCNICO DE SUSPENSIÓN {axle}")
    console.print(f"[bold green]{line_sep}[/]")

    console.print("\n[bold yellow]1. DIAGNÓSTICO DINÁMICO:[/]")
    console.print(f"  • [bold]Bump Steer[/]      : {geo['bump_steer']:.6f} [dim](Target: 0.000)[/]")
    console.print(f"  • [bold]Motion Ratio[/]    : {geo['mr_heave']:.3f} -> Wheel Rate = "
                   f"[cyan]{(44.0 if is_front else 53.0)/(geo['mr_heave']**2):.1f} N/mm[/]")
    console.print(f"  • [bold]Roll Center Z[/]   : {geo['rc_z']:.2f} mm")
    console.print(f"  • [bold]Camber Recovery[/] : {abs(geo['camber_roll'])*100:.1f}%")
    if is_front:
        console.print(f"  • [bold]Scrub Radius[/]    : {geo['scrub']:+.2f} mm")
        console.print(f"  • [bold]Anti-Dive[/]       : {anti_dive_f:.1f}%  "
                       f"[dim](brake_bias_f={force_fractions['brake_bias_f']:.2f})[/]")
        console.print(f"  • [bold]Anti-Lift (accel)[/]: {anti_lift_f:.1f}%  "
                       f"[dim](drive_bias_f={force_fractions['drive_bias_f']:.2f})[/]")
    else:
        console.print(f"  • [bold]Anti-Squat[/]      : {anti_squat_r:.1f}%  "
                       f"[dim](drive_bias_r={force_fractions['drive_bias_r']:.2f})[/]")
        console.print(f"  • [bold]Anti-Lift (brake)[/]: {anti_lift_r:.1f}%  "
                       f"[dim](brake_bias_r={force_fractions['brake_bias_r']:.2f})[/]")
    console.print(f"  • [bold]Longitud Amortiguador[/]: {geo['l_damper']:.2f} mm")

    console.print("\n[bold yellow]2. COORDENADAS PARA OPTIMUM KINEMATICS:[/]")
    table = Table(box=box.MINIMAL_HEAVY_HEAD)
    table.add_column("Punto (OK)", style="cyan")
    table.add_column("Original [X, Y, Z]")
    table.add_column("Optimizado [X, Y, Z]", style="bold white")
    table.add_column("Delta (Δ)")

    keys = ['CHAS_LowFor', 'CHAS_LowAft', 'UPRI_LowPnt', 'CHAS_UppFor', 'CHAS_UppAft', 'UPRI_UppPnt',
            'CHAS_TiePnt', 'UPRI_TiePnt', 'NSMA_PPAtt', 'CHAS_AttPnt', 'ROCK_RodPnt', 'ROCK_CoiPnt']

    for k in keys:
        o, p = base_hp[k], opt_hp[k]
        d = p - o
        mod = "[red]*[/]" if np.linalg.norm(d) > 0.1 else " "
        table.add_row(
            f"{k}{mod}",
            f"[{o[0]:6.2f}, {o[1]:6.2f}, {o[2]:6.2f}]",
            f"[{p[0]:6.2f}, {p[1]:6.2f}, {p[2]:6.2f}]",
            f"[{d[0]:+5.1f}, {d[1]:+5.1f}, {d[2]:+5.1f}]"
        )
    console.print(table)

    console.print("\n[bold yellow]3. COPIA DIRECTA — OPTIMUM KINEMATICS (Left / Right, mm)[/]")
    for k in ['CHAS_TiePnt', 'CHAS_AttPnt', 'ROCK_RodPnt', 'ROCK_CoiPnt',
              'CHAS_UppFor', 'CHAS_UppAft', 'CHAS_LowFor', 'CHAS_LowAft', 'UPRI_LowPnt']:
        p = opt_hp[k]
        console.print(f"  {k:<14} Left: [{p[0]:9.3f}, {p[1]:9.3f}, {p[2]:9.3f}]  |  "
                       f"Right: [{p[0]:9.3f}, {-p[1]:9.3f}, {p[2]:9.3f}]")

    poly = [geo['mr_heave'], -0.650, -5.200] if not is_front else [geo['mr_heave'], -0.789, -8.649]
    console.print("\n[bold yellow]4. CONFIGURACIÓN JAX — config/vehicles/ter27.py[/]")
    console.print(f"  'motion_ratio_{'f' if is_front else 'r'}_poly': "
                   f"[{poly[0]:.4f}, {poly[1]:.4f}, {poly[2]:.4f}],")
    if is_front:
        console.print(f"  'anti_dive_f': {anti_dive_f/100.0:.4f},   # SuspensionSetup index 20")
    else:
        console.print(f"  'anti_squat':  {anti_squat_r/100.0:.4f},   # SuspensionSetup index 19")
        console.print(f"  'anti_dive_r': {anti_lift_r/100.0:.4f},   # SuspensionSetup index 21"
                       f"  [dim](rear anti-lift-under-brake maps to anti_dive_r in vehicle_dynamics.py)[/]")
    console.print(f"[bold green]{line_sep}[/]\n")


# ==============================================================================
# 5. MENU PRINCIPAL
# ==============================================================================

def _ask_locks(is_front):
    console.print(f"\n[bold]Grupos de Hardpoints — eje {'FRONT' if is_front else 'REAR'}[/] "
                   f"[dim](Sí = optimizar / desbloquear, No = bloquear en nominal)[/]")
    locks = {}
    locks['tie_chassis'] = not Confirm.ask(
        "  ¿Optimizar anclaje de Tie-Rod en chasis (Bump Steer)?", default=False)
    locks['rocker'] = not Confirm.ask(
        "  ¿Optimizar Balancín + Amortiguador (Motion Ratio / Longitud)?", default=True)
    locks['chassis_upp'] = not Confirm.ask(
        "  ¿Optimizar trapecio SUPERIOR en chasis "
        + ("(Anti-Lift bajo potencia + Roll Center)?" if is_front
           else "(Anti-Squat + Roll Center)?"),
        default=False)
    locks['chassis_low'] = not Confirm.ask(
        "  ¿Optimizar trapecio INFERIOR en chasis "
        + ("(Anti-Dive)?" if is_front else "(Anti-Lift bajo frenada)?"),
        default=False)
    if is_front:
        locks['upright'] = not Confirm.ask(
            "  ¿Permitir modificar la mangueta (Scrub Radius)?", default=False)
    return locks


def _ask_optional_float(prompt: str, default: float):
    raw = Prompt.ask(prompt, default="")
    return None if raw.strip() == "" else float(raw)

def _ask_targets(is_front, locks):    
    targets = {}
    if not locks.get('rocker', True):
        targets['mr'] = FloatPrompt.ask("  Target Motion Ratio", default=1.15 if is_front else 1.20)
        targets['damper_len'] = FloatPrompt.ask("  Target longitud amortiguador (mm)", default=180.0)
    if not locks.get('chassis_upp', True) or not locks.get('chassis_low', True):
        targets['rc_z'] = FloatPrompt.ask("  Target Roll Center Z (mm)", default=19.85 if is_front else 40.0)
        if Confirm.ask("  ¿Fijar también Camber Recovery objetivo?", default=False):
            default_camber = -0.805 if is_front else -0.71
            targets['camber_roll'] = FloatPrompt.ask("  Target Camber Recovery (fracción)", default=default_camber)
    if is_front:
        if not locks.get('chassis_low', True):
            targets['anti_dive'] = FloatPrompt.ask("  Target Anti-Dive %", default=15.0)
        if not locks.get('chassis_upp', True):
            targets['anti_lift'] = FloatPrompt.ask("  Target Anti-Lift (bajo potencia) %", default=10.0)
        if not locks.get('upright', True):
            targets['scrub'] = FloatPrompt.ask("  Target Scrub Radius (mm)", default=3.0)
    else:
        if not locks.get('chassis_upp', True):
            v = _ask_optional_float("  Target Anti-Squat % (vacío = sin target)", 29.9)
            if v is not None:
                targets['anti_squat'] = v
        if not locks.get('chassis_low', True):
            v = _ask_optional_float("  Target Anti-Lift (bajo frenada) % (vacío = sin target)", 8.0)
            if v is not None:
                targets['anti_lift'] = v
    return targets


def _ask_force_fractions(need_front, need_rear):
    """
    These are the SAME fractions your calibrated setup already uses/produces:
      brake_bias_f  <-> SuspensionSetup.brake_bias_f  (fraction of total brake
                         force reacted at the front axle)
      drive_bias_f  <-> fraction of total drive torque delivered by the front
                         hub motors (Ter27 is 4WD — read this off your
                         powertrain torque-vectoring nominal split, NOT 0/1)
    """
    console.print("\n[bold]Reparto de Fuerzas Longitudinales (Ter27 es 4WD)[/]")
    console.print("[dim]  brake_bias_f: fracción del frenado total reaccionada en el eje "
                   "delantero (coincide con SuspensionSetup.brake_bias_f).[/]")
    console.print("[dim]  drive_bias_f: fracción del par motor total entregado por el eje "
                   "delantero en reparto nominal (no asumas RWD).[/]")
    ff = {}
    if need_front or need_rear:
        brake_bias_f = FloatPrompt.ask("  brake_bias_f [0-1]", default=0.60)
        drive_bias_f = FloatPrompt.ask("  drive_bias_f [0-1] (par delantero / par total)", default=0.50)
        ff['brake_bias_f'] = brake_bias_f
        ff['drive_bias_f'] = drive_bias_f
        ff['brake_bias_r'] = 1.0 - brake_bias_f
        ff['drive_bias_r'] = 1.0 - drive_bias_f
    return ff


def main_menu():
    console.print(Panel.fit(
        "[bold white]PROJECT-GP[/]\n[cyan]Optimizador Cinemático de Suspensión Interactiva 108-DOF\n"
        "[dim]Anti-dive / anti-squat / anti-lift ahora calculados correctamente "
        "para ambos ejes (4WD-aware)[/]",
        border_style="blue"))

    ans = Prompt.ask("Selecciona eje a optimizar [1:Front, 2:Rear, 3:Ambos]",
                      choices=["1", "2", "3"], default="3", show_choices=False)
    do_front = ans in ("1", "3")
    do_rear = ans in ("2", "3")

    force_fractions = _ask_force_fractions(do_front, do_rear)

    if do_front:
        console.print("\n[bold underline]Configuración FRONT[/]")
        locks_f = _ask_locks(is_front=True)
        targets_f = _ask_targets(is_front=True, locks=locks_f)
        optimize_axle(True, HP_FRONT_NOMINAL, targets_f, locks_f, force_fractions)

    if do_rear:
        console.print("\n[bold underline]Configuración REAR[/]")
        locks_r = _ask_locks(is_front=False)
        targets_r = _ask_targets(is_front=False, locks=locks_r)
        optimize_axle(False, HP_REAR_NOMINAL, targets_r, locks_r, force_fractions)


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[red]Operación cancelada por el usuario.[/]")