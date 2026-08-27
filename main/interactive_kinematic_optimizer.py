#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# PROJECT-GP: 3D INTERACTIVE MULTI-OBJECTIVE KINEMATIC OPTIMIZER (TER27)
# ==============================================================================
# Autor: Alex Revilla / Tecnun eRacing
# Archivo: main/interactive_kinematic_optimizer.py
# ==============================================================================

import numpy as np
from scipy.optimize import minimize
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

console = Console()

# ==============================================================================
# 1. HARDPOINTS BASE NOMINALES (TER27 - VELIS 2) [mm]
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
    'CP': np.array([0.000, 615.000, 0.000])
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
    'CP': np.array([0.000, 615.000, 0.000])
}

H_CG = 270.0
WHEELBASE = 1535.0

# ==============================================================================
# 2. MOTOR CINEMÁTICO ANALÍTICO 3D
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

def evaluate_full_geometry(hp, is_front=True):
    # Vista Frontal: FVIC, Roll Center, Camber Recovery
    lca_c_yz = (hp['CHAS_LowFor'][1:] + hp['CHAS_LowAft'][1:]) / 2.0
    uca_c_yz = (hp['CHAS_UppFor'][1:] + hp['CHAS_UppAft'][1:]) / 2.0
    fvic_yz = line_intersection_2d(lca_c_yz, hp['UPRI_LowPnt'][1:], uca_c_yz, hp['UPRI_UppPnt'][1:])
    
    y_cp, z_cp = hp['CP'][1], hp['CP'][2]
    denom = (fvic_yz[0] - y_cp)
    rc_z = z_cp - y_cp * (fvic_yz[1] - z_cp) / (denom + 1e-7)
    
    fvsa_len = np.linalg.norm(fvic_yz - hp['WC'][1:])
    camber_roll = -(1.0 - (y_cp / (fvsa_len + 1e-6)))

    # Vista Lateral: SVIC, Anti-Squat / Anti-Dive
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
    anti_pct = (tan_alpha / (H_CG / WHEELBASE)) * 100.0

    # Bump Steer
    vec_tr = hp['UPRI_TiePnt'][1:] - hp['CHAS_TiePnt'][1:]
    vec_ic = fvic_yz - hp['UPRI_TiePnt'][1:]
    cross_prod = (vec_tr[0] * vec_ic[1] - vec_tr[1] * vec_ic[0]) / (
        np.linalg.norm(vec_tr) * np.linalg.norm(vec_ic) + 1e-7
    )
    bump_steer = abs(cross_prod)
    
    # Motion Ratio Heave
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
    
    # Scrub Radius
    kp_vec = hp['UPRI_UppPnt'] - hp['UPRI_LowPnt']
    t_g = -hp['UPRI_LowPnt'][2] / (kp_vec[2] + 1e-6)
    kp_g = hp['UPRI_LowPnt'] + t_g * kp_vec
    scrub = hp['CP'][1] - kp_g[1]

    return rc_z, camber_roll, bump_steer, mr_heave, anti_pct, scrub, l_damper

# ==============================================================================
# 3. OPTIMIZACIÓN MULTIOBJETIVO INTERACTIVA
# ==============================================================================

class OptTracker:
    def __init__(self, targets):
        self.targets = targets
        self.iterations = 0
        self.current_metrics = {}

    def callback(self, loss, bs, mr, rc_z, c_roll, l_damp):
        self.iterations += 1
        self.current_metrics = {
            'loss': loss, 'bs': bs, 'mr': mr, 'rc_z': rc_z, 'c_roll': c_roll, 'l_damp': l_damp
        }

def build_hp(x, base_hp, is_front):
    hp = {k: v.copy() for k, v in base_hp.items()}
    if is_front:
        hp['CHAS_TiePnt'][2] += x[0]
        hp['ROCK_RodPnt'][0] += x[1]
        hp['ROCK_RodPnt'][1] += x[2]
        hp['ROCK_RodPnt'][2] += x[3]
        hp['ROCK_CoiPnt'][0] += x[4]
        hp['ROCK_CoiPnt'][2] += x[5]
        hp['CHAS_AttPnt'][0] += x[6]
        hp['CHAS_AttPnt'][2] += x[7]
        hp['UPRI_LowPnt'][1] += x[8]
    else:
        hp['CHAS_TiePnt'][2] += x[0]
        hp['CHAS_UppFor'][2] += x[1]
        hp['CHAS_UppAft'][2] += x[2]
        hp['ROCK_RodPnt'][0] += x[3]
        hp['ROCK_RodPnt'][1] += x[4]
        hp['ROCK_RodPnt'][2] += x[5]
        hp['ROCK_CoiPnt'][0] += x[6]
        hp['ROCK_CoiPnt'][2] += x[7]
        hp['CHAS_AttPnt'][2] += x[8]
    return hp

def optimize_suspension(is_front, targets, locks):
    base_hp = HP_FRONT_NOMINAL if is_front else HP_REAR_NOMINAL
    tracker = OptTracker(targets)

    def cost(x):
        hp = build_hp(x, base_hp, is_front)
        rc_z, c_roll, bs, mr, anti_pct, scrub, l_damp = evaluate_full_geometry(hp, is_front)
        
        loss = 0.0
        if not locks.get('tie_chassis', True):
            loss += 100000.0 * (bs ** 2)
        if not locks.get('rocker', True):
            loss += 2000.0 * ((mr - targets['mr']) ** 2)
            loss += 50.0   * ((l_damp - 180.0) ** 2)
        if not locks.get('chassis_upp', True):
            loss += 100.0  * ((rc_z - targets['rc_z']) ** 2)
            loss += 2000.0 * ((c_roll - targets['camber_roll']) ** 2)
            if not is_front:
                loss += 100.0 * ((anti_pct - targets['anti']) ** 2)
        if is_front and not locks.get('upright', True):
            loss += 200.0 * ((scrub - targets['scrub']) ** 2)

        tracker.callback(loss, bs, mr, rc_z, c_roll, l_damp)
        return loss

    if is_front:
        bounds = [
            (-25.0, 25.0) if not locks.get('tie_chassis', True) else (0.0, 0.0), # Tie Rod Z
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Rod X
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Rod Y
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Rod Z
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Coi X
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Coi Z
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Att X
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Att Z
            (-15.0, 15.0) if not locks.get('upright', True) else (0.0, 0.0)       # Scrub Y
        ]
    else:
        bounds = [
            (-25.0, 25.0) if not locks.get('tie_chassis', True) else (0.0, 0.0), # Tie Rod Z
            (-30.0, 30.0) if not locks.get('chassis_upp', True) else (0.0, 0.0), # UCA Fore Z
            (-30.0, 30.0) if not locks.get('chassis_upp', True) else (0.0, 0.0), # UCA Aft Z
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Rod X
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Rod Y
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Rod Z
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Coi X
            (-35.0, 35.0) if not locks.get('rocker', True) else (0.0, 0.0),      # Coi Z
            (-25.0, 25.0) if not locks.get('rocker', True) else (0.0, 0.0)       # Att Z
        ]

    x0 = np.zeros(9)
    console.print(f"\n[bold cyan]Lanzando Optimizador Multicuerpo para eje {'Frontal' if is_front else 'Trasero'}...[/]")
    with Live(generate_live_table(tracker), refresh_per_second=15) as live:
        def iter_cb(xk):
            live.update(generate_live_table(tracker))
        res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds, callback=iter_cb, options={'ftol': 1e-6, 'maxiter': 250})
        live.update(generate_live_table(tracker))

    opt_hp = build_hp(res.x, base_hp, is_front)
    print_final_report(base_hp, opt_hp, targets, is_front)

# ==============================================================================
# 4. REPORTES Y MENÚS
# ==============================================================================

def generate_live_table(tracker):
    table = Table(title="[bold blue]Optimizador Cinemático L-BFGS-B (En Vivo)", box=box.SIMPLE)
    table.add_column("Iteración", justify="center")
    table.add_column("Loss", justify="right", style="magenta")
    table.add_column("Bump Steer", justify="right")
    table.add_column("MR Heave", justify="right")
    table.add_column("RC Z (mm)", justify="right")
    table.add_column("Camber Rec.", justify="right")
    table.add_column("Damper Len", justify="right")
    
    if tracker.current_metrics:
        m = tracker.current_metrics
        bs_str = f"[green]{m['bs']:.5f}[/]" if m['bs'] < 0.005 else f"[red]{m['bs']:.5f}[/]"
        mr_str = f"[green]{m['mr']:.3f}[/]" if abs(m['mr'] - tracker.targets['mr']) < 0.02 else f"{m['mr']:.3f}"
        table.add_row(
            str(tracker.iterations), f"{m['loss']:.2f}", bs_str, mr_str,
            f"{m['rc_z']:.2f}", f"{abs(m['c_roll'])*100:.1f}%", f"{m['l_damp']:.2f}"
        )
    return table

def print_final_report(base_hp, opt_hp, targets, is_front):
    axle = "DELANTERA (FRONT)" if is_front else "TRASERA (REAR)"
    rc_z, c_roll, bs, mr, anti, scrub, l_damper = evaluate_full_geometry(opt_hp, is_front)

    line_sep = "═" * 90
    console.print(f"\n[bold green]{line_sep}[/]")
    console.print(f"  PROJECT-GP · REPORTE TÉCNICO DE SUSPENSIÓN {axle}")
    console.print(f"[bold green]{line_sep}[/]")
    
    console.print("\n[bold yellow]1. DIAGNÓSTICO DINÁMICO:[/]")
    console.print(f"  • [bold]Bump Steer[/]     : {bs:.6f} [dim](Target: 0.000)[/]")
    console.print(f"  • [bold]Motion Ratio[/]   : {mr:.3f} -> Wheel Rate = [cyan]{(44.0 if is_front else 53.0)/(mr**2):.1f} N/mm[/]")
    console.print(f"  • [bold]Roll Center Z[/]  : {rc_z:.2f} mm")
    console.print(f"  • [bold]Camber Recovery[/]: {abs(c_roll)*100:.1f}%")
    if is_front:
        console.print(f"  • [bold]Scrub Radius[/]   : {scrub:+.2f} mm")
    else:
        console.print(f"  • [bold]Anti-Squat[/]     : {anti:.1f}%")
    console.print(f"  • [bold]Longitud Amortiguador[/]: {l_damper:.2f} mm")
    
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
    
    poly = [mr, -0.650, -5.200] if not is_front else [mr, -0.789, -8.649]
    console.print("\n[bold yellow]3. CONFIGURACIÓN JAX (config/vehicles/ter27.py):[/]")
    console.print(f"  'motion_ratio_{'f' if is_front else 'r'}_poly': [{poly[0]:.4f}, {poly[1]:.4f}, {poly[2]:.4f}]")
    console.print(f"[bold green]{line_sep}[/]\n")

def main_menu():
    console.print(Panel.fit("[bold white]PROJECT-GP[/]\n[cyan]Optimizador Cinemático de Suspensión Interactiva 108-DOF", border_style="blue"))
    
    ans = Prompt.ask("Selecciona Tren a Optimizar [1:Front, 2:Rear, 3:Ambos]", choices=["1", "2", "3"], default="3", show_choices=False)
    
    do_front = ans in ["1", "3"]
    do_rear  = ans in ["2", "3"]

    if do_front:
        console.print("\n[bold]Configuración FRONT[/]")
        locks_f = {
            'upright': not Confirm.ask("¿Permitir modificar la mangueta (Scrub Radius)?", default=False),
            'tie_chassis': not Confirm.ask("¿Optimizar anclaje de Tie-Rod en chasis (Bump Steer)?", default=False),
            'rocker': not Confirm.ask("¿Optimizar Balancín y Amortiguador (Motion Ratio)?", default=True)
        }
        targets_f = {
            'mr': float(Prompt.ask("Target Motion Ratio", default="1.15")),
            'rc_z': float(Prompt.ask("Target Roll Center Z (mm)", default="19.85")),
            'camber_roll': -0.805,
            'scrub': float(Prompt.ask("Target Scrub Radius (mm)", default="3.0")) if not locks_f['upright'] else 10.0
        }
        optimize_suspension(True, targets_f, locks_f)

    if do_rear:
        console.print("\n[bold]Configuración REAR[/]")
        locks_r = {
            'tie_chassis': not Confirm.ask("¿Optimizar anclaje de Tie-Rod en chasis (Bump Steer)?", default=True),
            'chassis_upp': not Confirm.ask("¿Permitir variar trapecios superiores en chasis (Anti-Squat / RC)?", default=True),
            'rocker': not Confirm.ask("¿Optimizar Balancín y Amortiguador (Motion Ratio)?", default=True),
            'upright': True
        }
        targets_r = {
            'mr': float(Prompt.ask("Target Motion Ratio", default="1.20")),
            'rc_z': float(Prompt.ask("Target Roll Center Z (mm)", default="40.0")),
            'camber_roll': -float(Prompt.ask("Target Camber Recovery %", default="71")) / 100.0,
            'anti': float(Prompt.ask("Target Anti-Squat %", default="40.0")) if not locks_r['chassis_upp'] else 29.9
        }
        optimize_suspension(False, targets_r, locks_r)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[red]Operación cancelada por el usuario.[/]")