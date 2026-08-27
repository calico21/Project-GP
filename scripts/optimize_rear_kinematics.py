#!/usr/bin/env python3
# ==============================================================================
# PROJECT-GP: 3D REAR SUSPENSION OPTIMIZER (FROZEN UPRIGHT CONSTRAINTS)
# ==============================================================================
# Base File : Rear TeR27 - Velis 2.xlsx
# Path      : scripts/optimize_rear_kinematics.py
# ==============================================================================

import numpy as np
from scipy.optimize import minimize

# ==============================================================================
# 1. HARDPOINTS BASE (Rear TeR27 - Velis 2.xlsx) [mm]
# ==============================================================================
HP_VELIS2 = {
    # ── Double A-Arm (Chasis) ──
    'CHAS_LowFor': np.array([150.000, 240.000, 126.200]),   # Trapecio Inferior Chasis Fore
    'CHAS_LowAft': np.array([-150.000, 240.000, 120.000]),  # Trapecio Inferior Chasis Aft
    'CHAS_UppFor': np.array([150.000, 240.000, 282.000]),   # Trapecio Superior Chasis Fore
    'CHAS_UppAft': np.array([-150.000, 240.000, 250.000]),  # Trapecio Superior Chasis Aft
    'CHAS_TiePnt': np.array([-95.000, 240.000, 163.000]),   # Tie Rod Chasis

    # ── MANGUETA (UPRIGHT) -> 100% CONGELADA / INMUTABLE ──
    'UPRI_LowPnt': np.array([0.000, 576.780, 112.650]),     # Rótula Inferior Mangueta [BLOQUEADO]
    'UPRI_UppPnt': np.array([0.000, 520.001, 280.000]),     # Rótula Superior Mangueta [BLOQUEADO]
    'UPRI_TiePnt': np.array([-80.000, 590.000, 165.800]),   # Rótula Tie Rod Mangueta  [BLOQUEADO]
    'NSMA_PPAttPnt_L': np.array([8.930, 497.390, 297.580]), # Anclaje Pushrod Mangueta [BLOQUEADO]

    # ── Push-Pull & Balancín (Mecanizado y Chasis) ──
    'CHAS_AttPnt_L':   np.array([-30.000, 50.000, 430.000]), # Anclaje Amortiguador Chasis
    'CHAS_RocAxi_L':   np.array([74.510, 119.730, 580.040]), # Eje Balancín Chasis
    'CHAS_RocPiv_L':   np.array([107.430, 108.260, 547.130]),# Pivote Balancín Chasis
    'ROCK_RodPnt_L':   np.array([148.420, 144.100, 572.380]),# Pickup Pushrod Balancín
    'ROCK_CoiPnt_L':   np.array([97.280, 50.000, 557.280]),  # Pickup Amortiguador Balancín

    # ── Wheels & Referencia ──
    'WC': np.array([0.000, 613.227, 203.192]),
    'CP': np.array([0.000, 615.000, 0.000])
}

DESCRIPTIONS = {
    'CHAS_LowFor': ('Double A-Arm', 'Trapecio Inferior Chasis Fore', True),
    'CHAS_LowAft': ('Double A-Arm', 'Trapecio Inferior Chasis Aft', True),
    'CHAS_UppFor': ('Double A-Arm', 'Trapecio Superior Chasis Fore', True),
    'CHAS_UppAft': ('Double A-Arm', 'Trapecio Superior Chasis Aft', True),
    'UPRI_LowPnt': ('Double A-Arm', 'Mangueta Rótula Inferior', False),
    'UPRI_UppPnt': ('Double A-Arm', 'Mangueta Rótula Superior', False),
    'CHAS_TiePnt': ('Double A-Arm', 'Tie Rod (Tirante) en Chasis', True),
    'UPRI_TiePnt': ('Double A-Arm', 'Tie Rod (Tirante) en Mangueta', False),
    'NSMA_PPAttPnt_L': ('Push Pull', 'Pushrod Anclaje Exterior', False),
    'CHAS_AttPnt_L':   ('Push Pull', 'Amortiguador Anclaje Chasis', True),
    'CHAS_RocAxi_L':   ('Push Pull', 'Balancín Eje de Giro', True),
    'CHAS_RocPiv_L':   ('Push Pull', 'Balancín Pivote Chasis', True),
    'ROCK_RodPnt_L':   ('Push Pull', 'Balancín Pickup Pushrod', True),
    'ROCK_CoiPnt_L':   ('Push Pull', 'Balancín Pickup Amortiguador', True)
}

# ==============================================================================
# 2. SOLVER CINEMÁTICO 3D
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

def evaluate_rear_kinematics(hp_mod):
    lca_c_yz = (hp_mod['CHAS_LowFor'][1:] + hp_mod['CHAS_LowAft'][1:]) / 2.0
    lca_u_yz = hp_mod['UPRI_LowPnt'][1:]
    uca_c_yz = (hp_mod['CHAS_UppFor'][1:] + hp_mod['CHAS_UppAft'][1:]) / 2.0
    uca_u_yz = hp_mod['UPRI_UppPnt'][1:]
    
    fvic_yz = line_intersection_2d(lca_c_yz, lca_u_yz, uca_c_yz, uca_u_yz)
    if np.isnan(fvic_yz).any():
        return 999.0, 999.0, 999.0, 999.0
    
    y_cp, z_cp = hp_mod['CP'][1], hp_mod['CP'][2]
    denom = (fvic_yz[0] - y_cp)
    rc_z = z_cp - y_cp * (fvic_yz[1] - z_cp) / (denom + 1e-7)
    
    wc_yz = hp_mod['WC'][1:]
    fvsa_len = np.linalg.norm(fvic_yz - wc_yz)
    half_track = hp_mod['CP'][1]
    camber_gain_roll = -(1.0 - (half_track / (fvsa_len + 1e-6)))
    
    # Bump steer mediante alineación del tirante de convergencia al FVIC
    tr_c_yz = hp_mod['CHAS_TiePnt'][1:]
    tr_u_yz = hp_mod['UPRI_TiePnt'][1:]
    vec_tr = tr_u_yz - tr_c_yz
    vec_to_ic = fvic_yz - tr_u_yz
    cross_prod = (vec_tr[0] * vec_to_ic[1] - vec_tr[1] * vec_to_ic[0]) / (
        np.linalg.norm(vec_tr) * np.linalg.norm(vec_to_ic) + 1e-7
    )
    bump_steer_metric = abs(cross_prod)
    
    # Cinemática 3D de Balancín y Motion Ratio
    u_axis = (hp_mod['CHAS_RocAxi_L'] - hp_mod['CHAS_RocPiv_L']) / np.linalg.norm(hp_mod['CHAS_RocAxi_L'] - hp_mod['CHAS_RocPiv_L'])
    r_rod = hp_mod['ROCK_RodPnt_L'] - hp_mod['CHAS_RocPiv_L']
    r_dam = hp_mod['ROCK_CoiPnt_L'] - hp_mod['CHAS_RocPiv_L']
    
    v_prod = hp_mod['ROCK_RodPnt_L'] - hp_mod['NSMA_PPAttPnt_L']
    u_prod = v_prod / np.linalg.norm(v_prod)
    v_dam = hp_mod['CHAS_AttPnt_L'] - hp_mod['ROCK_CoiPnt_L']
    u_dam = v_dam / np.linalg.norm(v_dam)
    
    tau_prod = abs(np.dot(np.cross(r_rod, u_prod), u_axis))
    tau_dam = abs(np.dot(np.cross(r_dam, u_dam), u_axis))
    
    u_uca = (hp_mod['CHAS_UppFor'] - hp_mod['CHAS_UppAft']) / np.linalg.norm(hp_mod['CHAS_UppFor'] - hp_mod['CHAS_UppAft'])
    p0_uca = hp_mod['CHAS_UppAft']
    r_upr = hp_mod['UPRI_UppPnt'] - p0_uca
    r_pp = hp_mod['NSMA_PPAttPnt_L'] - p0_uca
    rad_upr = np.linalg.norm(r_upr - np.dot(r_upr, u_uca) * u_uca)
    rad_pp = np.linalg.norm(r_pp - np.dot(r_pp, u_uca) * u_uca)
    arm_geom_ratio = rad_pp / (rad_upr + 1e-6)
    
    sin_pr_vert = abs(u_prod[2])
    mr_heave = 1.0 / (arm_geom_ratio * sin_pr_vert * (tau_dam / (tau_prod + 1e-6)) + 1e-6)
    
    return rc_z, camber_gain_roll, bump_steer_metric, mr_heave

# ==============================================================================
# 3. OPTIMIZACIÓN (SOLO PUNTOS DE CHASIS Y BALANCÍN)
# ==============================================================================
def objective(x, base_hp):
    hp_test = {k: v.copy() for k, v in base_hp.items()}
    # 0: CHAS_TiePnt Z
    hp_test['CHAS_TiePnt'][2]   += x[0]
    # 1: ROCK_RodPnt_L X, Y, Z
    hp_test['ROCK_RodPnt_L'][0] += x[1]
    hp_test['ROCK_RodPnt_L'][1] += x[2]
    hp_test['ROCK_RodPnt_L'][2] += x[3]
    # 2: ROCK_CoiPnt_L X, Z
    hp_test['ROCK_CoiPnt_L'][0] += x[4]
    hp_test['ROCK_CoiPnt_L'][2] += x[5]
    # 3: CHAS_AttPnt_L Z
    hp_test['CHAS_AttPnt_L'][2] += x[6]
    
    rc_z, camber_roll, bump_steer, mr_heave = evaluate_rear_kinematics(hp_test)
    l_damper = np.linalg.norm(hp_test['CHAS_AttPnt_L'] - hp_test['ROCK_CoiPnt_L'])
    damper_len_err = (l_damper - 180.0) ** 2
    
    loss = (
        25000.0 * (bump_steer ** 2) +          # Anular Bump Steer
        500.0   * ((mr_heave - 1.190) ** 2) +   # Motion Ratio objetivo 1.19
        10.0    * damper_len_err                # Longitud nominal 180mm
    )
    return loss

bounds = [
    (-25.0, 25.0),   # CHAS_TiePnt Z
    (-30.0, 30.0),   # ROCK_RodPnt_L X
    (-30.0, 50.0),   # ROCK_RodPnt_L Y
    (-30.0, 30.0),   # ROCK_RodPnt_L Z
    (-30.0, 30.0),   # ROCK_CoiPnt_L X
    (-30.0, 30.0),   # ROCK_CoiPnt_L Z
    (-25.0, 25.0)    # CHAS_AttPnt_L Z
]

x0 = np.zeros(7)
res = minimize(objective, x0, args=(HP_VELIS2,), method='L-BFGS-B', bounds=bounds)

opt_hp = {k: v.copy() for k, v in HP_VELIS2.items()}
opt_hp['CHAS_TiePnt'][2]   += res.x[0]
opt_hp['ROCK_RodPnt_L'][0] += res.x[1]
opt_hp['ROCK_RodPnt_L'][1] += res.x[2]
opt_hp['ROCK_RodPnt_L'][2] += res.x[3]
opt_hp['ROCK_CoiPnt_L'][0] += res.x[4]
opt_hp['ROCK_CoiPnt_L'][2] += res.x[5]
opt_hp['CHAS_AttPnt_L'][2] += res.x[6]

rc_z, camber_roll, bump_steer, mr = evaluate_rear_kinematics(opt_hp)
l_damper = np.linalg.norm(opt_hp['CHAS_AttPnt_L'] - opt_hp['ROCK_CoiPnt_L'])

# ==============================================================================
# 4. REPORTE TÉCNICO
# ==============================================================================
print("\n" + "═" * 110)
print("  PROJECT-GP · OPTIMIZACIÓN TRASERA (VELIS 2 - MANGUETA 100% CONGELADA)")
print("═" * 110)
print(f"  • Bump Steer Residual          : {bump_steer:.6f}  (0.000°/mm ideal)")
print(f"  • Roll Center Z (sobre suelo)  : {rc_z:.2f} mm    (Fijo por mangueta Velis 2)")
print(f"  • Recuperación de Caída en Roll: {abs(camber_roll)*100:.1f}% ({camber_roll:.3f}°/°)")
print(f"  • Motion Ratio Heave           : {mr:.3f}       (Target: 1.18–1.20)")
print(f"  • Longitud Amortiguador Reposo : {l_damper:.2f} mm   (Target: 180.00 mm)")

print("\n" + "─" * 110)
print(f"  {'PESTAÑA OK':<14} | {'POINT NAME (OK)':<17} | {'ORIGINAL [X, Y, Z]':<26} | {'OPTIMIZADO [X, Y, Z]':<26} | {'DELTA [ΔX, ΔY, ΔZ]'}")
print("─" * 110)

pts_order = [
    'CHAS_LowFor', 'CHAS_LowAft', 'CHAS_UppFor', 'CHAS_UppAft',
    'UPRI_LowPnt', 'UPRI_UppPnt', 'CHAS_TiePnt', 'UPRI_TiePnt',
    'NSMA_PPAttPnt_L', 'CHAS_AttPnt_L', 'CHAS_RocAxi_L', 'CHAS_RocPiv_L',
    'ROCK_RodPnt_L', 'ROCK_CoiPnt_L'
]

for p in pts_order:
    tab, desc, modifiable = DESCRIPTIONS[p]
    orig = HP_VELIS2[p]
    opt = opt_hp[p]
    diff = opt - orig
    
    orig_str = f"[{orig[0]:7.2f}, {orig[1]:7.2f}, {orig[2]:7.2f}]"
    opt_str  = f"[{opt[0]:7.2f}, {opt[1]:7.2f}, {opt[2]:7.2f}]"
    diff_str = f"[{diff[0]:+5.1f}, {diff[1]:+5.1f}, {diff[2]:+5.1f}]"
    
    if not modifiable:
        tag = " [FROZEN]"
    elif np.linalg.norm(diff) > 1e-3:
        tag = " *MODIFICADO"
    else:
        tag = "  INTACTO"
        
    print(f"  {tab:<12} | {p:<17} | {orig_str:<26} | {opt_str:<26} | {diff_str} {tag}")

print("─" * 110)
print("  (*) Puntos a actualizar en Optimum Kinematics.")

print("\n" + "─" * 110)
print("  COORDENADAS PARA COPIAR Y PEGAR EN OPTIMUM KINEMATICS (Left / Right)")
print("─" * 110)
for p in ['CHAS_TiePnt', 'CHAS_AttPnt_L', 'ROCK_RodPnt_L', 'ROCK_CoiPnt_L']:
    opt = opt_hp[p]
    print(f"  {p:<17} Left: [{opt[0]:9.3f}, {opt[1]:9.3f}, {opt[2]:9.3f}]  |  Right: [{opt[0]:9.3f}, {-opt[1]:9.3f}, {opt[2]:9.3f}]")
print("═" * 110 + "\n")