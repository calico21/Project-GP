#!/usr/bin/env python3
"""
scripts/spring_arb_sizing.py
Tecnun eRacing · Formula Student Electric — Ter27 4WD
================================================================================
DISEÑO (no verificación) de rigideces de muelle y ARB para decisión de compra.

HISTORIAL DE ESTE SCRIPT (por qué tiene esta forma y no otra):

  v1: partía de muelles YA elegidos y calculaba qué frecuencia/reparto de
      balanceo resultaban -> eso es verificación, no ayuda a comprar.
  v2: invertía el problema (objetivo -> rigidez necesaria) pero RE-DERIVABA
      motion ratio, centro de balanceo y anti-dive desde cero con métodos
      geométricos propios (trabajo virtual, FVSA, SVSA) sin poder validarlos,
      y encima usaba una clave 'contact_patch' que no existe en el hp real.
  v3 (esta versión): tras inspeccionar vehicle_params_ter27 real, resulta que
      h_rc_f, h_rc_r, anti_dive_f, anti_dive_r, anti_squat, motion_ratio_f_poly
      y motion_ratio_r_poly YA ESTÁN en el config (casi seguro exportados de
      OptimumKinematics) -> se leen DIRECTAMENTE en vez de re-derivarlos. Es
      más fiable usar un dato ya validado por otra herramienta que reinventar
      el cálculo con mis propios métodos sin poder contrastarlos.

QUÉ SIGUE SIN RESOLVERSE / SUPUESTOS EXPLÍCITOS (no se han adivinado a ciegas):

  - Unidades de `arb_rate_f` / `arb_rate_r`: podría ser N/m (equivalente en
    rueda, mismo convenio que `spring_rate_f/r`) o Nm/rad (rigidez torsional
    pura de la barra). Se informa el resultado recomendado EN AMBAS unidades
    para que se pueda cotejar cuál de las dos hace que el valor actual (5000)
    tenga sentido físico, en vez de asumir una y arriesgarse a estar mal en
    todo el informe.
  - Signo/convenio de `motion_ratio_f_poly` respecto al heave (¿z positivo es
    compresión o extensión?): no afecta al PUNTO ESTÁTICO usado para
    dimensionar (poly[0], el término independiente, es el MR a z=0 sea cual
    sea el convenio), pero si se quiere usar la parte de rising-rate hay que
    confirmar el signo antes de fiarse de la dirección del efecto.
================================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

try:
    from config.vehicles.ter27 import vehicle_params_ter27
    HAS_VEHICLE_CONFIG = True
except ImportError:
    HAS_VEHICLE_CONFIG = False


# ==============================================================================
# 1. OBJETIVOS DE DISEÑO (lo único que decides tú)
# ==============================================================================

@dataclass
class DesignTargets:
    fn_f_hz: float = 3.6
    olley_ratio: float = 1.0          # fn_r / fn_f objetivo
    roll_angle_max_deg: float = 1.2
    lat_accel_g: float = 1.8
    tlltd_front: float = 0.45          # fracción de rigidez a balanceo TOTAL en el eje delantero
    brake_decel_g: float = 2.0


SPRING_CATALOG_N_MM: List[float] = [16, 18, 20, 22, 24, 26, 28, 30, 32, 35, 38, 40,
                                     42, 44, 46, 48, 50, 53, 56, 60, 65, 70, 75, 80]


def _nearest(value: float, catalog: List[float]) -> float:
    return min(catalog, key=lambda c: abs(c - value))


# ==============================================================================
# 2. LECTURA DIRECTA DEL VEHÍCULO (sin re-derivar lo que ya está en el config)
# ==============================================================================

def load_vehicle(p: dict) -> dict:
    m_total = p['total_mass']
    front_frac = p['mass_dist']
    wheelbase = p['wheelbase']
    m_us_f, m_us_r = p['unsprung_mass_f'], p['unsprung_mass_r']

    m_front = m_total * front_frac
    m_rear = m_total * (1.0 - front_frac)
    m_s_f = m_front / 2.0 - m_us_f
    m_s_r = m_rear / 2.0 - m_us_r

    return dict(
        m_total=m_total, front_frac=front_frac, wheelbase=wheelbase,
        lf=p['lf'], lr=p['lr'],
        m_s_f=m_s_f, m_s_r=m_s_r,
        h_cg_sprung=p['h_cg_sprung'],
        track_f=p['track_front'], track_r=p['track_rear'],
        h_rc_f=p['h_rc_f'], h_rc_r=p['h_rc_r'],
        anti_dive_f=p['anti_dive_f'], anti_dive_r=p['anti_dive_r'],
        anti_squat_r=p['anti_squat'], anti_squat_f=p['anti_squat_f'],
        brake_bias_f=p['brake_bias_f'],
        MR_f=p['motion_ratio_f_poly'][0], MR_r=p['motion_ratio_r_poly'][0],
        Ks_f_current=p['spring_rate_f'] / 1000.0,   # N/m -> N/mm
        Ks_r_current=p['spring_rate_r'] / 1000.0,
        arb_rate_f_current=p['arb_rate_f'],
        arb_rate_r_current=p['arb_rate_r'],
        bump_travel_available_mm=25.0,   # <- AJUSTAR si tenéis el dato real de recorrido a tope
    )


# ==============================================================================
# 3. DIMENSIONADO DE MUELLES
# ==============================================================================

def size_springs(v: dict, targets: DesignTargets) -> dict:
    fn_f, fn_r = targets.fn_f_hz, targets.fn_f_hz * targets.olley_ratio

    Kw_f_req = (2.0 * np.pi * fn_f) ** 2 * v['m_s_f']
    Kw_r_req = (2.0 * np.pi * fn_r) ** 2 * v['m_s_r']

    Ks_f_req = Kw_f_req * v['MR_f'] ** 2 / 1000.0
    Ks_r_req = Kw_r_req * v['MR_r'] ** 2 / 1000.0

    Ks_f_sel = _nearest(Ks_f_req, SPRING_CATALOG_N_MM)
    Ks_r_sel = _nearest(Ks_r_req, SPRING_CATALOG_N_MM)

    Kw_f_act = Ks_f_sel * 1000.0 / v['MR_f'] ** 2
    Kw_r_act = Ks_r_sel * 1000.0 / v['MR_r'] ** 2

    fn_f_act = (1.0 / (2.0 * np.pi)) * np.sqrt(Kw_f_act / v['m_s_f'])
    fn_r_act = (1.0 / (2.0 * np.pi)) * np.sqrt(Kw_r_act / v['m_s_r'])

    return dict(fn_f_target=fn_f, fn_r_target=fn_r, Ks_f_req=Ks_f_req, Ks_r_req=Ks_r_req,
                Ks_f_sel=Ks_f_sel, Ks_r_sel=Ks_r_sel, Kw_f_act=Kw_f_act, Kw_r_act=Kw_r_act,
                fn_f_act=fn_f_act, fn_r_act=fn_r_act)


# ==============================================================================
# 4. DIMENSIONADO DE ARB
# ==============================================================================

def size_arb(v: dict, targets: DesignTargets, sp: dict) -> dict:
    K_phi_spring_f = 0.5 * sp['Kw_f_act'] * v['track_f'] ** 2
    K_phi_spring_r = 0.5 * sp['Kw_r_act'] * v['track_r'] ** 2

    phi_max_rad = np.radians(targets.roll_angle_max_deg)
    a_lat = targets.lat_accel_g * 9.81
    m_s_total = 2.0 * (v['m_s_f'] + v['m_s_r'])

    z_rc_at_cg = v['h_rc_f'] + (v['h_rc_r'] - v['h_rc_f']) * (v['lf'] / v['wheelbase'])
    roll_arm = v['h_cg_sprung'] - z_rc_at_cg
    M_roll = m_s_total * a_lat * roll_arm
    K_phi_total_req = M_roll / phi_max_rad

    K_phi_f_target = targets.tlltd_front * K_phi_total_req
    K_phi_r_target = (1.0 - targets.tlltd_front) * K_phi_total_req

    def _arb_axle(K_phi_arb, track):
        feasible = K_phi_arb > 0
        Kw_equiv_n_per_mm = (2.0 * max(K_phi_arb, 0.0) / track ** 2) / 1000.0
        return dict(feasible=feasible, K_phi_arb_Nm_rad=K_phi_arb,
                    Kw_equiv_N_mm=Kw_equiv_n_per_mm)

    return dict(K_phi_spring_f=K_phi_spring_f, K_phi_spring_r=K_phi_spring_r,
                K_phi_total_req=K_phi_total_req, roll_arm_m=roll_arm,
                front=_arb_axle(K_phi_f_target - K_phi_spring_f, v['track_f']),
                rear=_arb_axle(K_phi_r_target - K_phi_spring_r, v['track_r']))


# ==============================================================================
# 5. PRESUPUESTO DE RECORRIDO EN FRENADA (usa anti_dive_f directo del config)
# ==============================================================================

def brake_travel_budget(v: dict, targets: DesignTargets, sp: dict) -> dict:
    a_x_g = targets.brake_decel_g
    W = v['m_total'] * 9.81
    dFz_front_total = W * a_x_g * (v['h_cg_sprung'] / v['wheelbase'])
    dFz_per_corner = dFz_front_total / 2.0

    elastic_fraction = 1.0 - v['anti_dive_f']
    dz_wheel_mm = (dFz_per_corner * elastic_fraction / sp['Kw_f_act']) * 1000.0
    margin_mm = v['bump_travel_available_mm'] - dz_wheel_mm

    return dict(dFz_per_corner=dFz_per_corner, elastic_fraction=elastic_fraction,
                dz_wheel_mm=dz_wheel_mm, margin_mm=margin_mm)


# ==============================================================================
# 6. INFORME
# ==============================================================================

def print_report(v: dict, targets: DesignTargets, sp: dict, arb: dict, travel: dict):
    print("=" * 96)
    print("  DIMENSIONADO DE MUELLES Y ARB PARA COMPRA — TeR27")
    print("=" * 96)

    print("\n[1] GEOMETRÍA/MASA LEÍDA DIRECTAMENTE DE vehicle_params_ter27 (sin re-derivar)")
    print(f"  MR_f={v['MR_f']:.3f}  MR_r={v['MR_r']:.3f}  "
          f"h_rc_f={v['h_rc_f']*1000:.0f}mm  h_rc_r={v['h_rc_r']*1000:.0f}mm  "
          f"anti_dive_f={v['anti_dive_f']*100:.1f}%")
    print(f"  m_s_f={v['m_s_f']:.2f} kg/esquina  m_s_r={v['m_s_r']:.2f} kg/esquina  "
          f"track_f={v['track_f']*1000:.0f}mm  track_r={v['track_r']*1000:.0f}mm")

    print("\n[2] MUELLES")
    print(f"  {'':26s} {'Delantero':>14s} {'Trasero':>14s}")
    print(f"  {'Ks ACTUAL [N/mm]':26s} {v['Ks_f_current']:14.1f} {v['Ks_r_current']:14.1f}")
    print(f"  {'Ks requerido [N/mm]':26s} {sp['Ks_f_req']:14.2f} {sp['Ks_r_req']:14.2f}")
    print(f"  {'Ks RECOMENDADO [N/mm]':26s} {sp['Ks_f_sel']:14.1f} {sp['Ks_r_sel']:14.1f}   <- COMPRAR")
    print(f"  {'fn resultante [Hz]':26s} {sp['fn_f_act']:14.2f} {sp['fn_r_act']:14.2f}")
    print(f"  {'fn objetivo [Hz]':26s} {sp['fn_f_target']:14.2f} {sp['fn_r_target']:14.2f}")

    print("\n[3] ARB  (unidades ambiguas en el config -> se dan las dos lecturas posibles)")
    print(f"  Rigidez balanceo TOTAL requerida = {arb['K_phi_total_req']:.0f} Nm/rad")
    for label, res, current in [("Delantero", arb['front'], v['arb_rate_f_current']),
                                 ("Trasero", arb['rear'], v['arb_rate_r_current'])]:
        print(f"\n  ARB {label} (arb_rate_{'f' if label=='Delantero' else 'r'} actual en config = {current}):")
        if not res['feasible']:
            print(f"    El muelle recomendado ya cubre de sobra el objetivo en este eje -> "
                  f"no hace falta ARB (o incluso sobra rigidez de muelle para el reparto pedido).")
        else:
            print(f"    Si arb_rate está en Nm/rad -> recomendado = {res['K_phi_arb_Nm_rad']:.0f} Nm/rad "
                  f"(actual={current}: {'coherente en orden de magnitud' if 0.3*res['K_phi_arb_Nm_rad'] < current < 3*res['K_phi_arb_Nm_rad'] else 'NO coherente en orden de magnitud'})")
            print(f"    Si arb_rate está en N/m (equiv. rueda) -> recomendado = "
                  f"{res['Kw_equiv_N_mm']*1000:.0f} N/m = {res['Kw_equiv_N_mm']:.2f} N/mm "
                  f"(actual={current}: {'coherente en orden de magnitud' if 0.3*res['Kw_equiv_N_mm']*1000 < current < 3*res['Kw_equiv_N_mm']*1000 else 'NO coherente en orden de magnitud'})")

    print("\n[4] PRESUPUESTO DE RECORRIDO EN FRENADA (usa anti_dive_f del config, no un supuesto)")
    print(f"  Transferencia de carga por rueda delantera @ {targets.brake_decel_g:.1f}G = "
          f"{travel['dFz_per_corner']:.0f} N")
    print(f"  Fracción que reacciona el muelle (1 - anti_dive_f) = {travel['elastic_fraction']*100:.1f} %")
    print(f"  Hundimiento de rueda estimado                       = {travel['dz_wheel_mm']:.1f} mm")
    print(f"  Recorrido disponible hasta bump-stop (AJUSTAR si procede) = "
          f"{v['bump_travel_available_mm']:.1f} mm")
    print(f"  Margen                                               = {travel['margin_mm']:.1f} mm")


if __name__ == "__main__":
    if not HAS_VEHICLE_CONFIG:
        raise SystemExit("No se encontró config/vehicles/ter27.py")

    v = load_vehicle(vehicle_params_ter27)
    targets = DesignTargets()
    sp = size_springs(v, targets)
    arb = size_arb(v, targets, sp)
    travel = brake_travel_budget(v, targets, sp)
    print_report(v, targets, sp, arb, travel)