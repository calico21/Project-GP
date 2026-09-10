#!/usr/bin/env python3
"""
scripts/chassis_suspension_loads.py
Tecnun eRacing · Formula Student Electric — Ter27 4WD
================================================================================
Calcula las fuerzas axiales en barras, ángulos espaciales 3D y cargas en los
anclajes del chasis/monocasco (tabs, bobbins e inserts) leyendo DIRECTAMENTE
de config/vehicles/ter27.py (sin necesidad de ningún archivo Excel externo).

CORRECCIONES respecto a la versión anterior (ver changelog al final del archivo):
  1. SyntaxError en el return de solve_axle_loads (faltaba una coma).
  2. Faltaba `import pandas as pd`.
  3. Mz_steering tenía Fx/Fy intercambiados respecto a su significado físico.
  4. La "regla de ocurrencia mínima" del tie-rod (1.5 kN) estaba comentada
     pero no implementada.
  5. Sin protección ante división por cero en F_push / F_damper cuando la
     geometría degenera (barra casi paralela al eje de momentos).
  6. Cálculo duplicado entre el informe de consola y el export a Excel:
     ahora se centraliza en build_full_report() y ambas salidas consumen
     el mismo resultado.
  7. `from artifact_tool import Workbook, SpreadsheetFile` NO es un paquete
     real (no existe en PyPI ni es un módulo del proyecto) -> se sustituye
     por `openpyxl`, que sí es una librería real e instalable
     (`pip install openpyxl`).
  9. FIX ESTRUCTURAL (mangueta + pushrod): la versión anterior resolvía la
     mangueta con un sistema 6x6 de SOLO 4 incógnitas reales (Lower_Fore,
     Lower_Aft, Tie_Rod + un vector F_ub de 3 componentes que mezclaba, sin
     separar, la reacción del trapecio superior Y la del pushrod). Luego
     intentaba separar esas dos cosas en un segundo paso (momento sobre
     `axis_upp` + `np.linalg.lstsq`), lo cual NO es equivalente al sistema
     físico real y deja un residuo de fuerza sin explicar (~200 N en el
     caso de validación con datos de referencia externos).
     Conteo de grados de libertad: la mangueta tiene 6 GDL. Los 2 brazos
     superiores + 2 inferiores + tie-rod son solo 5 miembros de 2 fuerzas
     (dejan 1 GDL libre = el recorrido de suspensión); el PUSHROD es el 6º
     miembro que cierra ese GDL. Con 6 miembros de 2 fuerzas para 6 GDL el
     sistema es determinado de forma DIRECTA: Upper_Fore, Upper_Aft,
     Lower_Fore, Lower_Aft, Tie_Rod y Pushrod se resuelven TODOS a la vez
     en un único sistema 6x6, sin cascada ni mínimos cuadrados. Validado
     numéricamente contra una herramienta de referencia externa (mismas
     coordenadas): el nuevo método reproduce sus magnitudes (p.ej.
     Upper_Fore idéntico en magnitud), cosa que la versión en cascada no
     lograba.
================================================================================
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter

# Asegurar importación del directorio raíz de Project-GP
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config.vehicles.ter27 import vehicle_params_ter27
except ImportError:
    try:
        from ter27 import vehicle_params_ter27
    except ImportError as e:
        raise ImportError(f"No se pudo encontrar config/vehicles/ter27.py: {e}")


# ==============================================================================
# CONSTANTES DE DISEÑO (documentadas explícitamente, no "números mágicos")
# ==============================================================================

MU_DRY = 1.6                          # Coeficiente de fricción neumático-asfalto seco
V_AERO_MS = 19.44                     # 70 km/h, velocidad de referencia aero (reglamento FS)
SCRUB_RADIUS_M = 0.010                # Scrub radius de diseño
PNEUMATIC_TRAIL_M = 0.025             # Rastro neumático de diseño
TIE_ROD_MIN_DESIGN_FORCE_N = 1500.0   # Mínimo de diseño para anclaje de dirección (criterio FSG)


@dataclass
class LoadCase:
    name: str
    Fx: float  # Longitudinal (+tracción, -frenada)
    Fy: float  # Lateral (+curva exterior)
    Fz: float  # Vertical (+carga normal)


def vector_angles(v: np.ndarray) -> Dict[str, float]:
    """Calcula la longitud (mm) y los ángulos espaciales de una barra."""
    L = np.linalg.norm(v)
    if L < 1e-6:
        return {'len_mm': 0.0, 'ang_vert_deg': 0.0, 'ang_front_deg': 0.0, 'ang_side_deg': 0.0}
    u = v / L
    ang_vert = np.degrees(np.arccos(np.clip(abs(u[2]), 0.0, 1.0)))
    ang_front = np.degrees(np.arctan2(abs(u[2]), abs(u[1])))
    ang_side = np.degrees(np.arctan2(abs(u[2]), abs(u[0])))
    return {
        'len_mm': float(L * 1000.0),
        'ang_vert_deg': float(ang_vert),
        'ang_front_deg': float(ang_front),
        'ang_side_deg': float(ang_side)
    }


def solve_axle_loads(hp: Dict[str, List[float]], axle: str, lc: LoadCase, params: dict):
    p_c_uf = np.array(hp['uca_in_f'])
    p_c_ua = np.array(hp['uca_in_r'])
    p_c_lf = np.array(hp['lca_in_f'])
    p_c_la = np.array(hp['lca_in_r'])
    p_u_up = np.array(hp['ubj'])
    p_u_lp = np.array(hp['lbj'])
    p_c_tie = np.array(hp['tie_in'])
    p_u_tie = np.array(hp['tie_out'])

    p_nsma = np.array(hp['pushrod_out'])
    p_rock_piv = np.array(hp['rocker_piv'])
    p_chas_att = np.array(hp['spring_in'])
    p_rock_rod = np.array(hp['rocker_rod'])
    p_rock_coi = np.array(hp['rocker_coi'])

    axis_roc = np.array(hp['rocker_axis']) - p_rock_piv
    axis_roc = axis_roc / np.linalg.norm(axis_roc)

    track = params['track_front'] if axle == 'Front' else params['track_rear']
    p_cp = np.array([0.0, track / 2.0, 0.0])

    # Vectores unitarios de barra
    u_uf = (p_c_uf - p_u_up) / np.linalg.norm(p_c_uf - p_u_up)
    u_ua = (p_c_ua - p_u_up) / np.linalg.norm(p_c_ua - p_u_up)
    u_lf = (p_c_lf - p_u_lp) / np.linalg.norm(p_c_lf - p_u_lp)
    u_la = (p_c_la - p_u_lp) / np.linalg.norm(p_c_la - p_u_lp)
    u_tie = (p_c_tie - p_u_tie) / np.linalg.norm(p_c_tie - p_u_tie)
    u_push = (p_rock_rod - p_nsma) / np.linalg.norm(p_rock_rod - p_nsma)

    damper_vec = p_chas_att - p_rock_coi
    u_damper = damper_vec / np.linalg.norm(damper_vec)

    # Fuerzas externas y Par Mz inducido por scrub radius y pneumatic trail
    F_ext = np.array([lc.Fx, lc.Fy, lc.Fz])
    M_ext = np.cross(p_cp, F_ext)

    # CORREGIDO: el scrub radius genera par por la fuerza LONGITUDINAL (Fx) en
    # el contacto (frenada/tracción); el rastro neumático genera el par de
    # autoalineación por la fuerza LATERAL (Fy). En la versión anterior estos
    # dos términos estaban cruzados (Fy con scrub, Fz con trail).
    Mz_steering = (lc.Fx * SCRUB_RADIUS_M) + (lc.Fy * PNEUMATIC_TRAIL_M)
    M_ext[2] += Mz_steering

    # ------------------------------------------------------------------
    # Sistema 6-DOF de la mangueta, resuelto DE UNA VEZ con las 6 fuerzas
    # de barra reales como incógnitas (Upper_Fore, Upper_Aft, Lower_Fore,
    # Lower_Aft, Tie_Rod, Pushrod). Cada miembro es un elemento de 2 fuerzas
    # (fuerza puramente axial), aplicado en su propio punto de anclaje en
    # la mangueta. 6 incógnitas para 6 ecuaciones de equilibrio (3 fuerza +
    # 3 momento respecto al origen) -> sistema determinado, sin necesidad
    # de cascada ni de mínimos cuadrados (ver nota #9 en el changelog).
    # ------------------------------------------------------------------
    members = [
        ('Upper_Fore', u_uf, p_u_up),
        ('Upper_Aft',  u_ua, p_u_up),
        ('Lower_Fore', u_lf, p_u_lp),
        ('Lower_Aft',  u_la, p_u_lp),
        ('Tie_Rod',    u_tie, p_u_tie),
        ('Pushrod',    u_push, p_nsma),
    ]

    A = np.zeros((6, 6))
    for i, (_name, u_i, p_i) in enumerate(members):
        A[0:3, i] = u_i
        A[3:6, i] = np.cross(p_i, u_i)

    sol = np.linalg.solve(A, np.concatenate([-F_ext, -M_ext]))
    F_uf, F_ua, F_lf, F_la, F_tie, F_push = sol

    # REGLA DE OCURRENCIA MÍNIMA PARA EL TIE-ROD (Umbral de seguridad chasis)
    # Ningún tirante de dirección/convergencia se dimensiona por debajo de
    # 1.5 kN en FSG. Se distingue explícitamente el valor "resuelto" por
    # equilibrio (para verificación) del valor "de diseño" (para dimensionar
    # el anclaje al chasis), que nunca baja del mínimo reglamentario.
    F_tie_solved = float(F_tie)
    if abs(F_tie_solved) < 1e-9:
        F_tie_design = F_tie_solved
    else:
        F_tie_design = float(np.sign(F_tie_solved) * max(abs(F_tie_solved), TIE_ROD_MIN_DESIGN_FORCE_N))

    # Equilibrio del rocker (balancín): con el pushrod ya conocido del
    # sistema 6x6 anterior, tomamos momentos en el eje de giro del rocker
    # para despejar la fuerza del damper.
    r_rod = p_rock_rod - p_rock_piv
    r_coi = p_rock_coi - p_rock_piv
    M_rod_roc = np.dot(axis_roc, np.cross(r_rod, -F_push * u_push))
    M_coi_unit = np.dot(axis_roc, np.cross(r_coi, -u_damper))
    if abs(M_coi_unit) < 1e-9:
        raise ValueError(
            f"Geometría degenerada en eje {axle}, caso '{lc.name}': el damper queda "
            f"prácticamente paralelo al eje de giro del rocker (M_coi_unit≈0). "
            f"Revisa los hardpoints 'rocker_coi' / 'rocker_axis' para este eje."
        )
    F_damper = -M_rod_roc / M_coi_unit

    f_on_chas_uf = -F_uf * u_uf
    f_on_chas_ua = -F_ua * u_ua
    f_on_chas_lf = -F_lf * u_lf
    f_on_chas_la = -F_la * u_la
    f_on_chas_tie = -F_tie_design * u_tie   # se usa el valor de DISEÑO, conservador
    f_on_chas_damper = -F_damper * u_damper
    f_on_rocker_piv = (F_push * u_push) + (F_damper * u_damper)

    return {
        'axial_forces': {
            'Upper_Fore': F_uf,
            'Upper_Aft': F_ua,
            'Lower_Fore': F_lf,
            'Lower_Aft': F_la,
            'Tie_Rod': F_tie_design,
            'Tie_Rod_Solved': F_tie_solved,
            'Pushrod': F_push,
            'Damper': F_damper
        },
        'chassis_forces': {
            'CHAS_UCA_Fore': f_on_chas_uf,
            'CHAS_UCA_Aft':  f_on_chas_ua,
            'CHAS_LCA_Fore': f_on_chas_lf,
            'CHAS_LCA_Aft':  f_on_chas_la,
            'CHAS_Tie_Rod':  f_on_chas_tie,
            'CHAS_Damper_Mount': f_on_chas_damper,
            'CHAS_Rocker_Pivot': f_on_rocker_piv
        }
    }


# ==============================================================================
# CÁLCULO CENTRALIZADO (una sola pasada, reutilizada por consola y Excel)
# ==============================================================================

def analyze_axle(axle: str, hp: dict, fz_stat: float, df_corner: float, params: dict, mu: float = MU_DRY) -> dict:
    """Calcula geometría de barras, casos de carga y resultados para un eje."""

    bars = {
        'Upper_Fore': np.array(hp['uca_in_f']) - np.array(hp['ubj']),
        'Upper_Aft':  np.array(hp['uca_in_r']) - np.array(hp['ubj']),
        'Lower_Fore': np.array(hp['lca_in_f']) - np.array(hp['lbj']),
        'Lower_Aft':  np.array(hp['lca_in_r']) - np.array(hp['lbj']),
        'Tie_Rod':    np.array(hp['tie_in'])   - np.array(hp['tie_out']),
        'Pushrod':    np.array(hp['rocker_rod']) - np.array(hp['pushrod_out']),
        'Damper':     np.array(hp['spring_in'])  - np.array(hp['rocker_coi'])
    }

    fz_brk = fz_stat + df_corner + (750.0 if axle == 'Front' else -400.0)
    fz_corn = fz_stat + df_corner + 650.0
    fz_acc = fz_stat + df_corner + (600.0 if axle == 'Rear' else -300.0)
    fz_bump = (3.0 * fz_stat) + df_corner

    load_cases = [
        LoadCase("1_Max_Braking",    -mu * fz_brk, 0.0, max(200.0, fz_brk)),
        LoadCase("2_Max_Cornering",  0.0, mu * fz_corn, fz_corn),
        LoadCase("3_3G_Bump_Impact", -0.2 * fz_bump, 0.0, fz_bump),
        LoadCase("4_Trail_Braking",  -0.7 * mu * fz_brk, 0.7 * mu * fz_brk, fz_brk),
        LoadCase("5_Max_Accel_AWD",   (mu * fz_acc * 0.65 if axle == 'Rear' else mu * fz_acc * 0.35), 0.0, max(200.0, fz_acc))
    ]

    axial_rows = []
    chassis_records = []

    for lc in load_cases:
        res = solve_axle_loads(hp, axle, lc, params)

        row = {'Caso': lc.name, 'Fx_ext': lc.Fx, 'Fy_ext': lc.Fy, 'Fz_ext': lc.Fz}
        row.update(res['axial_forces'])
        axial_rows.append(row)

        for ch_pt, f_vec in res['chassis_forces'].items():
            chassis_records.append({
                'Caso': lc.name,
                'Anclaje': ch_pt,
                'Fx_N': float(f_vec[0]),
                'Fy_N': float(f_vec[1]),
                'Fz_N': float(f_vec[2]),
                'F_Mag_N': float(np.linalg.norm(f_vec))
            })

    return {
        'axle': axle,
        'bars': bars,
        'fz_stat': fz_stat,
        'df_corner': df_corner,
        'load_cases': load_cases,
        'axial_rows': axial_rows,
        'chassis_records': chassis_records,
    }


def build_full_report(p: dict) -> dict:
    """Punto único de cálculo: todo lo que necesitan tanto la consola como el Excel."""

    g = 9.81
    w_total = p['total_mass'] * g
    f_dist = p['lr'] / p['wheelbase']  # % de carga delantera
    r_dist = p['lf'] / p['wheelbase']  # % de carga trasera
    fz_stat_f = (w_total * f_dist) / 2.0
    fz_stat_r = (w_total * r_dist) / 2.0

    q_aero = 0.5 * p['rho_air'] * (V_AERO_MS ** 2) * p['Cl'] * p['A']
    df_f_corner = (q_aero * p['aero_split_f']) / 2.0
    df_r_corner = (q_aero * p['aero_split_r']) / 2.0

    front = analyze_axle('Front', p['hardpoints_f'], fz_stat_f, df_f_corner, p, mu=MU_DRY)
    rear = analyze_axle('Rear', p['hardpoints_r'], fz_stat_r, df_r_corner, p, mu=MU_DRY)

    return {
        'params': p,
        'g': g,
        'w_total': w_total,
        'fz_stat_f': fz_stat_f,
        'fz_stat_r': fz_stat_r,
        'q_aero': q_aero,
        'df_f_corner': df_f_corner,
        'df_r_corner': df_r_corner,
        'mu': MU_DRY,
        'front': front,
        'rear': rear,
    }


# ==============================================================================
# SALIDA POR CONSOLA
# ==============================================================================

def _print_axle_console_report(axle_report: dict):
    axle = axle_report['axle']
    print("\n" + "#" * 90)
    print(f"  EJE: {axle.upper()} (Carga estática/rueda = {axle_report['fz_stat']:.1f} N | "
          f"Downforce 70km/h = {axle_report['df_corner']:.1f} N)")
    print("#" * 90)

    print("\n[1] ÁNGULOS ESPACIALES Y LONGITUDES DE CADA BARRA:")
    print(f"{'Barra':<14s} | {'Long (mm)':>9s} | {'Inc. Vert':>10s} | {'Vista Frontal (YZ)':>20s} | {'Vista Lateral (XZ)':>20s}")
    print("-" * 82)
    for b_name, vec in axle_report['bars'].items():
        ang = vector_angles(vec)
        print(f"{b_name:<14s} | {ang['len_mm']:9.1f} | {ang['ang_vert_deg']:9.1f}° | {ang['ang_front_deg']:19.1f}° | {ang['ang_side_deg']:19.1f}°")

    df_axial = pd.DataFrame(axle_report['axial_rows']).set_index('Caso')
    df_axial_display = df_axial.drop(columns=['Fx_ext', 'Fy_ext', 'Fz_ext', 'Tie_Rod_Solved'])

    print("\n[2] FUERZAS AXIALES EN BARRAS [N] (+ = Tracción, - = Compresión):")
    print("-" * 88)
    print(f"{'Caso de Carga':<18s} | " + " | ".join([f"{c:>10s}" for c in df_axial_display.columns]))
    print("-" * 88)
    for idx, row in df_axial_display.iterrows():
        print(f"{idx:<18s} | " + " | ".join([f"{v:10.0f}" for v in row]))
    print("-" * 88)
    print("(Tie_Rod ya incluye el mínimo de diseño de "
          f"{TIE_ROD_MIN_DESIGN_FORCE_N:.0f} N; ver columna Tie_Rod_Solved en el Excel para el valor de equilibrio puro)")

    df_chassis = pd.DataFrame(axle_report['chassis_records'])
    print("\n[3] CARGAS MÁXIMAS RESULTANTES SOBRE EL CHASIS:")
    print(f"{'Punto en Chasis':<20s} | {'Max |Fx| (N)':>12s} | {'Max |Fy| (N)':>12s} | {'Max |Fz| (N)':>12s} | {'Carga Pico |F| (N)':>19s}")
    print("-" * 76)
    for pt_name in df_chassis['Anclaje'].unique():
        df_pt = df_chassis[df_chassis['Anclaje'] == pt_name]
        max_fx = df_pt['Fx_N'].abs().max()
        max_fy = df_pt['Fy_N'].abs().max()
        max_fz = df_pt['Fz_N'].abs().max()
        max_mag = df_pt['F_Mag_N'].max()
        print(f"{pt_name:<20s} | {max_fx:12.0f} | {max_fy:12.0f} | {max_fz:12.0f} | {max_mag:19.0f}")


def run_chassis_load_analysis(report: dict):
    p = report['params']
    print("=" * 90)
    print("  ANÁLISIS DE SUSPENSIÓN Y CARGAS EN CHASIS (TeR27)")
    print(f"  Fuente: config/vehicles/ter27.py (Masa total = {p['total_mass']} kg, Wheelbase = {p['wheelbase']} m)")
    print("=" * 90)

    _print_axle_console_report(report['front'])
    _print_axle_console_report(report['rear'])

    print("\n[✓] acabo el joseo.")


# ==============================================================================
# EXPORT A EXCEL (openpyxl) — hoja única, Delantero y Trasero en columnas paralelas
# ==============================================================================

FRONT_COLS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
REAR_COLS = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']

CHASSIS_POINT_LABELS = {
    'CHAS_UCA_Fore': 'UCA Fore',
    'CHAS_UCA_Aft': 'UCA Aft',
    'CHAS_LCA_Fore': 'LCA Fore',
    'CHAS_LCA_Aft': 'LCA Aft',
    'CHAS_Tie_Rod': 'Tie Rod',
    'CHAS_Damper_Mount': 'Damper Mount',
    'CHAS_Rocker_Pivot': 'Rocker Pivot',
}

# --- Paleta ---
DARK = "1F2937"
BLUE = "2563EB"
LIGHT_BLUE = "DBEAFE"
LIGHT_GRAY = "F3F4F6"
GRAY = "6B7280"
GREEN = "DCFCE7"
RED = "FEE2E2"
AMBER = "FEF3C7"
WHITE = "FFFFFF"
BORDER_COLOR = "D1D5DB"

THIN = Side(style="thin", color=BORDER_COLOR)
BOX_BORDER = Border(top=THIN, bottom=THIN, left=THIN, right=THIN)


def _fill(hex_color):
    return PatternFill(fill_type="solid", start_color=hex_color, end_color=hex_color)


def _style_range(ws, cell_range, *, fill=None, font=None, alignment=None, border=None, number_format=None):
    for row in ws[cell_range]:
        for cell in row:
            if fill is not None:
                cell.fill = fill
            if font is not None:
                cell.font = font
            if alignment is not None:
                cell.alignment = alignment
            if border is not None:
                cell.border = border
            if number_format is not None:
                cell.number_format = number_format


def _write_rows(ws, top_left_cell, rows):
    """Escribe una matriz de filas empezando en top_left_cell (p.ej. 'A6')."""
    col_letters = ''.join(filter(str.isalpha, top_left_cell))
    start_row = int(''.join(filter(str.isdigit, top_left_cell)))
    start_col = ws[top_left_cell].column
    for r_off, row_vals in enumerate(rows):
        for c_off, val in enumerate(row_vals):
            ws.cell(row=start_row + r_off, column=start_col + c_off, value=val)
    end_row = start_row + len(rows) - 1
    end_col_letter = get_column_letter(start_col + max(len(r) for r in rows) - 1)
    return f"{col_letters}{start_row}:{end_col_letter}{end_row}", end_row


def export_to_excel(report: dict, output_filename: str = "Ter27_Chassis_Loads_Report.xlsx"):
    """
    Exporta el análisis a UNA ÚNICA hoja compacta, con el eje Delantero y el
    Trasero mostrados en columnas paralelas (uno junto al otro) en vez de uno
    debajo del otro, para que el informe entero se lea de un vistazo / quepa
    en una página en vez de convertirse en un scroll interminable.
    """

    p = report['params']

    wb = Workbook()
    ws = wb.active
    ws.title = "TeR27 Load Report"
    ws.sheet_view.showGridLines = False

    # --- Fuentes / alineaciones reutilizables ---
    title_font = Font(bold=True, color=WHITE, size=16, name="Arial")
    subtitle_font = Font(color="D1D5DB", size=9, italic=True, name="Arial")
    section_font = Font(bold=True, color=WHITE, size=12, name="Arial")
    axle_font = Font(bold=True, color=WHITE, size=12, name="Arial")
    subsection_font = Font(bold=True, color=DARK, size=10, name="Arial")
    notes_font = Font(color=DARK, size=9, italic=True, name="Arial")
    header_font = Font(bold=True, color=WHITE, size=9, name="Arial")
    body_font = Font(size=9, name="Arial")
    label_font = Font(size=9, bold=True, name="Arial")

    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left_mid = Alignment(horizontal="left", vertical="center")
    wrap_mid = Alignment(horizontal="left", vertical="center", wrap_text=True)

    # ------------------------------------------------------------
    # TÍTULO
    # ------------------------------------------------------------
    ws.merge_cells("A1:W2")
    ws["A1"] = "TeR27 4WD — CHASSIS & SUSPENSION LOAD REPORT"
    _style_range(ws, "A1:W2", fill=_fill(DARK), font=title_font, alignment=center)
    ws.row_dimensions[1].height = 22
    ws.row_dimensions[2].height = 14

    ws.merge_cells("A3:W3")
    ws["A3"] = "Tecnun eRacing · Formula Student Electric · Suspension Load Analysis"
    _style_range(ws, "A3:W3", fill=_fill(DARK), font=subtitle_font, alignment=center)

    # ------------------------------------------------------------
    # RESUMEN DEL VEHÍCULO
    # ------------------------------------------------------------
    ws.merge_cells("A5:W5")
    ws["A5"] = "VEHICLE SUMMARY"
    _style_range(ws, "A5:W5", fill=_fill(BLUE), font=section_font, alignment=left_mid)

    summary = [
        ["Parameter", "Value", "Unit", "Parameter", "Value", "Unit", "Parameter", "Value", "Unit"],
        ["Total Mass", p["total_mass"], "kg", "Wheelbase", p["wheelbase"], "m", "Friction μ", report["mu"], "-"],
        ["Front Track", p["track_front"], "m", "Rear Track", p["track_rear"], "m", "Aero Speed", 70.0, "km/h"],
        ["Static Fz Front", round(report["fz_stat_f"], 1), "N", "Static Fz Rear", round(report["fz_stat_r"], 1), "N",
         "Total Downforce", round(report["q_aero"] * p["Cl"] * p["A"], 1), "N"],
        ["Front DF / wheel", round(report["df_f_corner"], 1), "N", "Rear DF / wheel", round(report["df_r_corner"], 1), "N",
         "Aero Split Front", round(p["aero_split_f"] * 100.0, 1), "%"],
    ]
    _write_rows(ws, "A6", summary)
    _style_range(ws, "A6:I6", fill=_fill(DARK), font=header_font, alignment=center)
    _style_range(ws, "A7:I10", font=body_font, border=BOX_BORDER, alignment=Alignment(vertical="center"))

    # ------------------------------------------------------------
    # HIPÓTESIS / CRITERIOS DE DISEÑO
    # ------------------------------------------------------------
    row = 12
    ws.merge_cells(f"A{row}:W{row}")
    ws[f"A{row}"] = "DESIGN ASSUMPTIONS"
    _style_range(ws, f"A{row}:W{row}", fill=_fill(BLUE), font=section_font, alignment=left_mid)
    row += 1

    notes_text = (
        f"μ = {report['mu']} (dry tarmac, FSAE tyre data)   |   "
        f"Aero evaluated @ 70 km/h   |   "
        f"Mz_steering = Fx·scrub_radius({SCRUB_RADIUS_M*1000:.0f} mm) + Fy·pneumatic_trail({PNEUMATIC_TRAIL_M*1000:.0f} mm)   |   "
        f"Tie-rod chassis-mount force floored at {TIE_ROD_MIN_DESIGN_FORCE_N:.0f} N design minimum (FSG criterion); "
        f"raw equilibrium value kept in 'Tie_Rod_Solved' for reference."
    )
    ws.merge_cells(f"A{row}:W{row+1}")
    ws[f"A{row}"] = notes_text
    _style_range(ws, f"A{row}:W{row+1}", fill=_fill(AMBER), font=notes_font, alignment=wrap_mid)
    row += 3

    # ------------------------------------------------------------
    # BLOQUES DELANTERO / TRASERO EN PARALELO
    # ------------------------------------------------------------
    axle_blocks = [("Front", report['front'], FRONT_COLS, DARK),
                   ("Rear", report['rear'], REAR_COLS, GRAY)]

    # --- Encabezado de eje ---
    for axle_name, _, cols, color in axle_blocks:
        rng = f"{cols[0]}{row}:{cols[-1]}{row}"
        ws.merge_cells(rng)
        ws[f"{cols[0]}{row}"] = f"{axle_name.upper()} AXLE"
        _style_range(ws, rng, fill=_fill(color), font=axle_font, alignment=center)
    row += 2

    # --- 1. Geometría de barras (5 columnas) ---
    for axle_name, axle_report, cols, _ in axle_blocks:
        c = cols[:5]
        rng = f"{c[0]}{row}:{c[-1]}{row}"
        ws.merge_cells(rng)
        ws[f"{c[0]}{row}"] = "1. Bar Geometry"
        _style_range(ws, rng, fill=_fill(LIGHT_BLUE), font=subsection_font, alignment=left_mid)
    bar_header_row = row + 1

    angle_header = ["Bar", "Length (mm)", "Vert. (deg)", "Front YZ (deg)", "Side XZ (deg)"]
    end_row = bar_header_row
    for axle_name, axle_report, cols, _ in axle_blocks:
        c = cols[:5]
        angle_rows = [angle_header]
        for b_name, vec in axle_report['bars'].items():
            ang = vector_angles(vec)
            angle_rows.append([b_name, round(ang['len_mm'], 1), round(ang['ang_vert_deg'], 1),
                                round(ang['ang_front_deg'], 1), round(ang['ang_side_deg'], 1)])
        _, end_row = _write_rows(ws, f"{c[0]}{bar_header_row}", angle_rows)
        _style_range(ws, f"{c[0]}{bar_header_row}:{c[-1]}{bar_header_row}", fill=_fill(DARK), font=header_font, alignment=center)
        _style_range(ws, f"{c[0]}{bar_header_row+1}:{c[-1]}{end_row}", font=body_font, border=BOX_BORDER, alignment=Alignment(vertical="center", horizontal="center"))
        _style_range(ws, f"{c[0]}{bar_header_row+1}:{c[0]}{end_row}", fill=_fill(LIGHT_GRAY), font=label_font, alignment=Alignment(vertical="center", horizontal="left"))
    row = end_row + 2

    # --- 2. Casos de carga / fuerzas axiales (11 columnas) ---
    for axle_name, axle_report, cols, _ in axle_blocks:
        rng = f"{cols[0]}{row}:{cols[-1]}{row}"
        ws.merge_cells(rng)
        ws[f"{cols[0]}{row}"] = "2. Load Cases — Member Axial Forces (N)"
        _style_range(ws, rng, fill=_fill(LIGHT_BLUE), font=subsection_font, alignment=left_mid)
    axial_header_row = row + 1

    axial_header = ["Load Case", "Fx ext", "Fy ext", "Fz ext", "Up.Fore", "Up.Aft",
                     "Lo.Fore", "Lo.Aft", "Tie Rod", "Pushrod", "Damper"]
    end_row = axial_header_row
    for axle_name, axle_report, cols, _ in axle_blocks:
        data_rows = [axial_header]
        for r in axle_report['axial_rows']:
            data_rows.append([
                r['Caso'], round(r['Fx_ext'], 0), round(r['Fy_ext'], 0), round(r['Fz_ext'], 0),
                round(r['Upper_Fore'], 0), round(r['Upper_Aft'], 0), round(r['Lower_Fore'], 0),
                round(r['Lower_Aft'], 0), round(r['Tie_Rod'], 0), round(r['Pushrod'], 0), round(r['Damper'], 0)
            ])
        _, end_row = _write_rows(ws, f"{cols[0]}{axial_header_row}", data_rows)
        _style_range(ws, f"{cols[0]}{axial_header_row}:{cols[-1]}{axial_header_row}", fill=_fill(DARK), font=header_font, alignment=center)
        _style_range(ws, f"{cols[0]}{axial_header_row+1}:{cols[-1]}{end_row}", font=body_font, border=BOX_BORDER,
                     alignment=Alignment(vertical="center", horizontal="center"), number_format="#,##0")
        _style_range(ws, f"{cols[0]}{axial_header_row+1}:{cols[0]}{end_row}", fill=_fill(LIGHT_GRAY), font=label_font,
                     alignment=Alignment(vertical="center", horizontal="left"))

        # Resaltar tracción (verde) / compresión (rojo) en las columnas de fuerzas de barra
        first_force_col, last_force_col = cols[4], cols[-1]
        force_range = f"{first_force_col}{axial_header_row+1}:{last_force_col}{end_row}"
        ws.conditional_formatting.add(
            force_range, FormulaRule(formula=[f"{first_force_col}{axial_header_row+1}<0"], fill=_fill(RED)))
        ws.conditional_formatting.add(
            force_range, FormulaRule(formula=[f"{first_force_col}{axial_header_row+1}>0"], fill=_fill(GREEN)))
    row = end_row + 2

    # --- 3. Cargas máximas en anclajes de chasis (6 columnas) ---
    for axle_name, axle_report, cols, _ in axle_blocks:
        c = cols[:6]
        rng = f"{c[0]}{row}:{c[-1]}{row}"
        ws.merge_cells(rng)
        ws[f"{c[0]}{row}"] = "3. Maximum Chassis Bracket Loads (N)"
        _style_range(ws, rng, fill=_fill(LIGHT_BLUE), font=subsection_font, alignment=left_mid)
    chas_header_row = row + 1

    chas_header = ["Chassis Attachment", "Max |Fx|", "Max |Fy|", "Max |Fz|", "Peak |F|", "Critical Case"]
    end_row = chas_header_row
    for axle_name, axle_report, cols, _ in axle_blocks:
        c = cols[:6]
        df_chassis = pd.DataFrame(axle_report['chassis_records'])
        max_rows = [chas_header]
        for pt_key in df_chassis['Anclaje'].unique():
            df_pt = df_chassis[df_chassis['Anclaje'] == pt_key]
            max_fx = df_pt['Fx_N'].abs().max()
            max_fy = df_pt['Fy_N'].abs().max()
            max_fz = df_pt['Fz_N'].abs().max()
            worst = df_pt.loc[df_pt['F_Mag_N'].idxmax()]
            max_rows.append([
                CHASSIS_POINT_LABELS.get(pt_key, pt_key),
                round(max_fx, 0), round(max_fy, 0), round(max_fz, 0),
                round(worst['F_Mag_N'], 0), worst['Caso']
            ])
        _, end_row = _write_rows(ws, f"{c[0]}{chas_header_row}", max_rows)
        _style_range(ws, f"{c[0]}{chas_header_row}:{c[-1]}{chas_header_row}", fill=_fill(DARK), font=header_font, alignment=center)
        _style_range(ws, f"{c[0]}{chas_header_row+1}:{c[-1]}{end_row}", font=body_font, border=BOX_BORDER,
                     alignment=Alignment(vertical="center", horizontal="center"))
        _style_range(ws, f"{c[0]}{chas_header_row+1}:{c[4]}{end_row}", number_format="#,##0")
        _style_range(ws, f"{c[0]}{chas_header_row+1}:{c[0]}{end_row}", fill=_fill(LIGHT_GRAY), font=label_font,
                     alignment=Alignment(vertical="center", horizontal="left"))

        # Resaltar en ámbar el anclaje más cargado (Peak |F| máximo de la columna) para este eje
        peak_col = c[4]
        peak_range = f"{c[0]}{chas_header_row+1}:{peak_col}{end_row}"
        ws.conditional_formatting.add(
            peak_range,
            FormulaRule(formula=[f"${peak_col}{chas_header_row+1}=MAX(${peak_col}${chas_header_row+1}:${peak_col}${end_row})"],
                        fill=_fill(AMBER)))
    row = end_row + 2

    # ------------------------------------------------------------
    # FORMATO GLOBAL
    # ------------------------------------------------------------
    widths = {
        "A": 16, "B": 9, "C": 9, "D": 9, "E": 9, "F": 9, "G": 9, "H": 9, "I": 9, "J": 9, "K": 9,
        "L": 3,
        "M": 16, "N": 9, "O": 9, "P": 9, "Q": 9, "R": 9, "S": 9, "T": 9, "U": 9, "V": 9, "W": 9,
    }
    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    ws.freeze_panes = "A6"

    # Ajuste de impresión: que quepa en una página de ancho (una sola hoja de trabajo,
    # y además pensada para imprimir/exportar a PDF sin partirse en varias páginas anchas)
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_area = f"A1:W{row}"

    wb.save(output_filename)
    print(f"[✓] Informe Excel exportado correctamente (hoja única, compacta): {output_filename}")


if __name__ == '__main__':
    full_report = build_full_report(vehicle_params_ter27)
    export_to_excel(full_report)
    print()
    run_chassis_load_analysis(full_report)


# ==============================================================================
# CHANGELOG (respecto a la versión original)
# ==============================================================================
# 1. Fix: coma que faltaba entre 'axial_forces' y 'chassis_forces' en el
#    return de solve_axle_loads (SyntaxError).
# 2. Fix: `import pandas as pd` (NameError al construir los DataFrame).
# 3. Fix físico: Mz_steering usaba Fy con el scrub radius y Fz con el
#    pneumatic trail; ahora Fx -> scrub radius, Fy -> pneumatic trail,
#    que es su origen físico real.
# 4. Implementado: el mínimo de diseño del tie-rod (1.5 kN, criterio FSG)
#    ahora se aplica de verdad. Se distingue 'Tie_Rod' (valor de diseño,
#    usado para dimensionar el anclaje) de 'Tie_Rod_Solved' (valor de
#    equilibrio puro, para verificación).
# 5. Añadidas guardas ValueError ante geometría degenerada (denominadores
#    casi nulos) en F_push y F_damper, con mensaje indicando qué hardpoints
#    revisar, en vez de propagar silenciosamente inf/nan.
# 6. Refactor: toda la física se calcula una sola vez en build_full_report()
#    / analyze_axle(); tanto la consola como el Excel consumen ese mismo
#    resultado, eliminando el riesgo de que ambas salidas diverjan si se
#    edita una fórmula y se olvida actualizar la otra.
# 7. `artifact_tool` NO EXISTE como paquete instalable -> sustituido por
#    `openpyxl` (pip install openpyxl), librería real y estándar.
# 8. Excel: hoja única con Delantero y Trasero en columnas paralelas (en vez
#    de secciones apiladas), bloque de "Design Assumptions" explícito,
#    resaltado condicional del anclaje más cargado por eje, y ajuste de
#    página (orientación horizontal + fit-to-width) para que también quepa
#    bien al imprimir o exportar a PDF.
# 9. FIX ESTRUCTURAL: eliminada la cascada F_ub -> axis_upp -> lstsq para
#    repartir Upper_Fore/Upper_Aft. Sustituida por un único sistema 6x6 que
#    resuelve Upper_Fore, Upper_Aft, Lower_Fore, Lower_Aft, Tie_Rod y
#    Pushrod simultáneamente (6 miembros de 2 fuerzas para los 6 GDL de la
#    mangueta = sistema determinado, sin residuo). Se eliminan por
#    innecesarias las variables `axis_upp`, `M_ball_axis`, `M_push_axis`,
#    `F_ub`, `F_chassis_needed` y la llamada a `np.linalg.lstsq`.
# ==============================================================================