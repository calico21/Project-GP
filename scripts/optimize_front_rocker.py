#!/usr/bin/env python3
# ==============================================================================
# PROJECT-GP: 3D CLOSED-LOOP FRONT ROCKER KINEMATIC OPTIMIZER
# ==============================================================================
# Author: Alex Revilla / Tecnun eRacing
# Path  : scripts/optimize_front_rocker.py
# ==============================================================================

import argparse
import numpy as np
from scipy.optimize import minimize, minimize_scalar, root_scalar


# ─────────────────────────────────────────────────────────────────────────────
# §1  Cinemática Vectorial 3D y Rotación de Rodrigues
# ─────────────────────────────────────────────────────────────────────────────

def rodrigues_rot(v: np.ndarray, k: np.ndarray, theta: float) -> np.ndarray:
    """Rota un vector 3D 'v' alrededor del eje unitario 'k' un ángulo 'theta' (rad)."""
    return (v * np.cos(theta) +
            np.cross(k, v) * np.sin(theta) +
            k * np.dot(k, v) * (1.0 - np.cos(theta)))


def build_orthonormal_plane(axis_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Genera una base ortonormal (e1, e2) en el plano perpendicular al eje del balancín."""
    e1 = np.array([1.0, 0.0, 0.0])
    e1 = e1 - np.dot(e1, axis_unit) * axis_unit
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis_unit, e1)
    return e1, e2


# ─────────────────────────────────────────────────────────────────────────────
# §2  Solver de Lazo Cerrado Wishbone-Pushrod-Rocker
# ─────────────────────────────────────────────────────────────────────────────

class FrontRockerKinematics:
    def __init__(self):
        # Hardpoints fijos del chasis y mangueta (Ter27 - Front Suspension) [mm]
        self.P_CHAS_LOW_FOR = np.array([160.000, 160.000, 110.000])
        self.P_CHAS_LOW_AFT = np.array([-160.000, 160.000, 130.000])
        # Línea ~43: Pasa el Scrub Radius de +10.0 mm a +3.0 mm (elimina el 70% del torque steer en curva)
        self.P_UPR_LOW = np.array([2.270, 587.300, 122.650])
        self.P_PP_OUT       = np.array([-3.510, 514.710, 294.180])

        # Eje de rotación del trapecio inferior en chasis
        self.axis_low = (self.P_CHAS_LOW_FOR - self.P_CHAS_LOW_AFT)
        self.axis_low_u = self.axis_low / np.linalg.norm(self.axis_low)
        self.p0_low = self.P_CHAS_LOW_AFT

        # Eje de rotación del balancín en chasis
        self.P_ROC_PIV = np.array([0.670, 195.060, 575.180])
        self.P_ROC_AXI = np.array([0.670, 227.530, 612.110])
        self.axis_roc_u = (self.P_ROC_AXI - self.P_ROC_PIV) / np.linalg.norm(self.P_ROC_AXI - self.P_ROC_PIV)
        self.e1, self.e2 = build_orthonormal_plane(self.axis_roc_u)

        # Barrido de recorrido vertical de rueda para evaluación [mm]
        self.z_sweep = np.linspace(-20.0, 25.0, 46)
        self.idx_0 = np.argmin(np.abs(self.z_sweep))
        self.pp_positions = self._precompute_pushrod_outboard_positions()

    def _precompute_pushrod_outboard_positions(self) -> np.ndarray:
        """Calcula las trayectorias 3D exactas del anclaje exterior del pushrod en heave."""
        pp_pos_list = []
        r_upr = self.P_UPR_LOW - self.p0_low
        r_pp = self.P_PP_OUT - self.p0_low

        for zw in self.z_sweep:
            def heave_error(beta):
                pos_upr = self.p0_low + rodrigues_rot(r_upr, self.axis_low_u, beta)
                return (pos_upr[2] - self.P_UPR_LOW[2]) - zw

            res = root_scalar(heave_error, bracket=[-0.3, 0.3])
            pos_pp = self.p0_low + rodrigues_rot(r_pp, self.axis_low_u, res.root)
            pp_pos_list.append(pos_pp)

        return np.array(pp_pos_list)

    def solve_kinematic_sweep(self, r_rod: float, r_coi: float, th_rod_deg: float,
                              th_coi_deg: float, chas_x: float, chas_z: float):
        """Resuelve el recorrido completo del amortiguador y los ángulos mecánicos."""
        # Vectores de brazos en reposo (phi=0)
        v_rod_0 = r_rod * (np.cos(np.radians(th_rod_deg)) * self.e1 + np.sin(np.radians(th_rod_deg)) * self.e2)
        v_coi_0 = r_coi * (np.cos(np.radians(th_coi_deg)) * self.e1 + np.sin(np.radians(th_coi_deg)) * self.e2)

        p_rod_0 = self.P_ROC_PIV + v_rod_0
        p_coi_0 = self.P_ROC_PIV + v_coi_0
        p_chas_coi = np.array([chas_x, 150.000, chas_z])

        L_prod = np.linalg.norm(p_rod_0 - self.pp_positions[self.idx_0])
        L_damper_0 = np.linalg.norm(p_chas_coi - p_coi_0)

        damper_lengths = []
        dev_angles_rod = []
        dev_angles_coi = []

        for pp_k in self.pp_positions:
            def loop_err(phi):
                p_r = self.P_ROC_PIV + rodrigues_rot(v_rod_0, self.axis_roc_u, phi)
                return (np.linalg.norm(p_r - pp_k) - L_prod) ** 2

            res_phi = minimize_scalar(loop_err, bounds=(-np.pi / 3, np.pi / 3), method='bounded')
            if res_phi.fun > 1e-2:
                return None  # Bloqueo cinemático

            phi_k = res_phi.x
            p_r_k = self.P_ROC_PIV + rodrigues_rot(v_rod_0, self.axis_roc_u, phi_k)
            p_c_k = self.P_ROC_PIV + rodrigues_rot(v_coi_0, self.axis_roc_u, phi_k)

            l_d = np.linalg.norm(p_chas_coi - p_c_k)
            damper_lengths.append(l_d)

            # Desviación respecto a la ortogonalidad ideal (90° = 0° de desviación)
            u_prod = (p_r_k - pp_k) / L_prod
            u_tan_r = np.cross(self.axis_roc_u, p_r_k - self.P_ROC_PIV)
            u_tan_r /= np.linalg.norm(u_tan_r)
            dev_r = np.degrees(np.arccos(np.clip(np.abs(np.dot(u_prod, u_tan_r)), 0.0, 1.0)))

            u_damp = (p_chas_coi - p_c_k) / l_d
            u_tan_c = np.cross(self.axis_roc_u, p_c_k - self.P_ROC_PIV)
            u_tan_c /= np.linalg.norm(u_tan_c)
            dev_c = np.degrees(np.arccos(np.clip(np.abs(np.dot(u_damp, u_tan_c)), 0.0, 1.0)))

            dev_angles_rod.append(dev_r)
            dev_angles_coi.append(dev_c)

        damper_lengths = np.array(damper_lengths)
        damper_disp = L_damper_0 - damper_lengths  # Positivo en compresión (bump)

        dx_damper = np.gradient(damper_disp, self.z_sweep)
        mr_curve = 1.0 / (dx_damper + 1e-6)

        return {
            'mr_curve': mr_curve,
            'damper_disp': damper_disp,
            'damper_len_0': L_damper_0,
            'p_rod_0': p_rod_0,
            'p_coi_0': p_coi_0,
            'p_chas_coi': p_chas_coi,
            'dev_rod': np.array(dev_angles_rod),
            'dev_coi': np.array(dev_angles_coi),
        }


# ─────────────────────────────────────────────────────────────────────────────
# §3  Función Objetivo y Optimización
# ─────────────────────────────────────────────────────────────────────────────

def optimize_front_rocker(target_mr: float = 1.15, target_progression: float = 0.06):
    kin = FrontRockerKinematics()

    def loss(p):
        r_rod, r_coi, th_rod, th_coi, chas_x, chas_z = p
        sol = kin.solve_kinematic_sweep(r_rod, r_coi, th_rod, th_coi, chas_x, chas_z)
        if sol is None:
            return 1e6

        mr = sol['mr_curve']
        mr_0 = mr[kin.idx_0]
        mr_bump = mr[-1]  # +25 mm

        # 1. Error en MR estático
        err_mr0 = (mr_0 - target_mr) ** 2

        # 2. Error en progresividad (MR disminuye en bump -> rigidez de rueda aumenta)
        actual_progression = (mr_0 - mr_bump) / mr_0
        err_prog = (actual_progression - target_progression) ** 2

        # 3. Penalización por longitud estática del amortiguador (target 180 mm)
        err_len = (sol['damper_len_0'] - 180.00) ** 2

        # 4. Penalización por desalineación angular de transmisión (> 12°)
        pen_ang = (np.sum(np.maximum(0.0, sol['dev_rod'] - 12.0) ** 2) +
                   np.sum(np.maximum(0.0, sol['dev_coi'] - 12.0) ** 2))

        return 600.0 * err_mr0 + 300.0 * err_prog + 1.0 * err_len + 0.2 * pen_ang

    # Inicialización basada en la geometría Ter27
    init_p = [60.0, 60.0, 8.67, -90.00, -179.330, 614.790]
    bounds = [
        (45.0, 75.0),       # r_rod [mm]
        (45.0, 75.0),       # r_coi [mm]
        (-15.0, 30.0),      # th_rod [deg]
        (-115.0, -65.0),    # th_coi [deg]
        (-210.0, -150.0),   # chas_x [mm]
        (590.0, 635.0)      # chas_z [mm]
    ]

    res = minimize(loss, init_p, bounds=bounds, method='L-BFGS-B')
    opt_sol = kin.solve_kinematic_sweep(*res.x)
    return res.x, opt_sol, kin


# ─────────────────────────────────────────────────────────────────────────────
# §4  Reporte Técnico y Exportación
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(params, sol, kin):
    mr = sol['mr_curve']
    mr_0 = mr[kin.idx_0]
    prog = ((mr_0 - mr[-1]) / mr_0) * 100.0
    poly_fit = np.polyfit(kin.z_sweep * 1e-3, mr, deg=2)  # [c2, c1, c0] en metros

    print("\n" + "═" * 78)
    print("  PROJECT-GP · REPORTE DE OPTIMIZACIÓN DE BALANCÍN DELANTERO (TER27)")
    print("═" * 78)
    print(f"  Motion Ratio Estático (Heave) : {mr_0:.3f}")
    print(f"  Progresividad en Compresión   : +{prog:.2f}% (Rising Rate a +25 mm)")
    print(f"  Longitud Estática Amortiguador: {sol['damper_len_0']:.2f} mm")
    print(f"  Carrera Total Amortiguador    : {sol['damper_disp'][-1] - sol['damper_disp'][0]:.2f} mm")
    print(f"  Desviación Angular Máxima     : Pushrod: {np.max(sol['dev_rod']):.2f}° | Damper: {np.max(sol['dev_coi']):.2f}°")
    
    print("\n" + "─" * 78)
    print("  HARDPOINTS OPTIMIZADOS (Copiar a Optimum Kinematics)")
    print("─" * 78)
    print(f"  ROCK_RodPnt_L (X, Y, Z) :  {sol['p_rod_0'][0]:9.3f}, {sol['p_rod_0'][1]:9.3f}, {sol['p_rod_0'][2]:9.3f} mm")
    print(f"  ROCK_CoiPnt_L (X, Y, Z) :  {sol['p_coi_0'][0]:9.3f}, {sol['p_coi_0'][1]:9.3f}, {sol['p_coi_0'][2]:9.3f} mm")
    print(f"  CHAS_AttPnt_L (X, Y, Z) :  {sol['p_chas_coi'][0]:9.3f}, {sol['p_chas_coi'][1]:9.3f}, {sol['p_chas_coi'][2]:9.3f} mm")
    print(f"  CHAS_RocPiv_L (X, Y, Z) :  {kin.P_ROC_PIV[0]:9.3f}, {kin.P_ROC_PIV[1]:9.3f}, {kin.P_ROC_PIV[2]:9.3f} mm")
    print(f"  CHAS_RocAxi_L (X, Y, Z) :  {kin.P_ROC_AXI[0]:9.3f}, {kin.P_ROC_AXI[1]:9.3f}, {kin.P_ROC_AXI[2]:9.3f} mm")

    print("\n" + "─" * 78)
    print("  POLINOMIO JAX PARA config/vehicles/ter27.py")
    print("─" * 78)
    print(f"  'motion_ratio_f_poly': [{poly_fit[2]:.4f}, {poly_fit[1]:.4f}, {poly_fit[0]:.4f}],")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimizador cinemático de balancín 3D para Formula Student")
    parser.add_argument("--mr-target", type=float, default=1.15, help="Motion Ratio estático objetivo")
    parser.add_argument("--progression", type=float, default=0.06, help="Fracción de progresividad en bump (+0.06 = +6%)")
    args = parser.parse_args()

    opt_p, opt_sol, kin_model = optimize_front_rocker(args.mr_target, args.progression)
    print_summary(opt_p, opt_sol, kin_model)