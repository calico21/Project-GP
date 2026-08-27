#!/usr/bin/env python3
# main/optimize_setup.py — Project-GP 28-DOF Pareto Setup Optimizer
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from config.design_freeze import DesignFreeze, install_freeze

# h_cg (setup[25]) es geometría de chasis fija en CAD — no fabricable como
# parámetro de setup por pista. Sin este freeze el optimizer lo deja flotar
# libremente porque su gradiente en compute_skidpad_objective es ~0 (el
# objetivo analítico usa VP['h_cg_chassis'] fijo, no s.h_cg), así que el
# valor final es puro ruido del muestreo BO inicial en [0,1]^28.
freeze = DesignFreeze({
    'h_cg':            0.285,
    'yaw_target_gain':  0.80,   # pertenece al controlador TV, no a la física
                                 # del chasis — optimizarlo aquí no tiene
                                 # gradiente real y contamina el espacio 28D
})
install_freeze(freeze, 'ter26')

# Inclusión del path raíz del proyecto
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from optimization.evolutionary import MORL_SB_TRPO_Optimizer
from models.vehicle_dynamics import SETUP_NAMES


def main():
    parser = argparse.ArgumentParser(description="Project-GP 28-DOF Setup Optimization")
    parser.add_argument("--iterations", type=int, default=150, help="Iteraciones de gradiente TRPO")
    parser.add_argument("--ensemble", type=int, default=12, help="Tamaño del ensamble Chebyshev")
    parser.add_argument("--save-dir", type=Path, default=Path("reports/optimal_setups"))
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  PROJECT-GP: OPTIMIZACIÓN MULTI-OBJETIVO DE SETUP (28-DOF MORL-SB-TRPO)")
    print("=" * 78)

    opt = MORL_SB_TRPO_Optimizer(ensemble_size=args.ensemble)

    # Cargar calibración real de neumáticos
    mu_path = os.path.join(ROOT_DIR, "models", "mu_scale_calibrated.npy")
    if os.path.exists(mu_path):
        mu_cal = np.load(mu_path)
        print(f"[*] Calibración cargada: mu_f={mu_cal[0]:.3f}, mu_r={mu_cal[1]:.3f}")
        opt._vehicle.tire_cal = jnp.array([mu_cal[0], mu_cal[1], -1.0, 1.0], dtype=jnp.float32)

    # Ejecutar búsqueda de la frontera de Pareto
    p_setups, p_grips, p_stabs, p_gen = opt.run(iterations=args.iterations)

    if len(p_grips) == 0 or not np.isfinite(p_grips[0]):
        print("[!] No se encontraron setups válidos en la frontera.")
        return

    # Exportar setup de máximo agarre
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_grip_idx = int(np.argmax(p_grips))
    best_setup = p_setups[best_grip_idx]

    out_file = args.save_dir / "best_setup_28d.npy"
    np.save(out_file, best_setup)

    print("\n" + "=" * 78)
    print(f"  SETUP ÓPTIMO ENCONTRADO (Grip Máximo: {p_grips[best_grip_idx]:.4f} G)")
    print("=" * 78)
    for name, val in zip(SETUP_NAMES, best_setup):
        print(f"  {name:<18} : {val:10.3f}")

    print(f"\n[*] Setup exportado a: {out_file}\n")


if __name__ == "__main__":
    main()