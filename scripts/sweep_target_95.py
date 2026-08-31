#!/usr/bin/env python3
# scripts/sweep_target_95.py
# Project-GP — Sandbox Vectorizado para Alcanzar >95% Fleet Score
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.run_can_backtest import (
    decode_can_csv_to_dataframe, _extract_1d, _vy0_from_yaw_drift,
    run_session_backtest, WINDOW_LEN, _lowpass_corr,
)

DATA_DIR = Path("data/raw_can_logs")
DBC_PATH = Path("TER.dbc")
DT = 0.005


def load_fleet_data(steer_sign: float = 1.0):
    files = sorted(DATA_DIR.glob("*.csv"))
    session_data = []
    print(f"[Sandbox] Cargando y preprocesando {len(files)} sesiones...")
    for f in files:
        df = decode_can_csv_to_dataframe(f, dbc_path=DBC_PATH, dt=DT)
        session_data.append((f.stem, df))
    return session_data


def evaluate_candidate(vehicle, session_data, p: np.ndarray, steer_sign: float = 1.0):
    """
    Evalúa un candidato p = [mu_f, mu_r, steer_g, brake_g, trq_g, rby1, rby2, alpha_s]
    contra la flota completa usando el evaluador idéntico de run_can_backtest.
    """
    tire_cal = jnp.array([p[0], p[1], -1.0, p[7], p[5], p[6]], dtype=jnp.float32)
    steer_g, brake_g, trq_g = float(p[2]), float(p[3]), float(p[4])

    scores, r_ays, r_wzs, rmse_vxs = [], [], [], []
    per_session = {}

    for name, df in session_data:
        res = run_session_backtest(
            vehicle, df, dt=DT, steer_sign=steer_sign, verbose=False,
            tire_cal=tire_cal, steer_gain=steer_g,
            brake_gain=brake_g, torque_gain=trq_g,
        )
        scores.append(res['score'])
        r_ays.append(res['r_ay'])
        r_wzs.append(res['r_wz'])
        rmse_vxs.append(res['rmse_vx'])
        per_session[name] = res['score']

    fleet_score = float(np.mean(scores))
    mean_r_ay = float(np.mean(r_ays))
    mean_r_wz = float(np.mean(r_wzs))
    mean_rmse_vx = float(np.mean(rmse_vxs))

    return {
        "fleet_score": fleet_score,
        "mean_r_ay": mean_r_ay,
        "mean_r_wz": mean_r_wz,
        "mean_rmse_vx": mean_rmse_vx,
        "p": p,
        "per_session": per_session,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150, help="Pasos de optimización Adam")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--apply-best", action="store_true", help="Guardar los mejores parámetros en models/")
    args = parser.parse_args()

    print("=" * 78)
    print("  PROJECT-GP · SANDBOX AISLADO DE OPTIMIZACIÓN (>95% FLEET SCORE)")
    print("=" * 78)

    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
    session_data = load_fleet_data(steer_sign=1.0)

    # 1. Candidatos base físicamente fundamentados
    candidates = [
        # [mu_f, mu_r, steer_g, brake_g, trq_g, rby1, rby2, alpha_scale]
        np.array([1.463, 1.890, 0.917, 1.408, 1.219, 0.382, 3.000, 1.000]), # Baseline actual (91.95%)
        np.array([1.480, 2.100, 0.930, 1.400, 1.200, 0.350, 3.000, 1.050]),
        np.array([1.500, 2.200, 0.920, 1.450, 1.250, 0.300, 3.500, 1.100]),
        np.array([1.520, 2.300, 0.940, 1.400, 1.220, 0.250, 3.500, 1.150]),
        np.array([1.450, 2.000, 0.950, 1.350, 1.180, 0.400, 2.800, 1.000]),
    ]

    print("\n[Sandbox] Evaluando candidatos de barrido inicial...")
    leaderboard = []
    for i, cand in enumerate(candidates, 1):
        res = evaluate_candidate(vehicle, session_data, cand)
        res["name"] = f"Candidate_{i}"
        leaderboard.append(res)
        print(f"  {res['name']}: Fleet={res['fleet_score']:.2f}% | "
              f"r_ay={res['mean_r_ay']:+.3f} | r_wz={res['mean_r_wz']:+.3f} | "
              f"rmse_vx={res['mean_rmse_vx']:.3f}")

    # 2. Búsqueda de alta resolución mediante Coordinate Search alrededor del mejor candidato
    best_init = max(leaderboard, key=lambda x: x["fleet_score"])
    print(f"\n[Sandbox] Mejor punto de partida: {best_init['name']} ({best_init['fleet_score']:.2f}%)")
    print("[Sandbox] Ejecutando refinamiento multivariable...")

    p_best = best_init["p"].copy()
    score_best = best_init["fleet_score"]

    deltas = [
        (0, [1.40, 1.46, 1.52, 1.58]),         # mu_f
        (1, [1.80, 2.00, 2.20, 2.40]),         # mu_r
        (2, [0.90, 0.93, 0.96, 1.00]),         # steer_gain
        (3, [1.30, 1.40, 1.50]),               # brake_gain
        (4, [1.15, 1.22, 1.28]),               # torque_gain
        (5, [0.20, 0.30, 0.40, 0.50]),         # rby1
        (6, [2.50, 3.00, 3.50, 4.00]),         # rby2
        (7, [1.00, 1.05, 1.10, 1.15]),         # alpha_scale
    ]

    improved = True
    iteration = 0
    while improved and iteration < 3:
        improved = False
        iteration += 1
        for param_idx, test_vals in deltas:
            for val in test_vals:
                p_trial = p_best.copy()
                p_trial[param_idx] = val
                res_trial = evaluate_candidate(vehicle, session_data, p_trial)
                if res_trial["fleet_score"] > score_best:
                    score_best = res_trial["fleet_score"]
                    p_best = p_trial
                    res_trial["name"] = f"Refined_iter{iteration}_p{param_idx}_{val:.3f}"
                    leaderboard.append(res_trial)
                    improved = True
                    print(f"  [+] Nuevo Récord: {score_best:.2f}% | mu_f={p_best[0]:.3f} "
                          f"mu_r={p_best[1]:.3f} steer={p_best[2]:.3f} rby1={p_best[5]:.3f}")

    # 3. Resumen y Exportación
    df_board = pd.DataFrame([
        {
            "name": r["name"],
            "fleet_score": r["fleet_score"],
            "mean_r_ay": r["mean_r_ay"],
            "mean_r_wz": r["mean_r_wz"],
            "mean_rmse_vx": r["mean_rmse_vx"],
            "mu_f": r["p"][0],
            "mu_r": r["p"][1],
            "steer_g": r["p"][2],
            "brake_g": r["p"][3],
            "trq_g": r["p"][4],
            "rby1": r["p"][5],
            "rby2": r["p"][6],
            "alpha_s": r["p"][7],
            **r["per_session"],
        }
        for r in leaderboard
    ]).sort_values("fleet_score", ascending=False)

    out_dir = Path("reports") / "experiment_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "target_95_leaderboard.csv"
    df_board.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("  TOP 5 CONFIGURACIONES IDENTIFICADAS EN EL SANDBOX")
    print("=" * 78)
    print(df_board[["name", "fleet_score", "mean_r_ay", "mean_r_wz", "mu_f", "mu_r", "steer_g", "alpha_s"]].head(5).to_string(index=False))
    print(f"\n[saved] Leaderboard completo guardado en: {out_path}")

    winner = df_board.iloc[0]
    print(f"\n[*] GANADOR GLOBAL: {winner['fleet_score']:.2f}% (mu_f={winner['mu_f']:.3f}, mu_r={winner['mu_r']:.3f}, alpha_s={winner['alpha_s']:.3f})")

    if args.apply_best:
        os.makedirs("models", exist_ok=True)
        np.save("models/mu_scale_calibrated.npy", np.array([winner['mu_f'], winner['mu_r']]))
        np.save("models/gain_calibrated.npy", np.array([winner['steer_g'], winner['brake_g'], winner['trq_g']]))
        np.save("models/rby_scale_calibrated.npy", np.array([winner['rby1'], winner['rby2']]))
        np.save("models/alpha_scale_calibrated.npy", np.array([winner['alpha_s']]))
        print(f"[Sandbox] ¡Parámetros ganadores guardados exitosamente en models/!")
    else:
        print("\nPara aplicar los parámetros ganadores a producción sin editar código ejecuta:")
        print("  python -m scripts.sweep_target_95 --apply-best")


if __name__ == "__main__":
    main()