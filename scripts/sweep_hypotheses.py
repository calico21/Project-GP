#!/usr/bin/env python3
# scripts/sweep_hypotheses.py
# Project-GP — Barrido de hipótesis para el defecto estructural del eje trasero
# ═══════════════════════════════════════════════════════════════════════════════
#
# CONTEXTO (del backtest más reciente, score fleet=91.95%):
#   - mu_r convergió a 1.890 (ub=2.20) — NO pineado en el techo esta vez, así
#     que no es puramente un problema de ganancia de fricción.
#   - Las peores 10 ventanas por sesión son casi todas HIGH-regime +
#     DRIVE(rear)/COAST, con r_ay NEGATIVO (hasta -0.94). Eso es una firma de
#     FORMA, no de magnitud: la sim se aplana/decae mientras el ay real sigue
#     subiendo (o viceversa) — clásico de:
#       (a) transferencia de carga lateral trasera sub-modelada (dFz_lat_r en
#           _compute_derivatives es m*ay*h_cg/track_r plano, SIN el split
#           geométrico/elástico vía roll-centre que sí existe en
#           suspension_viz.py/aero_platform.py)
#       (b) rigidez de curva Ky trasera saturando antes de lo real (alpha_scale)
#       (c) reducción por combined-slip Gyk trasera demasiado agresiva (RBY1/RBY2)
#       (d) h_cg o Iz mal calibrados → transferencia de carga o momento de
#           inercia yaw incorrectos
#       (e) el filtro _lowpass_corr (butterworth 8Hz) puede estar introduciendo
#           artefactos de fase en ventanas cortas de 0.5s — vale la pena probar
#           sin él como control
#
# ESTE SCRIPT:
#   No modifica vehicle_dynamics.py. Para hipótesis "baratas" (solo tire_cal,
#   que ya es un jax.Array runtime) reutiliza el MISMO vehículo compilado y
#   barre valores. Para hipótesis "estructurales" (h_cg, track_r, Iz — que
#   vehicle_dynamics.py lee de self.vp en __init__, NO del setup vector)
#   instancia un vehículo nuevo con VP parcheado — más caro, así que esas
#   hipótesis solo se evalúan contra el batch de ventanas difíciles, no la
#   flota completa.
#
# USO:
#   python -m scripts.sweep_hypotheses                      # todo
#   python -m scripts.sweep_hypotheses --cheap-only         # solo tire_cal (rápido)
#   python -m scripts.sweep_hypotheses --n-hard 40          # más ventanas difíciles
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import argparse
import copy
import os
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from config.vehicles.ter26 import vehicle_params as VP_DICT_BASE
from config.tire_coeffs import tire_coeffs as TP_DICT
from scripts.run_can_backtest import (
    decode_can_csv_to_dataframe, _extract_1d, _vy0_from_yaw_drift,
    _probe_best_steer_sign, run_session_backtest, run_session_backtest_debug,
    WINDOW_LEN, _lowpass_corr,
)

DATA_DIR = Path("data/raw_can_logs")
DBC_PATH = Path("TER.dbc")
DT = 0.005


def load_production_calibration():
    """
    Carga EXACTAMENTE lo que usa scripts/run_can_backtest.py en producción:
    steer_sign GLOBAL (una sola sesión de probing, no por-sesión) y tire_cal
    de 6 elementos [mu_f, mu_r, T_opt_ovr, alpha_scale, rby1, rby2].

    CRÍTICO: no re-probar steer_sign por sesión aquí. _probe_best_steer_sign
    con un vehículo/tire_cal sin calibrar puede seleccionar signos distintos
    por sesión (visto en la corrida anterior: -1,-1,+1,+1,-1) mientras que
    producción usa un único signo global (+1) obtenido una vez con el
    tire_cal YA calibrado. Mezclar signos hace que el coche gire al revés en
    algunas sesiones — eso por sí solo destroza r_ay y no tiene nada que ver
    con las hipótesis físicas que queremos medir.
    """
    mu_path   = os.path.join("models", "mu_scale_calibrated.npy")
    rby_path  = os.path.join("models", "rby_scale_calibrated.npy")
    sign_path = os.path.join("models", "steer_sign_calibrated.npy")

    mu = np.load(mu_path) if os.path.exists(mu_path) else np.array([1.0, 1.0])
    rby = np.load(rby_path) if os.path.exists(rby_path) else np.array([1.0, 1.0])
    steer_sign = float(np.load(sign_path)[0]) if os.path.exists(sign_path) else 1.0

    tire_cal = jnp.array([mu[0], mu[1], -1.0, 1.0, rby[0], rby[1]], dtype=jnp.float32)
    print(f"[calib] steer_sign={steer_sign:+.0f}  mu_f={mu[0]:.3f}  mu_r={mu[1]:.3f}  "
          f"rby1={rby[0]:.3f}  rby2={rby[1]:.3f}")
    return steer_sign, tire_cal


# ─────────────────────────────────────────────────────────────────────────────
# §1  Construir el batch de ventanas difíciles (HIGH + no-braking) + control LOW
# ─────────────────────────────────────────────────────────────────────────────

def build_hard_batch(vehicle, steer_sign_global: float, tire_cal_baseline: jax.Array,
                      n_hard: int = 30, n_control: int = 15):
    """
    Reutiliza run_session_backtest_debug para clasificar todas las ventanas de
    todas las sesiones, se queda con las N peores por r_ay dentro de
    HIGH & axle!=BRAKE(front) (el segmento identificado como dominante del
    fallo), más un control aleatorio de ventanas LOW para vigilar regresiones.

    Usa el steer_sign GLOBAL y el tire_cal YA CALIBRADO (no identidad) para
    que la clasificación de "peores ventanas" coincida con la que ve
    producción — si clasificamos con física sin calibrar, seleccionamos las
    ventanas equivocadas como "duras".

    Devuelve arrays listos para vmap: u_batch (n,100,6), x0_batch (n,108),
    real_wz (n,100), real_ay (n,100), tags (list[str]).
    """
    files = sorted(DATA_DIR.glob("*.csv"))
    all_rows = []
    session_data = {}

    for f in files:
        df = decode_can_csv_to_dataframe(f, dbc_path=DBC_PATH, dt=DT)
        session_data[f.stem] = (df, steer_sign_global)

        res = run_session_backtest_debug(
            vehicle, df, dt=DT, steer_sign=steer_sign_global,
            tire_cal=tire_cal_baseline, session_name=f"_scan_{f.stem}")
        if not res or "df" not in res:
            continue
        d = res["df"].copy()
        d["session"] = f.stem
        all_rows.append(d)

    fleet = pd.concat(all_rows, ignore_index=True)

    hard = fleet[(fleet.regime == "HIGH") & (fleet.axle != "BRAKE(front)")].dropna(subset=["r_ay"])
    hard = hard.sort_values("r_ay").head(n_hard)

    control = fleet[fleet.regime == "LOW"].dropna(subset=["r_ay"])
    control = control.sample(n=min(n_control, len(control)), random_state=0)

    picked = pd.concat([hard.assign(tag="HARD"), control.assign(tag="CONTROL")])

    u_windows, x0_windows, wz_wins, ay_wins, tags = [], [], [], [], []

    for _, row in picked.iterrows():
        df, steer_sign = session_data[row.session]
        w = int(row.window)
        s, e = w * WINDOW_LEN, w * WINDOW_LEN + WINDOW_LEN

        steer_rad = np.deg2rad(_extract_1d(df, 'steer_deg')) * steer_sign
        t_fl, t_fr = _extract_1d(df, 't_fl'), _extract_1d(df, 't_fr')
        t_rl, t_rr = _extract_1d(df, 't_rl'), _extract_1d(df, 't_rr')
        p_hyd = _extract_1d(df, 'brake_press')
        u_all = np.stack([steer_rad, t_fl, t_fr, t_rl, t_rr, p_hyd], axis=1)
        u_all = np.nan_to_num(u_all, nan=0.0, posinf=0.0, neginf=0.0)
        u_all[:, 1:5] = np.clip(u_all[:, 1:5], -50.0, 400.0)
        u_all[:, 5] = np.clip(u_all[:, 5], 0.0, 2000.0)

        real_vx = _extract_1d(df, 'vx_mps')
        real_wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
        real_ay = _extract_1d(df, 'ay_mps2')

        vx0 = float(max(real_vx[s], 1.0))
        wz0 = float(real_wz[s])
        vy0 = _vy0_from_yaw_drift(vx0, wz0)

        x0 = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=vx0)
        x0 = x0.at[15].set(vy0).at[19].set(wz0)

        u_windows.append(u_all[s:e])
        x0_windows.append(x0)
        wz_wins.append(real_wz[s:e])
        ay_wins.append(real_ay[s:e])
        tags.append(f"{row.tag}:{row.session}#{w}")

    print(f"\n[batch] {sum(t.startswith('HARD') for t in tags)} HARD + "
          f"{sum(t.startswith('CONTROL') for t in tags)} CONTROL windows seleccionadas.")

    return (jnp.asarray(np.stack(u_windows)), jnp.asarray(np.stack(x0_windows)),
            np.stack(wz_wins), np.stack(ay_wins), tags)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Evaluador vectorizado (mismo vehículo, tire_cal variable)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(0,))
def _rollout_batch(vehicle, x0_batch, u_batch, tire_cal, ay_scale):
    setup = vehicle._default_setup_vec

    def one(x0, u_seq):
        def step_fn(x, u):
            x_next = vehicle.simulate_step(x, u, setup, dt=DT, n_substeps=4, tire_cal=tire_cal)
            vx_n = x_next[14]; wz_n = x_next[19]
            return x_next, jnp.array([wz_n, vx_n * wz_n * ay_scale])
        _, out = jax.lax.scan(step_fn, x0, u_seq)
        return out[:, 0], out[:, 1]

    return jax.vmap(one)(x0_batch, u_batch)


def score_hypothesis(vehicle, x0_batch, u_batch, real_wz, real_ay, tags,
                      tire_cal, ay_scale=1.0, use_lowpass=True):
    wz_sim, ay_sim = _rollout_batch(vehicle, x0_batch, u_batch, tire_cal, ay_scale)
    wz_sim, ay_sim = np.array(wz_sim), np.array(ay_sim)

    def _corr(a, b):
        if use_lowpass:
            a, b = _lowpass_corr(a), _lowpass_corr(b)
        if np.std(a) < 1e-4 or np.std(b) < 1e-4:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    hard_idx = [i for i, t in enumerate(tags) if t.startswith("HARD")]
    ctrl_idx = [i for i, t in enumerate(tags) if t.startswith("CONTROL")]

    r_ay_hard = np.nanmean([_corr(ay_sim[i], real_ay[i]) for i in hard_idx])
    r_ay_ctrl = np.nanmean([_corr(ay_sim[i], real_ay[i]) for i in ctrl_idx])
    rmse_hard = float(np.sqrt(np.mean((ay_sim[hard_idx] - real_ay[hard_idx]) ** 2)))
    rmse_ctrl = float(np.sqrt(np.mean((ay_sim[ctrl_idx] - real_ay[ctrl_idx]) ** 2)))

    return dict(r_ay_hard=r_ay_hard, r_ay_ctrl=r_ay_ctrl,
                rmse_hard=rmse_hard, rmse_ctrl=rmse_ctrl)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Hipótesis "baratas" — solo tire_cal (mu_f, mu_r, T_opt_ovr, alpha_scale,
#     rby1_scale, rby2_scale). tire_cal layout (ver tire_model.py):
#       [0]=mu_f  [1]=mu_r  [2]=T_opt_override  [3]=alpha_scale
#       [4]=rby1_scale  [5]=rby2_scale
# ─────────────────────────────────────────────────────────────────────────────

def cheap_hypotheses():
    base = [1.463, 1.890, -1.0, 1.0, 0.382, 3.000]  # valores calibrados actuales
    hyps = {"baseline (calibrado actual)": base}

    # H1: ¿sigue bajando el RMSE al subir mu_r más allá de 1.89? Si SÍ →
    # confirma que sigue siendo un problema de magnitud, no de forma.
    for mu_r in [1.6, 1.8, 2.0, 2.2, 2.5, 3.0]:
        hyps[f"mu_r={mu_r:.2f} (resto igual)"] = [1.463, mu_r, -1.0, 1.0, 0.382, 3.000]

    # H2: alpha_scale trasero — NOTA: alpha_scale es global (afecta las 4
    # ruedas vía tire_cal[3]), así que este barrido es diagnóstico: si mejora
    # mucho, hay que separar alpha_scale_f/alpha_scale_r en tire_cal.
    for a_scl in [0.7, 0.85, 1.0, 1.15, 1.3]:
        hyps[f"alpha_scale={a_scl:.2f}"] = [1.463, 1.890, -1.0, a_scl, 0.382, 3.000]

    # H3: combined-slip trasero (RBY1/RBY2) — controla cuánto cae Fy con
    # slip longitudinal simultáneo. rby1_scale ya calibrado a 0.382 (mucho
    # menos agresivo que 1.0) — probar si aún menos ayuda, o si se pasó de rosca.
    for rby1, rby2 in [(0.2, 3.0), (0.382, 3.0), (0.6, 3.0), (1.0, 1.0), (0.382, 5.0), (0.382, 1.5)]:
        hyps[f"rby1={rby1:.2f} rby2={rby2:.2f}"] = [1.463, 1.890, -1.0, 1.0, rby1, rby2]

    # H4: T_opt override — ¿el punto óptimo térmico calibrado (self.tire.T_opt,
    # típicamente ~90°C) es correcto para el eje trasero en ventanas HIGH
    # (más calientes)? Probar T_opt más bajo (neumático ya caliente/degradado).
    for t_opt in [-1.0, 70.0, 80.0, 100.0, 110.0]:
        hyps[f"T_opt_override={t_opt:.0f}"] = [1.463, 1.890, t_opt, 1.0, 0.382, 3.000]

    # H5: combinación mu_r moderado + alpha_scale más alto (rigidez de curva
    # trasera mayor en vez de solo más fricción — cambia la FORMA, no solo el
    # techo, que es justo lo que las ventanas HIGH con r_ay negativo sugieren).
    for mu_r, a_scl in [(1.6, 1.15), (1.7, 1.1), (1.8, 1.05)]:
        hyps[f"mu_r={mu_r:.2f}+alpha_scale={a_scl:.2f}"] = [1.463, mu_r, -1.0, a_scl, 0.382, 3.000]

    return hyps


# ─────────────────────────────────────────────────────────────────────────────
# §4  Hipótesis "estructurales" — requieren nueva instancia de vehículo
#     (VP parcheado). Solo se corren contra el batch difícil (más caras).
# ─────────────────────────────────────────────────────────────────────────────

def structural_hypotheses():
    """
    Cada entrada es (nombre, dict de overrides sobre VP_DICT_BASE).
    Los overrides tocan cantidades que _compute_derivatives lee DIRECTAMENTE
    de self.vp (NO del setup vector de 28 params) — h_cg, track_r, Iz.
    """
    hyps = {}

    # H6: h_cg — afecta transferencia de carga lateral en AMBOS ejes por
    # igual con la fórmula actual (dFz_lat_r = m*ay*h_cg/track_r). Si el
    # h_cg real es mayor de lo calibrado, la transferencia trasera está
    # sub-estimada exactamente en el régimen HIGH-ay.
    for h_cg in [0.285, 0.310, 0.330, 0.350]:
        hyps[f"h_cg={h_cg:.3f}"] = {"h_cg": h_cg}

    # H7: track_r — un track trasero efectivo más ESTRECHO (p.ej. por
    # compliance/roll de la mangueta bajo carga) aumenta dFz_lat_r a igual
    # ay, incrementando Fz_rl/Fz_rr en el pico y por tanto Fy_r disponible.
    for tr in [1.180, 1.120, 1.060, 1.000]:
        hyps[f"track_rear={tr:.3f}"] = {"track_rear": tr}

    # H8: Iz — momento de inercia yaw incorrecto no cambia ay directamente
    # pero sí wz, que retroalimenta a través de vx*wz en el kappa/alpha de
    # cada rueda. Incluido como control de la hipótesis "no es Iz".
    for iz in [120.0, 150.0, 180.0, 220.0]:
        hyps[f"Iz={iz:.0f}"] = {"Iz": iz}

    # H9: combinación h_cg alto + track_r estrecho — máxima transferencia
    # de carga trasera físicamente plausible, para ver el techo de mejora.
    hyps["h_cg=0.330+track_r=1.060"] = {"h_cg": 0.330, "track_rear": 1.060}

    return hyps


def run_structural_hypothesis(name, overrides, x0_batch_builder_args):
    """Instancia un vehículo nuevo con VP parcheado y puntúa contra el batch."""
    vp_patched = copy.deepcopy(VP_DICT_BASE)
    vp_patched.update(overrides)
    vehicle_h = DifferentiableMultiBodyVehicle(vp_patched, TP_DICT)

    # El batch de x0/u fue construido con el vehículo BASE (make_initial_state
    # no depende de vp salvo wheel_radius, que no tocamos) — reutilizable.
    x0_batch, u_batch, real_wz, real_ay, tags = x0_batch_builder_args
    tire_cal = jnp.array([1.463, 1.890, -1.0, 1.0, 0.382, 3.000], dtype=jnp.float32)
    return score_hypothesis(vehicle_h, x0_batch, u_batch, real_wz, real_ay, tags, tire_cal)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Control: ¿el filtro _lowpass_corr está distorsionando la métrica?
# ─────────────────────────────────────────────────────────────────────────────

def lowpass_control(vehicle, x0_batch, u_batch, real_wz, real_ay, tags):
    tire_cal = jnp.array([1.463, 1.890, -1.0, 1.0, 0.382, 3.000], dtype=jnp.float32)
    with_lp = score_hypothesis(vehicle, x0_batch, u_batch, real_wz, real_ay, tags,
                                tire_cal, use_lowpass=True)
    without_lp = score_hypothesis(vehicle, x0_batch, u_batch, real_wz, real_ay, tags,
                                   tire_cal, use_lowpass=False)
    return with_lp, without_lp


# ─────────────────────────────────────────────────────────────────────────────
# §6  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hard", type=int, default=30)
    ap.add_argument("--n-control", type=int, default=15)
    ap.add_argument("--cheap-only", action="store_true",
                     help="Solo hipótesis de tire_cal (rápido, ~1-2 min)")
    args = ap.parse_args()

    print("=" * 78)
    print("  PROJECT-GP · BARRIDO DE HIPÓTESIS — DEFECTO ESTRUCTURAL EJE TRASERO")
    print("=" * 78)

    vehicle = DifferentiableMultiBodyVehicle(VP_DICT_BASE, TP_DICT)

    # Cargar calibración global de producción para clasificar las peores ventanas reales
    steer_sign_global, tire_cal_baseline = load_production_calibration()

    t0 = time.time()
    u_batch, x0_batch, real_wz, real_ay, tags = build_hard_batch(
        vehicle, steer_sign_global, tire_cal_baseline,
        n_hard=args.n_hard, n_control=args.n_control)
    print(f"[batch] construido en {time.time()-t0:.1f}s")

    # ── Control: filtro lowpass ──────────────────────────────────────────
    print("\n" + "─" * 78)
    print("  CONTROL: ¿_lowpass_corr distorsiona la métrica en ventanas de 0.5s?")
    print("─" * 78)
    with_lp, without_lp = lowpass_control(vehicle, x0_batch, u_batch, real_wz, real_ay, tags)
    print(f"  {'':30s} {'r_ay HARD':>12} {'r_ay CTRL':>12} {'rmse HARD':>12}")
    print(f"  {'con lowpass (actual)':30s} {with_lp['r_ay_hard']:>12.3f} "
          f"{with_lp['r_ay_ctrl']:>12.3f} {with_lp['rmse_hard']:>12.3f}")
    print(f"  {'sin lowpass':30s} {without_lp['r_ay_hard']:>12.3f} "
          f"{without_lp['r_ay_ctrl']:>12.3f} {without_lp['rmse_hard']:>12.3f}")

    # ── Hipótesis baratas (tire_cal) ─────────────────────────────────────
    print("\n" + "─" * 78)
    print("  HIPÓTESIS BARATAS (solo tire_cal — mismo vehículo compilado)")
    print("─" * 78)
    results = []
    for name, tc in cheap_hypotheses().items():
        tire_cal = jnp.array(tc, dtype=jnp.float32)
        r = score_hypothesis(vehicle, x0_batch, u_batch, real_wz, real_ay, tags, tire_cal)
        r["name"] = name
        results.append(r)

    df_cheap = pd.DataFrame(results).sort_values("r_ay_hard", ascending=False)
    print(f"\n  {'Hipótesis':40s} {'r_ay HARD':>10} {'r_ay CTRL':>10} "
          f"{'rmse HARD':>10} {'rmse CTRL':>10}")
    for _, r in df_cheap.iterrows():
        print(f"  {r['name']:40s} {r['r_ay_hard']:>10.3f} {r['r_ay_ctrl']:>10.3f} "
              f"{r['rmse_hard']:>10.3f} {r['rmse_ctrl']:>10.3f}")

    out_dir = Path("reports") / "hypothesis_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_cheap.to_csv(out_dir / "cheap_hypotheses.csv", index=False)
    print(f"\n  [saved] {out_dir / 'cheap_hypotheses.csv'}")

    if args.cheap_only:
        print("\n[--cheap-only] Saltando hipótesis estructurales.")
        return

    # ── Hipótesis estructurales (VP parcheado, más caras) ────────────────
    print("\n" + "─" * 78)
    print("  HIPÓTESIS ESTRUCTURALES (VP parcheado — nueva compilación c/u)")
    print("─" * 78)
    struct_results = []
    batch_args = (x0_batch, u_batch, real_wz, real_ay, tags)
    for name, overrides in structural_hypotheses().items():
        t1 = time.time()
        r = run_structural_hypothesis(name, overrides, batch_args)
        r["name"] = name
        r["compile_s"] = time.time() - t1
        struct_results.append(r)
        print(f"  {name:35s} r_ay_hard={r['r_ay_hard']:+.3f}  "
              f"rmse_hard={r['rmse_hard']:.3f}  ({r['compile_s']:.1f}s)")

    df_struct = pd.DataFrame(struct_results).sort_values("r_ay_hard", ascending=False)
    df_struct.to_csv(out_dir / "structural_hypotheses.csv", index=False)
    print(f"\n  [saved] {out_dir / 'structural_hypotheses.csv'}")

    # ── Resumen final ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  TOP-5 GLOBAL (baratas + estructurales) POR r_ay EN VENTANAS DURAS")
    print("=" * 78)
    df_all = pd.concat([
        df_cheap[["name", "r_ay_hard", "r_ay_ctrl", "rmse_hard"]],
        df_struct[["name", "r_ay_hard", "r_ay_ctrl", "rmse_hard"]],
    ]).sort_values("r_ay_hard", ascending=False)
    print(df_all.head(5).to_string(index=False))

    print(f"\n  Baseline actual (fleet score 91.95%): r_ay_hard={with_lp['r_ay_hard']:+.3f}")
    print("  Combina el ganador de tire_cal con el ganador estructural y vuelve a\n"
          "  correr scripts/calibrate_mu_from_telemetry.py con esos VP fijos para\n"
          "  reoptimizar mu_f/mu_r/gains alrededor del nuevo punto — luego valida\n"
          "  con 'python -m scripts.run_can_backtest --debug' en la flota completa.")


if __name__ == "__main__":
    main()