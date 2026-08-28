#!/usr/bin/env python3
# scripts/compute_optimal_bias.py
# ==============================================================================
# PROJECT-GP: Cálculo del reparto óptimo de frenada y tracción (Ter27, 4WD)
# ==============================================================================
# Independiente de la geometría de suspensión: el bias óptimo de frenada
# iguala la utilización de fricción front/rear bajo la transferencia de
# carga dinámica a la deceleración de diseño. El anti-dive/anti-squat
# geométrico solo mitiga el squat/dive TRANSITORIO — no debe usarse como
# variable libre para "cuadrar" el bias, porque eso puede darte un reparto
# que frena mal el coche aunque el ángulo geométrico salga bonito.
#
# USO:
#   python scripts/compute_optimal_bias.py
#   python scripts/compute_optimal_bias.py --mass 280 --lf 0.806 --lr 0.744 \
#       --h-cg 0.310 --mu 1.4 --ax-decel-g 1.3 --ax-accel-g 0.8
# ==============================================================================

import argparse
import numpy as np


def compute_ideal_brake_bias(m, lf, lr, h_cg, ax_decel_g, g=9.81):
    """
    Bias de frenada que iguala Fz_f/Fz_r bajo transferencia de carga dinámica,
    es decir, el reparto que hace que ambos ejes alcancen el límite de
    fricción a la vez (frenada óptima, sin bloqueo prematuro de ningún eje).
    """
    L = lf + lr
    ax = ax_decel_g * g
    Fz_f = m * g * lr / L + m * ax * h_cg / L
    Fz_r = m * g * lf / L - m * ax * h_cg / L
    Fz_r = max(Fz_r, 1.0)  # evita división por cero / negativo a G extremas
    return Fz_f / (Fz_f + Fz_r)


def compute_ideal_drive_bias(m, lf, lr, h_cg, ax_accel_g, g=9.81):
    """
    Análogo para tracción 4WD bajo aceleración de diseño. Bajo aceleración
    el eje delantero se descarga, así que el drive_bias_f óptimo es MENOR
    que 0.5 — la mayoría del par debe ir al eje trasero, que gana carga.
    """
    L = lf + lr
    ax = ax_accel_g * g
    Fz_f = m * g * lr / L - m * ax * h_cg / L
    Fz_r = m * g * lf / L + m * ax * h_cg / L
    Fz_f = max(Fz_f, 1.0)
    return Fz_f / (Fz_f + Fz_r)


def bias_sweep(m, lf, lr, h_cg, mu, g=9.81, n=25):
    """
    Barrido de bias en función de la magnitud de la maniobra (ax_g variable),
    útil para ver cómo el óptimo se desplaza entre frenadas suaves y al límite.
    """
    ax_range = np.linspace(0.2, mu, n)   # de suave a límite de fricción
    print(f"\n{'ax [G]':>8}  {'brake_bias_f ideal':>20}  {'drive_bias_f ideal':>20}")
    print("-" * 52)
    for ax_g in ax_range:
        bb = compute_ideal_brake_bias(m, lf, lr, h_cg, ax_g, g)
        db = compute_ideal_drive_bias(m, lf, lr, h_cg, ax_g, g)
        print(f"{ax_g:8.2f}  {bb:20.4f}  {db:20.4f}")


def main():
    ap = argparse.ArgumentParser(description="Reparto óptimo de frenada/tracción — Ter27 4WD")
    ap.add_argument("--mass",       type=float, default=280.0,  help="masa total [kg]")
    ap.add_argument("--lf",         type=float, default=0.806,  help="CG->eje delantero [m]")
    ap.add_argument("--lr",         type=float, default=0.744,  help="CG->eje trasero [m]")
    ap.add_argument("--h-cg",       type=float, default=0.310,  help="altura CG [m]")
    ap.add_argument("--mu",         type=float, default=1.40,   help="coef. fricción neumático (para el sweep)")
    ap.add_argument("--ax-decel-g", type=float, default=1.30,   help="deceleración de diseño para brake_bias [G]")
    ap.add_argument("--ax-accel-g", type=float, default=0.80,   help="aceleración de diseño para drive_bias [G]")
    ap.add_argument("--sweep",      action="store_true", help="mostrar barrido completo de G")
    args = ap.parse_args()

    L = args.lf + args.lr
    print("=" * 62)
    print("  PROJECT-GP · REPARTO ÓPTIMO DE FRENADA / TRACCIÓN (TER27)")
    print("=" * 62)
    print(f"  m={args.mass} kg  lf={args.lf} m  lr={args.lr} m  "
          f"h_cg={args.h_cg} m  wheelbase={L:.3f} m")

    bb_ideal = compute_ideal_brake_bias(args.mass, args.lf, args.lr, args.h_cg, args.ax_decel_g)
    db_ideal = compute_ideal_drive_bias(args.mass, args.lf, args.lr, args.h_cg, args.ax_accel_g)

    print(f"\n  brake_bias_f ideal @ {args.ax_decel_g:.2f}G decel : {bb_ideal:.4f}  "
          f"({bb_ideal*100:.1f}% delante)")
    print(f"  drive_bias_f ideal @ {args.ax_accel_g:.2f}G accel : {db_ideal:.4f}  "
          f"({db_ideal*100:.1f}% delante)")

    # Estático (referencia, ax=0)
    static_f = args.lr / L
    print(f"\n  [ref] reparto de peso estático delante: {static_f*100:.1f}%  "
          f"(sin transferencia de carga)")

    if args.sweep:
        bias_sweep(args.mass, args.lf, args.lr, args.h_cg, args.mu)

    print("\n  → Copia estos valores como force_fractions en "
          "interactive_kinematic_optimizer.py:")
    print(f"    brake_bias_f = {bb_ideal:.3f}")
    print(f"    drive_bias_f = {db_ideal:.3f}")
    print("=" * 62)


if __name__ == "__main__":
    main()