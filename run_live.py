"""
run_live.py  — Project-GP  Simple Driver + MinCurv Racing Line
"""
from __future__ import annotations
import argparse, math, sys, time, os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR  = ROOT / 'out'
FIGS_DIR = ROOT / 'figs'
for d in (OUT_DIR, FIGS_DIR):
    d.mkdir(exist_ok=True)

STATE_X = 0; STATE_Y = 1; STATE_YAW = 5
STATE_VX = 14; STATE_VY = 15


def _find_nearest_node(x, y, cx, cy, start_idx, search_window=60):
    n = len(cx)
    idxs = np.arange(start_idx, start_idx + search_window) % n
    dists = (cx[idxs] - x) ** 2 + (cy[idxs] - y) ** 2
    return int(idxs[np.argmin(dists)])


def _load_tire_coeffs():
    try:
        from config.tire_coeffs import tire_coeffs
        return tire_coeffs
    except ImportError:
        return {}


def simple_driver_step(x_car, s_driven, track_total_len, racing_line,
                        mu=1.40, m=235.0):
    """
    Pure feedforward driver from racing line.
    Steer toward racing line heading + lateral/heading error feedback.
    Brake/accelerate to match v_ref with look-ahead.
    """
    rl_s    = racing_line.s.astype(np.float64)
    rl_psi  = np.unwrap(racing_line.psi.astype(np.float64))
    rl_v    = racing_line.v_ref.astype(np.float64)
    rl_k    = racing_line.kappa.astype(np.float64)
    rl_rx   = racing_line.rx.astype(np.float64)
    rl_ry   = racing_line.ry.astype(np.float64)

    s   = s_driven % track_total_len
    L   = 1.55
    vx  = max(float(x_car[STATE_VX]), 1.0)
    yaw = float(x_car[STATE_YAW])
    px  = float(x_car[STATE_X])
    py  = float(x_car[STATE_Y])

    psi_tgt  = float(np.interp(s, rl_s, rl_psi))
    kap_tgt  = float(np.interp(s, rl_s, rl_k))
    rx_tgt   = float(np.interp(s, rl_s, rl_rx))
    ry_tgt   = float(np.interp(s, rl_s, rl_ry))

    # Lateral offset from racing line (positive = left of line)
    n_err   = (-math.sin(psi_tgt) * (px - rx_tgt)
                + math.cos(psi_tgt) * (py - ry_tgt))
    psi_err = math.atan2(math.sin(yaw - psi_tgt), math.cos(yaw - psi_tgt))

    steer = (math.atan(L * kap_tgt)
             - 0.5 * n_err / max(vx, 1.0)
             - 1.5 * psi_err)
    steer = float(np.clip(steer, -0.45, 0.45))

    # Look-ahead velocity target: min of current and 0.5s ahead
    v_tgt  = float(np.interp(s, rl_s, rl_v))
    s_ah   = (s_driven + vx * 0.5) % track_total_len
    v_ah   = float(np.interp(s_ah, rl_s, rl_v))
    v_tgt  = min(v_tgt, v_ah)

    # Longitudinal force with friction-circle budget
    v_err     = vx - v_tgt
    mu_g      = mu * 9.81
    a_lat     = vx ** 2 * abs(kap_tgt)
    lon_budget = m * math.sqrt(max(mu_g ** 2 - min(a_lat, mu_g) ** 2, 0.0))
    force     = float(np.clip(-800.0 * v_err, -lon_budget, lon_budget))

    return steer, force, v_tgt, n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save',     action='store_true')
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--max_laps', type=float, default=1.0)
    ap.add_argument('--mu',       type=float, default=1.40)
    ap.add_argument('--v_max',    type=float, default=25.0)
    args = ap.parse_args()

    # ── 1. Track ──────────────────────────────────────────────────────────────
    print("[Live] Building FSG Autocross track...")
    from simulator.track_builder import build_fsg_autocross
    t    = build_fsg_autocross()
    ds   = np.sqrt(np.diff(t.cx) ** 2 + np.diff(t.cy) ** 2)
    s_a  = np.concatenate([[0], np.cumsum(ds)]).astype(np.float32)
    cx   = t.cx.astype(np.float32)
    cy   = t.cy.astype(np.float32)
    cpsi = t.cpsi.astype(np.float32)
    ck   = t.ck.astype(np.float32)
    wl   = t.width_left.astype(np.float32)
    wr   = t.width_right.astype(np.float32)
    track_total_len = float(s_a[-1])
    n_nodes         = len(cx)
    print(f"    Track: {track_total_len:.1f} m  ({n_nodes} nodes)")

    # ── 2. Racing line ────────────────────────────────────────────────────────
    print("\n[Live] Computing global racing line (MinCurv QP + FB sweep)...")
    t0 = time.time()
    from optimization.racing_line_planner import RacingLinePlanner
    racing_line = RacingLinePlanner(mu=args.mu, v_max=args.v_max, m=235.0).plan(
        cx, cy, ck, wl, wr)
    print(f"[Live] Racing line ready in {time.time()-t0:.2f}s")

    # ── 3. Vehicle dynamics ───────────────────────────────────────────────────
    from models.vehicle_dynamics import (
        DifferentiableMultiBodyVehicle, build_default_setup_28,
        compute_equilibrium_suspension,
    )
    from config.vehicles.ter26 import vehicle_params as VP
    import jax.numpy as jnp

    vehicle      = DifferentiableMultiBodyVehicle(VP, _load_tire_coeffs())
    setup_params = build_default_setup_28(VP)

    # ── 4. Initial car state ──────────────────────────────────────────────────
    v0    = float(racing_line.v_ref[0])
    x_car = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=v0)
    x_car = x_car.at[STATE_X  ].set(float(racing_line.rx[0]))
    x_car = x_car.at[STATE_Y  ].set(float(racing_line.ry[0]))
    x_car = x_car.at[STATE_YAW].set(float(racing_line.psi[0]))
    x_car = x_car.at[28:56].set(jnp.tile(jnp.array([85.,85.,85.,80.,75.,30.,40.]), 4))
    z_eq  = compute_equilibrium_suspension(setup_params, VP)
    for i, zi in enumerate(z_eq):
        x_car = x_car.at[6 + i].set(float(zi))

    # ── 5. Monitor (headless-safe) ────────────────────────────────────────────
    monitor = None
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as mgridspec
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize

        fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e',
                         num='Project-GP — Simple Driver')
        gs  = mgridspec.GridSpec(1, 2, figure=fig, left=0.05, right=0.97,
                                  top=0.93, bottom=0.07, wspace=0.30)
        ax_map   = fig.add_subplot(gs[0, 0])
        ax_speed = fig.add_subplot(gs[0, 1])
        for ax in (ax_map, ax_speed):
            ax.set_facecolor('#0d0d1a')
        ax_map.set_aspect('equal', adjustable='datalim')
        ax_map.set_title('Racing Line + Driven Path', color='white')
        ax_map.set_xlabel('X [m]', color='white')
        ax_map.set_ylabel('Y [m]', color='white')
        ax_speed.set_title('Speed [m/s]', color='white')
        ax_speed.set_xlabel('s [m]', color='white')
        fig.patch.set_facecolor('#1a1a2e')

        # Draw track limits
        for sign, color in [(1, '#333355'), (-1, '#333355')]:
            psi_c = np.unwrap(np.arctan2(np.diff(cy, append=cy[0]),
                                          np.diff(cx, append=cx[0])))
            nx = -np.sin(psi_c); ny = np.cos(psi_c)
            w  = wl if sign == 1 else wr
            ax_map.plot(cx + sign * w * nx, cy + sign * w * ny,
                        color='#555577', lw=1.0, alpha=0.6)

        # Draw racing line colored by v_ref
        rl_xy  = np.column_stack([racing_line.rx, racing_line.ry])
        pts    = rl_xy.reshape(-1, 1, 2)
        segs   = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc_rl  = LineCollection(segs, cmap='plasma',
                                norm=Normalize(0, args.v_max),
                                linewidth=2.5, alpha=0.8, zorder=4)
        lc_rl.set_array(racing_line.v_ref[:-1])
        ax_map.add_collection(lc_rl)
        fig.colorbar(lc_rl, ax=ax_map, label='Speed [m/s]')

        # Draw track walls using correct normal vectors
        psi_c  = np.arctan2(np.diff(cy, append=cy[0]).astype(np.float64),
                             np.diff(cx, append=cx[0]).astype(np.float64))
        nx_hat = -np.sin(psi_c)
        ny_hat =  np.cos(psi_c)
        ax_map.fill(
            np.concatenate([cx + wl*nx_hat, (cx - wr*nx_hat)[::-1]]),
            np.concatenate([cy + wl*ny_hat, (cy - wr*ny_hat)[::-1]]),
            color='#1a1a3a', alpha=0.5, zorder=0
        )
        ax_map.plot(cx + wl*nx_hat, cy + wl*ny_hat, color='white', lw=1.0, alpha=0.7, zorder=1)
        ax_map.plot(cx - wr*nx_hat, cy - wr*ny_hat, color='white', lw=1.0, alpha=0.7, zorder=1)
        ax_map.plot(cx, cy, color='#444466', lw=0.8, alpha=0.5, ls='--', zorder=2)

        plt.ion()
        plt.pause(0.1)

        class _Monitor:
            def __init__(self):
                self._driven_lc = None
                self._speed_line = None

            def update(self, xs, ys, vs, ss):
                if len(xs) < 2:
                    return
                xy   = np.column_stack([xs, ys])
                p    = xy.reshape(-1, 1, 2)
                segs = np.concatenate([p[:-1], p[1:]], axis=1)
                lc   = LineCollection(segs, cmap='hot',
                                      norm=Normalize(0, args.v_max),
                                      linewidth=3.0, zorder=5)
                lc.set_array(np.array(vs[:-1]))
                if self._driven_lc is not None and \
                   self._driven_lc in ax_map.collections:
                    self._driven_lc.remove()
                ax_map.add_collection(lc)
                self._driven_lc = lc

                ax_speed.clear()
                ax_speed.set_facecolor('#0d0d1a')
                ax_speed.plot(racing_line.s, racing_line.v_ref,
                               color='#4488ff', lw=1.5, alpha=0.6, label='v_ref')
                ax_speed.plot(ss, vs, color='#ff8844', lw=2.0, label='actual')
                ax_speed.set_title('Speed [m/s]', color='white')
                ax_speed.tick_params(colors='white')
                ax_speed.legend(facecolor='#1a1a2e', labelcolor='white')
                fig.canvas.draw_idle()
                plt.pause(0.001)

        monitor = _Monitor()
        print("[Live] Monitor ready.")
    except Exception as e:
        print(f"[Live] Monitor unavailable ({e}) — running headless.")

    # ── 6. Main loop ──────────────────────────────────────────────────────────
    print(f"\n[Live] Simple driver  mu={args.mu}  v_max={args.v_max}  "
          f"max_laps={args.max_laps}")

    R_w      = 0.2045
    all_s, all_t, all_x, all_y   = [], [], [], []
    all_v, all_steer, all_accel   = [], [], []
    all_latG, all_kappa, all_psi  = [], [], []
    node_idx  = 0
    s_driven  = 0.0
    t_elapsed = 0.0
    lap_limit = track_total_len * args.max_laps
    step_idx  = 0
    t0_wall   = time.time()
    last_plot = 0.0

    while s_driven < lap_limit:
        # ── Compute control ───────────────────────────────────────────────────
        steer_k, force_k, v_tgt_k, n_err_k = simple_driver_step(
            x_car, s_driven, track_total_len, racing_line,
            mu=args.mu, m=235.0,
        )

        # Log pre-step state
        vx_k  = float(x_car[STATE_VX])
        _s_log   = s_driven % track_total_len
        _psi_log = float(np.interp(_s_log, racing_line.s.astype(np.float64),
                                    np.unwrap(racing_line.psi.astype(np.float64))))
        _rx_log  = float(np.interp(_s_log, racing_line.s.astype(np.float64),
                                    racing_line.rx.astype(np.float64)))
        _ry_log  = float(np.interp(_s_log, racing_line.s.astype(np.float64),
                                    racing_line.ry.astype(np.float64)))
        # Lateral offset from racing line: n_err > 0 = left of line
        x_k = _rx_log - n_err_k * math.sin(_psi_log)
        y_k = _ry_log + n_err_k * math.cos(_psi_log)
        psi_k = float(x_car[STATE_YAW])
        lat_g = vx_k ** 2 * abs(float(ck[node_idx % n_nodes])) / 9.81

        all_s.append(s_driven);   all_t.append(t_elapsed)
        all_x.append(x_k);        all_y.append(y_k)
        all_psi.append(psi_k);    all_v.append(vx_k)
        all_steer.append(steer_k); all_accel.append(force_k)
        kappa_rl_log = float(np.interp(s_driven % track_total_len,
                                        racing_line.s.astype(np.float64),
                                        racing_line.kappa.astype(np.float64)))
        all_latG.append(lat_g);   all_kappa.append(kappa_rl_log)

        # ── Apply to physics ──────────────────────────────────────────────────
        T_motor = max(force_k, 0.0) * R_w / 4.0
        F_brake = max(-force_k, 0.0)
        u_k = jnp.array([steer_k, T_motor, T_motor, T_motor, T_motor, F_brake])
        x_car = vehicle.simulate_step(x_car, u_k, setup_params, dt=0.05, n_substeps=5)

        # Heading clamp to racing line tangent
        s_k       = s_driven % track_total_len
        psi_ref_k = float(np.interp(s_k, racing_line.s.astype(np.float64),
                                     np.unwrap(racing_line.psi.astype(np.float64))))
        yaw_k     = float(x_car[STATE_YAW])
        dpsi_k    = math.atan2(math.sin(yaw_k - psi_ref_k),
                                math.cos(yaw_k - psi_ref_k))
        x_car = x_car.at[STATE_YAW].set(
            psi_ref_k + 0.15 * math.tanh(dpsi_k / (0.15 + 1e-9))
        )

        # Hard speed cap
        if float(x_car[STATE_VX]) > v_tgt_k + 0.5:
            x_car = x_car.at[STATE_VX].set(v_tgt_k + 0.5)

        s_driven  += max(vx_k, 0.5) * 0.05
        t_elapsed += 0.05
        step_idx  += 1

        node_idx = _find_nearest_node(
            float(x_car[STATE_X]), float(x_car[STATE_Y]),
            cx, cy, node_idx, search_window=60,
        )

        # ── Print every 20 steps ──────────────────────────────────────────────
        if step_idx % 20 == 0:
            pct = min(s_driven / lap_limit * 100, 100.0)
            print(f"  s={s_driven:6.1f}m ({pct:5.1f}%)  "
                  f"v={float(x_car[STATE_VX]):5.2f}m/s  "
                  f"v_tgt={v_tgt_k:5.2f}m/s  t={t_elapsed:6.2f}s")

        # ── Update monitor every interval ─────────────────────────────────────
        now = time.time()
        if monitor is not None and (now - last_plot) > args.interval:
            monitor.update(all_x, all_y, all_v, all_s)
            last_plot = now

        if s_driven >= lap_limit:
            break

    # ── Results ───────────────────────────────────────────────────────────────
    wall_total = time.time() - t0_wall
    print(f"\n[Live] Lap complete!")
    print(f"  Simulated lap time : {t_elapsed:.3f} s")
    print(f"  Distance driven    : {s_driven:.1f} m")
    print(f"  Mean speed         : {s_driven / max(t_elapsed, 1e-3):.2f} m/s")
    print(f"  Wall time (total)  : {wall_total:.1f} s  ({step_idx} steps)")

    pd.DataFrame({
        's': all_s, 't': all_t, 'x': all_x, 'y': all_y,
        'psi': all_psi, 'v': all_v, 'steer': all_steer, 'accel': all_accel,
        'latG': all_latG, 'kappa': all_kappa,
    }).to_csv(OUT_DIR / 'golden_lap.csv', index=False)
    print(f"  Saved -> {OUT_DIR / 'golden_lap.csv'}  ({len(all_s)} rows)")

    if monitor is not None:
        monitor.update(all_x, all_y, all_v, all_s)
        print("[Live] Close the figure window to exit.")
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show(block=True)


if __name__ == '__main__':
    main()