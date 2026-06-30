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
# ── new module-level constants/helper, alongside STATE_X etc. ──────────────
STATE_WZ = 19

def _estimate_corner_loads(vx, ax_decel, ay_left, mass, lf, lr, h_cg, t_f, t_r,
                            Fz_floor: float = 80.0):
    """
    Quasi-static diagonal load transfer — same formula as
    sanity_checks.test_diagonal_load_transfer() / objectives.compute_skidpad_objective(),
    not a fresh derivation.

    ax_decel > 0 : braking   (load shifts FORWARD)
    ay_left  > 0 : left turn (load shifts to the RIGHT axle pair), matching the
                   project-wide curvature convention "positive kappa = left turn"
                   (differentiable_track.py). NOT independently re-verified against
                   racing_line_planner.py's sign convention — if RL/RR look swapped
                   in telemetry, this is the first place to check.

    A static Fz split is exactly wrong when it matters most: under trail braking,
    the SOCP/CBF friction budget on the unloaded rear-inside wheel is the binding
    constraint, and a flat 720/780N guess hides that from the allocator.
    """
    L = lf + lr
    Fz_fl = mass*9.81*lr/(L*2) - mass*ay_left*h_cg/(2*t_f)*0.5 + mass*ax_decel*h_cg/(2*L)
    Fz_fr = mass*9.81*lr/(L*2) + mass*ay_left*h_cg/(2*t_f)*0.5 + mass*ax_decel*h_cg/(2*L)
    Fz_rl = mass*9.81*lf/(L*2) - mass*ay_left*h_cg/(2*t_r)*0.5 - mass*ax_decel*h_cg/(2*L)
    Fz_rr = mass*9.81*lf/(L*2) + mass*ay_left*h_cg/(2*t_r)*0.5 - mass*ax_decel*h_cg/(2*L)
    return jnp.array([max(Fz_fl, Fz_floor), max(Fz_fr, Fz_floor),
                       max(Fz_rl, Fz_floor), max(Fz_rr, Fz_floor)])

def _make_powertrain_substep_scan(vehicle, setup_vec, pt_config, geom, n_substeps, dt_ctrl):
    m_veh, lf_g, lr_g, h_cg_g, t_f_g, t_r_g, r_w = geom

    @jax.jit
    def _scan_fn(x0, manager_state, steer, throttle, brake, force, kappa, mu_est):
        def body(carry, _):
            x_c, ms = carry
            vx_c, wz_c = x_c[STATE_VX], x_c[STATE_WZ]
            L = lf_g + lr_g
            ax_decel, ay_left = -force / m_veh, vx_c ** 2 * kappa
            Fz_fl = m_veh*9.81*lr_g/(L*2) - m_veh*ay_left*h_cg_g/(2*t_f_g)*0.5 + m_veh*ax_decel*h_cg_g/(2*L)
            Fz_fr = m_veh*9.81*lr_g/(L*2) + m_veh*ay_left*h_cg_g/(2*t_f_g)*0.5 + m_veh*ax_decel*h_cg_g/(2*L)
            Fz_rl = m_veh*9.81*lf_g/(L*2) - m_veh*ay_left*h_cg_g/(2*t_r_g)*0.5 - m_veh*ax_decel*h_cg_g/(2*L)
            Fz_rr = m_veh*9.81*lf_g/(L*2) + m_veh*ay_left*h_cg_g/(2*t_r_g)*0.5 - m_veh*ax_decel*h_cg_g/(2*L)
            Fz_k  = jnp.maximum(jnp.array([Fz_fl, Fz_fr, Fz_rl, Fz_rr]), 80.0)

            diag, ms_next = powertrain_step(
                throttle_raw=throttle, brake_raw=brake, delta=steer,
                vx=vx_c, vy=x_c[STATE_VY], wz=wz_c, Fz=Fz_k, Fy=jnp.zeros(4),
                omega_wheel=jnp.full(4, vx_c / r_w),
                alpha_t=x_c[56:72:4],
                T_tire=jnp.mean(x_c[28:56].reshape(4, 7)[:, :3], axis=1),
                mu_est=mu_est, gp_sigma=jnp.array(0.05), curvature=kappa,
                manager_state=ms, dt=jnp.array(dt_ctrl), config=pt_config,
            )
            u_k = jnp.concatenate([steer[None], diag.T_wheel, diag.F_brake_hydraulic[None]])
            x_next = vehicle.simulate_step(x_c, u_k, setup_vec, dt=dt_ctrl, n_substeps=1)
            return (x_next, ms_next), diag

        (x_final, ms_final), diags = jax.lax.scan(body, (x0, manager_state), None, length=n_substeps)
        return x_final, ms_final, diags

    return _scan_fn

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
    ap.add_argument('--log_powertrain', action='store_true',
                     help='Dump (state, T_wheel) pairs at 200 Hz for offline EDMD/Koopman retraining.')
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

    # ── 3. Vehicle dynamics + powertrain stack ────────────────────────────────
    import jax
    import jax.numpy as jnp
    from models.vehicle_dynamics import (
        DifferentiableMultiBodyVehicle, build_default_setup_28,
        compute_equilibrium_suspension,
    )
    from config.vehicles.ter26 import vehicle_params as VP
    from powertrain.powertrain_manager import make_powertrain_manager, powertrain_step

    vehicle      = DifferentiableMultiBodyVehicle(VP, _load_tire_coeffs())
    setup_params = build_default_setup_28(VP)

    pt_config, pt_manager_state = make_powertrain_manager(VP)
    pt_config = pt_config._replace(is_rwd=True)   # Ter26 reference platform is RWD, not Ter27 AWD

    r_w    = float(pt_config.geo.r_w)
    m_veh  = float(pt_config.geo.mass)
    lf_g   = float(VP.get('lf', 0.8525))
    lr_g   = float(VP.get('lr', 0.6975))
    h_cg_g = float(VP.get('h_cg', 0.285))
    t_f_g  = float(VP.get('track_front', 1.20))
    t_r_g  = float(VP.get('track_rear', 1.18))

    MACRO_DT   = 0.05      # visual/driver-update frame (unchanged from baseline)
    DT_CTRL    = 0.005     # native 200 Hz powertrain rate (powertrain_manager smoke_test, README §12.1)
    N_SUBSTEPS = round(MACRO_DT / DT_CTRL)   # 10

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

        for sign, color in [(1, '#333355'), (-1, '#333355')]:
            psi_c = np.unwrap(np.arctan2(np.diff(cy, append=cy[0]),
                                          np.diff(cx, append=cx[0])))
            nx = -np.sin(psi_c); ny = np.cos(psi_c)
            w  = wl if sign == 1 else wr
            ax_map.plot(cx + sign * w * nx, cy + sign * w * ny,
                        color='#555577', lw=1.0, alpha=0.6)

        rl_xy  = np.column_stack([racing_line.rx, racing_line.ry])
        pts    = rl_xy.reshape(-1, 1, 2)
        segs   = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc_rl  = LineCollection(segs, cmap='plasma',
                                norm=Normalize(0, args.v_max),
                                linewidth=2.5, alpha=0.8, zorder=4)
        lc_rl.set_array(racing_line.v_ref[:-1])
        ax_map.add_collection(lc_rl)
        fig.colorbar(lc_rl, ax=ax_map, label='Speed [m/s]')

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
                if self._driven_lc is not None and self._driven_lc in ax_map.collections:
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
    print(f"\n[Live] Driver + powertrain manager  mu={args.mu}  v_max={args.v_max}  "
          f"max_laps={args.max_laps}  ctrl={1/DT_CTRL:.0f}Hz")

    all_s, all_t, all_x, all_y   = [], [], [], []
    all_v, all_steer, all_accel   = [], [], []
    all_latG, all_kappa, all_psi  = [], [], []
    pt_log    = [] if args.log_powertrain else None
    node_idx  = 0
    s_driven  = 0.0
    t_elapsed = 0.0
    lap_limit = track_total_len * args.max_laps
    step_idx  = 0
    t0_wall   = time.time()
    last_plot = 0.0

    while s_driven < lap_limit:
        # ── Driver command (held constant across the 10 control substeps) ─────
        steer_k, force_k, v_tgt_k, n_err_k = simple_driver_step(
            x_car, s_driven, track_total_len, racing_line,
            mu=args.mu, m=235.0,
        )
        throttle_raw = float(np.clip(max(force_k, 0.0) / 6000.0, 0.0, 1.0))
        brake_raw    = float(np.clip(max(-force_k, 0.0) / 8000.0, 0.0, 1.0))

        # Log pre-step state
        vx_k     = float(x_car[STATE_VX])
        _s_log   = s_driven % track_total_len
        _psi_log = float(np.interp(_s_log, racing_line.s.astype(np.float64),
                                    np.unwrap(racing_line.psi.astype(np.float64))))
        _rx_log  = float(np.interp(_s_log, racing_line.s.astype(np.float64),
                                    racing_line.rx.astype(np.float64)))
        _ry_log  = float(np.interp(_s_log, racing_line.s.astype(np.float64),
                                    racing_line.ry.astype(np.float64)))
        x_k = _rx_log - n_err_k * math.sin(_psi_log)
        y_k = _ry_log + n_err_k * math.cos(_psi_log)
        psi_k = float(x_car[STATE_YAW])
        lat_g = vx_k ** 2 * abs(float(ck[node_idx % n_nodes])) / 9.81
        kappa_rl_log = float(np.interp(s_driven % track_total_len,
                                        racing_line.s.astype(np.float64),
                                        racing_line.kappa.astype(np.float64)))

        all_s.append(s_driven);   all_t.append(t_elapsed)
        all_x.append(x_k);        all_y.append(y_k)
        all_psi.append(psi_k);    all_v.append(vx_k)
        all_steer.append(steer_k); all_accel.append(force_k)
        all_latG.append(lat_g);   all_kappa.append(kappa_rl_log)

        # ── 200 Hz powertrain + plant substeps under the held driver command ──
        for _ in range(N_SUBSTEPS):
            x_c  = x_car
            vx_c = float(x_c[STATE_VX])
            wz_c = float(x_c[STATE_WZ])

            ax_decel = -force_k / m_veh                       # >0 under braking
            ay_left  = vx_c ** 2 * kappa_rl_log                # >0 in a left turn
            Fz_k = _estimate_corner_loads(vx_c, ax_decel, ay_left,
                                           m_veh, lf_g, lr_g, h_cg_g, t_f_g, t_r_g)

            omega_k   = jnp.full(4, vx_c / r_w)
            alpha_t_k = x_c[56:72:4]                            # corner-major stride: [α_t_FL..RR]
            T_tire_k  = jnp.mean(x_c[28:56].reshape(4, 7)[:, :3], axis=1)  # mean surface ribs/corner

            diag, pt_manager_state = powertrain_step(
                throttle_raw=jnp.array(throttle_raw),
                brake_raw=jnp.array(brake_raw),
                delta=jnp.array(steer_k),
                vx=jnp.array(vx_c), vy=x_c[STATE_VY], wz=jnp.array(wz_c),
                Fz=Fz_k, Fy=jnp.zeros(4),
                omega_wheel=omega_k, alpha_t=alpha_t_k, T_tire=T_tire_k,
                mu_est=jnp.array(args.mu), gp_sigma=jnp.array(0.05),
                curvature=jnp.array(kappa_rl_log),
                manager_state=pt_manager_state, dt=jnp.array(DT_CTRL), config=pt_config,
            )

            u_k = jnp.concatenate([
                jnp.array([steer_k]), diag.T_wheel, jnp.array([diag.F_brake_hydraulic]),
            ])
            x_car = vehicle.simulate_step(x_c, u_k, setup_params, dt=DT_CTRL, n_substeps=1)

            if pt_log is not None:
                pt_log.append(np.concatenate([
                    np.array(x_c[:28]),
                    np.array([float(diag.Mz_target), float(diag.Mz_actual)]),
                    np.array(diag.T_wheel), np.array([float(diag.F_brake_hydraulic)]),
                    np.array([args.mu, kappa_rl_log]),
                ]))

        # ── Catastrophic-divergence safety net only (not a steering controller —
        #    CBF/TC/TV now actually regulate yaw and slip; clamping every frame
        #    would hide whether that stack is doing anything at all) ───────────
        yaw_k  = float(x_car[STATE_YAW])
        dpsi_k = math.atan2(math.sin(yaw_k - _psi_log), math.cos(yaw_k - _psi_log))
        if abs(dpsi_k) > math.radians(75.0):
            x_car = x_car.at[STATE_YAW].set(_psi_log + math.copysign(math.radians(75.0), dpsi_k))
        if float(x_car[STATE_VX]) > v_tgt_k + 5.0:
            x_car = x_car.at[STATE_VX].set(v_tgt_k + 5.0)

        s_driven  += max(vx_k, 0.5) * MACRO_DT
        t_elapsed += MACRO_DT
        step_idx  += 1

        node_idx = _find_nearest_node(
            float(x_car[STATE_X]), float(x_car[STATE_Y]),
            cx, cy, node_idx, search_window=60,
        )

        if step_idx % 20 == 0:
            pct = min(s_driven / lap_limit * 100, 100.0)
            print(f"  s={s_driven:6.1f}m ({pct:5.1f}%)  "
                  f"v={float(x_car[STATE_VX]):5.2f}m/s  "
                  f"v_tgt={v_tgt_k:5.2f}m/s  t={t_elapsed:6.2f}s  "
                  f"Mz={float(diag.Mz_actual):+6.1f}Nm  cbf={float(diag.cbf_active):.0f}")

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
    print(f"  Wall time (total)  : {wall_total:.1f} s  ({step_idx} macro steps × {N_SUBSTEPS} ctrl substeps)")

    pd.DataFrame({
        's': all_s, 't': all_t, 'x': all_x, 'y': all_y,
        'psi': all_psi, 'v': all_v, 'steer': all_steer, 'accel': all_accel,
        'latG': all_latG, 'kappa': all_kappa,
    }).to_csv(OUT_DIR / 'golden_lap.csv', index=False)
    print(f"  Saved -> {OUT_DIR / 'golden_lap.csv'}  ({len(all_s)} rows)")

    if pt_log is not None:
        np.save(OUT_DIR / 'powertrain_edmd_log.npy', np.array(pt_log))
        print(f"  Saved -> {OUT_DIR/'powertrain_edmd_log.npy'}  ({len(pt_log)} rows @ {1/DT_CTRL:.0f}Hz)")

    if monitor is not None:
        monitor.update(all_x, all_y, all_v, all_s)
        print("[Live] Close the figure window to exit.")
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show(block=True)


if __name__ == '__main__':
    main()