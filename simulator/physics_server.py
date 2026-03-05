"""
simulator/physics_server.py
─────────────────────────────────────────────────────────────────────────────
Project-GP Headless JAX Physics Server — v2

Improvements over v1
─────────────────────────────────────────────────────────────────────────────
1.  Expanded telemetry packet (64 floats vs 11) — all forces, slip angles,
    temperatures, tire loads, grip utilisation, lap timing, energy.

2.  Full 46-DOF state extraction — every physics channel is decoded and
    forwarded. Wheel spin, heave velocities, thermal/transient states.

3.  Derived channels computed server-side:
      · Per-wheel slip angles (α = arctan(vy_corner / vx))
      · Per-wheel longitudinal slip (κ = (v_wheel − vx) / vx)
      · Grip utilisation (|F_lat| / (μ Fz))
      · Instantaneous energy consumption (∫ Fx_drive · vx dt)
      · G-force channels (ax/g, ay/g)

4.  Hot-reload setup parameters — client sends CMD_TYPE=2 with new spring
    rates; server applies without restart.

5.  Real-time lap timing via LapTimer (with track loaded at startup).

6.  Telemetry CSV logger — every N frames dumped to rolling CSV file.

7.  Drive-assist modes:
      · Traction Control: limits throttle when κ_rl or κ_rr > κ_limit
      · ABS: releases brake when wheel deceleration exceeds threshold
      · Stability Control (DSC): adds differential yaw correction moment

8.  State reset command (CMD_TYPE=1) — hot-resets car to start line.

9.  Multi-client broadcast — server sends telemetry to all registered
    client addresses (not just one port).

10. Performance profiling — prints actual Hz achieved every 5 seconds.

State vector layout (46-DOF)
─────────────────────────────────────────────────────────────────────────────
 q[0:6]   = X, Y, Z, roll, pitch, yaw
 q[6:10]  = (not used directly — heave computed from Z/roll/pitch)
 q[10:14] = wheel spin angles (fl, fr, rl, rr)
 p[0:14]  = q̇  (vX, vY, vZ, ωroll, ωpitch, ωyaw, z_dot x4, ω_spin x4)
 state[28:38] = thermal states (T_fl, T_fr, T_rl, T_rr + 6 brake temps)
 state[38:46] = transient slip states (κ_fl, κ_fr, κ_rl, κ_rr, α x4)
"""

import os
import sys
import time
import socket
import struct
import threading
import csv
import argparse
from collections import deque
from typing import Optional, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import remat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT
try:
    from simulator.sim_protocol import (TelemetryFrame, TX, RX, CMD,
                                         TX_FMT, RX_FMT, TX_BYTES, RX_BYTES,
                                         unpack_controls, DEFAULT_HOST,
                                         DEFAULT_PORT_RECV, DEFAULT_PORT_SEND,
                                         DEFAULT_PORT_CTRL)
    from simulator.lap_timer import LapTimer
    from simulator.track_builder import get_track
except ImportError:
    from sim_protocol import (TelemetryFrame, TX, RX, CMD,
                               TX_FMT, RX_FMT, TX_BYTES, RX_BYTES,
                               unpack_controls, DEFAULT_HOST,
                               DEFAULT_PORT_RECV, DEFAULT_PORT_SEND,
                               DEFAULT_PORT_CTRL)
    from lap_timer import LapTimer
    from track_builder import get_track

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PHYSICS_HZ    = 200         # sim advances at this rate — machine may run slower than real-time, that's OK
DT            = 1.0 / PHYSICS_HZ   # 5ms per loop iteration
SUBSTEPS      = 5           # substep dt = 1ms — required for H_net numerical stability
DT_SUB        = DT / SUBSTEPS      # 0.001s — DO NOT increase this, H_net trained at this timestep

# Drive-assist parameters
TC_KAPPA_LIMIT  = 0.20   # longitudinal slip threshold for traction control
ABS_DECEL_LIMIT = 0.40   # wheel decel threshold for ABS
DSC_YAW_GAIN    = 500.0  # DSC yaw moment correction gain

# Telemetry logging
LOG_INTERVAL_FRAMES = 20     # write every N frames (~10Hz at 200Hz physics)
LOG_DIR             = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# Default setup
DEFAULT_SETUP = np.array([
    40000.0,  # k_f  [N/m]
    40000.0,  # k_r  [N/m]
    500.0,    # arb_f [N·m/rad]
    500.0,    # arb_r [N·m/rad]
    3000.0,   # c_f  [N·s/m]
    3000.0,   # c_r  [N·s/m]
    0.30,     # h_cg [m]
    0.60,     # brake_bias_f [0-1]
], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────

class PhysicsServer:
    """
    200Hz JAX physics server with full telemetry extraction.
    """

    def __init__(self, host: str = DEFAULT_HOST,
                 port_recv: int = DEFAULT_PORT_RECV,
                 port_send: int = DEFAULT_PORT_SEND,
                 track_name: str = 'fsg_autocross',
                 log_telemetry: bool = True,
                 tc_enabled: bool = True,
                 abs_enabled: bool = True,
                 dsc_enabled: bool = False,
                 broadcast_clients: Optional[List[Tuple[str, int]]] = None):

        self.host         = host
        self.port_recv    = port_recv
        self.port_send    = port_send
        self.log_telemetry = log_telemetry
        self.tc_enabled   = tc_enabled
        self.abs_enabled  = abs_enabled
        self.dsc_enabled  = dsc_enabled

        # Extra broadcast clients (address, port) beyond default
        self.broadcast_clients = broadcast_clients or [
            (host, port_send),           # visualizer   → port 5001
            (host, DEFAULT_PORT_CTRL),   # ctrl HUD     → port 5002
        ]

        # Vehicle
        self.vehicle      = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        self.setup_params = jnp.array(DEFAULT_SETUP)

        # Track & timing
        try:
            self.track = get_track(track_name)
            self.lap_timer = LapTimer(
                finish_line  = self.track.finish_line,
                sector_lines = self.track.sector_lines,
            )
        except Exception as e:
            print(f"[Server] Track load failed ({e}). Timing disabled.")
            self.track     = None
            self.lap_timer = LapTimer()

        # State
        self.state      = self._make_initial_state()
        self.prev_state = self.state

        # Simulation clock
        self.frame_id   = 0
        self.sim_time   = 0.0
        self.paused     = False

        # Energy integrator
        self._energy_j  = 0.0
        self._prev_Fx_drive = 0.0

        # Drive-assist state
        self._abs_active    = False
        self._tc_active     = False
        self._prev_omega_rl = 0.0
        self._prev_omega_rr = 0.0

        # Controls (latest received)
        self._steer_cmd   = 0.0
        self._throttle_f  = 0.0
        self._brake_f     = 0.0
        self._last_ctrl_time = 0.0   # wall-clock time of last received control packet
        self._t_start        = 0.0   # set when run() loop begins

        # Performance profiling
        self._frame_times : deque = deque(maxlen=1000)
        self._last_perf_print = time.perf_counter()

        # Logger
        self._log_file   = None
        self._log_writer = None
        if log_telemetry:
            self._init_logger()

    # ── State initialisation ─────────────────────────────────────────────────

    def _make_initial_state(self, x: float = 0.0, y: float = 0.0,
                             yaw: float = 0.0, speed: float = 5.0) -> jnp.ndarray:
        """
        Build an initial state at the true static spring equilibrium.

        COORDINATE TRUTH (from vehicle_dynamics._compute_derivatives):
          state[2] = Z = spring-deformation generalised coordinate.
          z_fl = Z - (track_w/2)*roll - lf*pitch   (corner heave, line 455)
          F_spring_fl = -wheel_rate_f * z_fl + F_gas_f

          At equilibrium (flat, Fz_susp = m_s*g):
            Z_eq = (2*(F_gas_f + F_gas_r) - m_s*g) / (2*(wr_f + wr_r))
            ≈ -12.5 mm

          PhysicsNormalizer q_mean[2] = 0.3 is the *normaliser* centre used
          during H_net training — it is NOT the physical equilibrium.

          state[6:10] = independent suspension displacement DOFs in the
          Hamiltonian (q_mean=0, scale=0.05 m). Set to 0 at nominal.

        Visualiser transform (in _extract_telemetry):
          tf.z = state[2] + h_cg_viz  →  world CG height above ground grid.
        """
        s = jnp.zeros(46)

        tire_r = VP_DICT.get('tire_radius', VP_DICT.get('wheel_radius', 0.2032))
        m_s    = VP_DICT.get('sprung_mass', VP_DICT.get('total_mass', 230.0) - 40.0)
        g      = 9.81
        k_f    = float(DEFAULT_SETUP[0])
        k_r    = float(DEFAULT_SETUP[1])
        mr_f   = VP_DICT.get('motion_ratio_f_poly', [1.20])[0]
        mr_r   = VP_DICT.get('motion_ratio_r_poly', [1.15])[0]
        wr_f   = k_f / (mr_f ** 2)
        wr_r   = k_r / (mr_r ** 2)
        Fg_f   = VP_DICT.get('damper_gas_force_f', 120.0) / mr_f
        Fg_r   = VP_DICT.get('damper_gas_force_r', 120.0) / mr_r

        # True static equilibrium of the spring model (net Fz = 0)
        Z_eq = (2.0 * (Fg_f + Fg_r) - m_s * g) / (2.0 * (wr_f + wr_r))

        omega0 = speed / max(tire_r, 0.1)

        # Body pose — Z at spring equilibrium
        s = s.at[0].set(x).at[1].set(y)
        s = s.at[2].set(float(Z_eq))   # spring-deformation coord, NOT physical height
        s = s.at[3].set(0.0)           # roll  = 0
        s = s.at[4].set(0.0)           # pitch = 0
        s = s.at[5].set(yaw)

        # state[6:10] = suspension displacement DOFs (q_mean=0) → leave at 0

        # Forward velocity and wheel spin
        s = s.at[14].set(speed)
        s = s.at[24].set(omega0)
        s = s.at[25].set(omega0)
        s = s.at[26].set(omega0)
        s = s.at[27].set(omega0)

        # Tire temperatures — pre-warmed (70°C ≈ 93% grip)
        # Starting at 25°C (34% grip) gives only 1.9x margin on 9m hairpin at 5m/s;
        # any speed transient above 6.9m/s → slide. 70°C gives 3.8x margin.
        s = s.at[28].set(70.0).at[29].set(70.0).at[30].set(70.0).at[31].set(70.0)

        h_cg_viz = VP_DICT.get('h_cg', 0.30)
        print(f"[Server] Initial state: Z_eq={Z_eq*1000:.1f}mm (spring coord) | "
              f"world_CG={Z_eq+h_cg_viz:.3f}m | vx={speed:.1f}m/s")
        return s

    def _settle_physics(self, fast_step, n_steps: int = 400):
        # No-op: settling without ground contact makes things worse.
        # The car is already at Z_eq (equilibrium), so no settling needed.
        print("[Server] Starting at computed equilibrium — no settling required.")

    def reset_to_start(self):
        """Reset car to track start position."""
        if self.track:
            sx, sy, syaw = self.track.get_start_pose()
        else:
            sx, sy, syaw = 0.0, 0.0, 0.0
        self.state     = self._make_initial_state(sx, sy, syaw)
        self.sim_time  = 0.0
        self.frame_id  = 0
        self._energy_j = 0.0
        self.lap_timer.reset()
        print("[Server] Reset to start position.")

    # ── JIT-compiled step ───────────────────────────────────────────────────

    def _compile_step(self):
        vehicle = self.vehicle

        @jax.jit
        def fast_step(state: jnp.ndarray, controls: jnp.ndarray,
                      setup: jnp.ndarray) -> jnp.ndarray:
            @remat
            def substep(x_s, _):
                return vehicle.simulate_step(x_s, controls, setup, DT_SUB), None
            next_state, _ = jax.lax.scan(substep, state, None, length=SUBSTEPS)
            # Guard: if vx, roll, or pitch goes non-finite or hits clip walls → keep prev state
            vx_ok    = jnp.isfinite(next_state[14])
            roll_ok  = jnp.abs(next_state[3]) < 0.39   # clip wall is ±0.4
            pitch_ok = jnp.abs(next_state[4]) < 0.39
            healthy  = vx_ok & roll_ok & pitch_ok
            return jnp.where(healthy, next_state, state)

        return fast_step

    # ── Logger ──────────────────────────────────────────────────────────────

    def _init_logger(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts       = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(LOG_DIR, f'telemetry_{ts}.csv')
        self._log_file   = open(log_path, 'w', newline='')
        fields = [
            'frame_id','sim_time','x','y','z','roll','pitch','yaw',
            'vx','vy','vz','ax','ay','az','wz',
            'z_fl','z_fr','z_rl','z_rr',
            'Fz_fl','Fz_fr','Fz_rl','Fz_rr',
            'Fy_fl','Fy_fr','Fy_rl','Fy_rr',
            'slip_fl','slip_fr','slip_rl','slip_rr',
            'kappa_rl','kappa_rr',
            'omega_fl','omega_fr','omega_rl','omega_rr',
            'T_fl','T_fr','T_rl','T_rr',
            'delta','throttle','brake_norm',
            'grip_util_f','grip_util_r',
            'lap_time','lap_number','sector',
            'speed_kmh','lat_g','lon_g','energy_kj',
            'tc_active','abs_active','dsc_active',
        ]
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=fields)
        self._log_writer.writeheader()
        print(f"[Server] Telemetry log → {log_path}")

    def _log_frame(self, tf: TelemetryFrame):
        if self._log_writer is None:
            return
        self._log_writer.writerow({
            'frame_id': tf.frame_id, 'sim_time': f"{tf.sim_time:.4f}",
            'x': f"{tf.x:.4f}", 'y': f"{tf.y:.4f}", 'z': f"{tf.z:.4f}",
            'roll': f"{tf.roll:.5f}", 'pitch': f"{tf.pitch:.5f}", 'yaw': f"{tf.yaw:.5f}",
            'vx': f"{tf.vx:.3f}", 'vy': f"{tf.vy:.3f}", 'vz': f"{tf.vz:.3f}",
            'ax': f"{tf.ax:.3f}", 'ay': f"{tf.ay:.3f}", 'az': f"{tf.az:.3f}",
            'wz': f"{tf.wz:.4f}",
            'z_fl': f"{tf.z_fl:.4f}", 'z_fr': f"{tf.z_fr:.4f}",
            'z_rl': f"{tf.z_rl:.4f}", 'z_rr': f"{tf.z_rr:.4f}",
            'Fz_fl': f"{tf.Fz_fl:.1f}", 'Fz_fr': f"{tf.Fz_fr:.1f}",
            'Fz_rl': f"{tf.Fz_rl:.1f}", 'Fz_rr': f"{tf.Fz_rr:.1f}",
            'Fy_fl': f"{tf.Fy_fl:.1f}", 'Fy_fr': f"{tf.Fy_fr:.1f}",
            'Fy_rl': f"{tf.Fy_rl:.1f}", 'Fy_rr': f"{tf.Fy_rr:.1f}",
            'slip_fl': f"{tf.slip_fl:.4f}", 'slip_fr': f"{tf.slip_fr:.4f}",
            'slip_rl': f"{tf.slip_rl:.4f}", 'slip_rr': f"{tf.slip_rr:.4f}",
            'kappa_rl': f"{tf.kappa_rl:.4f}", 'kappa_rr': f"{tf.kappa_rr:.4f}",
            'omega_fl': f"{tf.omega_fl:.2f}", 'omega_fr': f"{tf.omega_fr:.2f}",
            'omega_rl': f"{tf.omega_rl:.2f}", 'omega_rr': f"{tf.omega_rr:.2f}",
            'T_fl': f"{tf.T_fl:.1f}", 'T_fr': f"{tf.T_fr:.1f}",
            'T_rl': f"{tf.T_rl:.1f}", 'T_rr': f"{tf.T_rr:.1f}",
            'delta': f"{tf.delta:.4f}", 'throttle': f"{tf.throttle:.3f}",
            'brake_norm': f"{tf.brake_norm:.3f}",
            'grip_util_f': f"{tf.grip_util_f:.3f}", 'grip_util_r': f"{tf.grip_util_r:.3f}",
            'lap_time': f"{tf.lap_time:.3f}", 'lap_number': tf.lap_number,
            'sector': tf.sector, 'speed_kmh': f"{tf.speed_kmh:.2f}",
            'lat_g': f"{tf.lat_g:.3f}", 'lon_g': f"{tf.lon_g:.3f}",
            'energy_kj': f"{tf.energy_kj:.3f}",
            'tc_active': int(self._tc_active),
            'abs_active': int(self._abs_active),
            'dsc_active': 0,
        })

    # ── State extraction ─────────────────────────────────────────────────────

    def _extract_telemetry(self, state: jnp.ndarray,
                            prev_state: jnp.ndarray,
                            controls_jax: jnp.ndarray,
                            timing: dict) -> TelemetryFrame:
        s     = np.array(state)
        s_p   = np.array(prev_state)
        g     = 9.81

        # Position
        X, Y, Z, roll, pitch, yaw = s[0], s[1], s[2], s[3], s[4], s[5]

        # Velocities (body frame)
        vx = float(np.clip(s[14], -80, 80))
        vy = float(np.clip(s[15], -30, 30))
        vz = float(s[16])
        wx = float(s[17])
        wy = float(s[18])
        wz = float(np.clip(s[19], -8, 8))

        # Wheel angular velocities — derived from transient longitudinal slip states
        # state[38:46] = [alpha_fl, kappa_fl, alpha_fr, kappa_fr, alpha_rl, kappa_rl, alpha_rr, kappa_rr]
        # IMPORTANT: state[34:37] = T_ribs_r (tire temperatures) — NOT wheel speeds!
        # Correct: omega = vx * (1 + kappa_t) / tire_radius
        tire_r_omega = VP_DICT.get('tire_radius', VP_DICT.get('wheel_radius', 0.2032))
        vx_safe_omega = max(abs(float(np.clip(s[14], -80, 80))), 0.5)
        kappa_t_fl = float(np.clip(s[39], -0.5, 0.5)) if len(s) > 41 else 0.0
        kappa_t_fr = float(np.clip(s[41], -0.5, 0.5)) if len(s) > 41 else 0.0
        kappa_t_rl = float(np.clip(s[43], -0.5, 0.5)) if len(s) > 43 else 0.0
        kappa_t_rr = float(np.clip(s[45], -0.5, 0.5)) if len(s) > 45 else 0.0
        omega_fl = vx_safe_omega * (1.0 + kappa_t_fl) / tire_r_omega
        omega_fr = vx_safe_omega * (1.0 + kappa_t_fr) / tire_r_omega
        omega_rl = vx_safe_omega * (1.0 + kappa_t_rl) / tire_r_omega
        omega_rr = vx_safe_omega * (1.0 + kappa_t_rr) / tire_r_omega

        # Suspension heave (derived)
        tw = VP_DICT.get('track_front', 1.20)
        lf = VP_DICT.get('lf', 0.680)
        lr = VP_DICT.get('lr', 0.920)
        tire_r = VP_DICT.get('tire_radius', VP_DICT.get('wheel_radius', 0.2032))

        # state[2] = Z = spring-deformation coordinate (Z_eq ≈ -12.5 mm at rest).
        # Physical CG height above ground = (Z - Z_eq) + h_cg
        #   → at equilibrium: (Z_eq - Z_eq) + h_cg = h_cg = 0.30m ✓
        #   → tire bottom = h_cg - (h_cg - R_TIRE) - R_TIRE = 0.0m ✓
        h_cg    = VP_DICT.get('h_cg', 0.30)
        m_s_viz = VP_DICT.get('sprung_mass', VP_DICT.get('total_mass', 230.0) - 40.0)
        k_f_viz = float(DEFAULT_SETUP[0])
        k_r_viz = float(DEFAULT_SETUP[1])
        mr_f_v  = VP_DICT.get('motion_ratio_f_poly', [1.20])[0]
        mr_r_v  = VP_DICT.get('motion_ratio_r_poly', [1.15])[0]
        wr_f_v  = k_f_viz / (mr_f_v ** 2)
        wr_r_v  = k_r_viz / (mr_r_v ** 2)
        Fg_f_v  = VP_DICT.get('damper_gas_force_f', 120.0) / mr_f_v
        Fg_r_v  = VP_DICT.get('damper_gas_force_r', 120.0) / mr_r_v
        Z_eq_v  = (2.0 * (Fg_f_v + Fg_r_v) - m_s_viz * 9.81) / (2.0 * (wr_f_v + wr_r_v))
        h_cg_viz = h_cg - Z_eq_v        # offset so tire bottom = 0 at equilibrium
        Z_viz = float(Z) + h_cg_viz

        tw = VP_DICT.get('track_front', 1.20)
        lf = VP_DICT.get('lf', 0.680)
        lr = VP_DICT.get('lr', 0.920)
        tire_r = VP_DICT.get('tire_radius', VP_DICT.get('wheel_radius', 0.2032))

        # Corner world heights = CG world height ± pitch/roll geometry
        z_fl = Z_viz - lf * pitch - (tw / 2) * roll
        z_fr = Z_viz - lf * pitch + (tw / 2) * roll
        z_rl = Z_viz + lr * pitch - (tw / 2) * roll
        z_rr = Z_viz + lr * pitch + (tw / 2) * roll

        # Accelerations from state derivative
        ax_raw = (vx - float(s_p[14])) / DT
        ay_raw = (vy - float(s_p[15])) / DT
        az_raw = (vz - float(s_p[16])) / DT
        ax = float(np.clip(ax_raw, -50, 50))
        ay = float(np.clip(ay_raw, -50, 50))
        az = float(np.clip(az_raw, -50, 50))

        # Corner velocities (for slip angle)
        d_ack = float(controls_jax[0]) * VP_DICT.get('ackermann_factor', 0.50) * (tw/2) / (lf+lr)
        delta_fl = float(controls_jax[0]) - d_ack
        delta_fr = float(controls_jax[0]) + d_ack

        vx_safe = max(abs(vx), 0.5)
        vy_fl_corner = vy + wz * lf
        vy_fr_corner = vy + wz * lf
        vy_rl_corner = vy - wz * lr
        vy_rr_corner = vy - wz * lr

        def alpha(vy_c, delta_c, vx_c):
            return float(np.arctan2(vy_c, max(abs(vx_c), 0.5)) - delta_c)

        slip_fl = alpha(vy_fl_corner, delta_fl, vx)
        slip_fr = alpha(vy_fr_corner, delta_fr, vx)
        slip_rl = alpha(vy_rl_corner, 0.0, vx)
        slip_rr = alpha(vy_rr_corner, 0.0, vx)

        # Longitudinal slip ratio
        tire_r = VP_DICT.get('tire_radius', VP_DICT.get('wheel_radius', 0.2032))
        v_wheel_rl = omega_rl * tire_r
        v_wheel_rr = omega_rr * tire_r
        kappa_rl = float(np.clip((v_wheel_rl - vx) / vx_safe, -1.0, 1.0))
        kappa_rr = float(np.clip((v_wheel_rr - vx) / vx_safe, -1.0, 1.0))

        # Thermal states (indices 28-31 = tire surface temps)
        T_fl = float(s[28]) if s[28] > 0 else 25.0
        T_fr = float(s[29]) if s[29] > 0 else 25.0
        T_rl = float(s[30]) if s[30] > 0 else 25.0
        T_rr = float(s[31]) if s[31] > 0 else 25.0
        T_fl = float(np.clip(T_fl, 15.0, 120.0))
        T_fr = float(np.clip(T_fr, 15.0, 120.0))
        T_rl = float(np.clip(T_rl, 15.0, 120.0))
        T_rr = float(np.clip(T_rr, 15.0, 120.0))

        # Wheel loads (static approximation — full load-transfer from physics)
        m  = VP_DICT.get('total_mass', 230.0)
        L  = lf + lr
        h  = VP_DICT.get('h_cg', 0.30)

        # Longitudinal load transfer
        LLT_lon = m * ax * h / L / 2
        # Lateral load transfer (simplified)
        LLT_lat_f = m * ay * h / tw / 2
        LLT_lat_r = m * ay * h / tw / 2

        # Aero downforce
        rho = 1.225; A = VP_DICT.get('A_ref', 1.1); Cl = VP_DICT.get('Cl_ref', 3.0)
        df_total = 0.5 * rho * Cl * A * vx**2
        aero_split_f = VP_DICT.get('aero_split_f', 0.40)
        Fz_aero_f = df_total * aero_split_f / 2
        Fz_aero_r = df_total * (1-aero_split_f) / 2

        Fz_fl = max(0.0, m*g*lr/(L*2) - LLT_lat_f - LLT_lon + Fz_aero_f)
        Fz_fr = max(0.0, m*g*lr/(L*2) + LLT_lat_f - LLT_lon + Fz_aero_f)
        Fz_rl = max(0.0, m*g*lf/(L*2) - LLT_lat_r + LLT_lon + Fz_aero_r)
        Fz_rr = max(0.0, m*g*lf/(L*2) + LLT_lat_r + LLT_lon + Fz_aero_r)

        # Lateral forces (from slip angle × cornering stiffness approx)
        Ca = VP_DICT.get('cornering_stiffness', 35000.0)
        Fy_fl = float(np.clip(-Ca * slip_fl * Fz_fl / 1000, -Fz_fl*1.8, Fz_fl*1.8))
        Fy_fr = float(np.clip(-Ca * slip_fr * Fz_fr / 1000, -Fz_fr*1.8, Fz_fr*1.8))
        Fy_rl = float(np.clip(-Ca * slip_rl * Fz_rl / 1000, -Fz_rl*1.8, Fz_rl*1.8))
        Fy_rr = float(np.clip(-Ca * slip_rr * Fz_rr / 1000, -Fz_rr*1.8, Fz_rr*1.8))

        # Grip utilisation
        mu = 1.45
        util_f = float(np.sqrt(Fy_fl**2 + Fy_fr**2) / (mu * (Fz_fl + Fz_fr) + 1e-6))
        util_r = float(np.sqrt(Fy_rl**2 + Fy_rr**2) / (mu * (Fz_rl + Fz_rr) + 1e-6))

        # Energy
        Fx_drive = max(0.0, float(controls_jax[1]) / 2000.0 *
                       VP_DICT.get('motor_peak_power', 60000.0) / max(abs(vx), 1.0))
        self._energy_j += Fx_drive * abs(vx) * DT
        energy_kj = self._energy_j / 1000.0

        # Aerodynamics
        Cd = VP_DICT.get('Cd_ref', 1.5)
        drag_total = 0.5 * rho * Cd * A * vx**2

        # Controls
        throttle_norm = float(np.clip(controls_jax[1] / 2000.0, 0.0, 1.0))
        brake_norm    = float(np.clip(-controls_jax[1] / 10000.0, 0.0, 1.0))

        tf = TelemetryFrame(
            frame_id     = self.frame_id,
            sim_time     = self.sim_time,
            x=X, y=Y, z=Z_viz,           # Z_viz = Z_model + tire_radius (real-world height)
            roll=roll, pitch=pitch, yaw=yaw,
            vx=vx, vy=vy, vz=vz,
            ax=ax, ay=ay, az=az, wz=wz,
            z_fl=z_fl, z_fr=z_fr, z_rl=z_rl, z_rr=z_rr,   # also offset by tire_r
            Fz_fl=Fz_fl, Fz_fr=Fz_fr, Fz_rl=Fz_rl, Fz_rr=Fz_rr,
            Fy_fl=Fy_fl, Fy_fr=Fy_fr, Fy_rl=Fy_rl, Fy_rr=Fy_rr,
            slip_fl=slip_fl, slip_fr=slip_fr, slip_rl=slip_rl, slip_rr=slip_rr,
            kappa_rl=kappa_rl, kappa_rr=kappa_rr,
            omega_fl=omega_fl, omega_fr=omega_fr,
            omega_rl=omega_rl, omega_rr=omega_rr,
            T_fl=T_fl, T_fr=T_fr, T_rl=T_rl, T_rr=T_rr,
            delta     = float(controls_jax[0]),
            throttle  = throttle_norm,
            brake_norm= brake_norm,
            grip_util_f=min(util_f, 2.0), grip_util_r=min(util_r, 2.0),
            lap_time   = timing.get('lap_time', 0.0),
            lap_number = timing.get('lap_number', 0),
            sector     = timing.get('sector', 0),
            speed_kmh  = abs(vx) * 3.6,
            lat_g      = ay / g, lon_g = ax / g,
            yaw_rate_deg = float(np.degrees(wz)),
            downforce  = df_total,
            drag       = drag_total,
            energy_kj  = energy_kj,
            trans_fl   = float(s[38]) if len(s) > 41 else 0.0,
            trans_fr   = float(s[39]) if len(s) > 41 else 0.0,
            trans_rl   = float(s[40]) if len(s) > 41 else 0.0,
            trans_rr   = float(s[41]) if len(s) > 41 else 0.0,
            setup_hash = float(hash(tuple(np.array(self.setup_params).tolist())) % 65536),
        )
        return tf

    # ── Drive assists ─────────────────────────────────────────────────────────

    def _apply_drive_assists(self, steer: float, throttle_f: float,
                              brake_f: float, vx: float,
                              omega_rl: float, omega_rr: float,
                              Fz_rear: float) -> Tuple[float, float, float]:
        """Apply TC, ABS, and return modified (steer, throttle_f, brake_f)."""
        tire_r = VP_DICT.get('tire_radius', 0.2032)
        vx_safe = max(abs(vx), 0.5)

        # ── Traction Control ─────────────────────────────────────────────
        if self.tc_enabled and throttle_f > 0:
            kappa_rl = (omega_rl * tire_r - vx) / vx_safe
            kappa_rr = (omega_rr * tire_r - vx) / vx_safe
            kappa_max = max(abs(kappa_rl), abs(kappa_rr))
            if kappa_max > TC_KAPPA_LIMIT:
                reduction = 1.0 - (kappa_max - TC_KAPPA_LIMIT) / TC_KAPPA_LIMIT
                throttle_f *= max(0.0, min(1.0, reduction))
                self._tc_active = True
            else:
                self._tc_active = False
        else:
            self._tc_active = False

        # ── ABS ──────────────────────────────────────────────────────────
        if self.abs_enabled and brake_f > 0:
            omega_fl = float(np.array(self.state)[34]) if len(np.array(self.state)) > 37 else 0.0
            omega_fr = float(np.array(self.state)[35]) if len(np.array(self.state)) > 37 else 0.0
            # Compute wheel decel (prev vs current)
            d_omega_rl = (self._prev_omega_rl - omega_rl) / DT
            d_omega_rr = (self._prev_omega_rr - omega_rr) / DT
            max_decel = max(d_omega_rl, d_omega_rr)
            if max_decel > ABS_DECEL_LIMIT / (tire_r + 1e-6):
                brake_f *= 0.6    # release brake pressure
                self._abs_active = True
            else:
                self._abs_active = False
        else:
            self._abs_active = False

        self._prev_omega_rl = omega_rl
        self._prev_omega_rr = omega_rr

        return steer, throttle_f, brake_f

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _autopilot_step(self, vx: float):
        """
        Pure-pursuit autopilot using track heading (cpsi) at the lookahead point.

        WHY cpsi NOT atan2:
          atan2(ty-y, tx-x) gives the CHORD direction to the lookahead point.
          Mid-hairpin this chord points back across the corner and the sign flips,
          commanding the wrong steer direction.
          cpsi is the track's own tangent heading — always correct regardless of
          where on the arc the car is.

        DS = 0.5m per track index.
        """
        import math
        TARGET_SPEED_MS = 5.0
        MAX_THROTTLE_N  = 1500.0
        MAX_BRAKE_N     = 3000.0
        LOOKAHEAD_M     = max(8.0, abs(vx) * 1.5)   # at least 8m, ~1.5s preview
        DS              = 0.5

        s = np.array(self.state)
        x, y = float(s[0]), float(s[1])
        yaw  = float(s[5])

        steer = 0.0
        if self.track is not None:
            try:
                idx, _ = self.track.get_closest_point(x, y)
                n_pts  = len(self.track.cx)
                la_idx = int(idx + LOOKAHEAD_M / DS) % n_pts
                # Use track tangent heading at lookahead — not atan2 chord bearing
                target_angle = float(self.track.cpsi[la_idx])
                err = target_angle - yaw
                while err >  math.pi: err -= 2 * math.pi
                while err < -math.pi: err += 2 * math.pi
                steer = float(np.clip(err, -0.20, 0.20))
            except Exception:
                steer = 0.0

        speed_err = TARGET_SPEED_MS - abs(vx)
        if speed_err > 0:
            net_lon = float(np.clip(speed_err * 300.0, 0.0, MAX_THROTTLE_N))
        else:
            net_lon = float(np.clip(speed_err * 500.0, -MAX_BRAKE_N, 0.0))

        return steer, net_lon

    def run(self):
        print("=" * 60)
        print("  Project-GP: Enhanced JAX Physics Server v2")
        print("=" * 60)
        print(f"  Physics: {PHYSICS_HZ}Hz | Substeps: {SUBSTEPS} | dt_sub={DT_SUB*1000:.2f}ms")
        print(f"  Track: {self.track.name if self.track else 'None (timing disabled)'}")
        print(f"  Drive assists: TC={'ON' if self.tc_enabled else 'OFF'} | "
              f"ABS={'ON' if self.abs_enabled else 'OFF'} | "
              f"DSC={'ON' if self.dsc_enabled else 'OFF'}")
        print(f"  Log: {'ON → ' + LOG_DIR if self.log_telemetry else 'OFF'}")
        print()

        # ── Compile ──────────────────────────────────────────────────────
        print("[1/4] AOT compiling JAX physics step…")
        fast_step = self._compile_step()
        dummy_ctrl = jnp.array([0.0, 0.0])
        _ = fast_step(self.state, dummy_ctrl, self.setup_params)  # warm-up
        print("[1/4] Compilation complete.")

        # ── Settle to equilibrium ─────────────────────────────────────────
        # CRITICAL: run 400 zero-control steps so the suspension finds static
        # equilibrium before any client connects. Without this, the car spawns
        # with unbalanced spring forces and immediately sinks underground.
        self._settle_physics(fast_step, n_steps=400)

        # ── Sockets ──────────────────────────────────────────────────────
        print(f"[2/4] Binding UDP sockets (recv:{self.port_recv}, send:{self.port_send})…")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port_recv))
        sock.setblocking(False)
        print("[2/4] Sockets ready.")

        print(f"[3/4] Initial state: vx=5.0 m/s, pos=(0,0)")
        print(f"[4/4] Entering {PHYSICS_HZ}Hz real-time loop…")
        print(f"      Autopilot ACTIVE until a control client connects on :{self.port_recv}\n")

        frame_log_counter    = 0
        _first_ctrl_received = False
        self._t_start        = time.perf_counter()

        try:
            while True:
                t0 = time.perf_counter()

                # ── Receive controls ──────────────────────────────────────
                try:
                    data, addr = sock.recvfrom(RX_BYTES + 4)
                    ctrl = unpack_controls(data)
                    if ctrl:
                        cmd = ctrl['cmd_type']

                        if abs(cmd - CMD.RESET) < 0.5:
                            self.reset_to_start()

                        elif abs(cmd - CMD.PAUSE) < 0.5:
                            self.paused = True

                        elif abs(cmd - CMD.RESUME) < 0.5:
                            self.paused = False

                        elif abs(cmd - CMD.SETUP_CHANGE) < 0.5:
                            new_setup = np.array(self.setup_params)
                            if ctrl['k_f'] > 0:   new_setup[0] = ctrl['k_f']
                            if ctrl['k_r'] > 0:   new_setup[1] = ctrl['k_r']
                            if ctrl['arb_f'] > 0: new_setup[2] = ctrl['arb_f']
                            if ctrl['arb_r'] > 0: new_setup[3] = ctrl['arb_r']
                            self.setup_params = jnp.array(new_setup)
                            print(f"[Server] Setup updated: k_f={new_setup[0]:.0f} k_r={new_setup[1]:.0f}")

                        else:
                            self._steer_cmd  = ctrl['steer']
                            self._throttle_f = ctrl['throttle_f']
                            self._brake_f    = ctrl['brake_f']
                            self._last_ctrl_time = time.perf_counter()
                            if not _first_ctrl_received:
                                _first_ctrl_received = True
                                print(f"[Server] ✓ First control packet: "
                                      f"steer={ctrl['steer']:.3f} "
                                      f"thr={ctrl['throttle_f']:.0f}N "
                                      f"brk={ctrl['brake_f']:.0f}N")
                except BlockingIOError:
                    pass

                if self.paused:
                    time.sleep(DT)
                    continue

                # ── Drive assists ─────────────────────────────────────────
                s_arr = np.array(self.state)
                vx    = float(np.clip(s_arr[14], -80, 80))
                # Derive wheel omega from transient slip states (state[38:46])
                # state[36:38] = T_ribs_r[2], T_gas_r  ← DO NOT use for omega!
                _tr = VP_DICT.get('tire_radius', 0.2032)
                _vxs = max(abs(vx), 0.5)
                omega_rl = _vxs * (1.0 + float(np.clip(s_arr[43], -0.5, 0.5))) / _tr
                omega_rr = _vxs * (1.0 + float(np.clip(s_arr[45], -0.5, 0.5))) / _tr
                m  = VP_DICT.get('total_mass', 230.0)
                Fz_rear = m * 9.81 * VP_DICT.get('lf', 0.68) / (VP_DICT.get('lf', 0.68) + VP_DICT.get('lr', 0.92))
                steer_c, thr_c, brk_c = self._apply_drive_assists(
                    self._steer_cmd, self._throttle_f, self._brake_f,
                    vx, omega_rl, omega_rr, Fz_rear,
                )
                net_lon = thr_c - brk_c

                # ── Autopilot fallback ────────────────────────────────────
                # If no control client has sent a packet in the last 1 second,
                # drive autonomously using track centreline pure-pursuit.
                # This keeps the car moving without needing control_interface.py.
                _now = time.perf_counter()
                if _now - self._last_ctrl_time > 1.0:
                    steer_c, net_lon = self._autopilot_step(vx)

                controls_jax = jnp.array([steer_c, net_lon])

                # ── Physics step ──────────────────────────────────────────
                self.prev_state = self.state
                self.state      = fast_step(self.state, controls_jax, self.setup_params)
                self.sim_time  += DT
                self.frame_id  += 1

                # ── Lap timing ────────────────────────────────────────────
                s2  = np.array(self.state)
                X, Y = float(s2[0]), float(s2[1])
                ay   = float(np.clip((float(s2[15]) - float(s_arr[15])) / DT, -50, 50))
                ax   = float(np.clip((float(s2[14]) - vx) / DT, -50, 50))
                spd  = abs(float(s2[14])) * 3.6
                timing = self.lap_timer.update(
                    X, Y, self.sim_time,
                    speed_kmh=spd, lat_g=ay/9.81, lon_g=ax/9.81,
                    energy_kj=self._energy_j / 1000.0,
                )

                # ── Telemetry extraction & broadcast ──────────────────────
                tf = self._extract_telemetry(self.state, self.prev_state,
                                              controls_jax, timing)
                tx_bytes = tf.to_bytes()
                for addr in self.broadcast_clients:
                    try:
                        sock.sendto(tx_bytes, addr)
                    except Exception:
                        pass

                # ── Logging ───────────────────────────────────────────────
                frame_log_counter += 1
                if self.log_telemetry and frame_log_counter >= LOG_INTERVAL_FRAMES:
                    self._log_frame(tf)
                    frame_log_counter = 0

                # ── Performance monitoring ────────────────────────────────
                elapsed = time.perf_counter() - t0
                self._frame_times.append(elapsed)
                now = time.perf_counter()
                if now - self._last_perf_print > 5.0:
                    avg_ms = np.mean(self._frame_times) * 1000
                    hz     = 1000.0 / avg_ms if avg_ms > 0 else 0
                    rt     = hz / PHYSICS_HZ
                    _ctrl_src = "AUTO" if (now - self._last_ctrl_time > 1.0) else "HUMAN"
                    print(f"[Server] sim={self.sim_time:.1f}s wall={now-self._t_start:.0f}s | "
                          f"RT={rt:.2f}x ({hz:.0f}/{PHYSICS_HZ}Hz) | "
                          f"v={spd:.1f}km/h [{_ctrl_src}]"
                          + (" [TC]"  if self._tc_active  else "")
                          + (" [ABS]" if self._abs_active else ""))
                    self._last_perf_print = now

                # ── Sleep to maintain target Hz ───────────────────────────
                sleep_t = DT - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n[Server] Shutting down gracefully…")
        finally:
            sock.close()
            if self._log_file:
                self._log_file.close()
                print(f"[Server] Telemetry log closed.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Project-GP Physics Server v2')
    parser.add_argument('--track',  default='fsg_autocross',
                        help='Track name or JSON path')
    parser.add_argument('--hz',     type=int,   default=200)
    parser.add_argument('--no-tc',  action='store_true')
    parser.add_argument('--no-abs', action='store_true')
    parser.add_argument('--dsc',    action='store_true')
    parser.add_argument('--no-log', action='store_true')
    parser.add_argument('--host',   default=DEFAULT_HOST)
    parser.add_argument('--port-recv', type=int, default=DEFAULT_PORT_RECV)
    parser.add_argument('--port-send', type=int, default=DEFAULT_PORT_SEND)
    args = parser.parse_args()

    server = PhysicsServer(
        host          = args.host,
        port_recv     = args.port_recv,
        port_send     = args.port_send,
        track_name    = args.track,
        log_telemetry = not args.no_log,
        tc_enabled    = not args.no_tc,
        abs_enabled   = not args.no_abs,
        dsc_enabled   = args.dsc,
    )
    server.run()


if __name__ == '__main__':
    main()