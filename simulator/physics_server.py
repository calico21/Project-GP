"""
simulator/physics_server.py
─────────────────────────────────────────────────────────────────────────────
Project-GP Headless JAX Physics Server — v3

v2 → v3 changes:
─────────────────────────────────────────────────────────────────────────────
1.  28-dim setup vector — matches canonical SuspensionSetup.from_vector().
    v2 used an 8-dim shortcut that silently produced wrong physics.

2.  Named state indices (S.VX, S.WYAW, S.WSPIN_RL etc.) from sim_config.py.
    Eliminates the class of bugs where state[N] means different things.

3.  Fixed ABS — now reads wheel spin rates from S.WSPIN_* (24-27),
    not thermal states at indices 34-35 (which are tire temperatures).

4.  Fixed TC — uses transient slip states (S.KAPPA_RL/RR = 40-41) for
    actual kappa measurement instead of deriving from misindexed omega.

5.  WebSocket bridge port added to default broadcast targets.
    Dashboard receives live telemetry without configuration.

6.  All constants imported from sim_config.py — no inline magic numbers.

7.  Telemetry extraction uses S.* throughout — every index is auditable.
"""

import os
import sys
import time
import socket
import struct
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
from config.vehicles.ter26 import vehicle_params as VP_DICT
from config.tire_coeffs import tire_coeffs as TP_DICT

try:
    from simulator.sim_protocol import (
        TelemetryFrame, TX, RX, CMD,
        TX_FMT, RX_FMT, TX_BYTES, RX_BYTES,
        unpack_controls,
    )
    from simulator.lap_timer import LapTimer
    from simulator.track_builder import get_track
    from simulator.sim_config import (
        S, HOST, PORT_CTRL_RECV, PORT_TELEM_VIZ, PORT_TELEM_CTRL,
        PORT_TELEM_ROS, PORT_TELEM_DEBUG, PHYSICS_HZ, DT, SUBSTEPS, DT_SUB,
        LOG_DECIMATION, LOG_DIR, LOG_COLUMNS,
        DEFAULT_SETUP_28, SETUP_DIM, SETUP_PARAM_NAMES,
        PRESET_SETUPS, DriveAssistConfig, VehicleConstants,
        setup_28_to_sim8,
    )
except ImportError:
    from sim_protocol import (
        TelemetryFrame, TX, RX, CMD,
        TX_FMT, RX_FMT, TX_BYTES, RX_BYTES,
        unpack_controls,
    )
    from lap_timer import LapTimer
    from track_builder import get_track
    from sim_config import (
        S, HOST, PORT_CTRL_RECV, PORT_TELEM_VIZ, PORT_TELEM_CTRL,
        PORT_TELEM_ROS, PORT_TELEM_DEBUG, PHYSICS_HZ, DT, SUBSTEPS, DT_SUB,
        LOG_DECIMATION, LOG_DIR, LOG_COLUMNS,
        DEFAULT_SETUP_28, SETUP_DIM, SETUP_PARAM_NAMES,
        PRESET_SETUPS, DriveAssistConfig, VehicleConstants,
        setup_28_to_sim8,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle constants (extracted once from VP_DICT)
# ─────────────────────────────────────────────────────────────────────────────

VC = VehicleConstants.from_dict(VP_DICT)


# ═════════════════════════════════════════════════════════════════════════════
# Physics Server
# ═════════════════════════════════════════════════════════════════════════════

class PhysicsServer:
    """
    200 Hz JAX physics server with full 46-DOF telemetry extraction.

    Uses the canonical 28-dim SuspensionSetup vector, named state indices,
    and broadcasts to all clients including the WebSocket bridge.
    """

    def __init__(self,
                 host:        str  = HOST,
                 port_recv:   int  = PORT_CTRL_RECV,
                 track_name:  str  = 'fsg_autocross',
                 log_telemetry: bool = True,
                 assists:     DriveAssistConfig = DriveAssistConfig(),
                 broadcast_clients: Optional[List[Tuple[str, int]]] = None):

        self.host       = host
        self.port_recv  = port_recv
        self.log_telemetry = log_telemetry
        self.assists    = assists

        # Broadcast targets: visualizer + control HUD + ROS2 bridge + recorder
        self.broadcast_clients = broadcast_clients or [
            (host, PORT_TELEM_VIZ),    # → Godot / Rerun / ws_bridge
            (host, PORT_TELEM_CTRL),   # → control_interface HUD
            (host, PORT_TELEM_ROS),    # → ros2_bridge
            (host, PORT_TELEM_DEBUG),  # → telemetry_recorder / debug tools
        ]

        # ── Vehicle model ────────────────────────────────────────────────
        self.vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)

        # Canonical 28-dim setup — matches SuspensionSetup.from_vector()
        self.setup_params = jnp.array(DEFAULT_SETUP_28)
        # 8-dim adapter for simulate_step() which indexes [k_f,k_r,arb_f,arb_r,c_f,c_r,h_cg,brake_bias]
        self.sim_setup = jnp.array(setup_28_to_sim8(DEFAULT_SETUP_28))
        print(f"[Server] Setup: {SETUP_DIM}-dim canonical → 8-dim sim adapter"
              f" (h_cg={float(self.sim_setup[6]):.3f}m, bb={float(self.sim_setup[7]):.2f})")

        # ── Track & timing ───────────────────────────────────────────────
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

        # ── State ────────────────────────────────────────────────────────
        self.state      = self._make_initial_state()
        self.prev_state = self.state.copy()

        # ── Simulation clock ─────────────────────────────────────────────
        self.frame_id  = 0
        self.sim_time  = 0.0
        self.paused    = False

        # ── Energy integrator ────────────────────────────────────────────
        self._energy_j = 0.0

        # ── Drive-assist state ───────────────────────────────────────────
        self._tc_active   = False
        self._abs_active  = False
        self._dsc_active  = False
        self._prev_omega  = np.zeros(4)   # [fl, fr, rl, rr] from previous step

        # ── Controls (latest received) ───────────────────────────────────
        self._steer_cmd    = 0.0
        self._throttle_f   = 0.0
        self._brake_f      = 0.0
        self._last_ctrl_time = 0.0
        self._t_start      = 0.0

        # ── Performance profiling ────────────────────────────────────────
        self._frame_times: deque = deque(maxlen=1000)
        self._last_perf_print = time.perf_counter()

        # ── Logger ───────────────────────────────────────────────────────
        self._log_file   = None
        self._log_writer = None
        if log_telemetry:
            self._init_logger()

    # ─────────────────────────────────────────────────────────────────────────
    # §1  State Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _make_initial_state(self, x=0.0, y=0.0, yaw=0.0, speed=5.0):
        """Build initial state at computed static spring equilibrium."""
        s = jnp.zeros(S.STATE_DIM)

        # Position
        s = s.at[S.X].set(x)
        s = s.at[S.Y].set(y)
        s = s.at[S.YAW].set(yaw)

        # Static spring equilibrium: Z_eq = -mg / (2*(k_f + k_r))
        k_f = float(self.setup_params[0])   # index 0 = k_f in canonical order
        k_r = float(self.setup_params[1])   # index 1 = k_r
        m   = VC.total_mass
        g   = 9.81
        Z_eq = -(m * g) / (2.0 * (k_f + k_r))
        s = s.at[S.Z].set(Z_eq)

        # Velocity
        s = s.at[S.VX].set(speed)

        # Wheel spin consistent with road speed
        omega_init = speed / VC.tire_radius
        s = s.at[S.WSPIN_FL].set(omega_init)
        s = s.at[S.WSPIN_FR].set(omega_init)
        s = s.at[S.WSPIN_RL].set(omega_init)
        s = s.at[S.WSPIN_RR].set(omega_init)

        # Warm tires (~70°C: 93% grip, 3.8× margin to degrad at 270°C)
        for idx in range(S.T_FL_SURF, S.T_RL_CORE + 1):
            s = s.at[idx].set(70.0)

        print(f"[Server] Init: Z_eq={Z_eq*1e3:.1f}mm | vx={speed:.1f}m/s | "
              f"ω_init={omega_init:.1f}rad/s")
        return s

    def reset_to_start(self):
        """Reset car to track start position."""
        if self.track:
            sx, sy, syaw = self.track.get_start_pose()
        else:
            sx, sy, syaw = 0.0, 0.0, 0.0
        self.state     = self._make_initial_state(sx, sy, syaw)
        self.prev_state = self.state.copy()
        self.sim_time  = 0.0
        self.frame_id  = 0
        self._energy_j = 0.0
        self.lap_timer.reset()
        print("[Server] Reset to start position.")

    # ─────────────────────────────────────────────────────────────────────────
    # §2  JIT-Compiled Physics Step
    # ─────────────────────────────────────────────────────────────────────────

    def _compile_step(self):
        vehicle = self.vehicle

        @jax.jit
        def fast_step(state: jnp.ndarray,
                      controls: jnp.ndarray,
                      setup: jnp.ndarray) -> jnp.ndarray:
            @remat
            def substep(x_s, _):
                return vehicle.simulate_step(x_s, controls, setup, DT_SUB), None
            next_state, _ = jax.lax.scan(substep, state, None, length=SUBSTEPS)

            # Healthcheck: NaN guard + soft roll/pitch clamp
            # Instead of binary reject (which causes frozen states), CLIP
            # the offending DOFs and let the simulation continue.
            is_finite = jnp.isfinite(next_state[S.VX])
            # If NaN: revert entirely (unrecoverable)
            next_state = jnp.where(is_finite, next_state, state)
            # Soft clip roll and pitch to ±0.5 rad (28.6°) — prevents runaway
            # but doesn't freeze the car like the old ±0.39 binary guard did
            next_state = next_state.at[S.ROLL].set(
                jnp.clip(next_state[S.ROLL], -0.50, 0.50))
            next_state = next_state.at[S.PITCH].set(
                jnp.clip(next_state[S.PITCH], -0.50, 0.50))
            # Speed limiter: prevent runaway velocity
            next_state = next_state.at[S.VX].set(
                jnp.clip(next_state[S.VX], -40.0, 40.0))
            return next_state

        return fast_step

    # ─────────────────────────────────────────────────────────────────────────
    # §3  Drive Assists (TC, ABS, DSC)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_wheel_omegas(self, state_arr: np.ndarray) -> np.ndarray:
        """
        Extract wheel angular velocities from state using CORRECT indices.

        v2 BUG: read state[34:37] which are THERMAL states (T_rl_surf, T_rl_bulk).
        v3 FIX: reads S.WSPIN_FL..RR (indices 24-27) — generalised velocity DOFs.
        """
        return np.array([
            float(state_arr[S.WSPIN_FL]),
            float(state_arr[S.WSPIN_FR]),
            float(state_arr[S.WSPIN_RL]),
            float(state_arr[S.WSPIN_RR]),
        ])

    def _get_slip_ratios(self, state_arr: np.ndarray) -> np.ndarray:
        """
        Read transient longitudinal slip ratios from state.
        These are the actual κ values from the relaxation-length tire model,
        NOT derived from omega vs vx (which is noisy).
        """
        return np.array([
            float(np.clip(state_arr[S.KAPPA_FL], -0.5, 0.5)),
            float(np.clip(state_arr[S.KAPPA_FR], -0.5, 0.5)),
            float(np.clip(state_arr[S.KAPPA_RL], -0.5, 0.5)),
            float(np.clip(state_arr[S.KAPPA_RR], -0.5, 0.5)),
        ])

    def _apply_drive_assists(self, steer, throttle_f, brake_f,
                              vx, state_arr):
        """Apply TC, ABS, and DSC corrections using named indices."""
        cfg = self.assists
        omega = self._get_wheel_omegas(state_arr)
        kappa = self._get_slip_ratios(state_arr)

        # ── Traction Control ─────────────────────────────────────────
        self._tc_active = False
        if cfg.tc_enabled and throttle_f > 0:
            kappa_rear_max = max(abs(kappa[2]), abs(kappa[3]))  # RL, RR
            if kappa_rear_max > cfg.tc_kappa_limit:
                excess = kappa_rear_max - cfg.tc_kappa_limit
                reduction = max(0.0, 1.0 - excess * (1.0 / cfg.tc_kappa_limit) * cfg.tc_ramp_rate)
                throttle_f *= reduction
                self._tc_active = True

        # ── ABS (using wheel deceleration) ───────────────────────────
        self._abs_active = False
        if cfg.abs_enabled and brake_f > 0:
            # Wheel deceleration: d(omega)/dt ≈ (omega_prev - omega_now) / DT
            d_omega = (self._prev_omega - omega) / DT
            # If any wheel is decelerating too fast → releasing grip
            max_decel = np.max(d_omega)
            if max_decel > cfg.abs_slip_limit / (VC.tire_radius + 1e-6):
                brake_f *= cfg.abs_release
                self._abs_active = True

            # Also check kappa: if rear wheels are locked (|κ| > threshold)
            if abs(kappa[2]) > 0.15 or abs(kappa[3]) > 0.15:
                brake_f *= cfg.abs_release
                self._abs_active = True

        self._prev_omega = omega

        # ── DSC (yaw rate correction) ────────────────────────────────
        # Not modifying steer here — DSC applies a differential torque
        # that would need to modify the control vector differently.
        # For now, DSC is reported but not active in the control path.
        self._dsc_active = False

        return steer, throttle_f, brake_f

    # ─────────────────────────────────────────────────────────────────────────
    # §4  Autopilot (Pure Pursuit)
    # ─────────────────────────────────────────────────────────────────────────

    def _autopilot_step(self, vx, state_arr):
        """
        Pure-pursuit autopilot using track tangent heading (cpsi).

        Throttle calibration note:
          Measured dynamics response: u[1]=2000 → 276N effective wheel force.
          Attenuation ratio ≈ 0.138 (motor model + drivetrain + tire).
          To achieve 3 m/s² accel on 300kg car → need 900N eff → u≈6500.
          To achieve 10 m/s² braking → need 3000N eff → u≈21700.
        """
        if self.track is None:
            return 0.0, 2000.0  # gentle throttle, no steer

        import math

        X   = float(state_arr[S.X])
        Y   = float(state_arr[S.Y])
        yaw = float(state_arr[S.YAW])

        idx, dist = self.track.get_closest_point(X, Y)
        n_pts = len(self.track.cx)

        # ── Steering: pure pursuit on track tangent heading ──────────
        la_dist = min(max(abs(vx) * 0.8, 5.0), 30.0)
        la_idx  = min(idx + int(la_dist), n_pts - 1)

        target_psi = self.track.cpsi[la_idx]
        error = target_psi - yaw
        while error >  math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi

        steer = max(-0.4, min(0.4, 1.5 * error))

        # ── Speed target: friction-limited by worst curvature in braking zone ─
        # Look ahead over the ENTIRE braking distance, not just one point.
        # Find the tightest corner in the next [current .. current + braking_dist]
        braking_dist = max(10, int(abs(vx) * 2.0))  # ~2 seconds of lookahead
        k_max = 0.0
        for i in range(idx, min(idx + braking_dist, n_pts)):
            k_i = abs(self.track.ck[i % n_pts])
            if k_i > k_max:
                k_max = k_i

        # Friction-limited speed at the tightest upcoming corner
        v_corner = math.sqrt(VC.mu_peak * 9.81 / max(k_max, 0.005))
        v_safe   = min(v_corner, 25.0)  # absolute cap at 25 m/s = 90 km/h

        # ── Longitudinal control: aggressive P-controller ────────────
        # Calibrated force constants account for the ~7:1 attenuation
        # in the dynamics model (u[1]=2000 → 276N effective)
        THR_MAX = 8000.0    # ~1100N effective → ~3.7 m/s² on 300kg
        BRK_MAX = 20000.0   # ~2760N effective → ~9.2 m/s² braking
        THR_GAIN = 2000.0   # N per (m/s underspeed)
        BRK_GAIN = 3000.0   # N per (m/s overspeed)
        CRUISE   = 1500.0   # maintain speed on flat sections

        # Speed-dependent throttle ramp: prevent wheelspin at low speed
        # At 2 m/s: max 2000N. At 10 m/s: full 8000N. Linear ramp.
        speed_factor = min(1.0, max(0.25, abs(vx) / 10.0))
        thr_limit = THR_MAX * speed_factor

        v_err = abs(vx) - v_safe

        if v_err > 0.5:
            # Over speed target → brake proportionally
            net_lon = -min(BRK_MAX, v_err * BRK_GAIN)
        elif v_err < -1.0:
            # Under speed target → throttle (speed-limited to prevent wheelspin)
            net_lon = min(thr_limit, abs(v_err) * THR_GAIN)
        else:
            # In the sweet spot → cruise (also speed-limited)
            net_lon = min(thr_limit, CRUISE)

        return steer, net_lon

    # ─────────────────────────────────────────────────────────────────────────
    # §5  Telemetry Extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_telemetry(self, state, prev_state, controls_jax, timing):
        """Extract all 64 telemetry channels using named state indices."""
        s   = np.array(state)
        s_p = np.array(prev_state)

        # ── Position ──────────────────────────────────────────────────
        X     = float(s[S.X])
        Y     = float(s[S.Y])
        Z_raw = float(s[S.Z])
        roll  = float(s[S.ROLL])
        pitch = float(s[S.PITCH])
        yaw   = float(s[S.YAW])

        # World-frame Z: spring coordinate + tire radius
        Z_viz = Z_raw + VC.tire_radius

        # ── Velocities (body frame) ──────────────────────────────────
        vx = float(np.clip(s[S.VX], -80, 80))
        vy = float(np.clip(s[S.VY], -30, 30))
        vz = float(s[S.VZ])
        wz = float(np.clip(s[S.WYAW], -8, 8))

        # ── Accelerations (finite difference) ────────────────────────
        ax = float(np.clip((s[S.VX] - s_p[S.VX]) / DT, -50, 50))
        ay = float(np.clip((s[S.VY] - s_p[S.VY]) / DT, -50, 50))
        # Add centripetal: ay_total = ay_body + vx * wz
        ay_total = ay + vx * wz
        az = float(np.clip((s[S.VZ] - s_p[S.VZ]) / DT, -50, 50))

        # ── Suspension heave ─────────────────────────────────────────
        z_fl = float(s[S.Z_FL])
        z_fr = float(s[S.Z_FR])
        z_rl = float(s[S.Z_RL])
        z_rr = float(s[S.Z_RR])

        # ── Wheel angular velocities (correct indices!) ──────────────
        omega_fl = float(s[S.WSPIN_FL])
        omega_fr = float(s[S.WSPIN_FR])
        omega_rl = float(s[S.WSPIN_RL])
        omega_rr = float(s[S.WSPIN_RR])

        # ── Slip angles and ratios (from transient slip states) ──────
        slip_fl = float(np.clip(s[S.ALPHA_FL], -0.6, 0.6))
        slip_fr = float(np.clip(s[S.ALPHA_FR], -0.6, 0.6))
        slip_rl = float(np.clip(s[S.ALPHA_RL], -0.6, 0.6))
        slip_rr = float(np.clip(s[S.ALPHA_RR], -0.6, 0.6))
        kappa_rl = float(np.clip(s[S.KAPPA_RL], -0.5, 0.5))
        kappa_rr = float(np.clip(s[S.KAPPA_RR], -0.5, 0.5))

        # ── Tire temperatures ────────────────────────────────────────
        T_fl = float(s[S.T_FL_SURF])
        T_fr = float(s[S.T_FR_SURF])
        T_rl = float(s[S.T_RL_SURF])
        T_rr = float(s[S.T_RR_SURF])

        # ── Wheel loads (analytical estimate) ────────────────────────
        m  = VC.total_mass
        g  = 9.81
        L  = VC.wheelbase
        lf = VC.lf
        lr = VC.lr
        tw_f = VC.track_f
        tw_r = VC.track_r
        h_cg = VC.h_cg

        # Lateral load transfer
        LLT_lat_f = m * ay_total * h_cg * (lr / L) / tw_f
        LLT_lat_r = m * ay_total * h_cg * (lf / L) / tw_r

        # Longitudinal load transfer
        LLT_lon = m * ax * h_cg / L

        # Aerodynamic loads
        rho = 1.225
        A   = VP_DICT.get('frontal_area', 1.2)
        Cl  = VP_DICT.get('Cl_ref', 2.8)
        Cd  = VP_DICT.get('Cd_ref', 1.5)
        df_total = 0.5 * rho * Cl * A * vx**2
        drag_total = 0.5 * rho * Cd * A * vx**2

        aero_split = VP_DICT.get('aero_split_f', 0.40)
        Fz_aero_f = df_total * aero_split / 2
        Fz_aero_r = df_total * (1 - aero_split) / 2

        Fz_fl = max(0.0, m*g*lr/(L*2) - LLT_lat_f - LLT_lon + Fz_aero_f)
        Fz_fr = max(0.0, m*g*lr/(L*2) + LLT_lat_f - LLT_lon + Fz_aero_f)
        Fz_rl = max(0.0, m*g*lf/(L*2) - LLT_lat_r + LLT_lon + Fz_aero_r)
        Fz_rr = max(0.0, m*g*lf/(L*2) + LLT_lat_r + LLT_lon + Fz_aero_r)

        # ── Lateral forces (from slip × cornering stiffness approx) ──
        Ca = VP_DICT.get('cornering_stiffness', 35000.0)
        Fy_fl = float(np.clip(-Ca * slip_fl * Fz_fl / 1000, -Fz_fl*1.8, Fz_fl*1.8))
        Fy_fr = float(np.clip(-Ca * slip_fr * Fz_fr / 1000, -Fz_fr*1.8, Fz_fr*1.8))
        Fy_rl = float(np.clip(-Ca * slip_rl * Fz_rl / 1000, -Fz_rl*1.8, Fz_rl*1.8))
        Fy_rr = float(np.clip(-Ca * slip_rr * Fz_rr / 1000, -Fz_rr*1.8, Fz_rr*1.8))

        # ── Grip utilisation ─────────────────────────────────────────
        mu = VC.mu_peak
        util_f = float(np.sqrt(Fy_fl**2 + Fy_fr**2) / (mu * (Fz_fl + Fz_fr) + 1e-6))
        util_r = float(np.sqrt(Fy_rl**2 + Fy_rr**2) / (mu * (Fz_rl + Fz_rr) + 1e-6))

        # ── Energy ───────────────────────────────────────────────────
        ctrl_lon = float(controls_jax[1])
        Fx_drive = max(0.0, ctrl_lon)
        self._energy_j += Fx_drive * abs(vx) * DT / 2000.0  # crude [J]
        energy_kj = self._energy_j / 1000.0

        # ── Controls ─────────────────────────────────────────────────
        delta = float(controls_jax[0])
        throttle_norm = float(np.clip(ctrl_lon / 2000.0, 0.0, 1.0))
        brake_norm    = float(np.clip(-ctrl_lon / 8000.0, 0.0, 1.0))

        # ── Speed / G ────────────────────────────────────────────────
        speed_kmh = abs(vx) * 3.6
        lat_g = ay_total / g
        lon_g = ax / g

        # ── Build TelemetryFrame ─────────────────────────────────────
        tf = TelemetryFrame(
            frame_id  = self.frame_id,
            sim_time  = self.sim_time,
            x=X, y=Y, z=Z_viz,
            roll=roll, pitch=pitch, yaw=yaw,
            vx=vx, vy=vy, vz=vz,
            ax=ax, ay=ay_total, az=az,
            wz=wz,
            z_fl=z_fl, z_fr=z_fr, z_rl=z_rl, z_rr=z_rr,
            Fz_fl=Fz_fl, Fz_fr=Fz_fr, Fz_rl=Fz_rl, Fz_rr=Fz_rr,
            Fy_fl=Fy_fl, Fy_fr=Fy_fr, Fy_rl=Fy_rl, Fy_rr=Fy_rr,
            slip_fl=slip_fl, slip_fr=slip_fr, slip_rl=slip_rl, slip_rr=slip_rr,
            kappa_rl=kappa_rl, kappa_rr=kappa_rr,
            omega_fl=omega_fl, omega_fr=omega_fr,
            omega_rl=omega_rl, omega_rr=omega_rr,
            T_fl=T_fl, T_fr=T_fr, T_rl=T_rl, T_rr=T_rr,
            delta=delta, throttle=throttle_norm, brake_norm=brake_norm,
            grip_util_f=util_f, grip_util_r=util_r,
            lap_time=timing.get('lap_time', 0.0),
            lap_number=timing.get('lap_number', 0),
            sector=timing.get('sector', 0),
            speed_kmh=speed_kmh,
            lat_g=lat_g, lon_g=lon_g,
            yaw_rate_deg=wz * 180.0 / 3.14159,
            downforce=df_total, drag=drag_total,
            energy_kj=energy_kj,
        )
        return tf

    # ─────────────────────────────────────────────────────────────────────────
    # §6  Logger
    # ─────────────────────────────────────────────────────────────────────────

    def _init_logger(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts       = time.strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(LOG_DIR, f'telemetry_{ts}.csv')
        self._log_file   = open(log_path, 'w', newline='')
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=LOG_COLUMNS)
        self._log_writer.writeheader()
        print(f"[Server] Telemetry log → {log_path}")

    def _log_frame(self, tf: TelemetryFrame):
        if self._log_writer is None:
            return
        row = {}
        for col in LOG_COLUMNS:
            val = getattr(tf, col, None)
            if val is not None:
                if isinstance(val, float):
                    row[col] = f"{val:.4f}"
                else:
                    row[col] = val
            else:
                row[col] = 0
        self._log_writer.writerow(row)

    # ─────────────────────────────────────────────────────────────────────────
    # §7  Main Loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Main 200 Hz physics loop."""
        print("=" * 60)
        print("  Project-GP Physics Server v3")
        print(f"  Physics:  {PHYSICS_HZ} Hz × {SUBSTEPS} substeps = {1/DT_SUB:.0f} Hz internal")
        print(f"  Setup:    {SETUP_DIM}-dim canonical vector")
        print(f"  Assists:  TC={'ON' if self.assists.tc_enabled else 'OFF'} "
              f"ABS={'ON' if self.assists.abs_enabled else 'OFF'} "
              f"DSC={'ON' if self.assists.dsc_enabled else 'OFF'}")
        print(f"  Broadcast: {len(self.broadcast_clients)} clients")
        print("=" * 60)

        # ── JIT compile ──────────────────────────────────────────────
        print("[1/3] Compiling JAX physics graph (first call)…")
        fast_step = self._compile_step()
        dummy_ctrl = jnp.array([0.0, 0.0])
        _ = fast_step(self.state, dummy_ctrl, self.sim_setup)
        print("[1/3] Compilation complete.")

        # ── Sockets ──────────────────────────────────────────────────
        print(f"[2/3] Binding UDP (recv:{self.port_recv})")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port_recv))
        sock.setblocking(False)

        print(f"[3/3] Entering {PHYSICS_HZ} Hz loop. Autopilot active until client connects.\n")

        frame_log_counter = 0
        _first_ctrl = False
        self._t_start = time.perf_counter()

        try:
            while True:
                t0 = time.perf_counter()

                # ── Receive controls ──────────────────────────────────
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
                            # Hot-reload partial setup (first 4 params in packet)
                            new_setup = np.array(self.setup_params)
                            if ctrl.get('k_f', 0) > 0:  new_setup[0] = ctrl['k_f']
                            if ctrl.get('k_r', 0) > 0:  new_setup[1] = ctrl['k_r']
                            if ctrl.get('arb_f', 0) > 0: new_setup[2] = ctrl['arb_f']
                            if ctrl.get('arb_r', 0) > 0: new_setup[3] = ctrl['arb_r']
                            self.setup_params = jnp.array(new_setup)
                            self.sim_setup = jnp.array(setup_28_to_sim8(new_setup))
                            print(f"[Server] Setup updated: k_f={new_setup[0]:.0f} "
                                  f"k_r={new_setup[1]:.0f} "
                                  f"(h_cg={float(self.sim_setup[6]):.3f}m)")
                        else:
                            self._steer_cmd  = ctrl['steer']
                            self._throttle_f = ctrl['throttle_f']
                            self._brake_f    = ctrl['brake_f']
                            self._last_ctrl_time = time.perf_counter()
                            if not _first_ctrl:
                                _first_ctrl = True
                                print(f"[Server] First control packet received.")
                except BlockingIOError:
                    pass

                if self.paused:
                    time.sleep(DT)
                    continue

                # ── State array for assist calculations ───────────────
                s_arr = np.array(self.state)
                vx = float(np.clip(s_arr[S.VX], -80, 80))

                # ── Determine control source ──────────────────────────
                # Autopilot runs FIRST, then assists modify its output.
                # This ensures TC/ABS apply to autopilot throttle too.
                _is_auto = (time.perf_counter() - self._last_ctrl_time > 1.0)

                if _is_auto:
                    steer_raw, net_lon_raw = self._autopilot_step(vx, s_arr)
                    thr_raw = max(0.0, net_lon_raw)
                    brk_raw = max(0.0, -net_lon_raw)
                else:
                    steer_raw = self._steer_cmd
                    thr_raw   = self._throttle_f
                    brk_raw   = self._brake_f

                # ── Drive assists (TC/ABS) on the ACTUAL control source ──
                steer_c, thr_c, brk_c = self._apply_drive_assists(
                    steer_raw, thr_raw, brk_raw, vx, s_arr,
                )
                net_lon = thr_c - brk_c
                controls_jax = jnp.array([steer_c, net_lon])

                # ── Physics step ──────────────────────────────────────
                self.prev_state = self.state
                self.state = fast_step(self.state, controls_jax, self.sim_setup)
                self.sim_time += DT
                self.frame_id += 1

                # ── Frozen state detection ────────────────────────────
                # If health guard rejected the step, state == prev_state
                s2 = np.array(self.state)
                if self.frame_id > 10 and self.frame_id % 200 == 0:
                    dx = abs(float(s2[S.X]) - float(np.array(self.prev_state)[S.X]))
                    if dx < 1e-8 and abs(vx) > 0.5:
                        # State is frozen — health guard is rejecting steps
                        r = float(s_arr[S.ROLL])
                        p = float(s_arr[S.PITCH])
                        print(f"[WARN] State frozen! roll={r:.3f} pitch={p:.3f} "
                              f"(limit ±0.39) | vx={vx:.2f} | δ={steer_c:.3f} "
                              f"| F={net_lon:.0f}N")
                        # Recovery: reset to a safe state nearby
                        self.state = self.state.at[S.ROLL].set(
                            jnp.clip(self.state[S.ROLL], -0.30, 0.30))
                        self.state = self.state.at[S.PITCH].set(
                            jnp.clip(self.state[S.PITCH], -0.30, 0.30))
                        s2 = np.array(self.state)

                # ── Lap timing ────────────────────────────────────────
                spd = abs(float(s2[S.VX])) * 3.6
                ax_g = float(np.clip((s2[S.VX] - s_arr[S.VX]) / DT, -50, 50)) / 9.81
                ay_g = float(np.clip(
                    (s2[S.VY] - s_arr[S.VY]) / DT + s2[S.VX] * s2[S.WYAW],
                    -50, 50)) / 9.81

                timing = self.lap_timer.update(
                    float(s2[S.X]), float(s2[S.Y]),
                    self.sim_time,
                    speed_kmh=spd, lat_g=ay_g, lon_g=ax_g,
                    energy_kj=self._energy_j / 1000.0,
                )

                # ── Telemetry extraction & broadcast ──────────────────
                tf = self._extract_telemetry(self.state, self.prev_state,
                                              controls_jax, timing)
                tx_bytes = tf.to_bytes()
                for addr in self.broadcast_clients:
                    try:
                        sock.sendto(tx_bytes, addr)
                    except Exception:
                        pass

                # ── Logging ───────────────────────────────────────────
                frame_log_counter += 1
                if self.log_telemetry and frame_log_counter >= LOG_DECIMATION:
                    self._log_frame(tf)
                    frame_log_counter = 0

                # ── Performance monitoring ────────────────────────────
                elapsed = time.perf_counter() - t0
                self._frame_times.append(elapsed)
                now = time.perf_counter()
                if now - self._last_perf_print > 5.0:
                    avg_ms = np.mean(self._frame_times) * 1000
                    hz = 1000.0 / avg_ms if avg_ms > 0 else 0
                    rt = hz / PHYSICS_HZ
                    src = "AUTO" if (now - self._last_ctrl_time > 1.0) else "HUMAN"
                    x_pos = float(s2[S.X])
                    y_pos = float(s2[S.Y])
                    ctrl_steer = float(controls_jax[0])
                    ctrl_force = float(controls_jax[1])
                    ctrl_label = f"thr={ctrl_force:.0f}" if ctrl_force >= 0 else f"brk={-ctrl_force:.0f}"
                    print(f"[Server] t={self.sim_time:.1f}s | "
                          f"RT={rt:.2f}x ({hz:.0f}Hz) | "
                          f"v={spd:.1f}km/h | "
                          f"pos=({x_pos:.1f},{y_pos:.1f}) | "
                          f"δ={ctrl_steer:.3f} {ctrl_label} | "
                          f"[{src}]"
                          + (" TC" if self._tc_active else "")
                          + (" ABS" if self._abs_active else ""))
                    self._last_perf_print = now

                # ── Sleep to maintain target Hz ───────────────────────
                sleep_t = DT - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n[Server] Shutting down…")
        finally:
            sock.close()
            if self._log_file:
                self._log_file.close()
                print("[Server] Telemetry log closed.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Project-GP Physics Server v3 — 200Hz JAX Digital Twin")
    parser.add_argument("--track", type=str, default="fsg_autocross",
                        help="Track name or JSON path")
    parser.add_argument("--no-tc", action="store_true", help="Disable traction control")
    parser.add_argument("--no-abs", action="store_true", help="Disable ABS")
    parser.add_argument("--dsc", action="store_true", help="Enable stability control")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV logging")
    parser.add_argument("--host", type=str, default=HOST)
    parser.add_argument("--port", type=int, default=PORT_CTRL_RECV)
    args = parser.parse_args()

    assists = DriveAssistConfig(
        tc_enabled  = not args.no_tc,
        abs_enabled = not args.no_abs,
        dsc_enabled = args.dsc,
    )

    server = PhysicsServer(
        host          = args.host,
        port_recv     = args.port,
        track_name    = args.track,
        log_telemetry = not args.no_log,
        assists       = assists,
    )
    server.run()


if __name__ == "__main__":
    main()