"""
simulator/control_interface.py
─────────────────────────────────────────────────────────────────────────────
Project-GP Control Interface — v3

Human input → rate limiting → UDP to physics server.

v2 → v3 changes:
  · Imports ports/constants from sim_config.py (no inline magic numbers)
  · Preset setups use 28-dim canonical vectors from sim_config.PRESET_SETUPS
  · Setup hotkeys send k_f, k_r, arb_f, arb_r from the 28-dim vector
  · Autopilot uses VehicleConstants for mu_peak

Modes:
  · Keyboard  — Arrow keys / WASD via termios raw-mode stdin (Wayland-safe)
  · Gamepad   — Xbox/PS controller via pygame.joystick (optional)
  · Autopilot — Path-following using track centreline + PD steering

Input is sent at 60Hz to the physics server via UDP.
"""

import os
import sys
import time
import socket
import struct
import threading
import math
from typing import Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from simulator.sim_protocol import (
        pack_controls, unpack_controls, CMD,
        TelemetryFrame, TX_BYTES, TX,
    )
    from simulator.sim_config import (
        HOST, PORT_CTRL_RECV, PORT_TELEM_CTRL,
        PRESET_SETUPS, SETUP_PARAM_NAMES,
        VehicleConstants,
    )
except ImportError:
    from sim_protocol import (
        pack_controls, unpack_controls, CMD,
        TelemetryFrame, TX_BYTES, TX,
    )
    from sim_config import (
        HOST, PORT_CTRL_RECV, PORT_TELEM_CTRL,
        PRESET_SETUPS, SETUP_PARAM_NAMES,
        VehicleConstants,
    )

VC = VehicleConstants()

# ─────────────────────────────────────────────────────────────────────────────
# Input parameters
# ─────────────────────────────────────────────────────────────────────────────
SEND_HZ           = 60
SEND_DT           = 1.0 / SEND_HZ
MAX_STEER_RAD     = 0.35
MAX_THROTTLE_N    = 3000.0
MAX_BRAKE_N       = 8000.0
STEER_RATE        = 2.5       # rad/s max steering rate
STEER_RETURN_RATE = 4.0       # rad/s centre-return
STEER_TAU         = 0.08      # rate limiter time constant
THROTTLE_RAMP     = 0.25      # seconds 0→1
BRAKE_RAMP        = 0.15

# Preset key mapping: keyboard key → sim_config preset name
PRESET_KEY_MAP = {
    '1': "1_soft",
    '2': "2_balanced",
    '3': "3_stiff",
    '4': "4_optimised",
}

PRESET_LABELS = {
    '1': "Soft / Understeer",
    '2': "Balanced (default)",
    '3': "Stiff / Oversteer",
    '4': "MORL Optimised",
}


# ─────────────────────────────────────────────────────────────────────────────
# Control State (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

class ControlState:
    """Thread-safe control state accumulator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._steer    = 0.0
        self._throttle = 0.0
        self._brake    = 0.0
        self._cmd_type = CMD.DRIVE if hasattr(CMD, 'DRIVE') else 0.0
        self._setup_key: Optional[str] = None
        self._quit     = False
        self._sensitivity = 1.0

    def set_steer(self, v):
        with self._lock: self._steer = max(-1.0, min(1.0, v))
    def set_throttle(self, v):
        with self._lock: self._throttle = max(0.0, min(1.0, v))
    def set_brake(self, v):
        with self._lock: self._brake = max(0.0, min(1.0, v))
    def issue_reset(self):
        with self._lock: self._cmd_type = CMD.RESET if hasattr(CMD, 'RESET') else 1.0
    def issue_pause(self):
        with self._lock: self._cmd_type = CMD.PAUSE if hasattr(CMD, 'PAUSE') else 3.0
    def issue_quit(self):
        with self._lock: self._quit = True
    def apply_setup(self, key: str):
        with self._lock:
            self._setup_key = key
            self._cmd_type = CMD.SETUP_CHANGE if hasattr(CMD, 'SETUP_CHANGE') else 2.0
    def adjust_sensitivity(self, delta):
        with self._lock: self._sensitivity = max(0.2, min(2.0, self._sensitivity + delta))
    def clear_cmd(self):
        with self._lock:
            self._cmd_type = CMD.DRIVE if hasattr(CMD, 'DRIVE') else 0.0
            self._setup_key = None

    def get_snapshot(self):
        with self._lock:
            return (self._steer, self._throttle, self._brake,
                    self._cmd_type, self._setup_key, self._quit,
                    self._sensitivity)


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard Input (Wayland-safe termios raw mode)
# ─────────────────────────────────────────────────────────────────────────────

class KeyboardInput:
    """
    Raw-mode stdin reader. Works on Wayland, X11, SSH, any terminal.
    """

    _ESC_MAP = {
        '\x1b[A': 'up',
        '\x1b[B': 'down',
        '\x1b[C': 'right',
        '\x1b[D': 'left',
    }

    def __init__(self, ctrl: ControlState):
        self.ctrl = ctrl
        self._keys = set()
        self._running = False
        self._thread = None
        self._old_settings = None

    def start(self) -> bool:
        import tty, termios
        try:
            self._old_settings = termios.tcgetattr(sys.stdin.fileno())
        except termios.error:
            print("[Keyboard] stdin is not a tty — using autopilot.")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[Keyboard] Raw-mode active (Wayland-safe).")
        print("[Keyboard] W/↑=thr  S/↓=brk  A/←=left  D/→=right  R=reset  Q=quit  1-4=setups")
        return True

    def _read_loop(self):
        import tty, termios, select
        fd = sys.stdin.fileno()
        tty.setraw(fd)
        try:
            while self._running:
                r, _, _ = select.select([sys.stdin], [], [], 0.01)
                if not r:
                    continue
                ch = sys.stdin.read(1)

                # Escape sequences (arrow keys)
                if ch == '\x1b':
                    r2, _, _ = select.select([sys.stdin], [], [], 0.02)
                    if r2:
                        ch2 = sys.stdin.read(1)
                        r3, _, _ = select.select([sys.stdin], [], [], 0.01)
                        if r3 and ch2 == '[':
                            ch3 = sys.stdin.read(1)
                            seq = '\x1b[' + ch3
                            name = self._ESC_MAP.get(seq)
                            if name:
                                self._keys.add(name)
                                # Auto-clear after 100ms
                                threading.Timer(0.10, lambda n=name: self._keys.discard(n)).start()
                    else:
                        self.ctrl.issue_quit()  # bare ESC = quit
                    continue

                cl = ch.lower()
                if cl == 'q':
                    self.ctrl.issue_quit()
                elif cl == 'r':
                    self.ctrl.issue_reset()
                elif cl == 'p':
                    self.ctrl.issue_pause()
                elif cl in PRESET_KEY_MAP:
                    self.ctrl.apply_setup(cl)
                    print(f"\r[Setup] → {PRESET_LABELS.get(cl, cl)}           ", end='')
                elif cl == '+' or cl == '=':
                    self.ctrl.adjust_sensitivity(0.1)
                elif cl == '-':
                    self.ctrl.adjust_sensitivity(-0.1)
                else:
                    # Movement keys: add with auto-clear
                    key_map = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
                    name = key_map.get(cl)
                    if name:
                        self._keys.add(name)
                        threading.Timer(0.10, lambda n=name: self._keys.discard(n)).start()
        finally:
            if self._old_settings:
                import termios
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)

    def poll(self):
        keys = self._keys.copy()
        self.ctrl.set_steer(-1.0 if 'left' in keys else (1.0 if 'right' in keys else 0.0))
        self.ctrl.set_throttle(1.0 if 'up' in keys else 0.0)
        self.ctrl.set_brake(1.0 if 'down' in keys else 0.0)

    def stop(self):
        self._running = False
        if self._old_settings:
            import termios
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Gamepad Input (pygame)
# ─────────────────────────────────────────────────────────────────────────────

class GamepadInput:
    STEER_AXIS    = 0
    THROTTLE_AXIS = 4
    BRAKE_AXIS    = 5

    def __init__(self, ctrl: ControlState):
        self.ctrl = ctrl
        self._joy = None
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._joy = pygame.joystick.Joystick(0)
                self._joy.init()
                print(f"[Gamepad] {self._joy.get_name()}")
        except ImportError:
            print("[Gamepad] pygame not installed. pip install pygame")

    @property
    def available(self): return self._joy is not None

    def poll(self):
        if not self._joy: return
        import pygame
        pygame.event.pump()
        steer = float(self._joy.get_axis(self.STEER_AXIS))
        if abs(steer) < 0.05: steer = 0.0
        steer = steer ** 3 if abs(steer) > 0.1 else steer * 0.3
        self.ctrl.set_steer(steer)
        thr = (float(self._joy.get_axis(self.THROTTLE_AXIS)) + 1.0) / 2.0
        brk = (float(self._joy.get_axis(self.BRAKE_AXIS)) + 1.0) / 2.0
        self.ctrl.set_throttle(thr)
        self.ctrl.set_brake(brk)


# ─────────────────────────────────────────────────────────────────────────────
# Autopilot (pure pursuit)
# ─────────────────────────────────────────────────────────────────────────────

class AutopilotInput:
    LOOKAHEAD_T   = 0.8
    MAX_LOOKAHEAD = 30.0
    MIN_LOOKAHEAD = 5.0
    KP_STEER      = 1.2
    KD_STEER      = 0.15
    TARGET_LAT_G  = VC.mu_peak * 0.82   # 82% of peak grip

    def __init__(self, ctrl: ControlState, track=None):
        self.ctrl = ctrl
        self.track = track
        self._prev_error = 0.0

    def set_track(self, track): self.track = track

    def poll(self, x=0.0, y=0.0, yaw=0.0, vx=5.0, lat_g=0.0):
        if self.track is None:
            self.ctrl.set_throttle(0.5)
            self.ctrl.set_steer(0.0)
            return

        idx, dist = self.track.get_closest_point(x, y)
        la_dist = min(max(abs(vx) * self.LOOKAHEAD_T, self.MIN_LOOKAHEAD), self.MAX_LOOKAHEAD)
        la_idx = min(idx + int(la_dist), len(self.track.cx) - 1)

        # Use track tangent heading (cpsi), not chord direction
        target_psi = self.track.cpsi[la_idx]
        error = target_psi - yaw
        while error >  math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi

        d_error = (error - self._prev_error) / SEND_DT
        steer_cmd = self.KP_STEER * error + self.KD_STEER * d_error
        self._prev_error = error
        self.ctrl.set_steer(max(-1.0, min(1.0, steer_cmd)))

        k_ahead = abs(self.track.ck[la_idx])
        v_limit = math.sqrt(self.TARGET_LAT_G * 9.81 / max(k_ahead, 0.01))
        v_safe = min(v_limit, 25.0)

        if abs(vx) > v_safe * 1.1:
            self.ctrl.set_throttle(0.0)
            self.ctrl.set_brake(min(1.0, (abs(vx) - v_safe) / v_safe))
        elif abs(vx) < v_safe * 0.9:
            self.ctrl.set_throttle(min(1.0, (v_safe - abs(vx)) / v_safe * 2))
            self.ctrl.set_brake(0.0)
        else:
            self.ctrl.set_throttle(0.3)
            self.ctrl.set_brake(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main Control Loop
# ─────────────────────────────────────────────────────────────────────────────

class ControlInterface:
    """60 Hz control loop: input → rate limiting → UDP send."""

    def __init__(self, host=HOST, port_send=PORT_CTRL_RECV,
                 port_recv=PORT_TELEM_CTRL, mode='keyboard'):
        self.host      = host
        self.port_send = port_send
        self.port_recv = port_recv
        self.mode      = mode
        self.ctrl      = ControlState()
        self._latest_telem = None

    def _make_input_source(self):
        if self.mode == 'keyboard':
            src = KeyboardInput(self.ctrl)
            if not src.start():
                self.mode = 'autopilot'
                return AutopilotInput(self.ctrl)
            return src
        elif self.mode == 'gamepad':
            src = GamepadInput(self.ctrl)
            if not src.available:
                self.mode = 'keyboard'
                return self._make_input_source()
            return src
        elif self.mode == 'autopilot':
            return AutopilotInput(self.ctrl)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def run(self):
        print(f"[Control] Starting (mode={self.mode})")
        src = self._make_input_source()

        sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        telem_active = False
        try:
            sock_recv.bind(('0.0.0.0', self.port_recv))
            sock_recv.settimeout(0.002)
            telem_active = True
            print(f"[Control] Telemetry on :{self.port_recv}")
        except OSError:
            print(f"[Control] Telemetry port {self.port_recv} busy — HUD disabled.")
            sock_recv.close()
            sock_recv = None

        print(f"[Control] Sending to {self.host}:{self.port_send}")

        steer_smooth  = 0.0
        throttle_prev = 0.0
        brake_prev    = 0.0
        last_print    = time.perf_counter()

        try:
            while True:
                t0 = time.perf_counter()

                # ── Receive telemetry ─────────────────────────────────
                telem = None
                if telem_active and sock_recv:
                    try:
                        data, _ = sock_recv.recvfrom(TX_BYTES + 8)
                        telem = TelemetryFrame.from_bytes(data)
                        self._latest_telem = telem
                    except socket.timeout:
                        pass

                # ── Poll input ────────────────────────────────────────
                if self.mode == 'autopilot' and telem:
                    src.poll(telem.x, telem.y, telem.yaw, telem.vx, telem.lat_g)
                else:
                    src.poll()

                snap = self.ctrl.get_snapshot()
                steer_raw, thr_raw, brk_raw, cmd_type, setup_key, quit_flag, sens = snap

                if quit_flag:
                    break

                # ── Steering rate limiter ─────────────────────────────
                steer_target = steer_raw * MAX_STEER_RAD
                max_step = STEER_RATE * SEND_DT
                delta_s = max(-max_step, min(max_step, steer_target - steer_smooth))

                if abs(steer_raw) < 0.05:
                    return_step = STEER_RETURN_RATE * SEND_DT
                    steer_smooth *= max(0.0, 1.0 - return_step / (abs(steer_smooth) + 1e-6))
                else:
                    steer_smooth += delta_s

                # ── Throttle / brake ramp ─────────────────────────────
                thr_target = thr_raw * sens
                brk_target = brk_raw
                thr_step = SEND_DT / THROTTLE_RAMP
                brk_step = SEND_DT / BRAKE_RAMP
                throttle_out = throttle_prev + max(-thr_step, min(thr_step, thr_target - throttle_prev))
                brake_out = brake_prev + max(-brk_step, min(brk_step, brk_target - brake_prev))
                throttle_prev, brake_prev = throttle_out, brake_out

                thr_N = throttle_out * MAX_THROTTLE_N
                brk_N = brake_out * MAX_BRAKE_N

                # ── Handle setup change from 28-dim preset ────────────
                setup_k_f = setup_k_r = setup_arb_f = setup_arb_r = 0.0
                if setup_key and setup_key in PRESET_KEY_MAP:
                    preset_name = PRESET_KEY_MAP[setup_key]
                    if preset_name in PRESET_SETUPS:
                        vec = PRESET_SETUPS[preset_name]
                        # Extract the 4 params the protocol supports for hot-reload
                        setup_k_f   = float(vec[0])   # k_f
                        setup_k_r   = float(vec[1])   # k_r
                        setup_arb_f = float(vec[2])   # arb_f
                        setup_arb_r = float(vec[3])   # arb_r

                # ── Pack & send ───────────────────────────────────────
                tx = pack_controls(
                    steer      = steer_smooth,
                    throttle_f = thr_N,
                    brake_f    = brk_N,
                    cmd_type   = cmd_type,
                    k_f        = setup_k_f,
                    k_r        = setup_k_r,
                    arb_f      = setup_arb_f,
                    arb_r      = setup_arb_r,
                )
                sock_send.sendto(tx, (self.host, self.port_send))
                self.ctrl.clear_cmd()

                # ── HUD ───────────────────────────────────────────────
                now = time.perf_counter()
                if telem and now - last_print > 0.5:
                    lat_bar = '█' * int(abs(telem.lat_g) * 5)
                    print(f"\r  {telem.speed_kmh:5.1f} km/h | "
                          f"Lat {telem.lat_g:+.2f}G {lat_bar:<5} | "
                          f"δ={math.degrees(telem.delta):+5.1f}° | "
                          f"Lap {telem.lap_number+1} {telem.lap_time:5.2f}s",
                          end='', flush=True)
                    last_print = now

                # ── Sleep ─────────────────────────────────────────────
                sleep_t = SEND_DT - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n[Control] Shutting down.")
        finally:
            if hasattr(src, 'stop'):
                src.stop()
            sock_send.close()
            if sock_recv:
                sock_recv.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Project-GP Control Interface v3')
    parser.add_argument('--mode', default='keyboard',
                        choices=['keyboard', 'gamepad', 'autopilot'])
    parser.add_argument('--host', default=HOST)
    args = parser.parse_args()
    ci = ControlInterface(host=args.host, mode=args.mode)
    ci.run()


if __name__ == '__main__':
    main()