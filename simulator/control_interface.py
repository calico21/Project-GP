"""
simulator/control_interface.py
─────────────────────────────────────────────────────────────────────────────
Human control interface for the physics server.

Modes:
  · Keyboard  — Arrow keys / WASD via termios raw-mode stdin (Wayland-safe)
  · Gamepad   — Xbox/PS controller via pygame.joystick (optional)
  · Autopilot — Path-following using track centreline + PD steering

Input is sent at 60Hz to the physics server via UDP.

Controls (keyboard):
  ↑ / W        — Throttle
  ↓ / S        — Brake
  ← / A        — Steer left
  → / D        — Steer right
  R            — Reset car to start
  P            — Pause / Resume
  T            — Toggle Traction Control
  B            — Toggle ABS
  +/-          — Adjust throttle sensitivity
  1-4          — Quick-select preset setups
  ESC / Q      — Quit

Steering model:
  · Exponential rate filter (tau=0.08s) prevents abrupt steer inputs
  · Steering angle ±0.35 rad at wheel (limited by rack)
  · Steering ramp: 0→max in ~0.3s
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
    from simulator.sim_protocol import (pack_controls, unpack_controls, CMD,
                                         TelemetryFrame,
                                         DEFAULT_HOST, DEFAULT_PORT_RECV,
                                         DEFAULT_PORT_SEND, DEFAULT_PORT_CTRL,
                                         TX_BYTES, TX)
except ImportError:
    from sim_protocol import (pack_controls, unpack_controls, CMD,
                               TelemetryFrame,
                               DEFAULT_HOST, DEFAULT_PORT_RECV,
                               DEFAULT_PORT_SEND, DEFAULT_PORT_CTRL,
                               TX_BYTES, TX)

# ─────────────────────────────────────────────────────────────────────────────
# Input parameters
# ─────────────────────────────────────────────────────────────────────────────
SEND_HZ          = 60
SEND_DT          = 1.0 / SEND_HZ
MAX_STEER_RAD    = 0.35      # maximum rack steering angle [rad]
MAX_THROTTLE_N   = 3000.0    # maximum throttle force [N] (motor torque × ratio)
MAX_BRAKE_N      = 8000.0    # maximum brake force [N] (disc clamp × pad)
STEER_RATE       = 2.5       # [rad/s] maximum steering rate
STEER_RETURN_RATE= 4.0       # [rad/s] centre-return rate when no key pressed
STEER_TAU        = 0.08      # [s] rate limiter time constant
THROTTLE_RAMP    = 0.25      # [s] throttle ramp time (0→1)
BRAKE_RAMP       = 0.15      # [s] brake ramp time (0→1)

# Preset setups (k_f, k_r, arb_f, arb_r)
PRESET_SETUPS = {
    '1': (30000, 32000, 800,  500,  'Soft/Understeery'),
    '2': (40000, 40000, 500,  500,  'Balanced (default)'),
    '3': (50000, 45000, 1200, 800,  'Stiff/Oversteery'),
    '4': (35000, 38000, 1000, 600,  'Optimiser Setup 1'),
}


# ─────────────────────────────────────────────────────────────────────────────

class ControlState:
    """Thread-safe control state."""

    def __init__(self):
        self._lock         = threading.Lock()
        self.steer_raw     = 0.0    # raw commanded steer [-1, 1]
        self.throttle_raw  = 0.0    # raw throttle [0, 1]
        self.brake_raw     = 0.0    # raw brake [0, 1]
        self.steer_smooth  = 0.0    # rate-limited steering angle [rad]
        self.throttle_N    = 0.0    # actual force [N]
        self.brake_N       = 0.0    # actual force [N]
        self.cmd_type      = CMD.DRIVE
        self.setup_payload = (0.0, 0.0, 0.0, 0.0)
        self.quit          = False
        self.sensitivity   = 1.0    # throttle sensitivity multiplier

    def set_steer(self, v):
        with self._lock: self.steer_raw  = float(max(-1, min(1, v)))

    def set_throttle(self, v):
        with self._lock: self.throttle_raw = float(max(0, min(1, v)))

    def set_brake(self, v):
        with self._lock: self.brake_raw = float(max(0, min(1, v)))

    def issue_reset(self):
        with self._lock: self.cmd_type = CMD.RESET

    def issue_pause(self):
        with self._lock: self.cmd_type = CMD.PAUSE

    def apply_setup(self, k_f, k_r, arb_f, arb_r):
        with self._lock:
            self.cmd_type      = CMD.SETUP_CHANGE
            self.setup_payload = (k_f, k_r, arb_f, arb_r)

    def get_snapshot(self):
        with self._lock:
            return (self.steer_raw, self.throttle_raw, self.brake_raw,
                    self.cmd_type, self.setup_payload, self.quit,
                    self.sensitivity)

    def clear_cmd(self):
        with self._lock: self.cmd_type = CMD.DRIVE


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard input (pynput — no display required)
# ─────────────────────────────────────────────────────────────────────────────

class KeyboardInput:
    """
    Terminal keyboard input using termios raw mode.

    Replaces pynput entirely — reads directly from stdin byte-by-byte.
    Works on Wayland, X11, SSH, and any terminal emulator.
    pynput silently fails on pure Wayland sessions without /dev/input access.

    Key map:
      W / ↑   = throttle        S / ↓   = brake
      A / ←   = steer left      D / →   = steer right
      R       = reset            P       = pause
      Q / ESC = quit             1-4     = preset setups
    """

    def __init__(self, ctrl: ControlState):
        self.ctrl     = ctrl
        self._keys    = set()      # currently held keys (string names)
        self._running = False
        self._thread  = None
        self._old_settings = None

        # Arrow-key escape sequences → friendly names
        self._ESC_MAP = {
            '\x1b[A': 'up',
            '\x1b[B': 'down',
            '\x1b[C': 'right',
            '\x1b[D': 'left',
        }

    def start(self) -> bool:
        import tty, termios, threading
        try:
            self._old_settings = termios.tcgetattr(sys.stdin.fileno())
        except termios.error:
            print("[Keyboard] stdin is not a tty — using autopilot mode instead.")
            return False

        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[Keyboard] Terminal raw-mode active (Wayland-safe).")
        print("[Keyboard] W/↑=throttle  S/↓=brake  A/←=left  D/→=right  R=reset  Q=quit")
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
                if ch == '\x1b':
                    # Read up to 2 more bytes for escape sequence (non-blocking)
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
                                # Arrow keys auto-release after a short hold
                                # (they repeat via the OS key-repeat mechanism)
                                threading.Timer(0.15, lambda n=name: self._keys.discard(n)).start()
                            continue
                    # Bare ESC = quit
                    self.ctrl.quit = True
                    continue

                c = ch.lower()
                self._keys.add(c)
                self._handle_oneshot(c)
                # Auto-release after short hold for held-key simulation
                threading.Timer(0.08, lambda k=c: self._keys.discard(k)).start()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)

    def _handle_oneshot(self, c: str):
        if c == 'r':
            self.ctrl.issue_reset()
            print("\n[Control] ↺ Reset")
        elif c == 'p':
            self.ctrl.issue_pause()
        elif c in ('q', '\x03'):   # q or Ctrl-C
            self.ctrl.quit = True
        elif c in PRESET_SETUPS:
            kf, kr, af, ar, name = PRESET_SETUPS[c]
            self.ctrl.apply_setup(kf, kr, af, ar)
            print(f"\n[Control] Setup {c}: {name}")

    def poll(self):
        keys = set(self._keys)

        # Steering
        s = 0.0
        if 'a' in keys or 'left'  in keys: s -= 1.0
        if 'd' in keys or 'right' in keys: s += 1.0
        self.ctrl.set_steer(s)

        # Throttle / brake
        if 'w' in keys or 'up' in keys:
            self.ctrl.set_throttle(self.ctrl.sensitivity)
            self.ctrl.set_brake(0.0)
        elif 's' in keys or 'down' in keys:
            self.ctrl.set_throttle(0.0)
            self.ctrl.set_brake(1.0)
        else:
            self.ctrl.set_throttle(0.0)
            self.ctrl.set_brake(0.0)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Gamepad input (pygame — optional)
# ─────────────────────────────────────────────────────────────────────────────

class GamepadInput:
    """
    Xbox/PS controller via pygame.joystick.
    Axis mapping: left-stick X = steer, RT = throttle, LT = brake.
    """
    STEER_AXIS   = 0
    THROTTLE_AXIS = 5   # RT on Xbox
    BRAKE_AXIS    = 4   # LT on Xbox

    def __init__(self, ctrl: ControlState, joystick_id: int = 0):
        self.ctrl = ctrl
        self._joy = None
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > joystick_id:
                self._joy = pygame.joystick.Joystick(joystick_id)
                self._joy.init()
                print(f"[Gamepad] Connected: {self._joy.get_name()}")
            else:
                print(f"[Gamepad] No joystick found at id={joystick_id}")
        except ImportError:
            print("[Gamepad] pygame not available. Install: pip install pygame")

    @property
    def available(self) -> bool:
        return self._joy is not None

    def poll(self):
        if not self._joy:
            return
        import pygame
        pygame.event.pump()

        # Steering (axis 0, -1=left, +1=right)
        steer = float(self._joy.get_axis(self.STEER_AXIS))
        # Dead zone
        if abs(steer) < 0.05: steer = 0.0
        # Cubic response for finer control at centre
        steer = steer ** 3 if abs(steer) > 0.1 else steer * 0.3
        self.ctrl.set_steer(steer)

        # Triggers (axis 4/5: rest at -1, full press at +1)
        throttle_raw = (float(self._joy.get_axis(self.THROTTLE_AXIS)) + 1.0) / 2.0
        brake_raw    = (float(self._joy.get_axis(self.BRAKE_AXIS))    + 1.0) / 2.0
        self.ctrl.set_throttle(throttle_raw)
        self.ctrl.set_brake(brake_raw)

        # Button shortcuts
        if self._joy.get_button(6):    # Start / Menu
            self.ctrl.issue_reset()
        if self._joy.get_button(7):    # Select / View
            self.ctrl.issue_pause()
        for i, preset_key in enumerate(['1', '2', '3', '4']):
            if self._joy.get_button(i):
                kf, kr, af, ar, name = PRESET_SETUPS[preset_key]
                self.ctrl.apply_setup(kf, kr, af, ar)


# ─────────────────────────────────────────────────────────────────────────────
# Autopilot (path-following PD controller)
# ─────────────────────────────────────────────────────────────────────────────

class AutopilotInput:
    """
    Pure-pursuit path follower using track centreline.
    Useful for automated testing and lap time benchmarking.
    """

    LOOKAHEAD_T   = 0.8    # [s] lookahead time
    MAX_LOOKAHEAD = 30.0   # [m] cap on lookahead distance
    MIN_LOOKAHEAD = 5.0    # [m] min lookahead
    KP_STEER      = 1.2    # proportional steer gain
    KD_STEER      = 0.15   # derivative steer gain
    TARGET_LAT_G  = 1.35   # target lateral G for speed control

    def __init__(self, ctrl: ControlState, track=None):
        self.ctrl   = ctrl
        self.track  = track
        self._prev_error = 0.0

    def set_track(self, track):
        self.track = track

    def poll(self, x: float, y: float, yaw: float, vx: float, lat_g: float):
        if self.track is None:
            self.ctrl.set_throttle(0.5)
            self.ctrl.set_steer(0.0)
            return

        # Closest track point
        idx, dist = self.track.get_closest_point(x, y)

        # Lookahead point
        la_dist = min(max(abs(vx) * self.LOOKAHEAD_T, self.MIN_LOOKAHEAD),
                      self.MAX_LOOKAHEAD)
        la_pts  = int(la_dist)
        la_idx  = min(idx + la_pts, len(self.track.cx) - 1)
        tx, ty  = self.track.cx[la_idx], self.track.cy[la_idx]

        # Heading error
        dx, dy = tx - x, ty - y
        target_angle = math.atan2(dy, dx)
        error = target_angle - yaw
        # Normalise to [-π, π]
        while error >  math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi

        d_error = (error - self._prev_error) / SEND_DT
        steer_cmd = self.KP_STEER * error + self.KD_STEER * d_error
        steer_cmd = max(-1.0, min(1.0, steer_cmd))
        self._prev_error = error
        self.ctrl.set_steer(steer_cmd)

        # Speed control: back off if at grip limit
        speed_factor = min(1.0, self.TARGET_LAT_G / (abs(lat_g) + 0.1))
        # Add a cornering brake if we're going too fast for the next corner
        k_ahead = abs(self.track.ck[la_idx])
        v_limit = math.sqrt(self.TARGET_LAT_G * 9.81 / max(k_ahead, 0.01))
        v_safe  = min(v_limit, 25.0)

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
# Main control loop
# ─────────────────────────────────────────────────────────────────────────────

class ControlInterface:
    """
    60Hz control loop: reads input → applies rate limiting → sends UDP.
    """

    def __init__(self, host: str = DEFAULT_HOST,
                 port_send: int = DEFAULT_PORT_RECV,    # send controls → server port 5000
                 port_recv: int = DEFAULT_PORT_CTRL,    # receive telemetry on port 5002 (NOT 5001 — that's the visualizer)
                 mode: str = 'keyboard'):
        self.host      = host
        self.port_send = port_send   # send controls to this port
        self.port_recv = port_recv   # receive telemetry from this port
        self.mode      = mode        # 'keyboard', 'gamepad', 'autopilot'

        self.ctrl  = ControlState()

        # Latest telemetry (for autopilot + HUD)
        self._latest_telem : Optional[TelemetryFrame] = None

    def _make_input_source(self):
        if self.mode == 'keyboard':
            src = KeyboardInput(self.ctrl)
            if not src.start():
                print("[Control] Keyboard input unavailable — using autopilot.")
                self.mode = 'autopilot'
                return AutopilotInput(self.ctrl)
            return src
        elif self.mode == 'gamepad':
            src = GamepadInput(self.ctrl)
            if not src.available:
                print("[Control] Gamepad unavailable — falling back to keyboard.")
                self.mode = 'keyboard'
                return self._make_input_source()
            return src
        elif self.mode == 'autopilot':
            return AutopilotInput(self.ctrl)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def run(self):
        print(f"[Control] Interface starting (mode={self.mode})")
        print(f"[Control] NOTE: Terminal will enter raw mode — typed keys won't echo.")
        src = self._make_input_source()

        # Sockets
        sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        telem_active = False
        try:
            sock_recv.bind(('0.0.0.0', self.port_recv))
            sock_recv.settimeout(0.002)
            telem_active = True
            print(f"[Control] Receiving telemetry on :{self.port_recv}")
        except OSError as e:
            print(f"[Control] Telemetry port {self.port_recv} busy ({e}).")
            print(f"[Control] Running controls-only mode (no HUD). "
                  f"Check physics_server broadcasts to port {self.port_recv}.")
            sock_recv.close()
            sock_recv = None

        print(f"[Control] Sending controls to {self.host}:{self.port_send}")
        print(f"[Control] Controls: W/S=throttle/brake | A/D=steer | "
              f"R=reset | P=pause | 1-4=setups | Q=quit")

        # Rate-limiter state
        steer_smooth   = 0.0
        throttle_prev  = 0.0
        brake_prev     = 0.0
        last_print     = time.perf_counter()

        try:
            while True:
                t0 = time.perf_counter()

                # ── Receive telemetry ─────────────────────────────────────
                telem = None
                if telem_active and sock_recv is not None:
                    try:
                        data, _ = sock_recv.recvfrom(TX_BYTES + 8)
                        telem = TelemetryFrame.from_bytes(data)
                        self._latest_telem = telem
                    except socket.timeout:
                        pass

                # ── Poll input ────────────────────────────────────────────
                if self.mode == 'autopilot' and telem is not None:
                    src.poll(telem.x, telem.y, telem.yaw, telem.vx, telem.lat_g)
                else:
                    src.poll()

                snap = self.ctrl.get_snapshot()
                steer_raw, thr_raw, brk_raw, cmd_type, setup, quit_flag, sens = snap

                if quit_flag:
                    break

                # ── Steering rate limiter ─────────────────────────────────
                steer_target = steer_raw * MAX_STEER_RAD
                max_step     = STEER_RATE * SEND_DT
                delta_s      = steer_target - steer_smooth
                delta_s      = max(-max_step, min(max_step, delta_s))
                # Centre return
                if abs(steer_raw) < 0.05:
                    return_step = STEER_RETURN_RATE * SEND_DT
                    steer_smooth *= max(0.0, 1.0 - return_step / (abs(steer_smooth) + 1e-6))
                else:
                    steer_smooth += delta_s

                # ── Throttle / brake ramp ─────────────────────────────────
                thr_target  = thr_raw * sens
                brk_target  = brk_raw
                thr_max_step = SEND_DT / THROTTLE_RAMP
                brk_max_step = SEND_DT / BRAKE_RAMP

                throttle_out = throttle_prev + max(-thr_max_step,
                               min(thr_max_step, thr_target - throttle_prev))
                brake_out    = brake_prev + max(-brk_max_step,
                               min(brk_max_step, brk_target - brake_prev))
                throttle_prev, brake_prev = throttle_out, brake_out

                thr_N = throttle_out * MAX_THROTTLE_N
                brk_N = brake_out    * MAX_BRAKE_N
                net_f = thr_N - brk_N

                # ── Pack & send ───────────────────────────────────────────
                tx = pack_controls(
                    steer     = steer_smooth,
                    throttle_f= thr_N,
                    brake_f   = brk_N,
                    cmd_type  = cmd_type,
                    k_f       = setup[0], k_r  = setup[1],
                    arb_f     = setup[2], arb_r = setup[3],
                )
                sock_send.sendto(tx, (self.host, self.port_send))
                self.ctrl.clear_cmd()

                # ── HUD console print ─────────────────────────────────────
                now = time.perf_counter()
                if telem and now - last_print > 0.5:
                    lat_g_bar = '█' * int(abs(telem.lat_g) * 5)
                    lon_g_bar = ('▲' if telem.lon_g > 0 else '▼') * int(abs(telem.lon_g) * 3)
                    print(f"\r  {telem.speed_kmh:5.1f} km/h | "
                          f"Lat {telem.lat_g:+.2f}G {lat_g_bar:<5} | "
                          f"Lon {telem.lon_g:+.2f}G | "
                          f"δ={math.degrees(telem.delta):+5.1f}° | "
                          f"Lap {telem.lap_number+1} {telem.lap_time:5.2f}s | "
                          f"T_fl={telem.T_fl:.0f}°C "
                          + (" [TC]" if telem.grip_util_r > 0.9 else "")
                          + (" [ABS]" if abs(telem.kappa_rl) > 0.2 else ""),
                          end='', flush=True)
                    last_print = now

                # ── Sleep ─────────────────────────────────────────────────
                elapsed = time.perf_counter() - t0
                sleep_t = SEND_DT - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\n[Control] Shutting down.")
        finally:
            if hasattr(src, 'stop'):
                src.stop()
            sock_send.close()
            if sock_recv is not None:
                sock_recv.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Project-GP Control Interface')
    parser.add_argument('--mode', default='keyboard',
                        choices=['keyboard', 'gamepad', 'autopilot'])
    parser.add_argument('--host', default=DEFAULT_HOST)
    args = parser.parse_args()
    ci = ControlInterface(host=args.host, mode=args.mode)
    ci.run()


if __name__ == '__main__':
    main()