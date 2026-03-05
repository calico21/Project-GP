#!/usr/bin/env python3
"""
simulator/debug_sim.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive diagnostic for the Project-GP simulator pipeline.

Tests every link in the chain and reports exactly what is broken.

Usage (run each test independently):

  # Test 1 — is the keyboard layer working at all?
  python simulator/debug_sim.py keyboard

  # Test 2 — are control packets reaching the server?
  # (run this in one terminal, run physics_server in another)
  python simulator/debug_sim.py sniff

  # Test 3 — does the physics integrator respond to force inputs?
  python simulator/debug_sim.py physics

  # Test 4 — end-to-end: inject throttle directly and watch speed
  # (run while physics_server.py is running)
  python simulator/debug_sim.py inject

  # Run all tests that don't need a running server
  python simulator/debug_sim.py all
"""

import sys
import os
import socket
import struct
import time
import threading
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from simulator.sim_protocol import (
        pack_controls, unpack_controls, TelemetryFrame,
        CMD, TX_BYTES, RX_BYTES, TX_FMT, RX_FMT,
        DEFAULT_HOST, DEFAULT_PORT_RECV, DEFAULT_PORT_SEND
    )
except ImportError:
    from sim_protocol import (
        pack_controls, unpack_controls, TelemetryFrame,
        CMD, TX_BYTES, RX_BYTES, TX_FMT, RX_FMT,
        DEFAULT_HOST, DEFAULT_PORT_RECV, DEFAULT_PORT_SEND
    )

SEP  = "─" * 60
PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
INFO = "  ℹ️  INFO"
WARN = "  ⚠️  WARN"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Keyboard input layer
# ─────────────────────────────────────────────────────────────────────────────

def test_keyboard():
    """
    Interactive keyboard test.
    Reads raw terminal bytes and shows exactly what each keystroke produces.
    This isolates whether termios raw mode works in this terminal session.
    """
    import tty, termios, select

    print(SEP)
    print("TEST 1: Keyboard raw-mode input")
    print(SEP)
    print("Press keys one at a time. Check that each key shows the right label.")
    print("Expected: W=throttle  S=brake  A=left  D=right  ↑↓←→=same")
    print("Press ESC twice to finish this test.\n")

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        print(FAIL, "stdin is not a tty. Are you running in a terminal?")
        print("       Try:  python simulator/debug_sim.py keyboard")
        return False

    ESC_MAP = {
        '\x1b[A': ('up',    'THROTTLE ↑'),
        '\x1b[B': ('down',  'BRAKE    ↓'),
        '\x1b[C': ('right', 'STEER R  →'),
        '\x1b[D': ('left',  'STEER L  ←'),
    }
    CHAR_MAP = {
        'w': 'THROTTLE',
        's': 'BRAKE',
        'a': 'STEER LEFT',
        'd': 'STEER RIGHT',
        'r': 'RESET',
        'p': 'PAUSE',
        'q': 'QUIT',
        '\x1b': 'ESC',
    }

    seen = set()
    esc_count = 0
    results = {}

    tty.setraw(fd)
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 5.0)
            if not r:
                print("  (waiting for keypress...)")
                continue

            ch = sys.stdin.read(1)

            if ch == '\x1b':
                r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r2:
                    ch2 = sys.stdin.read(1)
                    r3, _, _ = select.select([sys.stdin], [], [], 0.02)
                    if r3 and ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        seq = '\x1b[' + ch3
                        info = ESC_MAP.get(seq)
                        if info:
                            name, label = info
                            print(f"  KEY: {seq!r:15s}  →  {label}")
                            results[name] = True
                            seen.add(name)
                            continue
                    print(f"  KEY: ESC+{ch2!r}  (unknown sequence)")
                else:
                    esc_count += 1
                    print(f"  KEY: ESC  ({esc_count}/2 to exit test)")
                    if esc_count >= 2:
                        break
                continue

            label = CHAR_MAP.get(ch.lower(), f'unknown({ch!r})')
            hex_val = ch.encode().hex()
            print(f"  KEY: {ch!r:6s}  hex=0x{hex_val:4s}  →  {label}")
            results[ch.lower()] = True
            seen.add(ch.lower())

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print()
    needed = {'w', 'a', 'd', 's'}
    missing = needed - seen
    if not missing:
        print(PASS, "All primary keys detected correctly.")
    else:
        print(FAIL, f"Keys NOT detected: {missing}")
        print("       The termios reader isn't working in your terminal.")
        print("       Try running in a plain bash terminal (not tmux/screen).")
    return len(missing) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — UDP packet sniff (requires physics_server running)
# ─────────────────────────────────────────────────────────────────────────────

def test_sniff():
    """
    Sniffs the telemetry broadcast and the control socket simultaneously.
    Shows what speed, throttle, and brake values the server is actually
    processing and broadcasting — without relying on the control interface.
    Run this while physics_server.py is running.
    """
    print(SEP)
    print("TEST 2: Live packet sniffer (physics_server must be running)")
    print(SEP)

    # Listen for telemetry on 5001 (what server broadcasts)
    sock_telem = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_telem.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock_telem.bind(('0.0.0.0', DEFAULT_PORT_SEND))
        sock_telem.settimeout(3.0)
    except OSError as e:
        print(FAIL, f"Could not bind telemetry port {DEFAULT_PORT_SEND}: {e}")
        print("       Is visualizer_client.py already bound to this port?")
        sock_telem.close()
        return False

    print(f"  Listening for telemetry on :{DEFAULT_PORT_SEND} — waiting up to 5s...")

    received = 0
    prev_speed = None
    prev_throttle = None

    try:
        for _ in range(15):
            try:
                data, _ = sock_telem.recvfrom(TX_BYTES + 8)
                tf = TelemetryFrame.from_bytes(data)
                if tf is None:
                    continue

                speed_delta = ""
                if prev_speed is not None:
                    delta = tf.speed_kmh - prev_speed
                    if delta > 0.05:
                        speed_delta = f"  ↑ +{delta:.2f}"
                    elif delta < -0.05:
                        speed_delta = f"  ↓ {delta:.2f}"
                    else:
                        speed_delta = f"  ─ {delta:+.3f}"

                throttle_flag = ""
                if abs(tf.throttle) > 0.01:
                    throttle_flag = f"  THR={tf.throttle:.2f}"
                if abs(tf.brake_norm) > 0.01:
                    throttle_flag += f"  BRK={tf.brake_norm:.2f}"

                print(f"  frame={tf.frame_id:6d}  "
                      f"v={tf.speed_kmh:6.2f} km/h{speed_delta:12s}  "
                      f"thr={tf.throttle:.3f}  brk={tf.brake_norm:.3f}"
                      f"{throttle_flag}")

                prev_speed = tf.speed_kmh
                prev_throttle = tf.throttle
                received += 1
                time.sleep(0.3)

            except socket.timeout:
                print(WARN, "Timeout — no telemetry packet received.")
                break
    finally:
        sock_telem.close()

    if received == 0:
        print(FAIL, "No telemetry received. Is physics_server.py running?")
        return False

    print()
    print(INFO, f"{received} packets received.")
    if prev_throttle is not None and prev_throttle < 0.01:
        print(WARN, "Throttle is 0.0 in all received packets.")
        print("       This means either:")
        print("       a) control_interface.py is not sending throttle (keyboard broken)")
        print("       b) OR it IS sending but TC/drive_assists is clamping it to 0")
    print(PASS, "Telemetry pipeline is working.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Physics integrator sanity check
# ─────────────────────────────────────────────────────────────────────────────

def test_physics():
    """
    Runs the JAX physics integrator directly with known force inputs
    and checks that speed changes as expected.
    Does NOT need physics_server.py — runs the model directly.
    """
    print(SEP)
    print("TEST 3: JAX physics integrator direct test")
    print(SEP)

    try:
        import jax.numpy as jnp
        print(INFO, "JAX available.")
    except ImportError:
        print(FAIL, "JAX not installed. Cannot run this test.")
        return False

    try:
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from vehicle_dynamics import VehicleDynamicsH
        from data.configs.vehicle_params import vehicle_params as VP
        from data.configs.tire_coeffs import tire_coeffs as TC
        print(INFO, "VehicleDynamicsH loaded.")
    except Exception as e:
        print(FAIL, f"Could not import VehicleDynamicsH: {e}")
        return False

    # Build minimal setup params matching DEFAULT_SETUP in physics_server.py
    DEFAULT_SETUP = [40000.0, 40000.0, 500.0, 500.0, 3000.0, 3000.0, 0.30, 0.60]
    setup = jnp.array(DEFAULT_SETUP)

    vd = VehicleDynamicsH(VP, TC)

    # Build initial state: moving at 5 m/s, flat, equilibrium Z
    state = jnp.zeros(46)
    state = state.at[2].set(-0.015)   # Z_eq
    state = state.at[14].set(5.0)     # vx = 5 m/s
    state = state.at[28:32].set(25.0) # tire temps

    results = {}

    print()
    print("  Running 100 steps (0.5s) with different force inputs...\n")

    for label, u_val in [
        ("Zero force      [u=[0,0]]",      [0.0,    0.0]),
        ("Half throttle   [u=[0,1000]]",   [0.0, 1000.0]),
        ("Full throttle   [u=[0,2000]]",   [0.0, 2000.0]),
        ("Full brake      [u=[0,-8000]]",  [0.0, -8000.0]),
    ]:
        s = state
        ctrl = jnp.array(u_val)
        v_start = float(s[14])

        for _ in range(100):   # 100 × 5ms = 0.5s
            s = vd.simulate_step(s, ctrl, setup, dt=0.005)

        v_end = float(s[14])
        delta_v = v_end - v_start
        delta_a = delta_v / 0.5  # m/s²
        delta_g = delta_a / 9.81

        ok = True
        if "Zero" in label and abs(delta_v) > 1.0:
            ok = False
        if "throttle" in label and delta_v < 0.1:
            ok = False
        if "brake" in label and delta_v > -0.5:
            ok = False

        flag = PASS if ok else FAIL
        print(f"  {flag}  {label}")
        print(f"         v: {v_start:.2f} → {v_end:.2f} m/s  "
              f"(Δv={delta_v:+.3f}  Δa={delta_a:+.2f} m/s²  = {delta_g:+.3f}G)\n")
        results[label] = ok

    all_pass = all(results.values())
    if all_pass:
        print(PASS, "Physics integrator responds correctly to force inputs.")
    else:
        print(FAIL, "Physics integrator is NOT responding as expected.")
        print("       This means the dynamics model itself has a bug.")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Inject controls directly, watch server respond
# ─────────────────────────────────────────────────────────────────────────────

def test_inject():
    """
    Bypasses control_interface entirely.
    Sends throttle/brake/steer packets directly to the server and
    watches the telemetry to confirm the server responds.
    Run while physics_server.py is running.
    """
    print(SEP)
    print("TEST 4: Direct control injection (physics_server must be running)")
    print(SEP)
    print("  This bypasses the keyboard entirely.")
    print("  Injects: 0 → full throttle → full brake → steer left → center\n")

    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock_recv.bind(('0.0.0.0', 5003))  # use a fresh port, not 5001/5002
        sock_recv.settimeout(2.0)
    except OSError as e:
        print(WARN, f"Could not bind port 5003 for telemetry: {e}")
        print("       Will send packets but cannot read speed response.")
        sock_recv = None

    def get_speed():
        """Try to read current speed from server telemetry."""
        if sock_recv is None:
            return None
        try:
            data, _ = sock_recv.recvfrom(TX_BYTES + 8)
            tf = TelemetryFrame.from_bytes(data)
            return tf.speed_kmh if tf else None
        except socket.timeout:
            return None

    def send_ctrl(steer, thr_n, brk_n, label, duration=1.0):
        """Send a control for `duration` seconds and report speed change."""
        pkt = pack_controls(steer=steer, throttle_f=thr_n, brake_f=brk_n)
        t_end = time.perf_counter() + duration
        speeds = []
        while time.perf_counter() < t_end:
            sock_send.sendto(pkt, (DEFAULT_HOST, DEFAULT_PORT_RECV))
            spd = get_speed()
            if spd is not None:
                speeds.append(spd)
            time.sleep(0.05)

        if speeds:
            print(f"  {label}")
            print(f"         speed: {speeds[0]:.1f} → {speeds[-1]:.1f} km/h  "
                  f"(Δ = {speeds[-1]-speeds[0]:+.1f} km/h over {duration:.0f}s)")
        else:
            print(f"  {label}  [no telemetry available to confirm]")
        return speeds

    print("  Phase 1: Coasting (no input) — expect slow deceleration")
    s1 = send_ctrl(0.0, 0.0, 0.0, "COAST:", duration=2.0)

    print()
    print("  Phase 2: Full throttle — expect clear acceleration")
    s2 = send_ctrl(0.0, 3000.0, 0.0, "FULL THROTTLE (3000N):", duration=2.0)

    print()
    print("  Phase 3: Full brake — expect rapid deceleration")
    s3 = send_ctrl(0.0, 0.0, 8000.0, "FULL BRAKE (8000N):", duration=2.0)

    print()
    print("  Phase 4: Steer left — expect yaw rate increase")
    s4 = send_ctrl(-0.3, 1000.0, 0.0, "STEER LEFT + half throttle:", duration=2.0)

    sock_send.close()
    if sock_recv:
        sock_recv.close()

    # Evaluate
    print()
    issues = []

    if s1 and s2:
        if s2[-1] - s2[0] < 1.0:
            issues.append("Throttle had NO effect on speed.")
        else:
            print(PASS, "Throttle increases speed.")

    if s3:
        if s3[-1] - s3[0] > -1.0:
            issues.append("Brake had NO effect on speed.")
        else:
            print(PASS, "Brake decreases speed.")

    if not issues:
        print(PASS, "Server responds correctly to injected controls.")
        if s2 and s2[0] > 0:
            print(INFO, "If keyboard still does nothing, the problem is")
            print("       purely in control_interface.py keyboard capture.")
    else:
        print()
        for issue in issues:
            print(FAIL, issue)
        print()
        print("       The physics server is NOT responding to control packets.")
        print("       Possible causes:")
        print("       1. TC/ABS is clamping net force to zero")
        print("       2. The dynamics model produces no force for these inputs")
        print("       3. net_lon calculation has wrong sign somewhere")

    return len(issues) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Protocol sanity check
# ─────────────────────────────────────────────────────────────────────────────

def test_protocol():
    """
    Encodes and decodes a control packet, verifies values round-trip correctly.
    """
    print(SEP)
    print("TEST 5: Protocol encode/decode sanity")
    print(SEP)

    test_cases = [
        dict(steer=0.0,   throttle_f=3000.0,  brake_f=0.0,    label="Full throttle"),
        dict(steer=0.0,   throttle_f=0.0,     brake_f=8000.0, label="Full brake"),
        dict(steer=-0.35, throttle_f=1500.0,  brake_f=0.0,    label="Steer left + half thr"),
        dict(steer=0.0,   throttle_f=0.0,     brake_f=0.0,    label="All zero"),
    ]

    all_ok = True
    for tc in test_cases:
        raw = pack_controls(tc['steer'], tc['throttle_f'], tc['brake_f'])
        dec = unpack_controls(raw)

        ok = (
            abs(dec['steer']      - tc['steer'])      < 0.001 and
            abs(dec['throttle_f'] - tc['throttle_f']) < 0.1   and
            abs(dec['brake_f']    - tc['brake_f'])    < 0.1
        )
        flag = PASS if ok else FAIL
        if not ok:
            all_ok = False
        print(f"  {flag}  {tc['label']}")
        print(f"         steer: {tc['steer']:.3f} → {dec['steer']:.3f}  "
              f"thr: {tc['throttle_f']:.0f} → {dec['throttle_f']:.0f}N  "
              f"brk: {tc['brake_f']:.0f} → {dec['brake_f']:.0f}N")

    # Now check: what does the DYNAMICS actually see?
    print()
    print("  Checking vehicle_dynamics.py control mapping:")
    print("  (u[1]=net_lon in server, dynamics maps it to throttle/brake)")
    for net_lon, label in [
        (3000.0,  "Full throttle sent (3000N):"),
        (1000.0,  "Half throttle sent (1000N):"),
        (0.0,     "Coasting (0N):"),
        (-8000.0, "Full brake sent (-8000N):"),
    ]:
        throttle = max(0.0, min(1.0, net_lon / 2000.0))
        brake_f  = max(0.0, min(10000.0, -net_lon))
        print(f"  net_lon={net_lon:8.0f}N  →  "
              f"throttle={throttle:.3f}  brake_force={brake_f:.0f}N", end="")
        if throttle > 0 and brake_f > 0:
            print("  ⚠️  BOTH throttle AND brake active!", end="")
        print()

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'

    print()
    print("=" * 60)
    print("  Project-GP Simulator — Diagnostic Tool")
    print("=" * 60)
    print()

    if cmd == 'keyboard':
        test_keyboard()

    elif cmd == 'sniff':
        test_sniff()

    elif cmd == 'physics':
        test_physics()

    elif cmd == 'inject':
        test_inject()

    elif cmd == 'protocol':
        test_protocol()

    elif cmd == 'all':
        print("Running offline tests (keyboard, protocol, physics).")
        print("For live tests run: python simulator/debug_sim.py sniff|inject\n")
        test_protocol()
        print()
        test_physics()
        print()
        test_keyboard()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()