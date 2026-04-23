#!/usr/bin/env python3
"""
Physics Isolation Diagnostic
─────────────────────────────────────────────────────────────────────────────
Runs Project-GP physics server in isolation with constant throttle.
Logs state trajectories to detect erratic behavior.

Usage:
    cd ~/FS_Driver_Setup_Optimizer
    python physics_isolation_diagnostic.py
"""

import sys
import time
import socket
import struct
import numpy as np
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path.home() / "FS_Driver_Setup_Optimizer"))

from simulator.sim_protocol import TX, RX, CMD, TX_FMT, RX_FMT, TX_BYTES, RX_BYTES, unpack_controls
from simulator.sim_config import S, HOST, PORT_CTRL_RECV, PORT_TELEM_VIZ, PHYSICS_HZ

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Configuration
# ─────────────────────────────────────────────────────────────────────────────

DURATION_SEC = 30
PHYSICS_HZ_ACTUAL = 200
N_FRAMES = DURATION_SEC * PHYSICS_HZ_ACTUAL

# Test profile: constant 20% throttle
THROTTLE_PROFILE = 0.2
STEER_PROFILE = 0.0
BRAKE_PROFILE = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# UDP Client to Command Physics Server
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsClientDiagnostic:
    def __init__(self, host=HOST, port=PORT_CTRL_RECV, telem_port=PORT_TELEM_VIZ):
        self.host = host
        self.port = port
        self.telem_port = telem_port
        
        # Control socket (send commands)
        self.ctrl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctrl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Telemetry socket (receive state)
        self.telem_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telem_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.telem_socket.bind(("0.0.0.0", self.telem_port))
        self.telem_socket.settimeout(0.1)  # Non-blocking with timeout
        
        # State history (for divergence detection)
        self.state_history = deque(maxlen=200)  # 1 second @ 200 Hz
        self.time_history = deque(maxlen=200)
        
    def send_control(self, throttle, steer, brake, cmd_type=CMD.DRIVE):
        """Send control command to physics server."""
        # Pack into RX format (commands TO the server) — 8 floats
        payload = struct.pack(
            RX_FMT,
            throttle,      # [0]
            steer,         # [1]
            brake,         # [2]
            cmd_type,      # [3]
            0.0, 0.0, 0.0, 0.0,  # [4-7] padding
        )
        self.ctrl_socket.sendto(payload, (self.host, self.port))
    
    def recv_telemetry(self):
        """Receive telemetry from physics server. Returns None if no data."""
        try:
            data, _ = self.telem_socket.recvfrom(TX_BYTES)
            if len(data) != TX_BYTES:
                return None
            
            # Unpack telemetry frame (TX format)
            frame = struct.unpack(TX_FMT, data)
            return frame
        except socket.timeout:
            return None
    
    def extract_key_states(self, frame):
        """Extract key state variables from telemetry frame."""
        # TX frame layout: [t, x, y, z, roll, pitch, yaw, vx, vy, wz, ...]
        # See sim_protocol.py for exact mapping
        if frame is None:
            return None
        
        return {
            't': frame[0],
            'x': frame[1],
            'y': frame[2],
            'z': frame[3],
            'roll': frame[4],
            'pitch': frame[5],
            'yaw': frame[6],
            'vx': frame[7],
            'vy': frame[8],
            'wz': frame[9],
            'ax': frame[10] if len(frame) > 10 else 0.0,
            'ay': frame[11] if len(frame) > 11 else 0.0,
            'throttle': frame[12] if len(frame) > 12 else 0.0,
        }
    
    def run_diagnostic(self):
        """Run 30-second constant throttle test."""
        print("=" * 80)
        print("PROJECT-GP PHYSICS ISOLATION DIAGNOSTIC")
        print("=" * 80)
        print(f"Duration: {DURATION_SEC} seconds")
        print(f"Physics rate: {PHYSICS_HZ_ACTUAL} Hz")
        print(f"Throttle: {THROTTLE_PROFILE * 100:.1f}%")
        print(f"Steer: {STEER_PROFILE * 100:.1f}%")
        print("=" * 80)
        print()
        
        # Wait for server to be ready
        print("[DIAGNOSTIC] Waiting for physics server to be ready...")
        time.sleep(2)
        
        # Run diagnostic loop
        frame_count = 0
        start_time = time.time()
        last_print_time = start_time
        
        # Buffers for state trajectory
        x_traj = []
        y_traj = []
        z_traj = []
        vx_traj = []
        vy_traj = []
        wz_traj = []
        roll_traj = []
        pitch_traj = []
        yaw_traj = []
        ax_traj = []
        ay_traj = []
        
        try:
            while frame_count < N_FRAMES:
                # Send constant control
                self.send_control(THROTTLE_PROFILE, STEER_PROFILE, BRAKE_PROFILE)
                
                # Try to receive telemetry
                frame = self.recv_telemetry()
                if frame is not None:
                    state = self.extract_key_states(frame)
                    if state is not None:
                        # Record state
                        x_traj.append(state['x'])
                        y_traj.append(state['y'])
                        z_traj.append(state['z'])
                        vx_traj.append(state['vx'])
                        vy_traj.append(state['vy'])
                        wz_traj.append(state['wz'])
                        roll_traj.append(state['roll'])
                        pitch_traj.append(state['pitch'])
                        yaw_traj.append(state['yaw'])
                        ax_traj.append(state['ax'])
                        ay_traj.append(state['ay'])
                        
                        frame_count += 1
                
                # Print progress every 5 seconds
                now = time.time()
                if now - last_print_time >= 5.0:
                    elapsed = now - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"[{elapsed:5.1f}s] Frames: {frame_count:4d} | FPS: {fps:6.1f} | "
                          f"Pos: ({x_traj[-1]:7.2f}, {y_traj[-1]:7.2f}, {z_traj[-1]:6.3f}) | "
                          f"Vel: ({vx_traj[-1]:6.2f}, {vy_traj[-1]:6.2f}) m/s | "
                          f"Yaw: {np.degrees(yaw_traj[-1]):7.2f}°")
                    last_print_time = now
                
                # Sleep to not saturate CPU
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n[DIAGNOSTIC] Interrupted by user.")
        
        elapsed_total = time.time() - start_time
        
        # ─────────────────────────────────────────────────────────────────────
        # Analysis
        # ─────────────────────────────────────────────────────────────────────
        
        print()
        print("=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        
        if len(x_traj) < 100:
            print(f"[ERROR] Insufficient data: only {len(x_traj)} frames received.")
            print("        → Physics server may not be running or not responding.")
            return False
        
        # Convert to numpy for analysis
        x = np.array(x_traj)
        y = np.array(y_traj)
        z = np.array(z_traj)
        vx = np.array(vx_traj)
        vy = np.array(vy_traj)
        wz = np.array(wz_traj)
        roll = np.array(roll_traj)
        pitch = np.array(pitch_traj)
        yaw = np.array(yaw_traj)
        ax = np.array(ax_traj)
        ay = np.array(ay_traj)
        
        # 1. Check for NaN / Inf
        has_nan = np.isnan(x).any() or np.isnan(vx).any() or np.isnan(ax).any()
        has_inf = np.isinf(x).any() or np.isinf(vx).any() or np.isinf(ax).any()
        
        print(f"\n[NaN Check]        {'FAIL ✗' if has_nan else 'PASS ✓'}")
        print(f"[Inf Check]        {'FAIL ✗' if has_inf else 'PASS ✓'}")
        
        if has_nan or has_inf:
            print("   → Physics diverged to NaN/Inf. Check:")
            print("      1. Tire forces (may be too large)")
            print("      2. Suspension forces (may have negative energy)")
            print("      3. Control limits (steer/throttle saturation)")
            return False
        
        # 2. Check for unrealistic acceleration
        max_ax = np.abs(ax).max()
        max_ay = np.abs(ay).max()
        
        print(f"\n[Accel Check]      max|ax|={max_ax:6.2f} m/s² (expect <20)")
        print(f"                   max|ay|={max_ay:6.2f} m/s² (expect <20)")
        
        accel_ok = (max_ax < 25) and (max_ay < 25)
        print(f"                   {'PASS ✓' if accel_ok else 'WARN ⚠'}")
        
        # 3. Check velocity growth (should be monotonic with constant throttle)
        vx_growth = vx[-1] - vx[0]
        print(f"\n[Velocity Growth]  ΔVx = {vx_growth:.2f} m/s over {elapsed_total:.1f}s")
        print(f"                   Expected: ~2-4 m/s for 20% throttle")
        
        # 4. Check for oscillations (roll/pitch should be smooth)
        roll_rate = np.gradient(roll)
        pitch_rate = np.gradient(pitch)
        max_roll_rate = np.abs(roll_rate).max()
        max_pitch_rate = np.abs(pitch_rate).max()
        
        print(f"\n[Oscillations]     max|roll_rate|  = {np.degrees(max_roll_rate):.2f} °/s")
        print(f"                   max|pitch_rate| = {np.degrees(max_pitch_rate):.2f} °/s")
        
        roll_ok = max_roll_rate < np.radians(45)  # <45 deg/s
        pitch_ok = max_pitch_rate < np.radians(45)
        print(f"                   {'PASS ✓' if (roll_ok and pitch_ok) else 'WARN ⚠'}")
        
        # 5. Check Z position (height should be roughly constant)
        z_variance = np.var(z)
        z_max = np.abs(z).max()
        
        print(f"\n[Height Stability] Var(Z) = {z_variance:.6f} m²")
        print(f"                   max|Z| = {z_max:.4f} m (expect <0.2)")
        
        z_ok = z_variance < 0.01 and z_max < 0.2
        print(f"                   {'PASS ✓' if z_ok else 'WARN ⚠'}")
        
        # 6. Summary
        print()
        print("=" * 80)
        all_ok = (not has_nan) and (not has_inf) and accel_ok and z_ok
        if all_ok:
            print("✓ PHYSICS HEALTHY — Physics model appears stable")
            print("  Next: Run MirenaSim bridge and compare state divergence")
        else:
            print("✗ PHYSICS ISSUES DETECTED — Investigate above warnings")
            print("  Likely culprits:")
            print("    1. Tire model coefficients too large")
            print("    2. Suspension stiffness/damping misconfigured")
            print("    3. Bumpstop force too aggressive")
            print("    4. Thermal model causing grip loss")
        print("=" * 80)
        
        return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    import threading
    
    # Start physics server in background
    print("[STARTUP] Starting physics server in background...")
    server_proc = subprocess.Popen(
        ["python", "simulator/physics_server.py", "--track", "fsg_autocross"],
        cwd=str(Path.home() / "FS_Driver_Setup_Optimizer"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Give it time to start
    time.sleep(3)
    
    try:
        # Run diagnostic
        client = PhysicsClientDiagnostic()
        success = client.run_diagnostic()
        
        sys.exit(0 if success else 1)
    
    finally:
        # Cleanup
        print("\n[CLEANUP] Terminating physics server...")
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print("[CLEANUP] Done.")