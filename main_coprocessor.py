#!/usr/bin/env python3
"""
main_coprocessor.py — Project-GP 200 Hz Standalone CAN Co-Processor over UDP
"""
import time
import threading
import struct

# --- WSL1 KERNEL COMPATIBILITY PATCH ---
try:
    import can.interfaces.udp_multicast.bus as _udp_mod
    _orig_ioctl = _udp_mod.ioctl
    def _wsl_ioctl_patch(fd, op, arg=0, *args, **kwargs):
        try:
            return _orig_ioctl(fd, op, arg, *args, **kwargs)
        except OSError as e:
            if e.errno == 22:  # EINVAL on WSL1 socket query
                return struct.pack('i', 64)
            raise
    _udp_mod.ioctl = _wsl_ioctl_patch
except (ImportError, AttributeError):
    pass
# ---------------------------------------

import can
import cantools
import numpy as np
import jax.numpy as jnp

from powertrain.powertrain_manager import make_powertrain_manager, powertrain_step
from config.vehicles.ter27 import vehicle_params_ter27 as VP
from scripts.vcu_bridge import pack_tv_split_frame

# --- CAN & SYSTEM CONFIGURATION ---
CAN_INTERFACE = 'udp_multicast'
CAN_CHANNEL = '239.0.0.1'
CAN_PORT = 10000
DBC_PATH = "TER.dbc"
SBC_TV_CMD_FRAME_ID = 0x300
LOOP_RATE_HZ = 100.0 # de normal 200Hz, 100Hz para silenciar warnings de sobrecarga de loop
LOOP_PERIOD_S = 1.0 / LOOP_RATE_HZ

class TelemetryBuffer:
    def __init__(self, db: cantools.database.can.Database):
        self.db = db
        self.lock = threading.Lock()
        self.throttle = 0.0
        self.brake = 0.0
        self.steer_delta = 0.0
        self.vx = 10.0
        self.vy = 0.0
        self.wz = 0.0
        self.omega = np.full(4, 50.0, dtype=np.float32)
        self.T_tire = np.full(4, 85.0, dtype=np.float32)

    def update_from_can(self, msg: can.Message):
        try:
            signals = self.db.decode_message(msg.arbitration_id, msg.data)
        except (KeyError, ValueError):
            return

        with self.lock:
            if msg.arbitration_id == 0x003 and 'APPS_AV' in signals:
                self.throttle = float(signals['APPS_AV']) / 255.0
            elif msg.arbitration_id == 0x004 and 'BPPS' in signals:
                self.brake = min(float(signals['BPPS']) / 80.0, 1.0)
            elif msg.arbitration_id == 0x005 and 'ANGLE' in signals:
                self.steer_delta = np.radians(float(signals['ANGLE']))
            elif msg.arbitration_id == 0x123:
                if 'v_x' in signals: self.vx = float(signals['v_x'])
                if 'v_y' in signals: self.vy = float(signals['v_y'])
            elif msg.arbitration_id == 0x118 and 'Yaw_Rate_z' in signals:
                self.wz = float(signals['Yaw_Rate_z'])
            elif msg.arbitration_id == 0x115:
                if 'RPM_L' in signals: self.omega[0] = float(signals['RPM_L']) * 0.104719755
                if 'RPM_R' in signals: self.omega[1] = float(signals['RPM_R']) * 0.104719755
            elif msg.arbitration_id == 0x027:
                if 'rlRPM' in signals: self.omega[2] = float(signals['rlRPM']) * 0.104719755
                if 'rrRPM' in signals: self.omega[3] = float(signals['rrRPM']) * 0.104719755
            elif msg.arbitration_id == 0x3F0:
                self.T_tire[0] = np.mean([signals[f'TW{i}'] for i in range(1, 5)])
            elif msg.arbitration_id == 0x3F4:
                self.T_tire[1] = np.mean([signals[f'TW{i}'] for i in range(1, 5)])
            elif msg.arbitration_id == 0x3FC:
                self.T_tire[2] = np.mean([signals[f'TW{i}'] for i in range(1, 5)])
            elif msg.arbitration_id == 0x3F8:
                self.T_tire[3] = np.mean([signals[f'TW{i}'] for i in range(1, 5)])

    def get_state(self):
        with self.lock:
            return (self.throttle, self.brake, self.steer_delta, 
                    self.vx, self.vy, self.wz, self.omega.copy(), self.T_tire.copy())

def can_rx_thread(bus: can.Bus, buffer: TelemetryBuffer):
    while True:
        msg = bus.recv(timeout=1.0)
        if msg:
            buffer.update_from_can(msg)

def main():
    print("==================================================")
    print(" PROJECT-GP: 200 Hz CAN CO-PROCESSOR INITIALIZING ")
    print("==================================================")

    try:
        db = cantools.database.load_file(DBC_PATH)
        print(f"[INIT] Successfully loaded CAN database: {DBC_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load DBC file '{DBC_PATH}': {e}")
        return

    try:
        bus = can.interface.Bus(channel=CAN_CHANNEL, interface=CAN_INTERFACE, port=CAN_PORT)
        print(f"[INIT] Bound to UDP Multicast CAN: {CAN_CHANNEL}:{CAN_PORT}")
    except Exception as e:
        print(f"[ERROR] Failed to bind CAN channel '{CAN_CHANNEL}': {e}")
        return

    config, state = make_powertrain_manager(VP)
    telemetry = TelemetryBuffer(db)
    
    print("[INIT] Forcing JAX JIT compilation (this takes ~10-15 seconds)...")
    t0 = time.perf_counter()
    dummy_diag, state = powertrain_step(
        throttle_raw=jnp.array(0.0), brake_raw=jnp.array(0.0), delta=jnp.array(0.0),
        vx=jnp.array(10.0), vy=jnp.array(0.0), wz=jnp.array(0.0),
        Fz=jnp.full(4, 750.0), Fy=jnp.zeros(4), omega_wheel=jnp.full(4, 50.0),
        alpha_t=jnp.zeros(4), T_tire=jnp.full(4, 85.0), mu_est=jnp.array(1.4),
        gp_sigma=jnp.array(0.05), curvature=jnp.array(0.0),
        manager_state=state, dt=jnp.array(LOOP_PERIOD_S), config=config
    )
    _ = pack_tv_split_frame(dummy_diag)
    print(f"[INIT] JIT Warmup complete in {time.perf_counter() - t0:.2f}s. Entering live loop.")

    rx_thread = threading.Thread(target=can_rx_thread, args=(bus, telemetry), daemon=True)
    rx_thread.start()

    next_wake_time = time.perf_counter() + LOOP_PERIOD_S
    step_count = 0

    try:
        while True:
            t_start = time.perf_counter()

            throttle, brake, delta, vx, vy, wz, omega, T_tire = telemetry.get_state()

            diag, state = powertrain_step(
                throttle_raw=jnp.array(throttle), brake_raw=jnp.array(brake), delta=jnp.array(delta),
                vx=jnp.array(vx), vy=jnp.array(vy), wz=jnp.array(wz),
                Fz=jnp.full(4, 750.0), Fy=jnp.zeros(4), omega_wheel=jnp.array(omega),
                alpha_t=jnp.zeros(4), T_tire=jnp.array(T_tire), mu_est=jnp.array(1.4),
                gp_sigma=jnp.array(0.05), curvature=jnp.array(0.0),
                manager_state=state, dt=jnp.array(LOOP_PERIOD_S), config=config
            )

            payload_bytes = pack_tv_split_frame(diag)

            can_msg = can.Message(
                arbitration_id=SBC_TV_CMD_FRAME_ID,
                data=payload_bytes,
                is_extended_id=False
            )
            bus.send(can_msg)

            # --- E. Enforce Dynamic Loop Timing ---
            exec_time = time.perf_counter() - t_start
            step_count += 1
            
            # Dynamic frequency printing (e.g. prints every 1 second)
            if step_count % int(LOOP_RATE_HZ) == 0:
                print(f"[{int(LOOP_RATE_HZ)}Hz Loop] Exec: {exec_time*1000:.2f}ms | Split RL: {diag.T_wheel[2]:.1f}Nm | CBF: {int(diag.cbf_active)} | T_Tire FL: {T_tire[0]:.1f}C")

            # Calculate sleep time to maintain exact frequency
            sleep_time = next_wake_time - time.perf_counter()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                next_wake_time += LOOP_PERIOD_S
            else:
                # If we overran, print warning and RESET the schedule anchor
                # so subsequent normal frames don't cascade warnings!
                if step_count % 20 == 0:  # Throttled warning printing
                    print(f"[WARNING] Loop overrun! Execution took {exec_time*1000:.2f}ms (Budget: {LOOP_PERIOD_S*1000:.1f}ms)")
                next_wake_time = time.perf_counter() + LOOP_PERIOD_S

    except KeyboardInterrupt:
        print("\n[STOP] Shutting down CAN Co-Processor gracefully.")
        bus.shutdown()

if __name__ == '__main__':
    main()
