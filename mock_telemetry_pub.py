#!/usr/bin/env python3
"""
mock_telemetry_pub.py — Simulates TeR_ECU and Inverter CAN traffic over UDP
"""
import time
import math
import can
import cantools

DBC_PATH = "TER.dbc"
CAN_CHANNEL = "239.0.0.1"
CAN_PORT = 10000

def main():
    print("[Mock Pub] Loading TER.dbc...")
    db = cantools.database.load_file(DBC_PATH)
    bus = can.interface.Bus(channel=CAN_CHANNEL, interface='udp_multicast', port=CAN_PORT)

    print("[Mock Pub] Broadcasting dynamic cornering telemetry at 50 Hz...")
    t0 = time.time()
    
    try:
        while True:
            t = time.time() - t0
            
            # Simulate accelerating into a sweeping left turn
            throttle_val = min(0.3 + 0.2 * t, 1.0) * 255.0  # APPS_AV (0..255)
            steer_deg = 30.0 * math.sin(t * 0.5)            # Sweeping left/right (-30 to +30 deg)
            rpm_fl = 500.0 + 50.0 * math.sin(t)
            rpm_fr = 500.0 - 50.0 * math.sin(t)             # Outer wheel faster in turns

            # 1. Send Throttle (ID 3 / 0x003: APPS) — strict=False ignores DBC [0|1] bounds
            msg_apps = db.get_message_by_name('APPS')
            data_apps = msg_apps.encode({'APPS_AV': int(throttle_val), 'APPS_1': 0, 'APPS_2': 0, 'IMP_FLAG': 0}, strict=False)
            bus.send(can.Message(arbitration_id=0x003, data=data_apps, is_extended_id=False))

            # 2. Send Steering Angle (ID 5 / 0x005: STEER)
            msg_steer = db.get_message_by_name('STEER')
            data_steer = msg_steer.encode({'ANGLE': steer_deg}, strict=False)
            bus.send(can.Message(arbitration_id=0x005, data=data_steer, is_extended_id=False))

            # 3. Send Front Wheel RPMs (ID 277 / 0x115: Front_RPM)
            msg_rpm = db.get_message_by_name('Front_RPM')
            data_rpm = msg_rpm.encode({'RPM_L': rpm_fl, 'RPM_R': rpm_fr}, strict=False)
            bus.send(can.Message(arbitration_id=0x115, data=data_rpm, is_extended_id=False))

            time.sleep(0.02)  # 50 Hz publication rate

    except KeyboardInterrupt:
        print("\n[Mock Pub] Stopped.")
        bus.shutdown()

if __name__ == '__main__':
    main()
