#!/usr/bin/env python3
"""
can_monitor.py — Replaces Linux candump for UDP Multicast / WSL testing
"""
import can

def main():
    print("[Monitor] Listening to UDP Multicast CAN bus (239.0.0.1 port 10000)...")
    # Pass IP as channel, use 'interface' instead of 'bustype', and pass port explicitly
    bus = can.interface.Bus(channel='239.0.0.1', interface='udp_multicast', port=10000)
    
    try:
        while True:
            msg = bus.recv(timeout=1.0)
            if msg and msg.arbitration_id == 0x300:  # Filter for our Co-Processor frame
                hex_data = " ".join([f"{b:02X}" for b in msg.data])
                print(f"udp_can  300   [{msg.dlc}]  {hex_data}")
    except KeyboardInterrupt:
        print("\n[Monitor] Stopped.")
        bus.shutdown()

if __name__ == '__main__':
    main()
