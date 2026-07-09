#!/usr/bin/env python3
"""
can_monitor.py — Replaces Linux candump for UDP Multicast / WSL testing
"""
import struct

# --- WSL1 KERNEL COMPATIBILITY PATCH ---
# WSL1 does not support SIOCINQ/FIONREAD ioctls on UDP sockets (Errno 22).
# We intercept the OSError and return a safe 64-byte buffer count.
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

def main():
    print("[Monitor] Listening to UDP Multicast CAN bus (239.0.0.1 port 10000)...")
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
