"""
simulator/ws_bridge.py
─────────────────────────────────────────────────────────────────────────────
Project-GP WebSocket Bridge — UDP Telemetry → React Dashboard

Architecture:
  physics_server.py  ──UDP 256B──→  ws_bridge.py  ──WebSocket JSON──→  React Dashboard

The bridge:
  1. Listens for 64-float UDP telemetry packets from physics_server.py
  2. Decimates to ~60 Hz (dashboard doesn't need 200 Hz)
  3. Serialises to a compact JSON dict with named fields
  4. Broadcasts to all connected WebSocket clients
  5. Receives control commands from the dashboard and forwards via UDP

Usage:
  python simulator/ws_bridge.py                          # defaults
  python simulator/ws_bridge.py --ws-port 8765 --udp-port 5001

React client connection:
  const ws = new WebSocket('ws://localhost:8765');
  ws.onmessage = (e) => {
    const frame = JSON.parse(e.data);
    console.log(frame.speed_kmh, frame.lat_g, frame.Fz_fl);
  };

  // Send control commands back:
  ws.send(JSON.stringify({ type: 'control', steer: 0.1, throttle: 0.5, brake: 0.0 }));
  ws.send(JSON.stringify({ type: 'reset' }));
  ws.send(JSON.stringify({ type: 'setup', preset: '3_stiff' }));
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import time
import argparse
import signal
import sys
import os
from collections import deque
from typing import Set, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("[ws_bridge] ERROR: 'websockets' package not found.")
    print("           Install with: pip install websockets")
    sys.exit(1)

from simulator.sim_config import (
    HOST, PORT_TELEM_VIZ, PORT_CTRL_RECV,
    WS_HOST, WS_PORT, PHYSICS_HZ, TELEMETRY_HZ,
    S,
)

# ─────────────────────────────────────────────────────────────────────────────
# §1  Telemetry Frame Parser
# ─────────────────────────────────────────────────────────────────────────────

# sim_protocol.py TX format: 64 floats = 256 bytes
TX_FMT    = '<64f'
TX_BYTES  = struct.calcsize(TX_FMT)

# Named field mapping (index → key) for the 64-float packet.
# Must match sim_protocol.py TX layout exactly.
TELEM_FIELDS = {
    0:  "magic",
    1:  "frame_id",
    2:  "sim_time",
    3:  "x",
    4:  "y",
    5:  "z",
    6:  "roll",
    7:  "pitch",
    8:  "yaw",
    9:  "vx",
    10: "vy",
    11: "vz",
    12: "ax",
    13: "ay",
    14: "az",
    15: "wz",
    16: "z_fl",
    17: "z_fr",
    18: "z_rl",
    19: "z_rr",
    20: "Fz_fl",
    21: "Fz_fr",
    22: "Fz_rl",
    23: "Fz_rr",
    24: "Fy_fl",
    25: "Fy_fr",
    26: "Fy_rl",
    27: "Fy_rr",
    28: "slip_fl",
    29: "slip_fr",
    30: "slip_rl",
    31: "slip_rr",
    32: "kappa_rl",
    33: "kappa_rr",
    34: "omega_fl",
    35: "omega_fr",
    36: "omega_rl",
    37: "omega_rr",
    38: "T_fl",
    39: "T_fr",
    40: "T_rl",
    41: "T_rr",
    42: "delta",
    43: "throttle",
    44: "brake_norm",
    45: "speed_kmh",
    46: "lat_g",
    47: "lon_g",
    48: "grip_f",
    49: "grip_r",
    50: "energy_kj",
    51: "downforce",
    52: "drag",
    53: "lap_time",
    54: "lap_number",
    55: "sector",
    56: "yaw_rate_deg",
    57: "reserved_1",
    58: "reserved_2",
    59: "reserved_3",
    60: "reserved_4",
    61: "reserved_5",
    62: "reserved_6",
    63: "reserved_7",
}

# Fields to send to dashboard (skip magic, reserved, redundant)
DASHBOARD_FIELDS = [
    "frame_id", "sim_time",
    "x", "y", "z", "roll", "pitch", "yaw",
    "vx", "vy", "vz", "ax", "ay", "az", "wz",
    "z_fl", "z_fr", "z_rl", "z_rr",
    "Fz_fl", "Fz_fr", "Fz_rl", "Fz_rr",
    "Fy_fl", "Fy_fr", "Fy_rl", "Fy_rr",
    "slip_fl", "slip_fr", "slip_rl", "slip_rr",
    "kappa_rl", "kappa_rr",
    "omega_fl", "omega_fr", "omega_rl", "omega_rr",
    "T_fl", "T_fr", "T_rl", "T_rr",
    "delta", "throttle", "brake_norm",
    "speed_kmh", "lat_g", "lon_g",
    "grip_f", "grip_r",
    "energy_kj", "downforce", "drag",
    "lap_time", "lap_number", "sector",
]

# Build reverse lookup: field_name → index in the 64-float packet
_FIELD_TO_IDX = {name: idx for idx, name in TELEM_FIELDS.items()}


def parse_telemetry(raw: bytes) -> Optional[dict]:
    """Parse 256-byte UDP packet into a named dict for the dashboard."""
    if len(raw) < TX_BYTES:
        return None
    try:
        values = struct.unpack(TX_FMT, raw[:TX_BYTES])
    except struct.error:
        return None

    frame = {}
    for field_name in DASHBOARD_FIELDS:
        idx = _FIELD_TO_IDX.get(field_name)
        if idx is not None:
            v = values[idx]
            # Round floats to reduce JSON size (3 decimals for most, 0 for integers)
            if field_name in ("frame_id", "lap_number", "sector"):
                frame[field_name] = int(v)
            elif field_name in ("Fz_fl", "Fz_fr", "Fz_rl", "Fz_rr",
                                "Fy_fl", "Fy_fr", "Fy_rl", "Fy_rr",
                                "downforce", "drag"):
                frame[field_name] = round(v, 1)
            else:
                frame[field_name] = round(v, 4)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# §2  WebSocket Bridge Server
# ─────────────────────────────────────────────────────────────────────────────

class WSBridge:
    """
    Async WebSocket bridge between UDP telemetry and React dashboard.

    Decimates 200 Hz UDP to ~60 Hz WebSocket.
    Handles multiple simultaneous dashboard clients.
    Forwards control commands from dashboard → physics server.
    """

    def __init__(self,
                 udp_port:  int = PORT_TELEM_VIZ,
                 ws_host:   str = WS_HOST,
                 ws_port:   int = WS_PORT,
                 ctrl_host: str = HOST,
                 ctrl_port: int = PORT_CTRL_RECV,
                 target_hz: int = TELEMETRY_HZ):

        self.udp_port   = udp_port
        self.ws_host    = ws_host
        self.ws_port    = ws_port
        self.ctrl_host  = ctrl_host
        self.ctrl_port  = ctrl_port
        self.target_hz  = target_hz

        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._latest_frame: Optional[dict] = None
        self._frame_count    = 0
        self._ws_send_count  = 0
        self._last_stats     = time.monotonic()

        # UDP socket for forwarding controls to physics server
        self._ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Performance tracking
        self._latencies: deque = deque(maxlen=200)

    # ── WebSocket handler ────────────────────────────────────────────────

    async def _ws_handler(self, ws: websockets.WebSocketServerProtocol):
        """Handle a single WebSocket client connection."""
        self._clients.add(ws)
        client_addr = ws.remote_address
        print(f"[ws_bridge] Dashboard connected: {client_addr}  "
              f"(total: {len(self._clients)})")

        try:
            async for message in ws:
                await self._handle_dashboard_command(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            print(f"[ws_bridge] Dashboard disconnected: {client_addr}  "
                  f"(total: {len(self._clients)})")

    async def _handle_dashboard_command(self, raw_msg: str):
        """Parse and forward control commands from the dashboard."""
        try:
            cmd = json.loads(raw_msg)
        except json.JSONDecodeError:
            return

        cmd_type = cmd.get("type", "")

        if cmd_type == "control":
            # Forward steering/throttle/brake to physics server
            steer  = float(cmd.get("steer", 0.0))
            thr    = float(cmd.get("throttle", 0.0))
            brk    = float(cmd.get("brake", 0.0))
            # Pack as RX format: steer, throttle_force, brake_force, cmd_type, ...
            # Match sim_protocol.py RX_FMT = '<8f' (8 floats = 32 bytes)
            pkt = struct.pack('<8f', steer, thr * 2000.0, brk * 8000.0,
                              0.0, 0.0, 0.0, 0.0, 0.0)
            self._ctrl_sock.sendto(pkt, (self.ctrl_host, self.ctrl_port))

        elif cmd_type == "reset":
            # CMD.RESET = 1.0
            pkt = struct.pack('<8f', 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
            self._ctrl_sock.sendto(pkt, (self.ctrl_host, self.ctrl_port))

        elif cmd_type == "pause":
            pkt = struct.pack('<8f', 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0)
            self._ctrl_sock.sendto(pkt, (self.ctrl_host, self.ctrl_port))

        elif cmd_type == "resume":
            pkt = struct.pack('<8f', 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0)
            self._ctrl_sock.sendto(pkt, (self.ctrl_host, self.ctrl_port))

        elif cmd_type == "setup":
            # Hot setup change — forward preset name or raw array
            preset = cmd.get("preset")
            if preset:
                from simulator.sim_config import PRESET_SETUPS
                setup_arr = PRESET_SETUPS.get(preset)
                if setup_arr is not None:
                    # CMD.SETUP_CHANGE = 2.0; k_f, k_r at positions 4,5
                    pkt = struct.pack('<8f', 0.0, 0.0, 0.0, 2.0,
                                      float(setup_arr[0]), float(setup_arr[1]),
                                      float(setup_arr[2]), float(setup_arr[3]))
                    self._ctrl_sock.sendto(pkt, (self.ctrl_host, self.ctrl_port))

    # ── UDP receiver + WebSocket broadcaster ─────────────────────────────

    async def _udp_listener(self):
        """
        Async UDP listener using loop.sock_recvfrom().
        Decimates to target_hz before broadcasting.
        """
        loop = asyncio.get_running_loop()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.udp_port))
        sock.setblocking(False)

        print(f"[ws_bridge] Listening for UDP telemetry on :{self.udp_port}")

        min_interval = 1.0 / self.target_hz
        last_send_time = 0.0

        while True:
            try:
                data, addr = await loop.sock_recvfrom(sock, TX_BYTES + 16)
            except Exception:
                await asyncio.sleep(0.001)
                continue

            self._frame_count += 1

            # Decimation: only parse + broadcast at target_hz
            now = time.monotonic()
            if (now - last_send_time) < min_interval:
                continue

            frame = parse_telemetry(data)
            if frame is None:
                continue

            self._latest_frame = frame
            last_send_time = now

            # Broadcast to all connected dashboards
            if self._clients:
                msg = json.dumps(frame, separators=(',', ':'))
                stale = set()
                for ws in self._clients:
                    try:
                        await ws.send(msg)
                        self._ws_send_count += 1
                    except websockets.exceptions.ConnectionClosed:
                        stale.add(ws)
                self._clients -= stale

            # Stats logging every 5 seconds
            if now - self._last_stats > 5.0:
                udp_hz = self._frame_count / (now - self._last_stats)
                ws_hz  = self._ws_send_count / (now - self._last_stats)
                n_clients = len(self._clients)
                print(f"[ws_bridge] UDP: {udp_hz:.0f} Hz → WS: {ws_hz:.0f} Hz | "
                      f"{n_clients} client(s) connected")
                self._frame_count   = 0
                self._ws_send_count = 0
                self._last_stats    = now

    # ── Main entry point ─────────────────────────────────────────────────

    async def run(self):
        """Start WebSocket server and UDP listener concurrently."""
        print("=" * 60)
        print("  Project-GP WebSocket Bridge")
        print(f"  UDP listen   : :{self.udp_port}")
        print(f"  WebSocket    : ws://{self.ws_host}:{self.ws_port}")
        print(f"  Target rate  : {self.target_hz} Hz")
        print(f"  Ctrl forward : {self.ctrl_host}:{self.ctrl_port}")
        print("=" * 60)

        async with serve(
            self._ws_handler,
            self.ws_host,
            self.ws_port,
            ping_interval=20,
            ping_timeout=60,
            max_size=None,
            compression=None,     # disable per-message compression for latency
        ):
            print(f"[ws_bridge] WebSocket server running on ws://{self.ws_host}:{self.ws_port}")
            print(f"[ws_bridge] Waiting for physics_server.py telemetry...")
            await self._udp_listener()


# ─────────────────────────────────────────────────────────────────────────────
# §3  CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Project-GP WebSocket Bridge: UDP telemetry → React Dashboard")
    parser.add_argument("--udp-port", type=int, default=PORT_TELEM_VIZ,
                        help=f"UDP port to listen for telemetry (default: {PORT_TELEM_VIZ})")
    parser.add_argument("--ws-host", type=str, default=WS_HOST,
                        help=f"WebSocket bind address (default: {WS_HOST})")
    parser.add_argument("--ws-port", type=int, default=WS_PORT,
                        help=f"WebSocket port (default: {WS_PORT})")
    parser.add_argument("--target-hz", type=int, default=TELEMETRY_HZ,
                        help=f"Dashboard update rate (default: {TELEMETRY_HZ})")
    args = parser.parse_args()

    bridge = WSBridge(
        udp_port  = args.udp_port,
        ws_host   = args.ws_host,
        ws_port   = args.ws_port,
        target_hz = args.target_hz,
    )

    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("\n[ws_bridge] Shutting down.")


if __name__ == "__main__":
    main()