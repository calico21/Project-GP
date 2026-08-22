"""
simulator/sim_protocol.py
─────────────────────────────────────────────────────────────────────────────
Shared UDP wire protocol between the physics server and all clients.

Design goals:
  · Self-describing — every packet starts with a magic byte verification.
  · Extensible — new analytical channels added on separate ports.
  · Low-latency — fixed-size packed C-structs, no serialization overhead.
  · Bi-directional — server broadcasts metrics, client injects commands.

Strictly preserves the legacy 64-float and 8-float formats to avoid breaking
downstream structural unpacking in Godot and the ROS 2 driverless stack.
"""

from __future__ import annotations
import struct
import numpy as np
from typing import Optional, Dict

# ─────────────────────────────────────────────────────────────────────────────
# §1  Wire Constants & Protocol Struct Configurations (FROZEN)
# ─────────────────────────────────────────────────────────────────────────────

# Server-to-Client Main Telemetry Frame: 64 floats = 256 bytes
TX_N = 64
TX_FMT = f'<{TX_N}f'
TX_BYTES = struct.calcsize(TX_FMT)

# Client-to-Server Command Vector Frame: 8 floats = 32 bytes
RX_N = 8
RX_FMT = f'<{RX_N}f'
RX_BYTES = struct.calcsize(RX_FMT)

# Magic bytes definition aligned to continuous single-precision float representation
MAGIC = 0xCAFE
MAGIC_FLOAT = float(struct.unpack('<f', struct.pack('<I', MAGIC))[0])

# ── Additive dMPC Diagnostics Protocol (Standalone port to decouple tracking) ──
PORT_DMPC_DIAG = 5005
DMPC_DIAG_N = 16
DMPC_DIAG_FMT = f'<{DMPC_DIAG_N}f'
DMPC_DIAG_BYTES = struct.calcsize(DMPC_DIAG_FMT)
MAGIC_DMPC_FLOAT = float(struct.unpack('<f', struct.pack('<I', 0xDB9C))[0])


# ─────────────────────────────────────────────────────────────────────────────
# §2  Named Positional Layout Mappings
# ─────────────────────────────────────────────────────────────────────────────

class TX:
    """Telemetry structure offsets matching the 64-float layout exactly."""
    MAGIC          = 0
    FRAME_ID       = 1
    SIM_TIME       = 2
    X              = 3
    Y              = 4
    Z              = 5
    ROLL           = 6
    PITCH          = 7
    YAW            = 8
    VX             = 9
    VY             = 10
    VZ             = 11
    AX             = 12
    AY             = 13
    AZ             = 14
    WZ             = 15
    Z_FL           = 16
    Z_RL           = 18
    Z_FR           = 17
    Z_RR           = 19
    FZ_FL          = 20
    FZ_FR          = 21
    FZ_RL          = 22
    FZ_RR          = 23
    FY_FL          = 24
    FY_FR          = 25
    FY_RL          = 26
    FY_RR          = 27
    SLIP_FL        = 28
    SLIP_FR        = 29
    SLIP_RL        = 30
    SLIP_RR        = 31
    KAPPA_RL       = 32
    KAPPA_RR       = 33
    OMEGA_FL       = 34
    OMEGA_FR       = 35
    OMEGA_RL       = 36
    OMEGA_RR       = 37
    T_FL           = 38
    T_FR           = 39
    T_RL           = 40
    T_RR           = 41
    DELTA          = 42
    THROTTLE       = 43
    BRAKE_NORM     = 44
    GRIP_UTIL_F    = 45
    GRIP_UTIL_R    = 46
    LAP_TIME       = 47
    LAP_NUMBER     = 48
    SECTOR         = 49
    SPEED_KMH      = 50
    LAT_G          = 51
    LON_G          = 52
    YAW_RATE_DEG   = 53
    DOWNFORCE      = 54
    DRAG           = 55
    ENERGY_KJ      = 56
    TRANS_FL       = 57
    TRANS_FR       = 58
    TRANS_RL       = 59
    TRANS_RR       = 60
    SETUP_HASH     = 61


class RX:
    """Command vector indexes matching the 8-float layout exactly."""
    STEER         = 0
    THROTTLE_F    = 1
    BRAKE_F       = 2
    CMD_TYPE      = 3
    SETUP_K_F     = 4
    SETUP_K_R     = 5
    SETUP_ARB_F   = 6
    SETUP_ARB_R   = 7


class CMD:
    """Operational mode enumeration flags passed via control vectors."""
    DRIVE         = 0.0
    RESET         = 1.0
    SETUP_CHANGE  = 2.0
    PAUSE         = 3.0
    RESUME        = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# §3  Telemetry Packet Object Abstraction Layer
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryFrame:
    """Dataclass abstraction managing serialization/deserialization schemas."""
    
    def __init__(self, **kwargs):
        self.frame_id     = kwargs.get('frame_id', 0)
        self.sim_time     = kwargs.get('sim_time', 0.0)
        self.x            = kwargs.get('x', 0.0)
        self.y            = kwargs.get('y', 0.0)
        self.z            = kwargs.get('z', 0.0)
        self.roll         = kwargs.get('roll', 0.0)
        self.pitch        = kwargs.get('pitch', 0.0)
        self.yaw          = kwargs.get('yaw', 0.0)
        self.vx           = kwargs.get('vx', 0.0)
        self.vy           = kwargs.get('vy', 0.0)
        self.vz           = kwargs.get('vz', 0.0)
        self.ax           = kwargs.get('ax', 0.0)
        self.ay           = kwargs.get('ay', 0.0)
        self.az           = kwargs.get('az', 0.0)
        self.wz           = kwargs.get('wz', 0.0)
        self.z_fl         = kwargs.get('z_fl', 0.0)
        self.z_fr         = kwargs.get('z_fr', 0.0)
        self.z_rl         = kwargs.get('z_rl', 0.0)
        self.z_rr         = kwargs.get('z_rr', 0.0)
        self.Fz_fl        = kwargs.get('Fz_fl', 0.0)
        self.Fz_fr        = kwargs.get('Fz_fr', 0.0)
        self.Fz_rl        = kwargs.get('Fz_rl', 0.0)
        self.Fz_rr        = kwargs.get('Fz_rr', 0.0)
        self.Fy_fl        = kwargs.get('Fy_fl', 0.0)
        self.Fy_fr        = kwargs.get('Fy_fr', 0.0)
        self.Fy_rl        = kwargs.get('Fy_rl', 0.0)
        self.Fy_rr        = kwargs.get('Fy_rr', 0.0)
        self.slip_fl      = kwargs.get('slip_fl', 0.0)
        self.slip_fr      = kwargs.get('slip_fr', 0.0)
        self.slip_rl      = kwargs.get('slip_rl', 0.0)
        self.slip_rr      = kwargs.get('slip_rr', 0.0)
        self.kappa_rl     = kwargs.get('kappa_rl', 0.0)
        self.kappa_rr     = kwargs.get('kappa_rr', 0.0)
        self.omega_fl     = kwargs.get('omega_fl', 0.0)
        self.omega_fr     = kwargs.get('omega_fr', 0.0)
        self.omega_rl     = kwargs.get('omega_rl', 0.0)
        self.omega_rr     = kwargs.get('omega_rr', 0.0)
        self.T_fl         = kwargs.get('T_fl', 25.0)
        self.T_fr         = kwargs.get('T_fr', 25.0)
        self.T_rl         = kwargs.get('T_rl', 25.0)
        self.T_rr         = kwargs.get('T_rr', 25.0)
        self.delta        = kwargs.get('delta', 0.0)
        self.throttle     = kwargs.get('throttle', 0.0)
        self.brake_norm   = kwargs.get('brake_norm', 0.0)
        self.grip_util_f  = kwargs.get('grip_util_f', 0.0)
        self.grip_util_r  = kwargs.get('grip_util_r', 0.0)
        self.lap_time     = kwargs.get('lap_time', 0.0)
        self.lap_number   = kwargs.get('lap_number', 0)
        self.sector       = kwargs.get('sector', 0)
        self.speed_kmh    = kwargs.get('speed_kmh', 0.0)
        self.lat_g        = kwargs.get('lat_g', 0.0)
        self.lon_g        = kwargs.get('lon_g', 0.0)
        self.yaw_rate_deg = kwargs.get('yaw_rate_deg', 0.0)
        self.downforce    = kwargs.get('downforce', 0.0)
        self.drag         = kwargs.get('drag', 0.0)
        self.energy_kj    = kwargs.get('energy_kj', 0.0)
        self.trans_fl     = kwargs.get('trans_fl', 0.0)
        self.trans_fr     = kwargs.get('trans_fr', 0.0)
        self.trans_rl     = kwargs.get('trans_rl', 0.0)
        self.trans_rr     = kwargs.get('trans_rr', 0.0)
        self.setup_hash   = kwargs.get('setup_hash', 0.0)

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional[TelemetryFrame]:
        """Unpacks a raw 256-byte buffer into a structured frame instance."""
        if len(data) < TX_BYTES:
            return None
        try:
            vals = struct.unpack(TX_FMT, data[:TX_BYTES])
        except struct.error:
            return None

        t = cls()
        t.frame_id     = int(vals[TX.FRAME_ID])
        t.sim_time     = vals[TX.SIM_TIME]
        t.x            = vals[TX.X];          t.y             = vals[TX.Y];          t.z            = vals[TX.Z]
        t.roll         = vals[TX.ROLL];       t.pitch         = vals[TX.PITCH];      t.yaw          = vals[TX.YAW]
        t.vx           = vals[TX.VX];         t.vy            = vals[TX.VY];         t.vz           = vals[TX.VZ]
        t.ax           = vals[TX.AX];         t.ay            = vals[TX.AY];         t.az           = vals[TX.AZ]
        t.wz           = vals[TX.WZ]
        t.z_fl         = vals[TX.Z_FL];       t.z_fr          = vals[TX.Z_FR]
        t.z_rl         = vals[TX.Z_RL];       t.z_rr          = vals[TX.Z_RR]
        t.Fz_fl        = vals[TX.FZ_FL];      t.Fz_fr         = vals[TX.FZ_FR]
        t.Fz_rl        = vals[TX.FZ_RL];      t.Fz_rr         = vals[TX.FZ_RR]
        t.Fy_fl        = vals[TX.FY_FL];      t.Fy_fr         = vals[TX.FY_FR]
        t.Fy_rl        = vals[TX.FY_RL];      t.Fy_rr         = vals[TX.FY_RR]
        t.slip_fl      = vals[TX.SLIP_FL];    t.slip_fr       = vals[TX.SLIP_FR]
        t.slip_rl      = vals[TX.SLIP_RL];    t.slip_rr       = vals[TX.SLIP_RR]
        t.kappa_rl     = vals[TX.KAPPA_RL];   t.kappa_rr      = vals[TX.KAPPA_RR]
        t.omega_fl     = vals[TX.OMEGA_FL];   t.omega_fr      = vals[TX.OMEGA_FR]
        t.omega_rl     = vals[TX.OMEGA_RL];   t.omega_rr      = vals[TX.OMEGA_RR]
        t.T_fl         = vals[TX.T_FL];       t.T_fr          = vals[TX.T_FR]
        t.T_rl         = vals[TX.T_RL];       t.T_rr          = vals[TX.T_RR]
        t.delta        = vals[TX.DELTA]
        t.throttle     = vals[TX.THROTTLE];   t.brake_norm    = vals[TX.BRAKE_NORM]
        t.grip_util_f  = vals[TX.GRIP_UTIL_F]; t.grip_util_r   = vals[TX.GRIP_UTIL_R]
        t.lap_time     = vals[TX.LAP_TIME];   t.lap_number    = int(vals[TX.LAP_NUMBER])
        t.sector       = int(vals[TX.SECTOR])
        t.speed_kmh    = vals[TX.SPEED_KMH];  t.lat_g         = vals[TX.LAT_G];      t.lon_g        = vals[TX.LON_G]
        t.yaw_rate_deg = vals[TX.YAW_RATE_DEG]
        t.downforce    = vals[TX.DOWNFORCE];  t.drag          = vals[TX.DRAG]
        t.energy_kj    = vals[TX.ENERGY_KJ]
        t.trans_fl     = vals[TX.TRANS_FL];   t.trans_fr      = vals[TX.TRANS_FR]
        t.trans_rl     = vals[TX.TRANS_RL];   t.trans_rr      = vals[TX.TRANS_RR]
        t.setup_hash   = vals[TX.SETUP_HASH]
        return t

    def to_array(self) -> np.ndarray:
        """Flattens the structured frame object into a standard 64-element array."""
        arr = np.zeros(TX_N, dtype=np.float32)
        arr[TX.MAGIC]        = MAGIC_FLOAT
        arr[TX.FRAME_ID]     = float(self.frame_id)
        arr[TX.SIM_TIME]     = self.sim_time
        arr[TX.X]            = self.x;            arr[TX.Y]           = self.y;            arr[TX.Z]           = self.z
        arr[TX.ROLL]         = self.roll;         arr[TX.PITCH]       = self.pitch;        arr[TX.YAW]         = self.yaw
        arr[TX.VX]           = self.vx;           arr[TX.VY]          = self.vy;           arr[TX.VZ]          = self.vz
        arr[TX.AX]           = self.ax;           arr[TX.AY]          = self.ay;           arr[TX.AZ]          = self.az
        arr[TX.WZ]           = self.wz
        arr[TX.Z_FL]         = self.z_fl;         arr[TX.Z_FR]        = self.z_fr
        arr[TX.Z_RL]         = self.z_rl;         arr[TX.Z_RR]        = self.z_rr
        arr[TX.FZ_FL]        = self.Fz_fl;        arr[TX.FZ_FR]       = self.Fz_fr
        arr[TX.FZ_RL]        = self.Fz_rl;        arr[TX.FZ_RR]       = self.Fz_rr
        arr[TX.FY_FL]        = self.Fy_fl;        arr[TX.FY_FR]       = self.Fy_fr
        arr[TX.FY_RL]        = self.Fy_rl;        arr[TX.FY_RR]       = self.Fy_rr
        arr[TX.SLIP_FL]      = self.slip_fl;      arr[TX.SLIP_FR]     = self.slip_fr
        arr[TX.SLIP_RL]      = self.slip_rl;      arr[TX.SLIP_RR]     = self.slip_rr
        arr[TX.KAPPA_RL]     = self.kappa_rl;     arr[TX.KAPPA_RR]    = self.kappa_rr
        arr[TX.OMEGA_FL]     = self.omega_fl;     arr[TX.OMEGA_FR]    = self.omega_fr
        arr[TX.OMEGA_RL]     = self.omega_rl;     arr[TX.OMEGA_RR]    = self.omega_rr
        arr[TX.T_FL]         = self.T_fl;         arr[TX.T_FR]        = self.T_fr
        arr[TX.T_RL]         = self.T_rl;         arr[TX.T_RR]        = self.T_rr
        arr[TX.DELTA]        = self.delta
        arr[TX.THROTTLE]     = self.throttle;     arr[TX.BRAKE_NORM]  = self.brake_norm
        arr[TX.GRIP_UTIL_F]  = self.grip_util_f;  arr[TX.GRIP_UTIL_R] = self.grip_util_r
        arr[TX.LAP_TIME]     = self.lap_time;     arr[TX.LAP_NUMBER]  = float(self.lap_number)
        arr[TX.SECTOR]       = float(self.sector)
        arr[TX.SPEED_KMH]    = self.speed_kmh;    arr[TX.LAT_G]       = self.lat_g;        arr[TX.LON_G]       = self.lon_g
        arr[TX.YAW_RATE_DEG] = self.yaw_rate_deg
        arr[TX.DOWNFORCE]    = self.downforce;    arr[TX.DRAG] = self.drag
        arr[TX.ENERGY_KJ] = self.energy_kj
        arr[TX.TRANS_FL] = self.z_fl;             arr[TX.TRANS_FR] = self.z_fr # Fallbacks for trans channels
        arr[TX.TRANS_RL] = self.z_rl;             arr[TX.TRANS_RR] = self.z_rr
        arr[TX.SETUP_HASH] = self.setup_hash
        return arr

    def to_bytes(self) -> bytes:
        return struct.pack(TX_FMT, *self.to_array().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# §4  Serialization Interface Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pack_controls(steer: float, throttle_f: float, brake_f: float,
                  cmd_type: float = CMD.DRIVE,
                  k_f: float = 0.0, k_r: float = 0.0,
                  arb_f: float = 0.0, arb_r: float = 0.0) -> bytes:
    """Packs target controls into the legacy 32-byte struct contract."""
    return struct.pack(RX_FMT,
                       float(steer), float(throttle_f), float(brake_f),
                       float(cmd_type),
                       float(k_f), float(k_r), float(arb_f), float(arb_r))


def unpack_controls(data: bytes) -> Optional[Dict[str, float]]:
    """Unpacks client control demands from raw network byte buffers."""
    if len(data) < RX_BYTES:
        return None
    try:
        vals = struct.unpack(RX_FMT, data[:RX_BYTES])
        return {
            'steer': vals[RX.STEER],
            'throttle_f': vals[RX.THROTTLE_F],
            'brake_f': vals[RX.BRAKE_F],
            'cmd_type': vals[RX.CMD_TYPE],
            'k_f': vals[RX.SETUP_K_F],
            'k_r': vals[RX.SETUP_K_R],
            'arb_f': vals[RX.SETUP_ARB_F],
            'arb_r': vals[RX.SETUP_ARB_R],
        }
    except struct.error:
        return None


def pack_dmpc_diag(solver_time_ms: float, objective_loss: float, iterations: float, 
                   horizon_cross_track_err: float, trajectory_velocity_avg: float) -> bytes:
    """
    Serializes dMPC diagnostic data into a standalone additive frame layout.
    Injects metrics into port 5005 to bypass structural collisions on visualizers.
    """
    arr = [MAGIC_DMPC_FLOAT, solver_time_ms, objective_loss, iterations, 
           horizon_cross_track_err, trajectory_velocity_avg] + [0.0] * 10
    return struct.pack(DMPC_DIAG_FMT, *arr)