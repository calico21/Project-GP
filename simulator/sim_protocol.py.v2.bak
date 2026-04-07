"""
simulator/sim_protocol.py
─────────────────────────────────────────────────────────────────────────────
Shared UDP wire protocol between the physics server and all clients.

Design goals
────────────
· Self-describing — every packet starts with a magic byte + version.
· Extensible — new channels added without breaking old clients.
· Low-latency — fixed-size structs, no serialisation overhead.
· Bi-directional — server → clients (telemetry), clients → server (commands).

Packet layout (server → client)   TX_FMT   64 floats = 256 bytes
─────────────────────────────────────────────────────────────────────────────
 idx   name             units    description
   0   magic            —        Always 0xCAFE (as float32 cast for alignment)
   1   frame_id         count    Monotonic frame counter (detects dropped packets)
   2   sim_time         s        Simulation clock since start
   3   x                m        Global X position
   4   y                m        Global Y position
   5   z                m        Global Z (CG height)
   6   roll             rad      Chassis roll
   7   pitch            rad      Chassis pitch
   8   yaw              rad      Chassis yaw (heading)
   9   vx               m/s      Longitudinal velocity (body frame)
  10   vy               m/s      Lateral velocity (body frame)
  11   vz               m/s      Vertical velocity (body frame)
  12   ax               m/s²     Longitudinal acceleration
  13   ay               m/s²     Lateral acceleration
  14   az               m/s²     Vertical acceleration
  15   wz               rad/s    Yaw rate
  16   z_fl             m        Front-left suspension travel
  17   z_fr             m        Front-right suspension travel
  18   z_rl             m        Rear-left suspension travel
  19   z_rr             m        Rear-right suspension travel
  20   Fz_fl            N        Front-left wheel load
  21   Fz_fr            N        Front-right wheel load
  22   Fz_rl            N        Rear-left wheel load
  23   Fz_rr            N        Rear-right wheel load
  24   Fy_fl            N        Front-left lateral force
  25   Fy_fr            N        Front-right lateral force
  26   Fy_rl            N        Rear-left lateral force
  27   Fy_rr            N        Rear-right lateral force
  28   slip_fl          rad      Front-left slip angle
  29   slip_fr          rad      Front-right slip angle
  30   slip_rl          rad      Rear-left slip angle
  31   slip_rr          rad      Rear-right slip angle
  32   kappa_rl         —        Rear-left longitudinal slip ratio
  33   kappa_rr         —        Rear-right longitudinal slip ratio
  34   omega_fl         rad/s    Front-left wheel spin
  35   omega_fr         rad/s    Front-right wheel spin
  36   omega_rl         rad/s    Rear-left wheel spin
  37   omega_rr         rad/s    Rear-right wheel spin
  38   T_tire_fl        °C       Front-left tire surface temperature
  39   T_tire_fr        °C       Front-right tire surface temperature
  40   T_tire_rl        °C       Rear-left tire surface temperature
  41   T_tire_rr        °C       Rear-right tire surface temperature
  42   delta            rad      Steering angle
  43   throttle         [0,1]    Throttle demand
  44   brake_norm       [0,1]    Brake demand (normalised)
  45   grip_util_f      [0,1]    Front friction utilisation (|F|/μFz)
  46   grip_util_r      [0,1]    Rear friction utilisation
  47   lap_time         s        Current lap time (0 if timing not active)
  48   lap_number       count    Completed laps
  49   sector           count    Current sector (0-indexed)
  50   speed_kmh        km/h     Speed (for HUD)
  51   lat_g            G        Lateral G (ay/9.81)
  52   lon_g            G        Longitudinal G (ax/9.81)
  53   yaw_rate_deg     deg/s    Yaw rate in deg/s (for HUD)
  54   downforce_total  N        Total aerodynamic downforce
  55   drag_total       N        Total aerodynamic drag
  56   energy_consumed  kJ       Cumulative energy consumed
  57   trans_fl         m        Front-left transient slip state
  58   trans_fr         m        Front-right transient slip state
  59   trans_rl         m        Rear-left transient slip state
  60   trans_rr         m        Rear-right transient slip state
  61   setup_hash       —        Hash of current setup (detects setup changes)
  62   _reserved_1      —        Reserved for future use
  63   _reserved_2      —        Reserved for future use

Packet layout (client → server)   RX_FMT   8 floats = 32 bytes
─────────────────────────────────────────────────────────────────────────────
  0   steer_cmd        rad      Steering angle command
  1   throttle_force   N        Throttle force (positive)
  2   brake_force      N        Brake force (positive)
  3   cmd_type         enum     0=drive, 1=reset, 2=setup_change, 3=pause
  4   setup_k_f        N/m      Setup override: front spring (only if cmd_type=2)
  5   setup_k_r        N/m      Setup override: rear spring
  6   setup_arb_f      N·m/rad  Setup override: front ARB
  7   setup_arb_r      N·m/rad  Setup override: rear ARB
"""

import struct
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Magic number ─────────────────────────────────────────────────────────────
MAGIC         = 0xCAFE
MAGIC_FLOAT   = float(struct.unpack('<f', struct.pack('<I', MAGIC))[0])

# ── Network defaults ─────────────────────────────────────────────────────────
DEFAULT_HOST        = '127.0.0.1'
DEFAULT_PORT_RECV   = 5000   # server listens for controls
DEFAULT_PORT_SEND   = 5001   # server sends telemetry → visualizer
DEFAULT_PORT_CTRL   = 5002   # server sends telemetry copy → control interface HUD

# ── Struct formats ────────────────────────────────────────────────────────────
TX_N      = 64
RX_N      = 8
TX_FMT    = f'<{TX_N}f'
RX_FMT    = f'<{RX_N}f'
TX_BYTES  = struct.calcsize(TX_FMT)   # 256 bytes
RX_BYTES  = struct.calcsize(RX_FMT)   # 32 bytes

# ── Named indices for TX packet ───────────────────────────────────────────────
class TX:
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
    Z_FR           = 17
    Z_RL           = 18
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


# ── Named indices for RX packet ───────────────────────────────────────────────
class RX:
    STEER         = 0
    THROTTLE_F    = 1
    BRAKE_F       = 2
    CMD_TYPE      = 3
    SETUP_K_F     = 4
    SETUP_K_R     = 5
    SETUP_ARB_F   = 6
    SETUP_ARB_R   = 7


# ── Command type enum ─────────────────────────────────────────────────────────
class CMD:
    DRIVE         = 0.0
    RESET         = 1.0
    SETUP_CHANGE  = 2.0
    PAUSE         = 3.0
    RESUME        = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry packet dataclass (decoded TX)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TelemetryFrame:
    frame_id     : int   = 0
    sim_time     : float = 0.0
    x            : float = 0.0
    y            : float = 0.0
    z            : float = 0.0
    roll         : float = 0.0
    pitch        : float = 0.0
    yaw          : float = 0.0
    vx           : float = 0.0
    vy           : float = 0.0
    vz           : float = 0.0
    ax           : float = 0.0
    ay           : float = 0.0
    az           : float = 0.0
    wz           : float = 0.0
    z_fl         : float = 0.0
    z_fr         : float = 0.0
    z_rl         : float = 0.0
    z_rr         : float = 0.0
    Fz_fl        : float = 0.0
    Fz_fr        : float = 0.0
    Fz_rl        : float = 0.0
    Fz_rr        : float = 0.0
    Fy_fl        : float = 0.0
    Fy_fr        : float = 0.0
    Fy_rl        : float = 0.0
    Fy_rr        : float = 0.0
    slip_fl      : float = 0.0
    slip_fr      : float = 0.0
    slip_rl      : float = 0.0
    slip_rr      : float = 0.0
    kappa_rl     : float = 0.0
    kappa_rr     : float = 0.0
    omega_fl     : float = 0.0
    omega_fr     : float = 0.0
    omega_rl     : float = 0.0
    omega_rr     : float = 0.0
    T_fl         : float = 25.0
    T_fr         : float = 25.0
    T_rl         : float = 25.0
    T_rr         : float = 25.0
    delta        : float = 0.0
    throttle     : float = 0.0
    brake_norm   : float = 0.0
    grip_util_f  : float = 0.0
    grip_util_r  : float = 0.0
    lap_time     : float = 0.0
    lap_number   : int   = 0
    sector       : int   = 0
    speed_kmh    : float = 0.0
    lat_g        : float = 0.0
    lon_g        : float = 0.0
    yaw_rate_deg : float = 0.0
    downforce    : float = 0.0
    drag         : float = 0.0
    energy_kj    : float = 0.0
    trans_fl     : float = 0.0
    trans_fr     : float = 0.0
    trans_rl     : float = 0.0
    trans_rr     : float = 0.0
    setup_hash   : float = 0.0

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional['TelemetryFrame']:
        if len(data) < TX_BYTES:
            return None
        vals = struct.unpack(TX_FMT, data[:TX_BYTES])
        # Basic magic check (allow small float error)
        t = cls()
        t.frame_id     = int(vals[TX.FRAME_ID])
        t.sim_time     = vals[TX.SIM_TIME]
        t.x            = vals[TX.X];    t.y     = vals[TX.Y];     t.z     = vals[TX.Z]
        t.roll         = vals[TX.ROLL]; t.pitch = vals[TX.PITCH]; t.yaw   = vals[TX.YAW]
        t.vx           = vals[TX.VX];   t.vy    = vals[TX.VY];    t.vz    = vals[TX.VZ]
        t.ax           = vals[TX.AX];   t.ay    = vals[TX.AY];    t.az    = vals[TX.AZ]
        t.wz           = vals[TX.WZ]
        t.z_fl         = vals[TX.Z_FL]; t.z_fr  = vals[TX.Z_FR]
        t.z_rl         = vals[TX.Z_RL]; t.z_rr  = vals[TX.Z_RR]
        t.Fz_fl        = vals[TX.FZ_FL]; t.Fz_fr = vals[TX.FZ_FR]
        t.Fz_rl        = vals[TX.FZ_RL]; t.Fz_rr = vals[TX.FZ_RR]
        t.Fy_fl        = vals[TX.FY_FL]; t.Fy_fr = vals[TX.FY_FR]
        t.Fy_rl        = vals[TX.FY_RL]; t.Fy_rr = vals[TX.FY_RR]
        t.slip_fl      = vals[TX.SLIP_FL]; t.slip_fr = vals[TX.SLIP_FR]
        t.slip_rl      = vals[TX.SLIP_RL]; t.slip_rr = vals[TX.SLIP_RR]
        t.kappa_rl     = vals[TX.KAPPA_RL]; t.kappa_rr = vals[TX.KAPPA_RR]
        t.omega_fl     = vals[TX.OMEGA_FL]; t.omega_fr = vals[TX.OMEGA_FR]
        t.omega_rl     = vals[TX.OMEGA_RL]; t.omega_rr = vals[TX.OMEGA_RR]
        t.T_fl         = vals[TX.T_FL]; t.T_fr  = vals[TX.T_FR]
        t.T_rl         = vals[TX.T_RL]; t.T_rr  = vals[TX.T_RR]
        t.delta        = vals[TX.DELTA]
        t.throttle     = vals[TX.THROTTLE]; t.brake_norm = vals[TX.BRAKE_NORM]
        t.grip_util_f  = vals[TX.GRIP_UTIL_F]; t.grip_util_r = vals[TX.GRIP_UTIL_R]
        t.lap_time     = vals[TX.LAP_TIME]; t.lap_number = int(vals[TX.LAP_NUMBER])
        t.sector       = int(vals[TX.SECTOR])
        t.speed_kmh    = vals[TX.SPEED_KMH]; t.lat_g = vals[TX.LAT_G]; t.lon_g = vals[TX.LON_G]
        t.yaw_rate_deg = vals[TX.YAW_RATE_DEG]
        t.downforce    = vals[TX.DOWNFORCE]; t.drag = vals[TX.DRAG]
        t.energy_kj    = vals[TX.ENERGY_KJ]
        t.trans_fl     = vals[TX.TRANS_FL]; t.trans_fr = vals[TX.TRANS_FR]
        t.trans_rl     = vals[TX.TRANS_RL]; t.trans_rr = vals[TX.TRANS_RR]
        t.setup_hash   = vals[TX.SETUP_HASH]
        return t

    def to_array(self) -> np.ndarray:
        arr = np.zeros(TX_N, dtype=np.float32)
        arr[TX.MAGIC]        = MAGIC_FLOAT
        arr[TX.FRAME_ID]     = float(self.frame_id)
        arr[TX.SIM_TIME]     = self.sim_time
        arr[TX.X]            = self.x;  arr[TX.Y]   = self.y;   arr[TX.Z]   = self.z
        arr[TX.ROLL]         = self.roll; arr[TX.PITCH] = self.pitch; arr[TX.YAW] = self.yaw
        arr[TX.VX]           = self.vx; arr[TX.VY]  = self.vy;  arr[TX.VZ]  = self.vz
        arr[TX.AX]           = self.ax; arr[TX.AY]  = self.ay;  arr[TX.AZ]  = self.az
        arr[TX.WZ]           = self.wz
        arr[TX.Z_FL]         = self.z_fl; arr[TX.Z_FR] = self.z_fr
        arr[TX.Z_RL]         = self.z_rl; arr[TX.Z_RR] = self.z_rr
        arr[TX.FZ_FL]        = self.Fz_fl; arr[TX.FZ_FR] = self.Fz_fr
        arr[TX.FZ_RL]        = self.Fz_rl; arr[TX.FZ_RR] = self.Fz_rr
        arr[TX.FY_FL]        = self.Fy_fl; arr[TX.FY_FR] = self.Fy_fr
        arr[TX.FY_RL]        = self.Fy_rl; arr[TX.FY_RR] = self.Fy_rr
        arr[TX.SLIP_FL]      = self.slip_fl; arr[TX.SLIP_FR] = self.slip_fr
        arr[TX.SLIP_RL]      = self.slip_rl; arr[TX.SLIP_RR] = self.slip_rr
        arr[TX.KAPPA_RL]     = self.kappa_rl; arr[TX.KAPPA_RR] = self.kappa_rr
        arr[TX.OMEGA_FL]     = self.omega_fl; arr[TX.OMEGA_FR] = self.omega_fr
        arr[TX.OMEGA_RL]     = self.omega_rl; arr[TX.OMEGA_RR] = self.omega_rr
        arr[TX.T_FL]         = self.T_fl; arr[TX.T_FR] = self.T_fr
        arr[TX.T_RL]         = self.T_rl; arr[TX.T_RR] = self.T_rr
        arr[TX.DELTA]        = self.delta
        arr[TX.THROTTLE]     = self.throttle; arr[TX.BRAKE_NORM] = self.brake_norm
        arr[TX.GRIP_UTIL_F]  = self.grip_util_f; arr[TX.GRIP_UTIL_R] = self.grip_util_r
        arr[TX.LAP_TIME]     = self.lap_time; arr[TX.LAP_NUMBER] = float(self.lap_number)
        arr[TX.SECTOR]       = float(self.sector)
        arr[TX.SPEED_KMH]    = self.speed_kmh; arr[TX.LAT_G] = self.lat_g; arr[TX.LON_G] = self.lon_g
        arr[TX.YAW_RATE_DEG] = self.yaw_rate_deg
        arr[TX.DOWNFORCE]    = self.downforce; arr[TX.DRAG] = self.drag
        arr[TX.ENERGY_KJ]    = self.energy_kj
        arr[TX.TRANS_FL]     = self.trans_fl; arr[TX.TRANS_FR] = self.trans_fr
        arr[TX.TRANS_RL]     = self.trans_rl; arr[TX.TRANS_RR] = self.trans_rr
        arr[TX.SETUP_HASH]   = self.setup_hash
        return arr

    def to_bytes(self) -> bytes:
        return struct.pack(TX_FMT, *self.to_array().tolist())


def pack_controls(steer: float, throttle_f: float, brake_f: float,
                  cmd_type: float = CMD.DRIVE,
                  k_f: float = 0.0, k_r: float = 0.0,
                  arb_f: float = 0.0, arb_r: float = 0.0) -> bytes:
    return struct.pack(RX_FMT,
                       float(steer), float(throttle_f), float(brake_f),
                       float(cmd_type),
                       float(k_f), float(k_r), float(arb_f), float(arb_r))


def unpack_controls(data: bytes):
    if len(data) < RX_BYTES:
        return None
    vals = struct.unpack(RX_FMT, data[:RX_BYTES])
    return {
        'steer'     : vals[RX.STEER],
        'throttle_f': vals[RX.THROTTLE_F],
        'brake_f'   : vals[RX.BRAKE_F],
        'cmd_type'  : vals[RX.CMD_TYPE],
        'k_f'       : vals[RX.SETUP_K_F],
        'k_r'       : vals[RX.SETUP_K_R],
        'arb_f'     : vals[RX.SETUP_ARB_F],
        'arb_r'     : vals[RX.SETUP_ARB_R],
    }