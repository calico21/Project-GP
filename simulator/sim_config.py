"""
simulator/sim_config.py
─────────────────────────────────────────────────────────────────────────────
Project-GP Simulator — Centralised Configuration

Single source of truth for:
  · Network ports and addresses
  · Physics timing constants
  · State vector index map (46-DOF / 54-DOF AWD)
  · Canonical 28-dim SuspensionSetup default
  · Drive-assist tuning parameters
  · Telemetry logging settings

Every other simulator module imports from here — no magic numbers.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# §1  Network Configuration
# ─────────────────────────────────────────────────────────────────────────────

HOST              = "127.0.0.1"

# UDP ports (physics server is the hub)
PORT_CTRL_RECV    = 5000   # server listens for control packets here
PORT_TELEM_VIZ    = 5001   # telemetry → Rerun / Godot visualizer
PORT_TELEM_CTRL   = 5002   # telemetry → control_interface HUD
PORT_TELEM_ROS    = 5003   # telemetry → ROS 2 bridge
PORT_TELEM_DEBUG  = 5004   # telemetry → debug tools

# WebSocket (dashboard bridge)
WS_HOST           = "0.0.0.0"
WS_PORT           = 8765   # WebSocket server for React dashboard

# ROS 2 bridge
ROS2_NODE_NAME    = "projectgp_bridge"
ROS2_TF_FRAME     = "base_link"
ROS2_WORLD_FRAME   = "world"


# ─────────────────────────────────────────────────────────────────────────────
# §2  Physics Timing
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS_HZ    = 200          # main loop rate
DT            = 1.0 / PHYSICS_HZ   # 5 ms per tick
SUBSTEPS      = 5            # H_net requires 1 ms substeps
DT_SUB        = DT / SUBSTEPS      # 1 ms — DO NOT increase

TELEMETRY_HZ  = 60           # WebSocket / dashboard update rate
TELEM_DECIMATION = PHYSICS_HZ // TELEMETRY_HZ  # send every N physics frames

LOG_HZ        = 10           # CSV log rate
LOG_DECIMATION = PHYSICS_HZ // LOG_HZ


# ─────────────────────────────────────────────────────────────────────────────
# §3  State Vector Index Map (46-DOF RWD)
# ─────────────────────────────────────────────────────────────────────────────
# Canonical index names for the 46-element state vector.
# Usage: state[S.VX] instead of state[14]

@dataclass(frozen=True)
class _StateIndices:
    # Generalised positions q (0..13)
    X:      int = 0
    Y:      int = 1
    Z:      int = 2
    ROLL:   int = 3
    PITCH:  int = 4
    YAW:    int = 5
    Z_FL:   int = 6
    Z_FR:   int = 7
    Z_RL:   int = 8
    Z_RR:   int = 9
    W_FL:   int = 10   # wheel spin angle
    W_FR:   int = 11
    W_RL:   int = 12
    W_RR:   int = 13

    # Generalised velocities p = q̇ (14..27)
    VX:     int = 14
    VY:     int = 15
    VZ:     int = 16
    WROLL:  int = 17   # roll rate
    WPITCH: int = 18   # pitch rate
    WYAW:   int = 19   # yaw rate
    VZ_FL:  int = 20
    VZ_FR:  int = 21
    VZ_RL:  int = 22
    VZ_RR:  int = 23
    WSPIN_FL: int = 24  # wheel spin rate (ω)
    WSPIN_FR: int = 25
    WSPIN_RL: int = 26
    WSPIN_RR: int = 27

    # Thermal states (28..37)
    T_FL_SURF: int = 28
    T_FL_BULK: int = 29
    T_FL_CORE: int = 30
    T_FR_SURF: int = 31
    T_FR_BULK: int = 32
    T_RR_SURF: int = 33
    T_RR_BULK: int = 34
    T_RL_SURF: int = 35
    T_RL_BULK: int = 36
    T_RL_CORE: int = 37

    # Transient slip states (38..45)
    KAPPA_FL: int = 38
    KAPPA_FR: int = 39
    KAPPA_RL: int = 40
    KAPPA_RR: int = 41
    ALPHA_FL: int = 42
    ALPHA_FR: int = 43
    ALPHA_RL: int = 44
    ALPHA_RR: int = 45

    STATE_DIM: int = 46

S = _StateIndices()


# ─────────────────────────────────────────────────────────────────────────────
# §4  Canonical 28-dim SuspensionSetup Default (matches SuspensionSetup.from_vector)
# ─────────────────────────────────────────────────────────────────────────────
# This MUST match the ordering in data/configs/vehicle_params.py
# build_default_setup_28() and SuspensionSetup.from_vector().

SETUP_PARAM_NAMES = [
    "k_f",            # [N/m]      front spring rate
    "k_r",            # [N/m]      rear spring rate
    "arb_f",          # [N·m/rad]  front anti-roll bar
    "arb_r",          # [N·m/rad]  rear anti-roll bar
    "c_low_f",        # [N·s/m]    front low-speed compression
    "c_low_r",        # [N·s/m]    rear low-speed compression
    "c_hi_f",         # [N·s/m]    front high-speed compression
    "c_hi_r",         # [N·s/m]    rear high-speed compression
    "v_knee_f",       # [m/s]      front knee velocity
    "v_knee_r",       # [m/s]      rear knee velocity
    "reb_f",          # [ratio]    front rebound/compression ratio
    "reb_r",          # [ratio]    rear rebound/compression ratio
    "h_ride_f",       # [m]        front ride height
    "h_ride_r",       # [m]        rear ride height
    "camber_f",       # [rad]      front static camber
    "camber_r",       # [rad]      rear static camber
    "toe_f",          # [rad]      front static toe
    "toe_r",          # [rad]      rear static toe
    "castor",         # [rad]      castor angle
    "anti_sq",        # [%]        anti-squat percentage
    "anti_dive_f",    # [%]        front anti-dive
    "anti_dive_r",    # [%]        rear anti-dive
    "anti_lift",      # [%]        anti-lift percentage
    "diff_lock",      # [0-1]      differential lock ratio
    "brake_bias",     # [0-1]      front brake bias
    "h_cg",           # [m]        CG height
    "bs_f",           # [N/m]      front bump stop rate
    "bs_r",           # [N/m]      rear bump stop rate
]

SETUP_DIM = len(SETUP_PARAM_NAMES)  # 28

DEFAULT_SETUP_28 = np.array([
    35000.0,   # k_f
    38000.0,   # k_r
    800.0,     # arb_f
    600.0,     # arb_r
    2500.0,    # c_low_f
    2800.0,    # c_low_r
    900.0,     # c_hi_f
    1000.0,    # c_hi_r
    0.15,      # v_knee_f
    0.15,      # v_knee_r
    1.5,       # reb_f
    1.5,       # reb_r
    0.030,     # h_ride_f
    0.035,     # h_ride_r
    -0.026,    # camber_f  (≈ −1.5°)
    -0.017,    # camber_r  (≈ −1.0°)
    0.0017,    # toe_f     (≈ 0.1° out)
    -0.0009,   # toe_r     (≈ −0.05° in)
    0.070,     # castor    (≈ 4°)
    0.40,      # anti_sq   (40%)
    0.25,      # anti_dive_f
    0.30,      # anti_dive_r
    0.15,      # anti_lift
    0.30,      # diff_lock
    0.58,      # brake_bias (58% front)
    0.285,     # h_cg
    150000.0,  # bs_f
    150000.0,  # bs_r
], dtype=np.float32)

assert len(DEFAULT_SETUP_28) == SETUP_DIM, \
    f"DEFAULT_SETUP_28 has {len(DEFAULT_SETUP_28)} elements, expected {SETUP_DIM}"


def setup_28_to_sim8(setup_28: np.ndarray) -> np.ndarray:
    """
    Extract the 8 parameters that vehicle.simulate_step() reads from the
    canonical 28-dim SuspensionSetup vector.

    simulate_step() indexes: [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias]
    at positions [0, 1, 2, 3, 4, 5, 6, 7] of the array it receives.

    In the 28-dim canonical vector:
      h_cg       lives at index 25 (not 6!)
      brake_bias lives at index 24 (not 7!)

    Without this adapter, simulate_step reads c_hi_f (900) as h_cg and
    c_hi_r (1000) as brake_bias — the car thinks its CG is at 900 metres.
    """
    return np.array([
        setup_28[0],    # k_f       → sim[0]
        setup_28[1],    # k_r       → sim[1]
        setup_28[2],    # arb_f     → sim[2]
        setup_28[3],    # arb_r     → sim[3]
        setup_28[4],    # c_low_f   → sim[4] (c_f)
        setup_28[5],    # c_low_r   → sim[5] (c_r)
        setup_28[25],   # h_cg      → sim[6]  ← index 25 in 28-dim!
        setup_28[24],   # brake_bias→ sim[7]  ← index 24 in 28-dim!
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Preset Setups (for keyboard shortcuts 1-4)
# ─────────────────────────────────────────────────────────────────────────────

def _modify_setup(base: np.ndarray, **overrides) -> np.ndarray:
    """Return a copy of base with named parameter overrides."""
    s = base.copy()
    for name, val in overrides.items():
        idx = SETUP_PARAM_NAMES.index(name)
        s[idx] = val
    return s

PRESET_SETUPS = {
    "1_soft": _modify_setup(DEFAULT_SETUP_28,
        k_f=25000, k_r=28000, arb_f=500, arb_r=400,
        c_low_f=1800, c_low_r=2000),
    "2_balanced": DEFAULT_SETUP_28.copy(),
    "3_stiff": _modify_setup(DEFAULT_SETUP_28,
        k_f=45000, k_r=48000, arb_f=1200, arb_r=900,
        c_low_f=3500, c_low_r=3800),
    "4_optimised": DEFAULT_SETUP_28.copy(),  # placeholder — load from MORL Pareto
}


# ─────────────────────────────────────────────────────────────────────────────
# §6  Drive-Assist Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriveAssistConfig:
    tc_enabled:      bool  = True
    tc_kappa_limit:  float = 0.18    # slip ratio threshold
    tc_ramp_rate:    float = 0.85    # throttle reduction per excess kappa

    abs_enabled:     bool  = True
    abs_slip_limit:  float = 0.12    # target slip ratio under braking
    abs_release:     float = 0.60    # brake pressure multiplier when ABS intervenes

    dsc_enabled:     bool  = False
    dsc_yaw_gain:    float = 500.0   # Nm per (rad/s error)
    dsc_yaw_deadband: float = 0.05  # rad/s — no correction below this


# ─────────────────────────────────────────────────────────────────────────────
# §7  Vehicle Constants (for telemetry extraction — read from vehicle_params)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleConstants:
    """Subset of vehicle_params needed for telemetry derivations."""
    total_mass:    float = 300.0     # [kg]
    tire_radius:   float = 0.2032    # [m]   Hoosier 16x7.5-10
    wheelbase:     float = 1.55      # [m]
    lf:            float = 0.8525    # [m]   front axle to CG
    lr:            float = 0.6975    # [m]   rear axle to CG
    track_f:       float = 1.22      # [m]   front track width
    track_r:       float = 1.18      # [m]   rear track width
    h_cg:          float = 0.285     # [m]   CG height
    mu_peak:       float = 1.65      # [-]   peak friction coefficient

    @classmethod
    def from_dict(cls, d: dict) -> 'VehicleConstants':
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ─────────────────────────────────────────────────────────────────────────────
# §8  Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────

import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# CSV log column headers (must match telemetry extraction order)
LOG_COLUMNS = [
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
    "energy_kj",
    "downforce", "drag",
    "lap_time", "lap_number", "sector",
    "tc_active", "abs_active",
]