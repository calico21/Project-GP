# Project-GP Simulator v2

Real-time FSAE digital twin: 200Hz JAX physics + live 3D visualisation.

---

## Architecture

```
┌─────────────────────────┐     UDP 256B        ┌──────────────────────┐
│  physics_server.py      │ ──────────────────→ │ visualizer_client.py │
│  200Hz JAX integration  │ ← ─ ─ ─ ─ ─ ─ ─ ─  │  Rerun 3D + HUD      │
└─────────────────────────┘     UDP 32B          └──────────────────────┘
           ↑
           │ UDP 32B (controls)
┌─────────────────────────┐
│  control_interface.py   │
│  Keyboard / Gamepad /   │
│  Autopilot              │
└─────────────────────────┘
```

All three components run as **separate processes** on the same machine.

---

## Quick Start

### Terminal 1 — Physics Server
```bash
source ~/project_gp_env/bin/activate
export ACADOS_SOURCE_DIR=~/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/acados/lib

python simulator/physics_server.py --track fsg_autocross
```

### Terminal 2 — Visualizer
```bash
source ~/project_gp_env/bin/activate
pip install rerun-sdk  # first time only

python simulator/visualizer_client.py --track fsg_autocross
```

### Terminal 3 — Controls
```bash
source ~/project_gp_env/bin/activate
# pip install pygame  # gamepad input (optional)

python simulator/control_interface.py --mode keyboard
```

---

## Controls (Keyboard)

| Key       | Action                    |
|-----------|---------------------------|
| W / ↑     | Throttle                  |
| S / ↓     | Brake                     |
| A / ←     | Steer left                |
| D / →     | Steer right               |
| R         | Reset car to start        |
| P         | Pause / Resume            |
| 1         | Setup: Soft/Understeery   |
| 2         | Setup: Balanced (default) |
| 3         | Setup: Stiff/Oversteery   |
| 4         | Setup: Optimiser #1       |
| Q / ESC   | Quit                      |

---

## Available Tracks

| Name             | Description                          | Length |
|------------------|--------------------------------------|--------|
| `fsg_autocross`  | FS Germany autocross (3 sectors)     | ~750m  |
| `skidpad`        | FSAE skidpad (R=9.125m circles)      | ~200m  |
| `endurance_lap`  | Generic 1km endurance lap            | ~1000m |
| `acceleration`   | 75m straight-line acceleration       | 80m    |

---

## Autopilot Mode

The autopilot uses pure-pursuit path following:

```bash
python simulator/control_interface.py --mode autopilot
```

It automatically adjusts speed for corners and follows the track centreline.
Useful for benchmarking setup changes without driver variation.

---

## Drive Assists

Controlled via server command-line flags:

```bash
# All assists on (default)
python simulator/physics_server.py

# Disable traction control (raw throttle)
python simulator/physics_server.py --no-tc

# Disable ABS (raw braking)
python simulator/physics_server.py --no-abs

# Enable Differential Stability Control
python simulator/physics_server.py --dsc
```

---

## Hot Setup Changes

Send a setup command from the control interface (keys 1–4) or directly:

```python
from simulator.sim_protocol import pack_controls, CMD
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pkt  = pack_controls(0, 0, 0,
                     cmd_type=CMD.SETUP_CHANGE,
                     k_f=35000, k_r=38000,
                     arb_f=1000, arb_r=600)
sock.sendto(pkt, ('127.0.0.1', 5000))
```

No server restart required.

---

## Telemetry Analysis

After a session:

```bash
# Single session
python simulator/telemetry_analysis.py simulator/logs/telemetry_20260304_120000.csv

# Compare two setups
python simulator/telemetry_analysis.py \
    simulator/logs/telemetry_setup_A.csv \
    --compare simulator/logs/telemetry_setup_B.csv \
    --label "Setup A (soft)" --label2 "Setup B (stiff)"
```

Outputs:
- Speed + G-force trace (`*_speed_g_temp.png`)
- G-G diagram (`*_gg_diagram.png`)
- Suspension + wheel loads (`*_suspension.png`)
- XY trajectory map (`*_trajectory.png`)
- Setup comparison overlay (`comparison_*.png`)

---

## UDP Protocol

The server broadcasts 256-byte packets at 200Hz containing:

- Position (X, Y, Z, roll, pitch, yaw)
- Velocities (vx, vy, vz, wz)
- Accelerations (ax, ay, az) — all in body frame
- Per-wheel: Fz, Fy, slip angle α, longitudinal slip κ, temperature T
- Grip utilisation (front & rear)
- Lap timing (lap_time, lap_number, sector)
- Aerodynamic downforce and drag
- Cumulative energy consumption
- Active drive assist flags

Full protocol spec: `sim_protocol.py`

---

## File Structure

```
simulator/
├── __init__.py
├── physics_server.py       # 200Hz JAX physics + telemetry extraction
├── visualizer_client.py    # Rerun 3D visualizer + HUD
├── control_interface.py    # Keyboard / Gamepad / Autopilot
├── sim_protocol.py         # Shared UDP wire protocol
├── lap_timer.py            # Real-time lap & sector timing
├── track_builder.py        # Procedural track generation
├── telemetry_analysis.py   # Post-session analysis + plots
├── logs/                   # Auto-generated telemetry CSVs
└── README.md
```
