# Project-GP Simulator v3

Real-time FSAE digital twin: 200Hz JAX physics + live dashboard + ROS 2 + 3D visualization.

---

## Architecture

```
                                                  ┌──────────────────────────┐
                                          ┌──WS──→│ React Dashboard (Vercel) │
                                          │       │ useLiveTelemetry.js      │
┌─────────────────────────┐    ┌──────────┴───┐   └──────────────────────────┘
│  physics_server.py      │    │ ws_bridge.py  │
│  200Hz JAX integration  │──→ │ UDP→WebSocket │   ┌──────────────────────────┐
│  46-DOF Port-Hamiltonian│    └──────────────┘   │ Godot 3D Visualizer      │
│  + 28-dim setup space   │                       │ (driverless team)        │
└────────┬────────────────┘──UDP 256B────────────→└──────────────────────────┘
         │                           │
         │                    ┌──────┴────────┐   ┌──────────────────────────┐
         │                    │ ros2_bridge.py │──→│ Driverless Stack         │
         │                    │ UDP→ROS 2      │   │ nav2 / perception       │
         │                    └───────────────┘   └──────────────────────────┘
         │
         ↑ UDP 32B (controls)
┌────────┴────────────────┐
│  control_interface.py   │
│  Keyboard / Gamepad /   │
│  Autopilot / ROS 2      │
└─────────────────────────┘
```

All components run as **separate processes**. The physics server is the single
source of truth — every client receives the same 64-float telemetry packet.

---

## Quick Start

### Terminal 1 — Physics Server
```bash
source ~/project_gp_env/bin/activate
python simulator/physics_server.py --track fsg_autocross
```

### Terminal 2 — WebSocket Bridge (for React Dashboard)
```bash
pip install websockets
python simulator/ws_bridge.py
```
Then open the dashboard at `https://project-gp-ter26.vercel.app` and toggle **LIVE** mode.

### Terminal 3 — Controls
```bash
python simulator/control_interface.py --mode keyboard
```

### Optional: ROS 2 Bridge (for Driverless Team)
```bash
source /opt/ros/humble/setup.bash
python simulator/ros2_bridge.py
```

### Optional: Godot Visualizer
Open the Godot project in `visualizer/godot/` and press Play.
The visualizer listens on UDP port 5001 for the same telemetry packets.

---

## File Structure

```
simulator/
├── sim_config.py           # Centralised configuration (ports, state indices, setup)
├── sim_protocol.py         # 256-byte UDP wire protocol (64 floats)
├── physics_server.py       # 200Hz JAX physics engine + telemetry broadcast
├── ws_bridge.py            # UDP → WebSocket relay for React dashboard
├── ros2_bridge.py          # UDP → ROS 2 topics for driverless stack
├── control_interface.py    # Keyboard / Gamepad / Autopilot input
├── track_builder.py        # Procedural FSAE track generation
├── lap_timer.py            # Lap & sector timing
├── telemetry_recorder.py   # High-fidelity CSV/NPZ recording for validation
├── debug_sim.py            # Diagnostic tests
├── logs/                   # Auto-generated telemetry CSVs
└── README.md               # This file
```

Dashboard integration:
```
visualization/dashboard_react/src/
├── hooks/
│   └── useLiveTelemetry.js  # WebSocket hook — live data for all modules
└── context/
    └── SelectionContext.jsx  # Shared state bus (existing)
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
| 1         | Setup: Soft/Understeer    |
| 2         | Setup: Balanced (default) |
| 3         | Setup: Stiff/Oversteer    |
| 4         | Setup: MORL Optimised     |
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

## Data Flow

### Live Mode (Dashboard)

```
physics_server.py  ──200Hz UDP──→  ws_bridge.py  ──60Hz WS──→  React
                                                                 │
  useLiveTelemetry() hook provides:                              │
    • frame        — latest telemetry dict                       │
    • history      — ring buffer (last 600 frames = 10 sec)      │
    • connected    — boolean                                     │
    • sendControl  — (steer, throttle, brake) → physics server   │
    • resetCar     — reset to start line                         │
    • applySetup   — hot-swap 28-dim setup                       │
```

### Analyse Mode (Dashboard)

Modules load from static CSV files (simulation dumps or real telemetry).
The `DataAdapter` layer abstracts the source — modules don't know or care
whether data comes from WebSocket, CSV, or seeded RNG.

### ROS 2 Integration

```
physics_server.py ──UDP──→ ros2_bridge.py ──→ /vehicle/odom        (Odometry)
                                          ──→ /vehicle/imu         (Imu)
                                          ──→ /vehicle/joint_states (JointState)
                                          ──→ /vehicle/wheel_loads  (Float32MultiArray)
                                          ──→ /vehicle/slip         (Float32MultiArray)
                                          ──→ /vehicle/tire_temps   (Float32MultiArray)
                                          ──→ /tf                   (world → base_link)

driverless stack ──→ /cmd_vel (Twist) ──→ ros2_bridge.py ──UDP──→ physics_server.py
```

### Godot Integration

Godot receives the same 256-byte UDP packets as the Rerun visualizer.
Implementation in GDScript:

```gdscript
var udp := PacketPeerUDP.new()

func _ready():
    udp.bind(5001)

func _process(_delta):
    while udp.get_available_packet_count() > 0:
        var pkt = udp.get_packet()
        var vals = pkt.to_float32_array()  # 64 floats
        # vals[3]=x, vals[4]=y, vals[5]=z, vals[6]=roll, vals[7]=pitch, vals[8]=yaw
        $Car.global_position = Vector3(vals[3], vals[5], vals[4])
        $Car.rotation = Vector3(vals[6], vals[8], vals[7])
        # Wheel suspension: vals[16..19] = z_fl/fr/rl/rr
```

---

## Configuration

All constants are in `sim_config.py`:

| Parameter          | Default | Description                      |
|--------------------|---------|----------------------------------|
| `PHYSICS_HZ`      | 200     | Physics loop rate                |
| `SUBSTEPS`         | 5       | Substeps per tick (1ms each)     |
| `TELEMETRY_HZ`    | 60      | WebSocket update rate            |
| `LOG_HZ`          | 10      | CSV logging rate                 |
| `WS_PORT`         | 8765    | WebSocket bridge port            |
| `PORT_CTRL_RECV`  | 5000    | Physics server control port      |
| `PORT_TELEM_VIZ`  | 5001    | Telemetry → visualizer port      |
| `PORT_TELEM_CTRL` | 5002    | Telemetry → control HUD port     |
| `PORT_TELEM_ROS`  | 5003    | Telemetry → ROS 2 bridge port    |

Setup presets are in `sim_config.PRESET_SETUPS` — all use the canonical
28-dim `SuspensionSetup` vector (matching MORL optimizer output).

---

## Changes from v2

1. **28-dim setup** — `DEFAULT_SETUP` now matches the canonical `SuspensionSetup.from_vector()` ordering. The v2 8-dim shortcut silently produced wrong physics.

2. **Fixed ABS** — ABS now reads wheel spin rates from correct state indices (`S.WSPIN_FL..RR` = 24-27), not thermal states (34-35).

3. **WebSocket bridge** — `ws_bridge.py` relays 200Hz UDP to 60Hz JSON WebSocket for the React dashboard. Bidirectional: dashboard can send control commands back.

4. **ROS 2 bridge** — `ros2_bridge.py` publishes standard ROS 2 messages (Odometry, IMU, JointState, TF). Subscribes to `/cmd_vel` for driverless stack control.

5. **Centralised config** — `sim_config.py` is the single source of truth for all ports, indices, and parameters. No more magic numbers scattered across files.

6. **Removed legacy files** — `udp_server.py` (v1, 11-float protocol) and `dummy_client.py` (v1 test client) are removed. `SIM_ROADMAP.md` (UE5 fantasy) replaced by this README.

7. **State index safety** — Named indices (`S.VX`, `S.WYAW`, `S.KAPPA_FL`) replace raw integer offsets throughout. Eliminates the class of bugs where index N means different things in different files.