# Project-GP: Human-in-the-Loop Simulator Roadmap

## Objective
To build a zero-latency, human-in-the-loop driving simulator to validate the JAX/Flax `Diff-WMPC` digital twin. The architecture relies on a **Decoupled Client-Server Model**: the physics and optimal control run entirely in a headless Python environment, communicating via UDP to an Unreal Engine 5 (UE5) "dumb renderer" client.

---

## Phase 1: JAX Physics Server Optimization & Telemetry Validation
**Goal:** Ensure the 14-DOF physics step can execute well within real-time constraints and visualize it without UE5 overhead.

* [ ] **AOT Compilation Profiling:** Benchmark `DifferentiableMultiBodyVehicle.simulate_step` using `jax.jit`. Target execution time: `< 2.5ms` (to support a 400Hz physics loop).
* [ ] **Headless Server Loop:** Write a highly optimized Python `while True:` loop that ticks the physics engine continuously using a fixed `dt`.
* [ ] **Rapid Visualization (The "Quick Win"):** Stream the spatial outputs (`[x, y, z, roll, pitch, yaw]`) to an immediate-mode visualizer like Rerun.io or Foxglove to visually confirm the JAX integration is stable over long horizons.
* [ ] **Aligning Torque Extraction:** Ensure the pneumatic trail and $M_z$ (aligning torque) at the front tires are explicitly calculated and accessible in the output state array for future Force Feedback (FFB).

## Phase 2: UDP Bridge Development
**Goal:** Establish a bi-directional, microsecond-latency network bridge between Python and C++.

* [ ] **Python UDP Broadcaster:** Develop a socket script in Python that broadcasts the 46-dimensional vehicle state (specifically chassis transforms and wheel heave) at 200Hz-400Hz.
* [ ] **Python UDP Listener:** Set up a non-blocking listener in Python to receive the `[steer, throttle, brake]` packet array.
* [ ] **UE5 C++ Receiver:** Create an Unreal Engine C++ Actor/Component that listens to the UDP port and parses the incoming coordinate packets without stalling the game thread.
* [ ] **Latency Auditing:** Measure the round-trip time (RTT) of the local UDP loopback. Target RTT: `< 1ms`.

## Phase 3: Unreal Engine 5 Client Setup (The "Dumb" Renderer)
**Goal:** Map the incoming Python physics data to 3D meshes in UE5 and route hardware inputs back to Python.

* [ ] **Bypass Chaos Vehicles:** Create an empty UE5 project. Ensure the default Chaos physics engine is disabled or ignored for the vehicle actor.
* [ ] **Transform Mapping:** In the C++ Tick function, map the parsed UDP chassis coordinates to `SetActorLocationAndRotation`. Map the wheel suspension displacements to the relative transforms of the four tire meshes.
* [ ] **Hardware Input Routing:** Use UE5's RawInput plugin to read the physical steering wheel and pedal axes. Format these as a standardized float array (`[-1.0 to 1.0]`) and send them via UDP to the Python server.
* [ ] **Visual Smoothing:** Implement light linear interpolation (Lerp) on the UE5 side to match the 400Hz physics tick to the monitor's refresh rate (e.g., 144Hz) to eliminate micro-stutters.

## Phase 4: Force Feedback (FFB) Integration
**Goal:** Close the physical loop by sending tire saturation and torque data to the steering wheel.

* [ ] **UDP Payload Expansion:** Append the JAX-calculated $M_z$ (aligning torque) to the telemetry packet sent from Python to UE5.
* [ ] **Wheel SDK Integration:** Implement the specific C++ SDK for your hardware (Logitech Steering Wheel SDK, Fanatec API, or Simucube API) inside UE5.
* [ ] **Torque Mapping:** Route the $M_z$ value to the SDK's constant force/spring effects.
* [ ] **Tuning:** Calibrate the FFB multiplier so the Matérn 5/2 GP grip drop-off physically translates into steering weight loss (understeer feel) at the rim.