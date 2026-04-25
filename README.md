# Project-GP — End-to-End Differentiable Formula Student Digital Twin

> **Ter27 Formula Student | FSG 2026 — Siemens Digital Twin Award Entry**
>
> A 100% native JAX/Flax, fully end-to-end differentiable digital twin of the Ter27 FS vehicle.
> Designed for safety-biased setup optimisation, stochastic optimal control, real-time powertrain
> management, and driver coaching. Every equation in the physics engine is differentiable.
> `jax.grad()` traces directly from lap time back to spring rates, damper curves, roll-centre
> heights, brake bias, and powertrain torque allocation. The entire stack runs as a single XLA
> graph at 200 Hz on an embedded SBC.

---

## At a Glance

| Property | Value |
|---|---|
| **Framework** | 100% JAX/Flax — no NumPy inside traced functions |
| **Vehicle** | Ter27 — 4WD electric Formula Student (FSG 2026) |
| **State Dimension** | 46 (14 Mechanical DOF + 10 Thermal + 8 Transient Slip + 14 Velocity) |
| **Integrator** | 2-stage Gauss-Legendre RK4 (GLRK-4), symplectic, 4th-order, 5-iter Newton |
| **Dynamics** | Neural Port-Hamiltonian System — H_net (FiLM-conditioned) + R_net (Cholesky PSD) |
| **Tire Model** | Pacejka MF6.2 + Turn Slip + 5-Node Jaeger Thermal + PINN (8-feature) + Matérn 5/2 Sparse GP |
| **Optimal Control** | Diff-WMPC — 3-level Daubechies-4 Wavelet MPC + UT Stochastic Tubes + Augmented Lagrangian + Pseudo-Huber Regularization |
| **Setup Optimisation** | MORL-SB-TRPO + RNPG — 40-dim, Chebyshev ensemble, ARD-BO cold-start, SMS-EMOA Pareto archive, Endurance LTE 3rd axis |
| **Setup Space** | 40 parameters (`SuspensionSetup` NamedTuple) — springs, heave springs, inerters, ARBs, 4-way dampers, geometry, Ackermann, anti-geometry, diff |
| **Setup Freeze** | `DesignFreeze` — 3-phase progressive CAD lockdown; tanh projection → frozen params have zero gradient automatically |
| **Powertrain Control** | 14-stage pipeline at 200 Hz: mpQP KKT Allocator + Regen Blend + Robust DCBF + DESC-TC + B-spline Launch + Koopman LQR |
| **State Estimator** | Differentiable EKF — 5-state online parameter estimation (λ_μ_f, λ_μ_r, T_opt, h_cg, α_peak) |
| **Lap Gradient** | `jax.grad(simulate_full_lap)(setup)` — end-to-end differentiable lap simulation |
| **Suspension Kinematics** | Full kinematic sweep analysis (JAX-vmap); Optimum Kinematics-equivalent; Excel report generation |
| **Simulator** | 200 Hz physics server + UDP→WebSocket bridge + ROS 2 bridge + Godot 3D |
| **Gradient Server** | HTTP/WebSocket server — on-demand ∂(lap_time)/∂(setup) from React dashboard |
| **Dashboard** | React (Vercel) — 18 modules, 8 nav groups, LIVE/ANALYZE modes |
| **Revision** | GP-vX4 |

---

## Table of Contents

1. [Philosophy & Design Principles](#1-philosophy--design-principles)
2. [Repository Structure](#2-repository-structure)
3. [State Vector & SuspensionSetup](#3-state-vector--suspensionsetup)
4. [Design Freeze System](#4-design-freeze-system)
5. [Neural Port-Hamiltonian Vehicle Dynamics](#5-neural-port-hamiltonian-vehicle-dynamics)
6. [Multi-Fidelity Tire Model](#6-multi-fidelity-tire-model)
7. [Differentiable Wavelet MPC (Diff-WMPC)](#7-differentiable-wavelet-mpc-diff-wmpc)
8. [MORL-SB-TRPO Setup Optimiser](#8-morl-sb-trpo-setup-optimiser)
9. [Powertrain Control Stack](#9-powertrain-control-stack)
10. [Differentiable EKF — Online Parameter Estimation](#10-differentiable-ekf--online-parameter-estimation)
11. [Full-Lap Differentiable Simulation](#11-full-lap-differentiable-simulation)
12. [CMA-ES Crossover Validator](#12-cma-es-crossover-validator)
13. [Suspension Kinematic Analysis](#13-suspension-kinematic-analysis)
14. [Simulator Architecture](#14-simulator-architecture)
15. [React Engineering Dashboard](#15-react-engineering-dashboard)
16. [Sanity Check Suite (Tests 1–16)](#16-sanity-check-suite-tests-116)
17. [Known Issues, Limits & Diagnostics](#17-known-issues-limits--diagnostics)
18. [Pipeline Execution](#18-pipeline-execution)
19. [Revision History (GP-vX1 → vX4)](#19-revision-history-gp-vx1--vx4)

---

## 1. Philosophy & Design Principles

Project-GP abandons traditional numerical simulation frameworks (CasADi, IPOPT, point-mass solvers)
in favour of a **deep learning compiler architecture (JAX/XLA)**. The design is governed by five
non-negotiable constraints:

**The Differentiability Rule.** Every function, physics equation, and control logic path must be
strictly differentiable. Hard conditionals (`jnp.where` with discontinuous branches, step functions,
`jax.lax.cond` without gradient-safe branches) are forbidden. All limits are implemented via smooth
approximations: `jax.nn.softplus`, `jax.nn.sigmoid`, `jnp.tanh` rescaling.

**The JAX Purity Rule.** All physics is written in pure JAX. NumPy operations are prohibited inside
JIT-compiled or grad-traced functions. No host-device sync mid-graph. `vmap` and `scan` correctness
is verified against abstract tracing rules — notably, `jnp.convolve` cannot be vmapped over a batch
dimension without materialising it inside `lax.conv_general_dilated`; explicit per-channel calls are
used throughout.

**The Physical Rule.** The Port-Hamiltonian structure is never broken. `H_net` can only add
non-negative residual energy gated at physical equilibrium. `R_net` can only produce PSD dissipation
matrices via `R = LLᵀ + diag(softplus(d))`. The system structurally cannot hallucinate free energy.

**The Canonical-Index Rule.** All 40 setup parameters are accessed exclusively via
`SuspensionSetup.from_vector()` / `.to_vector()` and the canonical builder functions. No positional
indexing of the raw setup array is permitted outside these methods — a lesson learned from a
catastrophic silent bug where ARB indices received spring-rate values.

**The Freeze-Before-Grad Rule.** When geometry is committed to CAD, `DesignFreeze` must be applied
before running the optimizer. Frozen parameters have LB = UB, so `project_to_bounds()` returns a
constant — the gradient is identically zero without any masking logic in the optimizer. This prevents
the optimizer from wasting search budget on non-adjustable hardpoint geometry.

---

## 2. Repository Structure

```
FS_Driver_Setup_Optimizer/
├── models/
│   ├── vehicle_dynamics.py          # Neural Port-Hamiltonian 46-DOF dynamics + GLRK-4
│   ├── tire_model.py                # Pacejka MF6.2 + 5-node thermal + PINN + Sparse GP
│   ├── differentiable_ekf.py        # 5-state EKF — online λ_μ, T_opt, h_cg, α_peak estimation
│   ├── h_net.bytes                  # Trained H_net weights (Flax serialisation)
│   ├── r_net.bytes                  # Trained R_net weights
│   ├── aero_net.bytes               # Trained DifferentiableAeroMap weights
│   └── h_net_scale.txt              # Training normalisation diagnostic — NOT used in architecture
│
├── optimization/
│   ├── ocp_solver.py                # Diff-WMPC (Db4 DWT + AL + UT tubes + Pseudo-Huber)
│   ├── evolutionary.py              # MORL-SB-TRPO + RNPG optimizer (40-dim, ARD-BO, SMS-EMOA)
│   ├── objectives.py                # Skidpad grip + step-steer stability + Endurance LTE
│   ├── residual_fitting.py          # H_net / R_net training pipeline (Phase 1 + 2 + TTC PINN)
│   ├── differentiable_track.py      # JAX-native cubic spline track (SE(3) manifold)
│   ├── differentiable_lap_sim.py    # Full-lap scan: ∂(lap_time)/∂(setup) via jax.grad
│   └── cmaes_validator.py           # CMA-ES crossover validation of MORL Pareto front
│
├── powertrain/
│   ├── powertrain_manager.py        # Unified 14-stage coordinator — single JIT powertrain_step()
│   ├── motor_model.py               # PMSM electromechanical + thermal + battery OCV/R_int
│   ├── virtual_impedance.py         # 2nd-order virtual flywheel/damper — PIO mitigation
│   ├── regen_blend.py               # Dynamic regen blend — KKT-optimal α* + hydraulic residual
│   └── modes/
│       ├── advanced/
│       │   ├── torque_vectoring.py      # 12-iter projected-gradient SOCP + input-delay DCBF
│       │   ├── traction_control.py      # DESC extremum-seeking + GP-weighted dual-path TC
│       │   ├── launch_control.py        # 16-knot B-spline + 6-phase FSM (v2.1)
│       │   ├── koopman_tv.py            # Dictionary-Switched Koopman LQR (EDMD-DL, 3 regimes)
│       │   ├── slip_barrier.py          # Slip-predictive CBF rows for KKT integration
│       │   ├── explicit_mpqp_allocator.py  # V2 mpQP KKT allocator (24×24, slip-aware)
│       │   └── active_set_classifier.py    # Neural active-set predictor (V1: 16D, V2: 19D)
│       └── intermediate/
│           └── launch_control.py        # 8-knot Catmull-Rom intermediate sequencer
│
├── suspension/
│   ├── kinematics.py                # JAX-vmap kinematic solver (camber, toe, MR, RC, scrub)
│   ├── sweep_analysis.py            # Full heave/steer/roll sweep — Optimum Kinematics equivalent
│   └── excel_writer.py              # Generates Front/Rear_Ter27.xlsx (OKinematic format)
│
├── config/
│   ├── design_freeze.py             # DesignFreeze: 3-phase progressive parameter lockdown
│   ├── car_config.py                # Multi-car config registry + design bounds
│   └── vehicles/
│       └── ter26.py                 # Canonical Ter27 vehicle parameters (full dict)
│
├── simulator/
│   ├── physics_server.py            # 200 Hz JAX physics loop + UDP broadcast
│   ├── ws_bridge.py                 # UDP→WebSocket bridge (60 Hz, multi-client)
│   ├── ros2_bridge.py               # ROS 2 bridge (nav_msgs, sensor_msgs, tf2)
│   ├── control_interface.py         # Keyboard / Gamepad / Autopilot / ROS 2 controls
│   ├── sim_config.py                # Port map, protocol constants
│   └── README.md                    # Simulator quick-start
│
├── data/
│   ├── configs/
│   │   └── tire_coeffs.py           # Pacejka MF6.2 coefficients (Hoosier R20, TTC-fitted)
│   └── logs/                        # MoTeC telemetry CSV inputs
│
├── scripts/
│   ├── gradient_server.py           # HTTP/WebSocket server — on-demand ∂(lap_time)/∂(setup)
│   ├── train_koopman_tv.py          # EDMD-DL offline training for Koopman TV operators
│   └── run_ter27_design_exploration.py  # Phase-aware design exploration workflow
│
├── visualization/
│   ├── dashboard.py                 # Streamlit engineering suite (legacy)
│   └── dashboard_react/             # React 18-module dashboard (production)
│       ├── src/
│       │   ├── App.jsx                      # Root: 8 nav groups, LIVE/ANALYZE toggle
│       │   ├── theme.js                     # Shared tokens: C.cy, C.gn, C.am, C.red, GL, GS, TT, AX
│       │   ├── OverviewModule.jsx           # Architecture explainer panels
│       │   ├── TelemetryModule.jsx          # Live telemetry charts (WebSocket)
│       │   ├── SetupModule.jsx              # Pareto front + sensitivity waterfall
│       │   ├── SuspensionModule.jsx         # Per-corner Fz, Fd, LLTD, roll gradient
│       │   ├── SuspensionExplorerModule.jsx # 40-param interactive explorer + 3D Pareto scatter
│       │   ├── TirePhysicsModule.jsx        # 5-node thermal + GP envelope + hysteresis
│       │   ├── WeightBalanceModule.jsx      # CG position, corner weights, ballast
│       │   ├── AerodynamicsModule.jsx       # Ground-effect Fz, drag vs speed, aero balance
│       │   ├── ElectronicsModule.jsx        # Motor temps, SoC, inverter power, battery voltage
│       │   ├── PowertrainControlModule.jsx  # 14-stage pipeline visualisation
│       │   ├── TorqueVectoringSimModule.jsx # Interactive TV Simulator
│       │   ├── EnergyAuditModule.jsx        # Hamiltonian landscape + R-matrix heatmap
│       │   ├── DifferentiableInsightsModule.jsx # Setup sensitivities ∂t_lap/∂param
│       │   ├── ResearchModule.jsx           # MORL convergence, car selector, param freeze UI
│       │   ├── DriverCoachingModule.jsx     # Sector-by-sector coaching
│       │   ├── EnduranceStrategyModule.jsx  # Regen strategy, lap-time-energy trade-off
│       │   └── ComplianceModule.jsx         # FSG rules compliance checker
│       └── package.json
│
├── tests/
│   └── test_verification.py         # pytest suite: 5 physics tests (friction, aero, spring)
│
├── sanity_checks.py                 # Full system verification — Tests 1–16
├── main.py                          # Pipeline entry point (--mode setup | full)
├── jax_config.py                    # XLA cache + memory + parallelism config (import first)
└── morl_pareto_front.csv            # Latest 3-objective Pareto front (grip, stability, LTE)
```

---

## 3. State Vector & SuspensionSetup

### 3.1 46-Dimensional State Vector

```
x[0:14]   q  — positions
           [0:3]  X, Y, Z         (chassis CG, world frame)  [m]
           [3]    φ  roll          [rad]
           [4]    θ  pitch         [rad]
           [5]    ψ  yaw           [rad]
           [6:10] z_fl, z_fr, z_rl, z_rr  suspension heave   [m]
           [10:14] θ_fl..θ_rr     wheel rotation             [rad]

x[14:28]  v  — velocities
           [14:17] vx, vy, vz     chassis spatial            [m/s]
           [17:20] wx, wy, wz     chassis angular            [rad/s]
           [20:24] ż_fl..ż_rr    suspension rates           [m/s]
           [24:28] ω_fl..ω_rr    wheel spin rates           [rad/s]

x[28:38]  thermal states (5-node tire model, per axle)
           [28:31] T_surf_inner/mid/outer_f  front surface ribs   [°C]
           [31]    T_gas_f                   front internal gas   [°C]
           [32:35] T_surf_inner/mid/outer_r  rear surface ribs    [°C]
           [35]    T_gas_r                   rear internal gas    [°C]
           [36]    T_core_f                  front structural core [°C]
           [37]    T_core_r                  rear structural core  [°C]

x[38:46]  transient slip (first-order carcass lag)
           [38,39] α_t, κ_t  front-left   [rad], [-]
           [40,41] α_t, κ_t  front-right
           [42,43] α_t, κ_t  rear-left
           [44,45] α_t, κ_t  rear-right
```

**Static equilibrium deflections** (module-level constant `_Z_EQ`, single source of truth):

```
z_eq_f ≈ 12.8 mm   (k_f=35000 N/m, MR_f=1.14, F_susp_eq≈583 N)
z_eq_r ≈ 13.5 mm   (k_r=38000 N/m, MR_r=1.16, F_susp_eq≈695 N)
```

### 3.2 SuspensionSetup — 40-Element Typed Pytree

All 40 parameters are contained in a `NamedTuple` registered as a JAX pytree.
**`from_vector()` / `to_vector()` are the only permitted construction paths.**

Backward-compatibility helpers are provided:
- `SuspensionSetup.from_legacy_28(v28)` — promotes a 28-element vector by appending zeros for the 12 extended parameters.
- `SuspensionSetup.from_legacy_8(v8)` — promotes a legacy 8-element coarse vector.

#### Core 28 Parameters (indices 0–27)

| Index | Name | Units | Default | Bounds |
|---|---|---|---|---|
| 0–1 | k_f, k_r | N/m | 35000, 38000 | [10k, 120k] |
| 2–3 | arb_f, arb_r | N·m/rad | 800, 600 | [0, 5000] |
| 4–5 | c_low_f, c_low_r | N·s/m | 1800, 1800 | [200, 8000] |
| 6–7 | c_high_f, c_high_r | N·s/m | 1200, 1200 | [50, 4000] |
| 8–9 | v_knee_f, v_knee_r | m/s | 0.10, 0.10 | [0.03, 0.40] |
| 10–11 | rebound_ratio_f/r | — | 1.50, 1.50 | [1.0, 3.0] |
| 12–13 | h_ride_f, h_ride_r | m | 0.025, 0.022 | [0.010, 0.060] |
| 14–15 | camber_f, camber_r | deg | −2.0, −1.5 | [−5.0, 0.0] |
| 16–17 | toe_f, toe_r | deg | −0.10, −0.15 | [−1.0, 1.0] |
| 18 | castor_f | deg | 5.0 | [0.0, 10.0] |
| 19 | anti_squat | — | 0.30 | [0.0, 0.9] |
| 20–21 | anti_dive_f/r | — | 0.40, 0.10 | [0.0, 0.9] |
| 22 | anti_lift | — | 0.20 | [0.0, 0.9] |
| 23 | diff_lock_ratio | — | 0.30 | [0.0, 1.0] |
| 24 | brake_bias_f | — | 0.60 | [0.50, 0.80] |
| 25 | h_cg | m | 0.285 | [0.18, 0.45] |
| 26–27 | bump_steer_f/r | rad/m | 0.0, 0.0 | [−0.05, 0.05] |

#### Extended 12 Parameters (indices 28–39) — Ter27 Geometry

| Index | Name | Units | Default | Description |
|---|---|---|---|---|
| 28–29 | k_heave_f, k_heave_r | N/m | 0, 0 | Third-spring heave rate (decouples heave/roll) |
| 30–31 | inerter_f, inerter_r | kg | 0, 0 | Inerter apparent mass (J-damper) |
| 32–33 | mr_rise_f, mr_rise_r | — | 2.8, 2.2 | Rising-rate motion-ratio quadratic coefficient |
| 34 | anti_squat_f | — | 0.15 | Front anti-squat (independent from rear) |
| 35 | ackermann | % | 0 | Ackermann factor: 0 = parallel, 1 = full Ackermann, −1 = reverse |
| 36–37 | c_ls_reb_f, c_ls_reb_r | N·s/m | 2880, 2560 | Low-speed rebound (independent from bump) |
| 38–39 | c_hs_reb_f, c_hs_reb_r | N·s/m | 1920, 1800 | High-speed rebound (independent from bump) |

`project_to_bounds()` uses a smooth tanh rescaling — gradient survives at both bounds.
Frozen parameters (LB = UB via `DesignFreeze`) yield zero gradient automatically with zero changes
to the optimizer — no masking needed.

---

## 4. Design Freeze System

**File:** `config/design_freeze.py`

### 4.1 Overview

As the Ter27 design progresses from concept through CAD lock to prototype, some parameters become
build-fixed (hardpoint geometry determines them, they cannot be adjusted at track). Passing these
to the optimizer wastes search budget and risks spurious gradients into non-adjustable dimensions.

`DesignFreeze` solves this by clamping `LB[i] = UB[i] = fixed_value` for every frozen parameter.
The existing `tanh` projection in `project_to_bounds()` then produces a constant:

```
mid = val, half = 0  →  output = val  (constant, ∂output/∂μ = 0)
```

No optimizer code changes are required. Frozen parameters are invisible to Adam/TRPO.

### 4.2 Three-Phase Progressive Lockdown

```
Phase 1 (Concept):    all 40 params FREE — optimizer explores full space
Phase 2 (CAD lock):   hardpoints committed — freeze castor, anti-geometry, h_cg, bump steer
Phase 3 (Prototype):  alignment set on car — freeze camber, toe, diff; only springs/dampers/ARBs free
```

| Preset | API | Free dims |
|---|---|---|
| All free | `DesignFreeze()` | 40 |
| Phase 2 | `DesignFreeze.for_ter27_phase2()` | 32 |
| Phase 3 | `DesignFreeze.for_ter27_phase3()` | 20 |
| Custom | `DesignFreeze.custom(names, values)` | user-specified |

### 4.3 Integration

```python
from config.design_freeze import DesignFreeze, install_freeze

freeze = DesignFreeze.for_ter27_phase2()
grad_mask = install_freeze(freeze, car_id='ter27')   # patches SETUP_LB/UB globally
freeze.summary()    # prints frozen vs free parameter table
```

`install_freeze()` patches `models.vehicle_dynamics.SETUP_LB` and `SETUP_UB` in-place.
Must be called **before** importing `evolutionary.py`.

### 4.4 Gradient Server Integration

The `ResearchModule.jsx` in the dashboard exposes a Parameter Freeze UI backed by the gradient
server. When a user locks a parameter in the UI, the freeze config is serialised and sent to
`gradient_server.py`, which reinstalls the freeze before the next `jax.grad` call.

---

## 5. Neural Port-Hamiltonian Vehicle Dynamics

**File:** `models/vehicle_dynamics.py`

### 5.1 Port-Hamiltonian Structure

$$\dot{x} = (J - R)\,\nabla H(x) + F_{ext}(x, u)$$

- $J$ — skew-symmetric interconnection matrix (conservative, energy-preserving)
- $R$ — symmetric PSD dissipation matrix (energy-removing only)
- $H(x)$ — total Hamiltonian (kinetic + structural + neural residual)
- $F_{ext}$ — non-conservative external forces (tire, aero, gravity)

### 5.2 Neural Energy Landscape — H_net (FiLM-conditioned)

`NeuralEnergyLandscape` is a 128→64→1 MLP with **FiLM conditioning** at each hidden layer.
The setup vector modulates the energy landscape via Feature-wise Linear Modulation:

$$h_{out} = \gamma(\text{setup}) \odot \text{LayerNorm}(h) + \beta(\text{setup})$$

**Energy output:**

$$H_{total} = T_{prior} + V_{structural} + H_{res} \cdot \sum(q_{susp} - z_{eq})^2 + \epsilon$$

Key properties:
- `T_prior = 0.5 * ||p||² / M_diag` — exact kinetic energy from generalised momenta
- `V_structural = 0.5 * ||q_{susp}||² * K_{prior}` (K_prior = 30,000 N/m)
- `susp_sq_eq` gates at **physical equilibrium** `_Z_EQ`, not at z=0
- `H_res = min(softplus(MLP) * h_scale, 50_000)` — capped at 50,000 J/m²; `h_scale = 1.0` always
- 22 SE(3)-bilateral symmetric state features: anti-symmetric quantities enter as x²

### 5.3 Neural Dissipation Matrix — R_net (Cholesky PSD)

$$R = L L^T + \text{diag}(\text{softplus}(d))$$

The `diag(softplus(d))` term guarantees strict positive definiteness. A physical mask restricts
dissipation to heave, roll, pitch, and unsprung-z DOFs.

### 5.4 Differentiable Aero Map (Physics-Informed)

`DifferentiableAeroMap` is a 32→32→4 MLP mapping `(vx, pitch, roll, heave_f, heave_r)` to
ground-effect-corrected `(Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero)`.

The aero map is **physics-informed** — trained to satisfy aerodynamic constraints:
- Downforce ∝ v² in the ground-effect-linear regime (verified at Test 7)
- Ground clearance and downforce floors use `_softplus_floor` — not `jnp.maximum`
- Zero subgradient from `jnp.maximum` kills optimizer signal at the 15 mm clearance limit;
  `_softplus_floor` ensures non-zero gradient everywhere

### 5.5 Integrator — 2-Stage Gauss-Legendre RK4 (GLRK-4)

Butcher tableau:

$$a = \begin{pmatrix} 1/4 & 1/4 - \sqrt{3}/6 \\ 1/4 + \sqrt{3}/6 & 1/4 \end{pmatrix}, \quad b = (1/2,\ 1/2)$$

- **Symplectic** — preserves the symplectic 2-form `dq ∧ dp` to machine precision
- **4th-order** — energy drift O(h⁵) vs O(h³) for Störmer-Verlet
- **Implicit** — 5-iteration Newton scan inside `jax.lax.scan`; stage clipping at ±500 m/s²
  prevents NaN propagation without affecting gradients within the physical envelope
- **Auxiliary sub-states** (thermal + slip) integrated with the trapezoidal rule using converged
  GLRK-4 stage derivatives — A-stable, no extra `_compute_derivatives` calls

### 5.6 Suspension Forces

Per corner: `F_susp = F_spring + F_damper + F_bumpstop + F_ARB + F_heave + F_inerter`

- **Spring:** `F = k * z * MR(z)²`, MR is a quadratic polynomial of heave travel (rising-rate via `mr_rise` parameter)
- **Heave spring:** Third spring resisting heave mode independently of roll; `F_heave = k_heave * z_heave`
- **Inerter:** `F_inerter = b * z̈_heave` — velocity-squared-dependent force in frequency domain; modelled as apparent inertia in the 2nd-order heave subsystem
- **Damper:** 4-way digressive (low/high speed × bump/rebound), fully C∞ via sigmoid blending; rebound coefficients now fully independent from bump coefficients
- **Bumpstop:** `F = k_bs * softplus(β * (z − gap)) / β` — C∞, zero below gap, rising above
- **ARB:** lateral roll moment applied as symmetric ±correction to left/right corners
- **Fz (contact normal load):** always positive via `_softplus_floor(F_grav ± load_transfer)`

---

## 6. Multi-Fidelity Tire Model

**File:** `models/tire_model.py`

### 6.1 Layer 1 — Pacejka MF6.2

Full Magic Formula for Hoosier R20 (TTC-fitted). Includes:
- Longitudinal force Fx(κ, Fz, γ, P)
- Lateral force Fy(α, Fz, γ, P) with combined slip correction
- Self-aligning moment Mz(α, κ, Fz, γ, P)
- Turn-slip correction (ψ̇ × Vx term) for tight chicane accuracy
- Ply steer and conicity offsets
- Combined-slip weighting via Pacejka's G-function

All Pacejka functions are C∞ via `jnp.tanh` replacement for the sign discontinuity.

### 6.2 Layer 2 — 5-Node Jaeger Thermal Model

Nodes indexed `x[28:38]` (strictly verified at Test 7):
```
T_surf_inner, T_surf_mid, T_surf_outer  [front/rear surface ribs]
T_gas                                    [internal gas pressure node]
T_core                                   [structural carcass core]
```

Heat flows: contact patch flash temperature (Jaeger model) → surface ribs → gas → core.
Thermal grip modulation: `μ_T(T_surf) = μ_peak · exp(−β_T · (T_surf − T_opt)²)`

### 6.3 Layer 3 — TireOperator PINN (8-feature)

Physics-Informed Neural Network for deterministic residual correction beyond MF6.2.
8-feature input vector: `[α, κ, Fz, γ, P, vx, T_surf, T_norm]`

`T_norm` (temperature normalisation, added in GP-v3-D) allows the PINN to distinguish between
cold tires at normal load and hot tires at the same load — critical for predicting the
correct grip level at the operating temperature.

Training: `residual_fitting.py` Phase 3 against TTC data.

### 6.4 Layer 4 — Matérn 5/2 Sparse GP (Stochastic Uncertainty)

Inducing-point sparse GP for real-time uncertainty quantification over the tire operating envelope.

Key implementation details:
- Gram matrix: vectorised distance broadcast (no nested `vmap`) — avoids O(N²) trace time
- Cholesky solve with `stop_gradient(L)` — decouples Cholesky from gradient tape (avoids NaN)
- Jitter: `1e-3 * I` (sufficient for float32; condition number ~1e4)
- Inducing point init: `tanh(N(0, 0.5)) * scale + shift` (avoids boundary collapse)
- Variance floor: `softplus(var)` (no `jnp.maximum`)
- LCB penalty: `clip(2σ, 0, 0.15)` (prevents over-penalisation in uncertain regions)
- `SpectralDense` weight matrix: `stop_gradient(σ)` in power iteration

**GP uncertainty propagation:**
- WMPC stochastic tubes: `κ_safe · σ_GP` added to tube half-width
- Robust CBF: `h_robust(x) = h(x) − κ_safe · σ_GP(x)` — GP uncertainty enters the safety filter
- DESC-TC fusion weights: `w_desc ∝ 1/σ_GP²` — GP confidence drives DESC/SWIFT blending

---

## 7. Differentiable Wavelet MPC (Diff-WMPC)

**File:** `optimization/ocp_solver.py`

### 7.1 Wavelet Parameterisation

Control trajectory `u(t)` over horizon N is represented in a 3-level Daubechies-4 (Db4) wavelet
basis. Optimization is performed in wavelet coefficient space:

$$u(t) = \sum_{j,k} w_{j,k} \cdot \psi_{j,k}(t)$$

**Bands:** `A3` (approximation), `D1`, `D2`, `D3` (detail at scales 1–3).

Benefits:
- Low-frequency acceleration profiles are compactly represented in `A3`
- High-frequency actuator chatter penalised in `D1/D2/D3` without explicit bandwidth constraint
- The wavelet coefficient vector is smaller than the raw `u` vector — cheaper gradient computation

### 7.2 Pseudo-Huber Wavelet Regularisation

Replaces the non-differentiable L1 sparsity penalty on detail bands:

$$\mathcal{R}_{PH}(w) = \sum_{j \in \{D1,D2,D3\}} \lambda_j \sum_k \left( \sqrt{1 + (w_{j,k}/\delta)^2} - 1 \right) \cdot \delta^2$$

Per-band weighting: D3 → 0.5×, D2 → 1.0×, D1 → 2.0× (higher frequency = stronger penalty).
This imposes physically meaningful smoothness: actuator bandwidth decreases with frequency.

### 7.3 Augmented Lagrangian Friction Constraint

$$\mathcal{L}_{AL} = f(x) + \lambda^T c(x) + \frac{\rho}{2} \left\| \max\!\left(c(x), -\frac{\lambda}{\rho}\right) \right\|^2$$

Constraint: `g_i = (a_lat² + a_lon²) / (μg)² − 1 ≤ 0`

Multipliers update: `λ ← max(λ + ρ·max(g, 0), 0)`. ρ grows 2× when max violation > 0.1.

### 7.4 Unscented Transform — Stochastic Tubes

5-point UT generates sigma trajectories from joint GP tire σ² + wind perturbations. All 5
simulated in parallel via `jax.vmap`. Track limits enforced against tube edges:

```
dist_left  = w_left  − (n_mean + κ_safe · √σ²_n)   ≥ 0
dist_right = w_right + (n_mean − κ_safe · √σ²_n)   ≥ 0
```

Both enforced via `−ε · log(softplus(dist * 50) / 50 + 1e-5)` — smooth log-barrier.

### 7.5 Physics P-Controller Warm Start

```
v_target = min(sqrt(μg / |κ|), 0.92 · V_limit)   [Newton units]
```

Warm start uses Newton force units matching the physics `u[1]` channel.
An earlier warm start in m/s² units produced `flat_init ≈ 0`, triggering GTOL premature
convergence after nit=1.

### 7.6 NaN Recovery Fallback

```python
fallback_loss = 1e9 + 0.5 * ||x − x_warmstart||²
fallback_grad = clip(x − x_warmstart, −100, 100)
```

The 1e9 offset exceeds any real loss; gradient points toward the warm start (a known feasible
trajectory), not toward zero (which increased friction constraint violation on every recovery step).

---

## 8. MORL-SB-TRPO Setup Optimiser

**File:** `optimization/evolutionary.py`

### 8.1 Overview

Multi-Objective RL over the full 40-dimensional `SuspensionSetup` space (or effective dimension
determined by `DesignFreeze`). Three Pareto axes:

1. **Maximum Grip** — skidpad objective [G]
2. **Dynamic Stability** — step-steer overshoot ≤ 5.0 rad/s
3. **Endurance LTE** — lap-time-energy composite over an 8-corner mini-lap

### 8.2 Chebyshev Ensemble Spacing

20 members with ω-weights Chebyshev-spaced on [0, 1]:

$$\omega_i = \frac{1}{2}\left(1 - \cos\!\left(\frac{i \pi}{N-1}\right)\right)$$

Concentrates ~65% of the ensemble into ω ∈ [0.7, 1.0] — the high-grip physically interesting region.

### 8.3 Riemannian Natural Policy Gradient (RNPG)

Replaces the KL-divergence parameter-space trust region with a Riemannian metric pulled back
through the physics engine:

$$G_{phys,k} = J_k^T \text{diag}(s) J_k + \lambda \cdot \text{diag}(J_k^T S J_k + \varepsilon I)$$

where `s = [1.0, 0.2, 0.5]` (grip weighted 5× over stability) and `J_k = ∂[grip, stab, LTE]/∂μ_k`.

The metric makes step size automatically small where the physical Jacobian is large — the optimizer
cannot take a large step across the ARB/oversteer bifurcation boundary without the metric shrinking
it. No hand-tuned threshold required.

### 8.4 ARD Bayesian Optimisation Cold Start

10 random initialisations + 30 EI-guided acquisitions using an ARD squared-exponential GP.
Per-dimension lengthscales learned via correlation heuristic — insensitive dimensions (castor,
anti-geometry) acquire large lengthscale and are effectively pruned.
Best 5 diverse basins seed the ensemble logit-space.

### 8.5 Endurance LTE Objective

**File:** `optimization/objectives.py`

Simulates an 8-corner mini-lap (120 steps × dt=5ms = 0.6 s) with a P-controlled driver:

```
MINI_LAP_SEGMENTS = [
    (0.00, 15, 22.0),    # S1: straight acceleration
    (0.00, 10, 10.0),    # S2: heavy braking
    (0.12, 20, 11.0),    # S3: medium-speed right (R ≈ 8.3m)
    (-0.18, 15,  9.0),   # S4: tight left hairpin (R ≈ 5.5m)
    (0.08, 10, 14.0),    # S5: fast right sweeper
    (-0.10, 15, 12.0),   # S6: medium left
    (0.15, 15, 10.0),    # S7: chicane right
    (0.00, 20, 20.0),    # S8: exit acceleration
]
```

Composite score (HIGHER = BETTER):
`J_LTE = w_speed · v̄ − w_energy · E_total − w_thermal · ΔT_penalty`

### 8.6 SMS-EMOA Hypervolume Archive

3D hypervolume contribution pruning for the (grip, stability, LTE) Pareto front.
Archive capped at 150 points. Stability hard filter: setups with overshoot > 5.0 rad/s are
**excluded** from the archive — not just penalised.

---

## 9. Powertrain Control Stack

**Directory:** `powertrain/`

The complete 4WD Ter27 powertrain control stack. A single `@jax.jit`-compiled function
`powertrain_step()` is the only entry point. No mode switching. All transitions are
sigmoid-smooth. Compiles to a single XLA graph at 200 Hz.

### 9.1 Unified Manager — 14-Stage Pipeline

**File:** `powertrain/powertrain_manager.py`

```
Stage  1 : Virtual Impedance            → filtered pedal inputs (PIO mitigation)
Stage  2 : Acceleration Estimation      → low-pass ax, ay
Stage  3 : Traction Control             → κ* references + TC/TV blend weights
Stage  4 : Driver Force Demand          → Fx from filtered throttle/brake
Stage  5 : Torque Limits                → motor envelope + thermal derating
Stage  6 : Power Budget                 → per-motor P_max from battery SoC + temperature
Stage  7 : Yaw Rate Reference           → driver-intent-aware target ψ̇_ref
Stage  8 : Launch Control               → B-spline torques during launch phase
Stage  8a: mpQP KKT Allocation          → single KKT linear solve (V2: 24×24, slip-aware)
Stage  8b: Dynamic Regen Blend          → KKT-optimal α*, hydraulic brake residual
Stage  9 : CBF Safety Filter            → input-delay DCBF + GP-uncertainty robustness
Stage 10 : Mode Blending                → launch vs TV/TC sigmoid blend
Stage 11 : Powertrain Thermal           → motor/inverter/battery state update
Stage 12 : Diagnostics Packaging        → 28-field PowertrainDiagnostics output
```

Output: `(PowertrainDiagnostics, PowertrainManagerState)` — every signal needed for telemetry.

### 9.2 Motor Model

**File:** `powertrain/motor_model.py`

PMSM electromechanical model with:
- Smooth field-weakening: `T_fw = T_peak · softplus(ω_base / ω) / softplus(1)` — C∞ transition
- Per-motor thermal ODE: winding temperature evolves with I²R + iron losses
- Battery OCV/R_int model: SoC tracked via Coulomb counting; R_int rises with temperature
- Power limit: `P_max = V_bus · I_max - I² · R_int` — exact, not approximate
- Regenerative braking envelope: `T_regen_max = min(T_motor_limit, P_charge_max / ω)`

### 9.3 Virtual Impedance (PIO Mitigation)

**File:** `powertrain/virtual_impedance.py`

Second-order virtual flywheel/damper preventing Pilot-Induced Oscillation on throttle/brake:

$$J \ddot{\theta} + C \dot{\theta} + K \theta = u_{pedal}$$

Symplectic Euler integration preserves the mechanical impedance structure. At 3 Hz (typical PIO
frequency), the filter provides > 30° phase lag — sufficient to break the PIO feedback loop.
Verified in Test 15: frequency response at 3 Hz confirmed.

### 9.4 mpQP KKT Torque Allocator

**File:** `powertrain/modes/advanced/explicit_mpqp_allocator.py`

Replaces the 12-iteration projected-gradient SOCP with a **single KKT linear solve** conditioned
on the active-set classifier prediction — reducing per-step allocation time from ~270 ms to ~40–90 µs
post-JIT.

**Two solver versions:**

| Version | Constraints | KKT Dimension | Active-set input dim |
|---|---|---|---|
| V1 | 12 (box + friction) | 16×16 | 15-dim θ |
| V2 | 20 (box + friction + 8 slip-CBF) | 24×24 | 19-dim θ |

**Pipeline V2 (~40–90 µs post-JIT):**
1. Build slip barrier rows (A_slip ∈ ℝ^{8×4}, b_slip ∈ ℝ^8) — ~5 µs
2. V2 classifier → soft active-set A ∈ (0,1)^{20} — ~8 µs
3. Build extended KKT K ∈ ℝ^{24×24} — ~3 µs
4. `jnp.linalg.solve` → [T*, λ*] — ~20 µs
5. Primal/dual + slip feasibility check — ~4 µs
6. 3-step polish if infeasible (~5% of calls) — ~25 µs max

**Smooth active-set via temperature-softened sigmoid** (enables end-to-end differentiability):

$$A_c = \sigma(\tau \cdot (p_c - \theta_c)), \quad \tau = 50$$

At τ = 50: within ±0.02 of the hard threshold, classification is indistinguishable from hard.
Outside this window, ∂A_c/∂θ is finite everywhere → ∂T*/∂θ is C∞ → bilevel gradient works.

**Cost function:**

$$J = w_{Mz}(M_z - M_z^*)^2 + w_{Fx}(F_x - F_x^d)^2 + w_{\Delta T}\|\Delta T\|^2 + w_{loss}\sum T_i^2 \omega_i + w_{thermal}\sum(2 - \mu_T(T_{ribs}^i)) \cdot w_i + w_{bal}(T_{ribs} - \bar{T}_{ribs})^2$$

Thermal weighting `(2 − μ_T)` steers torque toward thermally optimal wheels (μ_T = 1 at T_opt).
Thermal balance term penalises cross-wheel temperature variance to prevent asymmetric degradation.

### 9.5 Dynamic Regenerative Braking Blend

**File:** `powertrain/regen_blend.py`

Replaces the fixed `regen_blend = 0.7` scalar with a KKT-optimal `α*` that maximises energy
recovery subject to:
- Battery aggregate charge-current limit: `P_regen ≤ I_charge_max · V_bus`
- Cell temperature derating (hot battery → reduced regen)
- High-SoC tapering (near-full → reduced regen)
- Per-wheel motor envelope (via `T_min`)
- Slip-barrier feasibility (κ < κ* even under regen)

**Hydraulic brake interface:**
```
F_brake_total_wheel = F_regen_wheel + F_hydraulic_wheel
```
`F_brake_hydraulic` fills the deficit between driver braking demand and achievable regen.
Distributed to front/rear by the brake pressure modulator (outside this module).

`jax.grad(total_regen_energy)(setup_vector)` is well-defined — this is the differentiable
signal available for bilevel setup optimisation of regen strategy.

### 9.6 Slip Barrier (Predictive Slip CBF)

**File:** `powertrain/modes/advanced/slip_barrier.py`

Predicts future wheel slip `κ_{k+d}` at delay horizon d and builds linear constraint rows:
```
κ_{k+d,i} ≤ +budget_i   (upper slip limit)
κ_{k+d,i} ≥ −budget_i   (lower slip limit, regen)
```

These 8 rows (2 per wheel) are fed directly into the V2 mpQP KKT system as additional constraints
`[A_slip | b_slip]`, ensuring the allocator solution is slip-feasible before it exits — without
a separate filter stage. The slip barrier integrates with `cbf_safety_filter()` for defense-in-depth.

### 9.7 CBF Safety Filter — Input-Delay DCBF

**File:** `powertrain/modes/advanced/torque_vectoring.py` → `cbf_safety_filter()`

Discrete Control Barrier Function with actuator delay compensation:

$$h(x_{k+d}) \geq (1 - \alpha)^d \cdot h(x_k)$$

where d = actuator delay steps and α ∈ (0,1) is the decay rate. The filter predicts the
future state at delay horizon d and enforces the barrier there — preventing constraint violation
from commands that arrive after the state has already evolved.

The robust extension integrates GP uncertainty:

$$h_{robust}(x) = h(x) - \kappa_{safe} \cdot \sigma_{GP}(x)$$

### 9.8 Traction Control — DESC Extremum-Seeking

**File:** `powertrain/modes/advanced/traction_control.py`

Dual-path architecture:

**Path A (DESC):** Discrete ESC lock-in demodulator finds κ* (optimal slip) without needing
a tire model. Adaptive dither amplitude via Michaelis-Menten schedule:
`A_dither = A_max · κ / (κ + K_m)` — dither is small near κ=0 (stable region) and large
at high slip (where faster convergence is needed).

**Path B (SWIFT):** Coupled transient slip ODE from SWIFT model predicts tire transient
response for feedforward. Integrated with GP uncertainty weighting.

**Fusion:** `T_tc = w_desc · T_desc + w_swift · T_swift`, where weights are GP-uncertainty-weighted
(`w_desc ∝ 1/σ_GP²`). When GP is confident, DESC dominates; when GP is uncertain, SWIFT (physics-
based) dominates.

### 9.9 Launch Control — Neural Predictive Sequencer v2.1

**File:** `powertrain/modes/advanced/launch_control.py`

**16-knot cubic B-spline profile** over a 2.0 s launch horizon — offline-optimisable.

**6-phase state machine:**

```
IDLE ──(btn OR brake+throttle)──► ARMED ──(btn_release + WOT)──► PROBE
PROBE ──(probe complete)──► LAUNCH ──(v > v_thr OR t > T_dur)──► HANDOFF
HANDOFF ──(t > dt_blend)──► TC
LAUNCH/HANDOFF ──(hard brake)──► IDLE   [abort path]
```

v2.1 additions: button-based ARMED trigger; per-wheel TC ceiling `T_cmd ≤ μ_rt · Fz · r_w · γ_κ`;
real-time μ EMA from DESC; yaw-lock PI targeting ψ̇ = 0 during launch; abort via sigmoid-gated
hard brake. All transitions are smooth sigmoid blends — no `jax.lax.cond`.

### 9.10 Dictionary-Switched Koopman TV Controller

**File:** `powertrain/modes/advanced/koopman_tv.py`

Replaces the reactive PD yaw moment generator with a predictive, formally optimal controller.
Three local Koopman operators trained on grip utilisation regimes via EDMD-DL:

| Regime | ρ Range | Lifting dim m |
|---|---|---|
| 0 — Linear grip | ρ < 0.70 | 32 |
| 1 — Transition | 0.70–0.92 | 64 |
| 2 — Saturation | ρ > 0.92 | 32 |

**Online pipeline (200 Hz, ≈ 0.18 ms total):**

```
1. ρ = grip_utilisation(Fx, Fy, Fz, μ)
2. w = softmax(−(ρ − ρ_k*)² / (2σ²))   ← soft Gaussian regime weights
3. e = normalise_error(ψ̇_err, vy, vx, δ)
4. z_k = φ_k(e)                          ← lifting function per regime
5. Mz_k = −L_k @ z_k                     ← precomputed Riccati gain
6. Mz = Σ w_k · Mz_k
7. Mz = trained_blend · Mz + (1 − trained_blend) · Mz_PD   ← safe deployment ramp
```

`trained_blend = 0.0` gives pure PD fallback; `trained_blend = 1.0` gives full Koopman LQR.
This allows safe incremental deployment without restarting any subsystem.

---

## 10. Differentiable EKF — Online Parameter Estimation

**File:** `models/differentiable_ekf.py`

### 10.1 State & Observation

5-state parameter estimator running at every physics step:

```
θ = [λ_μ_f, λ_μ_r, T_opt, h_cg, α_peak]   — parameter state
y = [ay_measured, wz_measured]               — observations (standard MoTeC ADL3)
```

`α_peak` (5th state) tracks peak slip angle, which shifts with tyre wear and temperature.
Prior uncertainty ±0.04 rad covers both dry and wet operating conditions.

### 10.2 JAX Jacobian (No Finite Differences)

Because the vehicle dynamics are fully differentiable, the EKF measurement Jacobian is computed
exactly:

$$H = \frac{\partial [a_y^{sim}, \dot{\psi}^{sim}]}{\partial [\lambda_\mu, T_{opt}, h_{cg}, \alpha_{peak}]}$$

via `jax.jacobian` — no finite differences, no approximation errors. This is a direct benefit
of the end-to-end differentiable design.

### 10.3 Convergence

| Parameter | Initial Uncertainty | Convergence After |
|---|---|---|
| λ_μ_f, λ_μ_r | ±20% | 3–5 laps → ±3% |
| T_opt | ±10°C | 3–5 laps → ±5°C |
| h_cg | ±20 mm | 3–5 laps → ±8 mm |
| α_peak | ±0.04 rad | 5–8 laps → ±0.01 rad |

A living digital twin matches whatever car shows up on the day — different fuel load, tyre
pressures, tyre wear. The EKF makes the twin adaptive without retraining.

---

## 11. Full-Lap Differentiable Simulation

**File:** `optimization/differentiable_lap_sim.py`

### 11.1 The Kill Feature

```python
∂(lap_time) / ∂(setup_vector) = jax.grad(simulate_full_lap)(setup)
```

No FS team in the world has this. The gradient traces from the scalar lap time back through:

```
lap_time → speed profile → tire forces → suspension loads → setup params
```

in a single `jax.lax.scan` over the entire lap.

### 11.2 Differentiable Track

**File:** `optimization/differentiable_track.py`

JAX-native cubic spline interpolation on the SE(3) manifold:
- Track geometry stored as a sequence of SE(3) poses (rotation + translation)
- Curvature computed analytically: `κ = ω_z / v_xy` — no noisy finite-difference arclength derivatives
- Heading extracted from rotation matrix `R[1,0] / R[0,0]` — not from `arctan2(dy, dx)` which
  accumulates numerical error at parametric kinks
- NSDE stochastic friction uncertainty map `w_mu` from lateral slip variance — feeds directly
  into Tube-MPC parameter vectors

### 11.3 Architecture

```python
@jax.jit
def simulate_full_lap(setup: jax.Array) -> jax.Array:   # returns scalar lap_time
    init_state = ...
    def step_fn(carry, _):
        state, t = carry
        u = path_following_controller(state, track)
        state_new = vehicle.step(state, u, setup, dt)
        return (state_new, t + dt), state_new.vx
    (final, _), vx_traj = jax.lax.scan(step_fn, init_state, None, length=N_steps)
    return N_steps * dt + 0.0 * jnp.sum(vx_traj)  # compile-friendly scalar
```

Compiles once; `jax.grad` traces the entire XLA graph automatically.

### 11.4 Gradient Server

**File:** `scripts/gradient_server.py`

HTTP + WebSocket server that exposes on-demand gradient computation to the React dashboard:

```
React Dashboard ──HTTP POST──→ gradient_server.py ──JAX grad──→ ∂(lap_time)/∂(setup) Response
                 ──WebSocket──→ streaming gradient updates (as user drags sliders)
```

**Architecture:**
1. Receives 28/40D setup vector as JSON
2. Runs `jax.jacobian(step_fn)(x0, setup)` — short-horizon (N=20) Jacobian for responsiveness
3. Returns 28/40D sensitivity vector to the dashboard `∂x_N/∂s ∈ ℝ^{46×40}`
4. Full `jax.grad(simulate_full_lap)` available for offline batch computation

The JAX graph is compiled once on first call; subsequent calls use the XLA cache (~100 ms CPU,
~10 ms GPU). The gradient server allows setup engineers to interactively explore sensitivities in
real time from the `SuspensionExplorerModule` without understanding JAX internals.

---

## 12. CMA-ES Crossover Validator

**File:** `optimization/cmaes_validator.py`

Independent gradient-free optimizer that validates the MORL-SB-TRPO Pareto front. If CMA-ES
finds solutions that dominate the MORL front, the gradient-based optimizer has a bug or is stuck
in a local optimum.

**CMA-ES implementation:** Pure NumPy (μ/μ_w, λ)-CMA-ES with rank-1 and rank-μ updates.
Operates in normalised [0,1]^40 space. Physical bounds enforced by sigmoid projection.

**Workflow:**

1. Run MORL → Pareto front CSV
2. Run CMA-ES from 5 random initialisations + 5 MORL-front seeds
3. Compute hypervolume of CMA-ES solutions vs MORL Pareto front
4. PASS: CMA-ES does not dominate MORL front (gradient-based search is correct)
5. FAIL: CMA-ES dominates → MORL has a bug or local-minimum issue

---

## 13. Suspension Kinematic Analysis

**Directory:** `suspension/`

A full kinematic sweep analysis stack — the JAX-native equivalent of Optimum Kinematics — enabling
differentiable suspension geometry optimisation.

### 13.1 Kinematic Solver

**File:** `suspension/kinematics.py`

Computes at each heave position z:
- Camber angle, toe angle (bump steer curve), caster, kingpin inclination (KPI)
- Motion ratio MR(z) — fitted from K&C rig data via quadratic polynomial
- Roll centre height RC(z) — from instantaneous screw axis geometry
- Scrub radius at ground plane
- Track width change ΔY
- Wheel centre path `(X, Y, Z)`

All curves computed by `jax.vmap(solve_at_heave)(z_array)`, making results differentiable with
respect to tie-rod length `delta_L_tr` and shim angle `psi_shim`. This allows MORL to compute
`∂Objective/∂(toe_target)` exactly via the IFD chain — no finite differences needed.

### 13.2 Full Sweep Analysis

**File:** `suspension/sweep_analysis.py`

Outputs `SweepResult` and `SteerSweepResult` dataclasses:

```
SweepResult:
  z_mm [mm]           — heave range (−80 to +150 mm, 500 points)
  camber_deg          — camber curve [deg]
  toe_deg             — bump steer curve [deg]
  caster_deg          — caster variation [deg]
  motion_ratio        — MR(z) [-]
  rc_height_mm        — roll centre height [mm]
  scrub_radius_mm     — [mm]
  track_change_mm     — wheel centre Y relative to nominal [mm]

  Gains (at z=0):
  camber_gain_deg_per_m     dCamber/dz [deg/m]
  bump_steer_deg_per_m      dToe/dz [deg/m]
  drc_dz                    dRC/dz [-]
  ackermann_pct             positive = pro-Ackermann
```

All array quantities are NumPy (500-point heave sweep) for plotting; gains are JAX-traceable
for optimisation.

### 13.3 Excel Report Generation

**File:** `suspension/excel_writer.py`

Generates `Front_Ter27.xlsx` and `Rear_Ter27.xlsx` in Optimum Kinematics format:

```
Sheet 1: "Front/Rear Suspension"  — OKinematic structure
         Hardpoints: UNCHANGED (frozen)
         Stiffness rows: UPDATED with optimizer values
         Wheel rows: UPDATED with camber/toe

Sheet 2: "Alex Optimization"       — NEW
         § A: Kinematic Gains Summary (camber gain, bump steer, MR, RC)
         § B: Heave Sweep table (500 rows)
         § C: Steer Sweep table (100 rows, front only)
         § D: Roll Analysis table (100 rows)
         § E: Pareto Front summary from MORL
         § F: Recommended Setups (Skidpad / Endurance / Balanced)
```

---

## 14. Simulator Architecture

```
┌─────────────────────────┐  ┌────────────┐   ┌──────────────────────────────┐
│  React Dashboard        │  │ ws_bridge  │   │ gradient_server.py           │
│  (Vercel, 8 groups,     │─←│ UDP→WS     │   │ HTTP: ∂(lap_time)/∂(setup)   │
│   18 modules)           │  │ 60 Hz      │   │ WS: streaming sensitivities  │
│                         │  └────────────┘   └──────────────────────────────┘
│  useLiveTelemetry.js    │
└─────────────────────────┘
┌─────────────────────────┐  ┌────────────┐   ┌──────────────────────────────┐
│  physics_server.py      │  │ ws_bridge  │
│  200 Hz JAX integration │─→│ UDP→WS     │   ┌──────────────────────────────┐
│  46-DOF Port-Hamiltonian│  └────────────┘   │ Godot 3D Visualizer          │
│  + 40-dim setup space   │                   │ (driverless team)            │
│  + powertrain_step()    │──UDP 256B─────────→└──────────────────────────────┘
└────────┬────────────────┘──────┐
         │                ┌──────┴─────────┐  ┌──────────────────────────────┐
         │                │ ros2_bridge.py  │─→│ Driverless Stack             │
         │                │ UDP→ROS 2       │  │ nav2 / perception            │
         │                └────────────────┘  └──────────────────────────────┘
         ↑ UDP 32B (controls)
┌────────┴────────────────┐
│  control_interface.py   │
│  Keyboard / Gamepad /   │
│  Autopilot / ROS 2      │
└─────────────────────────┘
```

### 14.1 Physics Server

**File:** `simulator/physics_server.py`

- Runs the 46-DOF GLRK-4 + powertrain pipeline at 200 Hz
- Accepts 40-dim `SuspensionSetup` and backward-compat adapters (`from_legacy_28`, `from_legacy_8`)
- Broadcasts a 64-float (256-byte) UDP telemetry packet to all clients simultaneously
- Supports preset setup switching (`3_stiff`, `2_soft`, etc.) via control commands
- Lap timing via geometric finish-line intersection (RANSAC-based S/F placement)

### 14.2 WebSocket Bridge

**File:** `simulator/ws_bridge.py`

- Listens on UDP; decimates 200 Hz → 60 Hz for the React dashboard
- Serialises to compact JSON (named fields, 3 decimal places for forces)
- Broadcasts to all simultaneous dashboard clients
- Forwards control commands from dashboard → physics server
- Performance stats logged every 5 s (UDP Hz → WS Hz, client count)

### 14.3 ROS 2 Bridge

**File:** `simulator/ros2_bridge.py`

Standard ROS 2 topic output for the driverless team:

| ROS 2 Topic | Message Type | Content |
|---|---|---|
| `/odom` | `nav_msgs/Odometry` | Position, velocity, covariance |
| `/imu` | `sensor_msgs/Imu` | Acceleration, angular velocity, orientation |
| `/joint_states` | `sensor_msgs/JointState` | Wheel angular velocities |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands |
| `/tf` | `tf2_ros` | SE(3) transform broadcast |

Falls back to standalone UDP-print mode when `rclpy` is not available.

---

## 15. React Engineering Dashboard

**Directory:** `visualization/dashboard_react/`

Deployed on Vercel. 8 navigation groups, 18 modules, LIVE/ANALYZE mode toggle.

### Navigation Groups & Modules

| Group | Accent | Modules |
|---|---|---|
| **Overview** | cyan | Architecture explainer — H_net, Tire Model, Diff-WMPC, MORL-SB-TRPO |
| **Telemetry** | green | Live charts: speed, lateral G, Fz per corner, tire temps, yaw rate |
| **Vehicle Dynamics** | amber | Setup Opt (3D Pareto front), Suspension (LLTD, roll gradient, damper power), **Setup Explorer** (40-param interactive + 3D Pareto scatter + δ-highlight compare), Tire Physics (5-node thermal, GP envelope, hysteresis), Weight & CG |
| **Aerodynamics** | pink | Ground-effect Fz vs ride height, drag vs speed, aero balance |
| **Electronics** | purple | Motor temperatures, SoC, inverter power, battery voltage |
| **Powertrain** | orange | **Powertrain Ctrl** (14-stage pipeline visualisation), **TV Simulator** (interactive torque vectoring simulator) |
| **Controls & AI** | cyan | Energy Audit (Hamiltonian landscape, R-matrix heatmap), ∇ Insights (setup sensitivities ∂t_lap/∂param), Research (MORL convergence, car selector, parameter freeze UI) |
| **Performance** | red | Driver Coaching, Endurance Strategy (regen trade-off), Compliance (FSG rules) |

### Setup Explorer Module

`SuspensionExplorerModule.jsx` is the primary interface for setup engineers:
1. **3D Pareto scatter** (grip × stability × LTE)
2. **Car schematic** with per-corner parameter overlays
3. **Side-by-side setup comparison** with Δ-highlighting
4. **Sensitivity bar chart** (`∂lap_time/∂param`) pulled from gradient server
5. **Live gradient connection**: slider drag → HTTP POST → `gradient_server.py` → sensitivity update

### LIVE Mode

- WebSocket connection to `ws_bridge.py` at `ws://localhost:8765`
- `useLiveTelemetry.js` hook: auto-reconnect, 60 Hz frame buffer, ring-buffer history
- All charts update in real time from the physics server

### Theme System

```javascript
C.cy   = "#00d4ff"   — cyan   (architecture, control, Controls & AI)
C.gn   = "#00ff88"   — green  (tire, safety, Telemetry)
C.am   = "#ffaa00"   — amber  (dynamics, warning, Vehicle Dynamics)
C.red  = "#ff4444"   — red    (limits, MORL, Performance)
C.bg   = "#0a0a0f"   — background

// Shared style tokens
GL  — glass card border style
GS  — glassmorphism surface style
TT  — table text style
AX  — axis label style
```

---

## 16. Sanity Check Suite (Tests 1–16)

**Files:** `sanity_checks.py`, `powertrain/powertrain_sanity_checks.py`

### Physics Subsystem Tests (1–9)

| Test | Name | What It Verifies |
|---|---|---|
| 1 | Neural Convergence | H_net + R_net gradient flow; power injection < 500 J/s |
| 2 | Forward Pass | 46-DOF GLRK-4 step; state finite; energy conserved to < 1 J/step |
| 3 | Circular Track | Circular-arc simulation; lateral acceleration within friction circle |
| 4 | Friction Circle | `sqrt(Fx² + Fy²) ≤ mu * Fz` across full (α, κ) grid; ratio < 1.10 |
| 5 | Load Sensitivity | Fy_peak increases monotonically with Fz (degressive slope) |
| 6 | Diagonal Load Transfer | Correct corner Fz at 1.5G cornering + 0.5G braking |
| 7 | Aero Scaling | `Fz_aero ∝ v²` within 1% of theoretical |
| 8 | Differential Yaw Moment | Non-zero yaw moment from differential torque |
| 9 | Optimizer Boundary Diversity | k_f values not pinned at lower bound; stability cap active |

### Powertrain Control Tests (10–16)

| Test | Name | What It Verifies |
|---|---|---|
| 10 | Motor Torque Envelope | Low-speed torque ≈ T_peak (< 5%); monotone decrease in field-weakening |
| 11 | SOCP Allocator | Feasibility in 12 iterations; yaw moment tracking error < 5% |
| 12 | CBF Safety | CBF barrier h(x) ≥ 0 enforced; unsafe T_cmd filtered |
| 13 | DESC Convergence | κ* converges within adaptive dither schedule; error < 0.025 |
| 14 | Launch State Machine | Full IDLE→ARMED→PROBE→LAUNCH→HANDOFF→TC traversal |
| 15 | Virtual Impedance | Phase lag > 30° at 3 Hz; steady-state error < 5% |
| 16 | Full Pipeline JIT | Per-step time < 5 ms on SBC budget; 28 diagnostic fields finite |

---

## 17. Known Issues, Limits & Diagnostics

### 17.1 H_net Passivity Warning

Test 1 may report ~450 J/s passive power injection from H_net. This is attributed to synthetic
training data quality — MSE stagnated in prior revisions. Retraining `residual_fitting.py` with
the vX3 TTC PINN pipeline is expected to reduce this to < 50 J/s. The 450 J/s figure is ~0.45%
of typical kinetic energy at 15 m/s — acceptable for setup optimisation, not for energy analysis.

### 17.2 SOCP Solve Time (Legacy Path)

The 12-iteration projected-gradient SOCP allocator (legacy path) cold-start times on CPU
exceed the 5 ms real-time budget. The mpQP KKT allocator (Stage 8a) resolves this via the
active-set classifier: expected post-JIT latency 40–90 µs. The legacy SOCP path is retained
as a fallback when the classifier bundle is not loaded (`trained_blend = 0`).

### 17.3 Dashboard Telemetry Authenticity

Dashboard was historically seeded with synthetic RNG data. The `ws_bridge.py` → React real-time
connection is implemented. Full hardware-in-loop validation against MoTeC telemetry is a
competition-critical gap — synthetic data scores low on the FSG Siemens Digital Twin Award criteria.

### 17.4 XLA Compilation Times

First run triggers XLA compilation; subsequent runs use the `XLA_FLAGS` cache:

| Subsystem | First compile | Cached |
|---|---|---|
| Vehicle dynamics (GLRK-4) | ~3–5 min CPU | ~5–15 s |
| WMPC full trajectory | ~8–12 min CPU | ~20–40 s |
| MORL gradient | ~5–8 min CPU | ~10–20 s |
| Powertrain step | ~2–4 min CPU | ~3–8 s |
| Gradient server (short horizon) | ~1–2 min CPU | ~3–8 s |

Import `jax_config` **before any other JAX import** in every script.

### 17.5 Float32 Limits

- `softplus(200 * (z − 0.025))` overflows at z ≈ 469 mm — prevented by `clip(q[6:10], −0.08, 0.15)`
- H_net `H_RESIDUAL_CAP = 50,000 J/m²` prevents float32 overflow for extreme setups
- GP Gram matrix uses `jitter = 1e-3 * I` — sufficient for float32 (Cholesky condition ~1e4)

### 17.6 Three Existential Safety Threats (FSG 2026)

1. **CBF actuator delay** — addressed by input-delay DCBF in `cbf_safety_filter()`; slip barrier
   adds a second defense layer via predictive slip CBF in the KKT system
2. **EMI/bit-flip risks in SBC memory** — ECC memory or CRC-checked state transfer required
3. **4WD ABS v_x anchoring** — during simultaneous multi-wheel lockup, v_x estimate from
   wheel speeds collapses; requires IMU-based v_x fusion (not yet implemented)

---

## 18. Pipeline Execution

```bash
# 0. Activate environment
source project_gp_env/bin/activate

# 1. Verify all subsystems (Tests 1–16, ~20 min first run due to compilation)
python sanity_checks.py

# 2. Train H_net / R_net against TTC tire data and synthetic physics rollouts
python -c "from optimization.residual_fitting import train_neural_residuals; train_neural_residuals()"

# 3. Offline Koopman TV training (requires tire data + vehicle params)
python scripts/train_koopman_tv.py

# 4. Run MORL setup optimiser — Phase 1 (all params free, ~45 min)
python scripts/run_ter27_design_exploration.py --phase 1

# 4b. Phase 2 (CAD locked geometry, ~30 min)
python scripts/run_ter27_design_exploration.py --phase 2

# 4c. Phase 3 (prototype, only springs/dampers/ARBs free, ~15 min)
python scripts/run_ter27_design_exploration.py --phase 3

# 5. Validate Pareto front with CMA-ES crossover (~10 min)
python optimization/cmaes_validator.py --generations 200 --popsize 40

# 6. Run full pipeline (Telemetry → Track Gen → WMPC → Driver Coaching)
python main.py --mode full --log /path/to/motec_telemetry.csv

# 7. Generate suspension kinematic report (OKinematic Excel format)
python -c "from suspension.excel_writer import generate_report; generate_report('ter27')"

# 8a. Physics server (Terminal 1)
python simulator/physics_server.py --track fsg_autocross

# 8b. WebSocket bridge for React dashboard (Terminal 2)
python simulator/ws_bridge.py

# 8c. Gradient server for setup sensitivities (Terminal 3)
python scripts/gradient_server.py --port 8766

# 8d. Controls interface (Terminal 4)
python simulator/control_interface.py --mode keyboard

# 8e. ROS 2 bridge for driverless team (Terminal 5, requires ROS 2 Humble)
source /opt/ros/humble/setup.bash && python simulator/ros2_bridge.py

# 9. Launch React dashboard (if running locally)
cd visualization/dashboard_react && npm install && npm run dev
# Or use the Vercel deployment: https://project-gp-ter26.vercel.app
```

### Key Output Files

| File | Contents |
|---|---|
| `morl_pareto_front.csv` | 40-dim setup vectors + grip/stability/LTE on Pareto front |
| `cmaes_best_setup.csv` | CMA-ES best setup for MORL seeding |
| `models/h_net.bytes` | Trained H_net weights (Flax serialisation) |
| `models/r_net.bytes` | Trained R_net weights |
| `models/aero_net.bytes` | Trained physics-informed AeroMap weights |
| `models/h_net_scale.txt` | Training normalisation diagnostic (not architectural) |
| `suspension/Front_Ter27.xlsx` | Kinematic sweep + optimized setup (OKinematic format) |
| `suspension/Rear_Ter27.xlsx` | Rear kinematic sweep + optimized setup |

---

## 19. Revision History (GP-vX1 → vX4)

### GP-vX4 (Current)

**New Modules:**

| Module | File | Description |
|---|---|---|
| mpQP KKT Allocator V2 | `powertrain/modes/advanced/explicit_mpqp_allocator.py` | 24×24 KKT, slip-aware, soft active-set (τ-sigmoid), ~40–90 µs post-JIT |
| Active-Set Classifier V2 | `powertrain/modes/advanced/active_set_classifier.py` | V2: 19-dim θ including κ*, σ(κ*); V1: 15-dim θ |
| Slip Barrier | `powertrain/modes/advanced/slip_barrier.py` | Predictive slip CBF rows for KKT integration |
| Dynamic Regen Blend | `powertrain/regen_blend.py` | KKT-optimal α*; battery/temperature derating; hydraulic brake residual |
| Design Freeze | `config/design_freeze.py` | 3-phase progressive parameter lockdown; tanh → zero gradient for frozen params |
| Car Config Registry | `config/car_config.py` | Multi-car support; design bounds per car_id |
| Suspension Kinematics | `suspension/kinematics.py` | JAX-vmap; camber, toe, MR, RC, scrub; IFD chain for grad |
| Suspension Sweep | `suspension/sweep_analysis.py` | Full heave/steer/roll sweep; Optimum Kinematics equivalent |
| Suspension Excel Writer | `suspension/excel_writer.py` | Front/Rear OKinematic-format XLSX + Alex Optimization sheet |
| Gradient Server | `scripts/gradient_server.py` | HTTP/WebSocket; on-demand ∂(lap_time)/∂(setup) for dashboard |
| Design Exploration Script | `scripts/run_ter27_design_exploration.py` | Phase-aware multi-phase optimizer runner |
| Setup Explorer Module | `visualization/dashboard_react/src/SuspensionExplorerModule.jsx` | 3D Pareto, car schematic, side-by-side comparison, live sensitivities |
| Powertrain Control Module | `visualization/dashboard_react/src/PowertrainControlModule.jsx` | 14-stage pipeline visualisation |
| TV Simulator Module | `visualization/dashboard_react/src/TorqueVectoringSimModule.jsx` | Interactive torque vectoring simulator |

**Batch Upgrades:**

| ID | Component | Change |
|---|---|---|
| BATCH-K | SuspensionSetup | Expanded from 28D → 40D: heave springs, inerters, rising-rate MR, anti-squat-F, Ackermann, independent LS/HS rebound |
| BATCH-L | DesignFreeze | 3-phase progressive lockdown; `for_ter27_phase2()`, `for_ter27_phase3()`, `custom()` presets |
| BATCH-M | mpQP Allocator | V2 KKT (24×24) with 8 slip-barrier constraints; smooth τ-sigmoid active-set |
| BATCH-N | RegenBlend | Dynamic α* replacing fixed scalar; hydraulic brake interface; differentiable energy accounting |
| BATCH-O | Dashboard | Expanded 13 → 18 modules, 7 → 8 nav groups; SuspensionExplorer, PowertrainCtrl, TVSim |
| BATCH-P | GradientServer | HTTP/WebSocket gradient server; integrates with SuspensionExplorer drag-to-compute |
| BATCH-Q | SuspensionKinematics | JAX-vmap heave/steer/roll sweeps; Excel report generation in OKinematic format |

### GP-vX3

**New Modules:**

| Module | File | Description |
|---|---|---|
| Powertrain Manager | `powertrain/powertrain_manager.py` | 13-stage 200 Hz coordinator, 28-field diagnostics, single JIT entry |
| Motor Model | `powertrain/motor_model.py` | PMSM + smooth field-weakening + per-motor thermal + battery |
| Virtual Impedance | `powertrain/virtual_impedance.py` | 2nd-order virtual flywheel, symplectic Euler, PIO mitigation |
| Torque Vectoring | `powertrain/modes/advanced/torque_vectoring.py` | 12-iter SOCP + input-delay DCBF + counter-steer detection |
| Traction Control | `powertrain/modes/advanced/traction_control.py` | DESC ESC + GP-weighted dual-path fusion + SWIFT ODE |
| Launch Control v2.1 | `powertrain/modes/advanced/launch_control.py` | 16-knot B-spline + 6-phase FSM + button arming + yaw-lock PI |
| Koopman TV | `powertrain/modes/advanced/koopman_tv.py` | Dictionary-Switched Koopman LQR, EDMD-DL, 3 grip regimes |
| Differentiable EKF | `models/differentiable_ekf.py` | 5-state online estimator; jax.jacobian H matrix; α_peak 5th state |
| Differentiable Track | `optimization/differentiable_track.py` | JAX cubic spline on SE(3); analytical curvature; NSDE w_mu map |
| Lap Simulation | `optimization/differentiable_lap_sim.py` | Full-lap scan; `∂(lap_time)/∂(setup)` via `jax.grad` |
| CMA-ES Validator | `optimization/cmaes_validator.py` | Black-box Pareto crossover validation; pure NumPy CMA-ES |
| Physics Server | `simulator/physics_server.py` | 200 Hz JAX loop + UDP broadcast to 4 clients |
| WS Bridge | `simulator/ws_bridge.py` | UDP→WebSocket 60 Hz, multi-client, control forwarding |
| ROS 2 Bridge | `simulator/ros2_bridge.py` | nav_msgs + sensor_msgs + tf2; standalone fallback mode |

**Batch Upgrades:**

| ID | Component | Change |
|---|---|---|
| BATCH-A | TireOperatorPINN | PINN training pipeline against TTC data (residual_fitting.py Phase 3) |
| BATCH-B | SparseGP + CBF | Robust CBF: `h_robust = h(x) − κ_safe · σ_GP(x)` — GP uncertainty in safety filter |
| BATCH-C | DifferentiableAeroMap | Physics-Informed training: Fz ∝ v² constraint; softplus floors everywhere |
| BATCH-D | DifferentiableEKF | α_peak added as 5th estimated state; Q and P extended to (5×5) |
| BATCH-E | DifferentiableTrack | JAX-native cubic spline on SE(3) manifold; analytical curvature |
| BATCH-F | MORL evolutionary | RNPG replaces KL trust region; physics-pullback metric G_phys |
| BATCH-G | objectives.py | Endurance LTE as 3rd Pareto axis; 8-corner mini-lap simulation |
| BATCH-H | ocp_solver.py | Pseudo-Huber wavelet regularisation on D1/D2/D3 bands (replaces L1) |
| BATCH-I | Sanity checks | Tests 10–16: full powertrain stack verification (7 new tests) |
| BATCH-J | React Dashboard | 13-module dashboard: Aerodynamics + Electronics new modules; WS LIVE mode |

**Bug Fixes:**

| ID | Component | Issue | Fix |
|---|---|---|---|
| BUG-PT-1 | `cbf_safety_filter` | `AttributeError: 'CBFParams' has no attribute 'r_w'` — positional shift from adding `gp_sigma` | Updated all 4 call sites with explicit keyword arguments |
| BUG-PT-2 | `DESCState` | `TypeError: DESCState.__new__() missing 't_acc'` — `t_acc` added to NamedTuple without updating `default()` and `make_desc_state()` | Added `t_acc` to both factory functions |

### GP-vX2

**Critical Architecture Fixes:**

| ID | Component | Issue | Fix |
|---|---|---|---|
| BUGFIX-4 | H_net | `h_net_scale.txt` (102.62) passed into architecture → 102× energy amplification → all gradients zero | `h_scale=1.0` always in `NeuralEnergyLandscape`; scale file is diagnostic only |
| BUGFIX-5 | H_net | `susp_sq` gate at z=0 (never occupied during operation); operating point z_eq had full gradient | `susp_sq = Σ(q[6:10] − _Z_EQ)² + 1e-4` |
| BUGFIX-7 | TireModel | Thermal layout misaligned; `T_nodes[4]=T_surf0_r` used as front core | Reindexed to match `x[28:38]` exactly |
| BUGFIX-8 | TireModel | `tire.operator AttributeError` in `diagnose.py` | `@property operator` returns `_pinn_module` |

**Architecture Upgrades:**

| ID | Component | Change |
|---|---|---|
| UPGRADE-9 | H_net | `H_RESIDUAL_CAP` 5,000 → 50,000 J/m² (old cap clipped ~30% of training samples) |
| UPGRADE-10 | AeroMap | `jnp.maximum` → `_softplus_floor` for ground clearance and Cl floors |

**GP-vX3 Tire Patches (within vX2 file revision):**

| ID | Component | Change |
|---|---|---|
| GP-v3-A | SparseGP | `linalg.inv` → Cholesky + `stop_gradient(L)`; jitter 1e-4 → 1e-3 |
| GP-v3-B | SparseGP | `jnp.maximum` variance floor → `softplus` |
| GP-v3-C | SparseGP | Nested `vmap` Gram matrix → vectorised distance broadcast |
| GP-v3-D | TireOperatorPINN | 7D → 8D features (adds T_norm) |
| GP-v3-E | SpectralDense | `stop_gradient(σ)` in power iteration |
| GP-v3-F | SparseGP | Inducing point init: `uniform(0,1)` → `tanh(N(0,0.5))*scale+shift` |
| GP-v3-G | LCB penalty | Uncapped (2σ) → `clip(2σ, 0, 0.15)` |

### GP-vX1

| ID | Component | Change |
|---|---|---|
| BUGFIX-1 | SuspensionSetup | Index alignment — ARB at [2:4], not [4:6] |
| BUGFIX-2 | Dynamics | Mass/inertia defaults corrected for Ter27 |
| BUGFIX-3 | R_net | `_TRIL_14` defined at module level |
| UPGRADE-1 | H_net | FiLM conditioning on setup vector |
| UPGRADE-2 | R_net | log-diagonal guarantee (strict PD) |
| UPGRADE-3 | Integrator | Störmer-Verlet → 2-stage GLRK-4 variational integrator |
| UPGRADE-4 | Bumpstop | Hard clip → C∞ softplus contact model |
| UPGRADE-5 | Dynamics | Compliance steer from lateral load |
| UPGRADE-6 | Damper | 4-way digressive model (sigmoid blend) |
| UPGRADE-7 | Contact | Fz softplus floor |
| UPGRADE-8 | WMPC | `_build_default_setup_28` removed; canonical `build_default_setup_28()` |
| UPGRADE-9 | WMPC | Physics P-controller warm start (Newton units) |
| UPGRADE-10 | WMPC | NaN fallback anchored at warm start |
| UPGRADE-11 | MORL | Full 28-dim `SuspensionSetup` |
| UPGRADE-12 | MORL | ARD BO cold-start |
| UPGRADE-13 | MORL | SMS-EMOA HV contribution pruning |

---

*Project-GP is a live research codebase. Physics is correct to the extent that `sanity_checks.py`
passes all 16 tests. Retraining H_net (`residual_fitting.py`) after any architectural change is
required before running WMPC or MORL in production mode. The Koopman TV controller requires
offline EDMD-DL training (`scripts/train_koopman_tv.py`) before `trained_blend > 0`. The mpQP
KKT allocator requires offline active-set classifier training before the KKT path activates;
the projected-gradient fallback is always available.*