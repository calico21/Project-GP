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
| **State Dimension** | 108 (14 positions + 14 momenta + 28 thermal 3D + 16 transient 2nd-order slip + 12 damper hysteresis + 24 elastokinematic compliance) |
| **Integrator** | 2-stage Gauss-Legendre RK4 (GLRK-4), symplectic, 4th-order, 5-iter Newton |
| **Dynamics** | Neural Port-Hamiltonian System — H_net (FiLM-conditioned) + R_net (Cholesky PSD) |
| **Tire Model** | Pacejka MF6.2 + Turn Slip + 3D Lateral Asymmetric Thermal (4 corners × 7 nodes) + 2nd-order Transient Slip + PINN (8-feature) + Matérn 5/2 Sparse GP |
| **Suspension** | Full double A-arm kinematic solver (IFD + custom VJP) + nonlinear Bouc-Wen elastokinematic bushing model |
| **Aero** | AeroPlatformModel with ground effect stall (physics-informed) |
| **Damper** | Maxwell ODE damper with thermal dynamics + Bouc-Wen hysteresis |
| **Track** | Track surface model with rubber build-up and grip asymmetry |
| **Slip Observer** | Koopman Slip Observer (primary κ* estimator, Batch 4) + RLS Slip-Slope Observer (secondary) |
| **Optimal Control** | Diff-WMPC — 3-level Daubechies-4 Wavelet MPC + UT Stochastic Tubes + Augmented Lagrangian + Pseudo-Huber Regularization |
| **Setup Optimisation** | MORL-SB-TRPO + RNPG — 40-dim, Chebyshev ensemble, ARD-BO cold-start, SMS-EMOA Pareto archive, Endurance LTE 3rd axis |
| **Setup Space** | 40 parameters (`SuspensionSetup` NamedTuple) — springs, heave springs, inerters, ARBs, 4-way dampers, geometry, Ackermann, anti-geometry, diff |
| **Setup Freeze** | `DesignFreeze` — 3-phase progressive CAD lockdown; tanh projection → frozen params have zero gradient automatically |
| **Powertrain Control** | 14-stage pipeline at 200 Hz: mpQP KKT Allocator + Regen Blend + Robust DCBF + DESC-TC + B-spline Launch + Koopman LQR |
| **State Estimator** | Differentiable EKF — 5-state online parameter estimation (λ_μ_f, λ_μ_r, T_opt, h_cg, α_peak) |
| **Lap Gradient** | `jax.grad(simulate_full_lap)(setup)` — end-to-end differentiable lap simulation |
| **Suspension Kinematics** | Full double A-arm IFD kinematic solver (JAX-vmap); Optimum Kinematics-equivalent; Excel report generation |
| **Elastokinematics** | Nonlinear Bouc-Wen bushing model; 24-state compliance subsystem in state vector |
| **Simulator** | 200 Hz physics server + UDP→WebSocket bridge + ROS 2 bridge + Godot 3D |
| **Gradient Server** | HTTP/WebSocket server — on-demand ∂(lap_time)/∂(setup) from React dashboard |
| **Dashboard** | React (Vercel) — 18 modules, 8 nav groups, LIVE/ANALYZE modes |
| **Revision** | GP-vX5 |

---

## Table of Contents

1. [Philosophy & Design Principles](#1-philosophy--design-principles)
2. [Repository Structure](#2-repository-structure)
3. [State Vector & SuspensionSetup](#3-state-vector--suspensionsetup)
4. [Design Freeze System](#4-design-freeze-system)
5. [Neural Port-Hamiltonian Vehicle Dynamics](#5-neural-port-hamiltonian-vehicle-dynamics)
6. [Suspension Package — Kinematics & Elastokinematics](#6-suspension-package--kinematics--elastokinematics)
7. [Multi-Fidelity Tire Model](#7-multi-fidelity-tire-model)
8. [Aero, Damper & Track Surface Models](#8-aero-damper--track-surface-models)
9. [Slip Observers — Koopman & RLS](#9-slip-observers--koopman--rls)
10. [Differentiable Wavelet MPC (Diff-WMPC)](#10-differentiable-wavelet-mpc-diff-wmpc)
11. [MORL-SB-TRPO Setup Optimiser](#11-morl-sb-trpo-setup-optimiser)
12. [Powertrain Control Stack](#12-powertrain-control-stack)
13. [Differentiable EKF — Online Parameter Estimation](#13-differentiable-ekf--online-parameter-estimation)
14. [Full-Lap Differentiable Simulation](#14-full-lap-differentiable-simulation)
15. [CMA-ES Crossover Validator](#15-cma-es-crossover-validator)
16. [Suspension Kinematic Analysis](#16-suspension-kinematic-analysis)
17. [Simulator Architecture](#17-simulator-architecture)
18. [React Engineering Dashboard](#18-react-engineering-dashboard)
19. [Sanity Check Suite (Tests 1–25)](#19-sanity-check-suite-tests-125)
20. [Known Issues, Limits & Diagnostics](#20-known-issues-limits--diagnostics)
21. [Pipeline Execution](#21-pipeline-execution)
22. [Revision History (GP-vX1 → vX5)](#22-revision-history-gp-vx1--vx5)

---

## 1. Philosophy & Design Principles

Project-GP abandons traditional numerical simulation frameworks (CasADi, IPOPT, point-mass solvers)
in favour of a **deep learning compiler architecture (JAX/XLA)**. The design is governed by five
non-negotiable constraints.

**The Differentiability Rule.** Every function, physics equation, and control logic path must be
strictly differentiable. Hard conditionals (`jnp.where` with discontinuous branches, step functions,
`jax.lax.cond` without gradient-safe branches) are forbidden. All limits are implemented via smooth
approximations: `jax.nn.softplus`, `jax.nn.sigmoid`, `jnp.tanh` rescaling. The Bouc-Wen bushing
hysteresis and the Maxwell damper ODE are implemented as differentiable ODEs rather than
lookup tables — both are fully traceable by `jax.grad`.

**The JAX Purity Rule.** All physics is written in pure JAX. NumPy operations are prohibited inside
JIT-compiled or grad-traced functions. No host-device sync mid-graph. `vmap` and `scan` correctness
is verified against abstract tracing rules — notably, `jnp.convolve` cannot be vmapped over a batch
dimension without materialising it inside `lax.conv_general_dilated`; explicit per-channel calls are
used throughout. The IFD (Implicit Function Differentiation) chain in `suspension/kinematics.py`
uses a custom VJP to differentiate through the kinematic constraint solver without unrolling Newton
iterations.

**The Physical Rule.** The Port-Hamiltonian structure is never broken. `H_net` can only add
non-negative residual energy gated at physical equilibrium. `R_net` can only produce PSD dissipation
matrices via `R = LLᵀ + diag(softplus(d))`. The Bouc-Wen hysteresis bushing model is energy-bounded
by construction — the dissipative term is always non-negative. The system structurally cannot
hallucinate free energy.

**The Canonical-Index Rule.** All 40 setup parameters are accessed exclusively via
`SuspensionSetup.from_vector()` / `.to_vector()` and the canonical builder functions. No positional
indexing of the raw setup array is permitted outside these methods — a lesson learned from a
catastrophic silent bug where ARB indices received spring-rate values. The 108-DOF state vector
is similarly accessed via named index ranges defined at module level; raw positional indexing
into `x` is forbidden outside `vehicle_dynamics.py`.

**The Freeze-Before-Grad Rule.** When geometry is committed to CAD, `DesignFreeze` must be applied
before running the optimizer. Frozen parameters have LB = UB, so `project_to_bounds()` returns a
constant — the gradient is identically zero without any masking logic in the optimizer. This prevents
the optimizer from wasting search budget on non-adjustable hardpoint geometry.

---

## 2. Repository Structure

```
FS_Driver_Setup_Optimizer/
├── models/
│   ├── vehicle_dynamics.py          # Neural Port-Hamiltonian 108-DOF dynamics + GLRK-4
│   ├── tire_model.py                # Pacejka MF6.2 + 3D thermal + transient + PINN + Sparse GP
│   ├── tire_thermal_3d.py           # 3D tire thermal with lateral asymmetry (camber load dist.)
│   ├── tire_transient.py            # Second-order tire transient dynamics (carcass + belt)
│   ├── aero_platform.py             # AeroPlatformModel with ground effect stall
│   ├── damper_hysteresis.py         # Maxwell ODE damper with thermal dynamics
│   ├── track_surface.py             # Track surface with rubber build-up and grip asymmetry
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
│   ├── kinematics.py                # Full double A-arm kinematic solver; IFD + custom VJP
│   │                                #   FIX-1 through FIX-4; camber, toe, MR, RC, scrub
│   ├── elastokinematics.py          # Nonlinear elastokinematic bushing model (Bouc-Wen hysteresis)
│   ├── sweep_analysis.py            # Full heave/steer/roll sweep — Optimum Kinematics equivalent
│   └── excel_writer.py              # Generates Front/Rear_Ter27.xlsx (OKinematic format)
│
├── observers/
│   ├── koopman_slip.py              # Koopman Slip Observer — primary κ* estimator (Batch 4)
│   └── rls_tc.py                    # Recursive Least-Squares Slip-Slope Observer (secondary)
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
│       │   ├── OverviewModule.jsx
│       │   ├── TelemetryModule.jsx
│       │   ├── SetupModule.jsx
│       │   ├── SuspensionModule.jsx
│       │   ├── SuspensionExplorerModule.jsx
│       │   ├── TirePhysicsModule.jsx
│       │   ├── WeightBalanceModule.jsx
│       │   ├── AerodynamicsModule.jsx
│       │   ├── ElectronicsModule.jsx
│       │   ├── PowertrainControlModule.jsx
│       │   ├── TorqueVectoringSimModule.jsx
│       │   ├── EnergyAuditModule.jsx
│       │   ├── DifferentiableInsightsModule.jsx
│       │   ├── ResearchModule.jsx
│       │   ├── DriverCoachingModule.jsx
│       │   ├── EnduranceStrategyModule.jsx
│       │   └── ComplianceModule.jsx
│       └── package.json
│
├── tests/
│   └── test_verification.py         # pytest suite: 5 physics tests (friction, aero, spring)
│
├── sanity_checks.py                 # Full system verification — Tests 1–25
├── main.py                          # Pipeline entry point (--mode setup | full)
├── jax_config.py                    # XLA cache + memory + parallelism config (import first)
└── morl_pareto_front.csv            # Latest 3-objective Pareto front (grip, stability, LTE)
```

---

## 3. State Vector & SuspensionSetup

### 3.1 108-Dimensional State Vector

The state vector has been significantly expanded from the 46-DOF representation of GP-vX4. The
current architecture is 108-dimensional, reflecting the addition of a full 3D tire thermal model,
second-order transient slip dynamics, Maxwell damper hysteresis states, and the elastokinematic
compliance subsystem from the new `suspension/` package.

```
x[0:14]   kinematics — positions and momenta (14 DOF)
           [0:3]   X, Y, Z          chassis CG, world frame   [m]
           [3]     φ  roll           [rad]
           [4]     θ  pitch          [rad]
           [5]     ψ  yaw            [rad]
           [6:10]  z_fl, z_fr, z_rl, z_rr   suspension heave  [m]
           [10:14] θ_fl..θ_rr        wheel rotation           [rad]

           NOTE: momenta (p = M·v, 14 components) are stored in the
           canonical Port-Hamiltonian sense. x[0:14] are generalised
           positions q; x[14:28] would correspond to the generalised
           momentum representation. The full kinematic block occupies
           x[0:28] (14 positions + 14 momenta).

x[0:28]   kinematics block (14 positions + 14 momenta)

x[28:56]  thermal 3D — 4 corners × 7 nodes (28 states)
           Per corner layout (7 nodes):
             [0]  T_surf_inner      inner surface rib          [°C]
             [1]  T_surf_mid        mid surface rib            [°C]
             [2]  T_surf_outer      outer surface rib          [°C]
             [3]  T_gas             internal gas pressure node [°C]
             [4]  T_core            structural carcass core    [°C]
             [5]  T_carcass_inner   carcass lateral node       [°C]
             [6]  T_carcass_outer   carcass lateral node       [°C]
           Corners ordered: FL, FR, RL, RR.
           Lateral asymmetry (nodes [5,6]) captures camber load distribution
           and the associated inboard/outboard surface temperature gradient.

x[56:72]  transient 2nd-order slip — 4 corners × 4 states (16 states)
           Per corner layout (4 states):
             [0]  α_t               transient slip angle (carcass lag) [rad]
             [1]  α_t_dot           rate                               [rad/s]
             [2]  κ_t               transient longitudinal slip [-]
             [3]  κ_t_dot           rate                               [1/s]
           Second-order dynamics model both the carcass compliance lag and
           the belt relaxation — a step improvement over the first-order
           model in GP-vX4.

x[72:84]  damper hysteresis — 12 states
           Maxwell ODE damper produces 3 internal states per corner:
             [0]  z_maxwell         Maxwell element displacement       [m]
             [1]  F_internal        internal spring force             [N]
             [2]  T_fluid           damper fluid temperature          [°C]
           These states are integrated alongside the vehicle ODE and feed
           back into the per-corner suspension force computation.

x[84:108] elastokinematic compliance — 24 states
           The Bouc-Wen bushing model introduces 6 compliance states per
           corner (one per bushing DOF of the double A-arm):
             [0:3]  δ_x, δ_y, δ_z   translational compliance         [m]
             [3:6]  δ_φ, δ_θ, δ_ψ  rotational compliance            [rad]
           These states evolve according to the nonlinear Bouc-Wen
           differential equation and couple back into camber, toe, and
           caster via the IFD kinematic chain in suspension/kinematics.py.
```

**Static equilibrium deflections** (module-level constant `_Z_EQ`, single source of truth):

```
z_eq_f ≈ 12.8 mm   (k_f=35000 N/m, MR_f=1.14, F_susp_eq≈583 N)
z_eq_r ≈ 13.5 mm   (k_r=38000 N/m, MR_r=1.16, F_susp_eq≈695 N)
```

The `susp_sq` gate in `H_net` is anchored at `_Z_EQ` (not at z=0), ensuring that H_net gradient
is non-zero at the normal operating point.

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
| 35 | ackermann | % | 0 | Ackermann factor: 0 = parallel, 1 = full, −1 = reverse |
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
build-fixed — hardpoint geometry determines them and they cannot be adjusted at track. Passing these
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
- $F_{ext}$ — non-conservative external forces (tire, aero, gravity, elastokinematic)

The 108-DOF state vector is partitioned into four subsystems, each with its own ODE formulation.
The GLRK-4 integrator solves the full coupled system simultaneously: the kinematics subsystem is
the primary Port-Hamiltonian ODE; the thermal, slip, hysteresis, and compliance subsystems are
treated as auxiliary ODEs integrated with the trapezoidal rule using the converged GLRK-4 stage
derivatives, ensuring A-stability with no additional `_compute_derivatives` calls.

### 5.2 Neural Energy Landscape — H_net (FiLM-conditioned)

`NeuralEnergyLandscape` is a 128→64→1 MLP with **FiLM conditioning** at each hidden layer.
The setup vector modulates the energy landscape via Feature-wise Linear Modulation:

$$h_{out} = \gamma(\text{setup}) \odot \text{LayerNorm}(h) + \beta(\text{setup})$$

**Energy output:**

$$H_{total} = T_{prior} + V_{structural} + H_{res} \cdot \sum(q_{susp} - z_{eq})^2 + \epsilon$$

Key properties:
- `T_prior = 0.5 * ||p||² / M_diag` — exact kinetic energy from generalised momenta
- `V_structural = 0.5 * ||q_susp||² * K_prior` (K_prior = 30,000 N/m)
- `susp_sq_eq` gates at **physical equilibrium** `_Z_EQ`, not at z=0
- `H_res = min(softplus(MLP) * h_scale, 50_000)` — capped at 50,000 J/m²; `h_scale = 1.0` always
- 22 SE(3)-bilateral symmetric state features: anti-symmetric quantities enter as x²

**H_net retraining requirement.** The 108-DOF state vector is architecturally incompatible with
H_net weights trained on the 46-DOF vector. H_net must be retrained via `residual_fitting.py`
before running WMPC or MORL in production mode. The `trained_blend = 0.0` lock on the Koopman
TV controller is also pending this retraining. See Section 20.

### 5.3 Neural Dissipation Matrix — R_net (Cholesky PSD)

$$R = L L^T + \text{diag}(\text{softplus}(d))$$

The `diag(softplus(d))` term guarantees strict positive definiteness. A physical mask restricts
dissipation to heave, roll, pitch, and unsprung-z DOFs.

### 5.4 Aero Model

The original `DifferentiableAeroMap` (32→32→4 MLP) has been replaced by `AeroPlatformModel`
in `models/aero_platform.py`. This is a physics-informed model that extends the previous
formulation with an explicit **ground effect stall** regime — the nonlinear region where
downforce drops sharply as ride height decreases below the stall threshold.

Key features:
- Ground effect stall modelled as a smooth sigmoid transition: `Fz_ge = Fz_linear * σ(h_ride − h_stall) + Fz_stall * (1 − σ(...))`
- Physics-informed training constraint: Fz ∝ v² in the linear ground-effect regime (verified Test 19)
- All clearance and downforce floors use `_softplus_floor` — no `jnp.maximum` subgradients
- Outputs: `(Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero)`

### 5.5 Integrator — 2-Stage Gauss-Legendre RK4 (GLRK-4)

Butcher tableau:

$$a = \begin{pmatrix} 1/4 & 1/4 - \sqrt{3}/6 \\ 1/4 + \sqrt{3}/6 & 1/4 \end{pmatrix}, \quad b = (1/2,\ 1/2)$$

- **Symplectic** — preserves the symplectic 2-form `dq ∧ dp` to machine precision
- **4th-order** — energy drift O(h⁵) vs O(h³) for Störmer-Verlet
- **Implicit** — 5-iteration Newton scan inside `jax.lax.scan`; stage clipping at ±500 m/s²
  prevents NaN propagation without affecting gradients within the physical envelope
- **Auxiliary sub-states** (thermal 3D + 2nd-order slip + hysteresis + compliance) integrated
  with the trapezoidal rule using converged GLRK-4 stage derivatives — A-stable

### 5.6 Suspension Forces

Per corner: `F_susp = F_spring + F_damper_maxwell + F_bumpstop + F_ARB + F_heave + F_inerter + F_elasto`

- **Spring:** `F = k * z * MR(z)²`, MR is a quadratic polynomial of heave travel (rising-rate via `mr_rise`)
- **Heave spring:** Third spring resisting heave mode independently of roll
- **Inerter:** `F_inerter = b * z̈_heave` — apparent inertia in the 2nd-order heave subsystem
- **Damper (Maxwell ODE):** Replaces the sigmoid-blend 4-way digressive model. The Maxwell
  element is a spring and viscous damper in series — the ODE evolves `z_maxwell` (the Maxwell
  element displacement) and `T_fluid` (fluid temperature). This captures velocity-dependent
  hysteresis and thermal fade in a single differentiable system. See Section 8.
- **Bumpstop:** `F = k_bs * softplus(β * (z − gap)) / β` — C∞, zero below gap
- **ARB:** lateral roll moment applied as symmetric ±correction to left/right corners
- **Elastokinematic correction:** `F_elasto` is the force contribution from bushing compliance
  states `x[84:108]`, propagated through the Bouc-Wen constitutive law. See Section 6.2.

---

## 6. Suspension Package — Kinematics & Elastokinematics

**Directory:** `suspension/`

The `suspension/` package is new in GP-vX5. It provides the full double A-arm kinematic and
elastokinematic foundation for the 108-DOF state vector, replacing the polynomial-approximation
approach used in earlier revisions. All solvers are JAX-native and fully differentiable.

### 6.1 Full Double A-arm Kinematic Solver

**File:** `suspension/kinematics.py`

The kinematic solver computes the instantaneous geometric state of each corner's double A-arm
linkage from the heave displacement and setup parameters. It solves a system of loop-closure
constraint equations using Newton's method, then differentiates through the solver using
**Implicit Function Differentiation (IFD)** with a custom VJP — avoiding the full unrolled
Newton tape that would make the gradient graph computationally intractable.

**Outputs per corner at each heave position `z`:**
- Camber angle γ(z), toe angle δ(z) (bump steer curve), caster angle, kingpin inclination (KPI)
- Motion ratio MR(z) — ratio of wheel travel to spring travel
- Roll centre height RC(z) — from instantaneous screw axis geometry
- Scrub radius at ground plane
- Track width change ΔY(z)
- Wheel centre path `(X, Y, Z)`

All quantities are computed by `jax.vmap(solve_at_heave)(z_array)` and are differentiable with
respect to tie-rod length `delta_L_tr` and shim angle `psi_shim` via the IFD chain. This allows
`MORL` to compute `∂Objective/∂(toe_target)` and `∂Objective/∂(camber_gain)` exactly.

**Applied fixes** (FIX-1 through FIX-4 documented inline in `kinematics.py`):
- **FIX-1:** Upper A-arm inboard point coordinate convention corrected (sign error on Y-axis)
- **FIX-2:** Newton convergence criterion tightened from 1e-6 to 1e-9 for float32 stability
- **FIX-3:** Custom VJP correctly handles the Jacobian transpose in the IFD chain (previous
  implementation transposed the wrong matrix)
- **FIX-4:** Roll centre height clipped to `[−0.05, 0.25]` m to prevent `softplus` overflow
  under extreme heave inputs during MORL exploration

### 6.2 Elastokinematic Bushing Model

**File:** `suspension/elastokinematics.py`

The elastokinematic model captures how rubber-bushing compliance in the A-arm pickup points
modifies the instantaneous wheel geometry under load — a significant contributor to understeer
gradient and brake steer that was not represented in the previous purely kinematic model.

Each bushing is modelled as a **Bouc-Wen hysteretic element**, which is the industry standard
for capturing the nonlinear stiffness and energy dissipation of elastomeric bushings:

$$\dot{z}_{BW} = A\dot{u} - \beta|\dot{u}||z_{BW}|^{n-1}z_{BW} - \gamma\dot{u}|z_{BW}|^n$$

$$F_{bushing} = k_{lin} \cdot u + k_{hys} \cdot z_{BW}$$

Key implementation properties:
- The Bouc-Wen ODE is fully differentiable via `jax.lax.scan` — no piecewise or conditional logic
- Parameters A, β, γ, n are per-bushing and can be identified from K&C rig measurements
- The 24 compliance states `x[84:108]` are the Bouc-Wen `z_BW` variables (6 DOF × 4 corners)
- Compliance output `(δ_camber, δ_toe, δ_caster)` feeds back into tire slip angle and
  longitudinal slip via the kinematic chain — closing the loop between compliance and tire force

The elastokinematic contribution is particularly significant under heavy braking, where longitudinal
bushing deflection induces compliance steer at the rear axle, affecting the understeer gradient
in the critical trail-braking phase of a corner entry.

### 6.3 Full Sweep Analysis

**File:** `suspension/sweep_analysis.py`

Outputs `SweepResult` and `SteerSweepResult` dataclasses:

```
SweepResult:
  z_mm [mm]                 heave range (−80 to +150 mm, 500 points)
  camber_deg                camber curve [deg]
  toe_deg                   bump steer curve [deg]
  caster_deg                caster variation [deg]
  motion_ratio              MR(z) [-]
  rc_height_mm              roll centre height [mm]
  scrub_radius_mm           [mm]
  track_change_mm           wheel centre Y relative to nominal [mm]

  Gains (at z=0, JAX-traceable for optimisation):
  camber_gain_deg_per_m     dCamber/dz [deg/m]
  bump_steer_deg_per_m      dToe/dz [deg/m]
  drc_dz                    dRC/dz [-]
  ackermann_pct             positive = pro-Ackermann
```

### 6.4 Excel Report Generation

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

## 7. Multi-Fidelity Tire Model

**Files:** `models/tire_model.py`, `models/tire_thermal_3d.py`, `models/tire_transient.py`

The tire model has been restructured around three new dedicated model files that replace the
monolithic `tire_model.py` approach. The four-layer architecture is retained but each layer is
now substantially more sophisticated.

### 7.1 Layer 1 — Pacejka MF6.2

Full Magic Formula for Hoosier R20 (TTC-fitted). Includes:
- Longitudinal force Fx(κ, Fz, γ, P), lateral force Fy(α, Fz, γ, P) with combined slip
- Self-aligning moment Mz(α, κ, Fz, γ, P), turn-slip correction, ply steer and conicity offsets
- Combined-slip weighting via Pacejka's G-function
- All Pacejka functions are C∞ via `jnp.tanh` replacement for the sign discontinuity

### 7.2 Layer 2 — 3D Tire Thermal Model with Lateral Asymmetry

**File:** `models/tire_thermal_3d.py`

The previous 5-node Jaeger axisymmetric thermal model (per-axle, `x[28:38]`) has been replaced
by a **full 3D thermal model** with 7 nodes per corner, occupying `x[28:56]` (28 states total).
The key architectural addition is **lateral asymmetry** from camber load distribution.

Camber angle shifts the contact patch load toward the inner or outer shoulder of the tread,
creating a temperature gradient across the 3 surface rib nodes (`T_surf_inner`, `T_surf_mid`,
`T_surf_outer`). The lateral carcass nodes (`T_carcass_inner`, `T_carcass_outer`) track the
thermal gradient through the carcass thickness. This asymmetry has first-order effects on
the grip distribution across the contact patch width and on the effective μ seen by the
Pacejka model.

Heat flows:
- Contact patch flash temperature (Jaeger model) → surface ribs (weighted by camber load distribution)
- Surface ribs ↔ gas (convection + conduction)
- Surface ribs → core (conduction through carcass)
- Carcass lateral nodes coupled to their respective surface ribs

Thermal grip modulation:
`μ_T(T_surf_effective) = μ_peak · exp(−β_T · (T_surf_eff − T_opt)²)`

where `T_surf_eff` is the camber-load-weighted average of the three surface rib temperatures.

### 7.3 Layer 3 — Second-Order Transient Slip Dynamics

**File:** `models/tire_transient.py`

The first-order carcass lag model of GP-vX4 (8 states, `x[38:46]`) has been replaced by a
**second-order model** that captures both carcass compliance and belt relaxation separately,
occupying `x[56:72]` (16 states, 4 states per corner).

The second-order transient dynamics are:

$$\sigma_c \ddot{\alpha}_t + (1 + \sigma_c/\sigma_b)\dot{\alpha}_t + \alpha_t/\sigma_b = \alpha_{ss} / \sigma_b$$

where σ_c is the carcass relaxation length and σ_b is the belt relaxation length. This formulation
correctly predicts the oscillatory overshoot in lateral force seen during rapid steer inputs at
high speed — a known weakness of first-order models that caused systematic underestimation of
peak lateral acceleration during chicanes.

### 7.4 Layer 4 — TireOperator PINN (8-feature)

Physics-Informed Neural Network for deterministic residual correction beyond MF6.2.
8-feature input vector: `[α, κ, Fz, γ, P, vx, T_surf, T_norm]`

`T_norm` allows the PINN to distinguish cold tires at normal load from hot tires at the same
load — critical for predicting grip at the operating temperature.

### 7.5 Layer 5 — Matérn 5/2 Sparse GP (Stochastic Uncertainty)

Inducing-point sparse GP for real-time uncertainty quantification over the tire operating envelope.
Implementation details (unchanged from GP-vX4):
- Gram matrix: vectorised distance broadcast (no nested `vmap`)
- Cholesky solve with `stop_gradient(L)`; jitter `1e-3 * I`
- Inducing point init: `tanh(N(0, 0.5)) * scale + shift`
- Variance floor: `softplus(var)`; LCB penalty: `clip(2σ, 0, 0.15)`

GP uncertainty propagation:
- WMPC stochastic tubes: `κ_safe · σ_GP` added to tube half-width
- Robust CBF: `h_robust(x) = h(x) − κ_safe · σ_GP(x)`
- DESC-TC fusion weights: `w_desc ∝ 1/σ_GP²`

---

## 8. Aero, Damper & Track Surface Models

### 8.1 AeroPlatformModel

**File:** `models/aero_platform.py`

Replaces `DifferentiableAeroMap`. The `AeroPlatformModel` adds an explicit **ground effect stall**
regime to the physics-informed aero model. In the previous model, downforce increased monotonically
as ride height decreased — an unphysical extrapolation below approximately 15 mm clearance.

The stall transition is implemented as a smooth sigmoid blend between the linear ground-effect
regime and the post-stall plateau:

```python
h_norm = (h_ride - h_stall) / sigma_stall
Fz_ge  = Fz_linear * sigmoid(h_norm * tau) + Fz_stall * (1 - sigmoid(h_norm * tau))
```

The stall threshold `h_stall` and transition sharpness `tau` are learned parameters, trained
against CFD data. The physics constraint Fz ∝ v² is enforced in the linear regime via a
physics-informed loss term in `residual_fitting.py` Phase 4. Test 19 verifies the v² scaling
holds within 1% in the linear regime and that the stall transition is smooth.

### 8.2 Maxwell ODE Damper with Thermal Dynamics

**File:** `models/damper_hysteresis.py`

The previous sigmoid-blend 4-way digressive damper model has been replaced by a **Maxwell ODE
damper** that explicitly models the viscoelastic character of a hydraulic damper and the thermal
dependence of damping force.

The Maxwell element is a spring (stiffness `k_m`) in series with a viscous dashpot (viscosity
`c_m`). The internal displacement `z_maxwell` evolves as:

$$\dot{z}_{maxwell} = \frac{F_{ext} - k_m z_{maxwell}}{c_m(T_{fluid})}$$

where `c_m(T_fluid) = c_m0 / (1 + α_T (T_fluid − T_ref))` — viscosity decreases linearly
with fluid temperature (thermal fade).

The fluid temperature `T_fluid` evolves as:

$$C_{fluid} \dot{T}_{fluid} = F_{damper} \cdot \dot{z}_{susp} - h_{conv}(T_{fluid} - T_{ambient})$$

This model produces three additional state variables per corner stored in `x[72:84]`:
`z_maxwell`, `F_internal`, and `T_fluid`. The Maxwell model naturally captures:
- Velocity-dependent hysteresis in the force-velocity diagram (characteristic figure-8 loop)
- Thermal fade during extended high-speed running (Endurance)
- Frequency-dependent apparent stiffness at high heave velocity (important for curb strikes)

### 8.3 Track Surface Model

**File:** `models/track_surface.py`

The track surface model adds two physics effects not previously represented:

**Rubber build-up.** Over the course of an event, tyre rubber deposits on the racing line
increase the available grip coefficient. The model tracks a rubber accumulation state per
track segment `ρ_rubber(s, t)` that evolves as:

$$\dot{\rho}_{rubber} = k_{dep} \cdot F_z \cdot v_{slip} - k_{decay} \cdot \rho_{rubber}$$

Grip modulation: `μ(ρ) = μ_asphalt · (1 + k_rubber · tanh(ρ / ρ_sat))`

**Grip asymmetry.** The model also represents the left-right grip asymmetry that develops when
the circuit has a predominant direction (as with most FS autocross layouts). The asymmetry state
`Δμ(s)` is initialised from MoTeC telemetry data and provides a per-side μ correction.

Both effects feed into the stochastic friction map `w_mu(s)` used by the Diff-WMPC tube
parameterisation and the Koopman Slip Observer.

---

## 9. Slip Observers — Koopman & RLS

### 9.1 Koopman Slip Observer (Primary)

**File:** `observers/koopman_slip.py`

The Koopman Slip Observer (KSO) replaces the previous Recursive Least-Squares estimator as
the **primary κ* (optimal slip ratio) estimator** from Batch 4 onward. It uses a Koopman
operator representation of the nonlinear tire slip dynamics, allowing a globally linear
predictor to track the optimal slip point across the full operating envelope.

**Architecture:**
- Lifting function `φ(e)` maps the 4D observation vector `(κ, F_x, ∂F_x/∂κ, v_x)` into
  a 32-dimensional lifted space
- Koopman operator `K_slip` is a 32×32 matrix trained offline via EDMD on TTC data
- The optimal slip point is the fixed point of `K_slip`: `κ* = argmax_κ F_x(κ)` estimated
  as the κ value at which the Koopman state derivative is zero
- Online update: `z_{k+1} = K_slip @ z_k` — a single matrix-vector product, ~3 µs post-JIT

**Why Koopman over RLS.** The RLS observer is a local linear model that must re-identify
the slip curve slope `∂F_x/∂κ` at each operating point. It works well near κ* but diverges
during transients when the tire is far from the optimal slip. The Koopman observer has global
validity — it was trained on the full (α, κ, Fz, γ) envelope and does not degrade during
transients.

The KSO output `κ*_est` feeds into the DESC traction control as the target slip reference.
The DESC inner loop still performs real-time extremum seeking, but now uses the KSO as a
warm start rather than tracking from scratch.

### 9.2 RLS Slip-Slope Observer (Secondary)

**File:** `observers/rls_tc.py`

The Recursive Least-Squares Slip-Slope Observer estimates the local slope `∂F_x/∂κ` in real
time using the standard RLS algorithm with exponential forgetting (`λ_forget = 0.98`).

It serves two roles: (a) a fallback κ* estimator when the KSO confidence score falls below
a threshold (e.g., sensor dropout, extreme operating conditions), and (b) a cross-validation
signal for the KSO — if the two estimates diverge by more than `Δκ = 0.02`, a diagnostic flag
is raised in the `PowertrainDiagnostics` output.

The RLS observer uses a 3-parameter linear model of the slip curve: `F_x ≈ a + b·κ + c·κ²`.
The optimal slip estimate is `κ*_rls = −b / (2c)` (vertex of the fitted parabola).

---

## 10. Differentiable Wavelet MPC (Diff-WMPC)

**File:** `optimization/ocp_solver.py`

### 10.1 Wavelet Parameterisation

Control trajectory `u(t)` over horizon N is represented in a 3-level Daubechies-4 (Db4) wavelet
basis. Optimization is performed in wavelet coefficient space:

$$u(t) = \sum_{j,k} w_{j,k} \cdot \psi_{j,k}(t)$$

**Bands:** `A3` (approximation), `D1`, `D2`, `D3` (detail at scales 1–3).

Benefits:
- Low-frequency acceleration profiles are compactly represented in `A3`
- High-frequency actuator chatter penalised in `D1/D2/D3` without explicit bandwidth constraint
- The wavelet coefficient vector is smaller than the raw `u` vector — cheaper gradient computation

### 10.2 Pseudo-Huber Wavelet Regularisation

Replaces the non-differentiable L1 sparsity penalty on detail bands:

$$\mathcal{R}_{PH}(w) = \sum_{j \in \{D1,D2,D3\}} \lambda_j \sum_k \left( \sqrt{1 + (w_{j,k}/\delta)^2} - 1 \right) \cdot \delta^2$$

Per-band weighting: D3 → 0.5×, D2 → 1.0×, D1 → 2.0× (higher frequency = stronger penalty).

### 10.3 Augmented Lagrangian Friction Constraint

$$\mathcal{L}_{AL} = f(x) + \lambda^T c(x) + \frac{\rho}{2} \left\| \max\!\left(c(x), -\frac{\lambda}{\rho}\right) \right\|^2$$

Constraint: `g_i = (a_lat² + a_lon²) / (μg)² − 1 ≤ 0`

Multipliers update: `λ ← max(λ + ρ·max(g, 0), 0)`. ρ grows 2× when max violation > 0.1.

### 10.4 Unscented Transform — Stochastic Tubes

5-point UT generates sigma trajectories from joint GP tire σ² + wind perturbations. All 5
simulated in parallel via `jax.vmap`. Track limits enforced against tube edges:

```
dist_left  = w_left  − (n_mean + κ_safe · √σ²_n)   ≥ 0
dist_right = w_right + (n_mean − κ_safe · √σ²_n)   ≥ 0
```

Both enforced via `−ε · log(softplus(dist * 50) / 50 + 1e-5)` — smooth log-barrier.

### 10.5 Physics P-Controller Warm Start

```
v_target = min(sqrt(μg / |κ|), 0.92 · V_limit)   [Newton units]
```

Warm start uses Newton force units matching the physics `u[1]` channel — essential for
preventing premature GTOL convergence after nit=1.

### 10.6 NaN Recovery Fallback

```python
fallback_loss = 1e9 + 0.5 * ||x − x_warmstart||²
fallback_grad = clip(x − x_warmstart, −100, 100)
```

The 1e9 offset exceeds any real loss; gradient points toward the warm start.

---

## 11. MORL-SB-TRPO Setup Optimiser

**File:** `optimization/evolutionary.py`

### 11.1 Overview

Multi-Objective RL over the full 40-dimensional `SuspensionSetup` space. Three Pareto axes:

1. **Maximum Grip** — skidpad objective [G]
2. **Dynamic Stability** — step-steer overshoot ≤ 5.0 rad/s
3. **Endurance LTE** — lap-time-energy composite over an 8-corner mini-lap

### 11.2 Chebyshev Ensemble Spacing

20 members with ω-weights Chebyshev-spaced on [0, 1]:

$$\omega_i = \frac{1}{2}\left(1 - \cos\!\left(\frac{i \pi}{N-1}\right)\right)$$

Concentrates ~65% of the ensemble into ω ∈ [0.7, 1.0] — the high-grip physically interesting region.

### 11.3 Riemannian Natural Policy Gradient (RNPG)

Replaces the KL-divergence trust region with a Riemannian metric pulled back through the
physics engine:

$$G_{phys,k} = J_k^T \text{diag}(s) J_k + \lambda \cdot \text{diag}(J_k^T S J_k + \varepsilon I)$$

where `s = [1.0, 0.2, 0.5]` (grip weighted 5× over stability) and `J_k = ∂[grip, stab, LTE]/∂μ_k`.

### 11.4 ARD Bayesian Optimisation Cold Start

10 random initialisations + 30 EI-guided acquisitions using an ARD squared-exponential GP.
Per-dimension lengthscales learned via correlation heuristic — insensitive dimensions
acquire large lengthscale and are effectively pruned. Best 5 diverse basins seed the ensemble.

### 11.5 Endurance LTE Objective

**File:** `optimization/objectives.py`

Simulates an 8-corner mini-lap (120 steps × dt=5ms = 0.6 s) with a P-controlled driver.
Composite score (HIGHER = BETTER):
`J_LTE = w_speed · v̄ − w_energy · E_total − w_thermal · ΔT_penalty`

### 11.6 SMS-EMOA Hypervolume Archive

3D hypervolume contribution pruning for the (grip, stability, LTE) Pareto front.
Archive capped at 150 points. Stability hard filter: setups with overshoot > 5.0 rad/s are
excluded from the archive — not just penalised.

---

## 12. Powertrain Control Stack

**Directory:** `powertrain/`

### 12.1 Unified Manager — 14-Stage Pipeline

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

### 12.2 Koopman TV Controller — Current Status

**File:** `powertrain/modes/advanced/koopman_tv.py`

The Koopman TV controller is architecturally complete (Dictionary-Switched Koopman LQR,
EDMD-DL, 3 grip regimes at ρ < 0.70 / 0.70–0.92 / > 0.92). However:

**`trained_blend = 0.0` is locked pending H_net retraining.**

The Koopman TV operators were trained on rollouts generated with the 46-DOF vehicle model.
The 108-DOF state vector changes the tire force predictions (via the 3D thermal and 2nd-order
transient models) and the suspension forces (via the Maxwell damper and Bouc-Wen compliance).
These changes mean the existing Koopman operators are trained on a distribution that no longer
matches the current dynamics. Deploying them at any `trained_blend > 0` would degrade yaw
moment tracking. The operators must be retrained via `scripts/train_koopman_tv.py` on
rollouts from the updated vehicle model before `trained_blend` is raised.

The pure PD fallback (`trained_blend = 0.0`) remains fully operational.

### 12.3 mpQP KKT Torque Allocator

**File:** `powertrain/modes/advanced/explicit_mpqp_allocator.py`

V2 KKT allocator (24×24, slip-aware). Single KKT linear solve conditioned on the active-set
classifier prediction — ~40–90 µs post-JIT.

| Version | Constraints | KKT Dimension | Active-set input dim |
|---|---|---|---|
| V1 | 12 (box + friction) | 16×16 | 15-dim θ |
| V2 | 20 (box + friction + 8 slip-CBF) | 24×24 | 19-dim θ |

### 12.4 Other Powertrain Stages

Sections 12.2–12.10 from the GP-vX4 README remain accurate for the motor model, virtual
impedance, regen blend, slip barrier, CBF safety filter, DESC traction control, launch control,
and other subsystems. These components were not structurally modified in the last three weeks.
Their detailed descriptions are retained from GP-vX4 and are not repeated here to avoid
document inflation.

---

## 13. Differentiable EKF — Online Parameter Estimation

**File:** `models/differentiable_ekf.py`

### 13.1 State & Observation

5-state parameter estimator running at every physics step:

```
θ = [λ_μ_f, λ_μ_r, T_opt, h_cg, α_peak]   — parameter state
y = [ay_measured, wz_measured]               — observations (standard MoTeC ADL3)
```

### 13.2 JAX Jacobian (No Finite Differences)

Because the vehicle dynamics are fully differentiable, the EKF measurement Jacobian is computed
exactly via `jax.jacobian` — no finite differences, no approximation errors.

$$H = \frac{\partial [a_y^{sim}, \dot{\psi}^{sim}]}{\partial [\lambda_\mu, T_{opt}, h_{cg}, \alpha_{peak}]}$$

### 13.3 Convergence

| Parameter | Initial Uncertainty | Convergence After |
|---|---|---|
| λ_μ_f, λ_μ_r | ±20% | 3–5 laps → ±3% |
| T_opt | ±10°C | 3–5 laps → ±5°C |
| h_cg | ±20 mm | 3–5 laps → ±8 mm |
| α_peak | ±0.04 rad | 5–8 laps → ±0.01 rad |

---

## 14. Full-Lap Differentiable Simulation

**File:** `optimization/differentiable_lap_sim.py`

### 14.1 The Kill Feature

```python
∂(lap_time) / ∂(setup_vector) = jax.grad(simulate_full_lap)(setup)
```

The gradient traces from the scalar lap time back through:

```
lap_time → speed profile → tire forces → suspension loads → setup params
```

in a single `jax.lax.scan` over the entire lap.

### 14.2 Differentiable Track

**File:** `optimization/differentiable_track.py`

JAX-native cubic spline interpolation on the SE(3) manifold. Curvature computed analytically
(`κ = ω_z / v_xy`). NSDE stochastic friction uncertainty map `w_mu` from lateral slip variance,
augmented with the rubber build-up and grip asymmetry states from `models/track_surface.py`.

### 14.3 Architecture

```python
@jax.jit
def simulate_full_lap(setup: jax.Array) -> jax.Array:   # returns scalar lap_time
    init_state = make_initial_state_108dof()              # 108-dimensional initial state
    def step_fn(carry, _):
        state, t = carry
        u = path_following_controller(state, track)
        state_new = vehicle.step(state, u, setup, dt)
        return (state_new, t + dt), state_new.vx
    (final, _), vx_traj = jax.lax.scan(step_fn, init_state, None, length=N_steps)
    return N_steps * dt + 0.0 * jnp.sum(vx_traj)
```

### 14.4 Gradient Server

**File:** `scripts/gradient_server.py`

HTTP + WebSocket server exposing on-demand gradient computation to the React dashboard.
Receives 40D setup vector as JSON; returns 40D sensitivity vector `∂t_lap/∂s`.
The full `jax.grad(simulate_full_lap)` is available for offline batch computation.

---

## 15. CMA-ES Crossover Validator

**File:** `optimization/cmaes_validator.py`

Independent gradient-free optimizer validating the MORL-SB-TRPO Pareto front. Pure NumPy
(μ/μ_w, λ)-CMA-ES with rank-1 and rank-μ updates, operating in normalised [0,1]^40 space.

Workflow: run MORL → Pareto front CSV → run CMA-ES from 5 random + 5 MORL-seeded initialisations
→ compute hypervolume comparison → PASS if CMA-ES does not dominate the MORL front.

---

## 16. Suspension Kinematic Analysis

Covered in full in Section 6 (Suspension Package). The kinematic analysis stack now resides
in the `suspension/` package and encompasses the full double A-arm solver, Bouc-Wen bushing
model, heave/steer/roll sweep analysis, and OKinematic-format Excel report generation.

---

## 17. Simulator Architecture

```
┌─────────────────────────┐  ┌────────────┐   ┌──────────────────────────────┐
│  React Dashboard        │  │ ws_bridge  │   │ gradient_server.py           │
│  (Vercel, 8 groups,     │←─│ UDP→WS     │   │ HTTP: ∂(lap_time)/∂(setup)   │
│   18 modules)           │  │ 60 Hz      │   │ WS: streaming sensitivities  │
│                         │  └────────────┘   └──────────────────────────────┘
│  useLiveTelemetry.js    │
└─────────────────────────┘
┌─────────────────────────┐  ┌────────────┐
│  physics_server.py      │  │ ws_bridge  │
│  200 Hz JAX integration │─→│ UDP→WS     │   ┌──────────────────────────────┐
│  108-DOF Port-Hamiltonian│  └────────────┘   │ Godot 3D Visualizer          │
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

The physics server now integrates the full 108-DOF GLRK-4 loop, including the suspension
elastokinematic ODE, Maxwell damper ODE, and 3D tire thermal model. The 256-byte UDP telemetry
packet has been extended accordingly (expanded state broadcast fields).

---

## 18. React Engineering Dashboard

**Directory:** `visualization/dashboard_react/`

Deployed on Vercel. 8 navigation groups, 18 modules, LIVE/ANALYZE mode toggle. No structural
changes since GP-vX4.

### Navigation Groups & Modules

| Group | Accent | Modules |
|---|---|---|
| **Overview** | cyan | Architecture explainer — H_net, Tire Model, Diff-WMPC, MORL-SB-TRPO |
| **Telemetry** | green | Live charts: speed, lateral G, Fz per corner, tire temps, yaw rate |
| **Vehicle Dynamics** | amber | Setup Opt (3D Pareto front), Suspension (LLTD, roll gradient, damper power), Setup Explorer (40-param interactive + 3D Pareto scatter), Tire Physics (7-node thermal, GP envelope, hysteresis), Weight & CG |
| **Aerodynamics** | pink | Ground-effect stall Fz vs ride height, drag vs speed, aero balance |
| **Electronics** | purple | Motor temperatures, SoC, inverter power, battery voltage |
| **Powertrain** | orange | Powertrain Ctrl (14-stage pipeline visualisation), TV Simulator (interactive torque vectoring simulator) |
| **Controls & AI** | cyan | Energy Audit (Hamiltonian landscape, R-matrix heatmap), ∇ Insights (setup sensitivities ∂t_lap/∂param), Research (MORL convergence, car selector, parameter freeze UI) |
| **Performance** | red | Driver Coaching, Endurance Strategy (regen trade-off), Compliance (FSG rules) |

---

## 19. Sanity Check Suite (Tests 1–25)

**File:** `sanity_checks.py`

The suite has been extended from 16 to 25 tests, covering the five new model files and the
two new observer modules introduced in GP-vX5.

### Physics Subsystem Tests (1–9) — Unchanged

| Test | Name | What It Verifies |
|---|---|---|
| 1 | Neural Convergence | H_net + R_net gradient flow; power injection < 500 J/s |
| 2 | Forward Pass | 108-DOF GLRK-4 step; state finite; energy conserved to < 1 J/step |
| 3 | Circular Track | Circular-arc simulation; lateral acceleration within friction circle |
| 4 | Friction Circle | `sqrt(Fx² + Fy²) ≤ mu * Fz` across full (α, κ) grid; ratio < 1.10 |
| 5 | Load Sensitivity | Fy_peak increases monotonically with Fz (degressive slope) |
| 6 | Diagonal Load Transfer | Correct corner Fz at 1.5G cornering + 0.5G braking |
| 7 | Aero Scaling | `Fz_aero ∝ v²` within 1% in linear regime; stall transition smooth |
| 8 | Differential Yaw Moment | Non-zero yaw moment from differential torque |
| 9 | Optimizer Boundary Diversity | k_f values not pinned at lower bound; stability cap active |

### Powertrain Control Tests (10–16) — Unchanged

| Test | Name | What It Verifies |
|---|---|---|
| 10 | Motor Torque Envelope | Low-speed torque ≈ T_peak (< 5%); monotone decrease in field-weakening |
| 11 | SOCP Allocator | Feasibility in 12 iterations; yaw moment tracking error < 5% |
| 12 | CBF Safety | CBF barrier h(x) ≥ 0 enforced; unsafe T_cmd filtered |
| 13 | DESC Convergence | κ* converges within adaptive dither schedule; error < 0.025 |
| 14 | Launch State Machine | Full IDLE→ARMED→PROBE→LAUNCH→HANDOFF→TC traversal |
| 15 | Virtual Impedance | Phase lag > 30° at 3 Hz; steady-state error < 5% |
| 16 | Full Pipeline JIT | Per-step time < 5 ms on SBC budget; 28 diagnostic fields finite |

### New Model Tests (17–25) — GP-vX5 Additions

| Test | Name | What It Verifies |
|---|---|---|
| 17 | Suspension Kinematics | IFD chain: `∂camber/∂delta_L_tr` finite and consistent with FD; Newton convergence to 1e-9 |
| 18 | Elastokinematics | Bouc-Wen dissipation non-negative; compliance states bounded; `∂F_elasto/∂x` finite |
| 19 | AeroPlatformModel | Fz ∝ v² within 1% in linear regime; ground effect stall transition at h_stall; C∞ |
| 20 | Maxwell Damper | F-v diagram shows hysteresis loop; T_fluid increases monotonically during sustained oscillation |
| 21 | Tire Thermal 3D | Lateral asymmetry: outer rib hotter than inner under positive camber; 7 nodes finite |
| 22 | Tire Transient 2nd-order | Lateral force overshoot during rapid steer input > 0% (belt relaxation effect present) |
| 23 | Track Surface | Rubber build-up state increases during slip; grip modulation bounded in [1.0, 1 + k_rubber] |
| 24 | Koopman Slip Observer | κ* converges to within 0.005 of DESC estimate on synthetic ramp test |
| 25 | RLS Observer Fallback | RLS κ* matches KSO within divergence threshold; diagnostic flag raised when they disagree |

---

## 20. Known Issues, Limits & Diagnostics

### 20.1 H_net Retraining Required

**This is the primary blocking issue for production operation.** The H_net and R_net weights
stored in `models/h_net.bytes` and `models/r_net.bytes` were trained against the 46-DOF state
vector. The 108-DOF state vector is architecturally incompatible — the FiLM-conditioned input
layer now receives a 108-dimensional feature vector instead of 46 dimensions. The existing
weights will produce garbage energy estimates and must be retrained before:

- Running Diff-WMPC in production mode (Test 1 will fail passivity check)
- Enabling `trained_blend > 0` on the Koopman TV controller
- Running MORL in full-physics mode

Retraining command: `python -c "from optimization.residual_fitting import train_neural_residuals; train_neural_residuals()"`

Test 1 passivity threshold is currently set at 500 J/s. Post-retraining, the target is < 50 J/s.

### 20.2 Koopman TV `trained_blend = 0.0` Lock

As described in Section 12.2, the Koopman TV operators must be retrained on 108-DOF rollouts
before deployment. The lock is enforced by a runtime assertion in `koopman_tv.py`:

```python
assert trained_blend == 0.0, (
    "Koopman TV operators trained on 46-DOF rollouts. "
    "Retrain via scripts/train_koopman_tv.py before raising trained_blend."
)
```

### 20.3 State Vector Migration

Code outside the `models/`, `suspension/`, and `observers/` directories that references
specific state indices `x[28:38]`, `x[38:46]` from the 46-DOF layout will produce silently
incorrect results. All such references must be updated to the 108-DOF index layout before
full integration testing. Known migration locations:
- `simulator/physics_server.py`: UDP telemetry packet construction references old state indices
- `optimization/differentiable_lap_sim.py`: initial state construction
- `tests/test_verification.py`: thermal state assertions

### 20.4 XLA Compilation Times

The 108-DOF state vector and the additional ODEs (Maxwell, Bouc-Wen, 3D thermal) increase XLA
compilation time relative to GP-vX4. Updated benchmarks:

| Subsystem | First compile | Cached |
|---|---|---|
| Vehicle dynamics (GLRK-4, 108-DOF) | ~5–8 min CPU | ~8–20 s |
| WMPC full trajectory | ~10–15 min CPU | ~25–50 s |
| MORL gradient | ~6–10 min CPU | ~12–25 s |
| Powertrain step | ~2–4 min CPU | ~3–8 s |
| Gradient server (short horizon) | ~1–2 min CPU | ~3–8 s |

Import `jax_config` **before any other JAX import** in every script.

### 20.5 Float32 Limits

- Bouc-Wen `z_BW` saturates at `±z_sat` via `jnp.tanh(z_BW / z_sat) * z_sat` — prevents
  float32 overflow during extreme compliance events
- Maxwell `T_fluid` clipped to `[T_ambient − 10, T_ambient + 180]` °C
- 3D thermal nodes clipped to `[−10, 200]` °C per node
- GP Gram matrix: `jitter = 1e-3 * I` (unchanged)

### 20.6 Dashboard Telemetry Authenticity

Dashboard modules displaying tire thermal data, damper hysteresis, and track surface state
currently show synthetic RNG data. The `ws_bridge.py` real-time connection is implemented
but the expanded 108-DOF state packet broadcast is not yet wired through to the relevant
`TirePhysicsModule.jsx` and `SuspensionModule.jsx` panels. This remains a competition-critical
gap for the FSG Siemens Digital Twin Award criteria.

### 20.7 Three Existential Safety Threats (FSG 2026)

1. **CBF actuator delay** — addressed by input-delay DCBF in `cbf_safety_filter()`; slip barrier
   adds a second defense layer via predictive slip CBF in the KKT system
2. **EMI/bit-flip risks in SBC memory** — ECC memory or CRC-checked state transfer required
3. **4WD ABS v_x anchoring** — during simultaneous multi-wheel lockup, v_x estimate from
   wheel speeds collapses; requires IMU-based v_x fusion (not yet implemented)

---

## 21. Pipeline Execution

```bash
# 0. Activate environment
source project_gp_env/bin/activate

# 1. Verify all subsystems (Tests 1–25, ~30 min first run due to compilation)
python sanity_checks.py

# 2. Train H_net / R_net against TTC tire data and synthetic 108-DOF physics rollouts
#    REQUIRED after state vector change — existing h_net.bytes is incompatible
python -c "from optimization.residual_fitting import train_neural_residuals; train_neural_residuals()"

# 3. Offline Koopman TV training on 108-DOF rollouts
#    Run AFTER H_net retraining — Koopman operators need the updated dynamics
python scripts/train_koopman_tv.py

# 4. Run MORL setup optimiser — Phase 1 (all params free, ~55 min with 108-DOF)
python scripts/run_ter27_design_exploration.py --phase 1

# 4b. Phase 2 (CAD locked geometry, ~35 min)
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
| `models/h_net.bytes` | Trained H_net weights — must be retrained for 108-DOF |
| `models/r_net.bytes` | Trained R_net weights — must be retrained for 108-DOF |
| `models/aero_net.bytes` | Trained AeroPlatformModel weights |
| `models/h_net_scale.txt` | Training normalisation diagnostic (not architectural) |
| `suspension/Front_Ter27.xlsx` | Kinematic sweep + optimized setup (OKinematic format) |
| `suspension/Rear_Ter27.xlsx` | Rear kinematic sweep + optimized setup |

---

## 22. Revision History (GP-vX1 → vX5)

### GP-vX5 (Current)

**Architectural Change: 46-DOF → 108-DOF State Vector**

This is the defining change of GP-vX5. The state vector has been expanded from 46 to 108
dimensions by integrating four new physics subsystems directly into the vehicle ODE. The
key motivation is that the previous model significantly underestimated tire force transients
during chicanes (first-order slip lag insufficient), underestimated damper thermal fade
during Endurance (no fluid temperature model), and could not represent bushing compliance
effects on understeer gradient (kinematic-only suspension model).

**New Packages & Files:**

| Module | File | Description |
|---|---|---|
| Suspension Package | `suspension/` | New top-level package; full double A-arm kinematics + elastokinematics |
| Kinematic Solver | `suspension/kinematics.py` | IFD + custom VJP; FIX-1 through FIX-4; full double A-arm geometry |
| Elastokinematic Model | `suspension/elastokinematics.py` | Nonlinear Bouc-Wen bushing model; 24 compliance states in x[84:108] |
| AeroPlatformModel | `models/aero_platform.py` | Ground effect stall; replaces DifferentiableAeroMap |
| Maxwell ODE Damper | `models/damper_hysteresis.py` | Viscoelastic Maxwell element + thermal fade; 12 states in x[72:84] |
| 3D Tire Thermal | `models/tire_thermal_3d.py` | 7 nodes per corner, lateral asymmetry; 28 states in x[28:56] |
| 2nd-order Tire Transient | `models/tire_transient.py` | Belt + carcass relaxation; 16 states in x[56:72] |
| Track Surface | `models/track_surface.py` | Rubber build-up + grip asymmetry |
| Koopman Slip Observer | `observers/koopman_slip.py` | Primary κ* estimator; EDMD-DL; replaces RLS as primary, Batch 4 |
| RLS TC Observer | `observers/rls_tc.py` | Secondary κ* estimator; divergence detection vs KSO |

**Batch Upgrades:**

| ID | Component | Change |
|---|---|---|
| BATCH-R | State Vector | 46-DOF → 108-DOF: kinematics (0:28), thermal 3D (28:56), transient slip 2nd-order (56:72), damper hysteresis (72:84), elastokinematic compliance (84:108) |
| BATCH-S | TireThermal | 5-node axisymmetric per-axle → 7-node 3D per-corner with lateral asymmetry (camber load distribution) |
| BATCH-T | TireTransient | First-order carcass lag → second-order carcass + belt relaxation (16 states vs 8) |
| BATCH-U | Damper | Sigmoid-blend 4-way digressive → Maxwell ODE with thermal dynamics (hysteresis + fade) |
| BATCH-V | Suspension | Polynomial-approximation kinematics → full double A-arm IFD solver (kinematics.py) |
| BATCH-W | Suspension | Rigid bushing assumption → nonlinear Bouc-Wen elastokinematic bushing model (elastokinematics.py) |
| BATCH-X | Aero | DifferentiableAeroMap → AeroPlatformModel with ground effect stall |
| BATCH-Y | Track | Static μ map → dynamic rubber build-up + grip asymmetry (track_surface.py) |
| BATCH-Z | Observers | RLS as sole κ* estimator → Koopman Slip Observer (primary) + RLS (secondary, divergence check) |
| BATCH-4 | KoopmanTV | `trained_blend` locked at 0.0 pending H_net retraining on 108-DOF rollouts |
| BATCH-AA | Sanity Checks | Tests 1–16 → Tests 1–25 (9 new tests for all new models and observers) |

### GP-vX4

**New Modules:**

| Module | File | Description |
|---|---|---|
| mpQP KKT Allocator V2 | `powertrain/modes/advanced/explicit_mpqp_allocator.py` | 24×24 KKT, slip-aware, ~40–90 µs post-JIT |
| Active-Set Classifier V2 | `powertrain/modes/advanced/active_set_classifier.py` | V2: 19-dim θ |
| Slip Barrier | `powertrain/modes/advanced/slip_barrier.py` | Predictive slip CBF rows for KKT |
| Dynamic Regen Blend | `powertrain/regen_blend.py` | KKT-optimal α*; battery/temperature derating |
| Design Freeze | `config/design_freeze.py` | 3-phase progressive lockdown |
| Suspension Kinematics (v1) | `suspension/kinematics.py` | JAX-vmap sweep; precursor to full IFD solver |
| Suspension Excel Writer | `suspension/excel_writer.py` | OKinematic-format XLSX |

**Batch Upgrades:**

| ID | Component | Change |
|---|---|---|
| BATCH-K | SuspensionSetup | Expanded 28D → 40D: heave springs, inerters, rising-rate MR, anti-squat-F, Ackermann, independent LS/HS rebound |
| BATCH-L | DesignFreeze | 3-phase progressive lockdown; tanh projection → zero gradient for frozen params |
| BATCH-M | mpQP Allocator | V2 KKT (24×24) with 8 slip-barrier constraints |
| BATCH-N | RegenBlend | Dynamic α* replacing fixed scalar; hydraulic brake interface |
| BATCH-O | Dashboard | Expanded 13 → 18 modules, 7 → 8 nav groups |
| BATCH-P | GradientServer | HTTP/WebSocket gradient server |
| BATCH-Q | SuspensionKinematics | JAX-vmap heave/steer/roll sweeps; Excel report generation |

### GP-vX3

**New Modules:** Powertrain Manager, Motor Model, Virtual Impedance, Torque Vectoring,
Traction Control, Launch Control v2.1, Koopman TV, Differentiable EKF, Differentiable Track,
Lap Simulation, CMA-ES Validator, Physics Server, WS Bridge, ROS 2 Bridge.

**Batch Upgrades:**

| ID | Component | Change |
|---|---|---|
| BATCH-A | TireOperatorPINN | PINN training pipeline against TTC data |
| BATCH-B | SparseGP + CBF | Robust CBF with GP uncertainty |
| BATCH-C | DifferentiableAeroMap | Physics-Informed training: Fz ∝ v² constraint |
| BATCH-D | DifferentiableEKF | α_peak added as 5th estimated state |
| BATCH-E | DifferentiableTrack | JAX-native cubic spline on SE(3) manifold |
| BATCH-F | MORL evolutionary | RNPG replaces KL trust region |
| BATCH-G | objectives.py | Endurance LTE as 3rd Pareto axis |
| BATCH-H | ocp_solver.py | Pseudo-Huber wavelet regularisation replaces L1 |
| BATCH-I | Sanity checks | Tests 10–16: full powertrain stack verification |
| BATCH-J | React Dashboard | 13-module dashboard; WS LIVE mode |

**Bug Fixes:**

| ID | Component | Issue | Fix |
|---|---|---|---|
| BUG-PT-1 | `cbf_safety_filter` | `AttributeError: 'CBFParams' has no attribute 'r_w'` — positional shift from adding `gp_sigma` | Updated all 4 call sites with explicit keyword arguments |
| BUG-PT-2 | `DESCState` | `TypeError: DESCState.__new__() missing 't_acc'` — `t_acc` added to NamedTuple without updating `default()` | Added `t_acc` to both factory functions |

### GP-vX2

**Critical Architecture Fixes:**

| ID | Component | Issue | Fix |
|---|---|---|---|
| BUGFIX-4 | H_net | `h_net_scale.txt` (102.62) passed into architecture → 102× energy amplification | `h_scale=1.0` always; scale file is diagnostic only |
| BUGFIX-5 | H_net | `susp_sq` gate at z=0 (never occupied during operation) | `susp_sq = Σ(q[6:10] − _Z_EQ)² + 1e-4` |
| BUGFIX-7 | TireModel | Thermal layout misaligned | Reindexed to match `x[28:38]` exactly |
| BUGFIX-8 | TireModel | `tire.operator AttributeError` in `diagnose.py` | `@property operator` returns `_pinn_module` |

### GP-vX1

| ID | Component | Change |
|---|---|---|
| BUGFIX-1 | SuspensionSetup | Index alignment — ARB at [2:4], not [4:6] |
| BUGFIX-2 | Dynamics | Mass/inertia defaults corrected for Ter27 |
| BUGFIX-3 | R_net | `_TRIL_14` defined at module level |
| UPGRADE-1–13 | Various | H_net FiLM conditioning; R_net log-diagonal guarantee; GLRK-4 integrator; C∞ bumpstop; 4-way digressive damper; Pacejka Fz floor; WMPC canonical builder; MORL 28-dim setup; ARD BO cold-start; SMS-EMOA HV pruning |

---

*Project-GP is a live research codebase. Physics is correct to the extent that `sanity_checks.py`
passes all 25 tests. The H_net and R_net weights (`models/h_net.bytes`, `models/r_net.bytes`)
must be retrained via `residual_fitting.py` after the 46→108-DOF state vector change before
running WMPC or MORL in production mode — this is the current top-priority task. The Koopman
TV controller (`trained_blend = 0.0` locked) and the Koopman Slip Observer both require
offline EDMD training on 108-DOF rollouts before production deployment. The mpQP KKT allocator
requires offline active-set classifier training before the KKT path activates; the
projected-gradient fallback is always available.*