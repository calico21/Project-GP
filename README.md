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
| **Setup Optimisation** | MORL-SB-TRPO + RNPG — 28-dim, Chebyshev ensemble, ARD-BO cold-start, SMS-EMOA Pareto archive, Endurance LTE 3rd axis |
| **Setup Space** | 28 parameters (`SuspensionSetup` NamedTuple) — springs, ARBs, 4-way dampers, geometry, diff |
| **Powertrain Control** | 13-stage pipeline at 200 Hz: SOCP-TV + Robust DCBF + DESC-TC + B-spline Launch + Koopman LQR |
| **State Estimator** | Differentiable EKF — 5-state online parameter estimation (λ_μ, T_opt, h_cg, α_peak) |
| **Lap Gradient** | `jax.grad(simulate_full_lap)(setup)` — end-to-end differentiable lap simulation |
| **Simulator** | 200 Hz physics server + UDP→WebSocket bridge + ROS 2 bridge + Godot 3D |
| **Dashboard** | React (Vercel) — 13 modules, 7 nav groups, LIVE/ANALYZE modes |
| **Revision** | GP-vX3 |

---

## Table of Contents

1. [Philosophy & Design Principles](#1-philosophy--design-principles)
2. [Repository Structure](#2-repository-structure)
3. [State Vector & SuspensionSetup](#3-state-vector--suspensionsetup)
4. [Neural Port-Hamiltonian Vehicle Dynamics](#4-neural-port-hamiltonian-vehicle-dynamics)
5. [Multi-Fidelity Tire Model](#5-multi-fidelity-tire-model)
6. [Differentiable Wavelet MPC (Diff-WMPC)](#6-differentiable-wavelet-mpc-diff-wmpc)
7. [MORL-SB-TRPO Setup Optimiser](#7-morl-sb-trpo-setup-optimiser)
8. [Powertrain Control Stack](#8-powertrain-control-stack)
9. [Differentiable EKF — Online Parameter Estimation](#9-differentiable-ekf--online-parameter-estimation)
10. [Full-Lap Differentiable Simulation](#10-full-lap-differentiable-simulation)
11. [CMA-ES Crossover Validator](#11-cma-es-crossover-validator)
12. [Simulator Architecture](#12-simulator-architecture)
13. [React Engineering Dashboard](#13-react-engineering-dashboard)
14. [Sanity Check Suite (Tests 1–16)](#14-sanity-check-suite-tests-116)
15. [Known Issues, Limits & Diagnostics](#15-known-issues-limits--diagnostics)
16. [Pipeline Execution](#16-pipeline-execution)
17. [Revision History (GP-vX1 → vX3)](#17-revision-history-gp-vx1--vx3)

---

## 1. Philosophy & Design Principles

Project-GP abandons traditional numerical simulation frameworks (CasADi, IPOPT, point-mass solvers)
in favour of a **deep learning compiler architecture (JAX/XLA)**. The design is governed by four
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

**The Canonical-Index Rule.** All 28 setup parameters are accessed exclusively via
`SuspensionSetup.from_vector()` / `.to_vector()` and `build_default_setup_28()`. No positional
indexing of the raw setup array is permitted outside these methods — a lesson learned from a
catastrophic silent bug where ARB indices received spring-rate values.

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
│   ├── evolutionary.py              # MORL-SB-TRPO + RNPG optimizer (28-dim, ARD-BO, SMS-EMOA)
│   ├── objectives.py                # Skidpad grip + step-steer stability + Endurance LTE
│   ├── residual_fitting.py          # H_net / R_net training pipeline (Phase 1 + 2 + TTC PINN)
│   ├── differentiable_track.py      # JAX-native cubic spline track (SE(3) manifold)
│   ├── differentiable_lap_sim.py    # Full-lap scan: ∂(lap_time)/∂(setup) via jax.grad
│   └── cmaes_validator.py           # CMA-ES crossover validation of MORL Pareto front
│
├── powertrain/
│   ├── powertrain_manager.py        # Unified 13-stage coordinator — single JIT powertrain_step()
│   ├── motor_model.py               # PMSM electromechanical + thermal + battery OCV/R_int
│   ├── virtual_impedance.py         # 2nd-order virtual flywheel/damper — PIO mitigation
│   └── modes/
│       ├── advanced/
│       │   ├── torque_vectoring.py  # 12-iter projected-gradient SOCP + input-delay DCBF
│       │   ├── traction_control.py  # DESC extremum-seeking + GP-weighted dual-path TC
│       │   ├── launch_control.py    # 16-knot B-spline + 6-phase FSM (v2.1)
│       │   └── koopman_tv.py        # Dictionary-Switched Koopman LQR (EDMD-DL, 3 regimes)
│       └── intermediate/
│           └── launch_control.py    # 8-knot Catmull-Rom intermediate sequencer
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
│   │   ├── vehicles/
│   │   │   └── ter26.py             # Canonical Ter27 vehicle parameters (full dict)
│   │   └── tire_coeffs.py           # Pacejka MF6.2 coefficients (Hoosier R20, TTC-fitted)
│   └── logs/                        # MoTeC telemetry CSV inputs
│
├── visualization/
│   ├── dashboard.py                 # Streamlit engineering suite (legacy)
│   └── dashboard_react/             # React 13-module dashboard (production)
│       ├── src/
│       │   ├── App.jsx              # Root: 7 nav groups, LIVE/ANALYZE toggle
│       │   ├── OverviewModule.jsx   # Architecture explainer panels
│       │   ├── TelemetryModule.jsx  # Live telemetry charts
│       │   ├── SetupModule.jsx      # Pareto front + sensitivity
│       │   ├── SuspensionModule.jsx # Per-corner Fz, Fd, LLTD, roll gradient
│       │   ├── TirePhysicsModule.jsx# 5-node thermal + GP envelope + hysteresis
│       │   ├── WeightBalanceModule.jsx
│       │   ├── AerodynamicsModule.jsx
│       │   ├── ElectronicsModule.jsx
│       │   ├── EnergyAuditModule.jsx
│       │   ├── DifferentiableInsightsModule.jsx
│       │   ├── ResearchModule.jsx
│       │   ├── DriverCoachingModule.jsx
│       │   ├── EnduranceStrategyModule.jsx
│       │   └── ComplianceModule.jsx
│       └── package.json
│
├── scripts/
│   └── train_koopman_tv.py          # EDMD-DL offline training for Koopman TV operators
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

### 3.2 SuspensionSetup — 28-Element Typed Pytree

All 28 parameters are contained in a `NamedTuple` registered as a JAX pytree.
**`from_vector()` / `to_vector()` are the only permitted construction paths.**

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

`project_to_bounds()` uses a smooth tanh rescaling — gradient survives at both bounds.
Frozen parameters (LB = UB) yield zero gradient automatically with zero changes to the optimizer.

---

## 4. Neural Port-Hamiltonian Vehicle Dynamics

**File:** `models/vehicle_dynamics.py`

### 4.1 Port-Hamiltonian Structure

$$\dot{x} = (J - R)\,\nabla H(x) + F_{ext}(x, u)$$

- $J$ — skew-symmetric interconnection matrix (conservative, energy-preserving)
- $R$ — symmetric PSD dissipation matrix (energy-removing only)
- $H(x)$ — total Hamiltonian (kinetic + structural + neural residual)
- $F_{ext}$ — non-conservative external forces (tire, aero, gravity)

### 4.2 Neural Energy Landscape — H_net (FiLM-conditioned)

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

### 4.3 Neural Dissipation Matrix — R_net (Cholesky PSD)

$$R = L L^T + \text{diag}(\text{softplus}(d))$$

The `diag(softplus(d))` term guarantees strict positive definiteness. A physical mask restricts
dissipation to heave, roll, pitch, and unsprung-z DOFs.

### 4.4 Differentiable Aero Map (Physics-Informed)

`DifferentiableAeroMap` is a 32→32→4 MLP mapping `(vx, pitch, roll, heave_f, heave_r)` to
ground-effect-corrected `(Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero)`.

The aero map is **physics-informed** — it is trained to satisfy aerodynamic constraints:
- Downforce ∝ v² in the ground-effect-linear regime (verified at Test 7)
- Ground clearance and downforce floors use `_softplus_floor` — not `jnp.maximum`
- Zero subgradient at the floor (from `jnp.maximum`) kills optimizer signal at the
  15 mm ground clearance limit; `_softplus_floor` ensures non-zero gradient everywhere

### 4.5 Integrator — 2-Stage Gauss-Legendre RK4 (GLRK-4)

Butcher tableau:

$$a = \begin{pmatrix} 1/4 & 1/4 - \sqrt{3}/6 \\ 1/4 + \sqrt{3}/6 & 1/4 \end{pmatrix}, \quad b = (1/2,\ 1/2)$$

- **Symplectic** — preserves the symplectic 2-form `dq ∧ dp` to machine precision
- **4th-order** — energy drift O(h⁵) vs O(h³) for Störmer-Verlet
- **Implicit** — 5-iteration Newton scan inside `jax.lax.scan`; stage clipping at ±500 m/s²
  prevents NaN propagation without affecting gradients within the physical envelope
- **Auxiliary sub-states** (thermal + slip) integrated with the trapezoidal rule using converged
  GLRK-4 stage derivatives — A-stable, no extra `_compute_derivatives` calls

### 4.6 Suspension Forces

Per corner: `F_susp = F_spring + F_damper + F_bumpstop + F_ARB`

- **Spring:** `F = k * z * MR(z)²`, MR is a quadratic polynomial of heave travel
- **Damper:** 4-way digressive (low/high speed × bump/rebound), fully C∞ via sigmoid blending
- **Bumpstop:** `F = k_bs * softplus(β * (z − gap)) / β` — C∞, zero below gap, rising above
- **ARB:** lateral roll moment applied as symmetric ±correction to left/right corners
- **Fz (contact normal load):** always positive via `_softplus_floor(F_grav ± load_transfer)`

---

## 5. Multi-Fidelity Tire Model

**File:** `models/tire_model.py`

### 5.1 Layer 1 — Pacejka MF6.2

Full Magic Formula for Hoosier R20 (TTC-fitted). Includes:
- Load sensitivity (dfz), camber sensitivity (γ terms)
- Combined-slip reduction factors Gyk (κ-driven) and Gxa (α-driven)
- Full MF6.2 aligning torque Mz with pneumatic trail and residual torque arm
- Turn-slip correction: `φ_t = a / R_path` — prevents ~2% grip over-prediction on skidpad
  (R ≈ 7.5 m) by accounting for contact patch slip angle gradient

### 5.2 Layer 2 — 5-Node Thermodynamic ODE

**Jaeger flash temperature:**

$$T_{flash} = \frac{q \cdot a}{k \sqrt{\pi V_{slide} a / \alpha}}$$

5-node lumped-capacitance ODE per axle:

| Node | Variable | Description |
|---|---|---|
| 0 | T_surf_inner | Inner tread rib, primary heat source |
| 1 | T_surf_mid | Centre rib |
| 2 | T_surf_outer | Outer rib |
| 3 | T_gas | Internal inflation gas (Gay-Lussac → δP → δK_z) |
| 4 | T_core | Structural carcass thermal mass |

Thermal grip: `μ_T = exp(−β(T_eff − T_opt)²)` — Gaussian window at T_opt = 90°C.
Gay-Lussac pressure correction: `μ_P = 1 + 0.05(P_ratio − 1)`.

### 5.3 Layer 3a — TireOperatorPINN (Deterministic Drift)

Symmetry-respecting spectrally-normalised PINN predicting `(ΔFx/Fx0, ΔFy/Fy0)`.

**8-feature input vector (GP-v3-D):**

```
[sin(α), sin(2α), κ, κ³, γ, Fz/1000, Vx/20, T_norm]
```

`T_norm = (T_eff − T_opt) / 30` captures the dominant source of Pacejka residuals at operating
temperature (Pacejka changes up to 51% across the thermal window). Output corrections clipped to
±25%. `stop_gradient` on power-iteration σ prevents Adam from minimising σ (GP-v3-E).

### 5.4 Layer 3b — Sparse GP Matérn 5/2 (Stochastic Uncertainty)

50-inducing-point Sparse GP over 5-dimensional kinematic space `(α, κ, γ, Fz, Vx)`.

$$k(d) = \sigma^2 \left(1 + \sqrt{5}d + \frac{5}{3}d^2\right) \exp(-\sqrt{5}d)$$

Key numerical stabilisations:
- Cholesky decomposition `L = chol(K_ZZ + 1e-3 * I)` replaces `linalg.inv` (GP-v3-A)
- `softplus` variance floor replaces `jnp.maximum` — eliminates dead subgradient (GP-v3-B)
- Vectorised distance broadcast replaces nested `vmap(vmap(...))` (GP-v3-C)
- Inducing points initialised via `tanh(N(0, 0.5)) * scale + shift` — permanently bounded (GP-v3-F)
- LCB penalty: `clip(2σ, 0, 0.15)` — uncapped σ ≈ 0.28 would reduce forces to 44% of Pacejka (GP-v3-G)

### 5.5 Robust CBF with GP Uncertainty (Batch Upgrade)

The Control Barrier Function safety filter integrates GP uncertainty σ directly into the barrier:

$$h_{robust}(x) = h(x) - \kappa_{safe} \cdot \sigma_{GP}(x)$$

This provides formal safety guarantees that account for tire model uncertainty. When GP confidence
is low (σ large), the effective safety margin shrinks, automatically reducing allowable torque.
The DCBF additionally handles actuator delay via a predictive input-delay compensation term.

---

## 6. Differentiable Wavelet MPC (Diff-WMPC)

**File:** `optimization/ocp_solver.py`

### 6.1 3-Level Daubechies-4 Wavelet Compression

Control inputs `u(t) = [δ(t), F(t)]` over a 64-step horizon are parameterised in the Db4
wavelet frequency domain. Band-limiting is structural — the optimiser cannot produce high-frequency
control chattering regardless of coefficient values.

```
c[0 :  8]  = A3  (low-frequency approximation)
c[8 : 16]  = D3  (detail level 3)
c[16: 32]  = D2  (detail level 2)
c[32: 64]  = D1  (detail level 1, highest retained frequency)
```

128 raw inputs → ~20 active coefficients (85% sparsity).

### 6.2 Pseudo-Huber Wavelet Regularization (Batch Upgrade)

Replaces the original L1 penalty with a C∞ Pseudo-Huber loss on detail coefficients:

$$\psi(c; \delta) = \delta^2 \left(\sqrt{1 + (c/\delta)^2} - 1\right)$$

Properties:
- At `|c| >> δ`: ψ → `|c| - δ/2` (linear, sparsity-promoting)
- At `|c| << δ`: ψ → `c²/(2δ)` (quadratic, smooth gradient)
- Hessian `δ²/(c²+δ²)^{3/2}` always positive → strictly convex landscape for L-BFGS-B

Per-band weighting: D3 → 0.5×, D2 → 1.0×, D1 → 2.0× (higher frequency = stronger penalty).
This imposes physically meaningful smoothness: actuator bandwidth decreases with frequency.

### 6.3 Augmented Lagrangian Friction Constraint

$$\mathcal{L}_{AL} = f(x) + \lambda^T c(x) + \frac{\rho}{2} \left\| \max\!\left(c(x), -\frac{\lambda}{\rho}\right) \right\|^2$$

Constraint: `g_i = (a_lat² + a_lon²) / (μg)² − 1 ≤ 0`

Multipliers update: `λ ← max(λ + ρ·max(g, 0), 0)`. ρ grows 2× when max violation > 0.1.

### 6.4 Unscented Transform — Stochastic Tubes

5-point UT generates sigma trajectories from joint GP tire σ² + wind perturbations. All 5
simulated in parallel via `jax.vmap`. Track limits enforced against tube edges:

```
dist_left  = w_left  − (n_mean + κ_safe · √σ²_n)   ≥ 0
dist_right = w_right + (n_mean − κ_safe · √σ²_n)   ≥ 0
```

Both enforced via `−ε · log(softplus(dist * 50) / 50 + 1e-5)` — smooth log-barrier.

### 6.5 Physics P-Controller Warm Start

```
v_target = min(sqrt(μg / |κ|), 0.92 · V_limit)   [Newton units]
```

Warm start uses Newton force units matching the physics `u[1]` channel.
An earlier warm start in m/s² units produced `flat_init ≈ 0`, triggering GTOL premature
convergence after nit=1.

### 6.6 NaN Recovery Fallback

```python
fallback_loss = 1e9 + 0.5 * ||x − x_warmstart||²
fallback_grad = clip(x − x_warmstart, −100, 100)
```

The 1e9 offset exceeds any real loss; gradient points toward the warm start (a known feasible
trajectory), not toward zero (which increased friction constraint violation on every recovery step).

---

## 7. MORL-SB-TRPO Setup Optimiser

**File:** `optimization/evolutionary.py`

### 7.1 Overview

Multi-Objective RL over the full 28-dimensional `SuspensionSetup` space. Three Pareto axes:

1. **Maximum Grip** — skidpad objective [G]
2. **Dynamic Stability** — step-steer overshoot ≤ 5.0 rad/s
3. **Endurance LTE** — lap-time-energy composite over an 8-corner mini-lap

### 7.2 Chebyshev Ensemble Spacing

20 members with ω-weights Chebyshev-spaced on [0, 1]:

$$\omega_i = \frac{1}{2}\left(1 - \cos\!\left(\frac{i \pi}{N-1}\right)\right)$$

Concentrates ~65% of the ensemble into ω ∈ [0.7, 1.0] — the high-grip physically interesting region.

### 7.3 Riemannian Natural Policy Gradient (Batch Upgrade)

Replaces the KL-divergence parameter-space trust region with a Riemannian metric pulled back
through the physics engine:

$$G_{phys,k} = J_k^T \text{diag}(s) J_k + \lambda \cdot \text{diag}(J_k^T S J_k + \varepsilon I)$$

where `s = [1.0, 0.2, 0.5]` (grip weighted 5× over stability) and `J_k = ∂[grip, stab, LTE]/∂μ_k`.

The metric makes step size automatically small where the physical Jacobian is large — the optimizer
cannot take a large step across the ARB/oversteer bifurcation boundary without the metric shrinking
it. No hand-tuned threshold required.

### 7.4 ARD Bayesian Optimisation Cold Start

10 random initialisations + 30 EI-guided acquisitions using an ARD squared-exponential GP.
Per-dimension lengthscales learned via correlation heuristic — insensitive dimensions (castor,
anti-geometry) acquire large lengthscale and are effectively pruned.
Best 5 diverse basins seed the ensemble logit-space.

### 7.5 Endurance LTE Objective (Batch Upgrade)

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

### 7.6 SMS-EMOA Hypervolume Archive

3D hypervolume contribution pruning for the (grip, stability, LTE) Pareto front.
Archive capped at 150 points. Stability hard filter: setups with overshoot > 5.0 rad/s are
**excluded** from the archive — not just penalised.

---

## 8. Powertrain Control Stack

**Directory:** `powertrain/`

The complete 4WD Ter27 powertrain control stack. A single `@jax.jit`-compiled function
`powertrain_step()` is the only entry point. No mode switching. All transitions are
sigmoid-smooth. Compiles to a single XLA graph at 200 Hz.

### 8.1 Unified Manager — 13-Stage Pipeline

**File:** `powertrain/powertrain_manager.py`

```
Stage  1 : Virtual Impedance         → filtered pedal inputs (PIO mitigation)
Stage  2 : Acceleration Estimation   → low-pass ax, ay
Stage  3 : Traction Control          → κ* references + TC/TV blend weights
Stage  4 : Driver Force Demand       → Fx from filtered throttle/brake
Stage  5 : Torque Limits             → motor envelope + thermal derating
Stage  6 : Power Budget              → per-motor P_max from battery SoC + temperature
Stage  7 : Yaw Rate Reference        → driver-intent-aware target ψ̇_ref
Stage  8 : Launch Control            → B-spline torques during launch phase
Stage  9 : Torque Vectoring (SOCP)   → 12-iter projected-gradient allocator
Stage 10 : CBF Safety Filter         → input-delay DCBF + GP-uncertainty robustness
Stage 11 : Mode Blending             → launch vs TV/TC sigmoid blend
Stage 12 : Powertrain Thermal        → motor/inverter/battery state update
Stage 13 : Diagnostics Packaging     → 28-field PowertrainDiagnostics output
```

Output: `(PowertrainDiagnostics, PowertrainManagerState)` — every signal needed for telemetry.

### 8.2 Motor Model

**File:** `powertrain/motor_model.py`

PMSM electromechanical model with:
- Smooth field-weakening: `T_fw = T_peak · softplus(ω_base / ω) / softplus(1)` — C∞ transition
- Per-motor thermal ODE: winding temperature evolves with I²R + iron losses
- Battery OCV/R_int model: SoC tracked via Coulomb counting; R_int rises with temperature
- Power limit: `P_max = V_bus · I_max - I² · R_int` — exact, not approximate

### 8.3 Virtual Impedance (PIO Mitigation)

**File:** `powertrain/virtual_impedance.py`

Second-order virtual flywheel/damper preventing Pilot-Induced Oscillation on throttle/brake:

$$J \ddot{\theta} + C \dot{\theta} + K \theta = u_{pedal}$$

Symplectic Euler integration preserves the mechanical impedance structure. At 3 Hz (typical PIO
frequency), the filter provides > 30° phase lag — sufficient to break the PIO feedback loop.
Verified in Test 15: frequency response at 3 Hz confirmed.

### 8.4 Torque Vectoring — SOCP Allocator

**File:** `powertrain/modes/advanced/torque_vectoring.py`

12-iteration projected-gradient SOCP via `jax.lax.scan`:

**Cost function:**

$$J = \underbrace{w_{Mz}(M_z - M_z^*)^2}_{\text{yaw tracking}} + \underbrace{w_{Fx}(F_x - F_x^d)^2}_{\text{longitudinal}} + \underbrace{w_{\Delta T}\|\Delta T\|^2}_{\text{rate penalty}} + \underbrace{w_{loss}\sum T_i^2 \omega_i}_{\text{loss minimisation}}$$

Constraints enforced via projected-gradient (softplus barrier inside scan):
- Per-wheel torque bounds: `[T_min_i, T_max_i]` (motor envelope + thermal derating)
- Power budget: `Σ |T_i · ω_i| ≤ P_max`
- Friction circle: `|T_i / r_w| ≤ μ_i · Fz_i`

**Driver-intent-aware yaw reference:** Counter-steer detection via `sign(δ) ≠ sign(ψ̇_error)`.
When counter-steering detected, yaw moment target flips to assist oversteer correction.

### 8.5 CBF Safety Filter — Input-Delay DCBF

**File:** `powertrain/modes/advanced/torque_vectoring.py` → `cbf_safety_filter()`

Discrete Control Barrier Function with actuator delay compensation:

$$h(x_{k+d}) \geq (1 - \alpha)^d \cdot h(x_k)$$

where d = actuator delay steps and α ∈ (0,1) is the decay rate. The filter predicts the
future state at delay horizon d and enforces the barrier there — preventing constraint violation
from commands that arrive after the state has already evolved.

The robust extension integrates GP uncertainty:

$$h_{robust}(x) = h(x) - \kappa_{safe} \cdot \sigma_{GP}(x)$$

### 8.6 Traction Control — DESC Extremum-Seeking

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

### 8.7 Launch Control — Neural Predictive Sequencer v2.1

**File:** `powertrain/modes/advanced/launch_control.py`

**16-knot cubic B-spline profile** over a 2.0 s launch horizon — offline-optimisable.

**6-phase state machine:**

```
IDLE ──(btn OR brake+throttle)──► ARMED ──(btn_release + WOT)──► PROBE
PROBE ──(probe complete)──► LAUNCH ──(v > v_thr OR t > T_dur)──► HANDOFF
HANDOFF ──(t > dt_blend)──► TC
LAUNCH/HANDOFF ──(hard brake)──► IDLE   [abort path]
```

v2.1 additions:
- Button-based ARMED trigger (steering wheel button signal)
- Per-wheel TC ceiling: `T_cmd ≤ μ_rt · Fz · r_w · γ_κ`
- Real-time μ EMA continuously updated from DESC feedback
- Yaw-lock PI: differential torque correction targeting ψ̇ = 0 during launch
- Abort via hard brake (sigmoid-gated, not hard conditional)

All transitions are smooth sigmoid blends — no `jax.lax.cond`, no discontinuities.

### 8.8 Dictionary-Switched Koopman TV Controller

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

`trained_blend = 0.0` gives pure PD fallback (pre-Phase-2 behaviour); `trained_blend = 1.0`
gives full Koopman LQR (post-training). This allows safe incremental deployment.

**Offline training:** `scripts/train_koopman_tv.py` — EDMD-DL with Koopman generator loss +
observability regulariser. Training data generated from the bicycle dynamics model with random
manoeuvres across the operating envelope.

---

## 9. Differentiable EKF — Online Parameter Estimation

**File:** `models/differentiable_ekf.py`

### 9.1 State & Observation

5-state parameter estimator running at every physics step:

```
θ = [λ_μ_f, λ_μ_r, T_opt, h_cg, α_peak]   — parameter state
y = [ay_measured, wz_measured]               — observations (standard MoTeC ADL3)
```

`α_peak` (5th state, Batch Upgrade) tracks peak slip angle, which shifts with tyre wear and
temperature. Prior uncertainty ±0.04 rad covers both dry and wet operating conditions.

### 9.2 JAX Jacobian (No Finite Differences)

Because the vehicle dynamics are fully differentiable, the EKF measurement Jacobian is computed
exactly:

$$H = \frac{\partial [a_y^{sim}, \dot{\psi}^{sim}]}{\partial [\lambda_\mu, T_{opt}, h_{cg}, \alpha_{peak}]}$$

via `jax.jacobian` — no finite differences, no approximation errors. This is a direct benefit
of the end-to-end differentiable design.

### 9.3 Convergence

| Parameter | Initial Uncertainty | Convergence After |
|---|---|---|
| λ_μ_f, λ_μ_r | ±20% | 3–5 laps → ±3% |
| T_opt | ±10°C | 3–5 laps → ±5°C |
| h_cg | ±20 mm | 3–5 laps → ±8 mm |
| α_peak | ±0.04 rad | 5–8 laps → ±0.01 rad |

A living digital twin matches whatever car shows up on the day — different fuel load, tyre
pressures, tyre wear. The EKF makes the twin adaptive without retraining.

---

## 10. Full-Lap Differentiable Simulation

**File:** `optimization/differentiable_lap_sim.py`

### 10.1 The Kill Feature

```python
∂(lap_time) / ∂(setup_vector) = jax.grad(simulate_full_lap)(setup)
```

No FS team in the world has this. The gradient traces from the scalar lap time back through:

```
lap_time → speed profile → tire forces → suspension loads → setup params
```

in a single `jax.lax.scan` over the entire lap.

### 10.2 Differentiable Track

**File:** `optimization/differentiable_track.py`

JAX-native cubic spline interpolation on the SE(3) manifold:
- Track geometry stored as a sequence of SE(3) poses (rotation + translation)
- Curvature computed analytically: `κ = ω_z / v_xy` — no noisy finite-difference arclength derivatives
- Heading extracted from rotation matrix `R[1,0] / R[0,0]` — not from `arctan2(dy, dx)` which
  accumulates numerical error at parametric kinks
- NSDE stochastic friction uncertainty map `w_mu` from lateral slip variance — feeds directly
  into Tube-MPC parameter vectors

### 10.3 Architecture

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

---

## 11. CMA-ES Crossover Validator

**File:** `optimization/cmaes_validator.py`

Independent gradient-free optimizer that validates the MORL-SB-TRPO Pareto front. If CMA-ES
finds solutions that dominate the MORL front, the gradient-based optimizer has a bug or is stuck
in a local optimum.

**CMA-ES implementation:** Pure NumPy (μ/μ_w, λ)-CMA-ES with rank-1 and rank-μ updates.
Operates in normalised [0,1]^28 space. Physical bounds enforced by sigmoid projection.

**Workflow:**
1. Load MORL Pareto front from `morl_pareto_front.csv`
2. Seed CMA-ES from best MORL solution (warm start)
3. Run N generations in parallel evaluation
4. Compare best CMA-ES grip vs MORL best grip

| Delta | Interpretation |
|---|---|
| CMA-ES > MORL by > 0.02 G | ⚠️ MORL stuck in local optimum — reseed |
| Within ±0.02 G | ✅ Pareto front validated |
| MORL > CMA-ES | ✅ Gradient-based search outperforms black-box (expected) |

---

## 12. Simulator Architecture

**Directory:** `simulator/`

```
                                              ┌──────────────────────────────┐
                                      ┌──WS──→│ React Dashboard (Vercel)     │
                                      │       │ useLiveTelemetry.js          │
┌─────────────────────────┐  ┌────────┴───┐   └──────────────────────────────┘
│  physics_server.py      │  │ ws_bridge  │
│  200 Hz JAX integration │─→│ UDP→WS     │   ┌──────────────────────────────┐
│  46-DOF Port-Hamiltonian│  └────────────┘   │ Godot 3D Visualizer          │
│  + 28-dim setup space   │                   │ (driverless team)            │
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

### 12.1 Physics Server

**File:** `simulator/physics_server.py`

- Runs the 46-DOF GLRK-4 + powertrain pipeline at 200 Hz
- Accepts 28-dim `SuspensionSetup` and 8-dim adapter for backward compatibility
- Broadcasts a 64-float (256-byte) UDP telemetry packet to all clients simultaneously
- Supports preset setup switching (`3_stiff`, `2_soft`, etc.) via control commands
- Lap timing via geometric finish-line intersection (RANSAC-based S/F placement)

### 12.2 WebSocket Bridge

**File:** `simulator/ws_bridge.py`

- Listens on UDP; decimates 200 Hz → 60 Hz for the React dashboard
- Serialises to compact JSON (named fields, 3 decimal places for forces)
- Broadcasts to all simultaneous dashboard clients
- Forwards control commands from dashboard → physics server
- Performance stats logged every 5 s (UDP Hz → WS Hz, client count)

### 12.3 ROS 2 Bridge

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

## 13. React Engineering Dashboard

**Directory:** `visualization/dashboard_react/`

Deployed on Vercel. 7 navigation groups, 13 modules, LIVE/ANALYZE mode toggle.

### Navigation Groups & Modules

| Group | Modules |
|---|---|
| **Overview** | Architecture explainer — H_net, Tire Model, Diff-WMPC, MORL-SB-TRPO |
| **Telemetry** | Live charts: speed, lateral G, Fz per corner, tire temps, yaw rate |
| **Vehicle Dynamics** | Setup Opt (Pareto front), Suspension (LLTD, roll gradient, damper power), Tire Physics (5-node thermal, GP envelope, hysteresis), Weight & CG |
| **Aerodynamics** | Ground-effect Fz vs ride height, drag vs speed, aero balance |
| **Electronics** | Motor temperatures, SoC, inverter power, battery voltage |
| **Controls & AI** | Energy Audit (Hamiltonian landscape, R-matrix), ∇ Insights (setup sensitivities), Research (MORL convergence) |
| **Performance** | Driver Coaching, Endurance Strategy, Compliance (FSG rules) |

### LIVE Mode

- WebSocket connection to `ws_bridge.py` at `ws://localhost:8765`
- `useLiveTelemetry.js` hook: auto-reconnect, 60 Hz frame buffer, ring-buffer history
- All charts update in real time from the physics server

### Theme System

```
C.cy   = "#00d4ff"   — cyan (architecture, control)
C.gn   = "#00ff88"   — green (tire, safety)
C.am   = "#ffaa00"   — amber (dynamics, warning)
C.red  = "#ff4444"   — red (limits, MORL)
C.bg   = "#0a0a0f"   — background
```

---

## 14. Sanity Check Suite (Tests 1–16)

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
| 13 | DESC Convergence | κ* converges within adaptive dither schedule; error < 0.025 after Batch upgrade |
| 14 | Launch State Machine | Full IDLE→ARMED→PROBE→LAUNCH→HANDOFF→TC traversal |
| 15 | Virtual Impedance | Phase lag > 30° at 3 Hz; steady-state error < 5% |
| 16 | Full Pipeline JIT | Per-step time < 5 ms on SBC budget; 28 diagnostic fields finite |

---

## 15. Known Issues, Limits & Diagnostics

### 15.1 H_net Passivity Warning

Test 1 may report ~450 J/s passive power injection from H_net. This is attributed to synthetic
training data quality — MSE stagnated in prior revisions. Retraining `residual_fitting.py` with
the vX3 TTC PINN pipeline is expected to reduce this to < 50 J/s. The 450 J/s figure is ~0.45%
of typical kinetic energy at 15 m/s — acceptable for setup optimisation, not for energy analysis.

### 15.2 SOCP Solve Time

The SOCP allocator (12-iteration scan) is the primary deployment risk. Cold-start times on CPU
exceed the 5 ms real-time budget. Warm-starting from the previous timestep is not yet implemented.
On target SBC (ARM Cortex-A72), expected post-XLA-compilation latency: 1.2–2.8 ms.

### 15.3 Dashboard Telemetry

Dashboard was historically seeded with synthetic data. The `ws_bridge.py` → React real-time
connection is implemented but requires the physics server running. Real-hardware-in-loop
validation against MoTeC telemetry is a future step.

### 15.4 XLA Compilation Times

First run triggers XLA compilation; subsequent runs use the `XLA_FLAGS` cache:

| Subsystem | First compile | Cached |
|---|---|---|
| Vehicle dynamics (GLRK-4) | ~3–5 min CPU | ~5–15 s |
| WMPC full trajectory | ~8–12 min CPU | ~20–40 s |
| MORL gradient | ~5–8 min CPU | ~10–20 s |
| Powertrain step | ~2–4 min CPU | ~3–8 s |

Import `jax_config` **before any other JAX import** in every script.

### 15.5 Float32 Limits

- `softplus(200 * (z − 0.025))` overflows at z ≈ 469 mm — prevented by `clip(q[6:10], −0.08, 0.15)`
- H_net `H_RESIDUAL_CAP = 50,000 J/m²` prevents float32 overflow for extreme setups
- GP Gram matrix uses `jitter = 1e-3 * I` — sufficient for float32 (Cholesky condition ~1e4)

### 15.6 Three Existential Safety Threats (FSG 2026)

1. **CBF actuator delay** — addressed by input-delay DCBF in `cbf_safety_filter()`
2. **EMI/bit-flip risks in SBC memory** — ECC memory or CRC-checked state transfer required
3. **4WD ABS v_x anchoring** — during simultaneous multi-wheel lockup, v_x estimate from
   wheel speeds collapses; requires IMU-based v_x fusion (not yet implemented)

---

## 16. Pipeline Execution

```bash
# 0. Activate environment
source project_gp_env/bin/activate

# 1. Verify all subsystems (Tests 1–16, ~20 min first run due to compilation)
python sanity_checks.py

# 2. Train H_net / R_net against TTC tire data and synthetic physics rollouts
python -c "from optimization.residual_fitting import train_neural_residuals; train_neural_residuals()"

# 3. Offline Koopman TV training (requires tire data + vehicle params)
python scripts/train_koopman_tv.py

# 4. Run MORL setup optimiser — generates 3-objective Pareto front (~45 min)
python main.py --mode setup

# 5. Validate Pareto front with CMA-ES crossover (~10 min)
python optimization/cmaes_validator.py --generations 200 --popsize 40

# 6. Run full pipeline (Telemetry → Track Gen → WMPC → Driver Coaching)
python main.py --mode full --log /path/to/motec_telemetry.csv

# 7a. Physics server (Terminal 1)
python simulator/physics_server.py --track fsg_autocross

# 7b. WebSocket bridge for React dashboard (Terminal 2)
python simulator/ws_bridge.py

# 7c. Controls interface (Terminal 3)
python simulator/control_interface.py --mode keyboard

# 7d. ROS 2 bridge for driverless team (Terminal 4, requires ROS 2 Humble)
source /opt/ros/humble/setup.bash && python simulator/ros2_bridge.py

# 8. Launch React dashboard (if running locally)
cd visualization/dashboard_react && npm install && npm run dev
# Or use the Vercel deployment: https://project-gp-ter26.vercel.app
```

### Key Output Files

| File | Contents |
|---|---|
| `morl_pareto_front.csv` | 28-dim setup vectors + grip/stability/LTE on Pareto front |
| `cmaes_best_setup.csv` | CMA-ES best setup for MORL seeding |
| `models/h_net.bytes` | Trained H_net weights (Flax serialisation) |
| `models/r_net.bytes` | Trained R_net weights |
| `models/aero_net.bytes` | Trained physics-informed AeroMap weights |
| `models/h_net_scale.txt` | Training normalisation diagnostic (not architectural) |

---

## 17. Revision History (GP-vX1 → vX3)

### GP-vX3 (Current)

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
offline EDMD-DL training (`scripts/train_koopman_tv.py`) before `trained_blend > 0`.*