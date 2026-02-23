# Project-GP — 46-DOF Port-Hamiltonian Formula Student Digital Twin

> **Ter26 Formula Student Team | 2026**
> 
> A fully differentiable digital twin of the Ter26 FS vehicle, built in JAX/XLA for setup optimisation, driver coaching, and lap-time prediction.

---

## At a Glance

| Property | Value |
|---|---|
| State dimension | 46 |
| Mechanical DOF | 14 |
| Tyre model | Hoosier R20 Pacejka — TTC-fitted (PDY1=2.218, PDY2=-0.25) |
| Optimal control | Differentiable Wavelet-MPC (WMPC) with L-BFGS |
| Setup search | MORL-DB / SB-TRPO — Pareto-optimal grip vs. stability |
| WMPC validation result | 14.09 m/s on R=20m — 1.02 G lateral |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [MORL-DB Setup Optimiser](#3-morl-db-setup-optimiser)
4. [End-to-End Pipeline](#4-end-to-end-pipeline)
5. [Sanity Checks](#5-sanity-checks)
6. [File Structure](#6-file-structure)
7. [Installation & Usage](#7-installation--usage)
8. [Known Issues & Next Steps](#8-known-issues--next-steps)

---

## 1. Project Overview

Project-GP is a fully differentiable digital twin designed to be the core computational engine for setup optimisation, driver coaching, and lap-time prediction. Every physics calculation — from suspension spring force to tyre thermal dynamics — has an exact automatic derivative via JAX. This makes it possible to use gradient-based optimal control (WMPC) and gradient-free evolutionary search (MORL-DB) within a single unified codebase.

The project targets the **FSG Siemens Digital Twin Award**. Judging criteria reward physical accuracy and evidence of real-world validation. The architecture is built to accept GPS/IMU telemetry and produce a quantifiable prediction error (yaw rate RMSE) as a validation metric.

### Design Philosophy

- **End-to-end differentiable** — every component is JAX-compatible, enabling exact gradients through the full physics engine
- **Structure-preserving** — Port-Hamiltonian formulation conserves energy by construction, preventing drift over long simulations
- **Hybrid physics/ML** — neural networks augment, not replace, physics: H_net corrects the energy landscape, R_net corrects the dissipation matrix
- **Symplectic integration** — Störmer-Verlet (leapfrog) integrator preserves the Hamiltonian structure; Runge-Kutta does not
- **Probabilistic grip** — Sparse Gaussian Process with Matérn 5/2 kernel provides calibrated confidence intervals on tyre grip

---

## 2. System Architecture

### 2.1 State Vector (46 Dimensions)

The full vehicle state is a 46-element vector evolved forward in time by the integrator.

| Indices | Contents |
|---|---|
| `x[0:14]` | Generalised positions **q**: X, Y, Z, roll, pitch, yaw (chassis) + z_fl, z_fr, z_rl, z_rr (suspension heave) + w_fl, w_fr, w_rl, w_rr (wheel spin) |
| `x[14:28]` | Generalised velocities **v**: vx, vy, vz, wx, wy, wz (chassis) + suspension and wheel spin rates |
| `x[28:38]` | Tyre thermal states: T_core, T_rib_in, T_rib_mid, T_rib_out, T_gas — front and rear axle |
| `x[38:46]` | Transient slip states: α_t and κ_t for each corner — first-order tyre carcass lag |

The mechanical DOF count is 14 (6 rigid body + 4 suspension heave + 4 wheel spin). The 46-element state dimension adds thermal and transient slip states on top.

### 2.2 Port-Hamiltonian Core (`models/vehicle_dynamics.py`)

Equations of motion in Port-Hamiltonian form:

```
dx/dt  =  (J - R) ∇H  +  F_ext
```

where **J** is the skew-symmetric interconnection matrix (conservative coupling), **R** is the symmetric positive semi-definite dissipation matrix (energy loss), **∇H** is the gradient of the Hamiltonian (total energy), and **F_ext** is the vector of external forces from tyres, aero, and gravity.

| Component | Description |
|---|---|
| **H_net** (NeuralEnergyLandscape) | 128-64-1 MLP learning the residual Hamiltonian not captured by the analytical spring-mass model. Trained on synthetic chassis flex data. |
| **R_net** (NeuralDissipationMatrix) | 128-64 MLP outputting the Cholesky factor L of the dissipation matrix, returning R = LLᵀ. Positive semi-definiteness guaranteed by construction. |
| **Aero map** (DifferentiableAeroMap) | 32-32 MLP modulating baseline Cl and Cd as a function of speed, pitch, roll, and chassis height. |

### 2.3 Suspension Physics — Four Levels

| Level | Name | Physical Mechanism |
|---|---|---|
| 1 | Roll Centre Geometry | Splits weight transfer into geometric (via roll centre height, free grip) and elastic (via springs/ARBs, loads tyres) components. The single most important missing parameter in a bicycle model. |
| 2 | Progressive Bump Stops | Rubber bump stop engages at 25mm compression at 50,000 N/m. Progressive spring rate increase at limit travel without numerical stiffness from hard contact. |
| 3 | Digressive Damper | Three-region velocity-sensitive damping: linear below 0.1 m/s, digressive above (c_high = 0.4 × c_low). Uses `jnp.where` instead of `jnp.sign` to avoid undefined gradients at vz=0. |
| 4 | Kinematic Camber Gain | Camber from body roll (camber_gain = -0.8 deg/deg) and suspension travel (-25 deg/m). Clipped to -10/+5 degrees to prevent overflow during MPC gradient exploration. |

### 2.4 Tyre Model (`models/tire_model.py`)

Three layers operating in sequence:

**Pacejka Magic Formula** — TTC-fitted Hoosier R20 coefficients (PDY1=2.218, PDY2=-0.25). Lateral force Fy and longitudinal force Fx computed analytically.

**5-node thermal model** — surface inner/mid/outer ribs, core, gas. Flash temperature via the Jaeger sliding contact solution. Grip degrades as contact patch temperature deviates from T_opt = 90°C.

**Transient slip dynamics** — first-order carcass lag for slip angle (relaxation length Ly=0.25m) and slip ratio (Lx=0.1m). `x[38:46]` are the delayed slip states that the force model actually uses.

**GP uncertainty** — Sparse GP with Matérn 5/2 kernel computes 1-sigma grip uncertainty. The MPC applies the Lower Confidence Bound (2σ pessimistic) for robust tube constraints.

### 2.5 WMPC Optimal Control (`optimization/ocp_solver.py`)

64-step horizon at dt=0.01s. The control trajectory is parameterised in the Haar wavelet basis, compressing the 64-step sequence into a small number of wavelet coefficients. L-BFGS optimises these coefficients, with gradients computed exactly via JAX autodiff through the full 46-DOF physics engine.

**Validation result:** 14.09 m/s on R=20m constant curvature — 1.02 G lateral. The Pacejka model predicts a theoretical limit of ~16.6 m/s; the MPC reaches 85% of that limit due to wavelet resolution and the GP uncertainty penalty.

---

## 3. MORL-DB Setup Optimiser

### 3.1 Search Space (7 Parameters)

| Parameter | Description | Range |
|---|---|---|
| `k_f` | Front spring rate | 15,000 – 55,000 N/m |
| `k_r` | Rear spring rate | 15,000 – 55,000 N/m |
| `arb_f` | Front anti-roll bar rate (at wheel) | 0 – 800 N/m |
| `arb_r` | Rear anti-roll bar rate (at wheel) | 0 – 800 N/m |
| `c_f` | Front damper coefficient | 500 – 4,000 N·s/m |
| `c_r` | Rear damper coefficient | 500 – 4,000 N·s/m |
| `h_cg` | Centre of gravity height | 0.25 – 0.35 m |

### 3.2 Algorithm

(μ, λ) evolution strategy with NSGA-II crowding distance for archive maintenance. Each generation evaluates 20 candidates, applies non-dominated sorting, and generates the next generation by mutation from elite parents. Step size σ decays from 0.15 to 0.02 over 400 iterations.

Safety constraint: only setups with positive safety margin are archived (rear LLTD > front LLTD + 2% at 1.5G reference load — understeer bias guaranteed).

### 3.3 Objectives (`optimization/objectives.py`)

**Grip objective** — steady-state cornering balance swept from 0.8G to 2.0G. Finds the highest achievable lateral acceleration where both axles are simultaneously below their grip limits. Includes:
- Roll centre geometry in weight transfer (geometric + elastic split)
- Aero downforce at 15 m/s cornering (Cl=3.0, A=1.5, 40/60 front/rear)
- Camber grip bonus — Gaussian peak at -3.5° effective camber on outer tyre
- Inner tyre lift penalty — activates below 50N inner Fz
- Stiffness penalty — scaled to 7mm RMS bump amplitude
- Ride frequency bounds — 1.0 to 3.5 Hz

**Stability objective** — damping ratio error across four vibration modes:

| Mode | Target ζ | Weight |
|---|---|---|
| Heave | 0.65 | 2.0 |
| Roll | 0.70 | 1.5 |
| Pitch | 0.60 | 1.0 |
| Wheel hop | 0.30 | 0.5 |

### 3.4 Current Results

```
Grip diagnostic:  soft setup 1.50G  |  hard setup 1.03G  ✓ (physics is sensitive to setup)
Safe count:       14–19 / 20 per generation  ✓
Stability margin: 0.285 – 0.41  ✓ (all setups understeer-biased)
Top grip result:  1.517 G
```

Known limitation: ~80% of top setups pin at k_f = k_r = 15,000 N/m (lower bound). See [Known Issues](#8-known-issues--next-steps).

---

## 4. End-to-End Pipeline

```bash
python main.py --mode <mode> [--log path/to/log.csv]
```

| Mode | Description |
|---|---|
| `pretrain` | Pre-trains H_net and R_net on synthetic chassis flex data. Serialises weights to `models/h_net.bytes` and `models/r_net.bytes`. Run once. |
| `telemetry` | Ingests GPS/IMU log. Fits continuous-time SE(3) trajectory on the Lie group. Outputs curvature profile, width corridor, and friction uncertainty map. |
| `ghost` | Runs WMPC on the fitted track to generate the AI ghost car trajectory. Outputs `stochastic_ghost_car.csv`. |
| `coach` | Compares human driver against ghost car via Actor-Critic evaluator. Identifies braking, apex, and exit throttle losses. Outputs `ac_mpc_coaching_report.csv`. |
| `setup` | Runs MORL-DB optimiser. Outputs `morl_pareto_front.csv` — 150-point Pareto front of grip vs. stability. |
| `full` | All phases in sequence on a single telemetry log. |
| `closed_loop` | Full pipeline + second WMPC solve with AI coaching weights injected. Produces an adapted ghost car accounting for track-specific grip variations. |

---

## 5. Sanity Checks

Run before any telemetry work:

```bash
python sanity_checks.py
```

| Test | Pass Criterion | Current Result |
|---|---|---|
| Neural convergence | H_net MSE < 20, R_net MSE < 15 at epoch 1000 | H_net 15.1, R_net 12.9 ✅ |
| Symplectic forward pass | All 46 states finite after single step | vx 10.000→10.006, wz 0.009 rad/s ✅ |
| WMPC circular track | Mean speed 13–18 m/s on R=20m | 14.09 m/s, 1.02 G ✅ |

---

## 6. File Structure

```
FS_Driver_Setup_Optimizer/
├── main.py                          # Master pipeline controller
├── sanity_checks.py                 # Pre-flight validation — must pass before real data
├── morl_pareto_front.csv            # Output: latest Pareto front
│
├── models/
│   ├── vehicle_dynamics.py          # 46-DOF Port-Hamiltonian integrator
│   └── tire_model.py                # Pacejka + 5-node thermal + transient slip + GP
│
├── optimization/
│   ├── ocp_solver.py                # DiffWMPCSolver — Haar wavelet MPC
│   ├── evolutionary.py              # MORL_SB_TRPO_Optimizer — Pareto search
│   ├── objectives.py                # Analytical grip & damping objectives
│   └── residual_fitting.py          # H_net / R_net pre-training
│
├── telemetry/
│   ├── log_ingestion.py             # GPS/IMU log parsing (ASC, CSV)
│   ├── filtering.py                 # SE(3) manifold trajectory estimator
│   ├── track_generator.py           # Curvature + width corridor + friction map
│   └── driver_coaching.py           # Actor-Critic ghost car evaluator
│
└── data/
    └── configs/
        ├── vehicle_params.py        # Mass, geometry, aero, suspension params
        └── tire_coeffs.py           # Fitted Hoosier R20 Pacejka coefficients
```

---

## 7. Installation & Usage

### Environment Setup

```bash
# Activate the Python environment
source ~/project_gp_env/bin/activate

# Set ACADOS library path (required for CasADi legacy compatibility)
export ACADOS_SOURCE_DIR=~/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/acados/lib
```

### Dependencies

- `jax`, `jaxlib` — differentiable numerical computing, XLA compilation
- `flax` — neural network definitions (Linen API)
- `optax` — optimisers for neural pre-training
- `numpy`, `pandas` — data handling
- `wandb` — experiment tracking and Pareto front logging
- `casadi`, `acados` — legacy CasADi bicycle model (retained for compatibility)

### Quickstart

```bash
# Step 1: Verify the physics engine
python sanity_checks.py

# Step 2: Run setup optimisation
python main.py --mode setup

# Step 3: With telemetry, run the full pipeline
python main.py --mode full --log path/to/your/log.csv
```

---

## 8. Known Issues & Next Steps

### Known Issues

**Spring rate lower-bound pinning** — ~80% of Pareto front setups are at k_f = k_r = 15,000 N/m. The stiffness and frequency penalties are insufficient to push the optimum into the interior of the feasible space. Fix: tighten ride frequency lower bound from 1.0 Hz to 1.2 Hz in `objectives.py`.

**Column naming** — `Stability_Overshoot` in the output CSV is actually the understeer margin (positive = understeer biased = safe). The 0.285–0.41 range in current results means all setups pass the safety constraint. The column should be renamed `Understeer_Margin` to avoid confusion.

**Neural networks trained on synthetic data only** — H_net and R_net are pre-trained on synthetic chassis flex. They are not yet fitted to real chassis flex measurements from a strain gauge or accelerometer array.

### Next Steps

**Priority 1 — Skidpad validation (highest leverage)**

Book a skidpad test session. Collect yaw rate and lateral acceleration at a minimum of two steady-state cornering speeds. One clean validation run producing yaw rate RMSE < 15% is the difference between an unvalidated simulation and a validated digital twin — and is decisive for the award.

| State | Architecture Score | Award Probability |
|---|---|---|
| Current (no validation) | 7.8 / 10 | 25–35% |
| After skidpad session | ~9.1 / 10 | 55–70% |

**Priority 2 — Measure real vehicle parameters**

Replace defaults with measured values from CAD and workshop:
- Motion ratios (currently defaulting to 1.2 / 1.15)
- Roll centre heights from suspension geometry
- Sprung/unsprung mass split
- Static camber at ride height

**Priority 3 — Fix spring rate optimisation**

In `optimization/objectives.py`, change:
```python
jax.nn.relu(1.0 - freq_heave_f)   # current lower bound
```
to:
```python
jax.nn.relu(1.2 - freq_heave_f)   # tightened lower bound
```

**Priority 4 — Rename output column**

In `main.py`, rename `Stability_Overshoot` to `Understeer_Margin` in the DataFrame construction.

---

*Project-GP — Ter26 Formula Student — 2026*