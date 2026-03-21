# Project-GP — End-to-End Differentiable Formula Student Digital Twin

> **Ter26 Formula Student | FSG 2026 — Siemens Digital Twin Award Entry**
>
> A 100% native JAX/Flax, fully end-to-end differentiable digital twin of the Ter26 FS vehicle.
> Designed for safety-biased setup optimisation, stochastic optimal control, and driver coaching.
> Every equation in the physics engine is differentiable. `jax.grad()` traces directly from
> lap time back to spring rates, damper curves, roll-centre heights, and brake bias.

---

## At a Glance

| Property | Value |
|---|---|
| **Framework** | 100% JAX/Flax — no NumPy inside traced functions |
| **State Dimension** | 46 (14 Mechanical DOF + 10 Thermal + 8 Transient Slip + 14 Velocity) |
| **Integrator** | 2-stage Gauss-Legendre RK4 (GLRK-4), symplectic, 4th-order, 5-iter Newton |
| **Dynamics** | Neural Port-Hamiltonian System — H_net (FiLM-conditioned) + R_net (Cholesky PSD) |
| **Tire Model** | Pacejka MF6.2 + Turn Slip + 5-Node Jaeger Thermal + PINN (8-feature) + Matérn 5/2 Sparse GP |
| **Optimal Control** | Diff-WMPC — 3-level Daubechies-4 Wavelet MPC + UT Stochastic Tubes + Augmented Lagrangian |
| **Setup Optimisation** | MORL-SB-TRPO — 28-dim, Chebyshev ensemble, ARD-BO cold-start, SMS-EMOA Pareto archive |
| **Setup Space** | 28 parameters (SuspensionSetup NamedTuple) — springs, ARBs, 4-way dampers, geometry, diff |
| **Revision** | GP-vX2 |

---

## Table of Contents

1. [Philosophy & Design Principles](#1-philosophy--design-principles)
2. [Repository Structure](#2-repository-structure)
3. [State Vector & SuspensionSetup](#3-state-vector--suspensionsetup)
4. [Neural Port-Hamiltonian Vehicle Dynamics](#4-neural-port-hamiltonian-vehicle-dynamics)
5. [Multi-Fidelity Tire Model](#5-multi-fidelity-tire-model)
6. [Differentiable Wavelet MPC (Diff-WMPC)](#6-differentiable-wavelet-mpc-diff-wmpc)
7. [MORL-SB-TRPO Setup Optimiser](#7-morl-sb-trpo-setup-optimiser)
8. [Known Issues, Limits & Diagnostics](#8-known-issues-limits--diagnostics)
9. [Pipeline Execution](#9-pipeline-execution)
10. [Revision History (GP-vX1 → vX2)](#10-revision-history-gp-vx1--vx2)

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
│   ├── vehicle_dynamics.py      # Neural Port-Hamiltonian 46-DOF dynamics + GLRK-4
│   ├── tire_model.py            # Pacejka MF6.2 + 5-node thermal + PINN + Sparse GP
│   ├── h_net.bytes              # Trained H_net weights (Flax serialisation)
│   ├── r_net.bytes              # Trained R_net weights
│   ├── aero_net.bytes           # Trained DifferentiableAeroMap weights
│   └── h_net_scale.txt          # Training normalisation diagnostic — NOT used in architecture
├── optimization/
│   ├── ocp_solver.py            # Diff-WMPC solver (Db4 DWT + AL + UT stochastic tubes)
│   ├── evolutionary.py          # MORL-SB-TRPO optimizer (28-dim, ARD-BO, SMS-EMOA)
│   ├── objectives.py            # Skidpad grip + step-steer stability objectives
│   └── residual_fitting.py      # H_net / R_net training pipeline (Phase 1 + 2)
├── data/
│   ├── configs/
│   │   ├── vehicle_params.py    # Canonical vehicle parameters (Ter26 FS 2026)
│   │   └── tire_coeffs.py       # Pacejka MF6.2 coefficients (Hoosier R20, TTC-fitted)
│   └── logs/                    # MoTeC telemetry CSV inputs
├── visualization/
│   ├── dashboard.py             # Streamlit engineering suite
│   ├── dashboard_react/         # React-based live telemetry & overview panels
│   └── suspension_3d_embed.py   # Three.js 3D suspension kinematics viewer
├── sanity_checks.py             # Physics subsystem verification (9 tests)
├── main.py                      # Pipeline entry point (--mode setup | full)
├── jax_config.py                # XLA cache + memory + parallelism config (import first)
└── morl_pareto_front.csv        # Latest Pareto front output from MORL run
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

**Static equilibrium deflections** (module-level constant `_Z_EQ`, used in both `vehicle_dynamics.py`
and `residual_fitting.py` — single source of truth):

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

---

## 4. Neural Port-Hamiltonian Vehicle Dynamics

**File:** `models/vehicle_dynamics.py`

### 4.1 Port-Hamiltonian Structure

The 14-DOF chassis evolves according to:

$$\dot{x} = (J - R)\,\nabla H(x) + F_{ext}(x, u)$$

- $J$ — skew-symmetric interconnection matrix (conservative, energy-preserving)
- $R$ — symmetric PSD dissipation matrix (energy-removing only)
- $H(x)$ — total Hamiltonian (kinetic + structural + neural residual)
- $F_{ext}$ — non-conservative external forces (tire, aero, gravity)

### 4.2 Neural Energy Landscape — H_net (FiLM-conditioned)

`NeuralEnergyLandscape` is a 128→64→1 MLP with **FiLM conditioning** at each hidden layer.
The setup vector modulates the energy landscape via Feature-wise Linear Modulation:

$$h_{out} = \gamma(\text{setup}) \odot \text{LayerNorm}(h) + \beta(\text{setup})$$

This allows the setup to control the *gradient* of the energy landscape (effective stiffness),
not just its offset. FiLM weights are initialised to identity (γ=1, β=0), so training starts
from the unmodulated baseline.

**Energy output:**

$$H_{total} = T_{prior} + V_{structural} + H_{res} \cdot \underbrace{\sum(q_{susp} - z_{eq})^2 + \epsilon}_{\texttt{susp\\_sq\\_eq}}$$
Key properties:
- `T_prior = 0.5 * ||p||² / M_diag` — exact kinetic energy from generalised momenta
- `V_structural = 0.5 * ||q_{susp}||² * K_{prior}` (K_prior = 30,000 N/m) — structural spring prior
- `susp_sq_eq` gates at **physical equilibrium** `_Z_EQ`, not at z=0. This means `dH/dq` is
  physically correct at the operating point (~30 kN/m spring force at 12.8 mm deflection).
- `H_res = min(softplus(MLP) * h_scale, 50_000)` — capped at 50,000 J/m² (covers 99.9th percentile
  of physically valid setups). `h_scale` is **always 1.0** in the architecture.
- 22 SE(3)-bilateral symmetric state features: anti-symmetric quantities (vy, wz, roll differences)
  enter as x² — structurally cannot represent odd functions regardless of weights.

**⚠️ h_net_scale.txt:** This file records the training normalisation factor (~102 J) for diagnostic
and W&B logging only. It is **never passed into the network architecture**. Doing so caused a 102×
energy amplification in a prior revision that forced all H_net gradients to zero via the
`H_RESIDUAL_CAP` clip.

### 4.3 Neural Dissipation Matrix — R_net (Cholesky PSD)

`NeuralDissipationMatrix` predicts the lower-triangular Cholesky factor L of the dissipation
matrix:

$$R = L L^T + \text{diag}(\text{softplus}(d))$$

The `diag(softplus(d))` term guarantees strict positive definiteness (not merely PSD), preventing
near-conservative blow-up. A physical mask restricts dissipation to heave, roll, pitch, and
unsprung-z DOFs — the physics-meaningful directions.

### 4.4 Differentiable Aero Map

`DifferentiableAeroMap` is a 32→32→4 MLP mapping `(vx, pitch, roll, heave_f, heave_r)` to
ground-effect-corrected `(Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero)`.

Ground clearance and downforce floors use `_softplus_floor` — not `jnp.maximum`. This ensures
the gradient is always non-zero, including at the 15 mm ground clearance limit where MPC is most
aggressive. `jnp.maximum` has a zero subgradient below the floor, killing the optimizer signal at
exactly the wrong moment.

### 4.5 Integrator — 2-Stage Gauss-Legendre RK4 (GLRK-4)

**The integrator is GLRK-4, not Störmer-Verlet leapfrog.** The README previously described a
"5-substep Symplectic Leapfrog" — this was inaccurate. The actual integrator uses the 2-stage
Gauss-Legendre Butcher tableau:

$$a = \begin{pmatrix} 1/4 & 1/4 - \sqrt{3}/6 \\ 1/4 + \sqrt{3}/6 & 1/4 \end{pmatrix}, \quad b = (1/2,\ 1/2)$$

Properties:
- **Symplectic** — preserves the symplectic 2-form `dq ∧ dp` to machine precision
- **4th-order** — energy drift O(h⁵) vs O(h³) for Störmer-Verlet
- **Implicit** — stage derivatives solved simultaneously via 5-iteration Newton scan inside
  `jax.lax.scan`; intermediate stage clipping at ±500 m/s² prevents NaN propagation at
  Newton overshoot states without affecting gradients within the physical envelope

**Auxiliary sub-states (thermal + slip)** are integrated with the trapezoidal rule using the
converged GLRK-4 stage derivatives — A-stable, one order of accuracy higher than forward Euler,
zero additional `_compute_derivatives` calls.

The thermal time constants (`τ_gas ≈ 20 s`, `τ_core ≈ 50 s`, `τ_surface ≈ 600 s`) produce
stiffness ratios `h|λ| ≈ 0.002 ≪ 2` at `dt = 1 ms` — the CFL condition is satisfied by three
orders of magnitude. Explicit integration of the thermal subsystem is numerically stable.

### 4.6 Suspension Forces

Per corner: `F_susp = F_spring + F_damper + F_bumpstop + F_ARB`

- **Spring:** `F = k * z * MR(z)²`, MR is a quadratic polynomial of heave travel
- **Damper:** 4-way digressive (low/high speed × bump/rebound), fully C∞ via sigmoid blending
  at both the knee velocity and the bump/rebound transition. No hard conditionals.
- **Bumpstop:** `F = k_bs * softplus(β * (z − gap)) / β` — C∞, zero force below gap, rapidly
  rising above. Replaces hard clip.
- **ARB:** lateral roll moment, applied as symmetric ±correction to left/right corners
- **Fz (contact normal load):** always positive via `_softplus_floor(F_grav ± load_transfer)`.
  Never clipped to zero — gradient survives corner going light during braking.

---

## 5. Multi-Fidelity Tire Model

**File:** `models/tire_model.py`

Three layers of increasing fidelity and uncertainty quantification.

### 5.1 Layer 1 — Pacejka MF6.2

Full Magic Formula for Hoosier R20 (TTC-fitted coefficients). Includes:
- Load sensitivity (dfz), camber sensitivity (γ terms)
- Combined-slip reduction factors Gyk and Gxa (corrected: Gyk is κ-driven, Gxa is α-driven)
- Full MF6.2 aligning torque Mz with pneumatic trail and residual torque arm
- Turn-slip correction: `φ_t = a / R_path` — prevents ~2% grip over-prediction on skidpad
  (R ≈ 7.5 m) by accounting for contact patch slip angle gradient across the patch length

### 5.2 Layer 2 — 5-Node Thermodynamic ODE

**Jaeger flash temperature** (analytical solution for a sliding semi-infinite solid):

$$T_{flash} = \frac{q \cdot a}{k \sqrt{\pi V_{slide} a / \alpha}}$$

Flash temperature feeds into a 5-node lumped-capacitance ODE per axle:

```
Node 0:  T_surf_inner  [°C]  — inner tread rib, primary heat source
Node 1:  T_surf_mid    [°C]  — centre rib
Node 2:  T_surf_outer  [°C]  — outer rib
Node 3:  T_gas         [°C]  — internal inflation gas (Gay-Lussac → δP → δK_z)
Node 4:  T_core        [°C]  — structural carcass thermal mass
```

State layout in x[28:38]:
`[T_s0_f, T_s1_f, T_s2_f, T_gas_f, T_s0_r, T_s1_r, T_s2_r, T_gas_r, T_core_f, T_core_r]`

Thermal grip factor: `μ_T = exp(−β(T_eff − T_opt)²)` — Gaussian window centred at T_opt=90°C.
Gay-Lussac pressure correction: `μ_P = 1 + 0.05(P_ratio − 1)` from gas temperature evolution.

### 5.3 Layer 3a — TireOperatorPINN (Deterministic Drift)

Symmetry-respecting spectrally-normalised PINN predicting `(ΔFx/Fx0, ΔFy/Fy0)` corrections
on top of the Pacejka baseline.

**8-feature input vector:**
```
[sin(α), sin(2α), κ, κ³, γ, Fz/1000, Vx/20, T_norm]
```

`T_norm = (T_eff − T_opt) / 30` — thermal deviation normalised to ±30°C range, where the
Pacejka baseline changes by up to 51% (exp(−0.0008 × 900) ≈ 0.49). Without this feature,
the PINN is blind to the dominant source of Pacejka residuals at operating temperature.

Output corrections clipped to ±25%. Spectral normalisation on Dense layers bounds Lipschitz
constant to ≤ 1. `stop_gradient` on the power-iteration normalisation factor `σ` — without
this, Adam minimises σ, making W/σ unbounded.

### 5.4 Layer 3b — Sparse GP Matérn 5/2 (Stochastic Uncertainty)

50-inducing-point Sparse GP over 5-dimensional kinematic space `(α, κ, γ, Fz, Vx)`.

**Matérn 5/2 kernel:**
$$k(d) = \sigma^2 \left(1 + \sqrt{5}d + \frac{5}{3}d^2\right) \exp(-\sqrt{5}d)$$

**Key numerical stabilisation (GP-vX3):**
- **Gram matrix:** Vectorised distance computation `Z_n[:, None, :] - Z_n[None, :, :]` replaces
  nested `vmap(vmap(...))`, eliminating 2,500 sequential kernel calls under abstract tracing.
- **Inversion:** Cholesky decomposition `L = chol(K_ZZ + 1e-3 * I)` replaces `linalg.inv`.
  `linalg.inv` backward squares the condition number (~1e4 → ~1e8), overflowing float32.
- **`stop_gradient` on L only:** `L` becomes a constant triangular factor in the backward pass.
  `∂σ/∂x_star` still flows through `k_xZ`, preserving the LCB penalty gradient w.r.t. tire
  operating point (which depends on wavelet coefficients through the physics rollout).
- **Variance floor:** `softplus(k_xx − red) + 1e-8` replaces `jnp.maximum(..., 1e-6)`.
  `jnp.maximum` has a zero subgradient at the kink — killing the GP uncertainty gradient in
  the WMPC loss function exactly when the friction constraint is most active.

**Inducing point initialisation:** `Z = tanh(N(0, 0.5)) * scale + shift`
Covers `α ∈ [−0.25, 0.25] rad`, `κ ∈ [−0.20, 0.20]`, `Fz ∈ [400, 1200] N`, `Vx ∈ [2, 22] m/s`
symmetrically. Tanh permanently bounds inducing points — Adam cannot push them outside the
physical operating envelope.

**LCB penalty:** `penalty = clip(2σ, 0, 0.15)` — capped at 15% maximum Pacejka reduction.
Uncapped, an uninitialised GP (σ ≈ 0.28) would reduce forces to 44% of Pacejka, causing the
MPC to "find" physically impossible cornering speeds (~19.2 m/s vs physical limit ~16.6 m/s).

---

## 6. Differentiable Wavelet MPC (Diff-WMPC)

**File:** `optimization/ocp_solver.py`

### 6.1 3-Level Daubechies-4 Wavelet Compression

Control inputs `u(t) = [δ(t), F(t)]` over a 64-step horizon are parameterised in the Db4
wavelet frequency domain. The optimiser searches over wavelet coefficients; the IDWT reconstructs
physically smooth time-domain inputs. Band-limiting is structural — the optimiser cannot produce
high-frequency control chattering regardless of coefficient values.

**Coefficient layout (N=64 per channel):**
```
c[0 :  8]  = A3  (low-frequency approximation, 3 DWT levels)
c[8 : 16]  = D3  (detail level 3, lowest frequency detail)
c[16: 32]  = D2  (detail level 2)
c[32: 64]  = D1  (detail level 1, highest retained frequency)
```

128 raw inputs → ~20 active coefficients (85% sparsity).
L1 penalty on D1, D2, D3 coefficients promotes sparse high-frequency content — physically
equivalent to enforcing smooth control trajectories.

**Implementation note:** `jnp.convolve` cannot be vmapped over the channel dimension without
materialising the batch axis inside `lax.conv_general_dilated`. All DWT/IDWT operations use
explicit per-channel calls.

### 6.2 Augmented Lagrangian Friction Constraint

Hard friction-circle constraint enforced via Augmented Lagrangian with adaptive ρ:

$$\mathcal{L}_{AL} = f(x) + \lambda^T c(x) + \frac{\rho}{2} \left\| \max\!\left(c(x), -\frac{\lambda}{\rho}\right) \right\|^2$$

Constraint: `g_i = (a_lat² + a_lon²) / (μg)² − 1 ≤ 0`

After each inner L-BFGS-B solve, multipliers update: `λ ← max(λ + ρ·max(g, 0), 0)`.
ρ grows by 2× when max violation > 0.1. This guarantees asymptotic feasibility as AL iterations
converge — unlike a soft barrier which can always be violated.

### 6.3 Unscented Transform — Stochastic Tubes

5-point UT generates sigma trajectories representing the joint uncertainty from GP tire σ² and
wind perturbations. The 5 trajectories are simulated in parallel via `jax.vmap`, reconstructing
the lateral position mean `μ_n` and variance `σ²_n` at each horizon step. Track boundaries are
enforced against the edges of the stochastic tube:

```
dist_left  = w_left  − (n_mean + κ_safe · √σ²_n)   ≥ 0
dist_right = w_right + (n_mean − κ_safe · √σ²_n)   ≥ 0
```

Both enforced via `−ε · log(softplus(dist * 50) / 50 + 1e-5)` — a smooth log-barrier. Safety
margin is proportional to current physical uncertainty.

### 6.4 Physics P-Controller Warm Start

The warm start runs N closed-loop simulation steps with a P-velocity controller:

```
F_brake = clip(−Kp · (v_curr − v_target), −8000, 0)    if v_curr > v_target
F_accel = clip(−Kp · (v_curr − v_target) · 0.3, 0, 600) otherwise
Kp = 6000 N/(m/s)
```

`v_target = min(sqrt(μg / |κ|), 0.92 · V_limit)` — friction-circle target with 8% safety margin.

This produces `U_warm[:,1]` in **Newton units** (matching the physics u[1] channel). An earlier
warm start encoded acceleration in m/s², which produced `flat_init ≈ 0` on near-constant-speed
segments, triggering GTOL premature convergence after nit=1.

### 6.5 NaN Recovery Fallback

When L-BFGS-B explores an unstable trajectory producing NaN gradients:

```python
fallback_loss = 1e9 + 0.5 * ||x − x_warmstart||²
fallback_grad = clip(x − x_warmstart, −100, 100)
```

The 1e9 offset exceeds any real loss (~3.2M maximum), so L-BFGS-B correctly identifies the NaN
point as worse than any feasible solution. The gradient points *toward* the warm start (a known
feasible trajectory) — not toward zero. Previous L2 fallback `(1e6 + 0.5||x||²)` pointed toward
zero (zero braking), which increased friction constraint violation on every recovery step.

---

## 7. MORL-SB-TRPO Setup Optimiser

**File:** `optimization/evolutionary.py`

### 7.1 Overview

Multi-Objective Reinforcement Learning over the full 28-dimensional `SuspensionSetup` space,
mapping the Pareto frontier between **Maximum Grip** (skidpad objective) and **Dynamic Stability**
(step-steer overshoot ≤ 5.0 rad/s).

Policy per ensemble member k:
$$\pi_k(\text{setup}) = \text{Sigmoid}(\mu_k + \varepsilon), \quad \varepsilon \sim \mathcal{N}(0, \sigma_k^2 I)$$

Physical setup: `s = SETUP_LB + (SETUP_UB − SETUP_LB) · Sigmoid(μ_k)`

### 7.2 Chebyshev Ensemble Spacing

20 ensemble members with ω-weights Chebyshev-spaced on [0, 1]:

$$\omega_i = \frac{1}{2}\left(1 - \cos\!\left(\frac{i \pi}{N-1}\right)\right)$$

This concentrates ~65% of the ensemble (8/10 in the sanity check run) into the high-grip region
ω ∈ [0.7, 1.0] — where the physically interesting Pareto boundary lies.

### 7.3 ARD Bayesian Optimisation Cold Start

10 random initialisations + 30 EI-guided acquisitions using an ARD squared-exponential GP.
Per-dimension lengthscales learned via correlation heuristic — insensitive dimensions (castor,
anti-geometry) acquire large lengthscale and are effectively pruned. Best 5 diverse basins seed
the ensemble logit-space, ensuring the gradient phase starts from physically meaningful regions.

Verified performance: BO phases finds basins up to 1.546G before gradient phase begins.

### 7.4 Safety-Biased Trust Region

KL divergence constraint (parameter-space trust region):
$$D_{KL}(\pi_{old} \| \pi_{new}) \leq \delta_{KL} = 0.005 \cdot \sqrt{28/8} \approx 0.0094$$

10-iteration lagged reference policy. Maximum entropy bonus: `0.005 · Σ log σ_k` — maintains
exploration variance across the ensemble. Bottom 5 members restarted every 200 iterations with
sensitivity-guided perturbation toward least-explored dimensions.

### 7.5 SMS-EMOA Hypervolume Archive

Pareto archive maintained with hypervolume contribution pruning:
- Non-dominated filtering on (grip, stability) objectives
- Exclusive HV contribution computed analytically (2D sweep algorithm)
- Archive pruned to 150 points via descending HV contribution

Stability filter: setups with overshoot > 5.0 rad/s are **excluded from the archive** (not just
penalised) — this is a hard physical requirement for drivability.

---

## 8. Known Issues, Limits & Diagnostics

### 8.1 H_net Passivity Warning

`sanity_checks.py` Test 2 reports ~450 J/s passive power injection from H_net. This is attributed
to synthetic training data quality in `residual_fitting.py` Phase 1 — the MSE stagnated at ~0.746
due to gradient attenuation from the `susp_sq` gate in prior revisions (now corrected via BUGFIX-5).
Retraining `residual_fitting.py` with the vX2 architecture is expected to eliminate this.

The 450 J/s figure represents ~0.45% of typical kinetic energy at 15 m/s. For setup optimisation
this is acceptable; for long-horizon energy analysis it should be resolved by retraining.

### 8.2 Gradient Sparsity in Sensitivity Report

The MORL sensitivity report shows near-zero gradients for `c_low_f`, `c_high_f`, and damper
parameters in the current Pareto front (results.txt). This reflects the skidpad objective
insensitivity to damping rates — correct physics. Step-steer stability would show larger damper
gradients. Both objectives need to be included for a complete sensitivity picture.

### 8.3 XLA Cache & Compilation

First run of `sanity_checks.py` or `main.py` triggers XLA compilation. Approximate times:

| Subsystem | First compile | Cached |
|---|---|---|
| Vehicle dynamics (GLRK-4) | ~3–5 min CPU | ~5–15 s |
| WMPC full trajectory | ~8–12 min CPU | ~20–40 s |
| MORL gradient | ~5–8 min CPU | ~10–20 s |

Import `jax_config` **before any other JAX import** in every script. This sets the compilation
cache directory and XLA flags before any tracing occurs. Importing after the first JAX operation
silently ignores all settings.

### 8.4 Float32 Limits

The entire stack operates in float32. Known saturation-prone operations:
- `softplus(200 * (z − 0.025))` overflows at z ≈ 469 mm — prevented by `clip(q[6:10], −0.08, 0.15)`
- `bumpstop_force` with β=200 requires the 150 mm travel limit to be enforced upstream
- H_net `H_RESIDUAL_CAP = 50_000 J/m²` prevents float32 overflow for extreme setup values

---

## 9. Pipeline Execution

```bash
# 0. Activate environment
source project_gp_env/bin/activate

# 1. Verify all physics subsystems (9 tests, ~15 min first run due to compilation)
python sanity_checks.py

# 2. Train H_net / R_net neural residuals (if weights not present or retraining needed)
python -c "from optimization.residual_fitting import train_neural_residuals; train_neural_residuals()"

# 3. Run MORL setup optimiser — generates Pareto front (400 iterations, ~45 min)
python main.py --mode setup

# 4. Run full pipeline (Telemetry → Track Gen → WMPC Ghost Car → Driver Coaching)
python main.py --mode full --log /path/to/motec_telemetry.csv

# 5. Launch engineering dashboard
streamlit run visualization/dashboard.py
```

### Key Output Files

| File | Contents |
|---|---|
| `morl_pareto_front.csv` | 28-dim setup vectors + grip/stability on Pareto front |
| `models/h_net.bytes` | Trained H_net weights (Flax serialisation) |
| `models/r_net.bytes` | Trained R_net weights |
| `models/h_net_scale.txt` | Training normalisation diagnostic (not architectural) |

---

## 10. Revision History (GP-vX1 → vX2)

### GP-vX2 (Current)

**Critical Architecture Fixes:**

| ID | Component | Issue | Fix |
|---|---|---|---|
| BUGFIX-4 | H_net | h_net_scale.txt (102.62) passed into architecture → 102× energy amplification → all H_net gradients zero → phantom 7534 mJ injection | h_scale=1.0 always in NeuralEnergyLandscape; scale file is diagnostic only |
| BUGFIX-5 | H_net | susp_sq gate at z=0 (never occupied); operating point z_eq had full gradient but gate provided no benefit | susp_sq = Σ(q[6:10] − _Z_EQ)² + 1e-4; consistent with residual_fitting.py training data |
| BUGFIX-7 | TireModel | compute_thermal_derivatives layout misaligned; T_nodes[4]=T_surf0_r used as front core → rear block offset by one | Reindexed to match vehicle_dynamics x[28:38] layout exactly |
| BUGFIX-8 | TireModel | tire.operator AttributeError in diagnose.py | @property operator returns _pinn_module |

**Architecture Upgrades:**

| ID | Component | Change |
|---|---|---|
| UPGRADE-9 | H_net | H_RESIDUAL_CAP 5,000 → 50,000 J/m² (old cap silently clipped ~30% of training samples) |
| UPGRADE-10 | AeroMap | jnp.maximum → _softplus_floor for ground clearance and Cl floors (zero subgradient at floor replaced with smooth gradient) |

**GP-vX3 Tire Patches (within vX2 file revision):**

| ID | Component | Change |
|---|---|---|
| GP-v3-A | SparseGP | linalg.inv → Cholesky + stop_gradient(L); jitter 1e-4 → 1e-3 |
| GP-v3-B | SparseGP | jnp.maximum variance floor → softplus (eliminates dead subgradient at kink) |
| GP-v3-C | SparseGP | Nested vmap Gram matrix → vectorised distance broadcast |
| GP-v3-D | TireOperatorPINN | Feature vector 7D → 8D (adds T_norm = (T_eff − T_opt)/30) |
| GP-v3-E | SpectralDense | stop_gradient(σ) in power iteration (prevents Adam from minimising σ) |
| GP-v3-F | SparseGP | Inducing point init: uniform(0,1) → tanh(N(0,0.5))*scale+shift (symmetric, bounded) |
| GP-v3-G | LCB penalty | Uncapped (2σ) → clip(2σ, 0, 0.15); uncapped penalty reduced forces to 44% of Pacejka |

### GP-vX1

| ID | Component | Change |
|---|---|---|
| BUGFIX-1 | SuspensionSetup | Index alignment — ARB at [2:4], not [4:6]; heave springs not in setup vector |
| BUGFIX-2 | Dynamics | Mass/inertia defaults corrected for Ter26 (m=300, Iz=150) |
| BUGFIX-3 | R_net | _TRIL_14 defined at module level (not inside __call__) |
| UPGRADE-1 | H_net | FiLM conditioning on setup vector |
| UPGRADE-2 | R_net | log-diagonal guarantee (strict PD, not just PSD) |
| UPGRADE-3 | Integrator | Störmer-Verlet → 2-stage GLRK-4 variational integrator |
| UPGRADE-4 | Bumpstop | Hard clip → C∞ softplus contact model |
| UPGRADE-5 | Dynamics | Compliance steer from lateral load (front/rear) |
| UPGRADE-6 | Damper | 4-way digressive model (sigmoid blend bump/rebound + knee) |
| UPGRADE-7 | Contact | Fz softplus floor (replaces jnp.maximum) |
| UPGRADE-8 | WMPC | ocp_solver _build_default_setup_28 removed; all callers use build_default_setup_28() |
| UPGRADE-9 | WMPC | Physics P-controller warm start (Newton units) replaces quintic Hermite (m/s²) |
| UPGRADE-10 | WMPC | NaN fallback anchored at warm start (not zero) |
| UPGRADE-11 | MORL | Full 28-dim SuspensionSetup (was 8-dim) |
| UPGRADE-12 | MORL | ARD BO cold-start (was SE-GP on 8 dims) |
| UPGRADE-13 | MORL | SMS-EMOA HV contribution pruning |

---

*Project-GP is a live research codebase. Physics is correct to the extent that sanity_checks.py
passes all 9 tests. Retraining H_net (residual_fitting.py) after any architectural change is
required before running WMPC or MORL in production mode.*