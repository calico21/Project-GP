# Project-GP — End-to-End Differentiable Formula Student Digital Twin

> **Ter27 Formula Student | FSG 2026 — Siemens Digital Twin Award Entry**
>
> A 100% native JAX/Flax, fully end-to-end differentiable digital twin of the Ter27 FS vehicle.
> Designed for safety-biased setup optimisation, stochastic optimal control, real-time powertrain
> management, and driver coaching. Every equation in the physics engine is differentiable.
> `jax.grad()` traces directly from lap time back to spring rates, damper curves, roll-centre
> heights, brake bias, and powertrain torque allocation. The entire stack runs as a single XLA
> graph at 200 Hz on an embedded SBC.
>
> **GP-vX6 milestone:** the full `sanity_checks.py` suite (physics, powertrain, and 108-DOF
> subsystem tests) is green end-to-end for the first time since the 46→108-DOF migration. This
> revision documents the architecture as it actually stands post-fix, including the structurally
> passive ICNN Hamiltonian, the UKF state estimator, the Cayley-stable Koopman operator, and the
> slip-aware mpQP allocator.

---

## At a Glance

| Property | Value |
|---|---|
| **Framework** | 100% JAX/Flax — no NumPy inside traced functions |
| **Vehicle** | Ter27 — 4WD electric Formula Student (FSG 2026) |
| **State Dimension** | 108 (14 positions + 14 momenta + 28 thermal 3D + 16 transient 2nd-order slip + 12 damper hysteresis + 24 elastokinematic compliance) |
| **Integrator** | 2-stage Gauss-Legendre RK4 (GLRK-4), symplectic, 4th-order; 4 stop-gradiented Newton iterations + 1 gradient-carrying final iteration |
| **Dynamics** | Neural Port-Hamiltonian System — **PassiveHNet** (ICNN kinetic/potential decomposition, structurally passive by construction) + R_net (Cholesky PSD dissipation) |
| **Tire Model** | Pacejka MF6.2 + Turn Slip + 3D Lateral Asymmetric Thermal (4 corners × 7 nodes) + 2nd-order Transient Slip + Spectrally-Normalised PINN (8-feature) + Spectral-Mixture Sparse GP (Cholesky, `stop_gradient(L)`) |
| **Suspension** | Full double A-arm kinematic solver (IFD + custom VJP, Rodrigues rotations) + nonlinear Bouc-Wen elastokinematic bushing model |
| **Aero** | `AeroPlatformModel` — physics-structured ground-effect stall envelope, pitch/roll/yaw sensitivity |
| **Damper** | Generalized Maxwell (2-branch) hysteretic damper with oil-temperature-dependent viscosity and cavitation model |
| **Track** | B-spline track geometry (periodic cubic, analytic curvature) + rubber build-up / grip-asymmetry surface model |
| **Slip Observers** | Koopman-Bilinear Slip Observer (8-term Pacejka-spanning dictionary, primary κ*) + RLS Slip-Slope Observer (secondary/fallback) + experimental Cayley-stable Koopman TV operator |
| **State Estimator** | 14-state Unscented Kalman Filter (`state_estimator.py`) fusing IMU + wheel speed + steering + GPS; feeds `vx, vy, wz, Fz, α_t` to the powertrain manager |
| **Optimal Control** | Diff-WMPC — 3-level Daubechies-4 Wavelet MPC + Coifman–Wickerhauser best-basis entropy regularisation + Unscented-Transform stochastic tubes + Augmented Lagrangian (friction + spatial) + Pseudo-Huber wavelet regularisation |
| **Setup Optimisation** | MORL-SB-TRPO + Riemannian NPG — 28-dim `SuspensionSetup` (40-dim schema, 12 extended params reserved), Chebyshev ensemble, ARD-BO cold-start, SMS-EMOA-style Pareto archive, 3-axis (grip / stability / Endurance-LTE) |
| **Powertrain Control** | 13-stage `powertrain_step()` pipeline at 200 Hz: Virtual Impedance → TC (Koopman+RLS fused) → mpQP KKT Allocator (V2, slip-aware, 24×24) → Dynamic Regen Blend → Robust input-delay DCBF → Launch Control v2.1 → Thermal update |
| **Lap Gradient** | `jax.grad` through the GLRK-4 scan; gradient server exposes short-horizon Jacobian sensitivities `∂v_x(100ms)/∂setup` to the dashboard |
| **Simulator** | 200 Hz physics server target; gradient server (`scripts/gradient_server.py`) serving live setup sensitivities over HTTP |
| **Sanity Suite** | `sanity_checks.py` — 25+ tests across physics, powertrain, and 108-DOF subsystems, **all passing** as of this revision |
| **Revision** | **GP-vX6** |

---

## Table of Contents

1. [Philosophy & Design Principles](#1-philosophy--design-principles)
2. [Repository Structure](#2-repository-structure)
3. [The 108-DOF State Vector & SuspensionSetup](#3-the-108-dof-state-vector--suspensionsetup)
4. [PassiveHNet — Structurally Passive Neural Hamiltonian](#4-passivehnet--structurally-passive-neural-hamiltonian)
5. [Vehicle Dynamics Integration (GLRK-4)](#5-vehicle-dynamics-integration-glrk-4)
6. [Suspension Package — Kinematics & Elastokinematics](#6-suspension-package--kinematics--elastokinematics)
7. [Multi-Fidelity Tire Model](#7-multi-fidelity-tire-model)
8. [Aero, Damper & Track Surface Models](#8-aero-damper--track-surface-models)
9. [Slip Observers — Koopman, RLS & Cayley-Stable Koopman TV](#9-slip-observers--koopman-rls--cayley-stable-koopman-tv)
10. [State Estimation — 14-State UKF](#10-state-estimation--14-state-ukf)
11. [Differentiable Wavelet MPC (Diff-WMPC)](#11-differentiable-wavelet-mpc-diff-wmpc)
12. [MORL-SB-TRPO Setup Optimiser](#12-morl-sb-trpo-setup-optimiser)
13. [Powertrain Control Stack](#13-powertrain-control-stack)
14. [Sanity Check Suite](#14-sanity-check-suite)
15. [Known Issues, Limits & Diagnostics](#15-known-issues-limits--diagnostics)
16. [Pipeline Execution](#16-pipeline-execution)
17. [Revision History](#17-revision-history)

---

## 1. Philosophy & Design Principles

Project-GP abandons traditional numerical simulation frameworks (CasADi, IPOPT, point-mass
solvers) in favour of a **deep learning compiler architecture (JAX/XLA)**. Five non-negotiable
constraints govern every module in the repo.

**The Differentiability Rule.** Every function, physics equation, and control-logic path must be
strictly differentiable. Hard conditionals (`jnp.where` with discontinuous branches, step
functions) are forbidden; all limits are smooth (`jax.nn.softplus`, `jax.nn.sigmoid`, `jnp.tanh`
rescaling, `safe_abs` via a Hessian-bounded softplus-V). Where a genuinely discrete decision is
required (e.g. launch-control phase transitions), it is implemented as a smooth sigmoid gate
rather than `jax.lax.cond` branching on a hard boolean, so `jax.grad` never hits a zero-measure
kink.

**The JAX Purity Rule.** All physics is pure JAX; NumPy never appears inside a `jit`/`grad`-traced
function. `vmap`/`scan` correctness is checked against abstract-tracing rules (e.g. `jnp.convolve`
is never vmapped directly over a batch axis — the wavelet transforms in `ocp_solver.py` do
explicit per-channel `jnp.convolve` calls instead). The IFD (Implicit Function Differentiation)
chain in `suspension/kinematics.py` uses a hand-written `jax.custom_vjp` to differentiate through
the Newton constraint solver without unrolling the Newton tape.

**The Physical Rule.** The Port-Hamiltonian structure is never broken. As of GP-vX6, this is
enforced **algebraically**, not just by training incentive: `PassiveHNet` (see §4) is built from
Input-Convex Neural Networks so that H ≥ 0, H(q_eq, 0, setup) = 0, ∇_p H(q, 0, setup) = 0, and
p·∇_p H ≥ 0 hold for *any* weight values — no amount of bad training data can produce a
Hamiltonian that injects energy. `R_net` can only produce PSD dissipation matrices via
`R = LLᵀ + diag(softplus(d))`. The Bouc-Wen bushing hysteresis is energy-bounded by construction.

**The Canonical-Index Rule.** All setup parameters are accessed exclusively via
`SuspensionSetup.from_vector()` / `.to_vector()`. No positional indexing of the raw setup array is
permitted outside these methods. The 108-DOF state vector is similarly accessed via named index
ranges defined at module level in `vehicle_dynamics.py`; raw positional indexing into `x` outside
that file is forbidden.

**The Freeze-Before-Grad Rule.** Frozen CAD parameters get `LB = UB`, so `project_to_bounds()`
returns a constant and the gradient is identically zero without any masking logic in the
optimizer.

---

## 2. Repository Structure

```
FS_Driver_Setup_Optimizer/
├── models/
│   ├── vehicle_dynamics.py          # 108-DOF Port-Hamiltonian dynamics + GLRK-4 integrator
│   ├── tire_model.py                # Pacejka MF6.2 + spectral-normalised PINN + SM-kernel Sparse GP
│   ├── tire_thermal_3d.py           # 3D tire thermal (4 corners × 7 nodes), camber load asymmetry
│   ├── tire_transient.py            # 2nd-order transient slip (carcass + belt relaxation)
│   ├── aero_platform.py             # AeroPlatformModel: ground-effect stall, pitch/roll/yaw coupling
│   ├── damper_hysteresis.py         # Generalized Maxwell (2-branch) damper + thermal fade + cavitation
│   ├── track_surface.py             # B-spline track geometry + rubber build-up / grip asymmetry
│   └── h_net_scale.txt              # Diagnostic-only training normalisation record (not architectural)
│
├── physics/
│   └── h_net_icnn.py                # PassiveHNet: ICNN-based structurally-passive Hamiltonian residual
│
├── optimization/
│   ├── ocp_solver.py                # Diff-WMPC: Db4 DWT/WPD + CW entropy + AL (friction+spatial) + UT tubes
│   ├── evolutionary.py              # MORL-SB-TRPO + Riemannian NPG optimizer, ARD-BO cold start
│   ├── objectives.py                # Skidpad grip, step-steer stability, Endurance-LTE mini-lap objective
│   ├── residual_fitting.py          # PassiveHNet / R_net training pipeline (density-matched targets)
│   └── pareto_continuation.py       # Pareto front continuation utilities used by sanity checks
│
├── powertrain/
│   ├── powertrain_manager.py        # Unified pipeline — single JIT powertrain_step()
│   ├── motor_model.py               # PMSM electromechanical + thermal + battery OCV/R_int model
│   ├── virtual_impedance.py         # 2nd-order virtual flywheel/damper pedal filter — PIO mitigation
│   ├── regen_blend.py               # Dynamic regen blend — battery/thermal-aware α*, hydraulic residual
│   ├── state_estimator.py           # 14-state Unscented Kalman Filter (vx, vy, wz, Fz×4, α_t×4, IMU bias×3)
│   ├── powertrain_wiring_v2.py      # HubMotorCommand packing + UKF-integrated PowertrainOutputV2
│   └── modes/
│       ├── advanced/
│       │   ├── torque_vectoring.py         # Projected-gradient SOCP fallback + input-delay robust DCBF
│       │   ├── traction_control.py         # DESC + Koopman/RLS fused κ* + TC/TV blend weights
│       │   ├── launch_control.py           # v2.1: button-armed FSM, TC ceiling, real-time μ EMA, yaw-lock PI
│       │   ├── koopman_tv.py               # Dictionary-switched Koopman LQR yaw controller (blend-locked)
│       │   ├── koopman_stable.py           # Cayley-parameterised (ρ(K)=1) Koopman + risk-sensitive LQR
│       │   ├── koopman_slip.py             # Koopman-bilinear slip observer (primary κ* estimator)
│       │   ├── rls_tc.py                   # RLS slip-slope observer (secondary κ* estimator)
│       │   ├── slip_barrier.py             # Predictive slip-CBF row builder for the KKT allocator
│       │   ├── explicit_mpqp_allocator.py  # V1 (16×16) / V2 (24×24, slip-aware) explicit KKT allocator
│       │   └── active_set_classifier.py    # Neural active-set predictor (V1: 15-dim θ, V2: 19-dim θ)
│       └── intermediate/                   # Reduced-fidelity TV/TC path for the SIMPLE/INTERMEDIATE modes
│
├── suspension/
│   ├── kinematics.py                # Full double A-arm solver; IFD + custom VJP (FIX-1..4)
│   └── elastokinematics.py          # Nonlinear Bouc-Wen elastokinematic bushing model
│
├── config/
│   └── vehicles/ter26.py            # Canonical vehicle_params dict (shared Ter26/Ter27 baseline)
│
├── scripts/
│   ├── gradient_server.py           # HTTP server — short-horizon Jacobian sensitivities for the dashboard
│   ├── benchmark_socp_latency.py    # Cold/warm-start latency benchmark for the SOCP/KKT allocator
│   ├── generate_qp_training_data.py # V1 (15D/12C) and V2 (19D/20C, slip-aware) QP training set generator
│   ├── train_koopman_hnet.py        # EDMD-DL Koopman TV retraining against 108-DOF PassiveHNet rollouts
│   ├── run_ter27_design_exploration.py  # Phase-aware (freeze) design exploration CLI
│   ├── run_twin_fidelity_demo.py    # FSG Digital Twin Award validation pipeline (R², xcorr, PSD metrics)
│   ├── diagnose_setup_graph_connectivity.py  # Gradient-connectivity diagnostic for the 28-param setup
│   └── vcu_bridge.py                # 4-byte CAN payload packer for the VCU torque-vectoring split
│
├── sanity_checks.py                 # Full system verification — 25+ tests, currently all green
├── README.md
└── jax_config.py                    # XLA cache + memory config (import first, always)
```

---

## 3. The 108-DOF State Vector & SuspensionSetup

### 3.1 State Layout

```
x[0:28]    kinematics — 14 generalised positions q + 14 generalised momenta p
             q[0:3]   X, Y, Z            chassis CG, world frame        [m]
             q[3]     φ  roll                                            [rad]
             q[4]     θ  pitch                                           [rad]
             q[5]     ψ  yaw                                             [rad]
             q[6:10]  z_fl, z_fr, z_rl, z_rr   suspension heave          [m]
             q[10:14] θ_fl..θ_rr        wheel rotation                   [rad]
             p = M_diag ⊙ v   (Port-Hamiltonian momenta, M_diag holds sprung/unsprung/wheel inertias)

x[28:56]   thermal 3D — 4 corners × 7 nodes (28 states)
             [0:3] T_surf_inner/mid/outer  (camber-load-weighted rib temps)  [°C]
             [3]   T_gas    [4] T_core     [5:7] T_carcass_inner/outer

x[56:72]   2nd-order transient slip — 4 corners × 4 states (16 states)
             [0] α_t  [1] α_t_dot  [2] κ_t  [3] κ_t_dot

x[72:84]   damper hysteresis — 4 corners × 3 states (12 states)
             [0] F_branch_1  [1] F_branch_2  [2] T_oil     (2-branch Maxwell + thermal ODE)

x[84:108]  elastokinematic compliance — 4 corners × 6 Bouc-Wen states (24 states)
             quasi-static reduced hysteresis proxy per corner (6 links: lower-fore/aft,
             upper-fore/aft, tie-rod, pushrod)
```

`DifferentiableMultiBodyVehicle.make_initial_state(T_env, vx0)` is the canonical constructor: it
sets wheel spin consistent with `vx0` (avoiding false lockup at t=0), warms the thermal block to
`T_env+5..+10 °C`, sets damper `T_oil=40°C`, and places the suspension at the physically correct
static equilibrium `_Z_EQ ≈ [12.8, 12.8, 14.2, 14.2] mm` rather than at `z=0` (which would
overflow the bumpstop softplus on the first integration step).

### 3.2 SuspensionSetup — 28-Element Canonical Pytree

`SuspensionSetup` is a `NamedTuple` registered as a JAX pytree; `from_vector()`/`to_vector()` are
the only construction paths. Bounds projection uses smooth tanh rescaling so gradients survive at
the bounds.

| Index | Name | Units | Default |
|---|---|---|---|
| 0–1 | k_f, k_r | N/m | 35000, 38000 |
| 2–3 | arb_f, arb_r | N·m/rad | 800, 600 |
| 4–7 | c_low_f/r, c_high_f/r | N·s/m | 1800/1800, 1200/1200 |
| 8–9 | v_knee_f, v_knee_r | m/s | 0.10, 0.10 |
| 10–11 | rebound_ratio_f/r | — | 1.50, 1.50 |
| 12–13 | h_ride_f, h_ride_r | m | 0.025, 0.022 |
| 14–17 | camber_f/r, toe_f/r | deg | −2.0/−1.5, −0.10/−0.15 |
| 18 | castor_f | deg | 5.0 |
| 19–22 | anti_squat, anti_dive_f/r, anti_lift | — | 0.30, 0.40/0.10, 0.20 |
| 23 | yaw_target_gain | — | 0.80 |
| 24 | brake_bias_f | — | 0.60 |
| 25 | h_cg | m | 0.285 |
| 26–27 | bump_steer_f/r | rad/m | 0.0, 0.0 |

`SuspensionSetup.from_legacy_8(v8)` upgrades the historical 8-parameter MORL vector
`[k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f]` into the full 28-vector by filling
unspecified fields with `DEFAULT_SETUP`. The 40-parameter extended schema described in earlier
revisions (heave springs, inerters, rising-rate MR, independent LS/HS rebound) is defined in the
setup-freeze/design-exploration tooling but the physics engine's canonical vector remains 28-wide
in the current `vehicle_dynamics.py`.

---

## 4. PassiveHNet — Structurally Passive Neural Hamiltonian

**File:** `physics/h_net_icnn.py`

This is the single biggest architectural change since the 46→108-DOF migration, and the reason
Test 1 (Neural Convergence) and Test 2 (Forward Pass energy budget) now pass reliably: passivity
is no longer a training objective the network can violate under distribution shift — it is
**baked into the function class**.

$$H_{net}(q, p, \text{setup}) = K(p) \cdot \psi(q, \text{setup}) + V(q, \text{setup})$$

with three independently-constructed submodules:

**`KineticNet` — `K(p) = ICNN(p²) − ICNN(0)`.** `_KineticICNN` is an Input-Convex Neural Network
with all-non-negative weights (`softplus`-parameterised), applied twice to the *same* parameters
(Flax reuses the param dict for a submodule invoked twice under one name) at `p²` and at `0`. This
gives, for **any** weights:
- P1 `K(p) ≥ 0` (ICNN is monotone non-decreasing from 0 on non-negative inputs)
- P2 `K(0) = 0` by construction
- P3 `∂K/∂p|_{p=0} = 0` (chain rule: `∂K/∂pᵢ = 2pᵢ·∂ICNN/∂(pᵢ²)`, vanishes at `p=0`)
- P4 `p·∂K/∂p ≥ 0` (both factors of `2Σpᵢ²·∂ICNN/∂(pᵢ²)` are non-negative)

**`PsiGate` — `ψ(q, setup) = softplus(MLP(q, setup)) ≥ 0`.** A small swish-MLP gate; strictly
positive, so it cannot flip the sign of `K`.

**`PotentialNet` — `V(q, setup) = ICNN(q_film) − ICNN(q_film|_{q=q_eq})`,** grounded at
equilibrium via the same submodule-reuse trick. FiLM conditioning (`γ(setup), β(setup)` applied
affinely to `q_centered = q − _Z_EQ`) preserves ICNN convexity in `q` while letting the setup
vector modulate the energy landscape's curvature and offset — this is what lets `MORL` see
`∂grip/∂k_f` through the neural residual.

The full residual is capped smoothly, `H = h_cap · tanh(H_raw / h_cap)` with `h_cap = 50,000 J/m²`
(raised from an earlier 15,000 J/m² cap to avoid clipping gradient signal during extreme
elastokinematic transients).

**Integration into `_compute_derivatives`.** `vehicle_dynamics.py` computes the *exact* linear
physics gradient (`dH_dq_phys`, `dH_dp_phys`, from the kinetic-prior and structural spring term)
directly, and adds the `PassiveHNet` gradient (`dH_dq_nn`, `dH_dp_nn`) computed via `jax.grad` and
wrapped in `jax.lax.stop_gradient` for the scan-internal Newton iterations — this severs the
second-derivative Hessian blow-up risk from differentiating *through* an already-differentiated
network inside a 5-substep GLRK-4 scan, while still letting gradients flow to `H_params` in the
dedicated `_compute_derivatives_with_h` path used by the online system-ID entry point
(`step_with_params`). Both `q` and `p` are fully left–right mirrored (including the previously
missing suspension-corner momenta) and the Hamiltonian is evaluated symmetrically,
`0.5·(H(q,p) + H(q_mirror,p_mirror))`, which is what makes the `test_mirror_symmetry_zero_wz`
control test pass: under a perfectly symmetric torque input at `wz=0`, `dwz/dt` is now exactly
zero rather than leaking a static asymmetry into the yaw channel.

---

## 5. Vehicle Dynamics Integration (GLRK-4)

**File:** `models/vehicle_dynamics.py`

$$\dot{x} = (J - R)\,\nabla H(x) + F_{ext}(x, u)$$

- `J` — skew-symmetric interconnection (`J[0:14,14:28]=I`, `J[14:28,0:14]=−I`)
- `R` — `NeuralDissipationMatrix`: `R = LLᵀ + diag(softplus(d))`, masked to heave/roll/pitch/
  unsprung-z DOFs via a fixed 0/1 mask
- `H` — `T_prior + V_structural + PassiveHNet_residual · susp_sq_eq` (see §4)
- `F_ext` — tire forces (Pacejka MF6.2 + PINN + GP), aero (`AeroPlatformModel`), gravity, and a
  **kinematic gauge lock**: a stiff virtual PD spring (`k_gauge=500,000`, `c_gauge=10,000`) tying
  the independently-integrated rigid-body DOFs (`Z, φ, θ`) to the suspension corner heights,
  preventing the body from tumbling when open-loop instability transiently disagrees with the
  4-corner kinematic constraint.

**Integrator.** 2-stage Gauss-Legendre RK4 with Butcher tableau
`a = [[1/4, 1/4−√3/6],[1/4+√3/6, 1/4]]`, `b=[1/2,1/2]`. The Newton fixed-point solve for the two
implicit stages runs for **4 iterations under `stop_gradient`** (severs the ~10¹⁵ Jacobian-chain
explosion that a naive 64-step backward scan would otherwise produce) followed by **1 final
Newton iteration with gradients enabled**, which reattaches the converged solution to `u` and
`x0` — giving a correct one-step gradient at the cost of truncated (not full) backpropagation
through the fixed-point iteration itself. Auxiliary substates (thermal/slip/damper/elastokin) are
integrated by trapezoidal rule at the converged stage derivatives — no extra
`_compute_derivatives` call.

`simulate_step()` wraps `n_substeps=5` GLRK-4 steps per call under `jax.checkpoint` to bound peak
memory during the backward pass of long rollouts (WMPC horizons, MORL mini-laps).

---

## 6. Suspension Package — Kinematics & Elastokinematics

**Files:** `suspension/kinematics.py`, `suspension/elastokinematics.py`

### 6.1 Full Double A-arm Kinematic Solver

Solves the double-A-arm loop-closure constraint `F(θ) = [F1, F2, F3] = 0` (upright rigid-body
distance, wheel-centre heave target, tie-rod length) for `θ = [θ_LA, θ_UA, ψ]` via 8-step Newton
under a `jax.custom_vjp` — the backward pass solves the adjoint system
`J_θᵀ v = g` once and propagates through `J_z`, `J_dL_tr`, `J_ψshim` rather than unrolling the
Newton tape. Rodrigues rotation is used throughout; `rotation_align` replaces
`jnp.linalg.norm(cross)` with `sqrt(dot(cross,cross)+ε)` because the rear ball joints have `X=0`
exactly in float32, making `cross=0` exactly at `θ=0` — `jnp.linalg.norm`'s gradient `x/|x|` is
NaN there, while the safe form's gradient is `0`, which is physically correct (identity rotation).

Motion ratio `MR = dL_spring/dz` is recovered via a second nested 8-step Newton (rocker angle
`φ`) plus one more implicit-function-theorem step (`dφ/dθ_arm`), branching at *trace time* on
`actuation_type` (pushrod vs pullrod) since `act_sign` is a Python float, not a traced value.

Outputs per corner: camber, toe, wheel position, motion ratio, roll-centre height (via the VPP
force-line intersection method), all differentiable w.r.t. `delta_L_tr` and `psi_shim`.

### 6.2 Elastokinematic Bushing Model

Six bushings per corner (lower-fore/aft, upper-fore/aft, tie-rod, pushrod), each a Bouc-Wen
hysteretic element:

$$\dot z = A\dot u - \beta|\dot u||z|^{n-1}z - \gamma \dot u|z|^n, \qquad F = K_{eff}(x)\cdot x + c\dot x + z$$

with a nonlinear hardening stiffness `K_eff(x) = K_base·(1 + α·(|x|/x_ref)^p)` and rate-dependent
stiffness boost at high deflection velocity. `compute_elastokinematic_corrections` derives
compliance steer (from tie-rod deflection, sign-gated by `sign(Fy)` for the stabilising
toe-in-under-load convention), compliance camber (from upper/lower A-arm deflection asymmetry),
and compliance caster (fore/aft bushing asymmetry) — all clipped to physical limits (±2.9° toe,
±1.7° camber, ±1.1° caster) via smooth tanh, not hard clip.

---

## 7. Multi-Fidelity Tire Model

**File:** `models/tire_model.py`

Four-layer stack, unchanged in spirit from earlier revisions but with three GP-vX2/vX3 numerical
fixes that materially affect solver behaviour:

1. **`SpectralDense`** — power-iteration spectral normalisation (`W_sn = W/σ`) with
   `stop_gradient(σ)`; without the stop-gradient, Adam sees `∂L/∂σ` and *minimises* σ, which makes
   `W/σ` — the opposite of the intended Lipschitz-≤1 guarantee.
2. **`SparseGPMatern52`** — now a genuine **Spectral Mixture kernel** (Q=4 mixture components,
   Bochner's theorem — reduces to Matérn/RBF-like behaviour at init but is a universal
   approximator of stationary covariances), with:
   - Cholesky factorisation (`jitter=1e-3`) instead of `jnp.linalg.inv` — inversion's backward
     pass squares the condition number (`cond² ≈ 1e8` at float32, overflowing) whereas the
     triangular-solve backward is `O(n²)` and numerically stable
   - `stop_gradient` applied **only to `L`**, not to `solve_triangular(L, k_xZ)` — this keeps
     `∂σ/∂x_star` alive (needed for the GP LCB penalty gradient into the WMPC objective) while
     freezing the ill-conditioned inverse itself
   - `softplus` variance floor instead of `jnp.maximum` — the latter has a zero subgradient
     exactly at the kink where the WMPC's LCB penalty needs signal
3. **`TireOperatorPINN`** — 8-feature input `[sin α, sin 2α, κ, κ³, γ, Fz/1000, Vx/20, T_norm]`;
   `T_norm = tanh((T_eff − T_opt)/30)` gives the deterministic drift correction access to thermal
   state, closing the loop between the 3D thermal model (§8) and the Pacejka residual.

LCB penalty is capped at `clip(2σ, 0, 0.15)` — a 15% maximum uncertainty derating, justified
because the PINN corrects a *residual* on top of Pacejka, not the baseline physics itself.
Combined-slip `Gyk`/`Gxa` reduction terms were corrected so `G_yk` is driven by shifted **κ**
(not α) and `G_xa` by shifted **α** (not κ) — the previous code accidentally made combined slip
have zero effect.

---

## 8. Aero, Damper & Track Surface Models

### 8.1 `AeroPlatformModel` (`models/aero_platform.py`)

Physics-structured (not black-box MLP) aero model:

$$C_{l,eff}(rh,\theta,\phi,\psi) = C_{l,max}\cdot\Gamma_{ge}(rh)\cdot\Gamma_{pitch}(\theta)\cdot\Gamma_{roll}(\phi)\cdot\Gamma_{yaw}(\psi)$$

`Γ_ge` is a product of two sigmoids (stall wall + high-ride-height decay wall) shaped by a
Gaussian bump centred at `rh_peak`, normalised so `Γ_ge(rh_peak)=1`. CoP migrates forward under
nose-down pitch via a smooth-tanh-clamped affine map. Roll always *reduces* Cl (`dCl_droll2 < 0`,
entering as `roll²` — even function, symmetric by construction). Yaw sensitivity uses an
effective-sideslip proxy `β_eff = atan2(wz·L/2, vx)` feeding a `cos²β` projection loss plus a
`sin(2β)` side-force coefficient.

### 8.2 Maxwell Damper (`models/damper_hysteresis.py`)

Two-branch generalized Maxwell model in series with the static bilinear damper: each branch
integrates `dF/dt = k·v − F/τ_eff(T,ẋ)`, where `τ_eff` scales with an Arrhenius-like viscosity
factor (`exp(0.015·(40−T_oil))`) and a rebound/bump asymmetry multiplier. Cavitation is modelled
as a smooth force-reduction factor active only during high-speed rebound
(`1 − is_rebound·(1 − max(cav_reduction, 0.3))`), never dropping below a 30% gas-spring residual.
Thermal ODE: `C_oil·dT/dt = P_dissipated − h_cool·(T_oil − T_env)`, with dissipated power computed
correctly as `F²/(k·τ)` per branch (only dashpots dissipate; springs store/return energy).

### 8.3 Track Surface (`models/track_surface.py`)

`track_surface.py` provides two complementary layers:
- **Geometric layer** (`TrackGeometry`): periodic cubic B-spline centerline/boundaries, an
  arc-length lookup table built via 4-point Gauss-Legendre quadrature per knot segment, and a
  differentiable `query(track, s)` returning position/tangent/normal/curvature/half-widths —
  used by the WMPC's arc-length-projected tracking cost.
- **Physics layer** (`TrackSurfaceState`): a `(N_s, N_n)` rubber-level grid that accumulates
  `+0.01·dt` per pass at the current `(s,n)` cell, and a track-temperature field with a static
  shadow mask; `query_track_friction` returns `μ = base_μ + rubber_level·boost`.

---

## 9. Slip Observers — Koopman, RLS & Cayley-Stable Koopman TV

### 9.1 Koopman-Bilinear Slip Observer (Primary, `powertrain/modes/advanced/koopman_slip.py`)

Lifts scalar slip into an 8-term dictionary `φ(κ) = [1, κ, sin(Bκ), cos(Bκ), atan(Bκ),
sin(C·atan(Bκ)), κ², tanh(Bκ)]` with `B=10, C=1.65` chosen so the Pacejka MF6.2 curve lies
*exactly* in the span of `φ`. A linear Kalman filter identifies coefficients `c ∈ ℝ⁸` from
`Fx ≈ cᵀφ(κ)` — because this is linear regression, the posterior covariance `P` is exact Gaussian
propagation, not an approximation. κ* is extracted by 4-step scalar Newton on
`f(κ) = cᵀφ'(κ) = 0`, globally convergent because the Pacejka peak is unique. Uncertainty
`σ(κ*)` comes from the implicit function theorem: `∂κ*/∂c = −φ'(κ*)/(cᵀφ''(κ*))`,
`σ² = (∂κ*/∂c)ᵀP(∂κ*/∂c)` — this feeds directly into the CBF's robust safety margin as a
`σ_GP` replacement. This architecture replaced a pure RLS-secant estimator specifically because
RLS's secant/gradient-blend method requires observations from *both* sides of the peak to locate
it reliably; on a monotone slip ramp, RLS overshoots past κ* before the sign of the slope has
had a chance to flip.

### 9.2 RLS Slip-Slope Observer (Secondary, `powertrain/modes/advanced/rls_tc.py`)

Scalar forgetting-factor RLS (`λ=0.985`, effective window ≈ 66.7 steps ≈ 333 ms at 200 Hz)
identifying `θ = dFx/dκ` directly from natural torque/slip variation — no dither injection
required, unlike DESC. κ* extraction blends a secant zero-crossing method (dominant when `Δθ` is
large — good two-point resolution) with a gradient-ascent step (dominant near the peak, where
`Δθ≈0`), via a sigmoid-conditioned blend weight. Both RLS and DESC feed a continuous
SNR-weighted fusion (`w_rls = σ((SNR_rls − SNR_desc)/τ)`), not a hard mode switch.

### 9.3 Cayley-Stable Koopman TV Operator (experimental, `koopman_stable.py`)

Addresses the root cause of why `koopman_tv.py`'s `trained_blend` remains locked at `0.0`
(spectral radius `ρ(K) > 1` in every trained operator, meaning the lifted-space propagation
diverges). `CayleyKoopman` parameterises `K = (I+A)(I−A)⁻¹` with `A=(W−Wᵀ)/2` skew-symmetric,
guaranteeing `ρ(K)=1` algebraically for *any* `W` — the same "constrain the function class, not
the loss" philosophy as `PassiveHNet`. The action of `K` is applied via a 6-term Neumann series on
a normalised `A` rather than materialising the matrix inverse, and per-mode learnable decay
(`log_decay → sigmoid`) allows genuinely dissipative modes below the unit circle. Paired with a
**risk-sensitive LQR** (exponential-cost Riccati recursion with risk parameter `θ`, quadratic in
grip utilisation `ρ_util`) that automatically becomes more conservative as the car approaches the
friction limit — this is the differentiable analogue of what the CBF enforces with hard
constraints. This module is validated but not yet wired into the production `tv_step()`; see §15.

---

## 10. State Estimation — 14-State UKF

**File:** `powertrain/state_estimator.py`

A JAX-native Unscented Kalman Filter estimating
`[vx, vy, wz, Fz×4, α_t×4, IMU_accel_bias×3]` (14-dim) from
`[ax_imu, ay_imu, wz_gyro, ω_wheel×4, δ_steer, vx_gps]` (10-dim, padded). The process model is a
lightweight bicycle-kinematics-plus-load-transfer surrogate (not the full 108-DOF plant — too
expensive for `2n+1=29` sigma points at 200 Hz); the measurement model maps state directly to
expected sensor readings (`ax_pred = −vy·wz + bias`, `ay_pred = vx·wz + bias`, wheel speeds from
`vx ± wz·track/2` over `R_wheel`). Sigma-point generation uses `α_ukf=1.0` (required for float32
numerical stability — the Wan–van der Merwe default `α=0.001` produces a near-singular scaling
term at 32-bit precision) and the covariance update uses the standard `P − KSKᵀ` form rather than
the Joseph form, which halves the number of `14×14` matrix products per step at this operating
precision. `extract_estimated_state()` is the sole interface consumed by
`powertrain_manager.powertrain_step_v2()`, so the powertrain control stack never reads simulated
ground truth directly in the UKF-integrated path.

---

## 11. Differentiable Wavelet MPC (Diff-WMPC)

**File:** `optimization/ocp_solver.py`

### 11.1 Wavelet Parameterisation & Best-Basis Entropy

Control trajectories are optimised in a 3-level Daubechies-4 coefficient space (`_db4_dwt`/
`_db4_idwt`, exact inverse pair via `mode='valid'` periodic convolution). A **separate**
Coifman–Wickerhauser best-basis computation (`_wpd_full_tree` + `_shannon_entropy`, soft
bottom-up node selection via `sigmoid(20·(H_parent − H_children))`) is used purely as an entropy
*regulariser* in the loss (`CW_ENTROPY_WEIGHT·(entropy_ch0 + entropy_ch1)`) — it is **not** used
as the actual transform basis, because the WP-tree coefficient layout is not exactly invertible by
`_idwt_1d_3level`, which would make the DWT∘IDWT round-trip non-identity and poison the gradient
landscape.

### 11.2 Pseudo-Huber Detail Regularisation

$$\mathcal{R}_{PH}(w;\delta) = \delta^2\left(\sqrt{1+(w/\delta)^2} - 1\right)$$

applied per detail band (D3×0.5, D2×1.0, D1×12.0 — heavily weighting the highest-frequency band
to suppress steering chatter) instead of a non-differentiable L1 penalty.

### 11.3 Dual Augmented Lagrangian — Friction + Spatial

Two independent AL constraint blocks, each with its own ρ-schedule
(`[0.5, 5.0, 50.0, 500.0, 2500.0]` for friction, `[0.5, 5.0, 25.0, 250.0, 2500.0]` for the spatial
track-boundary constraint):

$$\mathcal{L}_{AL} = \lambda^{\!\top}\max(c(x),0) + \tfrac{\rho}{2}\lVert\max(c(x), -\lambda/\rho)\rVert^2$$

Friction constraint: `g = (a_lat² + a_lon²)/(μg)² − 1 ≤ 0`. Spatial constraint uses the
Unscented-Transform-propagated lateral-position tube half-width
`κ_safe·√max(var_n, 1e-4)` against left/right track boundaries, **plus** a soft pre-boundary
quadratic ramp active 1.5 m before the wall — giving the optimizer gradient signal *before* the AL
term engages, since a trajectory that starts inside the track produces a zero AL gradient by
definition until it's already violated the boundary.

### 11.4 Backward-Pass Stability: Selective `stop_gradient` on the Scan Carry

The dominant historical NaN source (documented at length in `sanity_checks.py` §3) was a 64-step
`lax.scan` backward Jacobian chain overflowing float32 because the yaw-angle carry's per-step
eigenvalue (`sech²(dpsi/0.08)` times the H_net yaw Jacobian) compounds to `~1.7⁶³ ≈ 6e14`. The fix
(`_apply_carry_stop_gradient`) severs the cross-step gradient on the position/attitude/lateral-
velocity/angular-rate carry components while leaving the per-step emitted cost gradients
(computed on the *forward* value, not the carry) fully intact — a truncated-BPTT compromise that
is finite and still gives Adam/L-BFGS-B a useful descent direction.

### 11.5 Physics P-Controller Warm Start

`_build_physics_warmstart` runs a genuine closed-loop pure-pursuit + friction-budgeted
longitudinal P-controller through the *actual* vehicle model (not a kinematic proxy) to produce
`U_warm` in physical Newton units, matching the `u[1]` channel the optimizer sees — this fixed a
GP-vX3-era bug where a units-mismatched (m/s²) warm start produced `flat_init ≈ 0`, triggering
spurious `gtol` convergence after a single L-BFGS-B iteration.

---

## 12. MORL-SB-TRPO Setup Optimiser

**File:** `optimization/evolutionary.py`

20-member Chebyshev-spaced ensemble (`ω_i = 0.5(1−cos(iπ/(N−1)))`, concentrating ~65% of members
in the high-grip region `ω∈[0.7,1.0]`) optimising the 28-dim `SuspensionSetup` logit space.

**ARD Bayesian cold start** (`BayesianOptColdStart`): 20 random + 80 EI-guided evaluations with a
per-dimension-lengthscale squared-exponential GP; lengthscales are updated via a Pearson-
correlation heuristic every 5 iterations, so grip-insensitive dimensions (e.g. castor) acquire
large lengthscale and are effectively pruned from the acquisition search without explicit feature
selection. The best 5 diverse basins (greedy max-distance selection, `d_min > 0.10` in
normalised space) seed the ensemble's initial logits.

**Riemannian Natural Policy Gradient** (`_apply_rnpg_ensemble`): replaces the KL trust region with
a metric pulled back through the physics engine,
`G_phys = JᵀSJ + λ·diag(JᵀSJ + εI)` where `J = ∂[grip,stability]/∂μ`, refreshed every 10 gradient
steps (not every step — the Jacobian is the expensive part). Levenberg-Marquardt-style diagonal
damping (not Tikhonov `+λI`) preserves scale invariance across parameters with radically different
units (N/m vs degrees). Natural-gradient norm is clipped to 5.0 to guard against the
first-Jacobian-refresh overshoot when the BO-seeded basin has high curvature.

**Safety.** `SAFETY_THRESHOLD=0.10` G minimum grip (soft sigmoid gate in the loss), plus a hard
post-hoc archive filter: any setup with step-steer overshoot `> STABILITY_MAX = 5.0 rad/s` is
excluded from the Pareto archive entirely, not merely penalised. NaN-recovery logic snaps the
whole ensemble back to the BO basins (max 5 recoveries) if every member simultaneously produces a
non-finite evaluation — this happens occasionally when Adam's un-clipped natural gradient pushes a
member into a numerically pathological corner of setup space.

---

## 13. Powertrain Control Stack

**File:** `powertrain/powertrain_manager.py` — single `@jax.jit powertrain_step()` entry point.

```
 1. Virtual Impedance         → filtered throttle/brake (2nd-order flywheel+damper, PIO mitigation)
 2. Acceleration Estimation   → low-pass ax, ay (IMU-fused with a kinematic fallback for test envs)
 3. Traction Control          → tc_step(): Koopman-primary κ*, RLS-fallback, Pacejka clip guard,
                                  GP-sigma fusion, TC/TV blend weights
 4. Driver Force Demand       → Fx_demand from filtered pedals, speed-gated at low vx
 5. Motor Torque Limits       → field-weakening envelope × thermal derating (motor + inverter)
 6. Degradation Assessment    → sensor-confidence-scaled friction budget (quadratic shrinkage)
 7. Yaw Rate Reference        → driver-intent-aware, counter-steer-detecting target ψ̇_ref
 7b. Slip Barrier Inputs      → Koopman κ* + σ(κ*) packed into predictive slip-CBF rows
 8. mpQP KKT Allocation       → single 24×24 KKT solve (V2, slip-aware), classifier-conditioned
                                  active set, 3-step projected-gradient polish fallback
 8b. Dynamic Regen Blend      → battery/thermal-derated α*, hydraulic brake residual
 9. CBF Safety Filter         → input-delay robust DCBF (β, ψ̇, per-wheel slip), GP-uncertainty-
                                  shrunk safe set
10. Launch Control v2.1       → button-armed FSM, TC ceiling, real-time μ EMA, yaw-lock PI,
                                  smooth abort path
11. Output Smoothing          → EMA + tanh soft rate-limiter
12. Powertrain Thermal        → motor/inverter/battery state update
13. Diagnostics Packaging     → full PowertrainDiagnostics struct for telemetry
```

### 13.1 mpQP KKT Torque Allocator

**File:** `powertrain/modes/advanced/explicit_mpqp_allocator.py`

Replaces the 12-iteration projected-gradient SOCP (still retained as `solve_torque_allocation`,
the always-available fallback in `torque_vectoring.py`) with a single KKT linear solve
conditioned on a neural active-set classifier.

| Version | Constraints | KKT dim | θ dim |
|---|---|---|---|
| V1 | 12 (box × 2 + friction) | 16×16 | 15 |
| V2 | 20 (V1 + 8 slip-CBF rows) | 24×24 | 19 |

V2's active-set prediction is **soft** (`predict_active_set_soft`, temperature-scaled sigmoid on
classifier logits rather than a hard threshold), making `∂T*/∂θ` continuous everywhere — a
prerequisite for any future bilevel gradient closure through the allocator. `slip_barrier.py`
builds the 8 slip-CBF rows from a first-order Euler prediction of slip at the actuator-delay
horizon (`τ_delay=15ms`), linear in `T` by construction. If the V2 classifier bundle fails to load
(`active_set_classifier_v2.bytes` missing), `make_explicit_allocator_step_auto` falls back
gracefully to V1 plus an extended-polish safety net that still enforces the slip constraints via
projected gradient, logged as `v1+slip-polish`.

### 13.2 Dynamic Regen Blend

**File:** `powertrain/regen_blend.py`

Computes the battery-feasible regen power budget from SoC taper (`sigmoid`-gated above 92% SoC)
and cell-temperature derating (linear ramp 40→55°C), then scales *only* the regenerative
(negative-torque) component of the allocator output via a smooth `sigmoid(−k·T/50)` gate — drive
torques pass through unchanged. The hydraulic brake residual fills whatever the battery budget
cannot absorb: `F_hydraulic = hydraulic_gain · softplus(F_brake_demand − F_regen_achieved)`.

### 13.3 Launch Control v2.1

**File:** `powertrain/modes/advanced/launch_control.py`

Button-armed 6-phase FSM (`IDLE→ARMED→LAUNCH→HANDOFF→TC`, plus a smooth-sigmoid abort path back to
`IDLE` on hard braking) layered on top of the original brake+throttle legacy trigger for backward
compatibility. Per-wheel TC ceiling (`Fx_peak = μ_rt·Fz`, `T_ceiling = Fx_peak·r_w·kappa_margin`)
bounds the B-spline profile without depending on κ* directly — DESC/Koopman modulate finely
*inside* this conservative envelope. A yaw-lock PI (speed-gated, anti-windup-clamped) applies
differential left/right torque correction during LAUNCH/HANDOFF, clipped to `[0, T_ceiling]` per
wheel so it can never itself exceed the friction-derived ceiling.

---

## 14. Sanity Check Suite

**File:** `sanity_checks.py` — **all tests below currently pass.**

| # | Test | Verifies |
|---|---|---|
| — | Mirror symmetry (wz=0) | `dwz/dt = 0` exactly under symmetric torque at zero yaw rate |
| 1 | Neural Convergence | PassiveHNet/R_net training converges; weights + scale persisted |
| 2 | Forward Pass | 108-DOF GLRK-4 step finite; passive energy budget not injected (ΔKE ≤ 150 mJ) |
| 3 | Circular Track / WMPC | DWT/WPD round-trip, entropy regularisation, full Diff-WMPC solve, friction/spatial/heading/lap-time diagnostics |
| 4 | Friction Circle | Combined-slip Fy reduction 3–40%; Spectral-Mixture GP σ finite and correctly ordered in/out of distribution |
| 5 | Load Sensitivity | Fy_peak degressive with Fz (1.2×–1.9× per doubling) |
| 6 | Diagonal Load Transfer | Correct corner Fz ordering under combined braking+cornering |
| 7 | Aero v² Scaling | `Fz_aero ∝ v²` exactly (Cl/Cd constant) |
| 8 | Differential Yaw Moment | Correct-sign yaw response to asymmetric hub-motor torque |
| 9 | Optimizer Boundary Diversity | k_f not pinned at lower bound; stability cap enforced |
| 10 | Motor Torque Envelope | Low-speed torque ≈ T_peak; monotone field-weakening decrease; non-negative envelope |
| 11 | SOCP/KKT Allocator | Friction-circle feasibility; near-zero yaw moment on straight; sub-ms solve time |
| 12 | CBF Safety | Intervenes near sideslip/yaw limits; robust-CBF intervention grows with GP σ |
| 13 | DESC Convergence | κ_base converges to the true Pacejka peak within 0.02–0.05 |
| 14 | Launch State Machine | Full IDLE→ARMED→LAUNCH→HANDOFF→TC traversal via button arming |
| 15 | Virtual Impedance | >30° phase lag at 3 Hz (breaks PIO loop); <150 ms 90% rise time |
| 16 | Full Pipeline JIT | 28-field diagnostics finite; thermal/SoC state evolves under load |
| 17 | Koopman Slip Observer | κ* converges to true Pacejka peak within 0.015; correct slope sign above peak |
| 17b | Koopman E-ABS Slip Containment | Lockup torque flagged infeasible; polish restores feasibility; gradient finite; disabled-barrier passthrough |
| 18 | Dynamic Regen Blend | α monotone ↓ with SoC/temperature; zero spurious hydraulic on throttle; battery-limited deficit filled by hydraulic; ∂α/∂Fx finite |
| 19 | 108-DOF State Integrity | All state blocks correctly initialised (thermal, damper, slip, elastokin) |
| 20 | Ground Effect Stall | Γ_ge ≈ 1 at peak ride height, <0.5 below stall; CoP migrates forward under nose-down pitch |
| 21 | Damper Hysteresis | Rebound stiffer than compression; T_oil rises under oscillation; cavitation suppresses extreme rebound force |
| 22 | Tire Thermal 3D | Camber-induced load asymmetry; outer rib heats faster under negative camber; positive dT/dt under slip |
| 23 | Elastokinematics | Nonzero, load-scaling compliance steer; Fx–Fy coupling present; finite gradient |
| 24 | 2nd-Order Tire Transient | σ decreases with load and near-peak slip; bandwidth increases with speed; physically realistic settling time |
| 25 | Track Surface | Racing line has higher μ than dusty inside; shadow zone cooler; rubber deposits Gaussian around the racing line; finite ∂μ/∂s |

Running `python sanity_checks.py` executes the full suite end-to-end (~30 min on first run due to
XLA compilation; subsequent runs hit the compilation cache).

---

## 15. Known Issues, Limits & Diagnostics

### 15.1 Koopman TV `trained_blend = 0.0` — still locked, now with a validated path forward

The production `koopman_tv.py` yaw-moment controller remains PD-only. Two independent
prerequisites are now satisfied that were not true in earlier revisions: (a) `PassiveHNet` gives a
stable, passive 108-DOF plant to generate training rollouts from, and (b) `koopman_stable.py`
(§9.3) provides a Cayley-parameterised operator with `ρ(K)=1` guaranteed by construction, removing
the spectral-radius-blowup failure mode that caused the original lock. The remaining work is
purely a retraining/validation exercise: run `scripts/train_koopman_hnet.py` against
`PassiveHNet` rollouts (this script is already wired for the 108-DOF state and includes its own
sanity checks — `Mz(ψ̇_err=0)≈0`, correct-sign `Mz(ψ̇_err=1)`, finite gradient, ≥20% saturation
regime coverage), then shakedown-validate at `trained_blend=0.3` before raising further.

### 15.2 Dashboard / Telemetry Wiring

The React dashboard (referenced in prior revisions) and the live 200 Hz physics/WebSocket bridge
are not present in this snapshot of the repository; `scripts/gradient_server.py` is the only
externally-facing service currently implemented, exposing short-horizon Jacobian sensitivities
(`∂v_x(100ms)/∂setup`, extrapolated to `∂lap_time/∂setup` via a track-length scaling) over HTTP.
Any dashboard work should treat this as the sole ground-truth sensitivity source until a
full-lap gradient endpoint is (re-)added.

### 15.3 Float32 / Precision Notes

- `ocp_solver.py` runs under `jax.config.update("jax_enable_x64", True)` globally — the WMPC's
  64-step backward scan needs the extra mantissa bits even after the carry-stop-gradient fix;
  the rest of the stack (vehicle dynamics, powertrain, MORL) runs float32.
- `suspension/kinematics.py` also enables x64 locally, for the same reason (Newton residual
  tightened to `1e-9`).
- Bouc-Wen `z` states and Maxwell damper branch forces are hard-clipped (not just soft-bounded) as
  a last-resort float32 safety net (`±5000N` / `±3000N` for the two branches respectively) — this
  is intentionally *not* purely differentiable-smooth, since it is a physical impossibility guard,
  not a control-law shaping term.
- `active_set_classifier.py` V1/V2 both require their `.bytes`/`.npy` weight files to be trained
  via `scripts/generate_qp_training_data.py` (`--v2` flag for the 500k-sample slip-aware set)
  followed by `python -m powertrain.modes.advanced.active_set_classifier [--v2]` before the KKT
  path activates; the projected-gradient fallback is always available regardless.

### 15.4 Three Standing Safety Priorities (FSG 2026)

1. **Actuator-delay CBF** — addressed by the input-delay predictive DCBF in
   `cbf_safety_filter()`; the slip barrier adds a second, independent defence layer inside the
   KKT allocator itself.
2. **EMI/bit-flip resilience in embedded memory** — not yet addressed at the software layer; ECC
   memory or CRC-checked state transfer remains an open hardware-integration item.
3. **4WD ABS `v_x` anchoring under simultaneous multi-wheel lockup** — the UKF (§10) partially
   mitigates this by fusing GPS/IMU rather than relying solely on wheel-speed-derived `v_x`, but
   has not been stress-tested against a full 4-wheel lockup scenario.

---

## 16. Pipeline Execution

```bash
# 0. Always import jax_config first in any entry script
python -c "import jax_config"

# 1. Full sanity suite (physics + powertrain + 108-DOF subsystems)
python sanity_checks.py

# 2. Train PassiveHNet / R_net residuals against synthetic chassis-flex data
python -c "from optimization.residual_fitting import train_neural_residuals; train_neural_residuals()"

# 3. Generate mpQP training data and train the active-set classifiers
python -m scripts.generate_qp_training_data              # V1: 100k samples, 15D/12C
python -m scripts.generate_qp_training_data --v2          # V2: 500k samples, 19D/20C (slip-aware)
python -m powertrain.modes.advanced.active_set_classifier          # V1 classifier
python -m powertrain.modes.advanced.active_set_classifier --v2     # V2 classifier

# 4. (Optional, once PassiveHNet is retrained) Retrain the Koopman TV operators
python scripts/train_koopman_hnet.py --n_samples 500000 --n_epochs 500
python scripts/train_koopman_hnet.py --quick               # ~5 min pipeline smoke test

# 5. Run MORL-SB-TRPO design exploration
python scripts/run_ter27_design_exploration.py --phase 1   # all params free
python scripts/run_ter27_design_exploration.py --phase 2   # CAD-locked geometry
python scripts/run_ter27_design_exploration.py --phase 3   # springs/dampers/ARBs only

# 6. Benchmark the SOCP/KKT allocator against the 5 ms real-time budget
python scripts/benchmark_socp_latency.py --n-warmup 5 --n-trials 200

# 7. Twin-fidelity validation pipeline (FSG award submission artefact)
python scripts/run_twin_fidelity_demo.py --track fsg_autocross --duration 30

# 8. Gradient server (setup sensitivity endpoint)
python scripts/gradient_server.py --port 8766 --precompile
```

### Key Output Artefacts

| File | Contents |
|---|---|
| `models/h_net.bytes` / `r_net.bytes` | Trained PassiveHNet / R_net weights |
| `models/h_net_scale.txt` | Diagnostic normalisation record (not fed into the architecture) |
| `models/active_set_classifier[_v2].bytes` | Trained active-set classifier weights |
| `models/active_set_thresholds[_v2].npy` | Per-constraint calibrated recall thresholds |
| `models/qp_training_data[_v2].npz` | Offline QP ground-truth solver training set |
| `trained/koopman_hnet/` | Retrained Koopman TV operators (post-PassiveHNet) |
| `reports/socp_latency_report.json` | Cold/warm-start allocator latency benchmark |
| `reports/twin_fidelity/twin_fidelity_report.{json,txt}` | R²/xcorr/PSD digital-twin validation report |

---

## 17. Revision History

### GP-vX6 (current)

- **Full green sanity suite.** All tests in `sanity_checks.py` (25 core + the mirror-symmetry
  control test) pass for the first time since the 108-DOF migration.
- **`PassiveHNet` (ICNN) replaces `NeuralEnergyLandscape`** as the production Hamiltonian
  residual — passivity properties P1–P4 now hold algebraically, not just as a training incentive.
  Fixes the recurring energy-injection failures in Test 1/Test 2.
- **Mirror-symmetry fix**: full q/p mirroring (including previously-omitted suspension-corner
  momenta) plus symmetric Hamiltonian evaluation eliminates a static yaw-channel asymmetry at
  `wz=0` under symmetric torque.
- **`state_estimator.py`**: new 14-state UKF, `α_ukf=1.0` float32-stable sigma points, `P−KSKᵀ`
  covariance update; wired into `powertrain_step_v2` via `powertrain_wiring_v2.py`.
- **`koopman_stable.py`**: Cayley-parameterised (`ρ(K)=1` by construction) Koopman operator +
  risk-sensitive LQR, validated standalone as the path to unlocking `trained_blend > 0`.
- **`explicit_mpqp_allocator.py` V2**: 24×24 slip-aware KKT solve with soft (temperature-scaled)
  active-set prediction, graceful V1 fallback with extended slip-polish safety net.
  `regen_blend.py` and `slip_barrier.py` integrate directly into the allocator pipeline.
  `launch_control.py` v2.1 adds button arming, per-wheel TC ceiling, real-time μ EMA, and yaw-lock
  PI on top of the legacy brake+throttle trigger.
- **`ocp_solver.py`**: Coifman–Wickerhauser best-basis entropy regularisation added alongside the
  fixed 3-level DWT; dual (friction + spatial) Augmented Lagrangian with independent ρ-schedules;
  selective `stop_gradient` on the scan carry to eliminate the dominant float32 backward-pass NaN
  source; physics-based (closed-loop pure-pursuit) warm start in correct Newton units.
- **`tire_model.py`**: Spectral Mixture GP kernel (Q=4) with Cholesky/`stop_gradient(L)`
  numerically-stable posterior variance, replacing the earlier `linalg.inv`-based Matérn 5/2 GP;
  combined-slip `Gyk`/`Gxa` sign/variable bug fixed.

### GP-vX5

**Architectural change: 46-DOF → 108-DOF state vector.** Integrated four new physics subsystems
directly into the vehicle ODE: full double A-arm kinematics + Bouc-Wen elastokinematics
(`suspension/`), Maxwell ODE damper with thermal fade (`damper_hysteresis.py`), 3D 7-node tire
thermal model with camber-induced lateral asymmetry (`tire_thermal_3d.py`), and 2nd-order
carcass+belt transient slip dynamics (`tire_transient.py`). Koopman Slip Observer promoted to
primary κ* estimator; RLS demoted to secondary/fallback with divergence cross-checking.

### GP-vX4

`explicit_mpqp_allocator.py` (V1, 16×16 KKT) and `active_set_classifier.py` introduced, replacing
the pure projected-gradient SOCP as the default allocation path. `DesignFreeze` progressive
CAD-lockdown system added. `regen_blend.py` dynamic regen α* introduced.

### GP-vX3

Full powertrain control stack introduced: `powertrain_manager.py`, `motor_model.py`,
`virtual_impedance.py`, `torque_vectoring.py` (projected-gradient SOCP + input-delay DCBF),
`traction_control.py` (DESC), `launch_control.py`, `koopman_tv.py`.

### GP-vX2

Critical Hamiltonian fixes on the (then-current) `NeuralEnergyLandscape`: `h_scale` train/inference
mismatch corrected (was causing a 102× energy amplification), `susp_sq` gate re-anchored from
`z=0` to the physical equilibrium `_Z_EQ`.

### GP-vX1

Initial `SuspensionSetup` canonical-index fix (ARB/spring-rate index collision), corrected
mass/inertia defaults, `_TRIL_14` module-level constant fix, GLRK-4 integrator, C∞ bumpstop,
4-way digressive damper (later superseded by the Maxwell model in vX5).

---

*Project-GP is a live research codebase. Physics is validated to the extent that
`sanity_checks.py` passes — as of GP-vX6, it does, in full. The next priority is retraining the
Koopman TV operators against `PassiveHNet` rollouts (§15.1) to safely raise `trained_blend` above
zero, followed by re-establishing the live telemetry/dashboard bridge (§15.2).*