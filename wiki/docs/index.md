# Project-GP: End-to-End Differentiable Digital Twin

<div class="hero-banner" markdown>

**Tecnun eRacing** · **Ter27** · **FSG 2026**

*A 46-DOF vehicle digital twin where every equation is differentiable,
every controller is formally certified, and the gradient of lap time
with respect to any setup parameter is computed in a single backward pass.*

</div>

## Why This Exists

Traditional vehicle simulators treat physics as a black box: you evaluate a model,
observe an output, and if you want sensitivity information, you run finite differences —
perturbing each parameter one at a time. For a 28-parameter suspension setup, that's
29 forward evaluations per gradient.

Project-GP inverts this paradigm. The entire physics engine — from tire contact patch
thermodynamics to the neural Hamiltonian energy surface — compiles into a single
[XLA](https://openxla.org/) computation graph. One call to `jax.grad` traces the
gradient through the complete scan, yielding all 28 partial derivatives simultaneously.
The cost: one forward pass plus one backward pass. Total: **2× the compute of a single
simulation, regardless of parameter dimension.**

This is not an incremental improvement. It is a structural change in how vehicle
dynamics simulation can be done.

---

## The Kill Feature

```python
# This line differentiates through the entire lap simulation
∂_lap_time = jax.grad(simulate_full_lap)(setup_vector)
# ∂_lap_time is a (28,) vector: the sensitivity of lap time to every setup parameter
```

No Formula Student team in the world has published this capability. The gradient traces
through:

$$
\frac{\partial t_{\text{lap}}}{\partial \mathbf{s}} = 
\frac{\partial t_{\text{lap}}}{\partial \mathbf{v}} \cdot 
\frac{\partial \mathbf{v}}{\partial \mathbf{F}_{\text{tire}}} \cdot 
\frac{\partial \mathbf{F}_{\text{tire}}}{\partial \mathbf{q}_{\text{susp}}} \cdot 
\frac{\partial \mathbf{q}_{\text{susp}}}{\partial \mathbf{s}}
$$

where $\mathbf{s} \in \mathbb{R}^{28}$ is the full suspension setup vector, computed
via reverse-mode automatic differentiation through `jax.lax.scan` over the complete lap.

---

## Architecture At a Glance

| Subsystem | Method | Key Property |
|---|---|---|
| **Dynamics** | Neural Port-Hamiltonian (46 DOF) | Energy conservation by construction |
| **Tires** | Pacejka + PINN + Sparse GP | Multi-fidelity with quantified uncertainty |
| **Control** | Diff-WMPC (Db4 wavelets) | Frequency-domain sparsity, stochastic tubes |
| **Allocation** | SOCP (12-iter projected gradient) | Friction-circle feasible, ≤5 ms on SBC |
| **Safety** | Input-Delay DCBF + GP robustness | Formally certified barrier invariance |
| **Traction** | DESC extremum-seeking + SWIFT fusion | Model-free + physics-based hybrid |
| **Optimization** | MORL-SB-TRPO (28D, 3 objectives) | Pareto-optimal setups with CMA-ES validation |
| **Estimation** | Differentiable EKF (5 states) | Online tire/CG/grip calibration |

All subsystems compile to a **single XLA graph**. The target: **200 Hz on an embedded
ARM SBC** with deterministic, bounded-latency execution.

---

## Navigating This Documentation

This reference is organized from first principles to deployment:

- **[Architecture](architecture/philosophy.md)** — Design philosophy, state vector, XLA graph structure
- **[Physics Engine](physics/port_hamiltonian.md)** — The mathematical core: Hamiltonian mechanics, symplectic integration
- **[Tire Model](tires/architecture.md)** — Multi-fidelity tire forces with uncertainty quantification
- **[Optimal Control](control/wmpc_overview.md)** — Wavelet MPC, Augmented Lagrangian, stochastic safety
- **[Powertrain](powertrain/coordinator.md)** — 13-stage torque allocation, CBF safety, traction control
- **[Setup Optimization](optimization/morl.md)** — Multi-objective RL over the 28D setup space
- **[Validation](validation/methodology.md)** — Twin fidelity methodology and deployment benchmarks

Each page provides the **mathematical formulation**, the **design rationale** (why this
method over alternatives), and **references** to the JAX implementation.
