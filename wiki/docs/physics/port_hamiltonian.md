# Neural Port-Hamiltonian Vehicle Dynamics

## Motivation: Why Not Just Use an ODE?

A standard vehicle dynamics simulator might write $\dot{x} = f(x, u)$ and integrate
with RK4. This works — but it offers **no structural guarantee** on energy conservation.
Over a 22 km endurance event (~3600 seconds at 200 Hz = 720,000 integration steps),
numerical energy drift can accumulate to physically meaningless states: phantom
acceleration without input power, or dissipation that violates the second law.

Port-Hamiltonian (pH) systems solve this at the architectural level. Instead of
learning an arbitrary ODE, we learn the **energy function** $H(q, p)$ and the
**dissipation structure** $R(q, p)$, then derive the dynamics from these. Energy
conservation becomes a _theorem_, not an empirical hope.

---

## Mathematical Formulation

### The Port-Hamiltonian Structure

A Port-Hamiltonian system on a state space $\mathcal{X} = T^*\mathcal{Q}$ (the
cotangent bundle of the configuration manifold) is defined by:

$$
\begin{bmatrix} \dot{q} \\ \dot{p} \end{bmatrix} = 
\underbrace{\begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix}}_{J \text{ (interconnection)}}
\nabla H(q, p)
- \underbrace{\begin{bmatrix} 0 & 0 \\ 0 & R(q, p) \end{bmatrix}}_{\text{dissipation}}
\nabla H(q, p)
+ \underbrace{\begin{bmatrix} 0 \\ B \end{bmatrix} u}_{\text{input ports}}
$$

where:

- $q \in \mathbb{R}^{14}$ — generalised coordinates (chassis DOF + wheel DOF)
- $p \in \mathbb{R}^{14}$ — generalised momenta
- $H : \mathbb{R}^{28} \to \mathbb{R}_{\geq 0}$ — the Hamiltonian (total energy)
- $R : \mathbb{R}^{28} \to \mathbb{S}^{14}_+$ — the dissipation matrix (PSD)
- $J$ — the canonical symplectic structure (skew-symmetric, lossless)
- $B \in \mathbb{R}^{14 \times m}$ — input port matrix
- $u \in \mathbb{R}^m$ — control inputs (tyre forces, aero, powertrain)

### The Energy Balance

The key theorem: along any trajectory, the energy satisfies:

$$
\frac{dH}{dt} = \nabla H^\top (J - R) \nabla H + \nabla H^\top B u
= -\nabla H^\top R \nabla H + \nabla H^\top B u
\leq \nabla H^\top B u
$$

The $J$-term vanishes because $J$ is skew-symmetric ($x^\top J x = 0$ for all $x$).
The $R$-term is non-positive because $R \succeq 0$. Therefore:

$$
\boxed{H(t) \leq H(0) + \int_0^t \nabla H^\top B u \, d\tau}
$$

**Energy can only increase through the input ports.** The system structurally cannot
hallucinate free energy. This is not a numerical property — it is an algebraic identity
that holds regardless of integration scheme, time step, or parameter values.

---

## Neural Implementation

### H_net: Learning the Energy Landscape

The Hamiltonian is decomposed as:

$$
H(q, p) = \underbrace{H_{\text{kin}}(q, p)}_{\text{analytical}} + \underbrace{H_{\text{pot}}(q)}_{\text{analytical}} + \underbrace{\Delta H_{\text{net}}(q, p)}_{\text{learned residual}}
$$

where:

- $H_{\text{kin}} = \frac{1}{2} p^\top M^{-1}(q) p$ — kinetic energy (known mass matrix)
- $H_{\text{pot}} = mgh_{\text{cg}}(q) + \frac{1}{2} k_s z^2 + \ldots$ — potential energy (springs, gravity)
- $\Delta H_{\text{net}}$ — a neural network that learns the residual energy not captured by the analytical model

**Architectural constraint on $\Delta H_{\text{net}}$:**

$$
\Delta H_{\text{net}}(q, p) = \text{softplus}\big(\text{MLP}(q, p)\big) \cdot \sigma\Big(\frac{\|p\|^2}{\epsilon}\Big)
$$

- The `softplus` output guarantees $\Delta H_{\text{net}} \geq 0$ — the residual can only **add** energy to the baseline, never subtract it below the known physics floor.
- The sigmoid gate $\sigma(\|p\|^2 / \epsilon)$ ensures the residual vanishes at equilibrium ($p = 0$) — when the vehicle is stationary, the learned correction contributes nothing.
- The residual is capped: $\Delta H_{\text{net}} \leq H_{\text{RESIDUAL\_CAP}} = 50{,}000 \text{ J/m}^2$ to prevent float32 overflow.

### R_net: Learning the Dissipation Structure

The dissipation matrix must be positive semi-definite (PSD). We enforce this via
Cholesky factorisation:

$$
R(q, p) = L(q, p) \cdot L(q, p)^\top + \text{diag}\big(\text{softplus}(d(q, p))\big)
$$

where $L$ is a lower-triangular matrix predicted by a neural network, and $d$ is a
diagonal vector. The $LL^\top$ construction is PSD by construction, and the `softplus`
diagonal adds strictly positive definiteness when needed.

**Why Cholesky and not eigendecomposition?** Cholesky factorisation is numerically
stable in float32, has $O(n^2)$ parameters (vs $O(n^2)$ for eigendecomposition but
with orthogonality constraints that are expensive to enforce differentiably), and
the gradient through the Cholesky product is well-conditioned.

---

## Integration: Symplectic Leapfrog

Standard integrators (RK4, Euler) do not preserve the symplectic structure. Over long
simulations, they introduce artificial energy drift — exactly what the pH formulation
is designed to prevent.

We use a **Störmer-Verlet (leapfrog)** integrator:

$$
\begin{aligned}
p_{1/2} &= p_n - \frac{\Delta t}{2} \frac{\partial H}{\partial q}\Big|_{q_n, p_n} + \frac{\Delta t}{2} R(q_n, p_n) \nabla_p H + \frac{\Delta t}{2} B u_n \\
q_{n+1} &= q_n + \Delta t \frac{\partial H}{\partial p}\Big|_{q_n, p_{1/2}} \\
p_{n+1} &= p_{1/2} - \frac{\Delta t}{2} \frac{\partial H}{\partial q}\Big|_{q_{n+1}, p_{1/2}} + \frac{\Delta t}{2} R(q_{n+1}, p_{1/2}) \nabla_p H + \frac{\Delta t}{2} B u_n
\end{aligned}
$$

This is a second-order symplectic method. The energy error is bounded and oscillatory
(not growing), with magnitude $O(\Delta t^2)$ per step.

**Implementation detail:** We use 5 substeps per physics frame ($\Delta t_{\text{sub}} = \Delta t / 5 = 1 \text{ ms}$) to keep the symplectic error within float32 tolerance at the 15 m/s operating point. The entire 5-substep integration compiles as a single `jax.lax.scan` — no Python loop overhead.

---

## The 46-Dimensional State Vector

| Index | Symbol | Description | Unit |
|---|---|---|---|
| 0 | $v_x$ | Longitudinal velocity (body frame) | m/s |
| 1 | $v_y$ | Lateral velocity (body frame) | m/s |
| 2 | $\dot{\psi}$ | Yaw rate | rad/s |
| 3 | $\dot{\phi}$ | Roll rate | rad/s |
| 4 | $\dot{\theta}$ | Pitch rate | rad/s |
| 5 | $v_z$ | Heave velocity | m/s |
| 6–9 | $z_{FL,FR,RL,RR}$ | Suspension travel | m |
| 10–13 | $\dot{z}_{FL,FR,RL,RR}$ | Suspension velocity | m/s |
| 14–17 | $\omega_{FL,FR,RL,RR}$ | Wheel angular velocity | rad/s |
| 18–21 | $\alpha_{FL,FR,RL,RR}$ | Transient slip angle (relaxation length) | rad |
| 22–25 | $\kappa_{FL,FR,RL,RR}$ | Transient longitudinal slip | — |
| 26–29 | $T^{\text{rib}}_{FL,FR,RL,RR}$ | Tyre surface rib temperature | °C |
| 30–33 | $T^{\text{carcass}}_{FL,FR,RL,RR}$ | Tyre carcass temperature | °C |
| 34 | $T^{\text{gas}}$ | Tyre cavity gas temperature | °C |
| 35–38 | $T^{\text{core}}_{FL,FR,RL,RR}$ | Tyre core temperature | °C |
| 39–42 | $T^{\text{road}}_{FL,FR,RL,RR}$ | Road surface temperature (local) | °C |
| 43 | $\phi$ | Roll angle | rad |
| 44 | $\theta$ | Pitch angle | rad |
| 45 | $\psi$ | Yaw angle (global) | rad |

!!! info "Design Decision"
    The state is larger than necessary for a point-mass model but smaller than a
    full multi-body system. The 46-DOF count was chosen to capture all phenomena
    that affect lap time at the 0.1 s level: transient tyre slip (relaxation length
    dynamics), thermal grip variation, and full 4-corner suspension kinematics.
    Phenomena below the 0.1 s threshold (driveline torsional vibration, tyre belt
    dynamics) are omitted.

---

## Training Pipeline

### Phase 1: Physics-Informed Pre-Training

H_net and R_net are pre-trained against analytical vehicle dynamics (no TTC data
needed):

1. Generate rollouts using the analytical Pacejka + linear suspension model
2. Compute ground-truth $\dot{q}, \dot{p}$ from the analytical model
3. Train H_net to minimise $\|H_{\text{net}} - H_{\text{analytical}}\|^2$
4. Train R_net to minimise $\|R_{\text{net}} \nabla H - R_{\text{analytical}} \nabla H\|^2$

### Phase 2: TTC Data Fine-Tuning

After Phase 1, the PINN tire correction is trained against TTC Round 9 Calspan data.
The H_net is then fine-tuned end-to-end against tyre force residuals, allowing the
energy landscape to absorb unmodelled coupling effects.

### Phase 3: Online Adaptation (EKF)

The Differentiable EKF estimates 5 parameters online — tire grip scale $\lambda_\mu$,
optimal temperature $T_{\text{opt}}$, CG height $h_{\text{cg}}$, peak slip angle
$\alpha_{\text{peak}}$, and a catch-all scaling factor — closing the loop between the
model and reality without retraining the neural networks.

---

## References

1. van der Schaft, A. (2006). *Port-Hamiltonian systems: an introductory survey.* Proceedings of the International Congress of Mathematicians, Madrid.
2. Greydanus, S., Dzamba, M., & Yosinski, J. (2019). *Hamiltonian Neural Networks.* NeurIPS 2019.
3. Zhong, Y. D., et al. (2020). *Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control.* ICLR 2020.
4. Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration.* Springer.
