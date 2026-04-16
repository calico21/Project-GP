# Control Barrier Function Safety Filter

## The Safety Promise

The CBF guarantees that the vehicle state remains within a **safe invariant set**
$\mathcal{C} = \{x : h(x) \geq 0\}$, where $h$ is the barrier function. If the
torque allocator produces a command that would violate this set, the CBF filter
modifies the command minimally to maintain safety.

This is not a soft constraint or a penalty — it is a **hard guarantee** that holds
for all time, provided the barrier function is correctly constructed.

---

## Barrier Function Design

### Sideslip Barrier

The vehicle sideslip angle $\beta = \arctan(v_y / v_x)$ must remain bounded for
controllability:

$$
h_\beta(x) = \beta_{\max}^2 - \beta^2 = \beta_{\max}^2 - \left(\frac{v_y}{v_x}\right)^2
$$

When $h_\beta > 0$: safe (sideslip within limits). When $h_\beta = 0$: at the boundary.
The CBF prevents $h_\beta$ from becoming negative.

### Yaw Rate Barrier

The yaw rate must respect the friction-limited maximum:

$$
h_{\dot\psi}(x) = \dot\psi_{\max}^2(\mu, v_x) - \dot\psi^2
$$

where $\dot\psi_{\max} = \mu g / v_x$ is the steady-state yaw rate limit at the
current speed and friction estimate.

### Combined Barrier

$$
h(x) = \min\big(h_\beta(x),\; h_{\dot\psi}(x)\big)
$$

The `min` is smoothed via the log-sum-exp approximation:
$\min(a, b) \approx -\frac{1}{\tau}\log(e^{-\tau a} + e^{-\tau b})$ with $\tau = 10$.

---

## Input-Delay Compensation

### The Problem

Physical actuators have delay: a torque command issued at time $t$ arrives at the
motor at time $t + d\Delta t$, where $d$ is the delay in timesteps. During this delay,
the vehicle state evolves uncontrolled. A naive CBF that enforces $h(x_t) \geq 0$
may still violate the barrier because $x_{t+d}$ has already left the safe set by the
time the command takes effect.

### The Solution: Predictive DCBF

The discrete-time CBF with delay compensation enforces:

$$
\boxed{h\big(\hat{x}_{k+d}\big) \geq (1 - \alpha)^d \cdot h(x_k)}
$$

where:

- $\hat{x}_{k+d}$ is the **predicted** state at the delay horizon, computed by
  forward-simulating $d$ steps with the current state and the proposed command
- $(1 - \alpha)^d$ is the decay factor, ensuring the barrier decays gracefully
  rather than requiring instantaneous correction
- $\alpha \in (0, 1)$ is the CBF class-$\mathcal{K}$ function parameter

### Robust Extension (GP Uncertainty)

When the tyre model has quantified uncertainty (via the Sparse GP), the barrier
is tightened:

$$
h_{\text{robust}}(x) = h(x) - \kappa_{\text{safe}} \cdot \sigma_{\text{GP}}(x)
$$

where $\sigma_{\text{GP}}(x)$ is the GP posterior standard deviation at the current
operating point. When the GP is uncertain (high $\sigma$), the safe set shrinks —
the controller becomes more conservative. When the GP is confident (low $\sigma$),
the full envelope is available.

This is the connection between **learning** (the GP) and **safety** (the CBF): the
GP's uncertainty directly modulates the safety margin.

---

## Implementation

The CBF filter solves a 3-iteration QP:

$$
\min_{T} \|T - T_{\text{alloc}}\|^2 \quad \text{s.t.} \quad h(\hat{x}_{k+d}(T)) \geq (1-\alpha)^d h(x_k)
$$

This is implemented as 3 projected-gradient steps via `jax.lax.scan`, making the
filter differentiable and JIT-compilable. The total filter latency is ~0.3–0.5 ms
on ARM Cortex-A72.

```python
def cbf_safety_filter(T_alloc, T_prev, vx, vy, wz, Fz, Fy_total,
                      mu_est, omega_wheel, T_min, T_max, gp_sigma, geo, cbf):
    # 1. Compute current barrier value
    h_now = barrier_function(vx, vy, wz, mu_est, cbf)

    # 2. Predict state at delay horizon
    x_pred = predict_state(vx, vy, wz, T_alloc, Fz, geo, cbf.delay_steps)

    # 3. Evaluate barrier at predicted state (with GP robustness)
    h_pred = barrier_function(*x_pred, mu_est, cbf) - cbf.kappa_safe * gp_sigma

    # 4. CBF condition: h_pred >= (1-alpha)^d * h_now
    margin = h_pred - (1 - cbf.alpha) ** cbf.delay_steps * h_now

    # 5. If violated, project T_alloc toward safety
    T_safe = cbf_project(T_alloc, margin, ...)
    return T_safe
```

---

## Verification

**Test 12** in the sanity check suite verifies:

1. The barrier $h(x) \geq 0$ is maintained across a simulated emergency manoeuvre
2. The CBF intervention is non-zero when unsafe commands are injected
3. The filtered torques remain within motor limits

The CBF is the final safety layer before commands reach the motors. It cannot be
bypassed by any upstream controller.
