# benchmarks/baselines/pinn.py
# Project-GP — Physics-Informed Neural Network Baseline
# ═══════════════════════════════════════════════════════════════════════════════
"""
PINN baseline: MLP predicts dynamics, with an auxiliary physics residual loss
penalizing deviation from Hamilton's equations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class _PINNMLP(nn.Module):
    """MLP that maps (q, p, u, setup) → (dq/dt, dp/dt)."""
    hidden: tuple[int, ...] = (128, 128, 64)
    q_dim: int = 14

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array,
                 u: jax.Array, setup: jax.Array) -> tuple[jax.Array, jax.Array]:
        h = jnp.concatenate([q, p, u, setup])
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"d{i}")(h))
        out = nn.Dense(2 * self.q_dim, name="out")(h)
        return out[:self.q_dim], out[self.q_dim:]


class _HamiltonianMLP(nn.Module):
    """Auxiliary Hamiltonian estimator for physics residual."""
    hidden: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array,
                 setup: jax.Array) -> jax.Array:
        h = jnp.concatenate([q, p, setup])
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"h{i}")(h))
        return nn.Dense(1, name="hout")(h)[0]


class PINN:
    """
    Physics-Informed Neural Network baseline.

    Loss = MSE_data + λ_phys * MSE_physics_residual

    The physics residual penalizes deviation from Hamilton's equations:
        dq/dt = +∂H/∂p
        dp/dt = -∂H/∂q + F_ext
    """
    name = "PINN"

    def __init__(self, q_dim: int = 14, control_dim: int = 6,
                 setup_dim: int = 28, hidden: tuple[int, ...] = (128, 128, 64),
                 dt: float = 0.005, lambda_phys: float = 0.1):
        self.q_dim = q_dim
        self.state_dim = 2 * q_dim
        self.control_dim = control_dim
        self.setup_dim = setup_dim
        self.dt = dt
        self.lambda_phys = lambda_phys
        self.dyn_model = _PINNMLP(hidden=hidden, q_dim=q_dim)
        self.ham_model = _HamiltonianMLP()
        self.params = None

    def init_params(self, rng: jax.Array) -> dict:
        k1, k2 = jax.random.split(rng)
        q0 = jnp.zeros(self.q_dim)
        p0 = jnp.zeros(self.q_dim)
        u0 = jnp.zeros(self.control_dim)
        s0 = jnp.zeros(self.setup_dim)
        dyn_params = self.dyn_model.init(k1, q0, p0, u0, s0)["params"]
        ham_params = self.ham_model.init(k2, q0, p0, s0)["params"]
        self.params = {"dyn": dyn_params, "ham": ham_params}
        return self.params

    def predict_step(self, params: dict, x: jax.Array, u: jax.Array,
                     setup: jax.Array) -> jax.Array:
        """One-step: x_{k+1} = x_k + dt * f_θ(x_k)."""
        q, p = x[:self.q_dim], x[self.q_dim:self.state_dim]
        dq, dp = self.dyn_model.apply({"params": params["dyn"]}, q, p, u, setup)
        q_new = q + self.dt * dq
        p_new = p + self.dt * dp
        return jnp.concatenate([q_new, p_new])

    def predict_trajectory(self, params: dict, x0: jax.Array,
                           controls: jax.Array, setup: jax.Array) -> jax.Array:
        def step_fn(x, u):
            x_next = self.predict_step(params, x, u, setup)
            return x_next, x_next
        _, trajectory = jax.lax.scan(step_fn, x0, controls)
        return trajectory

    def loss_fn(self, params: dict, x_batch: jax.Array, u_batch: jax.Array,
                x_next_batch: jax.Array, setup_batch: jax.Array) -> jax.Array:
        """Data MSE + physics residual."""
        def single(x, u, x_next, s):
            # Data loss
            x_pred = self.predict_step(params, x, u, s)
            data_loss = jnp.mean((x_pred - x_next) ** 2)

            # Physics residual: dq_pred should ≈ ∂H/∂p, dp_pred should ≈ -∂H/∂q
            q, p = x[:self.q_dim], x[self.q_dim:self.state_dim]
            dq_pred, dp_pred = self.dyn_model.apply({"params": params["dyn"]}, q, p, u, s)

            dH_dp = jax.grad(lambda p_: self.ham_model.apply(
                {"params": params["ham"]}, q, p_, s))(p)
            dH_dq = jax.grad(lambda q_: self.ham_model.apply(
                {"params": params["ham"]}, q_, p, s))(q)

            phys_loss = jnp.mean((dq_pred - dH_dp) ** 2) + jnp.mean((dp_pred + dH_dq) ** 2)
            return data_loss + self.lambda_phys * phys_loss

        return jnp.mean(jax.vmap(single)(x_batch, u_batch, x_next_batch, setup_batch))

    def train(self, rng: jax.Array, dataset: dict, n_epochs: int = 500,
              lr: float = 3e-4, batch_size: int = 256) -> dict:
        if self.params is None:
            rng, sub = jax.random.split(rng)
            self.init_params(sub)

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(self.params)
        n = dataset["x"].shape[0]

        @jax.jit
        def train_step(params, opt_state, x_b, u_b, xn_b, s_b):
            loss, grads = jax.value_and_grad(self.loss_fn)(params, x_b, u_b, xn_b, s_b)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, opt_state_new, loss

        loss = jnp.inf
        for epoch in range(n_epochs):
            rng, sub = jax.random.split(rng)
            perm = jax.random.permutation(sub, n)
            for start in range(0, n - batch_size + 1, batch_size):
                idx = perm[start:start + batch_size]
                self.params, opt_state, loss = train_step(
                    self.params, opt_state,
                    dataset["x"][idx], dataset["u"][idx],
                    dataset["x_next"][idx], dataset["setup"][idx],
                )
        return {"params": self.params, "final_loss": float(loss)}
