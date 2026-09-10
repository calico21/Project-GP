# benchmarks/baselines/neural_ode.py
# Project-GP — Neural ODE Baseline
# ═══════════════════════════════════════════════════════════════════════════════
"""
Standard Neural ODE: dx/dt = f_θ(x, u, s) with MLP dynamics.
No physical structure imposed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class _DynamicsMLP(nn.Module):
    """MLP that maps (x, u, setup) → dx/dt."""
    hidden: tuple[int, ...] = (128, 128, 64)
    state_dim: int = 28  # q + v mechanical states

    @nn.compact
    def __call__(self, x: jax.Array, u: jax.Array, setup: jax.Array) -> jax.Array:
        h = jnp.concatenate([x, u, setup])
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"d{i}")(h))
        return nn.Dense(self.state_dim, name="out")(h)


class NeuralODE:
    """
    Neural ODE baseline for benchmark comparison.

    Identical training budget, data splits, and evaluation protocol as all
    other baselines. No physics inductive bias — pure black-box.
    """
    name = "NeuralODE"

    def __init__(self, state_dim: int = 28, control_dim: int = 6,
                 setup_dim: int = 28, hidden: tuple[int, ...] = (128, 128, 64),
                 dt: float = 0.005):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.setup_dim = setup_dim
        self.dt = dt
        self.model = _DynamicsMLP(hidden=hidden, state_dim=state_dim)
        self.params = None

    def init_params(self, rng: jax.Array) -> dict:
        x0 = jnp.zeros(self.state_dim)
        u0 = jnp.zeros(self.control_dim)
        s0 = jnp.zeros(self.setup_dim)
        self.params = self.model.init(rng, x0, u0, s0)["params"]
        return self.params

    def predict_step(self, params: dict, x: jax.Array, u: jax.Array,
                     setup: jax.Array) -> jax.Array:
        """One-step prediction: x_{k+1} = x_k + dt * f_θ(x_k, u_k, s)."""
        dx = self.model.apply({"params": params}, x, u, setup)
        return x + self.dt * dx

    def predict_trajectory(self, params: dict, x0: jax.Array,
                           controls: jax.Array, setup: jax.Array) -> jax.Array:
        """Roll out N steps. controls: (N, control_dim)."""
        def step_fn(x, u):
            x_next = self.predict_step(params, x, u, setup)
            return x_next, x_next
        _, trajectory = jax.lax.scan(step_fn, x0, controls)
        return trajectory

    def loss_fn(self, params: dict, x_batch: jax.Array, u_batch: jax.Array,
                x_next_batch: jax.Array, setup_batch: jax.Array) -> jax.Array:
        """MSE one-step prediction loss."""
        def single_loss(x, u, x_next, s):
            x_pred = self.predict_step(params, x, u, s)
            return jnp.mean((x_pred - x_next) ** 2)
        return jnp.mean(jax.vmap(single_loss)(x_batch, u_batch, x_next_batch, setup_batch))

    def train(self, rng: jax.Array, dataset: dict, n_epochs: int = 500,
              lr: float = 3e-4, batch_size: int = 256) -> dict:
        """Train with Adam on one-step MSE loss."""
        if self.params is None:
            rng, sub = jax.random.split(rng)
            self.init_params(sub)

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(self.params)

        x_train = dataset["x"]
        u_train = dataset["u"]
        x_next_train = dataset["x_next"]
        setup_train = dataset["setup"]
        n = x_train.shape[0]

        @jax.jit
        def train_step(params, opt_state, x_b, u_b, xn_b, s_b):
            loss, grads = jax.value_and_grad(self.loss_fn)(params, x_b, u_b, xn_b, s_b)
            updates, opt_state_new = optimizer.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, opt_state_new, loss

        for epoch in range(n_epochs):
            rng, sub = jax.random.split(rng)
            perm = jax.random.permutation(sub, n)
            for start in range(0, n - batch_size + 1, batch_size):
                idx = perm[start:start + batch_size]
                self.params, opt_state, loss = train_step(
                    self.params, opt_state,
                    x_train[idx], u_train[idx], x_next_train[idx], setup_train[idx]
                )

        return {"params": self.params, "final_loss": float(loss)}
