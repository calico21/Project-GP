# benchmarks/baselines/hnn.py
# Project-GP — Hamiltonian Neural Network Baseline
# ═══════════════════════════════════════════════════════════════════════════════
"""
HNN baseline: learn H(q, p) as unconstrained MLP, derive dynamics via
symplectic gradient structure: dq/dt = +∂H/∂p, dp/dt = -∂H/∂q.

No dissipation, no external forces in the structure — purely conservative.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class _HamiltonianNet(nn.Module):
    """Unconstrained MLP mapping (q, p, setup) → scalar H."""
    hidden: tuple[int, ...] = (128, 128, 64)

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array,
                 setup: jax.Array) -> jax.Array:
        h = jnp.concatenate([q, p, setup])
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"h{i}")(h))
        return nn.Dense(1, name="out")(h)[0]


class HNN:
    """
    Hamiltonian Neural Network baseline.

    Dynamics derived from learned Hamiltonian:
        dq/dt = +∂H/∂p
        dp/dt = -∂H/∂q

    Structural inductive bias: energy conservation (no dissipation).
    No guarantee on H ≥ 0, grounding, or passivity.
    """
    name = "HNN"

    def __init__(self, q_dim: int = 14, control_dim: int = 6,
                 setup_dim: int = 28, hidden: tuple[int, ...] = (128, 128, 64),
                 dt: float = 0.005):
        self.q_dim = q_dim
        self.state_dim = 2 * q_dim
        self.control_dim = control_dim
        self.setup_dim = setup_dim
        self.dt = dt
        self.model = _HamiltonianNet(hidden=hidden)
        self.params = None

    def init_params(self, rng: jax.Array) -> dict:
        q0 = jnp.zeros(self.q_dim)
        p0 = jnp.zeros(self.q_dim)
        s0 = jnp.zeros(self.setup_dim)
        self.params = self.model.init(rng, q0, p0, s0)["params"]
        return self.params

    def _symplectic_dynamics(self, params: dict, q: jax.Array, p: jax.Array,
                             setup: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute dq/dt, dp/dt from ∂H/∂p, -∂H/∂q."""
        def H_fn(q_, p_):
            return self.model.apply({"params": params}, q_, p_, setup)

        dH_dq = jax.grad(H_fn, argnums=0)(q, p)
        dH_dp = jax.grad(H_fn, argnums=1)(q, p)
        return dH_dp, -dH_dq

    def predict_step(self, params: dict, x: jax.Array, u: jax.Array,
                     setup: jax.Array) -> jax.Array:
        q, p = x[:self.q_dim], x[self.q_dim:self.state_dim]
        dq, dp = self._symplectic_dynamics(params, q, p, setup)
        return jnp.concatenate([q + self.dt * dq, p + self.dt * dp])

    def predict_trajectory(self, params: dict, x0: jax.Array,
                           controls: jax.Array, setup: jax.Array) -> jax.Array:
        def step_fn(x, u):
            x_next = self.predict_step(params, x, u, setup)
            return x_next, x_next
        _, trajectory = jax.lax.scan(step_fn, x0, controls)
        return trajectory

    def loss_fn(self, params: dict, x_batch: jax.Array, u_batch: jax.Array,
                x_next_batch: jax.Array, setup_batch: jax.Array) -> jax.Array:
        def single(x, u, x_next, s):
            x_pred = self.predict_step(params, x, u, s)
            return jnp.mean((x_pred - x_next) ** 2)
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
