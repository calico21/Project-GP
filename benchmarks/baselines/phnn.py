# benchmarks/baselines/phnn.py
# Project-GP — Port-Hamiltonian Neural Network Baseline
# ═══════════════════════════════════════════════════════════════════════════════
"""
PHNN baseline: extends HNN with learned dissipation R(x) ≥ 0.

    dq/dt = +∂H/∂p
    dp/dt = -∂H/∂q - R(x) ∂H/∂p + B u

R is parameterised as R = LLᵀ to guarantee PSD.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class _HamiltonianNet(nn.Module):
    hidden: tuple[int, ...] = (128, 128, 64)

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array,
                 setup: jax.Array) -> jax.Array:
        h = jnp.concatenate([q, p, setup])
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"h{i}")(h))
        return nn.Dense(1, name="out")(h)[0]


class _DissipationNet(nn.Module):
    """Learns lower-triangular L such that R = L Lᵀ ≽ 0."""
    hidden: tuple[int, ...] = (64, 32)
    p_dim: int = 14

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array,
                 setup: jax.Array) -> jax.Array:
        h = jnp.concatenate([q, p, setup])
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"r{i}")(h))
        # Output lower triangle entries
        n_tri = self.p_dim * (self.p_dim + 1) // 2
        L_flat = nn.Dense(n_tri, name="L_out")(h) * 0.01
        # Build lower triangular matrix
        L = jnp.zeros((self.p_dim, self.p_dim))
        idx = jnp.tril_indices(self.p_dim)
        L = L.at[idx].set(L_flat)
        return L @ L.T  # R = L Lᵀ ≽ 0


class _InputMatrix(nn.Module):
    """Learned input matrix B mapping controls to generalised forces."""
    hidden: tuple[int, ...] = (32,)
    p_dim: int = 14
    u_dim: int = 6

    @nn.compact
    def __call__(self, setup: jax.Array) -> jax.Array:
        h = setup
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"b{i}")(h))
        return nn.Dense(self.p_dim * self.u_dim, name="B_out")(h).reshape(
            self.p_dim, self.u_dim)


class PHNN:
    """
    Port-Hamiltonian Neural Network baseline.

    Extends HNN with:
    - PSD dissipation matrix R(x) = L(x)L(x)ᵀ
    - Input matrix B(setup) for external forcing

    Structural guarantee: energy dissipation (dH/dt ≤ uᵀy for port output y).
    No guarantee on H ≥ 0, grounding, or potential properties.
    """
    name = "PHNN"

    def __init__(self, q_dim: int = 14, control_dim: int = 6,
                 setup_dim: int = 28, hidden: tuple[int, ...] = (128, 128, 64),
                 dt: float = 0.005):
        self.q_dim = q_dim
        self.state_dim = 2 * q_dim
        self.control_dim = control_dim
        self.setup_dim = setup_dim
        self.dt = dt
        self.H_net = _HamiltonianNet(hidden=hidden)
        self.R_net = _DissipationNet(p_dim=q_dim)
        self.B_net = _InputMatrix(p_dim=q_dim, u_dim=control_dim)
        self.params = None

    def init_params(self, rng: jax.Array) -> dict:
        k1, k2, k3 = jax.random.split(rng, 3)
        q0 = jnp.zeros(self.q_dim)
        p0 = jnp.zeros(self.q_dim)
        s0 = jnp.zeros(self.setup_dim)
        self.params = {
            "H": self.H_net.init(k1, q0, p0, s0)["params"],
            "R": self.R_net.init(k2, q0, p0, s0)["params"],
            "B": self.B_net.init(k3, s0)["params"],
        }
        return self.params

    def _dynamics(self, params: dict, q: jax.Array, p: jax.Array,
                  u: jax.Array, setup: jax.Array) -> tuple[jax.Array, jax.Array]:
        def H_fn(q_, p_):
            return self.H_net.apply({"params": params["H"]}, q_, p_, setup)

        dH_dq = jax.grad(H_fn, argnums=0)(q, p)
        dH_dp = jax.grad(H_fn, argnums=1)(q, p)

        R = self.R_net.apply({"params": params["R"]}, q, p, setup)
        B = self.B_net.apply({"params": params["B"]}, setup)

        dq = dH_dp
        dp = -dH_dq - R @ dH_dp + B @ u
        return dq, dp

    def predict_step(self, params: dict, x: jax.Array, u: jax.Array,
                     setup: jax.Array) -> jax.Array:
        q, p = x[:self.q_dim], x[self.q_dim:self.state_dim]
        dq, dp = self._dynamics(params, q, p, u, setup)
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
