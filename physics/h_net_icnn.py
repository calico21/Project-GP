# physics/h_net_icnn.py
# Project-GP — Structurally Passive Neural Hamiltonian Residual
# ═══════════════════════════════════════════════════════════════════════════════
#
#     H_net(q, p, setup) = K(p) · ψ(q, setup) + V(q, setup)
#
# Structural properties guaranteed for ANY weight values:
#   P1  H_net ≥ 0
#   P2  H_net(q_eq, 0, setup) = 0
#   P3  ∇_p H_net(q, 0, setup) = 0
#   P4  pᵀ ∇_p H_net ≥ 0
#
# KEY DESIGN DECISION — why a submodule, not a nested function:
#   K(p) = ICNN(p²) - ICNN(0)  requires calling the same ICNN twice with
#   identical weights. Flax's nn.compact stores params by string name — calling
#   a nested function twice inside one __call__ tries to register e.g.
#   "k_W0_raw" twice → NameInUseError.
#
#   Fix: _KineticICNN is a standalone nn.Module. KineticNet instantiates it
#   once with name="core". Flax binds params on the first call `core(p_sq)`
#   and REUSES the same param dict on the second call `core(zeros)` because
#   the module name is the same. The subtraction K(p²) - K(0) is then
#   computed with provably identical weights.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


# ─────────────────────────────────────────────────────────────────────────────
# Equilibrium — matches residual_fitting.py _Z_EQ
# ─────────────────────────────────────────────────────────────────────────────

_Z_EQ_DEFAULT: jax.Array = jnp.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0128, 0.0128, 0.0142, 0.0142,
     0.0, 0.0, 0.0, 0.0]
)


# ─────────────────────────────────────────────────────────────────────────────
# §1  _KineticICNN — raw all-positive-weight ICNN on x = p²
# ─────────────────────────────────────────────────────────────────────────────

class _KineticICNN(nn.Module):
    """
    All-non-negative-weight ICNN. Monotone non-decreasing in each input
    component when inputs are ≥ 0 (which p² always is).

    Extracted as a standalone module so KineticNet can call it twice
    (at p² and at 0) with provably identical parameters — Flax reuses
    the param dict for any submodule invoked under the same name.
    """
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Input layer — all weights non-negative via softplus
        W0_raw = self.param("W0_raw", nn.initializers.normal(0.01),
                            (x.shape[-1], self.hidden[0]))
        b0 = self.param("b0", nn.initializers.zeros, (self.hidden[0],))
        z = nn.softplus(x @ nn.softplus(W0_raw) + b0)

        for i, h in enumerate(self.hidden[1:], start=1):
            Wz_raw = self.param(f"Wz{i}_raw", nn.initializers.normal(0.01),
                                (z.shape[-1], h))
            Wx_raw = self.param(f"Wx{i}_raw", nn.initializers.normal(0.01),
                                (x.shape[-1], h))
            b = self.param(f"b{i}", nn.initializers.zeros, (h,))
            z = nn.softplus(z @ nn.softplus(Wz_raw) + x @ nn.softplus(Wx_raw) + b)

        w_raw = self.param("w_out_raw", nn.initializers.normal(0.01), (z.shape[-1],))
        return jnp.sum(z * nn.softplus(w_raw))


# ─────────────────────────────────────────────────────────────────────────────
# §2  KineticNet — K(p) with P1–P4 via squared input + submodule reuse
# ─────────────────────────────────────────────────────────────────────────────

class KineticNet(nn.Module):
    """
    K(p) = ICNN(p²) - ICNN(0)

    Properties (all algebraic, weight-independent):
      P1  K(p) ≥ 0         ICNN is non-neg from 0, so ICNN(p²) ≥ ICNN(0)
      P2  K(0) = 0         by construction
      P3  ∂K/∂p|_{p=0} = 0  chain rule: ∂K/∂pᵢ = 2pᵢ · ∂ICNN/∂(pᵢ²) = 0 at p=0
      P4  p·∂K/∂p ≥ 0      = 2Σ pᵢ² · ∂ICNN/∂(pᵢ²) ≥ 0 (both factors ≥ 0)
    """
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, p: jax.Array) -> jax.Array:
        core = _KineticICNN(self.hidden, name="core")
        # Flax binds params on first call; second call reuses same param dict.
        return core(p * p) - core(jnp.zeros_like(p * p))


# ─────────────────────────────────────────────────────────────────────────────
# §3  PsiGate — ψ(q, setup) ≥ 0
# ─────────────────────────────────────────────────────────────────────────────

class PsiGate(nn.Module):
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, q: jax.Array, setup: jax.Array) -> jax.Array:
        h = jnp.concatenate([q, setup])
        for i, w in enumerate(self.hidden):
            h = nn.swish(nn.Dense(w, name=f"dense_{i}")(h))
        return nn.softplus(nn.Dense(1, name="out")(h)[0])


# ─────────────────────────────────────────────────────────────────────────────
# §4  _PotentialICNN — ICNN core for V(q, setup)
# ─────────────────────────────────────────────────────────────────────────────

class _PotentialICNN(nn.Module):
    """
    Standard ICNN convex in q (hidden-to-hidden weights non-negative).
    Input weights unconstrained because q can be negative.
    """
    hidden: tuple[int, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        z = nn.softplus(nn.Dense(self.hidden[0], name="b0_x")(x))

        for i, h in enumerate(self.hidden[1:], start=1):
            Wz_raw = self.param(f"Wz{i}_raw", nn.initializers.normal(0.01),
                                (z.shape[-1], h))
            z = nn.softplus(
                z @ nn.softplus(Wz_raw) + nn.Dense(h, name=f"b{i}_x")(x)
            )

        w_raw = self.param("w_out_raw", nn.initializers.normal(0.01), (z.shape[-1],))
        return jnp.sum(z * nn.softplus(w_raw))


# ─────────────────────────────────────────────────────────────────────────────
# §5  PotentialNet — V(q, setup), grounded at q_eq
# ─────────────────────────────────────────────────────────────────────────────

class PotentialNet(nn.Module):
    """
    V(q, setup) with V(q_eq, setup) = 0 for all setup.

    FiLM conditioning: affine map on q preserves ICNN convexity in q.
    Grounding: V = ICNN_V(q_film) - ICNN_V(q_film_ref)
    where q_film_ref = β(setup) is q_film evaluated at q = q_eq.
    Same submodule-reuse trick as KineticNet.
    """
    hidden: tuple[int, ...]
    q_dim: int
    setup_dim: int

    @nn.compact
    def __call__(self, q: jax.Array, setup: jax.Array) -> jax.Array:
        q_centered = q - _Z_EQ_DEFAULT

        film  = nn.Dense(2 * self.q_dim, name="film")(setup)
        gamma = 1.0 + 0.1 * jnp.tanh(film[: self.q_dim])
        beta  = 0.05 * jnp.tanh(film[self.q_dim:])

        q_film     = gamma * q_centered + beta
        q_film_ref = beta                              # q_film at q = q_eq

        core = _PotentialICNN(self.hidden, name="core")
        return core(q_film) - core(q_film_ref)


# ─────────────────────────────────────────────────────────────────────────────
# §6  PassiveHNet — full residual H_net(q, p, setup)
# ─────────────────────────────────────────────────────────────────────────────

class PassiveHNet(nn.Module):
    """
    H_net(q, p, setup) = K(p) · ψ(q, setup) + V(q, setup)

    Drop-in for NeuralEnergyLandscape:
        model.apply({"params": params}, q, p, setup)  →  scalar [J]

    Returns H_neural only. Callers that need H_total must add T_prior + V_struct.
    See vehicle_dynamics.py _compute_derivatives for the correct wrapper.
    """
    q_dim:      int          = 14
    p_dim:      int          = 14
    setup_dim:  int          = 28
    k_hidden:   tuple[int, ...] = (64, 64, 32)
    v_hidden:   tuple[int, ...] = (64, 64, 32)
    psi_hidden: tuple[int, ...] = (64, 32)
    h_cap:      float        = 50_000.0

    @nn.compact
    def __call__(self, q: jax.Array, p: jax.Array, setup: jax.Array) -> jax.Array:
        K   = KineticNet(self.k_hidden,                             name="K_net")(p)
        psi = PsiGate(self.psi_hidden,                              name="psi_gate")(q, setup)
        V   = PotentialNet(self.v_hidden, self.q_dim, self.setup_dim, name="V_net")(q, setup)

        H_raw = K * psi + V
        return self.h_cap * jnp.tanh(H_raw / self.h_cap)


# ─────────────────────────────────────────────────────────────────────────────
# §7  Init helper
# ─────────────────────────────────────────────────────────────────────────────

def init_passive_hnet(
    rng: jax.Array,
    q_dim: int = 14, p_dim: int = 14, setup_dim: int = 28,
    **kwargs,
) -> tuple[PassiveHNet, dict]:
    model  = PassiveHNet(q_dim=q_dim, p_dim=p_dim, setup_dim=setup_dim, **kwargs)
    params = model.init(
        rng,
        jnp.zeros(q_dim),
        jnp.zeros(p_dim),
        jnp.zeros(setup_dim),
    )["params"]
    return model, params