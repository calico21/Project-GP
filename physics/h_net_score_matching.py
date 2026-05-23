# physics/h_net_score_matching.py
# Project-GP — Score Matching Trainer for Neural Port-Hamiltonian H_net
#
# Replaces the standard MSE + P4 penalty training in residual_fitting.py.
# Key insight: Denoising Score Matching trains the Hamiltonian gradient field
# ∇H directly without fighting the ICNN convexity constraint — FiLM sensitivity
# is dramatically improved because the score gradient is a local operation that
# doesn't require globally consistent energy levels across setups.

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from physics.h_net_icnn import PassiveHNet


# ─────────────────────────────────────────────────────────────────────────────
# §1  Denoising Score Matching Loss
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(0,))
def dsm_loss(
    h_net:      PassiveHNet,
    h_params:   dict,
    q_batch:    jax.Array,    # (B, 14)  — suspension DOF positions
    p_batch:    jax.Array,    # (B, 14)  — generalised momenta
    s_batch:    jax.Array,    # (B, 28)  — setup vectors
    sigma:      float = 0.02,
    key:        jax.Array = None,
) -> jax.Array:
    """
    Denoising Score Matching loss for H_net.

    Target: learn the score ∇_q H(q, p, s) without normalisation.
    The Port-Hamiltonian dynamics require ∂H/∂p = q̇ and ∂H/∂q = -ṗ,
    so the score ∇_q H must be physically meaningful — specifically, it
    must be the negative of the generalised force, derivable from actual
    trajectory data.

    DSM objective:
      ℒ = E_q,p,s,ε [ ‖∇_{q̃} H(q̃, p, s) + ε/σ²‖² ]
    where q̃ = q + ε, ε ~ N(0, σ²I)

    The optimal solution satisfies:
      ∇_{q̃} H(q̃, p, s) = (q̃ - q) / σ²  — Tweedie's formula
    This is exactly the denoising direction, connecting DSM to the
    Stein score function without requiring partition function estimation.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    B = q_batch.shape[0]
    noise = jax.random.normal(key, q_batch.shape) * sigma
    q_noisy = q_batch + noise

    # Compute ∇_{q̃} H at noisy positions — this is a VJP call
    def h_score(q_n, p, s):
        return jax.grad(
            lambda q: h_net.apply(h_params, q, p, s)
        )(q_n)

    # vmap over batch
    scores = jax.vmap(h_score)(q_noisy, p_batch, s_batch)   # (B, 14)

    # DSM target: -noise / σ² (the denoising direction)
    targets = -noise / (sigma ** 2)

    # Weighted L2 loss: down-weight suspension DOFs, up-weight momentum DOFs
    # This is because the tire thermal and compliance DOFs have weaker signal
    domain_weights = jnp.array([
        1.0, 1.0, 1.0,   # X, Y, Z chassis (low relevance)
        2.0, 2.0, 1.0,   # roll, pitch, yaw (medium relevance)
        5.0, 5.0, 5.0, 5.0,  # suspension heave FL,FR,RL,RR (HIGH — this is what we learn)
        0.5, 0.5, 0.5, 0.5,  # wheel rotation angles (kinematic)
    ])
    weighted_error = (scores - targets) ** 2 * domain_weights[None, :]
    return jnp.mean(weighted_error)


# ─────────────────────────────────────────────────────────────────────────────
# §2  FiLM Contrastive Score Loss (drives setup sensitivity)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(0,))
def film_contrastive_score_loss(
    h_net:    PassiveHNet,
    h_params: dict,
    q:        jax.Array,   # (B, 14)
    p:        jax.Array,   # (B, 14)
    s_a:      jax.Array,   # (B, 28)  — "positive" setup
    s_b:      jax.Array,   # (B, 28)  — "negative" setup
    tau:      float = 0.1, # temperature
) -> jax.Array:
    """
    FiLM contrastive loss on the SCORE FIELD, not the energy values.

    Key insight: the previous contrastive loss compared H(q,p,s_a) vs H(q,p,s_b).
    This was insufficient because ICNN convexity allows two setups to have
    identical energy levels but different gradient directions (curvature changes).

    This loss compares ∇_q H(q,p,s_a) vs ∇_q H(q,p,s_b) — it forces the
    FiLM layers to modulate the Hamiltonian GRADIENT FIELD across setups,
    which is directly the physical quantity that matters (Hamilton's equations
    are written in terms of H's gradient, not H itself).

    InfoNCE lower bound on mutual information I(setup; score):
      ℒ_NCE = -E[log exp(s_q_a · s_q_b / τ) / Σ_j exp(s_q_a · s_q_j / τ)]
    """
    def score(q_i, p_i, s_i):
        return jax.grad(lambda q: h_net.apply(h_params, q, p_i, s_i))(q_i)

    scores_a = jax.vmap(score)(q, p, s_a)   # (B, 14)
    scores_b = jax.vmap(score)(q, p, s_b)   # (B, 14)

    # Normalize scores onto unit sphere for cosine similarity
    s_a_norm = scores_a / (jnp.linalg.norm(scores_a, axis=-1, keepdims=True) + 1e-8)
    s_b_norm = scores_b / (jnp.linalg.norm(scores_b, axis=-1, keepdims=True) + 1e-8)

    # InfoNCE: diagonal elements are positives (same q, different valid setups)
    # We WANT high cosine similarity between scores at physically similar setups
    # and LOW similarity between scores at very different setups (k_f differs by 2×)
    logits = s_a_norm @ s_b_norm.T / tau   # (B, B)
    labels = jnp.arange(logits.shape[0])
    loss   = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Hamilton's Equations Consistency Loss (physics-informed)
# ─────────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(0,))
def hamiltonian_consistency_loss(
    h_net:    PassiveHNet,
    h_params: dict,
    q:        jax.Array,   # (B, 14)
    p:        jax.Array,   # (B, 14)
    q_dot:    jax.Array,   # (B, 14) — from telemetry or physics rollout
    p_dot:    jax.Array,   # (B, 14) — from telemetry or physics rollout
    s:        jax.Array,   # (B, 28)
    M_inv:    jax.Array,   # (14, 14) — inverse mass matrix (diagonal approx ok)
) -> jax.Array:
    """
    Enforce Hamilton's equations directly:
      ∂H/∂p = q̇   (structure equation)
      ∂H/∂q = -ṗ  (force equation for conservative part)

    This is the ground-truth physics constraint that MSE training on target_H
    cannot express: we need the GRADIENT of H to match observed dynamics,
    not the value of H to match some synthetic target.

    Requires trajectory data (q, p, q̇, ṗ) — obtainable from:
      1. Physics rollouts (always available, but biased by current model)
      2. Telemetry + numerical differentiation (unbiased, but noisy)
      3. Hybrid: physics rollout + domain randomization

    The consistency loss is complementary to DSM:
      ℒ_total = λ_dsm · ℒ_dsm + λ_film · ℒ_film + λ_hcons · ℒ_hcons
    """
    def h_grads(q_i, p_i, s_i):
        h_val, grads = jax.value_and_grad(
            lambda qp: h_net.apply(h_params, qp[:14], qp[14:], s_i),
            argnums=0,
        )(jnp.concatenate([q_i, p_i]))
        return grads[:14], grads[14:]   # ∂H/∂q, ∂H/∂p

    dH_dq, dH_dp = jax.vmap(h_grads)(q, p, s)

    # ∂H/∂p = q̇: the momentum gradient gives the velocity
    structure_residual = dH_dp - q_dot

    # ∂H/∂q = -ṗ + R(q,p)·∂H/∂p: force equation (R accounts for dissipation)
    # For the conservative H_net alone: ∂H/∂q ≈ -ṗ (dissipation from R_net)
    force_residual = dH_dq + p_dot

    # Normalize by typical gradient magnitudes to balance terms
    structure_loss = jnp.mean(structure_residual ** 2) / (jnp.var(q_dot) + 1e-6)
    force_loss     = jnp.mean(force_residual ** 2) / (jnp.var(p_dot) + 1e-6)

    return structure_loss + force_loss


# ─────────────────────────────────────────────────────────────────────────────
# §4  Combined Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_h_net_score_matching(
    q_data:   jax.Array,   # (N, 14)
    p_data:   jax.Array,   # (N, 14)
    s_data:   jax.Array,   # (N, 28)  setup vectors
    q_dot_data: jax.Array, # (N, 14)  optional, from rollout
    p_dot_data: jax.Array, # (N, 14)  optional, from rollout
    n_epochs:   int = 2000,
    batch_size: int = 512,
    lr:         float = 3e-4,
    sigma_dsm:  float = 0.015,   # noise scale for DSM — ≈ 1% of typical suspension travel
    w_dsm:      float = 1.0,
    w_film:     float = 3.0,     # higher weight: FiLM sensitivity was too low
    w_hcons:    float = 5.0,
    tau_nce:    float = 0.07,    # InfoNCE temperature (lower = harder negatives)
) -> dict:
    """
    Complete score-matching training pipeline for H_net.

    Expected improvement over MSE:
      - FiLM sensitivity std(H|varied_setup): from 38 J/m² → >500 J/m²
      - Passive energy injection: from 237 mJ → <50 mJ
      - Training loss interpretability: ℒ_dsm is physically meaningful
        (score field accuracy), not an energy regression task
    """
    from physics.h_net_icnn import PassiveHNet
    import optax

    h_net = PassiveHNet(q_dim=14, p_dim=14, setup_dim=28)
    key = jax.random.PRNGKey(42)
    h_params = h_net.init(key, q_data[0], p_data[0], s_data[0])

    # Cosine annealing with warm restarts
    schedule = optax.cosine_decay_schedule(lr, n_epochs, alpha=0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(2.0),
        optax.adamw(schedule, weight_decay=1e-5),
    )
    opt_state = optimizer.init(h_params)

    N = q_data.shape[0]

    @jax.jit
    def step_fn(params, opt_state, key, q_b, p_b, s_b, qd_b, pd_b):
        k1, k2, k3 = jax.random.split(key, 3)

        # Contrastive pairs: shuffle setup assignments to create negatives
        s_b_neg = jax.random.permutation(k2, s_b, axis=0)

        def total_loss(params_):
            l_dsm  = w_dsm  * dsm_loss(h_net, params_, q_b, p_b, s_b, sigma=sigma_dsm, key=k1)
            l_film = w_film * film_contrastive_score_loss(h_net, params_, q_b, p_b, s_b, s_b_neg, tau=tau_nce)
            l_hc   = w_hcons * hamiltonian_consistency_loss(h_net, params_, q_b, p_b, qd_b, pd_b, s_b, jnp.eye(14))
            return l_dsm + l_film + l_hc, (l_dsm, l_film, l_hc)

        (loss, (l_dsm, l_film, l_hc)), grads = jax.value_and_grad(
            total_loss, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss, l_dsm, l_film, l_hc

    rng = jax.random.PRNGKey(0)
    for epoch in range(n_epochs):
        rng, k_perm, k_dsm = jax.random.split(rng, 3)
        perm = jax.random.permutation(k_perm, N)[:batch_size]
        q_b, p_b, s_b = q_data[perm], p_data[perm], s_data[perm]
        qd_b, pd_b = q_dot_data[perm], p_dot_data[perm]

        h_params, opt_state, loss, l_dsm, l_film, l_hc = step_fn(
            h_params, opt_state, k_dsm, q_b, p_b, s_b, qd_b, pd_b
        )

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:5d} | total={float(loss):.4f} "
                  f"| DSM={float(l_dsm):.4f} "
                  f"| FiLM-NCE={float(l_film):.4f} "
                  f"| Hcons={float(l_hc):.4f}")

    return h_params