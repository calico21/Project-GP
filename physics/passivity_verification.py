# physics/passivity_verification.py
# Project-GP — Algebraic Passivity Verification for PassiveHNet
# ═══════════════════════════════════════════════════════════════════════════════
#
# Run standalone or call run_verification() from sanity_checks.py.
# All jit-compiled checkers take params as a traced arg; model_fn is a Python
# closure (not traced), avoiding the "function as abstract array" error.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from physics.h_net_icnn import PassiveHNet, init_passive_hnet, _Z_EQ_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# §1  Operational envelope sampler
# ─────────────────────────────────────────────────────────────────────────────

def sample_envelope(rng: jax.Array, n: int,
                    q_dim: int = 14, p_dim: int = 14, setup_dim: int = 28,
                    ) -> tuple[jax.Array, jax.Array, jax.Array]:
    k_q, k_p, k_s = jax.random.split(rng, 3)
    q = 0.15 * jax.random.normal(k_q, (n, q_dim))
    q = q.at[:, 6:10].add(_Z_EQ_DEFAULT[6:10])
    p = 50.0 * jax.random.normal(k_p, (n, p_dim))
    setup = jax.random.uniform(k_s, (n, setup_dim))
    return q, p, setup


# ─────────────────────────────────────────────────────────────────────────────
# §2  Property checkers — model_fn is a Python closure, not a traced arg
# ─────────────────────────────────────────────────────────────────────────────

def make_checkers(model: PassiveHNet):
    """
    Returns four checker functions closed over `model`. Each accepts `params`
    as the only JAX-traced argument so jit compiles cleanly.
    """

    # ── P1: H_net ≥ 0 ────────────────────────────────────────────────────────
    @jax.jit
    def check_P1(params, q, p, setup):
        vals = jax.vmap(
            lambda qi, pi, si: model.apply({"params": params}, qi, pi, si)
        )(q, p, setup)
        return jnp.min(vals), jnp.max(vals)

    # ── P2: H_net(q_eq, 0, setup) = 0 ────────────────────────────────────────
    @jax.jit
    def check_P2(params, setup):
        n = setup.shape[0]
        q_eq = jnp.broadcast_to(_Z_EQ_DEFAULT, (n, _Z_EQ_DEFAULT.shape[0]))
        p0 = jnp.zeros(_Z_EQ_DEFAULT.shape[0])
        vals = jax.vmap(
            lambda qi, si: model.apply({"params": params}, qi, p0, si)
        )(q_eq, setup)
        return jnp.max(jnp.abs(vals))

    # ── P3: ∇_p H_net(q, 0, setup) = 0 ──────────────────────────────────────
    @jax.jit
    def check_P3(params, q, setup):
        p0 = jnp.zeros(q.shape[1])
        def grad_at_rest(qi, si):
            return jax.grad(
                lambda pi: model.apply({"params": params}, qi, pi, si)
            )(p0)
        grads = jax.vmap(grad_at_rest)(q, setup)
        return jnp.max(jnp.linalg.norm(grads, axis=-1))

    # ── P4: pᵀ ∇_p H_net ≥ 0 ─────────────────────────────────────────────────
    @jax.jit
    def check_P4(params, q, p, setup):
        def inner_product(qi, pi, si):
            g = jax.grad(
                lambda pi_: model.apply({"params": params}, qi, pi_, si)
            )(pi)
            return jnp.dot(pi, g)
        dots = jax.vmap(inner_product)(q, p, setup)
        return jnp.min(dots), jnp.mean(dots)

    return check_P1, check_P2, check_P3, check_P4


# ─────────────────────────────────────────────────────────────────────────────
# §3  Driver
# ─────────────────────────────────────────────────────────────────────────────

def run_verification(n_samples: int = 2048, rtol: float = 1e-4) -> bool:
    print("=" * 72)
    print("  PROJECT-GP  ·  Passive H_net Algebraic Verification")
    print("=" * 72)

    rng = jax.random.PRNGKey(0)
    model, params = init_passive_hnet(rng)
    check_P1, check_P2, check_P3, check_P4 = make_checkers(model)

    rng, sub = jax.random.split(rng)
    q, p, setup = sample_envelope(sub, n_samples)

    all_pass = True

    # P1
    h_min, h_max = check_P1(params, q, p, setup)
    p1_pass = bool(h_min >= -rtol * (abs(float(h_max)) + 1.0))
    print(f"\n  [P1] H_net ≥ 0")
    print(f"       min = {float(h_min):+.3e}   max = {float(h_max):+.3e}")
    print(f"       {'PASS ✓' if p1_pass else 'FAIL ✗'}")
    all_pass &= p1_pass

    # P2
    p2_err = check_P2(params, setup)
    p2_pass = bool(float(p2_err) < rtol)
    print(f"\n  [P2] H_net(q_eq, 0, setup) = 0")
    print(f"       max |error| = {float(p2_err):.3e}   (tol {rtol:.0e})")
    print(f"       {'PASS ✓' if p2_pass else 'FAIL ✗'}")
    all_pass &= p2_pass

    # P3
    p3_err = check_P3(params, q, setup)
    p3_pass = bool(float(p3_err) < rtol)
    print(f"\n  [P3] ∇_p H_net(q, 0, setup) = 0")
    print(f"       max ‖∇_p H_net‖ at p=0 = {float(p3_err):.3e}   (tol {rtol:.0e})")
    print(f"       {'PASS ✓' if p3_pass else 'FAIL ✗'}")
    all_pass &= p3_pass

    # P4
    p4_min, p4_mean = check_P4(params, q, p, setup)
    p4_pass = bool(float(p4_min) >= -1e-3 * (abs(float(p4_mean)) + 1.0))
    print(f"\n  [P4] pᵀ ∇_p H_net ≥ 0")
    print(f"       min = {float(p4_min):+.3e}   mean = {float(p4_mean):+.3e}")
    print(f"       {'PASS ✓' if p4_pass else 'FAIL ✗'}")
    all_pass &= p4_pass

    print("\n" + "=" * 72)
    verdict = "✓ ALL PROPERTIES ALGEBRAIC" if all_pass else "✗ STRUCTURAL DEFECT — see above"
    print(f"  OVERALL: {verdict}")
    print("=" * 72)
    return all_pass


if __name__ == "__main__":
    ok = run_verification()
    sys.exit(0 if ok else 1)