"""Project-GP — structural PassiveHNet verification (Batch 1.3A).

The structural claims are evaluated on H_raw and V directly.  The production
``tanh`` cap is reported separately as a numerical-safety diagnostic; it is not
used as evidence for P1/P4.
"""

from __future__ import annotations

import os

# Validation needs enough mantissa to distinguish the exact Bregman grounding
# identity from float32 cancellation. Set before importing jax/numpy arrays.
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from physics.h_net_icnn import PassiveHNet, _Z_EQ_DEFAULT


N_SAMPLES = 2048
TOL_P2 = 1e-10
TOL_P3 = 1e-10
TOL_P5 = 1e-10
TOL_NEG = 1e-10
H_CAP = 50_000.0


def _make_model(rng: jax.Array, *, output_mode: str):
    # Use a dimensionally coherent generalized-momentum scale: p_scale = M*v_ref.
    # Nominal values are the same order used by the vehicle model at v_ref=20 m/s.
    m_diag = jnp.array([
        370.0, 370.0, 370.0, 100.0, 100.0, 100.0,
        7.5, 7.5, 7.5, 7.5, 0.20, 0.20, 0.20, 0.20,
    ], dtype=jnp.float64)
    p_scale = tuple(float(x) for x in (m_diag * 20.0).tolist())
    model = PassiveHNet(
        q_dim=14,
        p_dim=14,
        setup_dim=28,
        p_scale=p_scale,
        h_cap=H_CAP,
        output_mode=output_mode,
    )
    params = model.init(
        rng,
        jnp.zeros(14, dtype=jnp.float64),
        jnp.zeros(14, dtype=jnp.float64),
        jnp.zeros(28, dtype=jnp.float64),
    )
    return model, params


def _apply(model, params, q, p, setup):
    return model.apply(params, q, p, setup)


def run_verification(seed: int = 42, n_samples: int = N_SAMPLES) -> bool:
    print("=" * 72)
    print("PROJECT-GP · PassiveHNet Structural Verification · Batch 1.3B")
    print("=" * 72)
    print("Precision: x64")
    print("Structural P1–P6 use H_raw / V directly; capped H is diagnostic only.\n")

    rng = jax.random.PRNGKey(seed)
    rng_h, rng_q, rng_p, rng_s = jax.random.split(rng, 4)
    raw_model, raw_params = _make_model(rng_h, output_mode="raw")
    v_model, v_params = _make_model(rng_h, output_mode="potential")
    capped_model = PassiveHNet(
        q_dim=14, p_dim=14, setup_dim=28,
        p_scale=raw_model.p_scale,
        h_cap=H_CAP,
        output_mode="capped",
    )

    # Same parameter tree is valid because output_mode changes only the returned view.
    capped_params = raw_params

    q = _Z_EQ_DEFAULT.astype(jnp.float64)[None, :] + 0.04 * jax.random.normal(
        rng_q, (n_samples, 14), dtype=jnp.float64
    )
    p_scale = jnp.asarray(raw_model.p_scale, dtype=jnp.float64)
    # Physical-envelope stress test: sample dimensionless momentum around
    # the nominal p = M*v_ref scale rather than assigning the same 200 kg·m/s
    # to every DOF (which is especially extreme for wheel inertia states).
    p = p_scale[None, :] * (0.75 * jax.random.normal(
        rng_p, (n_samples, 14), dtype=jnp.float64
    ))
    setup = jax.random.uniform(rng_s, (n_samples, 28), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    q_eq = jnp.broadcast_to(_Z_EQ_DEFAULT.astype(jnp.float64), (n_samples, 14))
    p_zero = jnp.zeros_like(p)

    raw_batched = jax.vmap(lambda qi, pi, si: _apply(raw_model, raw_params, qi, pi, si))(q, p, setup)
    v_batched = jax.vmap(lambda qi, si: _apply(v_model, v_params, qi, jnp.zeros(14, dtype=jnp.float64), si))(q, setup)
    cap_batched = jax.vmap(lambda qi, pi, si: _apply(capped_model, capped_params, qi, pi, si))(q, p, setup)

    # P1: raw H >= 0.
    p1_min = float(jnp.min(raw_batched))
    p1_ok = p1_min >= -TOL_NEG

    # P2: exact grounding at q_eq, p=0.
    p2_err = float(jnp.max(jnp.abs(jax.vmap(lambda si: _apply(
        raw_model, raw_params, _Z_EQ_DEFAULT.astype(jnp.float64),
        jnp.zeros(14, dtype=jnp.float64), si))(setup))))
    p2_ok = p2_err <= TOL_P2

    # P3: grad_p H_raw(q,0,s) = 0. Test off-equilibrium q so the result is not
    # accidentally relying on P2.
    grad_p = jax.vmap(jax.grad(lambda pi, qi, si: _apply(
        raw_model, raw_params, qi, pi, si), argnums=0))(p_zero, q, setup)
    p3_max = float(jnp.max(jnp.linalg.norm(grad_p, axis=1)))
    p3_ok = p3_max <= TOL_P3

    # P4: p^T grad_p H_raw >= 0.
    grad_p_random = jax.vmap(jax.grad(lambda pi, qi, si: _apply(
        raw_model, raw_params, qi, pi, si), argnums=0))(p, q, setup)
    directional = jnp.sum(p * grad_p_random, axis=1)
    p4_min = float(jnp.min(directional))
    p4_mean = float(jnp.mean(directional))
    p4_ok = p4_min >= -TOL_NEG

    # P5: grad_q V(q_eq,s)=0; evaluate V branch directly to avoid K/psi/cap.
    grad_v_eq = jax.vmap(jax.grad(lambda qi, si: _apply(
        v_model, v_params, qi, jnp.zeros(14, dtype=jnp.float64), si), argnums=0))(
            q_eq, setup
        )
    p5_norm = jnp.linalg.norm(grad_v_eq, axis=1)
    p5_max = float(jnp.max(p5_norm))
    p5_mean = float(jnp.mean(p5_norm))
    p5_ok = p5_max <= TOL_P5

    # P6: Bregman V >= 0.
    p6_min = float(jnp.min(v_batched))
    p6_ok = p6_min >= -TOL_NEG

    # Cap diagnostic: quantifies whether production H is entering tanh saturation.
    sat = jnp.abs(raw_batched) / H_CAP
    sat_fraction = float(jnp.mean(sat > 0.25))
    sat_hard_fraction = float(jnp.mean(sat > 0.80))
    cap_max = float(jnp.max(cap_batched))

    rows = [
        ("P1", "H_raw >= 0", p1_ok, f"min={p1_min:+.6e}"),
        ("P2", "H_raw(q_eq,0,s)=0", p2_ok, f"max|err|={p2_err:.6e}"),
        ("P3", "grad_p H_raw(q,0,s)=0", p3_ok, f"max norm={p3_max:.6e}"),
        ("P4", "p^T grad_p H_raw >= 0", p4_ok, f"min={p4_min:+.6e}, mean={p4_mean:.6e}"),
        ("P5", "grad_q V(q_eq,s)=0", p5_ok, f"max={p5_max:.6e}, mean={p5_mean:.6e}"),
        ("P6", "V >= 0", p6_ok, f"min={p6_min:+.6e}"),
    ]

    for name, desc, ok, detail in rows:
        print(f"[{name}] {desc:<32} {detail:<42} {'PASS' if ok else 'FAIL'}")

    print("\n[CAP] Numerical output diagnostic")
    print(f"      max H_capped       = {cap_max:.6e}")
    print(f"      frac |H_raw|/cap >0.25 = {sat_fraction:.3%}")
    print(f"      frac |H_raw|/cap >0.80 = {sat_hard_fraction:.3%}")
    if sat_hard_fraction > 0.01:
        print("      WARNING: significant cap saturation remains in the validation envelope.")
        print("               This is not a passivity failure; it means the NN scale/cap\n"
              "               should be revisited before gradient benchmarking.")

    overall = all(ok for _, _, ok, _ in rows)
    print("\nOVERALL:", "PASS — structural invariants verified." if overall else "FAIL — see failed properties above.")
    return overall


if __name__ == "__main__":
    raise SystemExit(0 if run_verification() else 1)
