"""Regression contracts for the full-vehicle gradient audit.

These check computational-map identities, rather than asserting that a finite
difference must agree with an IFT reference for a finite implicit solve.

Task 11 from the gradient-map audit specification.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from experiments.full_vehicle_gradient_audit import (
    AuditCase,
    _full_setup,
    _metrics,
    _state_objective,
    converged_stages,
    explicit_ift,
    finite_difference,
    implicit_context,
    make_case,
    production_stages,
    reconstruct_from_stages,
    stage_residual,
)


# ── Test 1: Production state == explicit P32 reconstruction ──────────────────

def test_x64_production_state_equals_exact_unrolled_stage_reconstruction():
    """production.simulate_step must be bitwise identical to the diagnostic map."""
    case = AuditCase(n_setup_params=1)
    vehicle, x, controls, setup, _ = make_case(case)
    assert x.dtype == jnp.float64, "x64 required for gradient audit"
    assert setup.dtype == jnp.float64, "x64 required for gradient audit"

    stages = production_stages(vehicle, x, controls, setup, case.dt)
    reconstructed = reconstruct_from_stages(x, stages, case.dt)
    production = vehicle.simulate_step(x, controls, setup, dt=case.dt, n_substeps=1)
    delta = float(jnp.max(jnp.abs(production - reconstructed)))
    assert delta < 1e-12, f"state identity failed: max_abs={delta:.3e}"


# ── Test 2: P32 AD == independently unrolled P32 AD ─────────────────────────

def test_production_ad_equals_unrolled_ad():
    """jax.grad through simulate_step must equal jax.grad through production_stages."""
    case = AuditCase(n_setup_params=6)
    vehicle, x, controls, setup, target = make_case(case)
    theta = setup[: case.n_setup_params]

    def production_obj(th):
        state = vehicle.simulate_step(x, controls, _full_setup(setup, th), dt=case.dt, n_substeps=1)
        return _state_objective(vehicle, state, target, _full_setup(setup, th), "benchmark")

    def unrolled_obj(th):
        full = _full_setup(setup, th)
        state = reconstruct_from_stages(x, production_stages(vehicle, x, controls, full, case.dt), case.dt)
        return _state_objective(vehicle, state, target, full, "benchmark")

    prod_grad = jax.grad(production_obj)(theta)
    unrolled_grad = jax.grad(unrolled_obj)(theta)
    m = _metrics(prod_grad, unrolled_grad)
    assert m["relative"] < 1e-8, (
        f"production AD != unrolled AD: relative={m['relative']:.3e}, cosine={m['cosine']:.6f}"
    )


# ── Test 3: P32 AD == FD_P32 within justified tolerance ─────────────────────

def test_production_ad_matches_fd():
    """jax.grad through simulate_step must agree with central FD of same map."""
    case = AuditCase(n_setup_params=6)
    vehicle, x, controls, setup, target = make_case(case)
    theta = setup[: case.n_setup_params]

    def production_obj(th):
        state = vehicle.simulate_step(x, controls, _full_setup(setup, th), dt=case.dt, n_substeps=1)
        return _state_objective(vehicle, state, target, _full_setup(setup, th), "benchmark")

    prod_grad = jax.grad(production_obj)(theta)
    fd_grad = finite_difference(production_obj, theta, 1e-4)
    m = _metrics(prod_grad, fd_grad)
    # Tolerance accounts for FD truncation error and potential branch effects
    assert m["relative"] < 0.1, (
        f"production AD vs FD: relative={m['relative']:.3e}, cosine={m['cosine']:.6f}"
    )
    assert m["cosine"] > 0.9, (
        f"production AD vs FD cosine too low: {m['cosine']:.6f}"
    )


# ── Test 4: Converged root residual is small ─────────────────────────────────

def test_converged_root_residual_small():
    """The converged solver must achieve ||F||_inf < 1e-10."""
    case = AuditCase(n_setup_params=1)
    vehicle, x, controls, setup, _ = make_case(case)
    z_star, info = converged_stages(vehicle, x, controls, setup, case.dt, tol=1e-10)
    assert info["converged"], f"converged_stages did not converge: res_inf={info['final_residual_inf']:.3e}"
    assert info["final_residual_inf"] < 1e-10, (
        f"residual too large: {info['final_residual_inf']:.3e}"
    )


# ── Test 5: IFT identity at converged root is small ─────────────────────────

def test_complete_ift_satisfies_its_linearized_216_state_identity():
    """At converged z*, ||F_z dz/dθ + F_θ|| must be small."""
    case = AuditCase(n_setup_params=1)
    vehicle, x, controls, setup, target = make_case(case)
    theta = setup[:1]

    z_star, info = converged_stages(vehicle, x, controls, setup, case.dt, tol=1e-10)
    assert info["converged"], "converged_stages did not converge"

    ctx = implicit_context(vehicle, x, controls, setup, theta, case.dt, z=z_star)
    assert float(ctx["residual_inf"]) < 1e-8, (
        f"residual at z* too large: {float(ctx['residual_inf']):.3e}"
    )
    assert float(ctx["ift_identity_l2"]) < 1e-6, (
        f"IFT identity residual too large: {float(ctx['ift_identity_l2']):.3e}"
    )


# ── Test 6: x64 enforced ────────────────────────────────────────────────────

def test_x64_enforced():
    """Verify that make_case returns x64 arrays."""
    case = AuditCase()
    vehicle, x, controls, setup, target = make_case(case)
    assert x.dtype == jnp.float64
    assert setup.dtype == jnp.float64
    assert controls.dtype == jnp.float64
    assert target.dtype == jnp.float64
