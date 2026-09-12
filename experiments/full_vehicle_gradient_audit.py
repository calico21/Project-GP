#!/usr/bin/env python3
"""Full-vehicle, one-step gradient-map audit for :mod:`vehicle_dynamics`.

This is deliberately a diagnostic, not a replacement integrator.  It keeps
four maps separate:

``production``
    ``simulate_step`` and its fixed 32 Picard iterations;
``unrolled``
    the same 32 iterations written out here and differentiated by JAX; and
``implicit root (P32)``
    ``F(z, theta) = 0`` IFT evaluated at P_32 output (approximate); and
``implicit root (converged)``
    ``F(z*, theta) = 0`` IFT at a genuinely converged root.

The distinction matters: the IFT is the derivative of a *converged root*,
whereas production is, by contract, the derivative of a finite iteration map.
No solver tolerance, finite-difference step, or vehicle equation is changed by
this experiment.

Run ``JAX_PLATFORMS=cpu python experiments/full_vehicle_gradient_audit.py``.
It writes a JSON report to ``results/full_vehicle_gradient_audit.json``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

os.environ.setdefault("JAX_ENABLE_X64", "true")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from config.tire_coeffs import tire_coeffs as TC
from config.vehicles.ter27 import vehicle_params_ter27 as VP
from models.vehicle_dynamics import (
    DEFAULT_SETUP,
    SETUP_LB,
    SETUP_NAMES,
    _GLRK_PICARD_ITERS,
    DifferentiableMultiBodyVehicle,
)


N_STATE = 108
N_STAGE = 2 * N_STATE
SQRT3 = jnp.sqrt(3.0)
A11, A12 = 0.25, 0.25 - SQRT3 / 6.0
A21, A22 = 0.25 + SQRT3 / 6.0, 0.25


@dataclass(frozen=True)
class AuditCase:
    """The fixed inputs defining the audited one-step map."""

    dt: float = 0.005
    n_setup_params: int = 6
    vx0: float = 20.0
    target_vx: float = 22.0


def make_case(case: AuditCase = AuditCase()):
    """Construct the deterministic, x64 operating point used by every map."""
    vehicle = DifferentiableMultiBodyVehicle(VP, TC)
    x0 = vehicle.make_initial_state(vx0=case.vx0).astype(jnp.float64)
    target = vehicle.make_initial_state(vx0=case.target_vx).astype(jnp.float64)[:28]
    setup = (0.5 * (DEFAULT_SETUP + SETUP_LB)).astype(jnp.float64)
    controls = jnp.array([0.03, 10.0, 10.0, 10.0, 10.0, 0.0], dtype=jnp.float64)
    return vehicle, x0, controls, setup, target


def _full_setup(setup: jax.Array, theta: jax.Array) -> jax.Array:
    return setup.at[: theta.shape[0]].set(theta)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Core GLRK stage equations (must match production _glrk4_step exactly)
# ─────────────────────────────────────────────────────────────────────────────

def stage_residual(vehicle, x, controls, setup, dt, z):
    """The actual complete 216-variable production GLRK residual ``F``."""
    k1, k2 = z.reshape(2, N_STATE)
    x1 = x + dt * (A11 * k1 + A12 * k2)
    x2 = x + dt * (A21 * k1 + A22 * k2)
    # These stage-rate clips are part of _glrk4_step and consequently part of F.
    f1 = jnp.clip(vehicle._compute_derivatives(x1, controls, setup), -500.0, 500.0)
    f2 = jnp.clip(vehicle._compute_derivatives(x2, controls, setup), -500.0, 500.0)
    return z - jnp.concatenate((f1, f2))


def reconstruct_from_stages(x, z, dt):
    """Exact final-state reconstruction used by the production GLRK step."""
    k1, k2 = z.reshape(2, N_STATE)
    out = x + 0.5 * dt * (k1 + k2)
    return out.at[28:N_STATE].set(jnp.clip(out[28:N_STATE], -1000.0, 1000.0))


def production_stages(vehicle, x, controls, setup, dt, iterations=_GLRK_PICARD_ITERS):
    """The production Picard map, including its *unclipped* initial guess.

    This differs subtly from starting with ``clip(f(x))``.  Matching this
    first iterate is essential when auditing derivatives of a finite solve.
    """
    dx0 = vehicle._compute_derivatives(x, controls, setup)
    z0 = jnp.concatenate((dx0, dx0))

    def body(z, _):
        return z - stage_residual(vehicle, x, controls, setup, dt, z), None

    z, _ = jax.lax.scan(body, z0, None, length=iterations)
    return z


# ─────────────────────────────────────────────────────────────────────────────
# §2  Converged root solver (diagnostic only — Task 5)
# ─────────────────────────────────────────────────────────────────────────────

def converged_stages(vehicle, x, controls, setup, dt, tol=1e-12, max_iter=500):
    """Solve F(z,θ)=0 by continued Picard iteration with residual monitoring.

    Returns (z_star, info_dict) where info_dict contains convergence diagnostics.
    This is NOT used in production — it provides the IFT reference only.
    """
    dx0 = vehicle._compute_derivatives(x, controls, setup)
    z = jnp.concatenate((dx0, dx0))

    residual_history = []
    for i in range(max_iter):
        F = stage_residual(vehicle, x, controls, setup, dt, z)
        res_inf = float(jnp.max(jnp.abs(F)))
        res_l2 = float(jnp.linalg.norm(F))
        residual_history.append({"iter": i, "res_inf": res_inf, "res_l2": res_l2})
        if res_inf < tol:
            break
        z = z - F  # Picard step: z_{n+1} = z_n - F(z_n)

    F_final = stage_residual(vehicle, x, controls, setup, dt, z)
    info = {
        "converged": float(jnp.max(jnp.abs(F_final))) < tol,
        "iterations": len(residual_history),
        "final_residual_inf": float(jnp.max(jnp.abs(F_final))),
        "final_residual_l2": float(jnp.linalg.norm(F_final)),
        "history_sample": residual_history[:5] + residual_history[-3:],
    }
    return z, info


# ─────────────────────────────────────────────────────────────────────────────
# §3  Objectives and metrics
# ─────────────────────────────────────────────────────────────────────────────

def _state_objective(vehicle, state, target, setup, kind: str):
    """Objective hierarchy: state component, linear state, energy, benchmark."""
    if kind == "state_component":
        return state[16]  # vertical body velocity; direct integrator output
    if kind == "linear_state":
        weights = jnp.linspace(-0.25, 0.35, 28, dtype=state.dtype)
        return weights @ state[:28]
    if kind == "mechanical_energy":
        q, v = state[:14], state[14:28]
        p = vehicle.M_diag * v
        return vehicle.H_net.apply(vehicle.H_params, q, p, setup)
    if kind == "benchmark":
        return jnp.sum((state[:28] - target) ** 2)
    raise ValueError(f"unknown objective kind: {kind}")


def _metrics(a, b):
    delta = a - b
    na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
    return {
        "l2": float(jnp.linalg.norm(delta)),
        "max_abs": float(jnp.max(jnp.abs(delta))),
        "relative": float(jnp.linalg.norm(delta) / (nb + 1e-30)),
        "cosine": float(jnp.vdot(a, b) / (na * nb + 1e-30)),
    }


def _host(x):
    return np.asarray(jax.device_get(x))


def _branch_signature(vehicle, x, controls, setup, dt, z):
    """Report active hard clips in the audited map (not a claim of smoothness)."""
    k1, k2 = z.reshape(2, N_STATE)
    x1 = x + dt * (A11 * k1 + A12 * k2)
    x2 = x + dt * (A21 * k1 + A22 * k2)
    raw1 = vehicle._compute_derivatives(x1, controls, setup)
    raw2 = vehicle._compute_derivatives(x2, controls, setup)
    out_unclipped = x + 0.5 * dt * (k1 + k2)
    return {
        "stage_rate_clip_counts": [int(jnp.sum(jnp.abs(raw1) >= 500.0)),
                                    int(jnp.sum(jnp.abs(raw2) >= 500.0))],
        "aux_output_clip_count": int(jnp.sum(jnp.abs(out_unclipped[28:]) >= 1000.0)),
    }


def finite_difference(f: Callable[[jax.Array], jax.Array], theta, eps):
    values_plus, values_minus = [], []
    for index in range(theta.size):
        direction = jnp.zeros_like(theta).at[index].set(eps)
        values_plus.append(f(theta + direction))
        values_minus.append(f(theta - direction))
    return jnp.stack([(p - m) / (2.0 * eps) for p, m in zip(values_plus, values_minus)])


def finite_difference_vector(f: Callable[[jax.Array], jax.Array], theta, eps):
    """Central FD for a vector-valued map f: R^n -> R^m.  Returns (m, n) Jacobian."""
    columns = []
    for index in range(theta.size):
        direction = jnp.zeros_like(theta).at[index].set(eps)
        col = (f(theta + direction) - f(theta - direction)) / (2.0 * eps)
        columns.append(col)
    return jnp.stack(columns, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# §4  IFT computation
# ─────────────────────────────────────────────────────────────────────────────

def implicit_context(vehicle, x, controls, setup, theta, dt, z=None):
    """Factor the complete IFT system once for several scalar objectives.

    If z is None, uses production_stages (P32).  Pass a converged z_star
    to evaluate the IFT at the genuine root.
    """
    if z is None:
        z = production_stages(vehicle, x, controls, setup, dt)

    def F(z_, theta_):
        return stage_residual(vehicle, x, controls, _full_setup(setup, theta_), dt, z_)

    fz = jax.jacrev(F, 0)(z, theta)
    ftheta = jax.jacrev(F, 1)(z, theta)
    dz_dtheta = jnp.linalg.solve(fz, -ftheta)
    residual = F(z, theta)
    singular_values = jnp.linalg.svd(fz, compute_uv=False)
    return {
        "z": z, "F": F, "fz": fz, "ftheta": ftheta, "dz_dtheta": dz_dtheta,
        "residual_l2": jnp.linalg.norm(residual), "residual_inf": jnp.max(jnp.abs(residual)),
        "ift_identity_l2": jnp.linalg.norm(fz @ dz_dtheta + ftheta),
        "condition_number": singular_values[0] / singular_values[-1],
        "sigma_min": singular_values[-1], "sigma_max": singular_values[0],
        "f_theta_l2": jnp.linalg.norm(ftheta), "dz_dtheta_l2": jnp.linalg.norm(dz_dtheta),
    }


def explicit_ift(vehicle, x, controls, setup, target, theta, dt, objective_kind, context=None):
    """Complete 216-dimensional IFT for the mathematical root map.

    ``G_theta`` is retained: it is zero for state-only objectives but nonzero
    for setup-conditioned mechanical energy.  Pass a shared ``context`` when
    comparing multiple objectives at the same state and parameter point.
    """
    context = context or implicit_context(vehicle, x, controls, setup, theta, dt)

    def G(z_, theta_):
        state = reconstruct_from_stages(x, z_, dt)
        return _state_objective(vehicle, state, target, _full_setup(setup, theta_), objective_kind)

    gz = jax.jacrev(G, 0)(context["z"], theta)
    gtheta = jax.jacrev(G, 1)(context["z"], theta)
    gradient = gtheta + gz @ context["dz_dtheta"]
    return {**context, "gradient": gradient}


def _parameter_rows(theta, fd, ift, unrolled, production):
    rows = []
    for i, name in enumerate(SETUP_NAMES[: theta.size]):
        reference = float(fd[i])
        row = {
            "index": i,
            "parameter": name,
            "value": float(theta[i]),
            "fd": reference,
            "ift": float(ift[i]),
            "unrolled_ad": float(unrolled[i]),
            "production_ad": float(production[i]),
            "production_fd_relative": abs(float(production[i] - fd[i])) / (abs(reference) + 1e-30),
            "sign_agreement_production_fd": bool(np.signbit(float(production[i])) == np.signbit(reference)),
        }
        rows.append(row)
    return sorted(rows, key=lambda r: r["production_fd_relative"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Iteration convergence sweep (Task 7)
# ─────────────────────────────────────────────────────────────────────────────

def iteration_convergence_sweep(vehicle, x, controls, setup, target, theta, dt, z_star, ift_root_grad,
                                N_values=(1, 2, 4, 8, 16, 32, 64, 128)):
    """Compare P_N maps for varying iteration counts.

    Reports: residual, ||z_N - z*||, AD_PN norm, FD_PN norm,
    AD_PN vs FD_PN, AD_PN vs IFT_root.
    """
    rows = []
    for N in N_values:
        # P_N map
        def P_N_objective(theta_, _N=N):
            full = _full_setup(setup, theta_)
            z = production_stages(vehicle, x, controls, full, dt, iterations=_N)
            state = reconstruct_from_stages(x, z, dt)
            return _state_objective(vehicle, state, target, full, "benchmark")

        def P_N_stages(theta_, _N=N):
            full = _full_setup(setup, theta_)
            return production_stages(vehicle, x, controls, full, dt, iterations=_N)

        # Stage vector at this N
        z_N = production_stages(vehicle, x, controls, setup, dt, iterations=N)
        residual_F = stage_residual(vehicle, x, controls, setup, dt, z_N)
        res_inf = float(jnp.max(jnp.abs(residual_F)))
        z_delta = float(jnp.linalg.norm(z_N - z_star))

        # AD gradient
        ad_grad = jax.grad(P_N_objective)(theta)
        ad_norm = float(jnp.linalg.norm(ad_grad))

        # FD gradient
        eps = 1e-4
        fd_grad = finite_difference(P_N_objective, theta, eps)
        fd_norm = float(jnp.linalg.norm(fd_grad))

        ad_fd = _metrics(ad_grad, fd_grad)
        ad_ift = _metrics(ad_grad, ift_root_grad)

        rows.append({
            "N": N,
            "residual_inf": res_inf,
            "z_delta_from_star": z_delta,
            "ad_norm": ad_norm,
            "fd_norm": fd_norm,
            "ad_vs_fd_relative": ad_fd["relative"],
            "ad_vs_fd_cosine": ad_fd["cosine"],
            "ad_vs_ift_relative": ad_ift["relative"],
            "ad_vs_ift_cosine": ad_ift["cosine"],
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# §6  Conditioning analysis (Task 10)
# ─────────────────────────────────────────────────────────────────────────────

def conditioning_analysis(context):
    """Analyse the 216×216 stage Jacobian F_z."""
    fz = context["fz"]
    sv = jnp.linalg.svd(fz, compute_uv=False)
    return {
        "sigma_min": float(sv[-1]),
        "sigma_max": float(sv[0]),
        "condition_number": float(sv[0] / sv[-1]),
        "f_theta_l2": float(context["f_theta_l2"]),
        "dz_dtheta_l2": float(context["dz_dtheta_l2"]),
        "ift_identity_l2": float(context["ift_identity_l2"]),
        "rank_216": int(jnp.sum(sv > 1e-12)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §7  Main audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_one_step(case: AuditCase = AuditCase(), eps_values=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)):
    """Execute the central one-step identity and gradient-map audit."""
    vehicle, x, controls, setup, target = make_case(case)
    theta = setup[: case.n_setup_params]
    if x.dtype != jnp.float64 or theta.dtype != jnp.float64:
        raise RuntimeError("gradient audit requires x64 state and setup arrays")

    print("=" * 70)
    print("FULL VEHICLE GRADIENT MAP AUDIT")
    print("=" * 70)

    # ── TASK 1: Production map identity ──────────────────────────────────
    print("\n── Task 1: Production map identity ──")
    z = production_stages(vehicle, x, controls, setup, case.dt)
    reconstructed = reconstruct_from_stages(x, z, case.dt)
    production_state = vehicle.simulate_step(x, controls, setup, dt=case.dt, n_substeps=1)
    state_delta = _host(production_state - reconstructed)
    state_identity = {
        "max_abs": float(np.max(np.abs(state_delta))),
        "l2": float(np.linalg.norm(state_delta)),
        "relative": float(np.linalg.norm(state_delta) / (np.linalg.norm(_host(reconstructed)) + 1e-30)),
        "max_index": int(np.argmax(np.abs(state_delta))),
        "per_block_max": {
            "q": float(np.max(np.abs(state_delta[:14]))), "p": float(np.max(np.abs(state_delta[14:28]))),
            "thermal": float(np.max(np.abs(state_delta[28:56]))), "slip": float(np.max(np.abs(state_delta[56:72]))),
            "damper": float(np.max(np.abs(state_delta[72:84]))), "elastokinematics": float(np.max(np.abs(state_delta[84:]))),
        },
    }
    print(f"  State identity max_abs: {state_identity['max_abs']:.3e}")
    print(f"  State identity relative: {state_identity['relative']:.3e}")

    # ── TASK 2: Primary gradient validation ──────────────────────────────
    print("\n── Task 2: Primary gradient validation (AD vs FD for P32) ──")

    def production_objective(theta_, kind="benchmark"):
        state = vehicle.simulate_step(x, controls, _full_setup(setup, theta_), dt=case.dt, n_substeps=1)
        return _state_objective(vehicle, state, target, _full_setup(setup, theta_), kind)

    def unrolled_objective(theta_, kind="benchmark"):
        full = _full_setup(setup, theta_)
        state = reconstruct_from_stages(x, production_stages(vehicle, x, controls, full, case.dt), case.dt)
        return _state_objective(vehicle, state, target, full, kind)

    prod_ad = jax.grad(production_objective)(theta)
    unrolled_ad = jax.grad(unrolled_objective)(theta)
    production_unrolled = _metrics(prod_ad, unrolled_ad)
    print(f"  production_ad vs unrolled_ad relative: {production_unrolled['relative']:.3e}")
    print(f"  production_ad vs unrolled_ad cosine:   {production_unrolled['cosine']:.6f}")

    fd = {str(eps): finite_difference(production_objective, theta, eps) for eps in eps_values}
    fd_sweep_metrics = {key: _metrics(prod_ad, value) for key, value in fd.items()}
    print("  FD sweep (production_ad vs FD):")
    for key in sorted(fd_sweep_metrics.keys(), key=lambda k: -float(k)):
        m = fd_sweep_metrics[key]
        print(f"    eps={key:>8s}  relative={m['relative']:.3e}  cosine={m['cosine']:.6f}")

    # ── TASK 3: Vector-level validation ──────────────────────────────────
    print("\n── Task 3: Vector-level validation ──")
    stage_map = lambda th: production_stages(vehicle, x, controls, _full_setup(setup, th), case.dt)
    state_map = lambda th: reconstruct_from_stages(x, stage_map(th), case.dt)
    dz_ad = jax.jacrev(stage_map)(theta)
    dx_ad = jax.jacrev(state_map)(theta)
    eps_mid = 1e-4
    dz_fd = finite_difference_vector(stage_map, theta, eps_mid)
    dx_fd = finite_difference_vector(state_map, theta, eps_mid)
    vector_metrics = {
        "stage_vector": _metrics(dz_ad, dz_fd),
        "reconstructed_state": _metrics(dx_ad, dx_fd),
    }
    print(f"  216-dim stage Jacobian AD vs FD: relative={vector_metrics['stage_vector']['relative']:.3e}")
    print(f"  108-dim state Jacobian AD vs FD: relative={vector_metrics['reconstructed_state']['relative']:.3e}")

    # ── TASK 4: Objective hierarchy ──────────────────────────────────────
    print("\n── Task 4: Objective hierarchy ──")
    # IFT at P32 (for hierarchy comparison — NOT the true converged root)
    context_p32 = implicit_context(vehicle, x, controls, setup, theta, case.dt)
    hierarchy = {}
    energy_fd_sweep = None

    for kind in ("state_component", "linear_state", "mechanical_energy", "benchmark"):
        p_ad = jax.grad(
            lambda th, k=kind: production_objective(th, k)
        )(theta)

        u_ad = jax.grad(
            lambda th, k=kind: unrolled_objective(th, k)
        )(theta)

        root = explicit_ift(
            vehicle,
            x,
            controls,
            setup,
            target,
            theta,
            case.dt,
            kind,
            context_p32,
        )

        if kind == "mechanical_energy":
            energy_fd = {
                str(eps): finite_difference(
                    lambda th, k=kind: production_objective(th, k),
                    theta,
                    eps,
                )
                for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
            }

            fd_k = energy_fd["0.0001"]

            energy_fd_sweep = {
                eps: _metrics(p_ad, fd_grad)
                for eps, fd_grad in energy_fd.items()
            }
        else:
            fd_k = finite_difference(
                lambda th, k=kind: production_objective(th, k),
                theta,
                1e-4,
            )

        hierarchy[kind] = {
            "production_unrolled": _metrics(p_ad, u_ad),
            "production_fd": _metrics(p_ad, fd_k),
            "production_ift": _metrics(p_ad, root["gradient"]),
        }

        if kind == "mechanical_energy":
            hierarchy[kind]["fd_epsilon_sweep"] = energy_fd_sweep

            print("  mechanical_energy FD epsilon sweep:")
            for eps in sorted(
                energy_fd_sweep.keys(),
                key=lambda e: -float(e),
            ):
                m = energy_fd_sweep[eps]
                print(
                    f"    eps={eps:>8s} "
                    f"relative={m['relative']:.3e} "
                    f"cosine={m['cosine']:.6f}"
                )

        print(
            f"  {kind:25s} "
            f"prod-unrolled={hierarchy[kind]['production_unrolled']['relative']:.3e}  "
            f"prod-fd={hierarchy[kind]['production_fd']['relative']:.3e}"
        )

    # ── TASK 5: Converged root ───────────────────────────────────────────
    print("\n── Task 5: Converged root solver ──")
    convergence_table = {}
    z_star = None
    z_star_info = None
    z_star_tol = None

    for tol in (1e-6, 1e-8, 1e-10, 1e-12):
        z_conv, info = converged_stages(
            vehicle, x, controls, setup, case.dt, tol=tol
        )
        convergence_table[str(tol)] = info

        print(
            f"  tol={tol:.0e}: converged={info['converged']}, "
            f"iters={info['iterations']}, "
            f"res_inf={info['final_residual_inf']:.3e}"
        )

        # Retain the tightest successfully converged root.
        if info["converged"]:
            z_star = z_conv
            z_star_info = info
            z_star_tol = tol

    if z_star is None:
        raise RuntimeError(
            "No converged root found at any requested tolerance; "
            "cannot construct a valid IFT_root reference."
        )

    print(
        f"  Selected z*: tol={z_star_tol:.0e}, "
        f"iterations={z_star_info['iterations']}, "
        f"res_inf={z_star_info['final_residual_inf']:.3e}"
    )

    z_p32 = production_stages(vehicle, x, controls, setup, case.dt)
    z_star_delta = float(jnp.linalg.norm(z_p32 - z_star))
    z_star_delta_inf = float(jnp.max(jnp.abs(z_p32 - z_star)))
    print(f"  ||z_P32 - z*||_2 = {z_star_delta:.3e}")
    print(f"  ||z_P32 - z*||_inf = {z_star_delta_inf:.3e}")

    # ── TASK 6: True IFT at converged root ───────────────────────────────
    print("\n── Task 6: True IFT at converged root ──")
    context_star = implicit_context(vehicle, x, controls, setup, theta, case.dt, z=z_star)
    ift_root = explicit_ift(vehicle, x, controls, setup, target, theta, case.dt, "benchmark", context_star)
    ift_root_grad = ift_root["gradient"]
    print(f"  Converged residual inf: {float(context_star['residual_inf']):.3e}")
    print(f"  IFT identity ||F_z dz/dθ + F_θ|| = {float(context_star['ift_identity_l2']):.3e}")
    print(f"  IFT gradient norm: {float(jnp.linalg.norm(ift_root_grad)):.6e}")
    print(f"  prod_ad vs IFT_root: relative={_metrics(prod_ad, ift_root_grad)['relative']:.3e}, "
          f"cosine={_metrics(prod_ad, ift_root_grad)['cosine']:.6f}")

    # ── TASK 7: Iteration convergence ────────────────────────────────────
    print("\n── Task 7: Iteration convergence sweep ──")
    iter_rows = iteration_convergence_sweep(
        vehicle, x, controls, setup, target, theta, case.dt, z_star, ift_root_grad)
    print(f"  {'N':>5s}  {'res_inf':>10s}  {'||z-z*||':>10s}  {'AD-FD rel':>10s}  {'AD-IFT rel':>10s}  {'cos(AD,FD)':>10s}")
    for row in iter_rows:
        print(f"  {row['N']:>5d}  {row['residual_inf']:>10.3e}  {row['z_delta_from_star']:>10.3e}  "
              f"{row['ad_vs_fd_relative']:>10.3e}  {row['ad_vs_ift_relative']:>10.3e}  "
              f"{row['ad_vs_fd_cosine']:>10.6f}")

    # ── TASK 8: Parameter localization ───────────────────────────────────
    print("\n── Task 8: Parameter localization ──")
    param_table = _parameter_rows(theta, fd["0.0001"], ift_root_grad, unrolled_ad, prod_ad)
    print(f"  {'param':>18s}  {'FD':>12s}  {'prod_AD':>12s}  {'IFT_root':>12s}  {'AD-FD rel':>10s}  {'sign ok':>7s}")
    for row in param_table:
        print(f"  {row['parameter']:>18s}  {row['fd']:>12.6e}  {row['production_ad']:>12.6e}  "
              f"{row['ift']:>12.6e}  {row['production_fd_relative']:>10.3e}  "
              f"{'Y' if row['sign_agreement_production_fd'] else 'N':>7s}")

    # ── TASK 9: Branch/clip audit ────────────────────────────────────────
    print("\n── Task 9: Branch/clip audit ──")
    branches = {"nominal": _branch_signature(vehicle, x, controls, setup, case.dt, z)}
    print(f"  Nominal clips: {branches['nominal']}")
    for eps in [1e-3, 1e-4]:
        signatures = []
        any_change = False
        for i in range(theta.size):
            direction = jnp.zeros_like(theta).at[i].set(eps)
            sig_plus = _branch_signature(vehicle, x, controls, _full_setup(setup, theta + direction), case.dt,
                                          stage_map(theta + direction))
            sig_minus = _branch_signature(vehicle, x, controls, _full_setup(setup, theta - direction), case.dt,
                                           stage_map(theta - direction))
            if sig_plus != branches["nominal"] or sig_minus != branches["nominal"]:
                any_change = True
            signatures.append({"index": i, "plus": sig_plus, "minus": sig_minus})
        branches[str(eps)] = signatures
        print(f"  eps={eps:.0e}: branch change under perturbation = {any_change}")

    # ── TASK 10: Conditioning analysis ───────────────────────────────────
    print("\n── Task 10: Conditioning analysis ──")
    cond_p32 = conditioning_analysis(context_p32)
    cond_star = conditioning_analysis(context_star)
    print(f"  At P32:")
    print(f"    σ_min={cond_p32['sigma_min']:.3e}, σ_max={cond_p32['sigma_max']:.3e}, "
          f"cond={cond_p32['condition_number']:.3e}")
    print(f"    ||F_θ||={cond_p32['f_theta_l2']:.3e}, ||dz/dθ||={cond_p32['dz_dtheta_l2']:.3e}")
    print(f"  At z*:")
    print(f"    σ_min={cond_star['sigma_min']:.3e}, σ_max={cond_star['sigma_max']:.3e}, "
          f"cond={cond_star['condition_number']:.3e}")
    print(f"    IFT identity={cond_star['ift_identity_l2']:.3e}")

    # ── Final classification ─────────────────────────────────────────────
    print("\n── Final Classification ──")
    # Classification is evidence-driven
    if production_unrolled["relative"] > 1e-6:
        classification = "E) PRODUCTION MAP PATH MISMATCH"
    elif fd_sweep_metrics["0.0001"]["relative"] <= 1e-4 and production_unrolled["relative"] <= 1e-10:
        classification = "A) PRODUCTION GRADIENT FULLY VALIDATED"
    elif fd_sweep_metrics["0.0001"]["relative"] <= 1e-3:
        classification = "B) PRODUCTION GRADIENT VALIDATED WITH NUMERICAL LIMITATION"
    elif production_unrolled["relative"] <= 1e-10 and fd_sweep_metrics["0.0001"]["relative"] > 1e-3:
        if context_star["residual_inf"] > 1e-8:
            classification = "G) CONVERGED ROOT / IFT NOT YET NUMERICALLY VALID"
        else:
            # AD agrees with unrolled but not FD — branch issue
            classification = "F) FINITE DIFFERENCE BRANCH ISSUE"
    elif context_p32["residual_inf"] < 1e-6 and _metrics(prod_ad, ift_root_grad)["relative"] < 0.1:
        classification = "C) AD_P32 AND FD_P32 AGREE, BUT IFT_ROOT DIFFERENCE IS DUE TO FINITE ITERATION"
    else:
        classification = "D) PRODUCTION AD INCORRECT"

    print(f"  {classification}")

    # ── Assemble report ──────────────────────────────────────────────────
    report = {
        "case": asdict(case),
        "classification": classification,
        "map_definitions": {
            "production_ad": "objective(simulate_step): 32 unrolled Picard iterations under jax.checkpoint",
            "unrolled_ad": "objective(reconstruct_from_stages(production_stages)): same initial guess and 32 Picard iterations",
            "ift_p32": "IFT evaluated at P32 stages (NOT a converged root reference)",
            "ift_root": "IFT evaluated at genuinely converged z* with ||F||_inf < tol",
            "finite_difference": "central difference of objective(simulate_step), i.e. the finite production iteration map",
        },
        "state_identity": state_identity,
        "stage_residual_p32": {"l2": float(context_p32["residual_l2"]), "inf": float(context_p32["residual_inf"])},
        "gradient_metrics": {
            "production_unrolled": production_unrolled,
            "production_ift_root": _metrics(prod_ad, ift_root_grad),
            "fd_sweep": fd_sweep_metrics,
            "production_ad_values": [float(v) for v in prod_ad],
            "unrolled_ad_values": [float(v) for v in unrolled_ad],
            "ift_root_values": [float(v) for v in ift_root_grad],
        },
        "intermediate_fd_vs_unrolled_ad": vector_metrics,
        "objective_hierarchy": hierarchy,
        "mechanical_energy_fd_sweep": energy_fd_sweep,
        "converged_root": {
            "convergence_table": convergence_table,
            "selected_root_tolerance": float(z_star_tol),
            "selected_root_iterations": int(z_star_info["iterations"]),
            "selected_root_residual_inf": float(z_star_info["final_residual_inf"]),
            "selected_root_residual_l2": float(z_star_info["final_residual_l2"]),
            "z_p32_vs_z_star_l2": z_star_delta,
            "z_p32_vs_z_star_inf": z_star_delta_inf,
            "converged_residual_inf": float(context_star["residual_inf"]),
            "ift_identity_l2": float(context_star["ift_identity_l2"]),
        },
        "iteration_convergence": iter_rows,
        "conditioning_p32": cond_p32,
        "conditioning_star": cond_star,
        "parameter_table": param_table,
        "branch_audit": branches,
    }
    return report


def write_report(report, path: Path | None = None):
    path = path or ROOT / "results" / "full_vehicle_gradient_audit.json"
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path


if __name__ == "__main__":
    result = audit_one_step()
    destination = write_report(result)
    print(f"\nReport written to {destination}")
    print(json.dumps({
        "classification": result["classification"],
        "state_identity": result["state_identity"],
        "gradient_metrics": {
            "production_unrolled": result["gradient_metrics"]["production_unrolled"],
            "production_ift_root": result["gradient_metrics"]["production_ift_root"],
        },
        "converged_root": result["converged_root"],
        "conditioning_p32": result["conditioning_p32"],
    }, indent=2))
