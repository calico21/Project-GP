# optimization/risk_averse_objective.py
# Project-GP — Stochastic Driveability Objective
# ═══════════════════════════════════════════════════════════════════════════
#
# J_RA(setup) = E_ε[T] + β₁·Std[T] + β₂·CVaR_α(T) + β₃·softplus(ExKurt[T])
#
# where T = LTE score under driver noise ε_steer ~ N(0,σ_δ²), ε_F ~ N(0,σ_F²).
#
# Var_ε[T] ≈ σ² Tr(∂²T/∂u²) [first-order Taylor] — the Monte Carlo variance
# is a free Hutchinson estimator of the control-Hessian trace. No Jacobian
# computation required.
#
# CVaR: Rockafellar-Uryasev formula with smoothed relu (softplus):
#   CVaR_α ≈ c + E[softplus(β(T-c))] / (β(1-α))
# where c = VaR proxy (stop_gradient). Gradient flows through the expectation.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from optimization.objectives import _CURV_PROFILE, _VTGT_PROFILE, _N_STEPS_LTE


class DriveabilityParams(NamedTuple):
    n_mc:          int   = 32        # MC samples — 32 sufficient for gradient, 256 for reporting
    sigma_steer:   float = 0.008     # rad RMS  ≈ 0.46° — measured from ADAC driver study
    sigma_throttle: float = 0.025    # fraction of K_speed — typical hall-effect pedal noise
    beta_std:      float = 0.40      # variance aversion (0 = robot driver, 1 = max conservatism)
    beta_cvar:     float = 0.30      # tail-risk (85th-percentile conditional expectation)
    alpha_cvar:    float = 0.85      # CVaR percentile
    cvar_temp:     float = 20.0      # softplus temperature for CVaR relu approximation
    beta_kurt:     float = 0.05      # excess-kurtosis penalty (fat-tail / outlier crash events)


@partial(jax.jit, static_argnums=(0,))
def compute_driveability_objective(
    sim_fn,
    setup:   jax.Array,             # (28,)
    x_init:  jax.Array,             # (N_state,)
    key:     jax.Array,
    params:  DriveabilityParams = DriveabilityParams(),
    dt:      float = 0.005,
    T_opt:   float = 90.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns (J_nominal, J_risk_averse, diagnostics[mean,std,cvar,kurt]).

    J_nominal      — zero-noise score (for Pareto archive quality)
    J_risk_averse  — what the optimizer should actually minimize
    """
    K_steer  = 1.0
    K_speed  = 4000.0
    K_brake  = 8000.0
    L_wb     = 1.55    # lf + lr [m]
    curv     = _CURV_PROFILE
    v_tgt    = _VTGT_PROFILE

    def single_rollout(noise_key: jax.Array) -> jax.Array:
        k1, k2 = jax.random.split(noise_key)
        eps_s  = jax.random.normal(k1, (_N_STEPS_LTE,)) * params.sigma_steer
        eps_t  = jax.random.normal(k2, (_N_STEPS_LTE,)) * params.sigma_throttle

        def step(x, inp):
            idx, es, et = inp[0].astype(jnp.int32), inp[1], inp[2]
            vx  = jnp.maximum(x[14], 1.0)
            # Perturbed controls: additive steer noise, multiplicative throttle noise
            delta = K_steer * curv[idx] * L_wb + es
            v_err = v_tgt[idx] - vx
            F = jnp.clip(
                jax.nn.softplus( v_err) * K_speed * (1.0 + et)
              - jax.nn.softplus(-v_err) * K_speed,
                -K_brake, 6000.0,
            )
            xn   = sim_fn(x, jnp.array([delta, F]), setup, dt)
            pwr  = jnp.abs(F * vx)
            T_sf = jnp.max(xn[28:31])   # FL surface ribs (108-DOF layout)
            return xn, (vx, pwr, T_sf)

        scanned_inputs = (
            jnp.arange(_N_STEPS_LTE, dtype=jnp.float32),
            eps_s, eps_t,
        )
        _, (vx_h, pw_h, T_h) = jax.lax.scan(step, x_init, scanned_inputs)

        d        = jnp.maximum(jnp.sum(vx_h) * dt, 1.0)
        ep_m     = jnp.sum(pw_h) * dt / d
        T_pen    = jax.nn.softplus((T_h[-1] - T_opt - 15.0) * 0.2) / 30.0
        # Negative: we minimize (lower LTE score = worse)
        return -(jnp.mean(vx_h) / 15.0
                 + 0.3 * (200.0 / jnp.maximum(ep_m, 10.0))
                 - 0.5 * T_pen)

    keys    = jax.random.split(key, params.n_mc)
    samples = jax.vmap(single_rollout)(keys)          # (n_mc,) — compiled as batched kernel

    # ── Risk statistics ────────────────────────────────────────────────────
    mu  = jnp.mean(samples)
    std = jnp.std(samples)

    # VaR proxy: stop_gradient keeps c fixed during backprop so ∂CVaR/∂setup
    # only flows through the expectation term (correct Rockafellar-Uryasev deriv)
    c    = jax.lax.stop_gradient(mu + 0.5 * std)
    cvar = c + jnp.mean(
        jax.nn.softplus((samples - c) * params.cvar_temp)
    ) / (params.cvar_temp * (1.0 - params.alpha_cvar))

    # Excess kurtosis: fat tails → outlier crash events; penalise positive values only
    m4   = jnp.mean((samples - mu) ** 4)
    kurt = jax.nn.softplus(m4 / (std ** 4 + 1e-8) - 3.0)

    J_RA = mu + params.beta_std * std + params.beta_cvar * cvar + params.beta_kurt * kurt

    # Zero-noise nominal (deterministic reference)
    J_nominal = single_rollout(jax.random.PRNGKey(0))

    return J_nominal, J_RA, jnp.array([mu, std, cvar, kurt])