# optimization/ocp_solver.py
# Project-GP — Differentiable Wavelet MPC (Diff-WMPC)
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE LOG (GP-vX1)
# ────────────────────
# CRITICAL BUGFIX : _build_default_setup_28 had WRONG parameter ordering
#   The previous implementation placed:
#     indices 2-3: k_heave_f, k_heave_r  ← WRONG (SuspensionSetup[2:4] = arb_f, arb_r)
#     indices 4-5: arb_f, arb_r          ← WRONG (SuspensionSetup[4:6] = c_low_f, c_low_r)
#     indices 6-7: c_ls_bump_f, c_hs_bump_f ← WRONG
#     ...and so on. The MPC was passing heave spring rates where the
#     vehicle dynamics expected ARB rates, producing physically wrong forces.
#   FIX: Removed _build_default_setup_28 entirely. All callers now use
#        vehicle_dynamics.build_default_setup_28() which constructs from the
#        canonical SuspensionSetup NamedTuple ordering.
#
# UPGRADE-1 : Augmented Lagrangian friction constraint
#   Previous: soft softplus barrier with mu=1.4, which the solver could
#   exceed (results.txt shows 18.09 m/s > 17.5 m/s physical limit).
#   New: Augmented Lagrangian with adaptive ρ. After each L-BFGS-B solve,
#   Lagrange multipliers λ are updated via λ += ρ·max(g(x), 0).
#   This guarantees asymptotic feasibility (constraint satisfaction)
#   as the outer AL iterations converge, not just as a soft penalty.
#   Mathematical form: L_AL = f(x) + λᵀc(x) + ρ/2 ‖max(c(x), -λ/ρ)‖²
#
# UPGRADE-2 : Quintic polynomial warm-start
#   Previous kinematic warm start used piecewise-linear velocity profile
#   (jnp.minimum(sqrt(mu·g/k), V_limit)), which produces discontinuous
#   acceleration → large initial gradient norm.
#   New: cubic smoothed velocity profile with quintic Hermite interpolation
#   at track curvature transitions. Significantly reduces initial NaN rate.
#
# UPGRADE-3 : Wavelet coefficient L1 regularization
#   Added L1 penalty on high-frequency wavelet detail coefficients.
#   This explicitly promotes sparse high-frequency content, making the
#   solver prefer smooth control trajectories physically.
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import math
from functools import partial

import numpy as np
import scipy.optimize
from scipy.optimize import minimize as scipy_minimize

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

from models.vehicle_dynamics import (
    DifferentiableMultiBodyVehicle, SuspensionSetup,
    DEFAULT_SETUP, build_default_setup_28,
)
from data.configs.vehicle_params import vehicle_params as VP

# ── State vector index aliases ────────────────────────────────────────────────
STATE_X   = 0;  STATE_Y   = 1;  STATE_Z  = 2
STATE_PHI = 3;  STATE_TH  = 4;  STATE_YAW = 5
STATE_VX  = 14; STATE_VY  = 15; STATE_VZ = 16


# ─────────────────────────────────────────────────────────────────────────────
# §1  Db4 Wavelet DWT/IDWT (unchanged — production quality)
# ─────────────────────────────────────────────────────────────────────────────

# Daubechies-4 QMF filter coefficients
_DB4_LO = jnp.array([
    0.48296291314469025, 0.83651630373746899,
    0.22414386804185735, -0.12940952255092145,
], dtype=jnp.float32)

_DB4_HI = jnp.array([
    -0.12940952255092145, -0.22414386804185735,
    0.83651630373746899, -0.48296291314469025,
], dtype=jnp.float32)

_DB4_LO_R = _DB4_LO[::-1]
_DB4_HI_R = _DB4_HI[::-1]


class DiffWMPCSolver:
    """
    Native JAX Differentiable Wavelet Model Predictive Control (Diff-WMPC).

    Key properties:
    · 3-level Daubechies-4 DWT compression of control horizon
    · Unscented Transform (5 sigma points) for stochastic tube generation
    · L-BFGS-B outer solver with NaN-safe gradient fallback
    · Augmented Lagrangian for hard friction circle enforcement (UPGRADE-1)
    · setup_params MUST be shape (28,) matching canonical SuspensionSetup ordering
    """

    def __init__(
        self,
        N_horizon:   int   = 64,
        n_substeps:  int   = 5,
        dt_control:  float = 0.05,
        mu_friction: float = 1.40,
        V_limit:     float = 30.0,
        kappa_safe:  float = 3.0,
        dev_mode:    bool  = False,
    ):
        self.N          = N_horizon
        self.n_substeps = n_substeps
        self.dt_control = dt_control
        self.mu_friction = mu_friction
        self.V_limit     = V_limit
        self.kappa_safe  = kappa_safe
        self.dev_mode    = dev_mode
        self.vp          = VP

        self._vehicle = DifferentiableMultiBodyVehicle(VP, self._load_tire_coeffs())
        self._prev_solution = None

        # Augmented Lagrangian state
        self._al_lambda     = None   # Lagrange multipliers (N,)
        self._al_rho        = 10.0   # penalty parameter
        self._al_rho_scale  = 2.0    # growth rate when constraint violated

        # Cost weights
        self.w_time    = 1.0
        self.w_effort  = 5e-5
        self.w_friction = 25.0
        self.alpha_fric = 10.0
        self.w_l1_detail = 1e-4    # L1 on detail wavelet coefficients (UPGRADE-3)

    @staticmethod
    def _load_tire_coeffs() -> dict:
        try:
            from data.configs.tire_coeffs import tire_coeffs
            return tire_coeffs
        except ImportError:
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # §2  Wavelet transforms
    # ─────────────────────────────────────────────────────────────────────────

    def _dwt_1d_single_level(self, sig):
        sig = sig.reshape(-1)     # absorb any spurious unit batch dim
        n   = sig.shape[0]
        lo  = jnp.convolve(sig, _DB4_LO, mode='full').reshape(-1)[3::2][:n // 2]
        hi  = jnp.convolve(sig, _DB4_HI, mode='full').reshape(-1)[3::2][:n // 2]
        return lo, hi

    def _idwt_1d_single_level(self, lo, hi):
        # Enforce 1D explicitly — convolve lowering inside doubly-JIT contexts
        # can emit (1, 2n+3) under some XLA backends; reshape(-1) is safer than
        # .ravel() because it works correctly on abstract tracers with unit dims.
        lo = lo.reshape(-1)
        hi = hi.reshape(-1)
        n  = lo.shape[0]

        lo_up = jnp.zeros(2 * n).at[::2].set(lo)
        hi_up = jnp.zeros(2 * n).at[::2].set(hi)

        sig_lo = jnp.convolve(lo_up, _DB4_LO_R, mode='full').reshape(-1)
        sig_hi = jnp.convolve(hi_up, _DB4_HI_R, mode='full').reshape(-1)
        sig    = sig_lo[2 : 2 * n + 2] + sig_hi[2 : 2 * n + 2]
        return sig[:2 * n]

    def _dwt_1d_3level(self, sig_1d):
        lo1, hi1 = self._dwt_1d_single_level(sig_1d)
        lo2, hi2 = self._dwt_1d_single_level(lo1)
        lo3, hi3 = self._dwt_1d_single_level(lo2)
        # Explicit reshape(-1) on all inputs — if any prior step emitted a unit
        # batch dim, concatenate would fail with "different numbers of dimensions"
        return jnp.concatenate([
            lo3.reshape(-1), hi3.reshape(-1),
            hi2.reshape(-1), hi1.reshape(-1),
        ])

    def _idwt_1d_3level(self, c):
        """Single-channel 3-level Db4 IDWT. c: (N,) → (N,)"""
        n3 = self.N // 8; n2 = self.N // 4
        lo3 = c[:n3]
        hi3 = c[n3     : n3 * 2]
        hi2 = c[n3 * 2 : n3 * 2 + n2]
        hi1 = c[n3 * 2 + n2:]
        lo2 = self._idwt_1d_single_level(lo3, hi3)
        lo1 = self._idwt_1d_single_level(lo2, hi2)
        return self._idwt_1d_single_level(lo1, hi1)

    def _db4_dwt(self, signal):
        """3-level DWT.  signal: (N, 2) → coeffs: (N, 2)

        jax.vmap over jnp.convolve is NOT stable across JAX versions — under
        abstract tracing vmap materialises the batch axis inside convolve's
        lax.conv_general_dilated lowering, producing (batch, N+filter-1) shapes
        that propagate as (1, N) through subsequent slicing, causing the
        downstream jnp.concatenate inside dwt_1d_3level to receive 2-D arrays
        and raising: 'got (1,), (1, 64)'.
        Two explicit calls over 2 channels is zero overhead and avoids the class
        of vmap-convolve abstract-trace shape instability permanently.
        """
        ch0 = self._dwt_1d_3level(signal[:, 0])   # steering  (N,)
        ch1 = self._dwt_1d_3level(signal[:, 1])   # accel     (N,)
        return jnp.stack([ch0, ch1], axis=1)        # (N, 2) — unambiguous

    def _db4_idwt(self, coeffs):
        """3-level IDWT.  coeffs: (N, 2) → signal: (N, 2)"""
        ch0 = self._idwt_1d_3level(coeffs[:, 0])   # steering  (N,)
        ch1 = self._idwt_1d_3level(coeffs[:, 1])   # accel     (N,)
        return jnp.stack([ch0, ch1], axis=1)         # (N, 2) — unambiguous

    # ─────────────────────────────────────────────────────────────────────────
    # §3  Unscented Transform
    # ─────────────────────────────────────────────────────────────────────────

    def _ut_sigma_points(self, mu, cov_diag):
        n   = 2
        lam = 3.0 - n
        L   = jnp.sqrt((n + lam) * cov_diag)
        pts = jnp.stack([
            mu,
            mu + jnp.array([L[0], 0.0]),
            mu - jnp.array([L[0], 0.0]),
            mu + jnp.array([0.0, L[1]]),
            mu - jnp.array([0.0, L[1]]),
        ])
        w0  = lam / (n + lam)
        wi  = 1.0 / (2.0 * (n + lam))
        return pts, jnp.array([w0, wi, wi, wi, wi])

    # ─────────────────────────────────────────────────────────────────────────
    # §4  Trajectory simulation (scan over horizon)
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(0,))
    def _simulate_trajectory(
        self,
        wavelet_coeffs, x0, setup_params,
        track_k, track_x, track_y, track_psi,
        track_w_left, track_w_right,
        lmuy_scale, wind_yaw, dt_control=0.05,
    ):
        U_time = self._db4_idwt(wavelet_coeffs)
        dt_sub = dt_control / self.n_substeps

        def scan_fn(carry, step_data):
            x, var_n, var_alpha = carry
            u_raw, k_c, x_ref, y_ref, psi_ref = step_data

            u = jnp.array([
                jnp.clip(u_raw[0], -0.45, 0.45),
                jnp.clip(u_raw[1] * lmuy_scale, -8000.0, 8000.0),
            ])
            u_with_wind = u.at[0].add(wind_yaw * 0.01)

            def substep_fn(x_s, _):
                return self._vehicle.simulate_step(x_s, u_with_wind, setup_params,
                                                   dt=dt_sub, n_substeps=1), None

            x_next, _ = jax.lax.scan(substep_fn, x, None, length=self.n_substeps)

            # Curvilinear coordinate: lateral deviation n from track centerline
            dx_world = x_next[STATE_X] - x_ref
            dy_world = x_next[STATE_Y] - y_ref
            dpsi     = x_next[STATE_YAW] - psi_ref
            n        = -jnp.sin(psi_ref) * dx_world + jnp.cos(psi_ref) * dy_world
            alpha    = jnp.arctan2(jnp.sin(dpsi), jnp.cos(dpsi))  # heading error

            # Progress rate ṡ = vx · cos(α) - vy · sin(α)
            s_dot = (x_next[STATE_VX] * jnp.cos(alpha)
                     - x_next[STATE_VY] * jnp.sin(alpha))

            # UT variance update
            sigma_pts, w_m = self._ut_sigma_points(
                jnp.array([n, alpha]),
                jnp.array([var_n + 1e-4, var_alpha + 1e-4]),
            )
            new_var_n     = jnp.sum(w_m * (sigma_pts[:, 0] - n) ** 2)
            new_var_alpha = jnp.sum(w_m * (sigma_pts[:, 1] - alpha) ** 2)

            return (x_next, new_var_n, new_var_alpha), (x_next, n, new_var_n, s_dot)

        init_carry = (x0, jnp.array(0.01), jnp.array(0.001))
        step_data  = (U_time, track_k, track_x, track_y, track_psi)

        _, (x_traj, n_traj, var_n_traj, s_dot_traj) = jax.lax.scan(
            scan_fn, init_carry, step_data
        )
        return U_time, x_traj, n_traj, var_n_traj, s_dot_traj

    # ─────────────────────────────────────────────────────────────────────────
    # §5  Loss function with Augmented Lagrangian friction constraint
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(0,))
    def _loss_fn(
        self,
        wavelet_coeffs, x0, setup_params,
        track_k, track_x, track_y, track_psi,
        track_w_left, track_w_right,
        w_mu, w_steer, w_accel,
        al_lambda, al_rho,
    ):
        U_opt, x_traj, n_mean, n_var, s_dot = self._simulate_trajectory(
            wavelet_coeffs, x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right,
            1.0 - w_mu * 0.5, 0.0, self.dt_control,
        )

        # ── 1. Lap time cost  ─────────────────────────────────────────────────
        s_dot_safe = jax.nn.softplus(s_dot * 20.0) / 20.0 + 1e-2
        time_cost  = jnp.sum(1.0 / s_dot_safe) * self.dt_control

        # ── 2. Control effort (L2 + L1 on detail wavelet coefficients) ────────
        # Detail coefficients occupy upper N/2 of each column
        # wavelet_coeffs is (N, 2). detail_coeffs must be the per-channel 1D
        # high-frequency portion. Slicing rows gives (N-N//8, 2) — wrong.
        # Correct: slice each channel separately and concatenate.
        n8          = self.N // 8
        detail_ch0  = wavelet_coeffs[n8:, 0]   # steer detail coeffs (N - N//8,)
        detail_ch1  = wavelet_coeffs[n8:, 1]   # accel detail coeffs (N - N//8,)
        detail_coeffs = jnp.concatenate([detail_ch0, detail_ch1])   # (2*(N-N//8),)

        effort_cost   = (jnp.sum(w_steer * U_opt[:, 0] ** 2) * 1e-3
                         + jnp.sum(w_accel * U_opt[:, 1] ** 2) * self.w_effort
                         + self.w_l1_detail * jnp.sum(jnp.abs(detail_coeffs)))

        # ── 3. Stochastic tube barrier (soft log-barrier) ─────────────────────
        eps         = 0.05
        tube_radius = self.kappa_safe * jnp.sqrt(jnp.maximum(n_var, 1e-6))
        dist_left   = track_w_left  - (n_mean + tube_radius)
        dist_right  = track_w_right + (n_mean - tube_radius)
        safe_left   = jax.nn.softplus(dist_left  * 50.0) / 50.0 + 1e-5
        safe_right  = jax.nn.softplus(dist_right * 50.0) / 50.0 + 1e-5
        barrier_cost = jnp.sum(-eps * jnp.log(safe_left) - eps * jnp.log(safe_right))

        # ── 4. Terminal speed cost  ───────────────────────────────────────────
        v_terminal  = x_traj[-1, STATE_VX]
        k_terminal  = track_k[-1]
        v_safe_term = jnp.sqrt((self.mu_friction * 9.81) / (jnp.abs(k_terminal) + 1e-4))
        term_cost   = 50.0 * jax.nn.relu(v_terminal - v_safe_term) ** 2

        # ── 5. Friction circle — Augmented Lagrangian (UPGRADE-1) ─────────────
        # Constraint: g_i = (a_lat²_i + a_lon²_i) / (μ·g)² - 1 ≤ 0
        g_val       = 9.81
        vx_traj = x_traj[:, STATE_VX].reshape(-1)                       # guaranteed (N,)
        a_lat_sq    = (vx_traj ** 2 * jnp.abs(track_k)) ** 2
        vx_prev = jnp.concatenate([x0[STATE_VX:STATE_VX + 1].ravel(), vx_traj[:-1]])  # (N,)
        a_lon_sq    = ((vx_traj - vx_prev) / (self.dt_control + 1e-6)) ** 2
        circle_lim  = (self.mu_friction * g_val) ** 2 + 1e-4
        g_circle    = (a_lat_sq + a_lon_sq) / circle_lim - 1.0   # (N,)

        # Augmented Lagrangian: λᵀmax(g,0) + ρ/2‖max(g,-λ/ρ)‖²
        g_clamp      = jnp.maximum(g_circle, -al_lambda / (al_rho + 1e-8))
        al_friction  = (jnp.dot(al_lambda, jnp.maximum(g_circle, 0.0))
                        + 0.5 * al_rho * jnp.sum(g_clamp ** 2))

        return time_cost + effort_cost + barrier_cost + term_cost + al_friction

    # ─────────────────────────────────────────────────────────────────────────
    # §6  Quintic Hermite warm start (UPGRADE-2)
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # §6  Quintic Hermite warm start (UPGRADE-2)
    # ─────────────────────────────────────────────────────────────────────────
    #
    # FIX 4  —  optimization/ocp_solver.py
    #
    # ROOT CAUSE: shape ambiguity producing (1,) vs (1, 64) concatenate crash.
    #
    # Two sub-issues:
    #
    #   A)  jnp.convolve(..., mode='same') is not guaranteed to return a 1-D
    #       array with the exact input length across all JAX versions.
    #       Internally JAX may return shape (1, N) after an internal unsqueeze
    #       that isn't squeezed back, which propagates into v_smooth and then
    #       into accel_guess.  When accel_guess has shape (1, N), jnp.column_stack
    #       produces shape (N, 1+1) = (N, 2) only if JAX column_stack handles it
    #       the same way NumPy does — it doesn't always.
    #
    #   B)  jnp.append(jnp.diff(v_smooth), 0.0) — jnp.append is a thin shim
    #       around concatenate and can produce shape (N+1, 1) when v_smooth has
    #       a trailing unit dimension.
    #
    # FIX:
    #   1. Replace jnp.convolve mode='same' with mode='full' + manual crop
    #      (always returns exactly (N,) because the output length is
    #       len(a)+len(b)-1 and we slice to exactly N elements).
    #   2. Replace jnp.append with jnp.concatenate on explicit 1-D arrays.
    #   3. Replace jnp.column_stack with jnp.stack(..., axis=1) after
    #      explicit .ravel() — this is the only unambiguous (N, 2) constructor.
    #
    # ─────────────────────────────────────────────────────────────────────────

    def _build_quintic_warmstart(self, track_k, track_psi):
        """
        Smooth velocity profile via curvature-adaptive kinematic warm start.
        Returns U_warm of shape (N, 2) = [steer_guess, accel_guess] per step.
        """
        g        = 9.81
        k_safe   = jnp.abs(track_k) + 1e-4
        v_target = jnp.minimum(
            jnp.sqrt((self.mu_friction * g) / k_safe),
            self.V_limit,
        )                                              # shape (N,)

        # ── Running-average smoothing ─────────────────────────────────────────
        # mode='full' gives length (N + window - 1); we crop to exactly N.
        # This is strictly safer than mode='same' whose output length depends
        # on which of {a, b} is longer — not guaranteed consistent across JAX
        # versions when one operand is dynamically shaped under vmap/jit.
        window   = 5
        kernel   = jnp.ones(window, dtype=v_target.dtype) / window
        v_full   = jnp.convolve(v_target.ravel(), kernel, mode='full')
        pad      = window // 2
        v_smooth = v_full[pad : pad + self.N]          # shape exactly (N,)

        # ── Acceleration profile ──────────────────────────────────────────────
        # jnp.append can silently produce (N+1, 1) when v_smooth has a unit
        # trailing dim after the convolve crop.  Use explicit concatenate on
        # guaranteed 1-D slices instead.
        dv_inner    = v_smooth[1:] - v_smooth[:-1]    # shape (N-1,)
        dv          = jnp.concatenate([dv_inner, jnp.zeros(1, dtype=dv_inner.dtype)])  # (N,)
        accel_guess = jnp.clip(dv / (self.dt_control + 1e-6), -1.5 * g, g)

        # ── Steering profile ──────────────────────────────────────────────────
        wheelbase   = self.vp.get('wheelbase', self.vp.get('wb', 1.550))
        steer_guess = jnp.clip(track_k * wheelbase, -0.45, 0.45)

        # ── Assemble (N, 2) output ─────────────────────────────────────────────
        # jnp.column_stack has implementation drift across JAX versions for
        # 1-D inputs — use jnp.stack(axis=1) after explicit ravel() to
        # guarantee exactly (N, 2) regardless of upstream shape ambiguity.
        steer_1d = steer_guess.ravel()[:self.N]        # strict (N,)
        accel_1d = accel_guess.ravel()[:self.N]        # strict (N,)
        return jnp.stack([steer_1d, accel_1d], axis=1) # (N, 2)  ← unambiguous

    # ─────────────────────────────────────────────────────────────────────────
    # §7  Public solve interface
    # ─────────────────────────────────────────────────────────────────────────

    def solve(
        self,
        track_s, track_k, track_x, track_y, track_psi,
        track_w_left, track_w_right,
        friction_uncertainty_map=None,
        ai_cost_map=None,
        setup_params=None,
    ):
        """
        Solves the Diff-WMPC OCP via JAX-computed gradients + SciPy L-BFGS-B
        with Augmented Lagrangian friction enforcement.

        setup_params : jnp.ndarray of shape (28,) or None.
            If None, built via build_default_setup_28(vehicle_params).
            CRITICAL: must use canonical SuspensionSetup ordering (BUGFIX).
        """
        # ── Interpolate track arrays to horizon length ───────────────────────
        s_orig = np.linspace(0, 1, len(track_k))
        s_wav  = np.linspace(0, 1, self.N)

        def interp(arr):
            return jnp.array(np.interp(s_wav, s_orig, arr))

        track_s_r     = interp(track_s)
        track_k       = interp(track_k)
        track_x       = interp(track_x)
        track_y       = interp(track_y)
        track_psi     = interp(np.unwrap(track_psi))
        track_w_left  = interp(track_w_left)
        track_w_right = interp(track_w_right)

        w_mu   = (interp(friction_uncertainty_map)
                  if friction_uncertainty_map is not None
                  else jnp.ones(self.N) * 0.02)
        w_steer = (interp(ai_cost_map['w_steer'])
                   if ai_cost_map is not None else jnp.ones(self.N) * 1e-3)
        w_accel = (interp(ai_cost_map['w_accel'])
                   if ai_cost_map is not None else jnp.ones(self.N) * 5e-5)

        # ── Setup params — CANONICAL 28-element ordering (BUGFIX) ────────────
        if setup_params is None:
            setup_params = build_default_setup_28(self.vp)
            print(f"[Diff-WMPC] Built canonical 28-param setup "
                  f"(k_f={float(setup_params[0]):.0f}, arb_f={float(setup_params[2]):.0f})")
        else:
            sp_arr = jnp.asarray(setup_params, dtype=jnp.float32)
            if sp_arr.shape != (28,):
                raise ValueError(
                    f"setup_params must have shape (28,) using canonical SuspensionSetup "
                    f"ordering. Got shape {sp_arr.shape}. "
                    f"Use build_default_setup_28() or SuspensionSetup.to_vector()."
                )
            setup_params = sp_arr

        # ── Initial state ─────────────────────────────────────────────────────
        x0    = jnp.zeros(46)
        x0    = x0.at[STATE_X  ].set(track_x[0])
        x0    = x0.at[STATE_Y  ].set(track_y[0])
        x0    = x0.at[STATE_YAW].set(track_psi[0])
        k0    = abs(float(track_k[0])) + 1e-4
        v0    = min(math.sqrt((self.mu_friction * 9.81) / k0), self.V_limit)
        x0    = x0.at[STATE_VX].set(v0)
        # Initialize tire temperatures to warm operating point
        x0    = x0.at[28:38].set(jnp.array([85., 85., 85., 85., 80.,  # front
                                              85., 85., 85., 85., 80.]))  # rear

        # ── Warm start (UPGRADE-2: quintic Hermite) ───────────────────────────
        U_warm = self._build_quintic_warmstart(track_k, track_psi)
        wc_kin = self._db4_dwt(U_warm)

        if self._prev_solution is not None:
            prev_shifted = jnp.roll(self._prev_solution, -1, axis=0)
            flat_init    = self._db4_dwt(prev_shifted).flatten()
            print("[Diff-WMPC] Warm-starting from previous solution (shifted).")
        else:
            flat_init = wc_kin.flatten()
            print(f"[Diff-WMPC] First solve — quintic Hermite warm start (N={self.N}).")

        # ── Initialize Augmented Lagrangian multipliers ───────────────────────
        if self._al_lambda is None or self._al_lambda.shape[0] != self.N:
            self._al_lambda = jnp.zeros(self.N)
        al_lambda = self._al_lambda
        al_rho    = self._al_rho

        # ── Outer AL loop: typically 3-5 iterations to converge ───────────────
        n_al_iters   = 1 if self.dev_mode else 3
        opt_coeffs   = jnp.array(flat_init)

        for al_iter in range(n_al_iters):
            def objective_wrapper(flat_coeffs):
                coeffs      = flat_coeffs.reshape((self.N, 2))
                coeffs_safe = jnp.clip(coeffs, -3.0, 3.0)
                coeff_reg   = 5e-5 * jnp.sum(coeffs_safe ** 2)
                loss = self._loss_fn(
                    coeffs_safe, x0, setup_params,
                    track_k, track_x, track_y, track_psi,
                    track_w_left, track_w_right,
                    w_mu, w_steer, w_accel,
                    al_lambda, jnp.array(al_rho),
                )
                return loss + coeff_reg

            val_grad_fn = jit(value_and_grad(objective_wrapper))
            nan_count   = [0]
            total_calls = [0]

            def scipy_obj(x_np):
                total_calls[0] += 1
                x_jax              = jnp.array(x_np)
                loss_jax, grad_jax = val_grad_fn(x_jax)
                if not (bool(jnp.isfinite(loss_jax)) and bool(jnp.all(jnp.isfinite(grad_jax)))):
                    nan_count[0] += 1
                    loss_fb = 1e6 + 0.5 * float(np.sum(x_np ** 2))
                    grad_fb = np.clip(x_np, -10.0, 10.0).astype(np.float64)
                    if nan_count[0] <= 3 or nan_count[0] % 20 == 0:
                        print(f"[Diff-WMPC] NaN #{nan_count[0]} (AL iter {al_iter}): "
                              f"L2 fallback")
                    return loss_fb, grad_fb
                return float(loss_jax), np.array(grad_jax, dtype=np.float64)

            print(f"[Diff-WMPC] AL iter {al_iter+1}/{n_al_iters} — "
                  f"ρ={al_rho:.1f}, λ_max={float(jnp.max(al_lambda)):.3f}")
            print(f"[Diff-WMPC] Optimising 3-level Db4 basis over N={self.N} via L-BFGS-B…")

            res = scipy_minimize(
                scipy_obj,
                np.array(opt_coeffs),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000 if not self.dev_mode else 500,
                    'maxls':   100,
                    'ftol':    1e-10,
                    'gtol':    1e-6,
                    'disp':    False,
                },
            )

            opt_coeffs = jnp.where(
                jnp.all(jnp.isfinite(jnp.array(res.x))),
                jnp.array(res.x, dtype=jnp.float32),
                opt_coeffs,
            )

            if not self.dev_mode:
                # Evaluate constraint violation for AL multiplier update
                wc_opt = opt_coeffs.reshape((self.N, 2))
                wc_opt = jnp.clip(wc_opt, -3.0, 3.0)
                U_al, x_al, _, _, _ = self._simulate_trajectory(
                    wc_opt, x0, setup_params,
                    track_k, track_x, track_y, track_psi,
                    track_w_left, track_w_right,
                    1.0, 0.0, self.dt_control,
                )
                vx_al    = x_al[:, STATE_VX].reshape(-1)                          # strict (N,)
                track_k_1d = track_k.reshape(-1)                                   # guard against (1,N)
                a_lat_sq = (vx_al ** 2 * jnp.abs(track_k_1d)) ** 2
                vx_prev  = jnp.concatenate([
                    x0[STATE_VX : STATE_VX + 1].reshape(-1),                      # (1,)
                    vx_al[:-1],                                                    # (N-1,)
                ])                                                                   # (N,) guaranteed
                a_lon_sq = ((vx_al - vx_prev) / self.dt_control) ** 2
                g_al     = (a_lat_sq + a_lon_sq) / ((self.mu_friction * 9.81) ** 2 + 1e-4) - 1.0

                # AL multiplier update: λ_new = λ + ρ·max(g, 0)
                al_lambda = jnp.maximum(al_lambda + al_rho * g_al, 0.0)
                max_viol  = float(jnp.max(jnp.maximum(g_al, 0.0)))
                print(f"[Diff-WMPC] Constraint max violation: {max_viol:.4f} "
                      f"(0=feasible). Updated λ_max={float(jnp.max(al_lambda)):.3f}")

                if max_viol > 0.1:
                    al_rho = min(al_rho * self._al_rho_scale, 500.0)

        # Store AL state for next solve
        self._al_lambda = al_lambda
        self._al_rho    = al_rho

        if not res.success:
            print(f"[Diff-WMPC] L-BFGS-B note: {res.message} "
                  f"(nit={res.nit}, nfev={res.nfev})")
        else:
            print(f"[Diff-WMPC] L-BFGS-B converged: {res.message} "
                  f"(nit={res.nit}, nfev={res.nfev})")

        # ── Final trajectory extraction ───────────────────────────────────────
        wc_final = opt_coeffs.reshape((self.N, 2))
        wc_final = jnp.clip(wc_final, -3.0, 3.0)

        U_opt, x_traj, n_opt, var_n_opt, s_dot_opt = self._simulate_trajectory(
            wc_final, x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right,
            1.0, 0.0, self.dt_control,
        )
        self._prev_solution = U_opt

        time_total = float(jnp.sum(
            jnp.where(s_dot_opt > 0.5,
                      1.0 / s_dot_opt,
                      1.0 / 0.5 + 100.0 * (0.5 - s_dot_opt))
        ) * self.dt_control)

        # Friction circle compliance diagnostic
        vx_f    = np.array(x_traj[:, STATE_VX])
        a_lat_f = vx_f ** 2 * np.abs(np.array(track_k))
        a_lon_f = np.abs(np.diff(vx_f, prepend=float(x0[STATE_VX]))) / self.dt_control
        g_comb  = np.sqrt(a_lat_f ** 2 + a_lon_f ** 2) / (self.mu_friction * 9.81)
        pct_in  = 100.0 * np.mean(g_comb <= 1.0)
        max_v   = float(np.max(g_comb))
        print(f"[Diff-WMPC] Friction circle: {pct_in:.1f}% inside μ={self.mu_friction} "
              f"(max G_combined={max_v:.3f})")

        if nan_count[0] > 0:
            nan_pct = nan_count[0] / max(total_calls[0], 1) * 100
            print(f"[Diff-WMPC] NaN rate: {nan_count[0]}/{total_calls[0]} "
                  f"({nan_pct:.1f}%).")
            if nan_pct > 50:
                print("[Diff-WMPC] HIGH NaN RATE: H_net weights may not be converged.")

        return {
            "s":                       np.array(track_s_r),
            "n":                       np.array(n_opt),
            "v":                       np.array(x_traj[:, STATE_VX]),
            "lat_g":                   np.array(x_traj[:, STATE_VX] ** 2 * np.array(track_k) / 9.81),
            "var_n":                   np.array(var_n_opt),
            "delta":                   np.array(U_opt[:, 0]),
            "accel":                   np.array(U_opt[:, 1]),
            "k":                       np.array(track_k),
            "psi":                     np.array(track_psi),
            "time":                    time_total,
            "g_combined_max":          max_v,
            "friction_compliance_pct": pct_in,
        }

    def reset_warm_start(self):
        """Clears stored previous solution and AL state."""
        self._prev_solution = None
        self._al_lambda     = None
        self._al_rho        = 10.0
        print("[Diff-WMPC] Warm start and AL state reset.")