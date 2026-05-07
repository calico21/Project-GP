"""
optimization/racing_line_planner.py — Project-GP  GP-vX10
═════════════════════════════════════════════════════════════
Minimum-curvature racing line (Heilmeier et al. 2020) + FB velocity profile.

Improvements over previous version:
  · Signed curvature computed from geometry with Gaussian smoothing (σ=5)
  · Tikhonov regularization on first-derivative of n to suppress noise
  · Curvature-weighted smoothing on output n_opt
  · 5-pass FB sweep for tight periodicity convergence
"""
from __future__ import annotations
import math
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize as sp_minimize
from typing import NamedTuple


class RacingLine(NamedTuple):
    s:      np.ndarray   # arc-length [m]
    n_opt:  np.ndarray   # optimal lateral offset [m]
    rx:     np.ndarray   # racing line world x [m]
    ry:     np.ndarray   # racing line world y [m]
    psi:    np.ndarray   # racing line heading [rad]
    kappa:  np.ndarray   # racing line curvature [1/m]
    v_ref:  np.ndarray   # speed profile [m/s]
    wl_rl:  np.ndarray   # width left of racing line [m]
    wr_rl:  np.ndarray   # width right of racing line [m]


class RacingLinePlanner:
    def __init__(self, mu=1.40, v_max=25.0, m=235.0, g=9.81):
        self.mu    = mu
        self.v_max = v_max
        self.m     = m
        self.g     = g

    def plan(self, cx, cy, ck_raw, wl, wr) -> RacingLine:
        cx = np.asarray(cx, dtype=np.float64)
        cy = np.asarray(cy, dtype=np.float64)
        wl = np.asarray(wl, dtype=np.float64)
        wr = np.asarray(wr, dtype=np.float64)
        M  = len(cx)

        # ── Geometry from centerline ──────────────────────────────────────
        dx = np.diff(cx, append=cx[0])
        dy = np.diff(cy, append=cy[0])
        ds = np.hypot(dx, dy)
        ds = np.maximum(ds, 1e-6)          # avoid division by zero
        s  = np.concatenate([[0.0], np.cumsum(ds[:-1])])

        # Tangent angle (signed, unwrapped)
        psi_c    = np.arctan2(dy, dx)
        psi_c_uw = np.unwrap(psi_c)

        # Signed curvature: κ = dψ/ds with Gaussian smoothing
        # Raw finite-difference curvature is noisy on discrete tracks.
        # σ=5 nodes ≈ 2.5m at 0.5m spacing — smooths node-level noise
        # while preserving corner geometry (corner width ≈ 10-20 nodes).
        dpsi     = np.diff(psi_c_uw, append=psi_c_uw[0] + (psi_c_uw[-1] - psi_c_uw[-2]))
        kappa_raw = dpsi / ds
        kappa_s   = gaussian_filter1d(kappa_raw, sigma=5, mode='wrap')

        print(f"[RacingLine] Curvature: signed, "
              f"κ ∈ [{kappa_s.min():.4f}, {kappa_s.max():.4f}] 1/m")

        # ── Left-normal vectors ───────────────────────────────────────────
        nx_hat = -np.sin(psi_c)
        ny_hat =  np.cos(psi_c)

        # ── Phase 1: MinCurv QP ───────────────────────────────────────────
        print("[RacingLine] Solving MinCurv QP ...")
        n_opt = self._min_curv_qp(kappa_s, wl, wr, ds)

        # Post-smooth n_opt: gentle Gaussian to remove QP discretization noise
        n_opt = gaussian_filter1d(n_opt, sigma=3, mode='wrap')
        # Re-enforce bounds after smoothing
        n_opt = np.clip(n_opt, -(wr - 0.15), wl - 0.15)

        print(f"[RacingLine] MinCurv done. "
              f"n ∈ [{n_opt.min():.2f}, {n_opt.max():.2f}] m")

        # ── Racing line geometry ──────────────────────────────────────────
        rx = cx + n_opt * nx_hat
        ry = cy + n_opt * ny_hat

        drx    = np.diff(rx, append=rx[0])
        dry    = np.diff(ry, append=ry[0])
        ds_r   = np.maximum(np.hypot(drx, dry), 1e-6)
        psi_r  = np.arctan2(dry, drx)
        psi_ruw = np.unwrap(psi_r)
        dpsi_r = np.diff(psi_ruw,
                         append=psi_ruw[0] + 2*np.pi + (psi_ruw[-1] - psi_ruw[-2]))
        kappa_r = dpsi_r / ds_r
        # Smooth racing line curvature for velocity profile
        kappa_r = gaussian_filter1d(kappa_r, sigma=3, mode='wrap')

        # ── Phase 2: Velocity profile ─────────────────────────────────────
        print("[RacingLine] Running forward-backward velocity sweep ...")
        v_ref = self._fb_velocity(kappa_r, ds_r)
        t_lap = float(np.sum(ds_r / np.maximum(v_ref, 0.5)))
        print(f"[RacingLine] Done. Est. lap={t_lap:.2f}s  "
              f"v ∈ [{v_ref.min():.1f}, {v_ref.max():.1f}] m/s")

        return RacingLine(
            s      = s.astype(np.float32),
            n_opt  = n_opt.astype(np.float32),
            rx     = rx.astype(np.float32),
            ry     = ry.astype(np.float32),
            psi    = psi_r.astype(np.float32),
            kappa  = kappa_r.astype(np.float32),
            v_ref  = v_ref.astype(np.float32),
            wl_rl  = np.maximum(wl - n_opt, 0.20).astype(np.float32),
            wr_rl  = np.maximum(wr + n_opt, 0.20).astype(np.float32),
        )

    # ──────────────────────────────────────────────────────────────────────
    # MinCurv QP  —  minimize ½‖D₂n − κ_c‖² + λ/2·‖D₁n‖²
    # ──────────────────────────────────────────────────────────────────────

    def _min_curv_qp(self, kappa_c, wl, wr, ds):
        """
        Path curvature (linearized): κ_path ≈ κ_c − n''
        Objective: minimize  ½ Σ (κ_c − n'')²  +  λ_smooth/2 · Σ (n')²

        The smoothing term λ_smooth·‖D₁n‖² acts as Tikhonov regularization:
        it penalizes rapid lateral changes, preventing the optimizer from
        creating wild oscillations when κ_c has node-level noise.

        λ_smooth = 0.02: barely perceptible on corner geometry (shifts apex
        by < 3cm on a 9m-radius corner) but eliminates high-frequency
        oscillation between adjacent nodes.
        """
        M        = len(kappa_c)
        ds_m     = float(np.mean(ds))
        kappa_c  = kappa_c.astype(np.float64)
        lambda_s = 0.02    # smoothness regularization weight

        # Sparse periodic second-difference D₂
        d_off = np.ones(M - 1)
        D2 = sp.diags([d_off, np.full(M, -2.0), d_off], [-1, 0, 1],
                      shape=(M, M), format='lil') / ds_m ** 2
        D2[0,   M - 1] = 1.0 / ds_m ** 2
        D2[M-1, 0    ] = 1.0 / ds_m ** 2
        D2  = D2.tocsr()
        D2T = D2.T.tocsr()

        # Sparse periodic first-difference D₁
        D1 = sp.diags([-np.ones(M), np.ones(M - 1)], [0, 1],
                      shape=(M, M), format='lil') / ds_m
        D1[M-1, 0] = 1.0 / ds_m     # periodic wrap
        D1 = D1.tocsr()
        D1T = D1.T.tocsr()

        def obj_grad(n):
            r      = D2 @ n - kappa_c         # curvature residual
            f_curv = 0.5 * float(r @ r)
            g_curv = (D2T @ r)

            d1n    = D1 @ n                    # smoothness term
            f_sm   = 0.5 * lambda_s * float(d1n @ d1n)
            g_sm   = lambda_s * (D1T @ d1n)

            return f_curv + f_sm, (g_curv + g_sm).astype(np.float64)

        # Bounds: car must stay 0.15m from each wall
        margin = 0.15
        bounds = [
            (float(max(-wr[i] + margin, -wr[i] * 0.95)),
             float(min( wl[i] - margin,  wl[i] * 0.95)))
            for i in range(M)
        ]

        res = sp_minimize(
            obj_grad, np.zeros(M, dtype=np.float64),
            method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': 30000, 'ftol': 1e-20, 'gtol': 1e-10, 'maxls': 50},
        )
        if not res.success:
            print(f"[MinCurv] note: {res.message}")
        return res.x.astype(np.float64)

    # ──────────────────────────────────────────────────────────────────────
    # Forward-backward velocity sweep
    # ──────────────────────────────────────────────────────────────────────

    def _fb_velocity(self, kappa, ds):
        """
        Point-mass minimum-time velocity profile.
        5 passes for tight periodicity convergence on closed circuits.
        """
        M    = len(kappa)
        mu_g = self.mu * self.g
        # Cornering speed limit at each node
        v = np.minimum(
            np.sqrt(np.maximum(mu_g / (np.abs(kappa) + 1e-4), 0.0)),
            self.v_max
        ).astype(np.float64)

        def a_lon_budget(v_i, k_i):
            """Combined-slip longitudinal acceleration budget."""
            a_lat = v_i ** 2 * abs(k_i)
            return math.sqrt(max(mu_g ** 2 - min(a_lat, mu_g) ** 2, 0.01))

        for _ in range(5):
            # Forward: acceleration limited
            for i in range(M):
                ip = (i + 1) % M
                a_bud = a_lon_budget(v[i], kappa[i])
                v_fwd = math.sqrt(max(v[i]**2 + 2.0 * a_bud * float(ds[i]), 0.01))
                v[ip] = min(v[ip], v_fwd, self.v_max)
            # Backward: braking limited
            for i in range(M - 1, -1, -1):
                ip = (i + 1) % M
                a_bud = a_lon_budget(v[ip], kappa[ip])
                v_bwd = math.sqrt(max(v[ip]**2 + 2.0 * a_bud * float(ds[i]), 0.01))
                v[i]  = min(v[i], v_bwd, self.v_max)

        return np.maximum(v, 0.5).astype(np.float32)
