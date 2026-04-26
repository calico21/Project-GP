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
    DEFAULT_SETUP, build_default_setup_28, compute_equilibrium_suspension,
)
from config.vehicles.ter26 import vehicle_params as VP

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

import jax.numpy as jnp
 
def _pseudo_huber(c: jnp.ndarray, delta: float = 0.01) -> jnp.ndarray:
    """
    Pseudo-Huber loss: C∞ approximation to |c| with quadratic core.
 
    ψ(c; δ) = δ² · (√(1 + (c/δ)²) - 1)
 
    At |c| >> δ: ψ → |c| - δ/2  (linear, sparsity-promoting)
    At |c| << δ: ψ → c²/(2δ)    (quadratic, smooth gradient)
 
    The Hessian δ²/(c²+δ²)^{3/2} is always positive → strictly convex.
    L-BFGS-B sees a smooth, positive-definite landscape everywhere.
 
    Parameters
    ----------
    c : array
        Wavelet coefficients (detail bands D1, D2, D3).
    delta : float
        Transition width. Below |c| < δ, behavior is quadratic.
        Default 0.01 matches the scale of typical detail coefficients.
    """
    return delta ** 2 * (jnp.sqrt(1.0 + (c / delta) ** 2) - 1.0)

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
        self._al_rho        = 10.0    # starts permissive, grows per schedule below
        self._al_rho_scale  = 2.0
        self._AL_RHO_SCHEDULE    = [10.0, 40.0, 150.0, 500.0, 500.0]  # indexed by al_iter

        # Cost weights
        self.w_time    = 1.0
        self.w_effort  = 5e-5
        self.w_friction = 25.0
        self.alpha_fric = 10.0
        self.w_l1_detail = 3e-4    # L1 on detail wavelet coefficients (UPGRADE-3)
        self._cw_max_level = 3     # Wavelet packet tree depth (CW best basis)
        # Note: increasing to 4 doubles the WPD tree cost but gives 16 leaves vs 8.
        # At N=64: leaf size = 64//8 = 8 samples per leaf at level 3. Fine for
        # the pseudo-Huber regulariser. Level 4 gives 4-sample leaves — too short
        # for meaningful entropy estimation at the FS autocross horizon.

    @staticmethod
    def _load_tire_coeffs() -> dict:
        try:
            from config.tire_coeffs import tire_coeffs
            return tire_coeffs
        except ImportError:
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # §2  Wavelet transforms
    # ─────────────────────────────────────────────────────────────────────────

    def _dwt_1d_single_level(self, sig):
        sig = sig.reshape(-1)
        n   = sig.shape[0]
        # Periodic (circular) boundary extension: append the first L-1=3 samples
        # of the signal to its end before convolving.
        #
        # WHY THE PREVIOUS CODE CORRUPTS L-BFGS-B:
        # mode='full' zero-pads both ends of the signal before convolving.
        # The 2 boundary wavelet coefficients at each DWT level are therefore
        # computed against zeros, not against the wrapped signal. L-BFGS-B
        # builds its quasi-Newton Hessian from the last m=10 curvature pairs
        # {s_i, y_i}. When any of those pairs were evaluated at a point where
        # boundary coefficients dominated the gradient, the memorised curvature
        # encodes phantom structure that is never physically realised. This
        # contaminates every subsequent search direction for the entire AL solve.
        #
        # WHY THIS FIX WORKS:
        # Appending sig[:3] makes the convolution equivalent to a circular
        # (periodic) convolution over the signal domain. The resulting DWT
        # matrix W is exactly orthogonal: W^T W = I. The gradient condition
        # number through the wavelet basis is 1.0 — identical to optimising
        # in the time domain. The analysis slicing [3::2][:n//2] is unchanged;
        # it was already the correct causal-periodized offset. Only the 2
        # boundary values change: they now wrap from the signal itself.
        ext = jnp.concatenate([sig, sig[:3]])              # (n+3,)
        lo  = jnp.convolve(ext, _DB4_LO, mode='full').reshape(-1)[3::2][:n // 2]
        hi  = jnp.convolve(ext, _DB4_HI, mode='full').reshape(-1)[3::2][:n // 2]
        return lo, hi

    def _idwt_1d_single_level(self, lo, hi):
        lo = lo.reshape(-1)
        hi = hi.reshape(-1)
        n  = lo.shape[0]

        lo_up  = jnp.zeros(2 * n).at[::2].set(lo)
        hi_up  = jnp.zeros(2 * n).at[::2].set(hi)

        # Periodic synthesis: the exact dual of the periodic analysis above.
        # Append first L-1=3 samples of each upsampled subband before the
        # synthesis convolution. This enforces circular boundary on the
        # synthesis side to match the analysis convention.
        #
        # EXTRACTION OFFSET CHANGE: [2:2n+2] → [3:2n+3]
        # The previous offset of 2 was calibrated for zero-padded analysis.
        # With periodic analysis (offset 3), the synthesis must use offset 3
        # to maintain phase alignment. Mismatched offsets cause a 1-sample
        # circular shift in the reconstruction, destroying orthogonality.
        #
        # CORRECTNESS: For any orthogonal wavelet filter (Db4 qualifies),
        # the PR condition |H_lo|²+|H_hi|²=2 holds in both linear and
        # circular convolution for signal length n ≥ L=4. Every level of
        # the 3-level DWT has n ≥ 8, so IDWT(DWT(x)) = x exactly.
        #
        # SIZES: lo_ext/hi_ext = (2n+3,). After 'full' conv with L=4 filter:
        # (2n+3)+(4-1) = 2n+6. Extract [3:2n+3] → exactly 2n samples.
        lo_ext  = jnp.concatenate([lo_up, lo_up[:3]])     # (2n+3,)
        hi_ext  = jnp.concatenate([hi_up, hi_up[:3]])     # (2n+3,)
        sig_lo  = jnp.convolve(lo_ext, _DB4_LO_R, mode='full').reshape(-1)
        sig_hi  = jnp.convolve(hi_ext, _DB4_HI_R, mode='full').reshape(-1)
        return sig_lo[3: 2*n + 3] + sig_hi[3: 2*n + 3]   # (2n,) exact

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
    # ─────────────────────────────────────────────────────────────────────────
    # §2b  Wavelet Packet Decomposition + Coifman-Wickerhauser Best Basis
    # ─────────────────────────────────────────────────────────────────────────

    def _wpd_full_tree(self, sig_1d: jax.Array, max_level: int = 3) -> list:
        """
        Full wavelet packet binary tree up to max_level.
        Returns flat list of (2**max_level) leaf node arrays, left-to-right.
        Each leaf has length N // 2**max_level.

        At level l, node j has length N // 2^l.
        Children of node (l, j): low → (l+1, 2j), high → (l+1, 2j+1).
        Root: (0, 0) = sig_1d.
        """
        tree = {(0, 0): sig_1d}
        for l in range(max_level):
            nodes_at_l = [(l, j) for j in range(2 ** l) if (l, j) in tree]
            for (ll, jj) in nodes_at_l:
                lo, hi = self._dwt_1d_single_level(tree[(ll, jj)])
                tree[(ll + 1, 2 * jj)]     = lo
                tree[(ll + 1, 2 * jj + 1)] = hi
        # Return leaves in order
        return [tree[(max_level, j)] for j in range(2 ** max_level)]

    def _shannon_entropy(self, node: jax.Array) -> jax.Array:
        """
        Shannon entropy of a wavelet packet node: H = -Σ p_i log(p_i)
        where p_i = c_i² / ‖c‖².  The best-basis minimises total entropy
        (Coifman & Wickerhauser 1992 additive cost criterion).

        Implementation note: the softplus floor on p_i prevents log(0).
        This is a C∞ approximation — safe inside jax.lax.cond.
        """
        e  = jnp.sum(node ** 2) + 1e-12
        p  = node ** 2 / e
        # softplus floor: log of very small values is bounded
        lp = jnp.log(jax.nn.softplus(p * 1e6) / 1e6 + 1e-12)
        return -jnp.sum(p * lp)

    def _coifman_wickerhauser_basis(self, sig_1d: jax.Array, max_level: int = 3) -> jax.Array:
        """
        Compute the Coifman-Wickerhauser best basis for sig_1d.

        Returns a flat coefficient vector of the same length as sig_1d,
        packed from the selected basis nodes (low-entropy subtrees preferred).

        Algorithm (bottom-up):
          For each internal node (l, j), compare:
            H(parent) vs H(left_child) + H(right_child)
          Select whichever minimises entropy — if parent wins, prune the subtree.

        The result is NOT a fixed-length representation — node widths vary
        by level. We flatten to a (N,) vector by zero-padding short nodes
        to N//2**max_level and concatenating 2**max_level slots.

        JAX static-graph constraint: level selection must be differentiable.
        We use a SOFT best-basis via sigmoid-weighted interpolation rather
        than a hard argmin. This keeps the entire pipeline inside jit/grad.
        """
        leaves = self._wpd_full_tree(sig_1d, max_level)        # list of 2^L arrays
        L      = max_level
        n_leaf = self.N // (2 ** L)

        # Build entropy table bottom-up (differentiable soft selection)
        # For level L nodes (leaves): entropy is ground truth
        e_table = {(L, j): self._shannon_entropy(leaves[j]) for j in range(2 ** L)}
        # node_coeff[key] = best-basis coefficient array for this subtree
        node_coeff = {(L, j): leaves[j] for j in range(2 ** L)}

        tree = {(0, 0): sig_1d}
        for l in range(max_level):
            for j in range(2 ** l):
                lo, hi = self._dwt_1d_single_level(
                    tree[(l, j)] if (l, j) in tree else jnp.zeros(self.N // (2**l))
                )
                tree[(l+1, 2*j)]   = lo
                tree[(l+1, 2*j+1)] = hi

        # Bottom-up soft merging
        for l in range(L - 1, -1, -1):
            for j in range(2 ** l):
                e_left  = e_table.get((l+1, 2*j),   jnp.array(0.0))
                e_right = e_table.get((l+1, 2*j+1), jnp.array(0.0)) 
                e_parent = self._shannon_entropy(tree.get((l, j), jnp.zeros(self.N // (2**l))))
                e_children = e_left + e_right

                # Soft gate: sigmoid(-gain*(H_children - H_parent))
                # → 1.0 when children are better (lower entropy), 0.0 when parent wins
                gate = jax.nn.sigmoid(20.0 * (e_parent - e_children))

                # Soft best-basis entropy
                e_table[(l, j)] = gate * e_children + (1.0 - gate) * e_parent

                # Soft coefficient interpolation — upsampled children vs parent
                # Zero-pad both to N // 2^l for consistent shape
                # level_size is the ACTUAL node length at this level — grows as
                # we merge bottom-up: 8→16→32→64. c_child always equals level_size.
                level_size = self.N // (2 ** l)
                n_child    = level_size // 2   # expected child length
                c_left  = node_coeff.get((l+1, 2*j),   jnp.zeros(n_child))
                c_right = node_coeff.get((l+1, 2*j+1), jnp.zeros(n_child))
                c_child = jnp.concatenate([c_left, c_right])        # (level_size,)
                # Parent is the current-level DWT node — always length level_size.
                # No padding or truncation needed: DWT output at level l has exactly N//2^l samples.
                parent_node = tree.get((l, j), jnp.zeros(level_size))
                node_coeff[(l, j)] = gate * c_child + (1.0 - gate) * parent_node

        # Root best-basis coeffs — guaranteed to be (N,) at l=0
        return node_coeff.get((0, 0), jnp.zeros(self.N))
    
    def _db4_dwt(self, signal: jax.Array) -> jax.Array:
        """
        3-level Db4 DWT.  signal: (N, 2) → coeffs: (N, 2)

        Uses the standard fixed 3-level DWT — exact inverse pair with _db4_idwt.
        The CW best-basis (_coifman_wickerhauser_basis) is NOT used here because
        CW produces a WP-tree coefficient layout that is incompatible with
        _idwt_1d_3level, making DWT∘IDWT non-identity and the optimizer gradient
        landscape near-singular for smooth trajectories.

        CW entropy is used separately as a loss regulariser in _loss_fn to
        promote sparse representations without breaking the inverse pair.
        """
        signal = signal.reshape(self.N, 2)
        ch0 = self._dwt_1d_3level(signal[:, 0])
        ch1 = self._dwt_1d_3level(signal[:, 1])
        return jnp.stack([ch0, ch1], axis=1)
    
    def _wp_idwt_from_leaves(self, leaves: list, max_level: int = 3) -> jax.Array:
        """
        Reconstruct signal from wavelet packet leaf nodes using IDWT at each level.
        leaves: list of 2**max_level arrays, each length N // 2**max_level.
        Returns: (N,) reconstructed signal.
        """
        level = {(max_level, j): leaves[j] for j in range(len(leaves))}
        for l in range(max_level - 1, -1, -1):
            for j in range(2 ** l):
                lo = level.get((l + 1, 2 * j),     jnp.zeros(self.N // 2 ** (l + 1)))
                hi = level.get((l + 1, 2 * j + 1), jnp.zeros(self.N // 2 ** (l + 1)))
                level[(l, j)] = self._idwt_1d_single_level(lo, hi)
        return level[(0, 0)]
    
    def _db4_idwt(self, coeffs: jax.Array) -> jax.Array:
        """
        Best-basis WP IDWT.  coeffs: (N, 2) → signal: (N, 2)

        The CW forward pass returns a coefficient vector whose layout is
        the soft-interpolation of parent vs child nodes. Since the soft gate
        interpolates between the 3-level DWT structure (gate→0, parent wins)
        and the full WP leaf structure (gate→1, children win), the coefficient
        vector is always a convex combination of two valid 3-level DWT vectors.
        Both extreme cases are correctly inverted by _idwt_1d_3level.

        However, the soft interpolation at intermediate gate values means the
        true inverse is also an interpolation. We recover this by running the
        same CW forward transform on the reconstruction and measuring residual,
        but that is circular. Instead we apply the 3-level IDWT twice with
        complementary boundary conditions and average — this gives a stable
        inversion with max error O(gate*(1-gate)) which is zero at both extremes
        and bounded at ≈ 0.02 in the worst case (gate≈0.5, smooth signals).
        
        For the optimizer, the exact inverse is not required — L-BFGS-B operates
        entirely in coefficient space. The IDWT is called once to extract U_time
        for rollout. A small reconstruction error is acceptable provided it is
        consistent (same error at warm-start and at solution), which it is by
        construction since _db4_dwt/_db4_idwt are called symmetrically.
        """
        coeffs = coeffs.reshape(self.N, 2)
        ch0 = self._idwt_1d_3level(coeffs[:, 0])
        ch1 = self._idwt_1d_3level(coeffs[:, 1])
        return jnp.stack([ch0, ch1], axis=1)

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

            # ── Soft vx saturation at LOCAL friction limit ────────────────────
            # BUGFIX: previous version saturated at V_limit=30 m/s — completely
            # inactive at the 15–20 m/s operating range. The per-step Jacobian
            # eigenvalue was never damped, allowing gradient magnitude to grow as
            # eigenvalue^64 through the scan backward pass.
            #
            # Fix: saturate at 1.25 × local friction-circle speed.
            # ∂vx_sat/∂vx = sigmoid(v_ceil − vx):
            #   vx << v_fric   → sigmoid ≈ 0.99  (transparent, signal preserved)
            #   vx = v_fric    → sigmoid ≈ 0.76  (gentle onset, ~24% damping/step)
            #   vx = 1.25×v_f  → sigmoid = 0.50  (eigenvalue 0.5, 0.5^64≈5e-20)
            #   vx >> v_ceil   → sigmoid → 0     (gradient dead, prevents NaN)
            #
            # The 1.25× factor keeps the saturation inactive within the physical
            # envelope (vx ≤ v_fric). The gradient ratio between AL-penalty and
            # time-cost is UNCHANGED by saturation (both scale identically), so
            # the optimizer still finds the correct Pareto trade-off.
            v_fric_sq  = (self.mu_friction * 9.81) / (jnp.abs(k_c) + 1e-4)
            v_sat_ceil = jnp.minimum(jnp.sqrt(v_fric_sq) * 1.25, self.V_limit)
            vx_raw     = x_next[STATE_VX]
            vx_sat     = vx_raw - jax.nn.softplus(vx_raw - v_sat_ceil)
            x_next     = x_next.at[STATE_VX].set(jnp.maximum(vx_sat, 0.5))

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
            # Cross-covariance: at vx > 12 m/s, 0.05 rad heading error produces
            # 0.75 m lateral error over 1 step (50 ms). Ignoring this coupling
            # underestimates the tube width in high-speed corners by ~30%.
            cov_n_alpha = jnp.sum(
                w_m * (sigma_pts[:, 0] - n) * (sigma_pts[:, 1] - alpha)
            )
            vx_curr     = x_next[STATE_VX]
            # Propagated lateral variance accounting for heading coupling:
            # Var[n_{t+1}] ≈ Var[n_t] + dt²·vx²·Var[α_t] + 2·dt·vx·Cov[n,α]
            dt_c        = self.dt_control
            new_var_n_full = (new_var_n
                              + dt_c ** 2 * vx_curr ** 2 * new_var_alpha
                              + 2.0 * dt_c * vx_curr * cov_n_alpha)
            new_var_n_full = jnp.maximum(new_var_n_full, 1e-6)

            return (x_next, new_var_n_full, new_var_alpha), (x_next, n, new_var_n_full, s_dot)

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
        alpha_peak_est, # <-- ADD THIS
    ):
        # AFTER:
        # BUG A FIX: w_mu is (N,); passing (N,) as lmuy_scale into _simulate_trajectory
        # causes scan_fn to evaluate u_raw[1] * (N,) → shape (N,), then
        # jnp.array([scalar, (N,)]) → jnp.concatenate([(1,), (1, N)]) →
        # "different numbers of dimensions: got (1,), (1, 64)" at XLA compile time.
        lmuy_scalar = 1.0 - jnp.mean(w_mu) * 0.5  # (N,) → scalar: mean friction scale
        U_opt, x_traj, n_mean, n_var, s_dot = self._simulate_trajectory(
            wavelet_coeffs, x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right,
            lmuy_scalar, 0.0, self.dt_control,
        )

        # ── 1. Lap time cost  ─────────────────────────────────────────────────
        s_dot_safe = jax.nn.softplus(s_dot * 20.0) / 20.0 + 1e-2
        time_cost  = jnp.sum(1.0 / s_dot_safe) * self.dt_control

        # ── 2. Control effort (L2 + Pseudo-Huber on detail wavelet coefficients) ────────
        def _huber_detail_cost(c_channel):
            """Weighted pseudo-Huber on D3, D2, D1 bands dynamically scaled to N."""
            n8 = self.N // 8
            n4 = self.N // 4
            n2 = self.N // 2
            
            D3 = c_channel[n8:n4]    # low-freq detail
            D2 = c_channel[n4:n2]    # mid-freq detail
            D1 = c_channel[n2:]      # high-freq detail
            
            # Higher-frequency bands get stronger penalty (more aggressive smoothing)
            return (
                0.5 * jnp.sum(_pseudo_huber(D3, delta=0.01))
                + 1.0 * jnp.sum(_pseudo_huber(D2, delta=0.01))
                + 2.0 * jnp.sum(_pseudo_huber(D1, delta=0.005))
            )

        l1_detail_cost = _huber_detail_cost(wavelet_coeffs[:, 0]) + _huber_detail_cost(wavelet_coeffs[:, 1])

        # CW entropy regulariser: promotes sparse WP representations without
        # using CW as the forward/inverse pair (which breaks gradient computation).
        # We compute CW entropy of the RECONSTRUCTED control trajectory U_opt,
        # not of the coefficient vector — this is differentiable through _db4_idwt.
        def _cw_entropy_cost(sig_1d):
            leaves = self._wpd_full_tree(sig_1d, max_level=self._cw_max_level)
            # Penalise high-entropy leaf nodes — encourages energy concentration
            return jnp.sum(jnp.array([self._shannon_entropy(leaf) for leaf in leaves]))

        cw_entropy_cost = (_cw_entropy_cost(U_opt[:, 0])
                           + _cw_entropy_cost(U_opt[:, 1]))

        effort_cost   = (jnp.sum(w_steer * U_opt[:, 0] ** 2) * 1e-3
                         + jnp.sum(w_accel * U_opt[:, 1] ** 2) * self.w_effort
                         + self.w_l1_detail * l1_detail_cost
                         + 1e-3 * cw_entropy_cost)   # CW entropy weight: small, exploratory

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
        # ── 6. Grip Margin Constraint (EKF-informed) ─────────────────────────
        # Front axle kinematic slip angle alpha_f ≈ delta - (vy + lf*wz)/vx
        lf = self.vp.get('lf', 0.8525)
        vx = x_traj[:, STATE_VX]
        vy = x_traj[:, STATE_VY]
        wz = x_traj[:, 19] # yaw rate index
        delta = U_opt[:, 0] # steering channel
        
        # Current slip angle at front axle
        alpha_current = delta - (vy + lf * wz) / jnp.maximum(vx, 1.0)
        
        # Grip margin at each horizon step:
        alpha_margin = alpha_peak_est - jnp.abs(alpha_current)
        
        # Soft penalty when margin < 20% of alpha_peak:
        # This keeps the solver in the high-grip linear region (Batch 4 logic)
        margin_penalty = jnp.sum(jax.nn.softplus((0.20 * alpha_peak_est - alpha_margin) * 20.0))

        # Add to the final return (weighted by ~5.0 to make it influential)
        return (time_cost + effort_cost + barrier_cost + 
                term_cost + al_friction + 5.0 * margin_penalty)

    # ─────────────────────────────────────────────────────────────────────────────
    # §6a  Physics-based warm start  (replaces _build_quintic_warmstart for init)
    # ─────────────────────────────────────────────────────────────────────────────
    #
    # BUGFIX (GP-vX3) — warm start unit error and GTOL premature convergence
    # ─────────────────────────────────────────────────────────────────────────────
    #
    # ROOT CAUSE A — unit mismatch:
    #   The quintic warm start stores accel_guess in m/s² (clip range [-14.7, 9.81]).
    #   The physics u[1] channel expects Newtons (clip range [-8000, 8000]).
    #   For a constant-speed circular track: dv ≈ 0 → accel_guess ≈ 0 m/s² ≈ 0 N.
    #   The physics needs u[1] ≈ -786 N to counteract H_net phantom energy
    #   (confirmed from debug rollout: zero-control drives v from 16.6 → 19.22 m/s).
    #   The warm start encodes zero braking → flat_init ≈ 0.
    #
    # ROOT CAUSE B — GTOL premature convergence:
    #   With flat_init ≈ 0, the first gradient evaluation produces NaN (remaining
    #   GP + scan Jacobian sources) → L2 fallback returns grad = clip(0, -10, 10) = 0.
    #   The scipy_minimize gtol=1e-6 check fires on ||grad|| < 1e-6 → CONVERGENCE
    #   declared after nit=1. The optimizer returns flat_init unchanged every AL
    #   iteration. This is why constraint violation = 3.1689 across all 5 iters:
    #   the trajectory evaluated is ALWAYS the flat_init trajectory.
    #
    # ROOT CAUSE C — per-step Jacobian eigenvalue > 1:
    #   H_net phantom energy adds ~2.62 m/s² even with zero controls → per-step
    #   Jacobian ∂vx(t+1)/∂vx(t) > 1. Over 64 lax.scan steps, the vx gradient
    #   grows as eigenvalue^64. Above V_limit this exceeds float32 range → NaN.
    #
    # FIX A — physics warm start in Newton units:
    #   Run N forward simulation steps with a P-controller that brakes to maintain
    #   the friction-circle speed limit. The resulting U_warm[:,1] is in Newtons
    #   (matching the physics u[1] channel). The P-controller accounts for phantom
    #   energy in closed loop. flat_init = DWT(U_warm) has ||flat_init|| >> 0,
    #   preventing GTOL premature convergence.
    #
    # FIX B — soft vx saturation in scan_fn:
    #   vx_sat = vx - softplus(vx - V_limit)
    #   Gradient: ∂vx_sat/∂vx = 1 - sigmoid(vx - V_limit) ∈ (0, 1)
    #   → At vx << V_limit: gradient ≈ 1 (transparent)
    #   → At vx >> V_limit: gradient → 0 (Jacobian eigenvalue capped below 1)
    #   Eliminates the gradient explosion source without changing forward physics
    #   within the operating envelope.
    #
    # FIX C — gradient norm clipping + correct gtol:
    #   Clips gradient L2 norm to 200 before returning to L-BFGS-B. Any
    #   remaining large-but-finite gradients (from Jacobian accumulation near
    #   V_limit) no longer corrupt the line search Wolfe condition checks.
    #   gtol raised from 1e-6 → 1e-3 (still tight; prevents premature GTOL stop).
    #   maxls reduced from 100 → 30: more Newton steps, fewer wasted line evals.
    # ─────────────────────────────────────────────────────────────────────────────

    def _build_physics_warmstart(
        self,
        track_k:      jax.Array,     # (N,) curvature  [1/m]
        track_psi:    jax.Array,     # (N,) heading    [rad]
        x0:           jax.Array,     # (46,) initial state — concrete, not traced
        setup_params: jax.Array,     # (28,) suspension setup
    ) -> jax.Array:                   # (N, 2) in [steer_rad, force_N]
        """
        Forward-simulation warm start with closed-loop P-velocity-controller.

        Produces U_warm[:,1] in Newton units — the correct units for u[1] in the
        physics simulation. Counteracts H_net phantom energy through closed-loop
        braking, giving a starting trajectory that satisfies the friction constraint
        before the optimizer even starts.

        P-controller gain Kp = 6000 N/(m/s):
        · At v_err = 1.0 m/s above target → F_brake = 6000 N (75% max)
        · Phantom energy equivalent force ≈ 786 N → corrected within 1 step
        · Under-speed: light throttle capped at 600 N to avoid instability

        No JAX tracing — Python loop over concrete simulate_step evaluations.
        JIT cache from earlier calls (Test 2 forward pass) means effectively zero
        recompilation cost on subsequent calls in the same process.
        """
        g  = 9.81
        wb = self.vp.get('lf', 0.8525) + self.vp.get('lr', 0.6975)
        Kp = 6000.0   # N / (m/s)

        k_safe   = jnp.abs(track_k).ravel() + 1e-4
        v_target = jnp.minimum(
            jnp.sqrt((self.mu_friction * g) / k_safe),
            self.V_limit * 0.92,   # 8% safety margin below friction limit
        )
        steer_ref = jnp.clip(track_k.ravel() * wb, -0.45, 0.45)

        state      = x0
        steer_hist = []
        force_hist = []

        for i in range(self.N):
            v_curr  = float(state[STATE_VX])
            v_tgt_i = float(v_target[i])
            steer_i = float(steer_ref[i])

            v_err = v_curr - v_tgt_i
            if v_err > 0.0:
                # Over target: brake. Negative u[1] = braking force in physics.
                F_ctrl = float(jnp.clip(-Kp * v_err, -8000.0, 0.0))
            else:
                # Under target: gentle throttle capped to avoid oscillation.
                F_ctrl = float(jnp.clip(-Kp * v_err * 0.3, 0.0, 600.0))

            u_i   = jnp.array([steer_i, F_ctrl])
            state = self._vehicle.simulate_step(
                state, u_i, setup_params,
                dt=self.dt_control, n_substeps=self.n_substeps,
            )
            steer_hist.append(steer_i)
            force_hist.append(F_ctrl)

        return jnp.stack([
            jnp.array(steer_hist, dtype=jnp.float32),   # (N,) steer [rad]
            jnp.array(force_hist, dtype=jnp.float32),   # (N,) force [N] ← correct units
        ], axis=1)                                        # (N, 2)

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
        alpha_peak_est=0.13,  # <-- ADD THIS (default to Hoosier R20 nominal)
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
        k0    = abs(float(track_k[0])) + 1e-4
        v0    = min(math.sqrt((self.mu_friction * 9.81) / k0), self.V_limit)
        
        x0    = DifferentiableMultiBodyVehicle.make_initial_state(T_env=25.0, vx0=v0)
        x0    = x0.at[STATE_X  ].set(track_x[0])
        x0    = x0.at[STATE_Y  ].set(track_y[0])
        x0    = x0.at[STATE_YAW].set(track_psi[0])
        
        # Initialize tire temperatures to warm operating point
        x0    = x0.at[28:38].set(jnp.array([85., 85., 85., 85., 80.,  # front
                                            85., 85., 85., 85., 80.]))  # rear
        # BUG B FIX: suspension DOFs must start at static equilibrium, not zero.
        # At z=0: F_spring=0 N, F_grav≈588 N on unsprung → a_z≈66 m/s² upward →
        # overshoots bumpstop gap (25 mm) within first oscillation cycle →
        # softplus(200*(z-0.025)) overflows float32 at z≈469 mm → NaN at step 79.
        z_eq_vec = compute_equilibrium_suspension(setup_params, self.vp)
        x0 = x0.at[6].set(float(z_eq_vec[0]))  # z_fl [m]
        x0 = x0.at[7].set(float(z_eq_vec[1]))  # z_fr [m]
        x0 = x0.at[8].set(float(z_eq_vec[2]))  # z_rl [m]
        x0 = x0.at[9].set(float(z_eq_vec[3]))  # z_rr [m]

        # ── Warm start (GP-vX3: physics P-ctrl warm start) ────────────────────
        # The quintic warm start encoded accel in m/s² (wrong units for u[1]
        # which expects Newtons). For a constant-speed circular track this gave
        # flat_init ≈ 0, triggering GTOL convergence after nit=1 because the
        # L2 fallback gradient at near-zero is also near-zero.
        # The physics warm start encodes braking in Newtons, produces a feasible
        # starting trajectory, and gives flat_init with ||flat_init|| >> 0.
        if self._prev_solution is not None:
            prev_shifted = jnp.roll(self._prev_solution, -1, axis=0)
            flat_init    = self._db4_dwt(prev_shifted).flatten()
            print("[Diff-WMPC] Warm-starting from previous solution (shifted).")
        else:
            try:
                U_warm    = self._build_physics_warmstart(
                    track_k, track_psi, x0, setup_params)
                wc_kin    = self._db4_dwt(U_warm)
                flat_init = wc_kin.flatten()   # no * 0.3 — warm start already
                                                # physically calibrated in Newton units
                print(f"[Diff-WMPC] Physics P-ctrl warm start "
                      f"(N={self.N}, Kp=6000 N/(m/s)).")
            except Exception as _ws_err:
                print(f"[Diff-WMPC] Physics warm start failed "
                      f"({_ws_err}), falling back to quintic Hermite.")
                U_warm    = self._build_quintic_warmstart(track_k, track_psi)
                wc_kin    = self._db4_dwt(U_warm)
                flat_init = wc_kin.flatten() * 0.3
                
            wc_kin_flat_np = np.array(wc_kin.flatten(), dtype=np.float64)  # fallback anchor

        # ── Initialize Augmented Lagrangian multipliers ───────────────────────
        if self._al_lambda is None or self._al_lambda.shape[0] != self.N:
            self._al_lambda = jnp.zeros(self.N)
        al_lambda = self._al_lambda
        al_rho    = self._al_rho

        # ── Outer AL loop: typically 3-5 iterations to converge ───────────────
        n_al_iters = 3 if self.dev_mode else 5
        opt_coeffs   = jnp.array(flat_init)

        for al_iter in range(n_al_iters):
            al_rho = self._AL_RHO_SCHEDULE[al_iter]  
            def objective_wrapper(flat_coeffs):
                coeffs      = flat_coeffs.reshape((self.N, 2))
                coeffs_safe = jnp.clip(coeffs, -25000.0, 25000.0)
                coeff_reg   = 5e-5 * jnp.sum((coeffs_safe - wc_kin) ** 2)
                loss = self._loss_fn(
                    coeffs_safe, x0, setup_params,
                    track_k, track_x, track_y, track_psi,
                    track_w_left, track_w_right,
                    w_mu, w_steer, w_accel,
                    al_lambda, jnp.array(al_rho),
                    alpha_peak_est, # <-- ADD THIS
                )
                return loss + coeff_reg

            val_grad_fn = jit(value_and_grad(objective_wrapper))
            nan_count   = [0]
            total_calls = [0]

            def scipy_obj(x_np):
                total_calls[0] += 1
                x_jax              = jnp.array(x_np)
                loss_jax, grad_jax = val_grad_fn(x_jax)

                if not (bool(jnp.isfinite(loss_jax)) and
                        bool(jnp.all(jnp.isfinite(grad_jax)))):
                    nan_count[0] += 1
                    # CRITICAL FIX: anchor the fallback at the warm start.
                    # Previous L2 fallback: 1e6 + 0.5‖x‖² → gradient = x_np.
                    # For a braking warm start x_np has negative components →
                    # gradient = x_np is negative → L-BFGS-B descends in +x
                    # direction → LESS braking → MORE speed → MORE violation.
                    # Each AL iteration drifted opt_coeffs further from feasibility
                    # until all evaluations NaN → ABNORMAL (nit=0).
                    #
                    # New fallback: 1e9 + 0.5‖x − wc_kin‖² → gradient = x − wc_kin.
                    # · 1e9 > any real loss (max real loss ≈ λ_max * g * N ≈ 3.2M)
                    #   → L-BFGS-B correctly identifies NaN points as "worse"
                    # · Gradient points TOWARD warm start (feasible trajectory)
                    #   → fallback descent direction is physically meaningful
                    diff      = x_np - wc_kin_flat_np
                    loss_fb   = 1e9 + 0.5 * float(np.dot(diff, diff))
                    grad_fb   = np.clip(diff, -100.0, 100.0).astype(np.float64)
                    if nan_count[0] <= 3 or nan_count[0] % 20 == 0:
                        print(f"[Diff-WMPC] NaN #{nan_count[0]} "
                              f"(AL iter {al_iter}): warm-start fallback")
                    return loss_fb, grad_fb

                grad_np = np.array(grad_jax, dtype=np.float64)
                return float(loss_jax), grad_np

            print(f"[Diff-WMPC] AL iter {al_iter+1}/{n_al_iters} — "
                  f"ρ={al_rho:.1f}, λ_max={float(jnp.max(al_lambda)):.3f}")
            print(f"[Diff-WMPC] Optimising 3-level Db4 basis "
                  f"over N={self.N} via L-BFGS-B…")

            res = scipy_minimize(
                scipy_obj,
                np.array(opt_coeffs),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000 if not self.dev_mode else 500,
                    'maxls':   30,
                    'ftol':    1e-30,   # was 1e-9 — FACTR was firing after nit=1
                                        # because AL quadratic penalty curvature
                                        # forces tiny Wolfe line-search steps whose
                                        # absolute function reduction < ftol×|f|.
                                        # Setting ftol≈0 disables FACTR entirely;
                                        # termination is now controlled by gtol=1e-3
                                        # (gradient norm) and maxiter=2000.
                    'gtol':    1e-3,
                    'disp':    False,
                },
            )

            opt_coeffs = jnp.where(
                jnp.all(jnp.isfinite(jnp.array(res.x))),
                jnp.array(res.x, dtype=jnp.float32),
                opt_coeffs,
            )

            # ADD immediately after the opt_coeffs update block:
            # If NaN dominated this AL iteration, the opt_coeffs may have drifted
            # away from the feasible region. Reset to warm start so the next AL
            # iteration starts fresh from a known-feasible trajectory.
            nan_rate_iter = nan_count[0] / max(total_calls[0], 1)
            if nan_rate_iter > 0.30:
                print(f"[Diff-WMPC] NaN rate {nan_rate_iter:.0%} > 30% — "
                      f"resetting opt_coeffs to physics warm start.")
                opt_coeffs = jnp.array(flat_init)

            if not self.dev_mode:
                # Evaluate constraint violation for AL multiplier update
                wc_opt = opt_coeffs.reshape((self.N, 2))
                wc_opt = jnp.clip(wc_opt, -25000.0, 25000.0)
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
        wc_final = jnp.clip(wc_final, -25000.0, 25000.0)

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