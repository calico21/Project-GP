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

# ── Device lock: must happen BEFORE any jax import ───────────────────────────
# Symptoms without this: JAX detects a GPU after the first solve, switches
# device, invalidates the XLA cache → full recompile every solve (~1000s each).
# The machine has CUDA hardware but jaxlib is CPU-only — locking prevents the
# device probe that triggers the cache miss.
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')  # no 90% GPU grab

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
# BUILD ONCE at module level — static shape, XLA constant-folds the mask
# ─────────────────────────────────────────────────────────────────────────────

# Dims where cross-step carry gradient causes float32 overflow at k=31.
# Determined empirically: all 108 dims NaN at k=31 in the full Jacobian,
# but individual per-dim scans are clean to k=64.
# The overflow is driven by the position/orientation accumulation chain:
#   X, Y accumulate at O(vx·dt) per step → Jacobian row norm grows as k
#   wz (yaw rate) couples into X,Y via rotation → eigenvalue > 1
#   Thermal states (T_tire) couple back via grip → exponential growth
# 
# Strategy: stop_gradient on POSITION + ORIENTATION + THERMAL CARRY.
# Free: vx (primary cost signal), suspension heave (spring forces matter),
#       transient slip (kappa tracking), wheel speeds.
# Stopped: absolute position (X,Y,Z — not in any cost directly),
#          orientation angles (phi,theta,psi — only heading error matters,
#          and we already clamp heading in scan_fn),
#          thermal states (slow dynamics, gradient contribution negligible
#          vs the 31-step overflow it causes),
#          damper/elastokin auxiliary states.

_N_STATE_OCP = 108   # must match state vector length

# Build boolean mask: 1.0 = stop_gradient, 0.0 = free
_SG_MASK_NP = np.zeros(_N_STATE_OCP, dtype=np.float32)

# Absolute position — not in any cost term directly
_SG_MASK_NP[0:3]   = 1.0   # X, Y, Z

# Orientation — heading error uses delta(psi), not absolute psi;
# roll/pitch not in any cost
_SG_MASK_NP[3:6]   = 1.0   # phi, theta, psi

_SG_MASK_NP[6:14]  = 1.0   # q[6:13]  suspension DOFs (discrete oscillator |λ|>1)

# Momenta corresponding to stopped positions — their Jacobian chains
# are equally explosive
_SG_MASK_NP[17:20] = 1.0   # p_phi, p_theta, p_psi (angular momenta)

_SG_MASK_NP[20:28] = 1.0   # p[20:27] suspension momenta (same oscillator)

# Thermal states — 28 nodes [28:56]; slow ODE, gradient contribution
# is negligible at the 1.5s horizon but their Jacobian rows accumulate
# multiplicatively with the position chain
_SG_MASK_NP[28:56] = 1.0   # T_tire (all 4×7 nodes)

_SG_MASK_NP[56:72] = 1.0   # slip[0:15] tire slip (stiff relaxation + force feedback)

# Damper auxiliary states [72:84] — hysteresis states, not in cost
_SG_MASK_NP[72:84] = 1.0

# Elastokinematic states [84:108] — compliance states, not in cost  
_SG_MASK_NP[84:]   = 1.0

# JAX constant — will be baked into XLA graph at first JIT
_SG_MASK: jax.Array = jnp.array(_SG_MASK_NP)
_SG_FREE: jax.Array = 1.0 - _SG_MASK


@jax.jit
def _apply_carry_stop_gradient(x: jax.Array) -> jax.Array:
    """
    Truncated BPTT carry filter.
    
    Mathematically: x_carry = sg(x)·M + x·(1−M)
    where M is the stop-gradient mask.
    
    This is NOT equivalent to zeroing gradients — it severs the
    cross-step dependency for masked dims while preserving the
    per-step emission gradient (x_next is emitted before masking,
    so cost terms that read x_next still get full gradients).
    
    The key invariant: ONLY the carry (what flows to the next scan
    step) is masked. The emitted value is unmasked. This gives us:
      - Forward pass: identical to full model (mask doesn't change values)
      - Backward pass: per-step cost gradients fully preserved
      - Cross-step accumulation: severed for overflow-prone dims
    """
    return jax.lax.stop_gradient(x) * _SG_MASK + x * _SG_FREE

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
 
def stable_softplus(x):
    # The jnp.where bypasses logaddexp for large values, 
    # preventing JAX VJP NaN explosions at extreme boundaries.
    return jnp.where(x > 10.0, x, jnp.logaddexp(0.0, x))

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
        # Augmented Lagrangian state
        self._al_lambda     = None   # Lagrange multipliers (N,)
        self._al_rho        = 0.1    # starts permissive, grows per schedule below
        self._al_rho_scale  = 2.0
        
        # Softened schedule to allow tolerance checking and avoid the gradient cliffs 
        # that were reaching 12,800x the lap-time gradient.
        self._AL_RHO_SCHEDULE = [0.1, 1.0, 10.0, 50.0, 200.0]
        # Rationale (GP-vX5 FIX): previous schedule [200,800,2000,5000,5000]
        # produced AL penalty gradient ~12800× the lap-time gradient from iter 1.
        # L-BFGS-B spent all iterations satisfying the constraint with zero
        # lap-time progress → nit=0, ABNORMAL at iter 5 (λ_max=1844, ρ=5000).
        # New schedule: gradual tightening so the optimizer finds a near-optimal
        # trajectory first, then progressively enforces the friction constraint.

        # Cost weights
        self.w_time    = 1.0
        self.w_center  = 0.005 # centreline guidance: at n=1m, cost≈0.16≈1.5×time_cost
                                # CRITICAL: must stay << time_cost. At w=2.0 it was 580×
                                # the time cost → optimizer zeroed velocity to keep n=0.
        self.w_heading = 8.0   # heading alignment cost weight.
                                # Must dominate barrier_cost (~2.5) when alpha>30°.
                                # At alpha=30° (0.52 rad): w_heading × N × 0.52² = 8×32×0.27=69.
                                # Gradient signal toward heading alignment >> all other terms.
                                # At warm-start alpha≈90° (π/2): cost=8×32×(π/2)²=1264 — 
                                # overwhelms barrier, forcing optimizer to align the car first.
        self.w_effort  = 5e-5
        self.w_friction = 25.0
        self.alpha_fric = 10.0
        self.w_l1_detail = 3e-4    # L1 on detail wavelet coefficients (UPGRADE-3)
        self._cw_max_level = 3     # Wavelet packet tree depth (CW best basis)
        # Note: increasing to 4 doubles the WPD tree cost but gives 16 leaves vs 8.
        # At N=64: leaf size = 64//8 = 8 samples per leaf at level 3. Fine for
        # the pseudo-Huber regulariser. Level 4 gives 4-sample leaves — too short
        # for meaningful entropy estimation at the FS autocross horizon.

        # ── Cache the JIT-compiled objective — compiled ONCE here, never again ──
        # ROOT CAUSE of 1000s per-solve recompilation (GP-vX8 FIX):
        # `_stable_objective` was a closure defined INSIDE `solve()`. Every call
        # to `solve()` created a NEW Python closure object → new id → JAX JIT
        # cache miss → full XLA recompilation (1179s, 521s, 1948s observed).
        #
        # Fix: define the objective as a closure over `self` ONCE in `__init__`.
        # `self` is the same Python object for the solver's entire lifetime, so
        # the closure id is stable → JAX compiles exactly once → all subsequent
        # `solve()` calls are ~5ms cache hits.
        #
        # All per-call varying inputs (track arrays, x0, etc.) are explicit JAX
        # array arguments — no captured Python variables that could change.
        self._val_grad_fn = self._build_jit_objective()
        self._jit_compiled = False  # flip to True after first block_until_ready

    def _build_jit_objective(self):
        """
        Returns jit(value_and_grad(objective)) compiled against a STABLE closure.

        The closure captures only `self` (fixed Python object) and `N` (int
        compile-time constant). Every `solve()` call passes track arrays, x0,
        setup_params etc. as explicit JAX array arguments — none are captured.

        This guarantees JAX compiles once (on first call) and produces cache
        hits (~5ms) on all subsequent solve() calls, regardless of how many
        times solve() is called or which track section is being optimised.
        """
        N        = self.N                 # compile-time constant (Python int)
        loss_fn  = self._loss_fn          # bound method — stable for solver lifetime
        w_mu_s   = jnp.ones(N, dtype=jnp.float32) * (1.0 / (1.40 * 9.81) ** 2)
        w_steer_s= jnp.ones(N, dtype=jnp.float32) * 1e-3
        w_accel_s= jnp.ones(N, dtype=jnp.float32) * 5e-5

        def _obj(flat_coeffs, wc_kin_flat, al_lambda, al_rho,
                 x0, setup_params,
                 track_k, track_x, track_y, track_psi,
                 track_wl, track_wr, alpha_peak):
            coeffs      = flat_coeffs.reshape((N, 2))
            coeffs_safe = jnp.clip(coeffs, -25000.0, 25000.0)
            wc_kin      = wc_kin_flat.reshape((N, 2))
            coeff_reg   = 5e-5 * jnp.sum((coeffs_safe - wc_kin) ** 2)
            loss = loss_fn(
                coeffs_safe, x0, setup_params,
                track_k, track_x, track_y, track_psi,
                track_wl, track_wr,
                w_mu_s, w_steer_s, w_accel_s,
                al_lambda, al_rho, alpha_peak,
            )
            return loss + coeff_reg

        # Apply buffer donation to the first argument (flat_coeffs)
        # This allows XLA to reuse the memory allocation during L-BFGS-B steps
        return jit(value_and_grad(_obj))

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
        # Periodic boundary: append the first 3 samples
        padded = jnp.concatenate([sig, sig[:3]])
        # mode='valid' automatically aligns the phase without messy slicing
        lo = jnp.convolve(padded, _DB4_LO, mode='valid')[::2]
        hi = jnp.convolve(padded, _DB4_HI, mode='valid')[::2]
        return lo, hi

    def _idwt_1d_single_level(self, lo, hi):
        lo = lo.reshape(-1)
        hi = hi.reshape(-1)
        n  = lo.shape[0]

        # Upsample by inserting zeros
        lo_up  = jnp.zeros(2 * n).at[::2].set(lo)
        hi_up  = jnp.zeros(2 * n).at[::2].set(hi)

        # Periodic wrap backward: PREPEND the last 3 samples (this was the bug!)
        lo_pad = jnp.concatenate([lo_up[-3:], lo_up])
        hi_pad = jnp.concatenate([hi_up[-3:], hi_up])
        
        # mode='valid' perfectly reconstructs the original signal with 0 shift
        sig_lo = jnp.convolve(lo_pad, _DB4_LO_R, mode='valid')
        sig_hi = jnp.convolve(hi_pad, _DB4_HI_R, mode='valid')
        
        return sig_lo + sig_hi

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
        """JAX-pure WPD: no Python dicts with traced values as keys."""
        # Use a fixed-shape array indexed by (level, node) encoded as flat index
        # Total nodes: 2^0 + 2^1 + ... + 2^max_level = 2^(max_level+1) - 1
        # But we only need the leaves, so build level-by-level with static shapes
        
        current_level = [sig_1d]  # level 0: one node of length N
        
        for _ in range(max_level):
            next_level = []
            for node in current_level:
                lo, hi = self._dwt_1d_single_level(node)
                next_level.extend([lo, hi])
            current_level = next_level
        
        return current_level  # 2^max_level leaves, each length N//2^max_level

    def _shannon_entropy(self, node: jax.Array) -> jax.Array:
        e = jnp.sum(node ** 2) + 1e-12
        p = node ** 2 / e
        
        # Safe masking prevents -inf in the backward pass
        safe_p = jnp.maximum(p, 1e-30)
        return -jnp.sum(p * jnp.log(safe_p))

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

            # ── CRITICAL FIX: expand 2-channel optimizer → 6-channel physics ──
            # _compute_derivatives expects u = [δ, T_fl, T_fr, T_rl, T_rr, F_brake]
            # Previous code passed u = [δ, force] (2 elements). JAX clips OOB
            # indices to u[1], so ALL 4 hub motors AND hydraulic brake received
            # the raw total force → 4× torque amplification + parasitic braking.
            steer_cmd = jnp.clip(u_raw[0], -0.45, 0.45) + wind_yaw * 0.01
            force_cmd = jnp.clip(u_raw[1] * lmuy_scale, -8000.0, 8000.0)

            R_w = 0.2045  # wheel radius [m]
            # ── RWD FIX: Split total thrust across 2 rear motors instead of 4 ──
            T_rear = jax.nn.relu(force_cmd) * R_w / 2.0   
            F_brake = jax.nn.relu(-force_cmd)             # brake: positive = braking

            # u_6 = [δ, T_fl, T_fr, T_rl, T_rr, F_brake_hyd]
            # Send 0.0 drive torque to the front wheels
            u_6 = jnp.array([steer_cmd, 0.0, 0.0, T_rear, T_rear, F_brake])

            def substep_fn(x_s, _):
                return self._vehicle.simulate_step(x_s, u_6, setup_params,
                                                   dt=dt_sub, n_substeps=1), None

            x_next, _ = jax.lax.scan(substep_fn, x, None, length=self.n_substeps)

            # ── Absolute heading error clamp (GP-vX7) ────────────────────────
            # DIAGNOSIS: H_net phantom torque drives +0.20 rad of yaw EVERY step
            # at the saturation limit. A per-step rate limiter (GP-vX6) still
            # accumulates: 0.20 × 32 = 6.4 rad total → alpha=176° by step 15.
            # Root: rate limiter prevents acceleration but not accumulation.
            #
            # Fix: clamp ABSOLUTE heading deviation from psi_ref, not the rate.
            # This makes alpha bounded regardless of how many steps are taken.
            # arctan2 handles 2π wraparound. tanh is C∞ with ∂/∂dpsi = sech² ≠ 0.
            # Cap = 0.08 rad (≈4.6°): s_dot = vx·cos(4.6°) ≈ 0.9968·vx — the
            # car appears nearly aligned to the optimizer, giving clean gradients.
            dpsi_raw    = x_next[STATE_YAW] - psi_ref
            dpsi_wrap   = jnp.arctan2(jnp.sin(dpsi_raw), jnp.cos(dpsi_raw))
            dpsi_capped = 0.08 * jnp.tanh(dpsi_wrap / 0.08)
            x_next      = x_next.at[STATE_YAW].set(psi_ref + dpsi_capped)

            # Clamp yaw rate x[19] to kinematic maximum (GP-vX10 fix).
            # H_net phantom torque regenerates wz→60 rad/s INSIDE simulate_step,
            # after STATE_YAW is clamped. The unclamped wz then drives:
            #   vy_dot ≈ −wz·vx = −60×16 = −960 m/s² per step
            # → STATE_Y drifts ~48m/step → n=12–30m → barrier gradient ~1.9M
            # → no Wolfe-satisfying step exists in 20 probes → ABNORMAL nit=0.
            # Physical wz at κ=0.05, vx=16: wz_phys = 16×0.05 = 0.8 rad/s.
            # Cap at 4× the kinematic rate to allow transient dynamics.
            wz_max = x_next[STATE_VX] * (jnp.sqrt(k_c**2 + 1e-8) + 0.05) * 4.0
            x_next = x_next.at[19].set(
                jnp.clip(x_next[19], -wz_max, wz_max)
            )
            # Clamp lateral velocity to physical sideslip limit β_max≈11°.
            # vy = vx·tan(β) ≈ vx·0.20. Without this, phantom wz integrates
            # into vy through the rigid-body equations each substep, causing
            # the same lateral drift even after the wz clamp is applied.
            vy_max = x_next[STATE_VX] * 0.20
            x_next = x_next.at[STATE_VY].set(
                jnp.clip(x_next[STATE_VY], -vy_max, vy_max)
            )

            v_fric_sq  = (self.mu_friction * 9.81) / (jnp.sqrt(k_c**2 + 1e-8) + 1e-4)
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
            v_sat_ceil = jnp.minimum(jnp.sqrt(v_fric_sq) * 1.25, self.V_limit)
            vx_raw     = x_next[STATE_VX]
            vx_sat = v_sat_ceil - jax.nn.softplus(v_sat_ceil - vx_raw)
            x_next     = x_next.at[STATE_VX].set(jnp.maximum(vx_sat, 0.5))

            # Curvilinear coordinate: lateral deviation n from track centerline
            dx_world = x_next[STATE_X] - x_ref
            dy_world = x_next[STATE_Y] - y_ref
            dpsi     = x_next[STATE_YAW] - psi_ref
            n        = -jnp.sin(psi_ref) * dx_world + jnp.cos(psi_ref) * dy_world
            alpha    = jnp.arctan2(jnp.sin(dpsi), jnp.cos(dpsi))  # heading error

            # Progress rate ṡ = vx·cos(α) - vy·sin(α)
            # FIX (GP-vX6): WITHOUT alpha clamping, yaw divergence (e.g. 60 rad/s
            # from H_net phantom torque) drives alpha → π/2 → cos(alpha) → 0 →
            # s_dot → 0 every step. This makes time_cost = Σ(1/s_dot)·dt → ∞
            # (reported as 1404s 'lap time') and the optimizer has zero gradient
            # signal to recover — the car spins in place.
            # Clamp to ±π/4: s_dot floor = vx · cos(π/4) = 0.707·vx > 0.
            # Gradient ∂s_dot/∂alpha = −vx·sin(alpha_safe): continuous, nonzero.
            alpha_safe = jnp.clip(alpha, -jnp.pi / 4, jnp.pi / 4)
            s_dot = (x_next[STATE_VX] * jnp.cos(alpha_safe)
                     - x_next[STATE_VY] * jnp.sin(alpha_safe))

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
            
            # Use clip instead of maximum to cap the variance at 25.0 (5m std dev)
            # This prevents the barrier soft-tube from exploding to infinity
            new_var_n_full = jnp.clip(new_var_n_full, 1e-4, 25.0)

            # ──────────────────────────────────────────────────────────────────────────────
            # ROOT CAUSE (§3h confirmed): The 64-step backward scan Jacobian chain overflows
            # float32 for any cost with all 64 gradient-active steps.
            # 
            # VX is already safe (softplus eigenvalue ~0.78%/step).
            # STATE_YAW carry eigenvalue ≈ sech²(dpsi/0.08) × H_net_yaw_jac ≈ 1.5–3.5/step
            # → 1.7^63 ≈ 6e14, cross-multiplied by position accumulators → float32 overflow.
            # STATE_X/Y have no saturation; cross-terms with YAW chain cause overflow even
            # for VX-only costs (∂vx/∂yaw × (overflowed yaw chain) = NaN).
            # 
            # Fix: carry uses stop_gradient copies of overflow-prone indices.
            #   • Forward dynamics: UNCHANGED (stop_gradient preserves values).
            #   • Backward pass: cross-step carry gradient = 0 instead of NaN/overflow.
            #   • Per-step cost gradients: PRESERVED via emitted x_next (full gradient).
            # This gives truncated-BPTT behaviour for these components, which is finite
            # and gives the optimizer a useful descent direction vs. the current 100% NaN.
            # ──────────────────────────────────────────────────────────────────────────────
            _CARRY_STOP_GRAD = [
                STATE_X,    # 0  — world position
                STATE_Y,    # 1  — world position
                STATE_Z,    # 2  — height
                STATE_PHI,  # 3  — roll angle
                STATE_TH,   # 4  — pitch angle
                STATE_YAW,  # 5  — forced heading
                STATE_VY,   # 15 — lateral velocity
                17,         # wx — roll rate
                18,         # wy — pitch rate
                19,         # wz — yaw rate (CRITICAL FIX)
            ]
            
            carry_x = _apply_carry_stop_gradient(x_next)

            return (carry_x, new_var_n_full, new_var_alpha), (x_next, n, new_var_n_full, s_dot)

        init_carry = (x0, jnp.array(0.01), jnp.array(0.001))
        step_data  = (U_time, track_k, track_x, track_y, track_psi)

        # CRITICAL RAM FIX: Checkpoint the horizon step
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
        # FIX (GP-vX6): previous formula Σ(1/s_dot)·dt is dimensionally s²/m,
        # not seconds, AND has a singularity when yaw divergence drives s_dot→0.
        # Root cause of 1404s "lap time": with alpha≈90° (yaw spin), s_dot→0,
        # so 1/s_dot→∞. With s_dot now clamped to ≥ vx·cos(π/4) > 0 in scan_fn,
        # the singularity is removed. BUT the correct time-domain MPC objective
        # for a fixed time horizon is to MAXIMIZE track progress covered:
        #   maximize Σ s_dot·dt  ≡  minimize −Σ s_dot·dt
        # This is numerically identical to the kinematic cost and gives clean
        # gradients everywhere. The factor 0.05 (= dt_control) makes the scale
        # match the old formulation at operating speed (~20 m/s).
        s_dot_safe = stable_softplus(s_dot * 20.0) / 20.0 + 1e-2  # floor at 1e-2 m/s
        time_cost  = -jnp.sum(s_dot_safe) * self.dt_control

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
            # Stack leaves into a 2D array BEFORE differentiation — single XLA node
            # Python list → jnp.array materializes traced values, severing the grad tape
            leaves_stacked = jnp.stack(leaves, axis=0)  # (2^L, N//2^L)
            entropies = jax.vmap(self._shannon_entropy)(leaves_stacked)
            return jnp.sum(entropies)

        CW_ENTROPY_WEIGHT = 0.0  # re-enable after xlogy fix validated

        # CRITICAL: CW_ENTROPY_WEIGHT is a Python-level constant, so this if-branch
        # is resolved at JIT trace time. When weight=0, _cw_entropy_cost is never
        # traced → its NaN VJP never enters the computation graph.
        # Do NOT use `0.0 * cw_entropy_cost` as a gate: in JAX, 0.0 * NaN_grad = NaN.
        if CW_ENTROPY_WEIGHT > 0.0:
            cw_entropy_cost = (_cw_entropy_cost(U_opt[:, 0])
                            + _cw_entropy_cost(U_opt[:, 1]))
        else:
            cw_entropy_cost = jnp.array(0.0)

        effort_cost = (jnp.sum(w_steer * U_opt[:, 0] ** 2) * 1e-3
            + self.w_effort * jnp.sum((U_opt[:, 1] / 8000.0) ** 2)
            + self.w_l1_detail * l1_detail_cost
            + CW_ENTROPY_WEIGHT * cw_entropy_cost)

        # ── 3. Track limits — smooth quadratic violation penalty ─────────────
        # CRITICAL FIX (GP-vX5): the log-barrier −ε·log(softplus(dist·50)/50+1e-5)
        # has ∇→0 when dist<<0 (car outside track). This is correct interior-point
        # behaviour but catastrophically wrong here: the warm start trajectory
        # routinely places the car outside the track boundary, and the optimizer
        # sees zero gradient → zero recovery signal → car never returns on track.
        #
        # Replacement: softplus-smoothed squared violation.
        # ψ(v) = w · softplus(v · sharpness)² / sharpness²
        # where v = (|n_mean| + tube_radius) − w_track  [violation ≥ 0]
        # ∂ψ/∂n = 2w · softplus(v·s)/s² · sigmoid(v·s) · ∂v/∂n
        # This gradient is PROPORTIONAL to the violation magnitude — larger
        # violations produce stronger restoring forces everywhere.
        # ── 3. Track limits — smooth quadratic violation penalty ─────────────
        sp_sharp    = 10.0    # was 20.0
        w_barrier   = 2000.0  # was 8000.0  
        tube_radius = self.kappa_safe * jnp.sqrt(jnp.maximum(n_var, 1e-4))

        raw_left  = ( n_mean + tube_radius - track_w_left ) * sp_sharp
        raw_right = (-n_mean + tube_radius - track_w_right) * sp_sharp
        
        viol_left  = stable_softplus(raw_left) / sp_sharp
        viol_right = stable_softplus(raw_right) / sp_sharp
        
        barrier_cost = w_barrier * jnp.sum(viol_left ** 2 + viol_right ** 2)

        # ── 3b. Centreline cost ───────────────────────────────────────────────
        center_cost = self.w_center * jnp.sum(n_mean ** 2)

        # ── 3c. Heading alignment cost — CRITICAL (GP-vX6) ───────────────────
        # Without this term the optimizer has ZERO direct gradient to keep the
        # car aligned with the track. Yaw divergence (60 rad/s in Test 8) drives
        # alpha → π/2, s_dot → 0, and the 'lap time' → ∞. The center_cost only
        # penalizes n (lateral position) NOT heading — the car can drift laterally
        # with zero heading error (if the track curves) or spin with n≈0.
        #
        # alpha = STATE_YAW - psi_ref: heading error vs track tangent.
        # ψ = Σ w_heading · α²  on every horizon step.
        # At the warm start (aligned), alpha≈0 → ψ≈0, no penalty.
        # At yaw divergence (alpha=π/2), ψ = w_heading · (π/2)² · N ≈ large.
        # Gradient ∂ψ/∂alpha = 2·w_heading·alpha is everywhere nonzero → the
        # optimizer always has a restoring signal when the car starts to spin.
        dpsi_traj    = x_traj[:, STATE_YAW] - track_psi
        alpha_traj   = jnp.arctan2(jnp.sin(dpsi_traj), jnp.cos(dpsi_traj))
        heading_cost = self.w_heading * jnp.sum(alpha_traj ** 2)

        # ── 4. Terminal speed cost  ───────────────────────────────────────────
        v_terminal  = x_traj[-1, STATE_VX]
        k_terminal  = track_k[-1]
        v_safe_term = jnp.sqrt((self.mu_friction * 9.81) / (jnp.sqrt(k_terminal**2 + 1e-8) + 1e-4))
        term_cost   = 50.0 * jax.nn.relu(v_terminal - v_safe_term) ** 2

        # ── 5. Friction circle — Augmented Lagrangian (UPGRADE-1) ─────────────
        # Constraint: g_i = (a_lat²_i + a_lon²_i) / (μ·g)² - 1 ≤ 0
        g_val       = 9.81
        vx_traj = x_traj[:, STATE_VX].reshape(-1)                       # guaranteed (N,)
        a_lat_sq    = (vx_traj ** 2 * jnp.sqrt(track_k**2 + 1e-8)) ** 2 # centripetal accel: a_lat = v²·κ, κ=0 → a_lat=0, no NaN
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
        
        dpsi     = jnp.diff(x_traj[:, STATE_YAW], prepend=x0[STATE_YAW])
        # In margin_penalty: replace diff-based wz with direct wz column
        wz_safe = x_traj[:, 19]  # direct wz — already clamped in scan_fn
        alpha_current = delta - (vy + lf * wz_safe) / jnp.maximum(vx, 1.0)
        
        # Grip margin at each horizon step:
        alpha_margin = alpha_peak_est - jnp.sqrt(alpha_current**2 + 1e-8)
        
        # Soft penalty when margin < 20% of alpha_peak:
        # This keeps the solver in the high-grip linear region (Batch 4 logic)
        # Grip margin penalty
        margin_penalty = jnp.sum(stable_softplus((0.20 * alpha_peak_est - alpha_margin) * 20.0))

        # Forward progress floor
        v_min_fwd  = 3.0   
        v_min_cost = 15.0 * jnp.mean(stable_softplus(v_min_fwd - vx_traj) ** 2)

        # Add to the final return (weighted by ~5.0 to make it influential)
        return (time_cost + effort_cost + barrier_cost + center_cost + heading_cost +
                term_cost + al_friction + 5.0 * margin_penalty + v_min_cost)

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
        x0:           jax.Array,     # (46,) initial state
        setup_params: jax.Array,     # (28,) suspension setup
        track_x:      'jax.Array | None' = None,   # (N,) centreline x [m] — optional
        track_y:      'jax.Array | None' = None,   # (N,) centreline y [m] — optional
    ) -> jax.Array:
        g  = 9.81
        wb = self.vp.get('lf', 0.8525) + self.vp.get('lr', 0.6975)
        Kp = 6000.0   # N / (m/s)

        k_safe   = jnp.sqrt(track_k**2 + 1e-8).ravel() + 1e-4
        v_target = jnp.minimum(
            jnp.sqrt((self.mu_friction * g) / k_safe),
            self.V_limit * 0.92,   # 8% safety margin below friction limit
        )
        steer_ref = jnp.clip(track_k.ravel() * wb, -0.45, 0.45)
        Kp_lat     = 0.25   # lateral error → steer correction [rad/m]
        Kp_heading = 2.0    # heading error → steer correction [rad/rad]

        def scan_fn(carry_state, step_idx):
            state = carry_state
            
            v_curr  = state[STATE_VX]
            v_tgt_i = v_target[step_idx]
            k_i     = track_k[step_idx]
            psi_ref_i = track_psi[step_idx]
            steer_ref_i = steer_ref[step_idx]

            # Lateral + heading feedback
            if track_x is not None and track_y is not None:
                x_car     = state[STATE_X]
                y_car     = state[STATE_Y]
                psi_car   = state[STATE_YAW]
                track_x_i = track_x[step_idx]
                track_y_i = track_y[step_idx]
                
                n_err = (-jnp.sin(psi_ref_i) * (x_car - track_x_i)
                         + jnp.cos(psi_ref_i) * (y_car - track_y_i))
                heading_err = jnp.arctan2(jnp.sin(psi_car - psi_ref_i),
                                          jnp.cos(psi_car - psi_ref_i))
                steer_fb = (-Kp_lat * n_err / jnp.maximum(v_curr, 1.0)
                            - Kp_heading * heading_err)
                steer_i = jnp.clip(steer_ref_i + steer_fb, -0.45, 0.45)
            else:
                steer_i = steer_ref_i

            v_err = v_curr - v_tgt_i
            m_veh      = self.vp.get('m', 235.0)
            F_lat_curr = m_veh * v_tgt_i ** 2 * jnp.sqrt(k_i**2 + 1e-8)
            F_fric_tot = self.mu_friction * 9.81 * m_veh          
            F_lon_max  = jnp.sqrt(jnp.maximum(F_fric_tot ** 2 - F_lat_curr ** 2, 0.0))
            
            F_ctrl = jnp.where(v_err > 0.0,
                               jnp.clip(-Kp * v_err, -F_lon_max, 0.0),
                               jnp.clip(-Kp * v_err * 0.3, 0.0, jnp.minimum(600.0, F_lon_max)))

            R_w   = 0.2045
            T_rear_i = jnp.maximum(F_ctrl, 0.0) * R_w / 2.0
            F_b_i = jnp.maximum(-F_ctrl, 0.0)
            u_i   = jnp.array([steer_i, 0.0, 0.0, T_rear_i, T_rear_i, F_b_i])
            
            state = self._vehicle.simulate_step(
                state, u_i, setup_params,
                dt=self.dt_control, n_substeps=self.n_substeps,
            )

            psi_ref_ws = psi_ref_i
            vx_ws      = jnp.maximum(state[STATE_VX], 1.0)
            k_ws       = k_i
            
            state = state.at[STATE_YAW].set(psi_ref_ws)          # exact heading
            state = state.at[19].set(vx_ws * k_ws)               # kinematic wz = vx·κ
            state = state.at[STATE_VY].set(0.0)                  # aligned with tangent
            if track_x is not None and track_y is not None:
                state = state.at[STATE_X].set(track_x[step_idx]) # on centreline
                state = state.at[STATE_Y].set(track_y[step_idx])

            vx_raw   = state[STATE_VX]
            v_ceil   = jnp.minimum(v_tgt_i * 1.05, self.V_limit)
            vx_safe  = jnp.minimum(vx_raw, v_ceil)
            state    = state.at[STATE_VX].set(jnp.maximum(vx_safe, 0.5))

            return state, jnp.array([steer_i, F_ctrl], dtype=jnp.float32)

        # Eradicate explicit Python loop, utilize XLA native scan
        step_indices = jnp.arange(self.N)
        final_state, U_warm = jax.lax.scan(scan_fn, x0, step_indices)
        
        return U_warm

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
        # ocp_solver.py  solve()  ← replace the resampling block

        # Estimate how far the car will travel over the horizon
        k0_abs       = abs(float(np.mean(track_k[:max(1, len(track_k)//8)]))) + 1e-4
        v_est        = min(math.sqrt((self.mu_friction * 9.81) / k0_abs), self.V_limit)
        horizon_dist = v_est * self.N * self.dt_control         # ≈ metres ahead

        track_total_len = float(track_s[-1]) if float(track_s[-1]) > 1.0 else 1.0
        # Fraction of the lap covered by this horizon
        frac = min(horizon_dist / track_total_len, 1.0)

        s_orig = np.linspace(0, 1, len(track_k))
        s_wav  = np.linspace(0, frac, self.N)   # ← was np.linspace(0, 1, self.N)

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
        
        # Initialize all 28 thermal nodes (x[28:56]) to warm operating conditions.
        # State layout: 4 corners × 7 nodes = [inner, mid, outer, bulk, carcass, gas, contact]
        # x[28:38] (old 10-node) only partially covered 2 corners and left
        # most nodes at make_initial_state default of T_env+5 = 30°C.
        # Thermal grip factor at 30°C: exp(-0.0008*(30-85)²) ≈ 0.089 — 9% of peak grip.
        _T_corner_warm = jnp.array([85., 85., 85., 80., 75., 30., 40.])
        x0 = x0.at[28:56].set(jnp.tile(_T_corner_warm, 4))
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
            prev_shifted    = jnp.roll(self._prev_solution, -1, axis=0)
            wc_kin          = self._db4_dwt(prev_shifted)          # ← ADD
            flat_init       = wc_kin.flatten()
            wc_kin_flat_np  = np.array(flat_init, dtype=np.float64)  # ← ADD
            print("[Diff-WMPC] Warm-starting from previous solution (shifted).")
        else:
            try:
                U_warm         = self._build_physics_warmstart(track_k, track_psi, x0, setup_params,
                                                                track_x=track_x, track_y=track_y)
                wc_kin         = self._db4_dwt(U_warm)
                flat_init      = wc_kin.flatten()
                wc_kin_flat_np = np.array(flat_init, dtype=np.float64)
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

        # ── Outer AL loop ─────────────────────────────────────────────────────
        n_al_iters = 5   # GP-vX8: always 3 (was `3 if dev else 5`). nit=1-9 observed
                         # in all working solves; 5 iters wastes 40% of wall time.
        opt_coeffs = jnp.array(flat_init)

        # FIX: Define these variables BEFORE the debug block uses them!
        val_grad_fn    = self._val_grad_fn
        wc_kin_flat_jx = wc_kin.reshape(-1).astype(jnp.float32)
        alpha_peak_jax = jnp.array(alpha_peak_est, dtype=jnp.float32)

        # ── DEBUG: inspect warm-start loss and gradient norm ─────────────
        _init_rho_jax_dbg = jnp.array(self._AL_RHO_SCHEDULE[0], dtype=jnp.float32)
        _loss_ws, _grad_ws = val_grad_fn(
            opt_coeffs, wc_kin_flat_jx, jnp.zeros(self.N), _init_rho_jax_dbg,
            x0, setup_params, track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right, alpha_peak_jax,
        )
        _init_rho_jax_dbg = jnp.array(self._AL_RHO_SCHEDULE[0], dtype=jnp.float32)
        _loss_ws, _grad_ws = val_grad_fn(
            opt_coeffs, wc_kin_flat_jx, jnp.zeros(self.N), _init_rho_jax_dbg,
            x0, setup_params, track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right, alpha_peak_jax,
        )
        _gnorm = float(jnp.linalg.norm(_grad_ws))
        print(f"[DEBUG WS] loss={float(_loss_ws):.4f}  ‖∇f‖={_gnorm:.2f}  "
              f"finite_loss={bool(jnp.isfinite(_loss_ws))}  "
              f"finite_grad={bool(jnp.all(jnp.isfinite(_grad_ws)))}")

        # Simulate warm-start trajectory and print per-step n and g_circle
        _U_ws, _x_ws, _n_ws, _, _s_ws = self._simulate_trajectory(
            opt_coeffs.reshape(self.N, 2), x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right, 1.0, 0.0, self.dt_control,
        )
        _vx_ws = _x_ws[:, STATE_VX]
        _vx_prev_ws = jnp.concatenate([x0[STATE_VX:STATE_VX+1].ravel(), _vx_ws[:-1]])
        _a_lat_ws = _vx_ws ** 2 * jnp.sqrt(track_k**2 + 1e-8)
        _a_lon_ws = jnp.sqrt((_vx_ws - _vx_prev_ws)**2 + 1e-8) / self.dt_control
        _g_ws = ((_a_lat_ws**2 + _a_lon_ws**2) / ((self.mu_friction*9.81)**2 + 1e-4)) - 1.0
        print(f"[DEBUG WS] n: min={float(jnp.min(_n_ws)):.2f}  max={float(jnp.max(_n_ws)):.2f}  "
              f"mean={float(jnp.mean(_n_ws)):.2f}")
        print(f"[DEBUG WS] vx: min={float(jnp.min(_vx_ws)):.2f}  max={float(jnp.max(_vx_ws)):.2f}  "
              f"mean={float(jnp.mean(_vx_ws)):.2f}")
        print(f"[DEBUG WS] g_circle: min={float(jnp.min(_g_ws)):.4f}  "
              f"max={float(jnp.max(_g_ws)):.4f}  "
              f"n_violated={int(jnp.sum(_g_ws > 0.0))}/{self.N}")
        print(f"[DEBUG WS] s_dot: min={float(jnp.min(_s_ws)):.2f}  "
              f"max={float(jnp.max(_s_ws)):.2f}")

        # Gradient breakdown: finite-difference the loss w.r.t. a small step
        # to confirm the Wolfe condition is geometrically possible
        _eps = 1e-3
        _d_ws = -_grad_ws / (_gnorm + 1e-12)  # steepest descent direction
        _loss_step, _ = val_grad_fn(
            opt_coeffs + _eps * _d_ws, wc_kin_flat_jx, jnp.zeros(self.N),
            _init_rho_jax_dbg, x0, setup_params, track_k, track_x, track_y,
            track_psi, track_w_left, track_w_right, alpha_peak_jax,
        )
        _expected_decrease = _eps * _gnorm  # c1=1, directional deriv = -‖∇f‖
        _actual_decrease = float(_loss_ws) - float(_loss_step)
        print(f"[DEBUG WS] Wolfe probe at ε={_eps}: "
              f"Δf_actual={_actual_decrease:.6f}  "
              f"Δf_expected(c1=1)={_expected_decrease:.6f}  "
              f"ratio={_actual_decrease/(_expected_decrease+1e-30):.4f}")
        print(f"[DEBUG WS] Wolfe c1=0.0001 threshold: "
              f"need Δf > {0.0001*_expected_decrease:.8f}  "
              f"{'SATISFIED' if _actual_decrease > 0.0001*_expected_decrease else 'VIOLATED — this is why ABNORMAL'}")
        # ── END DEBUG ─────────────────────────────────────────────────────

        # Use the pre-compiled JIT function from __init__ — guaranteed cache hit.
        # All varying inputs (track arrays, x0, wc_kin) are explicit arguments.
        wc_kin_flat_jx = wc_kin.reshape(-1).astype(jnp.float32)
        alpha_peak_jax = jnp.array(alpha_peak_est, dtype=jnp.float32)

        # Trigger compilation on very first solve() call, then print once.
        if not self._jit_compiled:
            _init_rho_jax = jnp.array(self._AL_RHO_SCHEDULE[0], dtype=jnp.float32)
            jax.block_until_ready(val_grad_fn(
                opt_coeffs, wc_kin_flat_jx, al_lambda, _init_rho_jax,
                x0, setup_params, track_k, track_x, track_y, track_psi,
                track_w_left, track_w_right, alpha_peak_jax,
            ))
            self._jit_compiled = True
            print("[Diff-WMPC] JIT compiled — starting AL loop.")
        else:
            print("[Diff-WMPC] JIT cache hit — starting AL loop.")

        for al_iter in range(n_al_iters):
            al_rho     = self._AL_RHO_SCHEDULE[al_iter]
            al_rho_jax = jnp.array(al_rho, dtype=jnp.float32)
            nan_count   = [0]
            total_calls = [0]

            def scipy_obj(x_np):
                total_calls[0] += 1
                x_jax              = jnp.array(x_np, dtype=jnp.float32)
                loss_jax, grad_jax = val_grad_fn(
                    x_jax, wc_kin_flat_jx, al_lambda, al_rho_jax,
                    x0, setup_params, track_k, track_x, track_y, track_psi,
                    track_w_left, track_w_right, alpha_peak_jax,
                )

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
                
                # --- APPLY GLOBAL NORM CLIPPING ---
                # We clip the overall magnitude to 1e8, but preserve the exact direction.
                # This stops the 10^15 AD explosion from causing float overflow in L-BFGS-B, 
                # but because 1e8 is still larger than the true FD gradient (~1e7), 
                # it doesn't break the Wolfe condition math like the old clip=500 did.
                g_norm = np.linalg.norm(grad_np)
                if g_norm > 1e8:
                    grad_np = grad_np * (1e8 / g_norm)
                # ----------------------------------

                # Debug: print first 3 calls per AL iter
                if total_calls[0] <= 3:
                    print(f"[DEBUG OBJ] call={total_calls[0]}  "
                          f"loss={float(loss_jax):.4f}  "
                          f"‖∇f‖={float(np.linalg.norm(grad_np)):.2f}  "
                          f"‖∇f‖_inf={float(np.max(np.abs(grad_np))):.2f}")
                return float(loss_jax), grad_np

            print(f"[Diff-WMPC] AL iter {al_iter+1}/{n_al_iters} — "
                  f"ρ={al_rho:.1f}, λ_max={float(jnp.max(al_lambda)):.3f}")
            print(f"[Diff-WMPC] Optimising 3-level Db4 basis "
                  f"over N={self.N} via L-BFGS-B…")

            res = scipy.optimize.minimize(  # MUST use module ref, not bound alias
                # Rationale: live_monitor.py patches scipy.optimize.minimize
                # (the module attribute) to intercept calls and update the live
                # plot. Using `from scipy.optimize import minimize as scipy_minimize`
                # (the old import) binds the name at import time — immune to the
                # monkey-patch. Using scipy.optimize.minimize reads the attribute
                # dynamically, so the monitor's patch is actually invoked.
                scipy_obj,
                np.array(opt_coeffs),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 30,   # GP-vX8: was 600/2000. Observed nit=1-9 in all
                                     # working solves → extra budget = failed Wolfe
                                     # line searches at 1.4s/eval = pure waste.
                                     # 30 iters × 3 AL × ~4 evals = ~168s/solve (N=64)
                                     # With --dev (N=32): ~42s/solve × 86 = 60min lap.
                    'maxls':   20,   # GP-vX9 set this to 2 to save time when NaN-fallback
                                    # was poisoning the Wolfe check. NaN rate is now 0% —
                                    # ABNORMAL is purely a step-size issue. With barrier
                                    # gradient ~1e4 at n=10m, L-BFGS-B needs >2 backtrack
                                    # probes to find a Wolfe-satisfying step. 20 matches
                                    # scipy default and is sufficient for smooth objectives.
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

            # Always evaluate constraint violation and update AL multipliers.
            # BUGFIX (GP-vX8): was guarded by `if not self.dev_mode:` → in dev
            # mode the friction constraint λ was NEVER updated → λ_max=0.000
            # forever → max G_combined=5.888 (completely unconstrained).
            wc_opt = opt_coeffs.reshape((self.N, 2))
            wc_opt = jnp.clip(wc_opt, -25000.0, 25000.0)
            U_al, x_al, _, _, _ = self._simulate_trajectory(
                wc_opt, x0, setup_params,
                track_k, track_x, track_y, track_psi,
                track_w_left, track_w_right,
                1.0, 0.0, self.dt_control,
            )
            vx_al    = x_al[:, STATE_VX].reshape(-1)
            track_k_1d = track_k.reshape(-1)
            a_lat_sq = (vx_al ** 2 * jnp.sqrt(track_k_1d**2 + 1e-8)) ** 2
            vx_prev  = jnp.concatenate([
                x0[STATE_VX : STATE_VX + 1].reshape(-1),
                vx_al[:-1],
            ])
            a_lon_sq = ((vx_al - vx_prev) / self.dt_control) ** 2
            g_al     = (a_lat_sq + a_lon_sq) / ((self.mu_friction * 9.81) ** 2 + 1e-4) - 1.0

            al_lambda = jnp.maximum(al_lambda + al_rho * g_al, 0.0)
            # Cap λ at 100: empirically sufficient to enforce the friction constraint
            # (penalty = 100×g >> time_cost≈0.05 for any g>0.0005). Without capping,
            # λ accumulates to 27,000 across receding horizon solves — the gradient
            # becomes 540,000× the time cost → L-BFGS-B Wolfe search fails on every
            # step → nit=0 ABNORMAL on every AL iter → solver is completely stuck.
            al_lambda = jnp.minimum(al_lambda, 200.0)  # <-- TWEAK 3: Up from 20.0   # GP-vX9: was 100. At λ=100,
            # gradient ≈ 100×14.5×√32=8207 → clipping was needed → Wolfe failed.
            # At λ=20: gradient ≈ 20×14.5×√32=1641 → no clipping needed → Wolfe works.
            max_viol  = float(jnp.max(jnp.maximum(g_al, 0.0)))
            print(f"[Diff-WMPC] Constraint max violation: {max_viol:.4f} "
                  f"(0=feasible). Updated λ_max={float(jnp.max(al_lambda)):.3f}")

            if max_viol > 0.1:
                al_rho = min(al_rho * self._al_rho_scale, 500.0)

            # Early exit: constraint is satisfied — further AL iters won't improve much
            if max_viol < 0.05:
                print(f"[Diff-WMPC] Constraint satisfied (viol={max_viol:.4f} < 0.05) — "
                      f"exiting AL loop at iter {al_iter+1}/{n_al_iters}")
                break

        # Store AL state — cap to prevent unbounded accumulation across receding horizon
        # GP-vX8: in receding horizon MPC each solve window sees shifted track geometry.
        # The λ from window t has no direct meaning for window t+K. However, partially
        # warm-starting λ (rather than resetting to 0) speeds up convergence when the
        # same corner appears in consecutive windows. We cap at 100 (already done above)
        # and store the clipped value.
        self._al_lambda = jnp.minimum(al_lambda, 20.0)
        self._al_rho    = min(al_rho, 20.0)

        nan_rate_iter = nan_count[0] / max(total_calls[0], 1)
        print(f"[Diff-WMPC] AL iter {al_iter+1} result: "
                f"nit={res.nit}  nfev={res.nfev}  "
                f"NaN={nan_count[0]}/{total_calls[0]} ({nan_rate_iter:.0%})  "
                f"{'CONVERGED' if res.success else res.message[:40]}")

        # ── Final trajectory extraction ───────────────────────────────────────
        wc_final = opt_coeffs.reshape((self.N, 2))
        wc_final = jnp.clip(wc_final, -25000.0, 25000.0)

        # OPEN-LOOP EVALUATION FIX: Use the exact same lmuy_scalar that _loss_fn used!
        lmuy_scalar_eval = 1.0 - jnp.mean(w_mu) * 0.5
        
        U_opt, x_traj, n_opt, var_n_opt, s_dot_opt = self._simulate_trajectory(
            wc_final, x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right,
            lmuy_scalar_eval, 0.0, self.dt_control,  # Replaced the hardcoded 1.0 here
        )
        self._prev_solution = U_opt

        # Lap time estimate: total arc-length covered / mean speed
        # GP-vX6: old formula Σ(1/s_dot)*dt was dimensionally s²/m — reported 1404s
        # for a 1.6s horizon because yaw-spin drove s_dot→0. Correct formula:
        # T_approx = track_arc / mean_s_dot, where track_arc = Σ(s_dot*dt).
        track_arc_m  = float(jnp.sum(s_dot_opt * self.dt_control))
        mean_s_dot   = float(jnp.mean(s_dot_opt))
        # Extrapolated full-lap estimate (only meaningful when horizon >> 1 corner)
        time_total = track_total_len / max(mean_s_dot, 0.5) if track_arc_m > 1.0 else float('inf')

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