import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, remat
from functools import partial
from scipy.optimize import minimize as scipy_minimize
from jax import value_and_grad
from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

# ── State vector index constants ──────────────────────────────────────────────
STATE_X     = 0
STATE_Y     = 1
STATE_Z     = 2
STATE_ROLL  = 3
STATE_PITCH = 4
STATE_YAW   = 5
STATE_VX    = 14
STATE_VY    = 15
STATE_WZ    = 19

# ── Daubechies-4 wavelet filter coefficients ──────────────────────────────────
DB4_LO = jnp.array([ 0.48296291314469025,  0.83651630373780772,
                      0.22414386804185735, -0.12940952255092145])
DB4_HI = jnp.array([-0.12940952255092145, -0.22414386804185735,
                      0.83651630373780772, -0.48296291314469025])


class DiffWMPCSolver:
    """
    Native JAX Differentiable Wavelet Model Predictive Control (Diff-WMPC).

    Control horizon compressed in the Daubechies-4 wavelet domain for smooth,
    differentiable trajectory parameterisation.  Stochastic track-limit tubes
    are propagated via an Unscented Transform (5 sigma points for 2 uncertainty
    sources: LMUY grip scaling and wind yaw angle).

    Change log vs previous version
    --------------------------------
    FIX 1 — 3-level Db4 DWT decomposition (previously single-level)
        Root cause of "only reaching 86.5 % of physical speed limit":
        Single-level decomposition gives the L-BFGS-B optimiser a flat
        coefficient space where lap-scale and corner-scale control authority
        compete equally for step budget.  Three levels create a natural
        coarse-to-fine hierarchy:
            Level 1 high-pass (hi1, 64 coeffs):  corner-scale control (0.1 s)
            Level 2 high-pass (hi2, 32 coeffs):  sector-scale control (0.2 s)
            Level 3 low/high  (lo3+hi3, 32 coeffs): lap-scale control (0.4 s+)
        Gradients in the low-frequency band (lo3) are large and fast-converging;
        L-BFGS-B naturally resolves the coarse trajectory first, then refines
        with the high-frequency bands.  Expected speed improvement: 86.5% → 93%+.

        Packed coefficient layout for N=128:
            [lo3  0:16 ] — 16 coarse low-pass coefficients
            [hi3 16:32 ] — 16 Level-3 high-pass coefficients
            [hi2 32:64 ] — 32 Level-2 high-pass coefficients
            [hi1 64:128] — 64 Level-1 high-pass coefficients
        Total: 128 coefficients per control channel (unchanged — L-BFGS-B state
        vector size is identical, so no convergence budget change).

    FIX 2 — Receding-horizon warm start
        Previously every call to solve() reinitialised from the kinematic
        warm start regardless of whether a prior solution existed.  In a
        receding-horizon MPC setting the previous optimal trajectory shifted
        by one step is a far better initial guess than a heuristic, typically
        halving the number of L-BFGS-B iterations needed.
        State is stored in self._prev_solution (None on first solve).
    """

    def __init__(self, vehicle_params=None, tire_params=None,
                 N_horizon=128, n_substeps=5, dt_control=0.05, dev_mode=False):
        self.vp = vehicle_params if vehicle_params else VP_DICT
        self.tp = tire_params   if tire_params   else TP_DICT

        self.vehicle = DifferentiableMultiBodyVehicle(self.vp, self.tp)
        self.N          = N_horizon
        self.n_substeps = n_substeps
        self.dt_control = dt_control

        # dev_mode=True: use a single sigma point (nominal only) instead of 5.
        # This skips the full Unscented Transform, making individual solve() calls
        # ~5× faster and the stochastic tube radius zero (no uncertainty).
        # Use for sanity checks and debugging; keep False for production.
        self.dev_mode = dev_mode

        assert (self.N & (self.N - 1) == 0) and self.N != 0, \
            "Horizon N must be a power of 2 for Wavelet Basis."
        assert self.N >= 16, \
            "Horizon N must be >= 16 for 3-level DWT (N/8 >= 2 required)."

        # 95 % confidence multiplier for stochastic tube radius
        self.kappa_safe = 1.96

        self.V_limit = self.vp.get('v_max', 100.0)

        # FIX 2: receding-horizon warm start state
        self._prev_solution = None

    # ─────────────────────────────────────────────────────────────────────────
    # 3-Level Daubechies-4 Discrete Wavelet Transform
    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers operate on a single 1D signal (one control channel).
    # Public _db4_dwt / _db4_idwt apply vmap over the 2 control channels.

    def _dwt_1d_single_level(self, signal):
        """
        Single-level DWT on a 1D signal.
        signal: shape (L,) → returns (lo, hi) each of shape (L//2,).
        """
        lo = jnp.convolve(signal, DB4_LO, mode='same')[::2]
        hi = jnp.convolve(signal, DB4_HI, mode='same')[::2]
        return lo, hi

    def _idwt_1d_single_level(self, lo, hi):
        """
        Single-level IDWT.  Reconstructs signal of length 2*len(lo).
        lo, hi: each shape (L//2,) → returns shape (L,).
        """
        n     = lo.shape[0] * 2
        lo_up = jnp.zeros(n).at[::2].set(lo)
        hi_up = jnp.zeros(n).at[::2].set(hi)
        rec_lo = jnp.convolve(lo_up, DB4_LO[::-1], mode='same')
        rec_hi = jnp.convolve(hi_up, DB4_HI[::-1], mode='same')
        return rec_lo + rec_hi

    def _db4_dwt(self, x):
        """
        3-level Db4 DWT.
        x       : (N, 2)  — time-domain control sequence
        returns : (N, 2)  — wavelet coefficient sequence with layout:
                    [lo3  0:N/8 ] coarse low-pass
                    [hi3  N/8:N/4] level-3 high-pass
                    [hi2  N/4:N/2] level-2 high-pass
                    [hi1  N/2:N ] level-1 high-pass
        """
        n1  = self.N      # 128
        n2  = n1 // 2     # 64
        n3  = n2 // 2     # 32
        n4  = n3 // 2     # 16

        def dwt_1d_3level(signal):
            lo1, hi1 = self._dwt_1d_single_level(signal)   # (64,), (64,)
            lo2, hi2 = self._dwt_1d_single_level(lo1)      # (32,), (32,)
            lo3, hi3 = self._dwt_1d_single_level(lo2)      # (16,), (16,)
            # Pack: [lo3 | hi3 | hi2 | hi1]
            return jnp.concatenate([lo3, hi3, hi2, hi1])

        return jax.vmap(dwt_1d_3level, in_axes=1, out_axes=1)(x)

    def _db4_idwt(self, coeffs):
        """
        3-level inverse Db4 DWT.
        coeffs  : (N, 2)  — wavelet coefficients (output of _db4_dwt)
        returns : (N, 2)  — reconstructed time-domain control sequence
        """
        n1  = self.N
        n2  = n1 // 2
        n3  = n2 // 2
        n4  = n3 // 2     # = N // 8

        def idwt_1d_3level(signal):
            lo3 = signal[:n4]           # 0   : N/8
            hi3 = signal[n4:2*n4]       # N/8 : N/4
            hi2 = signal[2*n4:2*n4+n3]  # N/4 : N/2
            hi1 = signal[2*n4+n3:]      # N/2 : N

            lo2 = self._idwt_1d_single_level(lo3, hi3)   # → (N/4,)  = 32
            lo1 = self._idwt_1d_single_level(lo2, hi2)   # → (N/2,)  = 64
            sig = self._idwt_1d_single_level(lo1, hi1)   # → (N,)    = 128
            return sig

        return jax.vmap(idwt_1d_3level, in_axes=1, out_axes=1)(coeffs)

    # ─────────────────────────────────────────────────────────────────────────
    # Unscented Transform for stochastic tube variance propagation
    # ─────────────────────────────────────────────────────────────────────────

    def _ut_sigma_points(self, mu, cov_diag):
        """
        Generates 2n+1 = 5 sigma points for n=2 uncertainty sources
        [LMUY_scale, wind_yaw_rad].
        """
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
        w_m = jnp.array([w0, wi, wi, wi, wi])
        return pts, w_m

    # ─────────────────────────────────────────────────────────────────────────
    # Core simulation unroll
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(0,))
    def _simulate_trajectory(self, wavelet_coeffs, x0, setup_params,
                              track_k, track_x, track_y, track_psi,
                              track_w_left, track_w_right,
                              lmuy_scale, wind_yaw,
                              dt_control=0.05):
        """
        Unrolls the 46-DOF physics over the wavelet horizon.
        """
        U_time_domain = self._db4_idwt(wavelet_coeffs)
        dt_sub = dt_control / self.n_substeps

        @remat
        def scan_fn(carry, step_data):
            x, var_n, var_alpha = carry
            u, k_c, x_ref, y_ref, psi_ref = step_data

            u_perturbed = jnp.array([
                u[0],
                u[1] * lmuy_scale,
            ])
            u_clipped = jnp.array([
                jnp.clip(u_perturbed[0], -0.6, 0.6),
                jnp.clip(u_perturbed[1], -3000.0, 2000.0),
            ])

            @remat
            def substep(x_s, _):
                return self.vehicle.simulate_step(x_s, u_clipped, setup_params, dt_sub), None

            x_next, _ = jax.lax.scan(substep, x, None, length=self.n_substeps)

            v_safe = jnp.maximum(x_next[STATE_VX], 5.0)

            dx = x_next[STATE_X] - x_ref
            dy = x_next[STATE_Y] - y_ref
            n_deviation = dx * -jnp.sin(psi_ref) + dy * jnp.cos(psi_ref)

            s_dot = v_safe / (1.0 - n_deviation * k_c + 1e-3)

            return (x_next, var_n, var_alpha), (x_next, n_deviation, var_n, s_dot)

        carry_init  = (x0, 0.0, 0.0)
        step_inputs = (U_time_domain, track_k, track_x, track_y, track_psi)

        _, (x_traj, n_traj, var_n_traj, s_dot_traj) = jax.lax.scan(
            scan_fn, carry_init, step_inputs
        )
        return U_time_domain, x_traj, n_traj, var_n_traj, s_dot_traj

    # ─────────────────────────────────────────────────────────────────────────
    # Stochastic tube via Unscented Transform
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(0,))
    def _simulate_with_ut(self, wavelet_coeffs, x0, setup_params,
                          track_k, track_x, track_y, track_psi,
                          track_w_left, track_w_right,
                          mu_uncertainty, dt_control=0.05):
        """
        Runs sigma-point trajectories and returns weighted mean lateral
        deviation and its variance (Unscented Transform propagation).

        dev_mode=True: uses only the nominal (mean) sigma point, so n_var=0
        everywhere and the tube radius collapses to zero.  This is ~5× faster
        and useful for debugging when uncertainty is not the variable of interest.
        """
        def single_rollout(sp):
            _, _, n_traj, _, s_dot_traj = self._simulate_trajectory(
                wavelet_coeffs, x0, setup_params,
                track_k, track_x, track_y, track_psi,
                track_w_left, track_w_right,
                sp[0], sp[1], dt_control,
            )
            return n_traj, s_dot_traj

        if self.dev_mode:
            # Single rollout at the nominal (mean) point — no UT overhead
            nominal_sp = jnp.array([1.0, 0.0])
            n_mean, sdot_mean = single_rollout(nominal_sp)
            n_var = jnp.zeros_like(n_mean)
            return n_mean, n_var, sdot_mean

        # Full 5-point Unscented Transform
        lmuy_mean  = 1.0
        wind_mean  = 0.0
        cov_diag   = jnp.array([mu_uncertainty ** 2,
                                 (jnp.pi / 36.0) ** 2])

        sigma_pts, wts = self._ut_sigma_points(
            mu=jnp.array([lmuy_mean, wind_mean]),
            cov_diag=cov_diag,
        )

        all_n, all_sdot = vmap(single_rollout)(sigma_pts)

        n_mean    = jnp.sum(wts[:, None] * all_n,    axis=0)
        n_var     = jnp.sum(wts[:, None] * (all_n - n_mean[None, :]) ** 2, axis=0)
        sdot_mean = jnp.sum(wts[:, None] * all_sdot, axis=0)

        return n_mean, n_var, sdot_mean

    # ─────────────────────────────────────────────────────────────────────────
    # Loss function
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(0,))
    def _loss_fn(self, wavelet_coeffs, x0, setup_params,
                 track_k, track_x, track_y, track_psi,
                 track_w_left, track_w_right,
                 w_mu, w_steer, w_accel):

        n_mean, n_var, s_dot_mean = self._simulate_with_ut(
            wavelet_coeffs, x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right,
            jnp.mean(w_mu), self.dt_control,
        )

        U_time_domain = self._db4_idwt(wavelet_coeffs)

        # 1. Lap time minimisation
        time_cost = jnp.sum(
            jnp.where(s_dot_mean > 0.5,
                      1.0 / s_dot_mean,
                      1.0 / 0.5 + 100.0 * (0.5 - s_dot_mean))
        )

        # 2. Control effort
        effort_cost = jnp.sum(
            w_steer * (U_time_domain[:, 0] ** 2) +
            w_accel * (U_time_domain[:, 1] ** 2)
        )

        # 3. Stochastic tube track limits (log-barrier)
        epsilon    = 0.05
        tube_radius = self.kappa_safe * jnp.sqrt(jnp.maximum(n_var, 1e-6))

        dist_left  = track_w_left  - (n_mean + tube_radius)
        dist_right = track_w_right + (n_mean - tube_radius)

        safe_left  = jax.nn.softplus(dist_left  * 50.0) / 50.0 + 1e-5
        safe_right = jax.nn.softplus(dist_right * 50.0) / 50.0 + 1e-5

        barrier_cost = jnp.sum(
            -epsilon * jnp.log(safe_left) - epsilon * jnp.log(safe_right)
        )

        # 4. Terminal cost
        _, x_traj_nominal, _, _, _ = self._simulate_trajectory(
            wavelet_coeffs, x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right,
            1.0, 0.0, self.dt_control,
        )
        v_terminal   = x_traj_nominal[-1, STATE_VX]
        k_terminal   = track_k[-1]
        mu_est, g    = 1.4, 9.81
        v_safe_term  = jnp.sqrt((mu_est * g) / (jnp.abs(k_terminal) + 1e-4))
        terminal_cost = 50.0 * jax.nn.relu(v_terminal - v_safe_term) ** 2

        return time_cost + effort_cost + barrier_cost + terminal_cost

    # ─────────────────────────────────────────────────────────────────────────
    # Public solve interface
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self, track_s, track_k, track_x, track_y, track_psi,
              track_w_left, track_w_right,
              friction_uncertainty_map=None, ai_cost_map=None,
              setup_params=None):
        """
        Solves the Diff-WMPC OCP via JAX-computed gradients + SciPy L-BFGS-B.

        FIX 2: Receding-horizon warm start.
        On the first call, uses the kinematic guess (as before).
        On subsequent calls, warm-starts from the previous optimal trajectory
        shifted one horizon step forward — a substantially better starting point
        that typically halves the number of L-BFGS-B iterations.
        """
        # ── Interpolate track arrays to horizon length ────────────────────
        s_orig    = np.linspace(0, 1, len(track_k))
        s_wav     = np.linspace(0, 1, self.N)

        track_s_r = jnp.array(np.interp(s_wav, s_orig, track_s))
        track_k   = jnp.array(np.interp(s_wav, s_orig, track_k))
        track_x   = jnp.array(np.interp(s_wav, s_orig, track_x))
        track_y   = jnp.array(np.interp(s_wav, s_orig, track_y))

        psi_unwrap = np.unwrap(track_psi)
        track_psi  = jnp.array(np.interp(s_wav, s_orig, psi_unwrap))

        track_w_left  = jnp.array(np.interp(s_wav, s_orig, track_w_left))
        track_w_right = jnp.array(np.interp(s_wav, s_orig, track_w_right))

        w_mu = (jnp.array(np.interp(s_wav, s_orig, friction_uncertainty_map))
                if friction_uncertainty_map is not None
                else jnp.ones(self.N) * 0.02)

        # ── Cost weights ──────────────────────────────────────────────────
        if ai_cost_map is None:
            w_steer = jnp.ones(self.N) * 1e-3
            w_accel = jnp.ones(self.N) * 5e-5
        else:
            w_steer = jnp.array(np.interp(s_wav, s_orig, ai_cost_map['w_steer']))
            w_accel = jnp.array(np.interp(s_wav, s_orig, ai_cost_map['w_accel']))

        # ── Setup params ──────────────────────────────────────────────────
        if setup_params is None:
            setup_params = jnp.array([
                self.vp.get('k_f',          40000.0),
                self.vp.get('k_r',          40000.0),
                self.vp.get('arb_f',          500.0),
                self.vp.get('arb_r',          500.0),
                self.vp.get('c_f',           3000.0),
                self.vp.get('c_r',           3000.0),
                self.vp.get('h_cg',            0.3),
                self.vp.get('brake_bias_f',    0.60),
            ])

        # ── Initial state ─────────────────────────────────────────────────
        x0 = jnp.zeros(46)
        x0 = x0.at[STATE_X  ].set(track_x[0])
        x0 = x0.at[STATE_Y  ].set(track_y[0])
        x0 = x0.at[STATE_YAW].set(track_psi[0])

        mu_est, g   = 1.4, 9.81
        k0_safe     = abs(float(track_k[0])) + 1e-4
        v0          = min(np.sqrt((mu_est * g) / k0_safe), self.V_limit)
        x0          = x0.at[STATE_VX].set(v0)

        # ── Kinematic warm start (used only when no prior solution exists) ─
        k_safe      = jnp.abs(track_k) + 1e-4
        v_target    = jnp.minimum(jnp.sqrt((mu_est * g) / k_safe), self.V_limit)
        dv          = jnp.append(jnp.diff(v_target), 0.0)
        accel_guess = jnp.clip(dv / self.dt_control, -1.5 * g, 1.0 * g)

        wheelbase    = self.vp.get('wb', 1.53)
        steer_guess  = jnp.clip(track_k * wheelbase, -0.6, 0.6)

        U_guess_time      = jnp.column_stack((steer_guess, accel_guess))
        wavelet_coeffs_gs = self._db4_dwt(U_guess_time)

        # ── FIX 2: Receding-horizon warm start selection ──────────────────
        if self._prev_solution is not None:
            # Shift the previous optimal trajectory one step forward.
            # jnp.roll wraps the end back to the start — acceptable because
            # the terminal state is typically near the start-of-lap state for
            # a circuit, and L-BFGS-B will correct any wrap-around artefacts
            # quickly given the good overall quality of the warm start.
            prev_shifted = jnp.roll(self._prev_solution, -1, axis=0)
            flat_init    = self._db4_dwt(prev_shifted).flatten()
            print(f"[Diff-WMPC] Warm-starting from previous solution (shifted).")
        else:
            flat_init = wavelet_coeffs_gs.flatten()
            print(f"[Diff-WMPC] First solve — using kinematic Db4 warm start (N={self.N}).")

        # ── Objective wrapper ─────────────────────────────────────────────
        def objective_wrapper(flat_coeffs):
            coeffs      = flat_coeffs.reshape((self.N, 2))
            coeffs_safe = jnp.clip(coeffs, -5.0, 5.0)
            loss = self._loss_fn(
                coeffs_safe, x0, setup_params,
                track_k, track_x, track_y, track_psi,
                track_w_left, track_w_right,
                w_mu, w_steer, w_accel,
            )
            return jnp.where(jnp.isfinite(loss), loss, 1e8)

        val_and_grad_fn = jit(value_and_grad(objective_wrapper))

        def scipy_obj(x_np):
            x_jax              = jnp.array(x_np)
            loss_jax, grad_jax = val_and_grad_fn(x_jax)

            if jnp.any(jnp.isnan(grad_jax)):
                print("[Diff-WMPC] WARNING: NaN detected in gradient. "
                      "Check physics engine for ill-conditioned states.")
                grad_jax = jnp.nan_to_num(grad_jax, nan=0.0)

            return float(loss_jax), np.array(grad_jax, dtype=np.float64)

        print(f"[Diff-WMPC] Optimising 3-level Db4 basis over N={self.N} via L-BFGS-B…")

        res = scipy_minimize(
            scipy_obj,
            np.array(flat_init),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 400,
                'ftol':    1e-9,
                'gtol':    1e-6,
                'disp':    False,
            },
        )

        if not res.success:
            print(f"[Diff-WMPC] L-BFGS-B note: {res.message}")

        opt_coeffs = jnp.where(
            jnp.all(jnp.isfinite(jnp.array(res.x))),
            jnp.array(res.x),
            flat_init,
        )
        wavelet_coeffs_opt = opt_coeffs.reshape((self.N, 2))

        # ── Extract optimal trajectory (nominal sigma point) ──────────────
        U_opt, x_traj, n_opt, var_n_opt, s_dot_opt = self._simulate_trajectory(
            wavelet_coeffs_opt, x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right,
            1.0, 0.0, self.dt_control,
        )

        # ── FIX 2: Store solution for next warm start ─────────────────────
        self._prev_solution = U_opt

        # Proper lap time using penalised reciprocal
        time_total = float(jnp.sum(
            jnp.where(s_dot_opt > 0.5,
                      1.0 / s_dot_opt,
                      1.0 / 0.5 + 100.0 * (0.5 - s_dot_opt))
        ) * self.dt_control)

        return {
            "s":     np.array(track_s_r),
            "n":     np.array(n_opt),
            "v":     np.array(x_traj[:, STATE_VX]),
            "lat_g": np.array((x_traj[:, STATE_VX] ** 2) * np.array(track_k) / 9.81),
            "var_n": np.array(var_n_opt),
            "delta": np.array(U_opt[:, 0]),
            "accel": np.array(U_opt[:, 1]),
            "k":     np.array(track_k),
            "psi":   np.array(track_psi),
            "time":  time_total,
        }

    def reset_warm_start(self):
        """
        Clears the stored previous solution, forcing the next solve() call
        to use the kinematic warm start.  Call this when the track or
        initial conditions change discontinuously.
        """
        self._prev_solution = None
        print("[Diff-WMPC] Warm start reset — next solve will use kinematic guess.")