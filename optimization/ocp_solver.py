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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the 28-parameter default setup from vehicle_params dict
# ─────────────────────────────────────────────────────────────────────────────

def _build_default_setup_28(vp: dict) -> jnp.ndarray:
    """
    Construct a 28-element setup_params vector from vehicle_params.

    P10 of vehicle_dynamics.py expanded setup_params from 8 → 28 scalars
    (full 4-way damper, heave springs, geometric alignment, etc.).

    Index table (matches vehicle_dynamics.py exactly):
     0  k_f              N/m   front wheel-centre spring rate
     1  k_r              N/m   rear  wheel-centre spring rate
     2  k_heave_f        N/m   front heave / third-spring
     3  k_heave_r        N/m   rear  heave / third-spring
     4  arb_f            N/m   front ARB wheel-centre rate
     5  arb_r            N/m   rear  ARB wheel-centre rate
     6  c_ls_bump_f      Ns/m  front LS bump damping
     7  c_hs_bump_f      Ns/m  front HS bump damping
     8  c_ls_reb_f       Ns/m  front LS rebound damping
     9  c_hs_reb_f       Ns/m  front HS rebound damping
    10  c_ls_bump_r      Ns/m  rear  LS bump
    11  c_hs_bump_r      Ns/m  rear  HS bump
    12  c_ls_reb_r       Ns/m  rear  LS rebound
    13  c_hs_reb_r       Ns/m  rear  HS rebound
    14  v_knee_f         m/s   front LS/HS knee velocity
    15  v_knee_r         m/s   rear  LS/HS knee velocity
    16  toe_f            rad   front static toe
    17  toe_r            rad   rear  static toe
    18  camber_f_deg     deg   front static camber
    19  camber_r_deg     deg   rear  static camber
    20  caster_deg       deg   front caster angle
    21  h_cg_setup       m     CG height
    22  brake_bias_f     —     front brake fraction
    23  diff_lock        —     rear diff lock ratio
    24  arb_preload_f    N     front ARB preload
    25  arb_preload_r    N     rear  ARB preload
    26  bumpstop_gap_f   m     front bumpstop clearance
    27  bumpstop_gap_r   m     rear  bumpstop clearance

    Damper derivation
    -----------------
    vehicle_params.py stores the legacy two-coefficient model (low/high speed).
    These map to the 4-way model as:
        c_ls_bump  = c_low          (low-speed compression)
        c_hs_bump  = c_high         (high-speed compression)
        c_ls_reb   = c_low  × 1.5   (low-speed extension — typically stiffer)
        c_hs_reb   = c_high × 1.5   (high-speed extension)

    Spring rate derivation
    ----------------------
    vehicle_params.py stores the spring rate AT THE SPRING.
    The 28-param vector wants the WHEEL-CENTRE rate:
        k_wheel = k_spring × MR²
    where MR_f ≈ 1.14 and MR_r ≈ 1.16 from the motion_ratio_f/r_poly[0].
    """
    import math

    mr_f = vp.get('motion_ratio_f_poly', [1.14])[0]
    mr_r = vp.get('motion_ratio_r_poly', [1.16])[0]

    # Springs — wheel-centre rates
    k_f = vp.get('spring_rate_f', 35030.) * (mr_f ** 2)
    k_r = vp.get('spring_rate_r', 52540.) * (mr_r ** 2)

    # Heave / third-spring (not in vehicle_params → use PhysicsNormalizer mean)
    k_heave_f = vp.get('k_heave_f', 5000.)
    k_heave_r = vp.get('k_heave_r', 5000.)

    # ARBs
    arb_f = vp.get('arb_rate_f', vp.get('arb_f', 200.))
    arb_r = vp.get('arb_rate_r', vp.get('arb_r', 150.))

    # 4-way damper — derived from legacy low/high coefficients
    c_low_f  = vp.get('damper_c_low_f',  vp.get('c_f', 3000.) * 0.60)
    c_high_f = vp.get('damper_c_high_f', vp.get('c_f', 3000.) * 0.40)
    c_low_r  = vp.get('damper_c_low_r',  vp.get('c_r', 3000.) * 0.60)
    c_high_r = vp.get('damper_c_high_r', vp.get('c_r', 3000.) * 0.40)

    c_ls_bump_f = c_low_f
    c_hs_bump_f = c_high_f
    c_ls_reb_f  = c_low_f  * 1.5
    c_hs_reb_f  = c_high_f * 1.5

    c_ls_bump_r = c_low_r
    c_hs_bump_r = c_high_r
    c_ls_reb_r  = c_low_r  * 1.5
    c_hs_reb_r  = c_high_r * 1.5

    v_knee = vp.get('damper_v_knee', 0.10)

    # Alignment (vehicle_params uses degrees for camber, degrees for toe)
    toe_f_deg    = vp.get('static_toe_f', -0.10)          # degrees
    toe_r_deg    = vp.get('static_toe_r', -0.15)          # degrees
    toe_f_rad    = math.radians(toe_f_deg)
    toe_r_rad    = math.radians(toe_r_deg)
    camber_f_deg = vp.get('static_camber_f', -2.0)        # already in degrees
    camber_r_deg = vp.get('static_camber_r', -1.5)
    caster_deg   = vp.get('caster', vp.get('caster_f', 5.0))

    h_cg         = vp.get('h_cg',         0.330)
    brake_bias_f = vp.get('brake_bias_f',  0.60)
    diff_lock    = vp.get('diff_lock_ratio', 1.0)

    # ARB preload — not in vehicle_params, default 0 (symmetric)
    arb_preload_f = vp.get('arb_preload_f', 0.0)
    arb_preload_r = vp.get('arb_preload_r', 0.0)

    # Bumpstop gap
    bumpstop_gap_f = vp.get('bump_stop_engage', vp.get('bumpstop_gap_f', 0.025))
    bumpstop_gap_r = vp.get('bumpstop_gap_r',   0.018)

    return jnp.array([
        k_f,           k_r,            # 0–1
        k_heave_f,     k_heave_r,      # 2–3
        arb_f,         arb_r,          # 4–5
        c_ls_bump_f,   c_hs_bump_f,    # 6–7
        c_ls_reb_f,    c_hs_reb_f,     # 8–9
        c_ls_bump_r,   c_hs_bump_r,    # 10–11
        c_ls_reb_r,    c_hs_reb_r,     # 12–13
        v_knee,        v_knee,         # 14–15
        toe_f_rad,     toe_r_rad,      # 16–17
        camber_f_deg,  camber_r_deg,   # 18–19
        caster_deg,                    # 20
        h_cg,                          # 21
        brake_bias_f,                  # 22
        diff_lock,                     # 23
        arb_preload_f, arb_preload_r,  # 24–25
        bumpstop_gap_f, bumpstop_gap_r,# 26–27
    ], dtype=jnp.float32)


class DiffWMPCSolver:
    """
    Native JAX Differentiable Wavelet Model Predictive Control (Diff-WMPC).

    Change log vs previous version
    ─────────────────────────────────────────────────────────────────────────
    P10 SETUP FIX — setup_params expanded from 8 → 28 scalars
    ─────────────────────────────────────────────────────────────────────────
    Root cause of  TypeError: sub got incompatible shapes (8,) vs (28,):

    vehicle_dynamics.py was upgraded to P10 which extended setup_params from
    8 scalars to 28 (full 4-way damper, heave springs, alignment, bumpstops).
    ocp_solver.py was still building the OLD 8-element default vector.
    The crash occurred in PhysicsNormalizer.normalize_setup(setup_params)
    which does  (s - setup_mean) / scale  where s.shape=(8,) and
    setup_mean.shape=(28,) → broadcast failure.

    Fix: replace the inline 8-element jnp.array([...]) with a call to
    _build_default_setup_28(vp) which correctly derives all 28 values
    from vehicle_params.py keys (with sensible physical defaults).

    GRIP LEAK FIX — Friction Circle Exact Penalty (retained from prev. version)
    ─────────────────────────────────────────────────────────────────────────
    See _loss_fn docstring for full explanation.
    µ=1.35, α=8.0, w=200.0 tuned on circular-track sanity check.

    RETAINED from previous version:
    ─────────────────────────────────────────────────────────────────────────
    FIX 1  — 3-level Db4 DWT decomposition
    FIX 2  — Receding-horizon warm start
    BUG 5  — L-BFGS-B maxls=100, maxiter=2000
    NaN FIX— Python-level NaN detection + L2 fallback gradient
    Coefficient clip −3.0, L2 regularisation 5e-5, scan NaN guard
    """

    def __init__(self, vehicle_params=None, tire_params=None,
                 N_horizon=128, n_substeps=5, dt_control=0.05, dev_mode=False):
        self.vp = vehicle_params if vehicle_params else VP_DICT
        self.tp = tire_params   if tire_params   else TP_DICT

        self.vehicle    = DifferentiableMultiBodyVehicle(self.vp, self.tp)
        self.N          = N_horizon
        self.n_substeps = n_substeps
        self.dt_control = dt_control
        self.dev_mode   = dev_mode

        assert (self.N & (self.N - 1) == 0) and self.N != 0, \
            "Horizon N must be a power of 2 for Wavelet Basis."
        assert self.N >= 16, \
            "Horizon N must be >= 16 for 3-level DWT (N/8 >= 2 required)."

        self.kappa_safe   = 1.96
        self.V_limit      = self.vp.get('v_max', 100.0)
        self._prev_solution = None

        # Friction circle constraint parameters (tuned on circular-track test)
        self.mu_friction   = 1.35   # conservative — 95% of 1.4 Pacejka nominal
        self.alpha_fric    = 8.0    # exponential barrier steepness
        self.w_friction    = 200.0  # weight: comparable to terminal_cost at violation≈0.3

    # ─────────────────────────────────────────────────────────────────────────
    # 3-Level Daubechies-4 DWT / IDWT (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _dwt_1d_single_level(self, signal):
        lo = jnp.convolve(signal, DB4_LO, mode='same')[::2]
        hi = jnp.convolve(signal, DB4_HI, mode='same')[::2]
        return lo, hi

    def _idwt_1d_single_level(self, lo, hi):
        n     = lo.shape[0] * 2
        lo_up = jnp.zeros(n).at[::2].set(lo)
        hi_up = jnp.zeros(n).at[::2].set(hi)
        rec_lo = jnp.convolve(lo_up, DB4_LO[::-1], mode='same')
        rec_hi = jnp.convolve(hi_up, DB4_HI[::-1], mode='same')
        return rec_lo + rec_hi

    def _db4_dwt(self, x):
        def dwt_1d_3level(signal):
            lo1, hi1 = self._dwt_1d_single_level(signal)
            lo2, hi2 = self._dwt_1d_single_level(lo1)
            lo3, hi3 = self._dwt_1d_single_level(lo2)
            return jnp.concatenate([lo3, hi3, hi2, hi1])
        return jax.vmap(dwt_1d_3level, in_axes=1, out_axes=1)(x)

    def _db4_idwt(self, coeffs):
        n4 = self.N // 8

        def idwt_1d_3level(signal):
            n3  = self.N // 4
            lo3 = signal[:n4]
            hi3 = signal[n4:2*n4]
            hi2 = signal[2*n4:2*n4+n3]
            hi1 = signal[2*n4+n3:]
            lo2 = self._idwt_1d_single_level(lo3, hi3)
            lo1 = self._idwt_1d_single_level(lo2, hi2)
            sig = self._idwt_1d_single_level(lo1, hi1)
            return sig

        return jax.vmap(idwt_1d_3level, in_axes=1, out_axes=1)(coeffs)

    # ─────────────────────────────────────────────────────────────────────────
    # Unscented Transform (unchanged)
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
        w_m = jnp.array([w0, wi, wi, wi, wi])
        return pts, w_m

    # ─────────────────────────────────────────────────────────────────────────
    # Core simulation unroll (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(0,))
    def _simulate_trajectory(self, wavelet_coeffs, x0, setup_params,
                              track_k, track_x, track_y, track_psi,
                              track_w_left, track_w_right,
                              lmuy_scale, wind_yaw,
                              dt_control=0.05):
        U_time_domain = self._db4_idwt(wavelet_coeffs)
        dt_sub = dt_control / self.n_substeps

        @remat
        def scan_fn(carry, step_data):
            x, var_n, var_alpha = carry
            u, k_c, x_ref, y_ref, psi_ref = step_data

            u_perturbed = jnp.array([u[0], u[1] * lmuy_scale])
            u_clipped   = jnp.array([
                jnp.clip(u_perturbed[0], -0.6, 0.6),
                jnp.clip(u_perturbed[1], -3000.0, 2000.0),
            ])

            @remat
            def substep(x_s, _):
                return self.vehicle.simulate_step(x_s, u_clipped, setup_params, dt_sub), None

            x_next, _ = jax.lax.scan(substep, x, None, length=self.n_substeps)

            # Scan NaN guard
            x_next = jnp.where(jnp.isfinite(x_next[STATE_VX]), x_next, x)

            v_safe = jnp.maximum(x_next[STATE_VX], 5.0)
            dx     = x_next[STATE_X] - x_ref
            dy     = x_next[STATE_Y] - y_ref
            n_deviation = dx * -jnp.sin(psi_ref) + dy * jnp.cos(psi_ref)
            s_dot  = v_safe / (1.0 - n_deviation * k_c + 1e-3)

            return (x_next, var_n, var_alpha), (x_next, n_deviation, var_n, s_dot)

        carry_init  = (x0, 0.0, 0.0)
        step_inputs = (U_time_domain, track_k, track_x, track_y, track_psi)

        _, (x_traj, n_traj, var_n_traj, s_dot_traj) = jax.lax.scan(
            scan_fn, carry_init, step_inputs
        )
        return U_time_domain, x_traj, n_traj, var_n_traj, s_dot_traj

    # ─────────────────────────────────────────────────────────────────────────
    # Stochastic tube via Unscented Transform (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(0,))
    def _simulate_with_ut(self, wavelet_coeffs, x0, setup_params,
                          track_k, track_x, track_y, track_psi,
                          track_w_left, track_w_right,
                          mu_uncertainty, dt_control=0.05):
        def single_rollout(sp):
            _, _, n_traj, _, s_dot_traj = self._simulate_trajectory(
                wavelet_coeffs, x0, setup_params,
                track_k, track_x, track_y, track_psi,
                track_w_left, track_w_right,
                sp[0], sp[1], dt_control,
            )
            return n_traj, s_dot_traj

        if self.dev_mode:
            nominal_sp      = jnp.array([1.0, 0.0])
            n_mean, sdot_mean = single_rollout(nominal_sp)
            n_var = jnp.zeros_like(n_mean)
            return n_mean, n_var, sdot_mean

        lmuy_mean = 1.0
        wind_mean = 0.0
        cov_diag  = jnp.array([mu_uncertainty ** 2, (jnp.pi / 36.0) ** 2])
        sigma_pts, wts = self._ut_sigma_points(
            mu=jnp.array([lmuy_mean, wind_mean]), cov_diag=cov_diag,
        )
        all_n, all_sdot = vmap(single_rollout)(sigma_pts)
        n_mean    = jnp.sum(wts[:, None] * all_n,    axis=0)
        n_var     = jnp.sum(wts[:, None] * (all_n - n_mean[None, :]) ** 2, axis=0)
        sdot_mean = jnp.sum(wts[:, None] * all_sdot, axis=0)
        return n_mean, n_var, sdot_mean

    # ─────────────────────────────────────────────────────────────────────────
    # Loss function (unchanged)
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

        # ── 1. Lap time minimisation ──────────────────────────────────────────
        time_cost = jnp.sum(
            jnp.where(s_dot_mean > 0.5,
                      1.0 / s_dot_mean,
                      1.0 / 0.5 + 100.0 * (0.5 - s_dot_mean))
        )

        # ── 2. Control effort ─────────────────────────────────────────────────
        effort_cost = jnp.sum(
            w_steer * (U_time_domain[:, 0] ** 2) +
            w_accel * (U_time_domain[:, 1] ** 2)
        )

        # ── 3. Stochastic tube track limits (log-barrier) ─────────────────────
        epsilon     = 0.05
        tube_radius = self.kappa_safe * jnp.sqrt(jnp.maximum(n_var, 1e-6))

        dist_left  = track_w_left  - (n_mean + tube_radius)
        dist_right = track_w_right + (n_mean - tube_radius)

        safe_left  = jax.nn.softplus(dist_left  * 50.0) / 50.0 + 1e-5
        safe_right = jax.nn.softplus(dist_right * 50.0) / 50.0 + 1e-5

        barrier_cost = jnp.sum(
            -epsilon * jnp.log(safe_left) - epsilon * jnp.log(safe_right)
        )

        # ── 4. Terminal speed cost ─────────────────────────────────────────────
        _, x_traj_nominal, _, _, _ = self._simulate_trajectory(
            wavelet_coeffs, x0, setup_params,
            track_k, track_x, track_y, track_psi,
            track_w_left, track_w_right,
            1.0, 0.0, self.dt_control,
        )
        v_terminal    = x_traj_nominal[-1, STATE_VX]
        k_terminal    = track_k[-1]
        mu_est, g_val = 1.4, 9.81
        v_safe_term   = jnp.sqrt((mu_est * g_val) / (jnp.abs(k_terminal) + 1e-4))
        terminal_cost = 50.0 * jax.nn.relu(v_terminal - v_safe_term) ** 2

        # ── 5. FRICTION CIRCLE EXACT PENALTY ──────────────────────────────────
        vx_traj = x_traj_nominal[:, STATE_VX]
        a_lat_sq = (vx_traj ** 2 * jnp.abs(track_k)) ** 2
        vx_prev  = jnp.concatenate([x0[STATE_VX:STATE_VX + 1], vx_traj[:-1]])
        a_lon_sq = ((vx_traj - vx_prev) / (self.dt_control + 1e-6)) ** 2
        circle_limit = (self.mu_friction * g_val) ** 2 + 1e-4
        violation    = (a_lat_sq + a_lon_sq) / circle_limit - 1.0
        friction_cost = (self.w_friction
                         * jnp.sum(jax.nn.softplus(self.alpha_fric * violation))
                         / self.alpha_fric)

        return time_cost + effort_cost + barrier_cost + terminal_cost + friction_cost

    # ─────────────────────────────────────────────────────────────────────────
    # Public solve interface
    # ─────────────────────────────────────────────────────────────────────────

    def solve(self, track_s, track_k, track_x, track_y, track_psi,
              track_w_left, track_w_right,
              friction_uncertainty_map=None, ai_cost_map=None,
              setup_params=None):
        """
        Solves the Diff-WMPC OCP via JAX-computed gradients + SciPy L-BFGS-B.

        setup_params : jnp.ndarray of shape (28,) or None.
            If None, a physically consistent 28-element default is built from
            vehicle_params using _build_default_setup_28().

            ── BREAKING CHANGE from pre-P10 callers ──────────────────────────
            Pre-P10 code passed a custom 8-element array here.  Those callers
            MUST be updated.  See the MORL optimizer (optimization/morl_sb_trpo.py)
            which also constructs setup_params and must be migrated to 28 params.
            Use vehicle_dynamics.DifferentiableMultiBodyVehicle.default_setup_params()
            or _build_default_setup_28(vehicle_params) as the canonical source.
        """
        # ── Interpolate track arrays to horizon length ────────────────────────
        s_orig = np.linspace(0, 1, len(track_k))
        s_wav  = np.linspace(0, 1, self.N)

        track_s_r     = jnp.array(np.interp(s_wav, s_orig, track_s))
        track_k       = jnp.array(np.interp(s_wav, s_orig, track_k))
        track_x       = jnp.array(np.interp(s_wav, s_orig, track_x))
        track_y       = jnp.array(np.interp(s_wav, s_orig, track_y))
        psi_unwrap    = np.unwrap(track_psi)
        track_psi     = jnp.array(np.interp(s_wav, s_orig, psi_unwrap))
        track_w_left  = jnp.array(np.interp(s_wav, s_orig, track_w_left))
        track_w_right = jnp.array(np.interp(s_wav, s_orig, track_w_right))

        w_mu = (jnp.array(np.interp(s_wav, s_orig, friction_uncertainty_map))
                if friction_uncertainty_map is not None
                else jnp.ones(self.N) * 0.02)

        if ai_cost_map is None:
            w_steer = jnp.ones(self.N) * 1e-3
            w_accel = jnp.ones(self.N) * 5e-5
        else:
            w_steer = jnp.array(np.interp(s_wav, s_orig, ai_cost_map['w_steer']))
            w_accel = jnp.array(np.interp(s_wav, s_orig, ai_cost_map['w_accel']))

        # ── Setup params — P10 FIX: always 28 elements ───────────────────────
        if setup_params is None:
            setup_params = _build_default_setup_28(self.vp)
            print(f"[Diff-WMPC] Built 28-param default setup from vehicle_params "
                  f"(k_f={float(setup_params[0]):.0f}, k_r={float(setup_params[1]):.0f}, "
                  f"h_cg={float(setup_params[21]):.3f})")
        else:
            # ── Guard: reject stale 8-element vectors from old callers ────────
            sp_arr = jnp.asarray(setup_params, dtype=jnp.float32)
            if sp_arr.shape != (28,):
                raise ValueError(
                    f"setup_params must have shape (28,) for P10 vehicle_dynamics. "
                    f"Got shape {sp_arr.shape}. "
                    f"Callers that previously passed an 8-element vector "
                    f"(k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f) "
                    f"must be updated to use _build_default_setup_28() or "
                    f"DifferentiableMultiBodyVehicle.default_setup_params()."
                )
            setup_params = sp_arr

        # ── Initial state ─────────────────────────────────────────────────────
        x0 = jnp.zeros(46)
        x0 = x0.at[STATE_X  ].set(track_x[0])
        x0 = x0.at[STATE_Y  ].set(track_y[0])
        x0 = x0.at[STATE_YAW].set(track_psi[0])
        mu_est, g = 1.4, 9.81
        k0_safe   = abs(float(track_k[0])) + 1e-4
        v0        = min(np.sqrt((mu_est * g) / k0_safe), self.V_limit)
        x0        = x0.at[STATE_VX].set(v0)

        # ── Kinematic warm start ──────────────────────────────────────────────
        k_safe      = jnp.abs(track_k) + 1e-4
        v_target    = jnp.minimum(jnp.sqrt((mu_est * g) / k_safe), self.V_limit)
        dv          = jnp.append(jnp.diff(v_target), 0.0)
        accel_guess = jnp.clip(dv / self.dt_control, -1.5 * g, 1.0 * g)
        wheelbase   = self.vp.get('wb', 1.53)
        steer_guess = jnp.clip(track_k * wheelbase, -0.6, 0.6)

        U_guess_time      = jnp.column_stack((steer_guess, accel_guess))
        wavelet_coeffs_gs = self._db4_dwt(U_guess_time)

        if self._prev_solution is not None:
            prev_shifted = jnp.roll(self._prev_solution, -1, axis=0)
            flat_init    = self._db4_dwt(prev_shifted).flatten()
            print(f"[Diff-WMPC] Warm-starting from previous solution (shifted).")
        else:
            flat_init = wavelet_coeffs_gs.flatten()
            print(f"[Diff-WMPC] First solve — using kinematic Db4 warm start (N={self.N}).")

        # ── Objective wrapper ─────────────────────────────────────────────────
        def objective_wrapper(flat_coeffs):
            coeffs      = flat_coeffs.reshape((self.N, 2))
            coeffs_safe = jnp.clip(coeffs, -3.0, 3.0)
            coeff_reg   = 5e-5 * jnp.sum(coeffs_safe ** 2)
            loss = self._loss_fn(
                coeffs_safe, x0, setup_params,
                track_k, track_x, track_y, track_psi,
                track_w_left, track_w_right,
                w_mu, w_steer, w_accel,
            )
            return loss + coeff_reg

        val_and_grad_fn = jit(value_and_grad(objective_wrapper))

        # ── NaN gradient fix: Python-level detection + L2 fallback ───────────
        nan_count   = [0]
        total_calls = [0]

        def scipy_obj(x_np):
            total_calls[0] += 1
            x_jax              = jnp.array(x_np)
            loss_jax, grad_jax = val_and_grad_fn(x_jax)

            loss_ok = bool(jnp.isfinite(loss_jax))
            grad_ok = bool(jnp.all(jnp.isfinite(grad_jax)))

            if not (loss_ok and grad_ok):
                nan_count[0] += 1
                coeff_rms = float(np.sqrt(np.mean(x_np ** 2)))
                loss_fb   = 1e6 + 0.5 * float(np.sum(x_np ** 2))
                grad_fb   = np.clip(x_np, -10.0, 10.0).astype(np.float64)
                if nan_count[0] <= 3 or nan_count[0] % 20 == 0:
                    loss_str = 'NaN' if not loss_ok else f'{float(loss_jax):.2f}'
                    print(f"[Diff-WMPC] NaN #{nan_count[0]} "
                          f"(call {total_calls[0]}): "
                          f"L2 fallback — coeff_rms={coeff_rms:.3f}, "
                          f"loss={loss_str}, grad_nan={not grad_ok}")
                return loss_fb, grad_fb

            return float(loss_jax), np.array(grad_jax, dtype=np.float64)

        print(f"[Diff-WMPC] Optimising 3-level Db4 basis over N={self.N} via L-BFGS-B…")
        print(f"[Diff-WMPC] Friction circle: µ={self.mu_friction}, "
              f"α={self.alpha_fric}, w={self.w_friction} — grip leak fix active.")

        res = scipy_minimize(
            scipy_obj,
            np.array(flat_init),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'maxls':   100,
                'ftol':    1e-9,
                'gtol':    1e-6,
                'disp':    False,
            },
        )

        # ── Post-solve diagnostics ────────────────────────────────────────────
        if nan_count[0] > 0:
            nan_pct = nan_count[0] / max(total_calls[0], 1) * 100
            print(f"[Diff-WMPC] NaN summary: {nan_count[0]}/{total_calls[0]} "
                  f"evaluations used L2 fallback ({nan_pct:.1f}%).")
            if nan_pct > 50:
                print(f"[Diff-WMPC] HIGH NaN RATE: H_net weights likely not converged.")

        if not res.success:
            print(f"[Diff-WMPC] L-BFGS-B note: {res.message} "
                  f"(nit={res.nit}, nfev={res.nfev})")
        else:
            print(f"[Diff-WMPC] L-BFGS-B converged: {res.message} "
                  f"(nit={res.nit}, nfev={res.nfev})")

        opt_coeffs = jnp.where(
            jnp.all(jnp.isfinite(jnp.array(res.x))),
            jnp.array(res.x),
            flat_init,
        )
        wavelet_coeffs_opt = opt_coeffs.reshape((self.N, 2))

        U_opt, x_traj, n_opt, var_n_opt, s_dot_opt = self._simulate_trajectory(
            wavelet_coeffs_opt, x0, setup_params,
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

        # ── Friction circle compliance diagnostic ─────────────────────────────
        vx_final    = np.array(x_traj[:, STATE_VX])
        a_lat_final = vx_final ** 2 * np.abs(np.array(track_k))
        a_lon_final = np.abs(np.diff(vx_final, prepend=float(x0[STATE_VX]))) / self.dt_control
        g_combined  = np.sqrt(a_lat_final ** 2 + a_lon_final ** 2) / (self.mu_friction * g)
        pct_in_circle = 100.0 * np.mean(g_combined <= 1.0)
        max_violation = float(np.max(g_combined))
        print(f"[Diff-WMPC] Friction circle compliance: "
              f"{pct_in_circle:.1f}% of steps inside µ={self.mu_friction} circle "
              f"(max G-combined = {max_violation:.3f})")

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
            "g_combined_max": max_violation,
            "friction_compliance_pct": pct_in_circle,
        }

    def reset_warm_start(self):
        """Clears stored previous solution, forcing kinematic warm start next call."""
        self._prev_solution = None
        print("[Diff-WMPC] Warm start reset — next solve will use kinematic guess.")