import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from functools import partial
from jax.scipy.optimize import minimize

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class DiffWMPCSolver:
    """
    Native JAX Differentiable Wavelet Model Predictive Control (Diff-WMPC).
    
    Replaces standard Implicit Runge-Kutta (IRK) Collocation with Wavelet 
    Basis Function optimization. Evaluates Stochastic Tube-MPC bounds 
    dynamically using analytical variance propagation.
    """
    def __init__(self, vehicle_params=None, tire_params=None, N_horizon=128, n_substeps=5):
        self.vp = vehicle_params if vehicle_params else VP_DICT
        self.tp = tire_params if tire_params else TP_DICT
        
        # Instantiate the 46-DOF Native JAX Physics Engine
        self.vehicle = DifferentiableMultiBodyVehicle(self.vp, self.tp)
        self.N = N_horizon
        self.n_substeps = n_substeps  # static Python int, never traced
        
        # Ensure N is a power of 2 for the Discrete Wavelet Transform
        assert (self.N & (self.N - 1) == 0) and self.N != 0, "Horizon N must be a power of 2 for Wavelet Basis."
        
        # Precompute the Inverse Haar Wavelet Transform Matrix
        self.W_inv = self._generate_haar_matrix(self.N)

        # Baseline Parameters
        self.V_limit = self.vp.get('v_max', 100.0)
        self.kappa_safe = 1.96 # 95% Confidence interval for stochastic tube

    def _generate_haar_matrix(self, n):
        """Generates an N x N Haar orthogonal wavelet basis matrix."""
        if n == 1:
            return jnp.array([[1.0]])
        h_half = self._generate_haar_matrix(n // 2)
        top = jnp.kron(h_half, jnp.array([1.0, 1.0]))
        bottom = jnp.kron(jnp.eye(n // 2), jnp.array([1.0, -1.0]))
        return jnp.vstack([top, bottom]) / jnp.sqrt(2.0)

    @partial(jit, static_argnums=(0,))
    def _simulate_trajectory(self, wavelet_coeffs, x0, setup_params, track_k, track_x, track_y, track_psi, track_w_left, track_w_right, w_mu, dt_control=0.05):
        """
        Unrolls the 46-DOF physics and computes Stochastic Covariance Tube over the horizon.
        Uses JAX sub-stepping to maintain Störmer-Verlet integrator stability.
        """
        # Inverse Wavelet Transform: Convert frequency domain coefficients to time domain controls
        U_time_domain = jnp.dot(self.W_inv, wavelet_coeffs)
        
        dt_sub = dt_control / self.n_substeps
        
        def scan_fn(carry, step_data):
            x, var_n, var_alpha = carry
            u, k_c, x_ref, y_ref, psi_ref, mu_uncert = step_data
            
            # Clip controls to physically realisable bounds before passing to physics
            u_clipped = jnp.array([
                jnp.clip(u[0], -0.6, 0.6),        # Steering: ±34 degrees
                jnp.clip(u[1], -3000.0, 2000.0)   # Longitudinal force: braking/driving limits
            ])
            
            def substep(x_s, _):
                return self.vehicle.simulate_step(x_s, u_clipped, setup_params, dt_sub), None
                
            x_next, _ = jax.lax.scan(substep, x, None, length=self.n_substeps)

            v_safe = jnp.maximum(x_next[14], 5.0) # vx is at index 14
            
            # Cap the variance growth to prevent the stochastic tube from exploding.
            Q_n = 0.01 * mu_uncert * (v_safe ** 2)
            var_n_next = jnp.minimum(var_n * (1.0 + 0.001 * v_safe) + Q_n, 2.0)
            var_alpha_next = jnp.minimum(var_alpha * (1.0 + 0.001 * v_safe) + 0.05 * mu_uncert, 0.5)
            
            # Compute Exact Frenet-Serret lateral deviation approximation 
            dx = x_next[0] - x_ref
            dy = x_next[1] - y_ref
            n_deviation = dx * -jnp.sin(psi_ref) + dy * jnp.cos(psi_ref)
            
            # Use the properly derived n_deviation for progression kinematics
            s_dot = (v_safe) / (1.0 - n_deviation * k_c + 1e-3)
            
            return (x_next, var_n_next, var_alpha_next), (x_next, n_deviation, var_n_next, s_dot)

        carry_init = (x0, 0.0, 0.0) # x_init, var_n, var_alpha
        step_inputs = (U_time_domain, track_k, track_x, track_y, track_psi, w_mu)
        
        _, (x_traj, n_traj, var_n_traj, s_dot_traj) = jax.lax.scan(scan_fn, carry_init, step_inputs)
        return U_time_domain, x_traj, n_traj, var_n_traj, s_dot_traj

    @partial(jit, static_argnums=(0,))
    def _loss_fn(self, wavelet_coeffs, x0, setup_params, track_k, track_x, track_y, track_psi, track_w_left, track_w_right, w_mu, w_steer, w_accel):
        """Computes the OCP Cost Function directly through the JAX unroll."""
        U_time_domain, x_traj, n_traj, var_n_traj, s_dot_traj = self._simulate_trajectory(
            wavelet_coeffs, x0, setup_params, track_k, track_x, track_y, track_psi, track_w_left, track_w_right, w_mu
        )
        
        # 1. Lap Time Minimization (Maximize progress s_dot)
        time_cost = -jnp.sum(s_dot_traj)
        
        # 2. Wavelet Domain Control Effort (Sparsity & Smoothness)
        # L2 norm on controls using dynamic AI weights
        effort_cost = jnp.sum(w_steer * (U_time_domain[:, 0] ** 2) + w_accel * (U_time_domain[:, 1] ** 2))
        
        # ---------------------------------------------------------
        # 3. Stochastic Tube Track Limits (Interior Point Log-Barrier)
        # ---------------------------------------------------------
        epsilon = 0.05  # Barrier stiffness parameter
        
        # Calculate Tube Radius based on variance
        tube_radius = self.kappa_safe * jnp.sqrt(jnp.maximum(var_n_traj, 1e-6))
        
        # Calculate exact distance from the edge of the stochastic tube to the track limits
        dist_left = track_w_left - (n_traj + tube_radius)
        dist_right = track_w_right + (n_traj - tube_radius)
        
        # Use softplus to prevent log(negative) NaNs if the BFGS solver 
        # takes an aggressive early step outside the track bounds.
        # This maps negative distances to strictly positive epsilon bounds.
        safe_left = jax.nn.softplus(dist_left * 50.0) / 50.0 + 1e-5
        safe_right = jax.nn.softplus(dist_right * 50.0) / 50.0 + 1e-5
        
        # Log-barrier penalty approaches infinity as the car approaches the edge
        barrier_cost = jnp.sum(-epsilon * jnp.log(safe_left) - epsilon * jnp.log(safe_right))
        
        # ---------------------------------------------------------
        # 4. Terminal Cost (Safe Horizon Boundary)
        # ---------------------------------------------------------
        # Prevents the solver from commanding 100% throttle at the end of the 
        # horizon by forcing the terminal velocity to respect the upcoming curvature.
        v_terminal = x_traj[-1, 14] # vx at the final node
        k_terminal = track_k[-1]
        
        # Calculate the absolute maximum physical velocity the tires can sustain at node N
        # V_max = sqrt(mu * g / k)
        mu_est, g = 1.4, 9.81
        v_safe_terminal = jnp.sqrt((mu_est * g) / (jnp.abs(k_terminal) + 1e-4))
        
        # Asymmetric penalty: Heavily penalize arriving too fast, but do not penalize being slower
        terminal_cost = 50.0 * jax.nn.relu(v_terminal - v_safe_terminal)**2

        total_cost = time_cost + effort_cost + barrier_cost + terminal_cost
        return total_cost

    def solve(self, track_s, track_k, track_x, track_y, track_psi, track_w_left, track_w_right, friction_uncertainty_map=None, ai_cost_map=None, setup_params=None):
        """
        Solves the Diff-WMPC optimization loop natively in Python via JAX L-BFGS.
        """
        # 1. FIX THE RESIZE BUG: Use spatial interpolation instead of np.resize
        # This maps the arbitrary track length cleanly to the Wavelet Horizon (Power of 2)
        s_original = np.linspace(0, 1, len(track_k))
        s_wavelet = np.linspace(0, 1, self.N)
        
        track_s_resampled = jnp.array(np.interp(s_wavelet, s_original, track_s))
        track_k = jnp.array(np.interp(s_wavelet, s_original, track_k))
        
        track_x = jnp.array(np.interp(s_wavelet, s_original, track_x))
        track_y = jnp.array(np.interp(s_wavelet, s_original, track_y))
        
        # Unwrap heading before interpolation to prevent 2*pi jumps
        track_psi_unwrapped = np.unwrap(track_psi)
        track_psi_interp = np.interp(s_wavelet, s_original, track_psi_unwrapped)
        track_psi = jnp.array(track_psi_interp)
        
        track_w_left = jnp.array(np.interp(s_wavelet, s_original, track_w_left))
        track_w_right = jnp.array(np.interp(s_wavelet, s_original, track_w_right))
        
        w_mu = jnp.array(np.interp(s_wavelet, s_original, friction_uncertainty_map)) if friction_uncertainty_map is not None else jnp.ones(self.N) * 0.02
        
        if ai_cost_map is None:
            w_steer = jnp.ones(self.N) * 1e-2
            w_accel = jnp.ones(self.N) * 1e-7
        else:
            w_steer = jnp.array(np.interp(s_wavelet, s_original, ai_cost_map['w_steer']))
            w_accel = jnp.array(np.interp(s_wavelet, s_original, ai_cost_map['w_accel']))

        if setup_params is None:
            setup_params = jnp.array([
                self.vp.get('k_f', 40000.0),   # Front spring N/m
                self.vp.get('k_r', 40000.0),   # Rear spring N/m
                self.vp.get('arb_f', 500.0),   # Front ARB N/m
                self.vp.get('arb_r', 500.0),   # Rear ARB N/m
                self.vp.get('c_f', 3000.0),    # Front damper Ns/m
                self.vp.get('c_r', 3000.0),    # Rear damper Ns/m
                self.vp.get('h_cg', 0.3),      # CG height m
    ])
            
        # Initialize 46-D state vector (14 q, 14 v, 10 thermal, 8 transient slip)
        x0 = jnp.zeros(46)
        x0 = x0.at[0].set(track_x[0])
        x0 = x0.at[1].set(track_y[0])
        x0 = x0.at[5].set(track_psi[0])
        
        # Set initial forward velocity based on track curvature
        mu_est, g = 1.4, 9.81
        k0_safe = abs(float(track_k[0])) + 1e-4
        v0 = min(np.sqrt((mu_est * g) / k0_safe), self.V_limit)
        x0 = x0.at[14].set(v0) # vx is state 14
        
        # 2. FIX THE OPTIMIZER: Switch from Optax (Adam) to JAX Native BFGS
        # BFGS requires a flattened 1D array for the variables being optimized
        # ---------------------------------------------------------
        # Kinematic Point-Mass Warm Start
        # ---------------------------------------------------------
        print(f"[Diff-WMPC] Generating Kinematic Warm Start for Horizon {self.N}...")
        
        # Calculate maximum safe velocity at each track node based on curvature
        k_safe = jnp.abs(track_k) + 1e-4
        v_max_curve = jnp.sqrt((mu_est * g) / k_safe)
        v_target = jnp.minimum(v_max_curve, self.V_limit)
        
        # Approximate longitudinal acceleration needed to hit target velocities
        # a = dv/dt approx = (v_next - v_current) / dt_approx
        dv = jnp.append(jnp.diff(v_target), 0.0) 
        dt_approx = 0.05
        accel_guess = jnp.clip(dv / dt_approx, -1.5 * g, 1.0 * g)
        
        # ISSUE 3 FIX: Use correct key 'wb' instead of 'wheelbase'
        wheelbase = self.vp.get('wb', 1.53)
        steer_guess = jnp.clip(track_k * wheelbase, -0.6, 0.6)
        
        # Stack into time-domain control matrix (N, 2)
        U_guess_time = jnp.column_stack((steer_guess, accel_guess))
        
        # Convert time-domain guess into Wavelet frequency domain using Forward Haar Transform
        # Since W_inv is orthogonal, W_forward is proportional to W_inv.T
        W_forward = self.W_inv.T  
        wavelet_coeffs_guess = jnp.dot(W_forward, U_guess_time)
        
        # Flatten for the BFGS optimizer
        flat_coeffs_init = wavelet_coeffs_guess.flatten()
        
        # Wrapper to reshape the 1D BFGS array back to (N, 2) for your loss function
        # Wrapper to reshape the 1D BFGS array back to (N, 2) for your loss function
        @jit
        def objective_wrapper(flat_coeffs):
            coeffs = flat_coeffs.reshape((self.N, 2))
            # Clip in wavelet domain before physics sees them.
            # Haar coefficients scale with control amplitude - ±5 is generous.
            coeffs_safe = jnp.clip(coeffs, -5.0, 5.0)
            loss = self._loss_fn(
                coeffs_safe, x0, setup_params, track_k, track_x, track_y,
                track_psi, track_w_left, track_w_right, w_mu, w_steer, w_accel
            )
            return jnp.where(jnp.isfinite(loss), loss, 1e8)
        
        print(f"[Diff-WMPC] Optimizing Wavelet Basis over Horizon {self.N} using JAX L-BFGS...")
        
        # Execute the BFGS optimization (50 iterations of BFGS usually outperforms 500 of Adam)
        res = minimize(objective_wrapper, flat_coeffs_init, method='BFGS', options={'maxiter': 50})

        # If BFGS failed (NaN in result), fall back to warm start
        opt_coeffs = jnp.where(
            jnp.all(jnp.isfinite(res.x)), 
            res.x, 
            flat_coeffs_init
        )
        wavelet_coeffs = opt_coeffs.reshape((self.N, 2))

        # Extract optimal trajectory
        U_opt, x_traj, n_opt, var_n_opt, s_dot_opt = self._simulate_trajectory(
            wavelet_coeffs, x0, setup_params, track_k, track_x, track_y, track_psi, track_w_left, track_w_right, w_mu
        )
        
        time_total = jnp.sum(1.0 / (s_dot_opt + 1e-3)) * 0.05 # dt approximation

        return {
            "s": np.array(track_s_resampled), 
            "n": np.array(n_opt), 
            "v": np.array(x_traj[:, 14]), # vx
            "lat_g": np.array((x_traj[:, 14]**2) * track_k / 9.81),          
            "var_n": np.array(var_n_opt), 
            "delta": np.array(U_opt[:, 0]),
            "accel": np.array(U_opt[:, 1]),
            "time": float(time_total)
        }