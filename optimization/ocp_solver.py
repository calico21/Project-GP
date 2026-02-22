import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd
import optax
import numpy as np
from functools import partial

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class DiffWMPCSolver:
    """
    Native JAX Differentiable Wavelet Model Predictive Control (Diff-WMPC).
    
    Replaces standard Implicit Runge-Kutta (IRK) Collocation with Wavelet 
    Basis Function optimization. Evaluates Stochastic Tube-MPC bounds 
    dynamically using automatic differentiation Jacobians.
    """
    def __init__(self, vehicle_params=None, tire_params=None, N_horizon=128):
        self.vp = vehicle_params if vehicle_params else VP_DICT
        self.tp = tire_params if tire_params else TP_DICT
        
        # Instantiate the 14-DOF Native JAX Physics Engine
        self.vehicle = DifferentiableMultiBodyVehicle(self.vp, self.tp)
        self.N = N_horizon
        
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
    def _simulate_trajectory(self, wavelet_coeffs, x0, setup_params, track_k, track_w_left, track_w_right, w_mu, dt=0.05):
        """
        Unrolls the 14-DOF physics and computes Stochastic Covariance Tube over the horizon.
        """
        # Inverse Wavelet Transform: Convert frequency domain coefficients to time domain controls
        # Control inputs U: [Steering delta, Longitudinal force]
        U_time_domain = jnp.dot(self.W_inv, wavelet_coeffs)
        
        def scan_fn(carry, step_data):
            x, var_n, var_alpha = carry
            u, k_c, mu_uncert = step_data
            
            # Forward physics step (14-DOF)
            x_next = self.vehicle.simulate_step(x, u, setup_params, dt)
            
            # Automatic Jacobian for Stochastic Tube Variance Propagation
            # We map the 38-D state down to planar kinematics (n, alpha) for tube bounds
            v_safe = jnp.maximum(x[14], 5.0) # vx is at index 14
            J_full = jacfwd(self.vehicle.simulate_step, argnums=0)(x, u, setup_params, dt)
            
            # Heuristic projection of spatial covariance 
            Q_n = 0.01 * mu_uncert * (v_safe ** 2)
            var_n_next = var_n + (J_full[0, 0] ** 2) * var_n + Q_n
            var_alpha_next = var_alpha + (J_full[1, 1] ** 2) * var_alpha + 0.05 * mu_uncert
            
            # Convert Cartesian output to Frenet-Serret approx for track limits
            s_dot = (v_safe) / (1.0 - x[1] * k_c + 1e-3) # x[1] approx n
            n_deviation = x[1] 
            
            return (x_next, var_n_next, var_alpha_next), (x_next, n_deviation, var_n_next, s_dot)

        carry_init = (x0, 0.0, 0.0) # x_init, var_n, var_alpha
        step_inputs = (U_time_domain, track_k, w_mu)
        
        _, (x_traj, n_traj, var_n_traj, s_dot_traj) = jax.lax.scan(scan_fn, carry_init, step_inputs)
        return U_time_domain, x_traj, n_traj, var_n_traj, s_dot_traj

    @partial(jit, static_argnums=(0,))
    def _loss_fn(self, wavelet_coeffs, x0, setup_params, track_k, track_w_left, track_w_right, w_mu, w_steer, w_accel):
        """Computes the OCP Cost Function directly through the JAX unroll."""
        U_time_domain, x_traj, n_traj, var_n_traj, s_dot_traj = self._simulate_trajectory(
            wavelet_coeffs, x0, setup_params, track_k, track_w_left, track_w_right, w_mu
        )
        
        # 1. Lap Time Minimization (Maximize progress s_dot)
        time_cost = -jnp.sum(s_dot_traj)
        
        # 2. Wavelet Domain Control Effort (Sparsity & Smoothness)
        # L2 norm on controls using dynamic AI weights
        effort_cost = jnp.sum(w_steer * (U_time_domain[:, 0] ** 2) + w_accel * (U_time_domain[:, 1] ** 2))
        
        # 3. Stochastic Tube Track Limits Constraints (Soft penalty)
        tube_radius = self.kappa_safe * jnp.sqrt(jnp.maximum(var_n_traj, 1e-6))
        left_violation = jax.nn.relu((n_traj + tube_radius) - track_w_left)
        right_violation = jax.nn.relu(-track_w_right - (n_traj - tube_radius))
        boundary_cost = 1e5 * jnp.sum(left_violation ** 2 + right_violation ** 2)
        
        total_cost = time_cost + effort_cost + boundary_cost
        return total_cost

    def solve(self, track_s, track_k, track_w_left, track_w_right, friction_uncertainty_map=None, ai_cost_map=None, setup_params=None):
        """
        Solves the Diff-WMPC optimization loop natively in Python via Optax gradient descent.
        """
        # Padding or truncating track inputs to fit Wavelet Horizon (Power of 2)
        track_k = jnp.array(np.resize(track_k, self.N))
        track_w_left = jnp.array(np.resize(track_w_left, self.N))
        track_w_right = jnp.array(np.resize(track_w_right, self.N))
        
        w_mu = jnp.array(friction_uncertainty_map) if friction_uncertainty_map is not None else jnp.ones(self.N) * 0.02
        
        if ai_cost_map is None:
            w_steer = jnp.ones(self.N) * 1e-2
            w_accel = jnp.ones(self.N) * 1e-7
        else:
            w_steer = jnp.array(np.resize(ai_cost_map['w_steer'], self.N))
            w_accel = jnp.array(np.resize(ai_cost_map['w_accel'], self.N))

        if setup_params is None:
            setup_params = jnp.zeros(7)
            
        # Initialize 38-D state vector (14 q, 14 v, 10 thermal)
        x0 = jnp.zeros(38)
        # Set initial forward velocity based on track curvature
        mu_est, g = 1.4, 9.81
        k0_safe = abs(float(track_k[0])) + 1e-4
        v0 = min(np.sqrt((mu_est * g) / k0_safe), self.V_limit)
        x0 = x0.at[14].set(v0) # vx is state 14
        
        # Initialize Wavelet coefficients (Zero initial guess)
        wavelet_coeffs = jnp.zeros((self.N, 2))
        
        # Define Optax optimizer (Adam)
        optimizer = optax.adam(learning_rate=0.05)
        opt_state = optimizer.init(wavelet_coeffs)
        
        # JIT compiled gradient step
        @jit
        def step(coeffs, state):
            loss, grads = jax.value_and_grad(self._loss_fn)(
                coeffs, x0, setup_params, track_k, track_w_left, track_w_right, w_mu, w_steer, w_accel
            )
            updates, state = optimizer.update(grads, state, coeffs)
            coeffs = optax.apply_updates(coeffs, updates)
            return coeffs, state, loss
        
        # Diff-WMPC Optimization Loop
        print(f"[Diff-WMPC] Optimizing Wavelet Basis over Horizon {self.N}...")
        for i in range(100): # 100 Iterations for real-time MPC tracking
            wavelet_coeffs, opt_state, loss_val = step(wavelet_coeffs, opt_state)

        # Extract optimal trajectory
        U_opt, x_traj, n_opt, var_n_opt, s_dot_opt = self._simulate_trajectory(
            wavelet_coeffs, x0, setup_params, track_k, track_w_left, track_w_right, w_mu
        )
        
        time_total = jnp.sum(1.0 / (s_dot_opt + 1e-3)) * 0.05 # dt approximation

        return {
            "s": track_s[:self.N], 
            "n": np.array(n_opt), 
            "v": np.array(x_traj[:, 14]), # vx
            "lat_g": np.array((x_traj[:, 14]**2) * track_k / 9.81),           
            "var_n": np.array(var_n_opt), 
            "time": float(time_total)
        }