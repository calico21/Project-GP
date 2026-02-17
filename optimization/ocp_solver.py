import casadi as ca
import numpy as np
from models.vehicle_dynamics import DynamicBicycleModel
from models.tire_model import PacejkaTire
# Load configs
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class OptimalLapSolver:
    """
    Optimization engine for finding the 'Perfect Lap' (Minimum Time).
    """
    def __init__(self, vehicle_params=None, tire_params=None):
        self.vp = vehicle_params if vehicle_params else VP_DICT
        self.tp = tire_params if tire_params else TP_DICT
        self.model = DynamicBicycleModel(self.vp, self.tp)
        self.f_dyn = self.model.get_equations() 

    def solve(self, track_s, track_k, track_w_left, track_w_right, N=100):
        """
        Solves the Optimal Control Problem.
        """
        opti = ca.Opti() 
        
        # --- Decision Variables ---
        # State: [s, n, alpha, v, delta, r]
        X = opti.variable(6, N+1)
        s_var       = X[0, :]
        n_var       = X[1, :]
        alpha_var   = X[2, :]
        v_var       = X[3, :]
        delta_var   = X[4, :]
        r_var       = X[5, :]

        # Control: [der_delta, Fx_req]
        U = opti.variable(2, N)
        d_delta_var = U[0, :]
        Fx_var      = U[1, :]
        
        # Parameters
        K_param = opti.parameter(N)
        opti.set_value(K_param, track_k[:-1]) 

        # --- Objective: Minimize Time ---
        T_total = 0
        
        for k in range(N):
            # Spatial Dynamics
            x_k = X[:, k]
            u_k = U[:, k]
            k_c = K_param[k] 
            
            v_k = x_k[3]
            alpha_k = x_k[2]
            n_k = x_k[1]
            
            # s_dot = (v * cos(alpha)) / (1 - n * k)
            s_dot = (v_k * ca.cos(alpha_k)) / (1 - n_k * k_c)
            dt_ds = 1.0 / (s_dot + 1e-3)
            
            dx_dt = self.f_dyn(x_k, u_k, k_c)
            dx_ds = dx_dt * dt_ds
            
            # Euler Integration
            x_next = x_k + dx_ds * (track_s[k+1] - track_s[k])
            opti.subject_to(X[:, k+1] == x_next)
            
            # Constraints
            w_l = track_w_left[k]
            w_r = track_w_right[k]
            opti.subject_to(opti.bounded(-w_r, n_k, w_l))
            opti.subject_to(v_k > 1.0) 
            
            # Limits
            max_steer = self.vp.get('max_steer', 0.4)
            opti.subject_to(opti.bounded(-max_steer, delta_var[k], max_steer))
            opti.subject_to(opti.bounded(-2500, Fx_var[k], 2500)) # Force limits

            T_total += dt_ds * (track_s[k+1] - track_s[k])
            
            # Regularization
            if k > 0:
                T_total += 0.1 * (U[0, k] - U[0, k-1])**2 # Smooth steering
                T_total += 1e-4 * (U[1, k] - U[1, k-1])**2 # Smooth throttle

        opti.minimize(T_total)

        # Boundary Conditions
        opti.subject_to(s_var[0] == track_s[0])
        opti.subject_to(n_var[0] == 0) 
        opti.subject_to(alpha_var[0] == 0)
        opti.subject_to(v_var[0] == 10.0) # Start speed
        
        # Initial Guess (Critical for convergence)
        opti.set_initial(v_var, 15.0)
        opti.set_initial(s_var, track_s)

        # Solver Options
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 2000,
            "tol": 1e-2,             # Relaxed tolerance for robustness
            "print_level": 4
            # REMOVED "accept_after_max_steps" to fix type mismatch crash
        }
        opti.solver("ipopt", p_opts, s_opts)
        
        try:
            sol = opti.solve()
            return {
                "s": sol.value(s_var),
                "n": sol.value(n_var),
                "v": sol.value(v_var),
                "delta": sol.value(delta_var),
                "time": sol.value(T_total)
            }
            
        except Exception as e:
            print(f"[OCP] Solver Exception: {e}")
            print("[OCP] Returning Fallback (Zero) Solution to keep Dashboard alive.")
            
            # Fallback: Return Arrays of Zeros matching the track shape
            # This prevents the pipeline from crashing so you can still open the dashboard
            zeros = np.zeros_like(track_s)
            return {
                "s": track_s,
                "n": zeros,
                "v": zeros + 10.0, # Flat speed line
                "delta": zeros,
                "time": 0.0,
                "error": str(e)
            }