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
        Solves the Optimal Control Problem (OCP) using RK4 Integration and Thermal constraints.
        """
        opti = ca.Opti() 
        
        # --- Decision Variables ---
        # State: [s, n, alpha, v, delta, r, T_tire]
        # Added T_tire (Index 6)
        n_states = 7
        X = opti.variable(n_states, N+1)
        
        s_var     = X[0, :]
        n_var     = X[1, :]
        alpha_var = X[2, :]
        v_var     = X[3, :]
        delta_var = X[4, :]
        r_var     = X[5, :]
        T_var     = X[6, :]

        # Control: [der_delta, Fx_req]
        U = opti.variable(2, N)
        d_delta_var = U[0, :]
        Fx_var      = U[1, :]
        
        # Parameters
        K_param = opti.parameter(N)
        opti.set_value(K_param, track_k[:-1]) 

        # --- Helper: System Dynamics (RK4 Ready) ---
        def get_spatial_derivatives(x, u, k_c):
            # 1. Extract Vehicle States (First 6)
            s, n, alpha, v, delta, r, T = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
            
            # CasADi slicing for Function call
            x_dyn = ca.vertcat(s, n, alpha, v, delta, r)
            
            # 2. Get Mechanical Derivatives (from Vehicle Model)
            # Returns [s_dot, n_dot, alpha_dot, v_dot, delta_dot, r_dot]
            dx_dt_mech = self.f_dyn(x_dyn, u, k_c)
            
            # 3. Calculate Thermal Derivative (dT/dt) manually
            # (We need to re-calculate forces here to get heat generation)
            # Approximation: We model the FRONT tire temp (dominant for steering limit)
            
            m, lf, lr = self.vp['m'], self.vp['lf'], self.vp['lr']
            rho, A, Cl = 1.225, self.vp['A'], self.vp.get('Cl', 3.0)
            
            # Kinematic Slip (Front)
            alpha_f = delta - (lf * r) / (v + 1.0) # +1.0 avoids singularity
            
            # Vertical Load (Front)
            Fz_aero = 0.5 * rho * Cl * A * v**2
            Fz_f = (m * 9.81 * lr)/(lf + lr) + Fz_aero/2
            
            # Compute Forces (using current Temp T)
            # We assume pure lateral for heat calc to keep OCP stable
            Fx_f, Fy_f = self.model.tire.compute_force(alpha_f, 0, Fz_f, Vx=v, T_tire=T)
            
            # Compute dT/dt
            dT_dt = self.model.tire.compute_thermal_dynamics(Fx_f, Fy_f, alpha_f, 0, v, T)
            
            # 4. Combine Derivatives
            dx_dt = ca.vertcat(dx_dt_mech, dT_dt)
            
            # 5. Convert Time -> Space Domain (dx/ds = dx/dt * dt/ds)
            # dt/ds = 1 / s_dot
            s_dot = dx_dt_mech[0]
            dt_ds = 1.0 / (s_dot + 1e-2) # Prevent division by zero
            
            return dx_dt * dt_ds, dt_ds

        # --- Objective: Minimize Time ---
        T_total = 0
        
        for k in range(N):
            ds = track_s[k+1] - track_s[k]
            k_c = K_param[k]
            
            # --- RK4 Integration Step ---
            # k1
            dx_ds1, dt_ds1 = get_spatial_derivatives(X[:, k], U[:, k], k_c)
            k1 = dx_ds1
            
            # k2
            x_k2 = X[:, k] + 0.5 * ds * k1
            dx_ds2, _ = get_spatial_derivatives(x_k2, U[:, k], k_c)
            k2 = dx_ds2
            
            # k3
            x_k3 = X[:, k] + 0.5 * ds * k2
            dx_ds3, _ = get_spatial_derivatives(x_k3, U[:, k], k_c)
            k3 = dx_ds3
            
            # k4
            x_k4 = X[:, k] + ds * k3
            dx_ds4, _ = get_spatial_derivatives(x_k4, U[:, k], k_c)
            k4 = dx_ds4
            
            # Final RK4 update
            x_next = X[:, k] + (ds / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            opti.subject_to(X[:, k+1] == x_next)
            
            # --- Path Constraints ---
            w_l = track_w_left[k]
            w_r = track_w_right[k]
            opti.subject_to(opti.bounded(-w_r, n_var[k], w_l))
            
            # --- Physical Constraints ---
            opti.subject_to(v_var[k] > 2.0) # Prevent stopping
            opti.subject_to(T_var[k] < 120.0) # Overheat limit
            opti.subject_to(T_var[k] > 20.0)  # Freeze limit
            
            # --- Accumulate Cost ---
            # Use dt_ds from k1 (start of interval approximation)
            T_total += dt_ds1 * ds
            
            # Regularization (Smoothness)
            if k > 0:
                T_total += 0.5 * (U[0, k] - U[0, k-1])**2 # Steering smoothness
                T_total += 1e-4 * (U[1, k] - U[1, k-1])**2 # Throttle smoothness

        opti.minimize(T_total)

        # --- Boundary Conditions ---
        opti.subject_to(s_var[0] == track_s[0])
        opti.subject_to(n_var[0] == 0) 
        opti.subject_to(alpha_var[0] == 0)
        opti.subject_to(v_var[0] == 15.0) # Flying lap start
        opti.subject_to(T_var[0] == 60.0) # Start with warm tires
        
        # --- Initial Guesses ---
        opti.set_initial(v_var, 20.0)
        opti.set_initial(s_var, track_s)
        opti.set_initial(T_var, 60.0)

        # --- Solver Settings ---
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 3000,
            "tol": 1e-3, # Looser tolerance for complex RK4/Thermal constraints
            "print_level": 4,
            "mumps_mem_percent": 100 # Allow more memory
        }
        opti.solver("ipopt", p_opts, s_opts)
        
        try:
            sol = opti.solve()
            return {
                "s": sol.value(s_var),
                "n": sol.value(n_var),
                "v": sol.value(v_var),
                "delta": sol.value(delta_var),
                "T_tire": sol.value(T_var),
                "time": sol.value(T_total)
            }
            
        except Exception as e:
            print(f"[OCP] Solver Exception: {e}")
            print("[OCP] Returning Fallback (Zero) Solution.")
            
            zeros = np.zeros_like(track_s)
            return {
                "s": track_s,
                "n": zeros,
                "v": zeros + 10.0,
                "delta": zeros,
                "T_tire": zeros + 60.0,
                "time": 0.0,
                "error": str(e)
            }