import casadi as ca
import numpy as np
from models.vehicle_dynamics import DynamicBicycleModel
# Load configs
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class OptimalLapSolver:
    """
    State-of-the-Art OCP Solver using Radau IIA Orthogonal Collocation.
    Includes 3-Node Thermal-Pressure coupling and strict L-stability integration.
    """
    def __init__(self, vehicle_params=None, tire_params=None):
        self.vp = vehicle_params if vehicle_params else VP_DICT
        self.tp = tire_params if tire_params else TP_DICT
        self.model = DynamicBicycleModel(self.vp, self.tp)
        self.f_dyn = self.model.get_equations() 

    def solve(self, track_s, track_k, track_w_left, track_w_right, N=100):
        """
        Solves the Minimum Lap Time problem.
        """
        opti = ca.Opti() 
        
        # --- 1. PRE-COMPUTE WARM START (CRITICAL FOR SKIDPAD) ---
        mu_est = 1.4 
        g = 9.81
        v_guess = []
        for k_val in track_k:
            k_safe = abs(k_val) + 1e-4
            v_lim = np.sqrt((mu_est * g) / k_safe)
            v_lim = min(v_lim, self.vp.get('v_max', 100.0))
            v_guess.append(v_lim * 0.95) 
        
        v_guess = np.array(v_guess)
        v_guess[0] = 15.0 

        # --- 2. DECISION VARIABLES ---
        n_states = 8 # [n, alpha, v, delta, r, T_core, T_surf, T_gas]
        X = opti.variable(n_states, N+1)
        
        n_var       = X[0, :]
        alpha_var   = X[1, :]
        v_var       = X[2, :]
        delta_var   = X[3, :]
        r_var       = X[4, :]
        T_core_var  = X[5, :]
        T_surf_var  = X[6, :]
        T_gas_var   = X[7, :] # New: Inflation Gas Node

        # Controls (N segments)
        U = opti.variable(2, N)
        d_delta_var = U[0, :]
        Fx_net_var  = U[1, :]
        
        K_param = opti.parameter(N)
        opti.set_value(K_param, track_k[:-1]) 

        # Physics Constants
        m = self.vp['m']
        Cd = self.vp.get('Cd', 0.8)
        A = self.vp.get('A', 1.0)
        rho = 1.225
        P_max = self.vp.get('power_max', 80000.0) 
        F_brake_max = self.vp['m'] * 9.81 * 1.5   
        V_limit = self.vp.get('v_max', 100.0)

        # --- 3. DYNAMICS HELPER ---
        def get_spatial_derivatives(x, u, k_c):
            n, alpha, v, delta, r, T_core, T_surf, T_gas = \
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
            
            # Mech Dynamics
            x_mech = ca.vertcat(n, alpha, v, delta, r)
            dx_dt_mech = self.f_dyn(x_mech, u, k_c)
            
            # 3-Node Thermal Dynamics
            lf, lr = self.vp['lf'], self.vp['lr']
            alpha_f = delta - (alpha + lf*r/(v+0.1)) 
            
            Fz_aero = 0.5 * rho * self.vp['Cl'] * A * v**2
            Fz_f = (m * 9.81 * lr)/(lf + lr) + Fz_aero/2
            
            # Compute dT/dt with Pressure-Coupling
            Fx_f, Fy_f = self.model.tire.compute_force(alpha_f, 0, Fz_f, T_surf, T_gas)
            
            dt_core_f, dt_surf_f, dt_gas_f = self.model.tire.compute_thermal_dynamics(
                Fx_f, Fy_f, Fz_f, alpha_f, 0, v, T_core, T_surf, T_gas
            )
            
            dx_dt = ca.vertcat(dx_dt_mech, dt_core_f, dt_surf_f, dt_gas_f)
            
            # Space Conversion
            s_dot = (v * ca.cos(alpha)) / (1 - n * k_c)
            dt_ds = 1.0 / (s_dot + 1e-1) 
            
            return dx_dt * dt_ds, dt_ds

        # --- 4. RADAU IIA (3rd ORDER) OPTIMIZATION LOOP ---
        T_total = 0
        
        for k in range(N):
            ds = track_s[k+1] - track_s[k]
            k_c = K_param[k]
            
            X_k = X[:, k]
            X_next = X[:, k+1]
            U_k = U[:, k]
            
            # Radau IIA Collocation Variables (2 Stages)
            K1 = opti.variable(n_states)
            K2 = opti.variable(n_states)
            
            # Internal Stage States based on Radau Butcher Tableau
            # c = [1/3, 1], A = [[5/12, -1/12], [3/4, 1/4]]
            X_c1 = X_k + ds * ((5/12)*K1 - (1/12)*K2)
            X_c2 = X_k + ds * ((3/4)*K1 + (1/4)*K2)
            
            # Evaluate Derivatives at Collocation Points
            dx_ds_1, dt_ds_1 = get_spatial_derivatives(X_c1, U_k, k_c)
            dx_ds_2, dt_ds_2 = get_spatial_derivatives(X_c2, U_k, k_c)
            
            # Collocation Constraints
            opti.subject_to(K1 == dx_ds_1)
            opti.subject_to(K2 == dx_ds_2)
            
            # Continuity Constraint (For Radau IIA, the final stage is the node end)
            opti.subject_to(X_next == X_c2)
            
            # Quadrature for Minimum Time Objective
            # b = [3/4, 1/4]
            step_time = ds * ((3/4)*dt_ds_1 + (1/4)*dt_ds_2)
            T_total += step_time

            # --- CONSTRAINTS ---
            v_k = v_var[k]
            
            F_drag = 0.5 * rho * Cd * A * v_k**2
            F_eng_limit = P_max / (v_k + 1.0)
            opti.subject_to(Fx_net_var[k] <= (F_eng_limit - F_drag))
            opti.subject_to(Fx_net_var[k] >= (-F_brake_max - F_drag))
            
            w_l = track_w_left[k]
            w_r = track_w_right[k]
            opti.subject_to(opti.bounded(-w_r, n_var[k], w_l))
            
            opti.subject_to(v_k > 5.0)
            opti.subject_to(v_k <= V_limit)
            
            opti.subject_to(T_core_var[k] < 120.0) 
            opti.subject_to(T_surf_var[k] < 160.0)
            
            if k > 0:
                opti.subject_to(opti.bounded(-5.0, U[0, k] - U[0, k-1], 5.0))

        opti.minimize(T_total)

        # --- 5. BOUNDARY CONDITIONS ---
        opti.subject_to(n_var[0] == 0)
        opti.subject_to(alpha_var[0] == 0)
        opti.subject_to(v_var[0] == 15.0) 
        opti.subject_to(delta_var[0] == 0)
        opti.subject_to(T_core_var[0] == 60.0)
        opti.subject_to(T_surf_var[0] == 60.0)
        opti.subject_to(T_gas_var[0] == 25.0) # Start with cold tire pressures
        
        # --- 6. INITIAL GUESS (WARM START) ---
        opti.set_initial(v_var, v_guess)
        opti.set_initial(T_core_var, 60.0)
        opti.set_initial(T_surf_var, 60.0)
        opti.set_initial(T_gas_var, 25.0)
        opti.set_initial(r_var, v_guess * track_k)

        # --- 7. SOLVER SETTINGS ---
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 5000,
            "tol": 1e-3,
            "print_level": 5,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based"
        }
        opti.solver("ipopt", p_opts, s_opts)
        
        try:
            sol = opti.solve()
            
            v_sol = sol.value(v_var)
            lat_g_sol = (v_sol**2) * track_k / 9.81
            
            return {
                "s": track_s,
                "n": sol.value(n_var),        
                "v": v_sol,                   
                "lat_g": lat_g_sol,           
                "T_core": sol.value(T_core_var),
                "T_surf": sol.value(T_surf_var),
                "T_gas": sol.value(T_gas_var),
                "time": float(sol.value(T_total)) 
            }
        except Exception as e:
            print(f"[OCP] Solver Failed: {e}")
            return {
                "error": str(e), 
                "s": track_s,
                "v": opti.debug.value(v_var)
            }