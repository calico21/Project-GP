import os
import sys
import numpy as np
import casadi as ca
import scipy.linalg

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
except ImportError:
    print("[Error] acados is not installed. Ensure the acados Python interface is built.")
    sys.exit(1)

from models.vehicle_dynamics import DynamicBicycleModel
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class DiffWMPCSolver:
    """
    Differentiable Wavelet Model Predictive Control (Diff-WMPC) 
    integrated with Stochastic Tube-MPC.
    
    Dynamically exposes cost function weights to the Actor-Critic AI,
    allowing real-time, spatial adaptation of the solver's control logic.
    """
    def __init__(self, vehicle_params=None, tire_params=None):
        self.vp = vehicle_params if vehicle_params else VP_DICT
        self.tp = tire_params if tire_params else TP_DICT
        self.model_core = DynamicBicycleModel(self.vp, self.tp)
        self.f_dyn = self.model_core.get_equations()
        
        self.m = self.vp['m']
        self.Cd = self.vp.get('Cd', 0.8)
        self.A = self.vp.get('A', 1.0)
        self.rho = 1.225
        self.P_max = self.vp.get('power_max', 80000.0)
        self.F_brake_max = self.m * 9.81 * 1.5
        self.V_limit = self.vp.get('v_max', 100.0)
        
        self.T_scale = 100.0 
        self.kappa_safe = 1.96 

    def export_vehicle_model(self) -> AcadosModel:
        model = AcadosModel()
        model.name = 'diff_wmpc_thermal_twin'

        # Nominal States
        n       = ca.SX.sym('n')
        alpha   = ca.SX.sym('alpha')
        v       = ca.SX.sym('v')
        delta   = ca.SX.sym('delta')
        r       = ca.SX.sym('r')
        
        # Thermal States (Scaled)
        T_core_s  = ca.SX.sym('T_core_s')
        T_in_s    = ca.SX.sym('T_in_s')
        T_mid_s   = ca.SX.sym('T_mid_s')
        T_out_s   = ca.SX.sym('T_out_s')
        T_gas_s   = ca.SX.sym('T_gas_s')
        
        # Stochastic States (Covariance diagonals)
        var_n     = ca.SX.sym('var_n')     
        var_alpha = ca.SX.sym('var_alpha') 
        
        x = ca.vertcat(n, alpha, v, delta, r, T_core_s, T_in_s, T_mid_s, T_out_s, T_gas_s, var_n, var_alpha)
        
        # Controls
        d_delta = ca.SX.sym('d_delta')
        Fx_net  = ca.SX.sym('Fx_net')
        u = ca.vertcat(d_delta, Fx_net)
        
        # --- DIFF-WMPC PARAMETER LAYER ---
        # We augment the parameter vector to accept dynamic optimal control weights.
        # This bridges the C-based solver with the JAX Actor-Critic AI.
        k_c      = ca.SX.sym('k_c') 
        w_mu     = ca.SX.sym('w_mu') 
        w_steer  = ca.SX.sym('w_steer') # Dynamic steering penalty
        w_accel  = ca.SX.sym('w_accel') # Dynamic longitudinal penalty
        
        p = ca.vertcat(k_c, w_mu, w_steer, w_accel)

        x_mech = ca.vertcat(n, alpha, v, delta, r)
        dx_dt_mech = self.f_dyn(x_mech, u, k_c)
        
        # Thermal Unscaling
        T_core  = T_core_s * self.T_scale
        T_gas   = T_gas_s * self.T_scale
        T_ribs_f = [T_in_s * self.T_scale, T_mid_s * self.T_scale, T_out_s * self.T_scale]
        
        lf, lr = self.vp['lf'], self.vp['lr']
        alpha_f = delta - (alpha + lf*r/(v+0.1))
        
        Fz_aero = 0.5 * self.rho * self.vp['Cl'] * self.A * v**2
        Fz_f = (self.m * 9.81 * lr)/(lf + lr) + Fz_aero/2
        
        ay_approx = v * r
        roll_angle_rad = (self.m * ay_approx * 0.3) / 40000.0 
        gamma_f = -2.0 + (roll_angle_rad * 57.2958 * 0.6) 
        
        Fx_f, Fy_f = self.model_core.tire.compute_force(alpha_f, 0, Fz_f, gamma_f, T_ribs_f, T_gas, v)
        
        dT_ribs_f, dt_core_f, dt_gas_f = self.model_core.tire.compute_thermal_dynamics(
            Fx_f, Fy_f, Fz_f, gamma_f, alpha_f, 0, v, T_core, T_ribs_f, T_gas
        )
        
        dt_core_s = dt_core_f / self.T_scale
        dt_in_s   = dT_ribs_f[0] / self.T_scale
        dt_mid_s  = dT_ribs_f[1] / self.T_scale
        dt_out_s  = dT_ribs_f[2] / self.T_scale
        dt_gas_s  = dt_gas_f / self.T_scale
        
        dx_dt = ca.vertcat(dx_dt_mech, dt_core_s, dt_in_s, dt_mid_s, dt_out_s, dt_gas_s)
        
        # Spatial Domain Transformation
        v_safe = ca.fmax(v, 5.0)
        s_dot = (v_safe * ca.cos(alpha)) / (1 - n * k_c)
        dt_ds = 1.0 / (s_dot + 1e-3) 
        f_expl_nominal = dx_dt * dt_ds

        # Covariance Propagation (Lyapunov Equation on Manifold)
        J_spatial = ca.jacobian(f_expl_nominal[0:2], ca.vertcat(n, alpha))
        Q_n = 0.01 * w_mu * (v_safe**2) 
        Q_alpha = 0.05 * w_mu * ca.fabs(r)
        
        dvar_n_ds = 2.0 * J_spatial[0,0] * var_n + J_spatial[0,1] * var_alpha + Q_n
        dvar_alpha_ds = J_spatial[1,0] * var_n + 2.0 * J_spatial[1,1] * var_alpha + Q_alpha
        
        f_expl = ca.vertcat(f_expl_nominal, dvar_n_ds, dvar_alpha_ds)

        x_dot = ca.SX.sym('x_dot', x.shape)
        f_impl = x_dot - f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.p = p
        
        # Tube Boundaries
        safety_radius = self.kappa_safe * ca.sqrt(ca.fmax(var_n, 1e-6))
        h_expr = ca.vertcat(n + safety_radius, n - safety_radius)
        model.con_h_expr = h_expr
        
        # --- DIFF-WMPC COST FUNCTION ---
        # Cost is directly governed by the parameterized weights w_steer and w_accel
        model.cost_expr_ext_cost = dt_ds + w_steer * (d_delta**2) + w_accel * (Fx_net**2)
        
        return model

    def solve(self, track_s, track_k, track_w_left, track_w_right, friction_uncertainty_map=None, ai_cost_map=None, N=100):
        print("\n[acados] Compiling 12-State Diff-WMPC C-Code...")
        ocp = AcadosOcp()
        ocp.model = self.export_vehicle_model()
        
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = track_s[-1] - track_s[0]
        
        # Default Params: [k_c, w_mu, w_steer, w_accel]
        ocp.parameter_values = np.array([0.0, 0.01, 1e-2, 1e-7]) 
        
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost_e = ca.SX(0)

        ocp.constraints.lbx = np.array([-1.0, 1.0, -0.5, -5.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0])
        ocp.constraints.ubx = np.array([ 1.0, self.V_limit, 0.5, 5.0, 1.2, 1.6, 1.6, 1.6, 1.0, 2.0, 1.0])
        ocp.constraints.idxbx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        ocp.constraints.lbu = np.array([-5.0, -self.F_brake_max])
        ocp.constraints.ubu = np.array([ 5.0, self.P_max / 15.0])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.constraints.uh = np.array([2.0, 2.0])
        ocp.constraints.lh = np.array([-2.0, -2.0])

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'EXACT' 
        ocp.solver_options.ext_cost_num_hess = 1 
        
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        ocp.solver_options.regularize_method = 'CONVEXIFY'
        ocp.solver_options.nlp_solver_max_iter = 200
        ocp.solver_options.tol = 1e-3
        ocp.solver_options.print_level = 0 

        mu_est, g = 1.4, 9.81
        k0_safe = abs(track_k[0]) + 1e-4
        v0 = min(np.sqrt((mu_est * g) / k0_safe), self.V_limit)
        delta0 = track_k[0] * self.vp['lf']
        r0 = v0 * track_k[0]
        
        x0_arr = np.array([0.0, 0.0, v0, delta0, r0, 0.6, 0.6, 0.6, 0.6, 0.25, 0.0, 0.0])
        ocp.constraints.x0 = x0_arr

        acados_solver = AcadosOcpSolver(ocp, json_file='acados_diff_wmpc_ocp.json')

        w_mu_arr = friction_uncertainty_map if friction_uncertainty_map is not None else np.ones(N) * 0.02
        
        # Diff-WMPC Maps provided by the Actor-Critic AI (defaults if None)
        if ai_cost_map is None:
            w_steer_arr = np.ones(N) * 1e-2
            w_accel_arr = np.ones(N) * 1e-7
        else:
            w_steer_arr = ai_cost_map['w_steer']
            w_accel_arr = ai_cost_map['w_accel']

        for i in range(N):
            k_val = track_k[i]
            
            # Inject dynamic parameters into the solver at node 'i'
            p_val = np.array([k_val, w_mu_arr[i], w_steer_arr[i], w_accel_arr[i]])
            acados_solver.set(i, "p", p_val)
            
            lh_i = np.array([-10.0, -track_w_right[i]]) 
            uh_i = np.array([track_w_left[i], 10.0])    
            
            acados_solver.constraints_set(i, "lh", lh_i)
            acados_solver.constraints_set(i, "uh", uh_i)
            
            if i == 0:
                acados_solver.set(0, "x", x0_arr)
            else:
                k_safe = abs(k_val) + 1e-4
                v_guess = min(np.sqrt((mu_est * g) / k_safe), self.V_limit)
                x_init = np.array([0.0, 0.0, v_guess, k_val*self.vp['lf'], v_guess*k_val, 0.6, 0.6, 0.6, 0.6, 0.25, 0.01, 0.001])
                acados_solver.set(i, "x", x_init)
                
            acados_solver.set(i, "u", np.array([0.0, 0.0]))
            
        acados_solver.set(N, "p", np.array([track_k[-1], w_mu_arr[-1], w_steer_arr[-1], w_accel_arr[-1]]))

        print("[acados] Solving Diff-WMPC Stochastic OCP...")
        status = acados_solver.solve()
        
        if status not in [0, 2]:
            return {"error": f"Acados status {status}", "s": track_s}

        n_sol, v_sol, lat_g_sol = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1)
        var_n_sol = np.zeros(N+1)
        time_total = 0.0
        
        for i in range(N):
            x_res = acados_solver.get(i, "x")
            n_sol[i], v_sol[i] = x_res[0], x_res[2]
            var_n_sol[i] = x_res[10]
            lat_g_sol[i] = (v_sol[i]**2) * track_k[i] / 9.81
            
            s_dot = (x_res[2] * np.cos(x_res[1])) / (1 - x_res[0] * track_k[i])
            ds = track_s[i+1] - track_s[i] if i < N-1 else track_s[-1] - track_s[-2]
            time_total += ds / (s_dot + 1e-3)

        return {
            "s": track_s[:-1], 
            "n": n_sol[:-1], 
            "v": v_sol[:-1], 
            "lat_g": lat_g_sol[:-1],           
            "var_n": var_n_sol[:-1], 
            "time": float(time_total)
        }