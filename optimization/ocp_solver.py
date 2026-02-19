import os
import sys
import numpy as np
import casadi as ca
import scipy.linalg

try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
except ImportError:
    print("[Error] acados is not installed.")
    sys.exit(1)

from models.vehicle_dynamics import DynamicBicycleModel
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class OptimalLapSolver:
    """
    State-of-the-Art OCP Solver using acados SQP.
    Includes 5-Node Asymmetric Thermal-Pressure coupling with EXPLICIT STATE SCALING.
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

    def export_vehicle_model(self) -> AcadosModel:
        model = AcadosModel()
        model.name = 'formula_student_thermal_twin'

        n       = ca.SX.sym('n')
        alpha   = ca.SX.sym('alpha')
        v       = ca.SX.sym('v')
        delta   = ca.SX.sym('delta')
        r       = ca.SX.sym('r')
        
        T_core_s  = ca.SX.sym('T_core_s')
        T_in_s    = ca.SX.sym('T_in_s')
        T_mid_s   = ca.SX.sym('T_mid_s')
        T_out_s   = ca.SX.sym('T_out_s')
        T_gas_s   = ca.SX.sym('T_gas_s')
        
        x = ca.vertcat(n, alpha, v, delta, r, T_core_s, T_in_s, T_mid_s, T_out_s, T_gas_s)
        
        d_delta = ca.SX.sym('d_delta')
        Fx_net  = ca.SX.sym('Fx_net')
        u = ca.vertcat(d_delta, Fx_net)
        
        k_c = ca.SX.sym('k_c') 
        p = ca.vertcat(k_c)

        x_mech = ca.vertcat(n, alpha, v, delta, r)
        dx_dt_mech = self.f_dyn(x_mech, u, k_c)
        
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
        
        v_safe = ca.fmax(v, 5.0)
        s_dot = (v_safe * ca.cos(alpha)) / (1 - n * k_c)
        dt_ds = 1.0 / (s_dot + 1e-3) 
        f_expl = dx_dt * dt_ds

        # --- THE FIX: RESTORE IMPLICIT DIFFERENTIAL EQUATION FOR IRK ---
        x_dot = ca.SX.sym('x_dot', x.shape)
        f_impl = x_dot - f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.p = p
        
        R_steer = 1e-2
        R_accel = 1e-7
        model.cost_expr_ext_cost = dt_ds + R_steer * (d_delta**2) + R_accel * (Fx_net**2)
        
        return model

    def solve(self, track_s, track_k, track_w_left, track_w_right, N=100):
        print("\n[acados] Compiling 10-State Scaled Multi-Rib OCP C-Code...")
        ocp = AcadosOcp()
        ocp.model = self.export_vehicle_model()
        
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = track_s[-1] - track_s[0]
        ocp.parameter_values = np.array([0.0])
        
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost_e = ca.SX(0)

        ocp.constraints.lbx = np.array([-5.0, -1.0, 1.0, -0.5, -5.0, 0.2, 0.2, 0.2, 0.2, 0.2])
        ocp.constraints.ubx = np.array([ 5.0,  1.0, self.V_limit, 0.5, 5.0, 1.2, 1.6, 1.6, 1.6, 1.0])
        ocp.constraints.idxbx = np.array(range(10))

        ocp.constraints.lbu = np.array([-5.0, -self.F_brake_max])
        ocp.constraints.ubu = np.array([ 5.0, self.P_max / 15.0])
        ocp.constraints.idxbu = np.array([0, 1])

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'EXACT' 
        ocp.solver_options.ext_cost_num_hess = 1 
        
        # --- THE FIX: IRK INTEGRATOR FOR STIFF THERMAL DYNAMICS ---
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.sim_method_num_stages = 2
        ocp.solver_options.sim_method_num_steps = 2
        
        ocp.solver_options.qp_solver_cond_N = max(1, N // 10) 
        ocp.solver_options.qp_solver_iter_max = 50
        
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        
        # --- THE FIX: CONVEXIFY REGULARIZATION ---
        ocp.solver_options.regularize_method = 'CONVEXIFY'
        
        ocp.solver_options.nlp_solver_max_iter = 200
        ocp.solver_options.tol = 1e-3
        ocp.solver_options.print_level = 0 

        mu_est, g = 1.4, 9.81
        k0_safe = abs(track_k[0]) + 1e-4
        v0 = min(np.sqrt((mu_est * g) / k0_safe), self.V_limit)
        delta0 = track_k[0] * self.vp['lf']
        r0 = v0 * track_k[0]
        
        ocp.constraints.x0 = np.array([0.0, 0.0, v0, delta0, r0, 0.6, 0.6, 0.6, 0.6, 0.25])

        acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

        for i in range(N):
            k_val = track_k[i]
            acados_solver.set(i, "p", np.array([k_val]))
            
            if i > 0:
                lbx_i = np.array([-track_w_right[i], -1.0, 1.0, -0.5, -5.0, 0.2, 0.2, 0.2, 0.2, 0.2])
                ubx_i = np.array([ track_w_left[i],   1.0, self.V_limit, 0.5, 5.0, 1.2, 1.6, 1.6, 1.6, 1.0])
                acados_solver.constraints_set(i, "lbx", lbx_i)
                acados_solver.constraints_set(i, "ubx", ubx_i)
            
            if i == 0:
                acados_solver.set(0, "x", ocp.constraints.x0)
            else:
                k_safe = abs(k_val) + 1e-4
                v_guess = min(np.sqrt((mu_est * g) / k_safe), self.V_limit)
                x_init = np.array([0.0, 0.0, v_guess, k_val*self.vp['lf'], v_guess*k_val, 0.6, 0.6, 0.6, 0.6, 0.25])
                acados_solver.set(i, "x", x_init)
                
            acados_solver.set(i, "u", np.array([0.0, 0.0]))
            
        acados_solver.set(N, "p", np.array([track_k[-1]]))

        print("[acados] Solving 10-State Multi-Rib OCP...")
        status = acados_solver.solve()
        
        if status not in [0, 2]:
            return {"error": f"Acados status {status}", "s": track_s}

        n_sol, v_sol, lat_g_sol = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1)
        T_core_sol, T_in_sol, T_mid_sol, T_out_sol, T_gas_sol = [np.zeros(N+1) for _ in range(5)]
        time_total = 0.0
        
        for i in range(N):
            x_res = acados_solver.get(i, "x")
            n_sol[i], v_sol[i] = x_res[0], x_res[2]
            lat_g_sol[i] = (v_sol[i]**2) * track_k[i] / 9.81
            
            T_core_sol[i] = x_res[5] * self.T_scale
            T_in_sol[i]   = x_res[6] * self.T_scale
            T_mid_sol[i]  = x_res[7] * self.T_scale
            T_out_sol[i]  = x_res[8] * self.T_scale
            T_gas_sol[i]  = x_res[9] * self.T_scale
            
            s_dot = (x_res[2] * np.cos(x_res[1])) / (1 - x_res[0] * track_k[i])
            ds = track_s[i+1] - track_s[i] if i < N-1 else track_s[-1] - track_s[-2]
            time_total += ds / (s_dot + 1e-3)

        return {
            "s": track_s[:-1], "n": n_sol[:-1], "v": v_sol[:-1], "lat_g": lat_g_sol[:-1],           
            "T_core": T_core_sol[:-1], "T_in": T_in_sol[:-1], "T_mid": T_mid_sol[:-1], 
            "T_out": T_out_sol[:-1], "T_gas": T_gas_sol[:-1], "time": float(time_total)
        }