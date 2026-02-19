import os
import sys
import numpy as np
import casadi as ca
import scipy.linalg

# --- ACADOS IMPORTS (The SOTA Upgrade) ---
try:
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
except ImportError:
    print("[Error] acados is not installed. Please install acados and acados_template.")
    sys.exit(1)

from models.vehicle_dynamics import DynamicBicycleModel
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class OptimalLapSolver:
    """
    State-of-the-Art OCP Solver using acados SQP.
    Includes 3-Node Thermal-Pressure coupling and strict L-stability integration via IRK.
    """
    def __init__(self, vehicle_params=None, tire_params=None):
        self.vp = vehicle_params if vehicle_params else VP_DICT
        self.tp = tire_params if tire_params else TP_DICT
        self.model_core = DynamicBicycleModel(self.vp, self.tp)
        self.f_dyn = self.model_core.get_equations()
        
        # Physics Constants
        self.m = self.vp['m']
        self.Cd = self.vp.get('Cd', 0.8)
        self.A = self.vp.get('A', 1.0)
        self.rho = 1.225
        self.P_max = self.vp.get('power_max', 80000.0)
        self.F_brake_max = self.m * 9.81 * 1.5
        self.V_limit = self.vp.get('v_max', 100.0)

    def export_vehicle_model(self) -> AcadosModel:
        """Constructs the symbolic CasADi model to be compiled into C code by acados."""
        model = AcadosModel()
        model.name = 'formula_student_thermal_twin'

        # --- States ---
        n       = ca.SX.sym('n')
        alpha   = ca.SX.sym('alpha')
        v       = ca.SX.sym('v')
        delta   = ca.SX.sym('delta')
        r       = ca.SX.sym('r')
        T_core  = ca.SX.sym('T_core')
        T_surf  = ca.SX.sym('T_surf')
        T_gas   = ca.SX.sym('T_gas')
        x = ca.vertcat(n, alpha, v, delta, r, T_core, T_surf, T_gas)

        # --- Controls ---
        d_delta = ca.SX.sym('d_delta')
        Fx_net  = ca.SX.sym('Fx_net')
        u = ca.vertcat(d_delta, Fx_net)

        # --- Parameters (Spatial Curvature) ---
        k_c = ca.SX.sym('k_c') 
        p = ca.vertcat(k_c)

        # --- Dynamics ---
        x_mech = ca.vertcat(n, alpha, v, delta, r)
        dx_dt_mech = self.f_dyn(x_mech, u, k_c)
        
        # Thermal Dynamics Computation
        lf, lr = self.vp['lf'], self.vp['lr']
        alpha_f = delta - (alpha + lf*r/(v+0.1))
        
        Fz_aero = 0.5 * self.rho * self.vp['Cl'] * self.A * v**2
        Fz_f = (self.m * 9.81 * lr)/(lf + lr) + Fz_aero/2
        
        Fx_f, Fy_f = self.model_core.tire.compute_force(alpha_f, 0, Fz_f, T_surf, T_gas)
        dt_core_f, dt_surf_f, dt_gas_f = self.model_core.tire.compute_thermal_dynamics(
            Fx_f, Fy_f, Fz_f, alpha_f, 0, v, T_core, T_surf, T_gas
        )
        
        dx_dt = ca.vertcat(dx_dt_mech, dt_core_f, dt_surf_f, dt_gas_f)
        
        # Time to Space Conversion (Spatial Formulation)
        s_dot = (v * ca.cos(alpha)) / (1 - n * k_c)
        dt_ds = 1.0 / (s_dot + 1e-3) 
        
        f_expl = dx_dt * dt_ds

        # Define Implicit ODEs for the IRK integrator (crucial for thermal stiffness)
        x_dot = ca.SX.sym('x_dot', x.shape)
        f_impl = x_dot - f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.p = p
        
        # Cost: Minimize Time -> Minimize integral of dt/ds
        model.cost_expr_ext_cost = dt_ds 
        
        return model

    def solve(self, track_s, track_k, track_w_left, track_w_right, N=100):
        print("\n[acados] Compiling C-Code & Initializing SQP Solver...")
        
        # 1. Setup Acados OCP
        ocp = AcadosOcp()
        ocp.model = self.export_vehicle_model()
        ocp.dims.N = N
        
        # Track Length parameters
        S_total = track_s[-1] - track_s[0]
        ocp.solver_options.tf = S_total 
        
        # 2. Setup Cost & Constraints
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.model.cost_expr_ext_cost_e = ca.SX.zeros(1) # Terminal cost is zero

        # State Bounds
        # x = [n, alpha, v, delta, r, T_core, T_surf, T_gas]
        ocp.constraints.lbx = np.array([-5.0, -1.0, 5.0, -0.5, -5.0, 20.0, 20.0, 20.0])
        ocp.constraints.ubx = np.array([ 5.0,  1.0, self.V_limit, 0.5, 5.0, 120.0, 160.0, 100.0])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 6, 7])

        # Control Bounds [d_delta, Fx_net]
        ocp.constraints.lbu = np.array([-5.0, -self.F_brake_max])
        ocp.constraints.ubu = np.array([ 5.0, self.P_max / 15.0]) # Approx max force
        ocp.constraints.idxbu = np.array([0, 1])

        # Initial conditions bounds
        ocp.constraints.x0 = np.array([0.0, 0.0, 15.0, 0.0, 0.0, 60.0, 60.0, 25.0])

        # 3. Solver Options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # BLASFEO backend
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK' # Implicit Runge Kutta (L-Stable)
        ocp.solver_options.nlp_solver_type = 'SQP' # Real-Time iteration capability
        ocp.solver_options.nlp_solver_max_iter = 150
        ocp.solver_options.tol = 1e-4

        # 4. Create Solver
        acados_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')

        # 5. Initialization / Warm Start
        mu_est = 1.4
        g = 9.81
        for i in range(N):
            k_val = track_k[i]
            k_safe = abs(k_val) + 1e-4
            v_guess = min(np.sqrt((mu_est * g) / k_safe), self.V_limit)
            
            # Set parameters (curvature)
            acados_solver.set(i, "p", np.array([k_val]))
            
            # Update Track Boundaries dynamically via constraints
            lbx_i = np.array([-track_w_right[i], -1.0, 5.0, -0.5, -5.0, 20.0, 20.0, 20.0])
            ubx_i = np.array([ track_w_left[i],   1.0, self.V_limit, 0.5, 5.0, 120.0, 160.0, 100.0])
            acados_solver.constraints_set(i, "lbx", lbx_i)
            acados_solver.constraints_set(i, "ubx", ubx_i)
            
            # Warm Start States
            x_init = np.array([0.0, 0.0, v_guess, 0.0, v_guess * k_val, 60.0, 60.0, 25.0])
            acados_solver.set(i, "x", x_init)
            acados_solver.set(i, "u", np.array([0.0, 0.0]))
            
        # Terminal Node Parameter
        acados_solver.set(N, "p", np.array([track_k[-1]]))

        # 6. Solve NLP
        print("[acados] Solving OCP...")
        status = acados_solver.solve()
        
        if status not in [0, 2]: # 0 is success, 2 is max_iter (often still usable)
            print(f"[acados] Solver Failed with status {status}")
            return {"error": f"Acados status {status}", "s": track_s}

        # 7. Extract Results
        print(f"[acados] Optimization Completed in {acados_solver.get_stats('time_tot')*1000:.2f} ms")
        
        n_sol = np.zeros(N+1)
        v_sol = np.zeros(N+1)
        lat_g_sol = np.zeros(N+1)
        T_core_sol = np.zeros(N+1)
        T_surf_sol = np.zeros(N+1)
        T_gas_sol = np.zeros(N+1)
        time_total = 0.0
        
        for i in range(N):
            x_res = acados_solver.get(i, "x")
            n_sol[i] = x_res[0]
            v_sol[i] = x_res[2]
            lat_g_sol[i] = (v_sol[i]**2) * track_k[i] / 9.81
            T_core_sol[i] = x_res[5]
            T_surf_sol[i] = x_res[6]
            T_gas_sol[i] = x_res[7]
            
            # Reconstruct time 
            s_dot = (x_res[2] * np.cos(x_res[1])) / (1 - x_res[0] * track_k[i])
            ds = track_s[i+1] - track_s[i] if i < N-1 else track_s[-1] - track_s[-2]
            time_total += ds / (s_dot + 1e-3)

        return {
            "s": track_s,
            "n": n_sol[:-1],        
            "v": v_sol[:-1],                   
            "lat_g": lat_g_sol[:-1],           
            "T_core": T_core_sol[:-1],
            "T_surf": T_surf_sol[:-1],
            "T_gas": T_gas_sol[:-1],
            "time": float(time_total)
        }