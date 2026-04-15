import casadi as ca
from acados_template import AcadosSim, AcadosSimSolver, AcadosModel
from system_dynamics import ASystemDynamics
from typing import Literal
import numpy as np

class AcadosSimBuilder:
    def __init__(self, system_dynamics: ASystemDynamics, model_type: Literal["explicit", "implicit", "timescaling_implicit"], T: float):

        #---------------------------------
        # X and U Vectors (State and Control)
        #---------------------------------

        X = ca.MX.sym('X', 6)   # [x, y, phi, u, v, omega]
        if model_type == "timescaling_implicit":
            U = ca.MX.sym('U', 3)   # [ax, delta, dt]
        else:
            U = ca.MX.sym('U', 2)   # [ax, delta]
        X_dot = ca.MX.sym('X_Dot', 6)

        X_dot_func = system_dynamics.get_system_dynamics()
        #---------------------------------
        # Acados model
        #---------------------------------#
        self.model = AcadosModel()
        self.model.name = system_dynamics.get_model_name()+(''.join(word.title() for word in model_type.split('_')))
        self.model.x = X
        self.model.u = U
        self.model.xdot = X_dot
        if model_type == "explicit":
            self.model.f_expl_expr = X_dot_func(X, U)
        elif model_type == "implicit":
            self.model.f_impl_expr = X_dot - X_dot_func(X, U)
        elif model_type == "timescaling_implicit":
            dt = U[2]
            self.model.f_impl_expr = X_dot - X_dot_func(X, U[:2]) * ca.fmax(dt, 0.001)
        else:
            raise TypeError(f"Unknown/unimplemented model type: {model_type}")

        # Acados simulation
        self.sim = AcadosSim()
        self.sim.model = self.model
        self.sim.solver_options.num_steps = 1            # One step per iteration
        self.sim.solver_options.T = T
        if model_type == "explicit":
            self.sim.solver_options.num_stages = 4           # RK4
            self.sim.solver_options.integrator_type = 'ERK'  # Explicit RK
        elif model_type == "implicit":
            self.sim.solver_options.num_stages = 2           # RK2
            self.sim.solver_options.integrator_type = 'IRK'  # Implicit RK
        elif model_type == "timescaling_implicit":
            self.sim.solver_options.num_stages = 1           # RK1 (backwards euler) es el unico IRK que garantiza estabilidad con time scaling
            self.sim.solver_options.integrator_type = 'IRK'  # Implicit RK
        else:
            raise TypeError(f"Unknown/unimplemented model type: {model_type}")
        
    def generate(self, code_export_dir: str):
        self.sim.code_export_directory = code_export_dir
        self.sim_solver = AcadosSimSolver(self.sim, json_file=code_export_dir+"/acados_sim.json", generate=True)