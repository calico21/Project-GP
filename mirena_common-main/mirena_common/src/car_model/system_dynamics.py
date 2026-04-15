import casadi as ca
import numpy as np

class SymCarParameters:

    class Defaults():
        """
        Car Parameters
        ----------
        m [kg] : Total vehicle mass. 
        Iz[kg*m²] : Yaw moment of inertia
        Lf [m]: Distance from center of gravity (CG) to front axle.
        Lr [m]: Distance from center of gravity (CG) to rear axle.     
        kf [N/rad]: Front axle cornering stiffness.
        kr [N/rad]: Rear axle cornering stiffness:   
        """
        m: float = 250      # kg - Total vehicle mass
        Iz: float = 125     # kg*m² - Yaw moment of inertia  
        Lf: float = 0.8     # m - Distance from CG to front axle
        Lr: float = 0.75      # m - Distance from CG to rear axle
        kf: float = 35000    # N/rad - Front axle cornering stiffness
        kr: float = 35000    # N/rad - Rear axle cornering stiffness

        C_M = 1.3*m     # Coeficiente de comanda
        C_R0 = 0        # Coeficiente de fricción
        C_R2 = 0        # Coeficiente cuadratico de drag

        B = 8.6
        C = 1.64
        D = 2.23

    N_OF_PARAMS = 12

    def __init__(self, name: str):
        self._params = ca.MX.sym(name, SymCarParameters.N_OF_PARAMS)

    def get_sym(self) -> ca.MX:
        return self._params

    @property
    def m(self):
        return self._params[0]
    
    @property
    def Iz(self):
        return self._params[1]
    
    @property
    def Lf(self):
        return self._params[2]
    
    @property
    def Lr(self):
        return self._params[3]
    
    @property
    def kf(self):
        return self._params[4]
    
    @property
    def kr(self):
        return self._params[5]
    
    @property
    def C_M(self):
        return self._params[6]
    
    @property
    def C_R0(self):
        return self._params[7]
    
    @property
    def C_R2(self):
        return self._params[8]
    
    @property
    def B(self):
        return self._params[9]
    
    @property
    def C(self):
        return self._params[10]
    
    @property
    def D(self):
        return self._params[11]
    
    @staticmethod
    def get_default_parameters_getter() -> ca.Function:
        """
        Returns a casadi function that takes no parameters and returns a symbolic matrix of default car parameters
        Done this way so that this shit can be exported to C as a casadi function lmao this is bad design
        Use:
        param_getter = SymCarParameters.get_default_parameters_getter()
        params_sym: ca.DM = param_getter(0)
        El 0 es dummy porque la interfaz de casadi es estupida y no te deja crear funciones sin argumentos de forma comoda ugh 
        """
        return ca.Function('CarParameters_get_defaults', [ca.MX.sym("dummy")], [ca.vertcat(
            SymCarParameters.Defaults.m,
            SymCarParameters.Defaults.Iz,
            SymCarParameters.Defaults.Lf,
            SymCarParameters.Defaults.Lr,
            SymCarParameters.Defaults.kf,
            SymCarParameters.Defaults.kr,
            SymCarParameters.Defaults.C_M,
            SymCarParameters.Defaults.C_R0,
            SymCarParameters.Defaults.C_R2,
            SymCarParameters.Defaults.B,
            SymCarParameters.Defaults.C,
            SymCarParameters.Defaults.D
        )])

class ASystemDynamics:
    def __init__(self):
        raise AssertionError("Called method on abstract class")
    
    def get_model_name(self) -> str:
        raise AssertionError("Called method on abstract class")
 
    def get_system_dynamics(self) -> ca.Function:
        raise AssertionError("Called method on abstract class")

class DynamicBicycleSystemDynamics(ASystemDynamics):
    def __init__(self):
        # ----------------------
        # State & Control
        # ----------------------
        
        X = ca.MX.sym('X', 6)  # [x, y, phi, u, v, omega]
        U = ca.MX.sym('U', 2)  # [ax, delta] 
        _param = SymCarParameters("P")

        phi, u, v, omega = X[2], X[3], X[4], X[5]
        g, delta = U[0], U[1]

        # ----------------------
        # Continous-time dynamics: X_dot = dX/dt
        # ----------------------
        
        epsilon = 1; # For aysntote prevention 
        
        Fy1 = - _param.kf * ((v + _param.Lf * omega) * u / (u**2 + epsilon**2) - delta)  # front sideslip force
        Fy2 = - _param.kr * ((v - _param.Lr * omega) * u / (u**2 + epsilon**2))           # rear sideslip force

        f_x = _param.C_M * g - _param.C_R0 - _param.C_R2 * u**2
        a = f_x /_param.m

        X_dot  = ca.Function('StateDerivative', [X, U, _param.get_sym()], [ca.vertcat(
            u * ca.cos(phi) - v * ca.sin(phi),      # x_dot
            u * ca.sin(phi) + v * ca.cos(phi),      # y_dot  
            omega,                                  # phi_dot
            a + v * omega - (1/_param.m) * Fy1 * ca.sin(delta),  # u_dot
            - u * omega + (1/_param.m) * (Fy1 * ca.cos(delta) + Fy2),  # v_dot
            (1/_param.Iz) * (_param.Lf * Fy1 * ca.cos(delta) - _param.Lr * Fy2)    # omega_dot
        )])

        self._system_dynamics = X_dot
    
    def get_model_name(self) -> str:
        return "DynamicBicycle"

    def get_system_dynamics(self) -> ca.Function:
        """
        Get system dynamics: X_dot(X, U, P), where:
        X = [x, y, phi, u, v, omega]
        U = [ax, delta] 
        P = Parameters. Get them via CarParameters__get_defaults
        """
        return self._system_dynamics
    

class KinematicBicycleSystemDynamics(ASystemDynamics):
    def __init__(self):
        # ----------------------
        # State & Control
        # ----------------------
        
        X = ca.MX.sym('X', 6)  # [x, y, phi, u, v, omega]
        U = ca.MX.sym('U', 2)  # [ax, delta] 
        _param = SymCarParameters("P")

        phi, u, v, omega = X[2], X[3], X[4], X[5]
        g, delta = U[0], U[1]

        # ----------------------
        # Continous-time dynamics: X_dot = dX/dt
        # ----------------------
        f_x = _param.C_M * g - _param.C_R0 - _param.C_R2 * u**2
        a = f_x /_param.m

        X_dot  = ca.Function('StateDerivative', [X, U, _param.get_sym()], [ca.vertcat(
            u*ca.cos(phi) - v*ca.sin(phi),          # X_dot
            u*ca.sin(phi) + v*ca.cos(phi),          # Y_dot
            ca.tan(delta)*u/(_param.Lf+_param.Lr),  # Phi_dot
            a,                                      # u_dot
            0,                                      # v_dot
            0                                       # omega_dot
        )])
        
        self._system_dynamics = X_dot
    
    def get_model_name(self) -> str:
        return "KinematicBicycle"
 
    def get_system_dynamics(self) -> ca.Function:
        """
        Get system dynamics: X_dot(X, U, P), where:
        X = [x, y, phi, u, v, omega]
        U = [ax, delta] 
        P = Parameters. Get them via CarParameters__get_defaults
        """
        return self._system_dynamics
    

class BlendedBicycleSystemDynamics(ASystemDynamics):
    def __init__(self):
        # ----------------------
        # State & Control
        # ----------------------
        
        X = ca.MX.sym('X', 6)  # [x, y, phi, u, v, omega]
        U = ca.MX.sym('U', 2)  # [ax, delta] 
        _param = SymCarParameters("P")

        u = X[3]
        
        # ----------------------
        # Continous-time dynamics: X_dot = dX/dt
        # ----------------------
        dyn_X_dot = DynamicBicycleSystemDynamics().get_system_dynamics()
        kin_X_dot = KinematicBicycleSystemDynamics().get_system_dynamics()

        u_blend_min = 3 # m/s
        u_blend_max = 7 # m/s
        blend = ca.fmin(ca.fmax((u-u_blend_min)/(u_blend_max-u_blend_min),0),1)
        
        blend_X_dot  = ca.Function('StateDerivative', [X, U, _param.get_sym()], [
            (1-blend)*kin_X_dot(X, U, _param.get_sym()) + blend*dyn_X_dot(X, U, _param.get_sym())
        ])

        self._system_dynamics = blend_X_dot
        
    def get_model_name(self) -> str:
        return "BlendedBicycle"
    
    def get_system_dynamics(self) -> ca.Function:
        """
        Get system dynamics: X_dot(X, U), where:
        X = [x, y, phi, u, v, omega]
        U = [ax, delta] 
        """
        return self._system_dynamics
