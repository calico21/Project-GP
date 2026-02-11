import casadi as ca
import numpy as np
from fsae_core.dynamics.vehicle_14dof import Vehicle14DOF

class OptimalLapSolver:
    """
    Solves the Minimum Lap Time Problem (MLTP) using Direct Collocation.
    """

    def __init__(self, track_data, vehicle_params):
        self.track = track_data
        self.params = vehicle_params
        self.model = Vehicle14DOF(vehicle_params)

    def solve(self, N_segments=200):
        """
        Solves for the optimal control inputs over a fixed distance.
        
        Args:
            N_segments: Number of discretization points (mesh nodes).
                        More segments = higher accuracy but slower solve time.
        """
        opti = ca.Opti() # The CasADi Optimization Interface

        # --- 1. Discretization Setup ---
        # We assume a fixed track length L, divided into N segments of length ds
        total_length = self.track['total_length']
        ds = total_length / N_segments
        
        # --- 2. Decision Variables ---
        # State Vector: 28 States x (N+1) Nodes
        # (Positions, Orientations, Velocities, Rates, etc.)
        n_states = 28
        X = opti.variable(n_states, N_segments + 1)
        
        # Control Vector: 3 Controls x N Nodes
        # [Steering Angle, Throttle (0-1), Brake (0-1)]
        n_controls = 3
        U = opti.variable(n_controls, N_segments)
        
        # --- 3. Objective Function: Minimize Time ---
        # Time = Integral of (1 / velocity) ds
        # We use the longitudinal velocity 'u' (State index 14 in our model)
        u_vel = X[14, :-1] 
        
        # Regularization: Add small term to minimize control jerk (smoother inputs)
        J_time = ca.sum1(ds / (u_vel + 1e-3)) 
        J_smoothness = ca.sum1(ca.sum2((U[:, 1:] - U[:, :-1])**2)) # Penalize rapid control changes
        
        opti.minimize(J_time + 0.1 * J_smoothness)

        # --- 4. Constraints ---
        
        for k in range(N_segments):
            # A. Dynamic Constraints (Runge-Kutta 4 Integration)
            # x_next = x_curr + ds * (dx/ds)
            # dx/ds = dynamics(x, u) / velocity
            
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k+1]
            
            # Get time-derivative from physics engine
            # Note: We must ensure velocity is not zero to avoid division by zero
            velocity_k = ca.fmax(x_k[14], 1.0) # Lower bound 1 m/s
            
            # RK4 Integration Step (Spatial)
            k1 = self.model.get_dynamics(x_k, u_k)
            k2 = self.model.get_dynamics(x_k + (ds/2) * (k1/velocity_k), u_k)
            k3 = self.model.get_dynamics(x_k + (ds/2) * (k2/velocity_k), u_k)
            k4 = self.model.get_dynamics(x_k + ds * (k3/velocity_k), u_k)
            
            x_next_rk4 = x_k + (ds / 6) * (k1 + 2*k2 + 2*k3 + k4) / velocity_k
            
            opti.subject_to(x_next == x_next_rk4)

            # B. Track Boundary Constraints
            # Car position (x, y) must be within track width
            # In a full spatial solver, we normally convert X,Y to Frenet coordinates (s, n)
            # Here, we assume the inputs are global X,Y and we constrain deviation from center line.
            
            # Simple approximation for this file:
            # Distance from center line < track_width / 2
            # (Requires spline lookup in real implementation)
            center_x = self.track['x_center'][k]
            center_y = self.track['y_center'][k]
            width    = self.track['width'][k]
            
            car_x = x_k[0]
            car_y = x_k[1]
            
            distance_sq = (car_x - center_x)**2 + (car_y - center_y)**2
            opti.subject_to(distance_sq <= (width/2)**2)
            
            # C. Physical Limits
            # Friction Circle is handled inside 'get_dynamics', but we can add safety bounds
            opti.subject_to(opti.bounded(-0.5, u_k[0], 0.5)) # Steering limit (rad)
            opti.subject_to(opti.bounded(0, u_k[1], 1))      # Throttle 0-1
            opti.subject_to(opti.bounded(0, u_k[2], 1))      # Brake 0-1
            
            # No simultaneous Throttle + Brake (optional logic)
            # opti.subject_to(u_k[1] * u_k[2] == 0) # Hard for solvers, use with care

        # --- 5. Boundary Conditions ---
        # Start at initial state (e.g., start line velocity)
        opti.subject_to(X[:, 0] == self.params['initial_state'])
        
        # Cyclic Constraint (for a lap): End state = Start state
        # opti.subject_to(X[:, -1] == X[:, 0])

        # --- 6. Initial Guess (Warm Start) ---
        # Solvers fail if you start with all Zeros.
        # We initialize with a constant velocity "cruise"
        opti.set_initial(X[14, :], 20.0) # Guess 20 m/s everywhere
        opti.set_initial(U[1, :], 0.5)   # Guess 50% throttle

        # --- 7. Solve ---
        # IPOPT is the standard interior-point solver for this
        p_opts = {"expand": True}
        s_opts = {"max_iter": 500, "tol": 1e-4}
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
            return sol.value(X), sol.value(U)
        except Exception as e:
            print("Solver failed to converge. Returning debug values.")
            return opti.debug.value(X), opti.debug.value(U)