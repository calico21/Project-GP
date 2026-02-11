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
        """
        opti = ca.Opti() # The CasADi Optimization Interface

        # --- 1. Discretization Setup ---
        total_length = self.track['total_length']
        ds = total_length / N_segments
        
        # --- 2. Decision Variables ---
        # State Vector: 28 States x (N+1) Nodes
        n_states = 28
        X = opti.variable(n_states, N_segments + 1)
        
        # Control Vector: 3 Controls x N Nodes
        # [Steering Angle, Throttle (0-1), Brake (0-1)]
        n_controls = 3
        U = opti.variable(n_controls, N_segments)
        
        # --- 3. Objective Function: Minimize Time ---
        # Time = Integral of (1 / velocity) ds
        # u_vel is State index 14
        u_vel = X[14, :-1] 
        
        # FIXED: Use sum2 (horizontal sum) for the time integration
        J_time = ca.sum2(ds / (u_vel + 1e-3)) 
        
        # Smoothness regularization
        J_smoothness = ca.sum1(ca.sum2((U[:, 1:] - U[:, :-1])**2)) 
        
        opti.minimize(J_time + 0.1 * J_smoothness)

        # --- 4. Constraints ---
        for k in range(N_segments):
            # A. Dynamic Constraints (RK4 Integration)
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k+1]
            
            # Lower bound velocity to prevent div/0
            velocity_k = ca.fmax(x_k[14], 1.0) 
            
            k1 = self.model.get_dynamics(x_k, u_k)
            k2 = self.model.get_dynamics(x_k + (ds/2) * (k1/velocity_k), u_k)
            k3 = self.model.get_dynamics(x_k + (ds/2) * (k2/velocity_k), u_k)
            k4 = self.model.get_dynamics(x_k + ds * (k3/velocity_k), u_k)
            
            x_next_rk4 = x_k + (ds / 6) * (k1 + 2*k2 + 2*k3 + k4) / velocity_k
            
            opti.subject_to(x_next == x_next_rk4)

            # B. Track Boundary Constraints (Simplified)
            # Distance from center line < track_width / 2
            center_x = self.track['x_center'][k]
            center_y = self.track['y_center'][k]
            width    = self.track['width'][k]
            
            car_x = x_k[0]
            car_y = x_k[1]
            
            distance_sq = (car_x - center_x)**2 + (car_y - center_y)**2
            opti.subject_to(distance_sq <= (width/2)**2)
            
            # C. Physical Limits
            opti.subject_to(opti.bounded(-0.5, u_k[0], 0.5)) # Steering limit
            opti.subject_to(opti.bounded(0, u_k[1], 1))      # Throttle
            opti.subject_to(opti.bounded(0, u_k[2], 1))      # Brake

        # --- 5. Boundary Conditions (FIXED) ---
        # Instead of looking for a missing config key, we calculate the start state here.
        
        # Constants
        R_tire = self.params.get('tire_radius', 0.23)
        h_cg   = self.params.get('cg_height', 0.28)
        
        # Build the initial state vector (28 elements)
        # We use a standard Python list, CasADi accepts this in subject_to
        X0 = [0.0] * 28
        
        # Position: Start at the first point of the track
        X0[0] = self.track['x_center'][0]
        X0[1] = self.track['y_center'][0]
        X0[2] = h_cg  # Z position roughly at CG height
        
        # Velocity: Rolling start at 20 m/s (72 kph)
        # This helps the solver converge faster than a standing start
        v_start = 20.0
        X0[14] = v_start
        
        # Wheel Speeds: Match vehicle speed (No slip)
        # omega = v / r
        w_start = v_start / R_tire
        X0[24] = w_start # FL
        X0[25] = w_start # FR
        X0[26] = w_start # RL
        X0[27] = w_start # RR
        
        # Suspension: Unsprung mass height = tire radius
        X0[6] = R_tire
        X0[7] = R_tire
        X0[8] = R_tire
        X0[9] = R_tire

        # Apply the Start Constraint
        opti.subject_to(X[:, 0] == X0)

        # ... (Previous code remains the same up to Step 6)

        # --- 6. Improved Initial Guess (Warm Start) ---
        # Guess a constant velocity of 20 m/s
        opti.set_initial(X[14, :], 20.0) 
        
        # Guess wheel speeds matching that velocity (No slip condition)
        R_tire = self.params.get('tire_radius', 0.23)
        omega_guess = 20.0 / R_tire
        opti.set_initial(X[24, :], omega_guess) # FL
        opti.set_initial(X[25, :], omega_guess) # FR
        opti.set_initial(X[26, :], omega_guess) # RL
        opti.set_initial(X[27, :], omega_guess) # RR
        
        # Guess correct ride height (prevent "falling" at start)
        opti.set_initial(X[6, :], R_tire)
        opti.set_initial(X[7, :], R_tire)
        opti.set_initial(X[8, :], R_tire)
        opti.set_initial(X[9, :], R_tire)

        # Guess 50% throttle to maintain speed
        opti.set_initial(U[1, :], 0.5)   

        # --- 7. Solve ---
        p_opts = {"expand": True, "print_time": True}
        s_opts = {
            "max_iter": 3000,        # <-- INCREASED from 500
            "tol": 1e-3,             # <-- RELAXED from 1e-4
            "acceptable_tol": 1e-2,  # Allow loose solution if stuck
            "acceptable_iter": 50,
            "print_level": 5
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
            return sol.value(X), sol.value(U)
        except Exception as e:
            print(f"⚠️ Solver Message: {e}")
            print("Returning debug values (trajectory might be incomplete).")
            return opti.debug.value(X), opti.debug.value(U)