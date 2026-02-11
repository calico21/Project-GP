import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination

# Import our custom physics engine and solver
from fsae_core.optimal_control.solver import OptimalLapSolver
from fsae_core.dynamics.vehicle_14dof import Vehicle14DOF

class SetupOptimizationProblem(ElementwiseProblem):
    """
    Multi-Objective Optimization Problem for FSAE Vehicle Setup.
    
    Objectives:
    1. Minimize Lap Time (f1)
    2. Maximize Drivability / Minimize Driver Workload (f2)
    
    Design Variables (Genes):
    1. Front Spring Stiffness
    2. Rear Spring Stiffness
    3. Front Wing Angle (Cl balance)
    4. Rear Wing Angle (Drag/Downforce trade)
    5. ARB Stiffness (Roll distribution)
    """

    def __init__(self, track_data, base_params):
        # Define ranges for the 5 design variables
        # [k_f, k_r, wing_f, wing_r, arb_stiff]
        self.xl = np.array([15000, 15000, 0.5, 1.0, 500])   # Lower bounds
        self.xu = np.array([45000, 45000, 2.5, 4.0, 5000])  # Upper bounds
        
        self.track_data = track_data
        self.base_params = base_params
        
        super().__init__(n_var=5, n_obj=2, n_constr=0, xl=self.xl, xu=self.xu)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        The 'Fitness Function'. 
        Simulates one specific car setup on the track.
        """
        # 1. Update Vehicle Parameters with new Genes
        current_params = self.base_params.copy()
        current_params['k_spring'] = [x[0], x[0], x[1], x[1]] # FL, FR, RL, RR
        current_params['Cl'] = x[2] + x[3] # Simplified aero map
        current_params['Cd'] = 0.5 + 0.1 * (x[2] + x[3]) # Drag penalty
        current_params['roll_stiffness_dist'] = x[4] 

        # 2. Run the "Ghost Car" Simulation
        solver = OptimalLapSolver(self.track_data, current_params)
        
        # We use a lower resolution (N=50) for the optimization loop to save time
        states, controls = solver.solve(N_segments=50)
        
        # 3. Calculate Objective 1: Lap Time
        # Sum of dt. Velocity is state index 14.
        velocity = states[14, :]
        ds = self.track_data['total_length'] / 50
        lap_time = np.sum(ds / (velocity + 1e-3))
        
        # 4. Calculate Objective 2: Drivability Metric (Minimize Instability)
        # We define "Instability" as the integral of Yaw Acceleration squared.
        # A twitchy car requires rapid steering corrections.
        yaw_rate = states[19, :] # r
        # Approximate yaw accel (dr/dt) via finite difference
        yaw_accel = np.diff(yaw_rate) / (ds / (velocity[:-1] + 1))
        instability_score = np.sum(yaw_accel**2)
        
        # Another drivability factor: "Tire Utilization Variance"
        # If the car relies on 100% grip at one axle and 50% at the other, it's "edgy".
        # (Simplified implementation here)
        
        # Store results
        # We minimize both, so Drivability is actually "Instability Score"
        out["F"] = [lap_time, instability_score]
        
        print(f"Gene: {x[:2]} -> Time: {lap_time:.3f}s, Stability: {instability_score:.1f}")

def run_genetic_algorithm(track_data, default_car_params):
    """
    Executes the NSGA-II evolution.
    """
    # Initialize the problem
    problem = SetupOptimizationProblem(track_data, default_car_params)

    # Configure the Algorithm
    algorithm = NSGA2(
        pop_size=40,            # Population size (cars per generation)
        n_offsprings=10,        # New cars per gen
        sampling=np.random.random,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # Termination Criteria
    termination = get_termination("n_gen", 20) # Run for 20 generations

    print("--- Starting Evolutionary Setup Optimization ---")
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    print(f"Optimization finished. Found {len(res.X)} optimal setups.")
    
    # The result 'res.F' contains the Pareto Front [LapTime, Instability]
    # The result 'res.X' contains the Setup Parameters
    return res

if __name__ == "__main__":
    # Dummy data for testing direct execution
    dummy_track = {'total_length': 1000, 'x_center': np.zeros(50), 'y_center': np.zeros(50), 'width': np.ones(50)*5}
    dummy_params = {'mass': 250, 'k_spring': [20000]*4, 'Cl': 2.0, 'Cd': 1.0}
    
    run_genetic_algorithm(dummy_track, dummy_params)