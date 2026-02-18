import sys
import os
import numpy as np
import pandas as pd
import time

# --- IMPORT PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS ---
try:
    from models.vehicle_dynamics import MultiBodyVehicle
    from data.configs.vehicle_params import vehicle_params as VP_DICT
    from data.configs.tire_coeffs import tire_coeffs as TP_DICT
except ImportError as e:
    print(f"[Error] Import Failed: {e}")
    sys.exit(1)

# --- PYMOO IMPORTS (The "Real" NSGA-II) ---
try:
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
except ImportError:
    print("\n[CRITICAL ERROR] 'pymoo' library is missing.")
    print("Please install it to use the new optimization engine:")
    print(">> pip install pymoo\n")
    sys.exit(1)


class VehicleSetupProblem(ElementwiseProblem):
    """
    Defines the Multi-Objective Optimization Problem for pymoo.
    
    Variables (7):
        k_f, k_r       : Spring Rates [N/m]
        arb_f, arb_r   : Anti-Roll Bar Stiffness [Nm/deg]
        c_f, c_r       : Damper Damping [N/(m/s)]
        h_cg           : Ride Height / CG Height [m]
        
    Objectives (2):
        1. Minimize( Lat_G_Score )      -> Effectively Maximize Lateral G
        2. Minimize( Stability_Score )  -> Minimize Yaw Overshoot
    """
    def __init__(self):
        # Define bounds for the 7 variables
        # Order: k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg
        self.var_keys = ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', 'h_cg']
        
        xl = np.array([15000., 15000., 0.,   0.,   1000., 1000., 0.25])
        xu = np.array([60000., 60000., 2000.,1500., 5000., 5000., 0.35])
        
        super().__init__(n_var=7, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Called by the optimizer for every individual in the population.
        x: The array of parameters for this individual.
        out: Dictionary where we store the results ("F").
        """
        # 1. Map array 'x' back to dictionary for the simulator
        ind = {k: v for k, v in zip(self.var_keys, x)}
        
        # 2. Run the Physics Simulation
        f_score, overshoot = self._run_physics_simulation(ind)
        
        # 3. Store objectives
        # pymoo minimizes by default.
        # f_score is negative G (e.g. -1.5). Minimizing -1.5 pushes it towards -2.0 (Better).
        # overshoot is positive % (e.g. 0.05). Minimizing pushes it to 0.0 (Better).
        out["F"] = [f_score, overshoot]

    def _run_physics_simulation(self, ind):
        """
        Runs a Step Steer maneuver. Includes CRASH DETECTION.
        """
        # Setup Parameters
        params = [
            ind['k_f'], ind['k_r'], 
            ind['arb_f'], ind['arb_r'],
            ind['c_f'], ind['c_r']
        ]
        
        # Initialize Vehicle
        vehicle = MultiBodyVehicle(VP_DICT, TP_DICT)
        
        # Simulation Config
        dt = 0.005 # 200Hz
        T_max = 2.5
        steps = int(T_max / dt)
        
        # Initial State: 20 m/s entry
        x_curr = np.zeros(10)
        x_curr[3] = 20.0 
        
        yaw_rates = []
        lat_accels = []
        crashed = False

        # Run Integration Loop
        for t in np.linspace(0, T_max, steps):
            # Input: Step Steer at 0.2s
            steer = 0.0
            if t > 0.2:
                steer = 0.1 
            u_curr = [steer, 0]
            
            # Integrate one step
            try:
                x_next = vehicle.simulate_step(x_curr, u_curr, params, dt=dt)
            except RuntimeError:
                crashed = True
                break

            # --- CRASH DETECTION ---
            if not np.all(np.isfinite(x_next)):
                crashed = True
                break
                
            vx, r = x_next[3], x_next[5]
            
            # Spinout check (Yaw rate > 5 rad/s) or Stop check
            if abs(r) > 5.0 or abs(vx) > 100.0:
                crashed = True
                break
                
            # Calculate Lat G
            ay = vx * r / 9.81
            yaw_rates.append(r)
            lat_accels.append(ay)
            x_curr = x_next
            
        # --- CALCULATE OBJECTIVES ---
        if crashed or len(lat_accels) < 10:
            # Penalties:
            # Grip: 0.0 G (Worst, since normal is ~ -1.5)
            # Stability: 5.0 (500% Overshoot, Terrible)
            return 0.0, 5.0 
            
        # Obj 1: Maximize Grip (Minimize Negative G)
        # We take the steady state G (last 20% of data)
        steady_state_ay = np.mean(lat_accels[-int(steps*0.2):])
        f_score = -abs(steady_state_ay) 

        # Obj 2: Stability (Overshoot)
        peak_yaw = np.max(np.abs(yaw_rates))
        steady_yaw = np.mean(np.abs(yaw_rates[-int(steps*0.2):]))
        
        if steady_yaw < 0.01: 
            overshoot = 0.0 
        else:
            overshoot = (peak_yaw - steady_yaw) / steady_yaw
            
        return f_score, overshoot


class SetupOptimizer:
    """
    Wrapper for the NSGA-II Algorithm.
    """
    def __init__(self, pop_size=50, generations=20):
        self.pop_size = pop_size
        self.generations = generations

    def run(self):
        print(f"[Optimizer] Initializing NSGA-II (pymoo) with Pop={self.pop_size}, Gen={self.generations}...")
        
        # 1. Instantiate the Physics Problem
        problem = VehicleSetupProblem()
        
        # 2. Configure the Algorithm
        algorithm = NSGA2(
            pop_size=self.pop_size,
            n_offsprings=20,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # 3. Define Termination
        termination = get_termination("n_gen", self.generations)
        
        # 4. RUN OPTIMIZATION
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            save_history=False,
            verbose=True  # Prints progress to console
        )
        
        print(f"[Optimizer] Optimization Complete. Found {len(res.X)} solutions on the Pareto Front.")
        
        # 5. Format Results for Dashboard compatibility
        # res.X = Parameters [N_sol, 7]
        # res.F = Objectives [N_sol, 2]
        
        final_pop = []
        for row in res.X:
            ind = {k: v for k, v in zip(problem.var_keys, row)}
            final_pop.append(ind)
            
        final_obj = res.F
        
        return final_pop, final_obj

if __name__ == "__main__":
    opt = SetupOptimizer(pop_size=40, generations=10)
    final_pop, final_obj = opt.run()
    
    # Save for inspection
    df = pd.DataFrame(final_pop)
    df['Lat_G_Score'] = final_obj[:, 0]
    df['Stability_Overshoot'] = final_obj[:, 1]
    
    out_file = os.path.join(project_root, 'optimization_results.csv')
    df.to_csv(out_file, index=False)
    print(f"[Success] Results saved to {out_file}")