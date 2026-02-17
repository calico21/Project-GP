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

class SetupOptimizer:
    """
    NSGA-II Evolutionary Algorithm for Formula Student Setup Optimization.
    Optimizes: Springs, ARBs, Dampers, Ride Height
    Objectives: Minimize Lateral G Score (Maximize Grip), Maximize Stability (Minimize Overshoot)
    """
    def __init__(self, pop_size=50, generations=20):
        self.pop_size = pop_size
        self.generations = generations
        
        # Expanded Search Space
        self.bounds = {
            'k_f': (15000, 60000),   # Front Spring [N/m]
            'k_r': (15000, 60000),   # Rear Spring [N/m]
            'arb_f': (0, 2000),      # Front ARB [Nm/deg]
            'arb_r': (0, 1500),      # Rear ARB [Nm/deg]
            'c_f': (1000, 5000),     # Front Damping [N/(m/s)]
            'c_r': (1000, 5000),     # Rear Damping [N/(m/s)]
            'h_cg': (0.25, 0.35)     # Ride Height proxy [m]
        }
        self.param_names = list(self.bounds.keys())

    def run(self):
        print(f"[Optimizer] Initializing Population of {self.pop_size} setups...")
        
        # 1. Initialize
        population = self._init_population()
        objectives = self._evaluate_population(population)
        
        print(f"[Optimizer] Generation 0 Complete.")
        
        for gen in range(self.generations):
            # 2. Tournament Selection
            parents = self._selection(population, objectives)
            
            # 3. Crossover & Mutation
            offspring = self._generate_offspring(parents)
            
            # 4. Evaluate Offspring
            off_objectives = self._evaluate_population(offspring)
            
            # 5. Merge & Sort (Elitism)
            combined_pop = population + offspring
            combined_obj = np.vstack((objectives, off_objectives))
            
            # Sorting: Weighted sum
            # 70% Weight on Grip (Obj 0), 30% on Stability (Obj 1)
            # Filter out NaNs before sorting just in case
            combined_obj = np.nan_to_num(combined_obj, nan=10.0) # Penalty for NaNs

            scores = 0.7 * combined_obj[:, 0] + 0.3 * combined_obj[:, 1]
            indices = np.argsort(scores)
            
            best_indices = indices[:self.pop_size]
            
            population = [combined_pop[i] for i in best_indices]
            objectives = combined_obj[best_indices]
            
            # Display stats (handle potential bad values for display)
            best_g = -objectives[0, 0]
            if best_g < -9.0: best_g = 0.0 # Clean up display for crashed cars
            
            print(f"[Optimizer] Gen {gen+1}/{self.generations} | Best LatG: {best_g:.3f}g | Overshoot: {objectives[0, 1]*100:.1f}%")

        return population, objectives

    def _init_population(self):
        pop = []
        for _ in range(self.pop_size):
            ind = {}
            for key, (low, high) in self.bounds.items():
                ind[key] = np.random.uniform(low, high)
            pop.append(ind)
        return pop

    def _evaluate_population(self, population):
        results = []
        for i, ind in enumerate(population):
            try:
                f_score, stab_score = self._run_physics_simulation(ind)
                results.append([f_score, stab_score])
            except Exception as e:
                # Fallback for any unexpected errors
                results.append([0.0, 5.0]) # 0G, 500% Overshoot
                
        return np.array(results)

    def _run_physics_simulation(self, ind):
        """
        Runs a Step Steer maneuver. Includes CRASH DETECTION.
        """
        # 1. Setup Parameters
        params = [
            ind['k_f'], ind['k_r'], 
            ind['arb_f'], ind['arb_r'],
            ind['c_f'], ind['c_r']
        ]
        
        # 2. Initialize Vehicle
        vehicle = MultiBodyVehicle(VP_DICT, TP_DICT)
        
        # 3. Simulation Config
        dt = 0.005 # Reduced timestep for stability (200Hz)
        T_max = 2.5
        steps = int(T_max / dt)
        
        # Initial State
        x_curr = np.zeros(10)
        x_curr[3] = 20.0 # 20 m/s
        
        yaw_rates = []
        lat_accels = []
        
        crashed = False

        # 4. Run Integration Loop
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
            # If values explode (NaN or Infinity), stop immediately
            if not np.all(np.isfinite(x_next)):
                crashed = True
                break
                
            vx, r = x_next[3], x_next[5]
            
            # If car spins out (Yaw rate > 5 rad/s approx 286 deg/s) or stops
            if abs(r) > 5.0 or abs(vx) > 100.0:
                crashed = True
                break
                
            # Calculate Lat G
            ay = vx * r / 9.81
            
            yaw_rates.append(r)
            lat_accels.append(ay)
            x_curr = x_next
            
        # 5. Calculate Objectives
        if crashed or len(lat_accels) < 10:
            # PENALTY for unstable cars
            return 0.0, 5.0 # 0 Gs, 500% Overshoot
            
        # Obj 1: Maximize Grip (Minimize Negative G)
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

    def _selection(self, population, objectives):
        parents = []
        for _ in range(len(population)):
            i, j = np.random.choice(len(population), 2, replace=False)
            
            # Handle NaNs in objectives during selection
            score_i = 0.7*np.nan_to_num(objectives[i, 0]) + 0.3*np.nan_to_num(objectives[i, 1])
            score_j = 0.7*np.nan_to_num(objectives[j, 0]) + 0.3*np.nan_to_num(objectives[j, 1])
            
            if score_i < score_j:
                parents.append(population[i])
            else:
                parents.append(population[j])
        return parents

    def _generate_offspring(self, parents):
        offspring = []
        for i in range(0, len(parents), 2):
            if i+1 >= len(parents): break
            p1 = parents[i]
            p2 = parents[i+1]
            
            child = {}
            for key in self.bounds:
                alpha = np.random.random()
                val = p1[key]*alpha + p2[key]*(1-alpha)
                
                if np.random.random() < 0.3: 
                    mutation_scale = (self.bounds[key][1] - self.bounds[key][0]) * 0.10
                    val += np.random.normal(0, mutation_scale)
                
                val = np.clip(val, self.bounds[key][0], self.bounds[key][1])
                child[key] = val
            offspring.append(child)
        return offspring

if __name__ == "__main__":
    opt = SetupOptimizer(pop_size=50, generations=20)
    final_pop, final_obj = opt.run()
    
    df = pd.DataFrame(final_pop)
    df['Lat_G_Score'] = final_obj[:, 0]
    df['Stability_Overshoot'] = final_obj[:, 1]
    
    out_file = os.path.join(project_root, 'optimization_results.csv')
    df.to_csv(out_file, index=False)
    print(f"[Success] Results saved to {out_file}")