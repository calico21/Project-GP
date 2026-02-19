import os
import sys
import numpy as np
import pandas as pd
import argparse
import time

# --- ML & OPTIMIZATION IMPORTS ---
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from scipy.stats import qmc # Quasi-Monte Carlo for Latin Hypercube
except ImportError:
    print("[Main] Warning: sklearn/scipy not found. Surrogate optimization will fail.")

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- PROJECT IMPORTS ---
try:
    from telemetry.log_ingestion import LogIngestion
    from telemetry.track_generator import TrackGenerator
    from optimization.ocp_solver import OptimalLapSolver
    # We will invoke the logic directly here to ensure Surrogate integration
    from models.vehicle_dynamics import DynamicBicycleModel
    from data.configs.vehicle_params import vehicle_params as VP_DEF
    from data.configs.tire_coeffs import tire_coeffs as TP_DEF
except ImportError as e:
    print(f"[Critical Error] Module import failed: {e}")
    sys.exit(1)


# =============================================================================
#  HELPER: SIMULATION BENCHMARK
# =============================================================================
def evaluate_vehicle_setup(setup_vector):
    """
    The 'Expensive' Ground Truth Function.
    Maps normalized [0-1] inputs to physical vehicle parameters and runs a test.
    
    Design Variables (Simplified for Demo):
    0: Front Spring Stiffness (k_f)
    1: Rear Spring Stiffness (k_r)
    2: Wing Angle (Cl_boost)
    """
    # 1. Denormalize Parameters
    # k_f range: [20000, 80000]
    k_f = 20000 + setup_vector[0] * 60000
    # k_r range: [20000, 80000]
    k_r = 20000 + setup_vector[1] * 60000
    # Aero range: [0.5, 3.0] (Cl multiplier)
    cl_mult = 0.5 + setup_vector[2] * 2.5
    
    # 2. Update Vehicle Config
    vp = VP_DEF.copy()
    vp['k_roll_f'] = k_f # Approximation for roll stiffness
    vp['k_roll_r'] = k_r
    vp['Cl'] = vp.get('Cl', 1.0) * cl_mult
    
    # 3. Run Physics Test (e.g., Constant Radius Turn)
    # We use the DynamicBicycleModel to find steady-state lateral G
    model = DynamicBicycleModel(vp, TP_DEF)
    
    # Objective 1: Max Lateral G (Grip)
    # Objective 2: Stability (Understeer Gradient)
    
    # Simulating a Ramp Steer to find peak G
    v_test = 20.0 # 20 m/s
    peak_g = 0.0
    stability_metric = 0.0
    
    # Pseudo-simulation loop (Fast approximation for the demo)
    # In a real scenario, this would integrate the ODEs
    try:
        # Physics proxy:
        # Fz = m*g + Aero
        Fz = vp['m']*9.81 + 0.5*1.225*vp['Cl']*vp['A']*(v_test**2)
        mu_peak = 1.3 # Average tire mu
        
        # Simple Grip Calc
        max_lat_force = Fz * mu_peak
        peak_g = max_lat_force / (vp['m'] * 9.81)
        
        # Stability: Balance between front/rear stiffness
        # Ideal: Front slightly stiffer than rear (Understeer)
        balance = k_f / (k_f + k_r)
        target_balance = 0.55
        stability_metric = abs(balance - target_balance) # Minimize this (0 = perfect balance)
        
    except Exception as e:
        print(f"Sim Failed: {e}")
        return [0.0, 1.0] # Bad score

    return [peak_g, stability_metric]


# =============================================================================
#  CLASS: SURROGATE OPTIMIZER
# =============================================================================
class SurrogateOptimizer:
    def __init__(self, n_samples=50):
        self.n_samples = n_samples
        self.model = None
        
        # Kernel: Matern is best for physics landscapes
        # Nu=2.5 handles smooth functions with some irregularities
        self.kernel = 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        
    def generate_doe(self):
        """Generates Latin Hypercube Samples."""
        print(f"[Surrogate] Generating {self.n_samples} design points (LHS)...")
        sampler = qmc.LatinHypercube(d=3)
        sample = sampler.random(n=self.n_samples)
        return sample # Returns [0, 1] normalized vectors
    
    def train(self):
        """Runs the expensive simulations and fits the Kriging model."""
        X_train = self.generate_doe()
        y_train = []
        
        print(f"[Surrogate] Running {self.n_samples} physics simulations (Ground Truth)...")
        start_t = time.time()
        
        for i, setup in enumerate(X_train):
            # The 'Expensive' Call
            res = evaluate_vehicle_setup(setup)
            y_train.append(res)
            
            if i % 10 == 0:
                print(f"   ... Sim {i}/{self.n_samples} Complete")
                
        duration = time.time() - start_t
        print(f"[Surrogate] Training Data Collection took {duration:.2f}s")
        
        # Fit Gaussian Process
        # We predict 2 targets: [Grip, Stability]
        print("[Surrogate] Fitting Gaussian Process (Kriging)...")
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)
        self.model.fit(X_train, np.array(y_train))
        print(f"[Surrogate] Model Trained. R2 Score: {self.model.score(X_train, y_train):.4f}")
        
    def optimize(self):
        """Runs NSGA-II on the Surrogate Model."""
        if self.model is None:
            self.train()
            
        print("\n[Surrogate] Running Genetic Algorithm on Predicted Surface...")
        
        # Simple Random Search on the Surrogate (NSGA-II proxy)
        # Since prediction is cheap (ms), we can evaluate 100,000 points instantly
        
        # 1. Generate massive random population
        n_pop = 50000
        X_pop = np.random.rand(n_pop, 3)
        
        # 2. Predict all instantly
        y_pred = self.model.predict(X_pop)
        
        # 3. Filter (Pareto-ish)
        # We want MAX Grip (idx 0) and MIN Instability (idx 1)
        
        # Create a DataFrame to sort
        df = pd.DataFrame(X_pop, columns=['k_f_norm', 'k_r_norm', 'aero_norm'])
        df['pred_grip'] = y_pred[:, 0]
        df['pred_stability'] = y_pred[:, 1]
        
        # Sort by Grip
        best_grip = df.sort_values('pred_grip', ascending=False).head(5)
        
        # Sort by Stability
        best_stable = df.sort_values('pred_stability', ascending=True).head(5)
        
        return best_grip, best_stable


# =============================================================================
#  MAIN EXECUTION
# =============================================================================

def generate_synthetic_track():
    """FS Skidpad Generator."""
    print("[Main] No log file found. Generating SYNTHETIC TRACK (FS Skidpad)...")
    n_points = 500
    radius = 15.25 # FS Skidpad is small
    theta = np.linspace(0, 2*np.pi, n_points)
    
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    s = radius * theta
    k = np.full_like(s, 1.0/radius)
    
    return {
        's': s, 'x': x, 'y': y, 'k': k,
        'w_left': np.full_like(s, 3.0),
        'w_right': np.full_like(s, 3.0),
        'total_length': 2 * np.pi * radius
    }

def run_ocp_pipeline(track_data):
    """Executes the Optimal Control Problem."""
    print("\n" + "="*50)
    print("PHASE 1: GHOST CAR GENERATION (OCP)")
    print("="*50)
    
    solver = OptimalLapSolver()
    
    # Subsample track for OCP speed
    N_nodes = 150
    idx = np.linspace(0, len(track_data['s'])-1, N_nodes+1).astype(int)
    
    try:
        result = solver.solve(
            track_data['s'][idx], 
            track_data['k'][idx], 
            track_data['w_left'][idx], 
            track_data['w_right'][idx], 
            N=N_nodes
        )
        
        if "error" in result:
            print(f"[OCP] Solver returned error: {result['error']}")
        else:
            print(f"[OCP] Optimal Lap Time: {result['time']:.3f} s")
            df = pd.DataFrame(result)
            df.to_csv(os.path.join(current_dir, 'ghost_car_telemetry.csv'), index=False)
            
    except Exception as e:
        print(f"[OCP] Crash: {e}")

def run_optimization_pipeline():
    """Executes the Surrogate-Assisted Setup Optimization."""
    print("\n" + "="*50)
    print("PHASE 2: SETUP OPTIMIZATION (Kriging Surrogate)")
    print("="*50)
    
    opt = SurrogateOptimizer(n_samples=50) # 50 simulations to learn the physics
    best_grip, best_stable = opt.optimize()
    
    print("\n[Results] Best Grip Configurations:")
    print(best_grip[['pred_grip', 'pred_stability', 'k_f_norm', 'aero_norm']].to_string(index=False))
    
    print("\n[Results] Best Stability Configurations:")
    print(best_stable[['pred_grip', 'pred_stability', 'k_f_norm', 'aero_norm']].to_string(index=False))
    
    # Save
    best_grip.to_csv(os.path.join(current_dir, 'opt_results_grip.csv'), index=False)
    print(f"\n[Main] Results saved to {current_dir}")


def main():
    parser = argparse.ArgumentParser(description="Project-GP Digital Twin Engine")
    parser.add_argument('--mode', type=str, default='all', choices=['ocp', 'opt', 'all'])
    parser.add_argument('--log', type=str, default=None)
    args = parser.parse_args()

    # 1. Track Prep
    track_data = None
    if args.mode in ['ocp', 'all']:
        if args.log:
            ingestor = LogIngestion(args.log)
            df = ingestor.process()
            gen = TrackGenerator(df)
            track_data = gen.generate()
        else:
            track_data = generate_synthetic_track()
            
        run_ocp_pipeline(track_data)

    # 2. Optimization
    if args.mode in ['opt', 'all']:
        run_optimization_pipeline()
        
    print("\n[Main] Full Execution Complete.")

if __name__ == "__main__":
    main()