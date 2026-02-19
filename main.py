import os
import sys
import numpy as np
import pandas as pd
import argparse

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- PROJECT IMPORTS ---
try:
    from telemetry.log_ingestion import LogIngestion
    from telemetry.track_generator import TrackGenerator
    from optimization.ocp_solver import OptimalLapSolver
    from optimization.evolutionary import SetupOptimizer  # <-- The New SOTA AI
except ImportError as e:
    print(f"[Critical Error] Module import failed: {e}")
    sys.exit(1)


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
    """Executes the Surrogate-Assisted Multi-Fidelity Setup Optimization."""
    print("\n" + "="*50)
    print("PHASE 2: SETUP OPTIMIZATION (Multi-Fidelity Co-Kriging & NSGA-II)")
    print("="*50)
    
    # Instantiate the new SOTA optimizer
    opt = SetupOptimizer(pop_size=100, generations=50)
    final_pop, final_obj = opt.run()
    
    # Convert Pareto Front results to DataFrame
    df = pd.DataFrame(final_pop)
    df['Lat_G_Score'] = final_obj[:, 0]
    df['Stability_Overshoot'] = final_obj[:, 1]
    
    # Identify the extreme ends of the Pareto Front for the user
    best_grip = df.sort_values('Lat_G_Score', ascending=True).head(5)
    best_stable = df.sort_values('Stability_Overshoot', ascending=True).head(5)
    
    print("\n[Results] Best Grip Configurations (Max Cornering):")
    print(best_grip[['Lat_G_Score', 'Stability_Overshoot', 'k_f', 'k_r', 'arb_f', 'arb_r', 'h_cg']].to_string(index=False))
    
    print("\n[Results] Best Stability Configurations (Zero Overshoot):")
    print(best_stable[['Lat_G_Score', 'Stability_Overshoot', 'k_f', 'k_r', 'arb_f', 'arb_r', 'h_cg']].to_string(index=False))
    
    # Save Full Pareto Front
    out_file = os.path.join(current_dir, 'optimization_results.csv')
    df.to_csv(out_file, index=False)
    print(f"\n[Main] Full Pareto Front saved to {out_file}")


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