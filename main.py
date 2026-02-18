import os
import sys
import numpy as np
import pandas as pd
import argparse

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- IMPORTS ---
try:
    from telemetry.log_ingestion import LogIngestion
    from telemetry.track_generator import TrackGenerator
    from optimization.ocp_solver import OptimalLapSolver
    from optimization.evolutionary import SetupOptimizer
except ImportError as e:
    print(f"[Critical Error] Module import failed: {e}")
    print("Ensure you have run the previous fixes (files 1-5) correctly.")
    sys.exit(1)

def generate_synthetic_track():
    """
    Creates a simple 200m radius circle track if no logs are found.
    Allows the optimizer to be tested immediately.
    """
    print("[Main] No log file found. Generating SYNTHETIC TRACK (Circle)...")
    n_points = 500
    radius = 200.0
    theta = np.linspace(0, 2*np.pi, n_points)
    
    # Create a circle path
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    # Calculate s, psi, k analytically for a perfect circle
    s = radius * theta
    k = np.full_like(s, 1.0/radius) # Constant curvature
    w_left = np.full_like(s, 5.0)
    w_right = np.full_like(s, 5.0)
    
    track_data = {
        's': s,
        'x': x,
        'y': y,
        'psi': theta + np.pi/2, # Tangent is perpendicular to radius
        'k': k,
        'w_left': w_left,
        'w_right': w_right,
        'total_length': 2 * np.pi * radius
    }
    return track_data

def run_ocp_pipeline(track_data):
    """
    Runs the Optimal Control Problem to find the theoretical best lap.
    """
    print("\n" + "="*50)
    print("PHASE 1: GHOST CAR GENERATION (OCP)")
    print("="*50)
    print(f"[OCP] Solving for track length: {track_data['total_length']:.1f}m")
    
    solver = OptimalLapSolver()
    
    # Solve
    # We downsample the track for OCP speed (N=200 nodes is usually good for a full lap)
    N_nodes = 200
    idx = np.linspace(0, len(track_data['s'])-1, N_nodes+1).astype(int)
    
    s_coarse = track_data['s'][idx]
    k_coarse = track_data['k'][idx]
    wl_coarse = track_data['w_left'][idx]
    wr_coarse = track_data['w_right'][idx]
    
    result = solver.solve(s_coarse, k_coarse, wl_coarse, wr_coarse, N=N_nodes)
    
    if "error" in result:
        print(f"[OCP] Failed: {result['error']}")
    else:
        print(f"[OCP] Success! Lap Time: {result['time']:.3f} s")
        print(f"[OCP] Peak Velocity: {np.max(result['v'])*3.6:.1f} km/h")
        print(f"[OCP] Peak Lateral G: {np.max(np.abs(result['v']**2 * k_coarse[:-1]))/9.81:.2f} G")
        
        # Save result
        df_res = pd.DataFrame(result)
        out_path = os.path.join(current_dir, 'ghost_car_telemetry.csv')
        df_res.to_csv(out_path, index=False)
        print(f"[OCP] Telemetry saved to {out_path}")

def run_optimization_pipeline():
    """
    Runs the NSGA-II Genetic Algorithm to find optimal setups.
    """
    print("\n" + "="*50)
    print("PHASE 2: SETUP OPTIMIZATION (NSGA-II)")
    print("="*50)
    
    # Settings
    pop_size = 40
    generations = 10
    
    optimizer = SetupOptimizer(pop_size=pop_size, generations=generations)
    final_pop, final_obj = optimizer.run()
    
    # Process Results
    df_opt = pd.DataFrame(final_pop)
    df_opt['Lat_G_Score'] = final_obj[:, 0]
    df_opt['Stability_Overshoot'] = final_obj[:, 1]
    
    # Convert negative G back to positive for readability
    df_opt['Max_Lat_G'] = -df_opt['Lat_G_Score']
    
    out_path = os.path.join(current_dir, 'optimization_results.csv')
    df_opt.to_csv(out_path, index=False)
    
    print("\n[Optimization] Summary:")
    print(f"Top Solution (Grip): {df_opt['Max_Lat_G'].max():.3f} G")
    print(f"Top Solution (Stability): {df_opt['Stability_Overshoot'].min()*100:.1f} % Overshoot")
    print(f"[Optimization] Full Pareto front saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Project-GP Main Execution Engine")
    parser.add_argument('--mode', type=str, default='all', choices=['ocp', 'opt', 'all'],
                        help="Select execution mode: 'ocp' (Ghost Car), 'opt' (Setup), or 'all'")
    parser.add_argument('--log', type=str, default=None,
                        help="Path to CSV telemetry log for track generation")
    args = parser.parse_args()

    # --- STEP 1: TRACK GENERATION ---
    if args.mode in ['ocp', 'all']:
        track_data = None
        
        # Try to load log
        log_path = args.log
        if not log_path:
            # Check default location
            default_log = os.path.join(current_dir, 'data', 'logs', 'sample_log.csv')
            if os.path.exists(default_log):
                log_path = default_log
        
        if log_path and os.path.exists(log_path):
            print(f"[Main] Loading log: {log_path}")
            ingestor = LogIngestion(log_path)
            df_log = ingestor.process()
            
            gen = TrackGenerator(df_log)
            track_data = gen.generate(s_step=1.0, smoothing_s=2.0)
            gen.save_track(track_data)
        else:
            # Fallback
            track_data = generate_synthetic_track()
            
        # --- STEP 2: RUN OCP ---
        run_ocp_pipeline(track_data)

    # --- STEP 3: RUN SETUP OPTIMIZATION ---
    if args.mode in ['opt', 'all']:
        run_optimization_pipeline()

    print("\n[Main] Execution Complete.")

if __name__ == "__main__":
    main()