import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- IMPORTS ---
from telemetry.log_ingestion import LogIngestion
from telemetry.track_generator import TrackGenerator
from telemetry.validation import ModelValidator
from telemetry.driver_coaching import DriverCoach
from optimization.ocp_solver import OptimalLapSolver
from optimization.evolutionary import SetupOptimizer # <--- NEW IMPORT
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

# --- DBC CONFIG ---
DBC_CONFIG = {
    0x5: {'name': 'STEER', 'signals': [
        {'name': 'ANGLE', 'start_bit': 0, 'length': 16, 'factor': 0.01, 'offset': 0, 'signed': True}
    ]},
    0x27: {'name': 'WheelInfo', 'signals': [
        {'name': 'speed', 'start_bit': 48, 'length': 8,  'factor': 1.0, 'offset': 0, 'signed': False}
    ]},
    0x119: {'name': 'GPS_Lat_Long', 'signals': [
        {'name': 'Latitude',  'start_bit': 0,  'length': 32, 'factor': 1e-07, 'offset': 0, 'signed': True},
        {'name': 'Longitude', 'start_bit': 32, 'length': 32, 'factor': 1e-07, 'offset': 0, 'signed': True}
    ]}
}

def main():
    print("=== FS Project-GP: Driver & Setup Optimization Engine ===")

    # 1. LOCATE LOG
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(base_dir, "data", "logs", "20250816_185811.asc") 

    # 2. INGESTION
    print(f"[1/6] Processing Log: {os.path.basename(log_file)}...")
    ingestor = LogIngestion(DBC_CONFIG)
    if os.path.exists(log_file):
        ingestor.parse_asc(log_file)
        channel_map = {'ANGLE': 'steer', 'speed': 'velocity', 'Latitude': 'lat', 'Longitude': 'lon'}
        ingestor.apply_schema(channel_map)
        ingestor.process_units()          
        ingestor.project_gps_to_cartesian()
        real_data = ingestor.export()
    else:
        print("[Warning] Log file not found. Using dummy data.")
        real_data = {}

    # 3. TRACK GENERATION
    print("[2/6] Generating Track Model...")
    track_gen = TrackGenerator(smoothing_factor=2.0)
    track_generated = False
    
    if 'x' in real_data and len(real_data['x']) > 100:
        try:
            track_gen.from_centerline(real_data['x'], real_data['y'], track_width=4.0)
            track_generated = True
        except ValueError as e:
            print(f"[Warning] GPS Track Gen failed: {e}")

    if not track_generated:
        print("⚠️  GPS Data Invalid/Empty. Generating SYNTHETIC SKIDPAD.")
        theta = np.linspace(0, 2*np.pi, 200)
        x_dummy = 12 * np.cos(theta)
        y_dummy = 12 * np.sin(theta)
        track_gen.from_centerline(x_dummy, y_dummy, track_width=4.0, closed_loop=True)
        t_arrays = track_gen.get_arrays()
        
        N_dummy = len(t_arrays['s'])
        real_data['x'] = t_arrays['x']
        real_data['y'] = t_arrays['y']
        real_data['s'] = t_arrays['s']
        real_data['velocity'] = np.ones(N_dummy) * 15.0 
        real_data['steer']    = np.ones(N_dummy) * 0.1
        real_data['throttle'] = np.ones(N_dummy) * 0.5
        real_data['brake']    = np.zeros(N_dummy)

    track_arrays = track_gen.get_arrays()

    # 4. SOLVE OPTIMAL CONTROL
    print("[3/6] Solving Optimal Lap (Ghost Car)...")
    solver = OptimalLapSolver(VP_DICT, TP_DICT)
    N_solver = 100
    s_interp = np.linspace(track_arrays['s'][0], track_arrays['s'][-1], N_solver)
    k_interp = np.interp(s_interp, track_arrays['s'], track_arrays['k'])
    w_interp = np.ones(N_solver) * 3.0 
    
    ghost_data = solver.solve(s_interp, k_interp, w_interp, w_interp, N=N_solver-1)
    
    # 5. VALIDATION & COACHING
    print("[4/6] Validating Driver Data...")
    ghost_track_model = {'s': ghost_data['s'], 'x': np.zeros(N_solver), 'y': np.zeros(N_solver), 'k': k_interp}
    validator = ModelValidator(ghost_track_model)
    
    if track_generated:
        real_synced = validator.sync_telemetry_to_track(real_data)
    else:
        real_synced = {}
        for key in ['velocity', 'steer', 'throttle', 'brake']:
            if key in real_data:
                real_synced[key] = np.interp(ghost_data['s'], real_data['s'], real_data[key])
        real_synced['s'] = ghost_data['s']
        
    print("[5/6] Generating Coaching Report...")
    coach = DriverCoach(ghost_track_model)
    coaching_df = coach.analyze_lap(real_synced, ghost_data)
    print("\n=== DRIVER ADVICE (Top 3) ===")
    if not coaching_df.empty:
        print(coaching_df.head(3).to_string(index=False))

    # 6. SETUP OPTIMIZATION (NEW STEP)
    print("\n[6/6] Running Setup Optimization (Genetic Algorithm)...")
    print("      (This may take 30-60 seconds)")
    
    # Using the improved optimizer with larger population
    optimizer = SetupOptimizer(pop_size=50, generations=20) 
    final_pop, final_obj = optimizer.run()
    
    # Save Results
    results_df = pd.DataFrame(final_pop)
    results_df['Lat_G_Score'] = final_obj[:, 0]
    results_df['Stability_Overshoot'] = final_obj[:, 1]
    
    out_file = os.path.join(base_dir, 'optimization_results.csv')
    results_df.to_csv(out_file, index=False)
    print(f"✅ Optimization Complete. Results saved to {out_file}")
    
    print("\n🎉 ALL TASKS DONE. Run 'streamlit run visualization/dashboard.py' to view results.")

if __name__ == "__main__":
    main()