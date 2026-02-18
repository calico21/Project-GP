import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from telemetry.log_ingestion import LogIngestion
from models.vehicle_dynamics import MultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

class InertiaEstimator:
    def __init__(self, log_path):
        self.ingestor = LogIngestion(log_path)
        self.df = self.ingestor.process()
        
        # --- IMU COLUMN MAPPING ---
        # We need to find Yaw Rate (rad/s). LogIngestion might not have caught it.
        self.yaw_col = None
        candidates = ['yaw_rate', 'gyro_z', 'ang_vel_z', 'r', 'yawspeed', 'rotation_speed']
        
        # Check normalized columns first
        for col in self.df.columns:
            if col in candidates:
                self.yaw_col = col
                break
        
        # Check original raw columns if not found
        if not self.yaw_col:
            # We might have to look at the raw CSV again or ask user
            print(f"[Estimator] Warning: Could not auto-detect Yaw Rate column.")
            print(f"Available columns: {list(self.df.columns)}")
            # For this tool, we'll try to guess or fail gracefully
            pass

    def select_dynamic_segment(self):
        """
        Finds a segment of the log with high steering activity (Chicane/Corner).
        Inertia only matters when inputs change!
        """
        if 'delta' not in self.df.columns:
            raise ValueError("Steering ('delta') missing from logs.")
            
        # Calculate steering derivative (Speed of hand movement)
        d_steer = np.gradient(self.df['delta'])
        
        # Find index of max steering activity
        idx_max = np.argmax(np.abs(d_steer))
        
        # Take a window around that event (e.g., +/- 2 seconds)
        window = 200 # assuming 100Hz -> 2s
        start = max(0, idx_max - window)
        end = min(len(self.df), idx_max + window)
        
        return self.df.iloc[start:end].copy()

    def run_simulation(self, df_segment, test_Iz):
        """
        Simulates the segment using a specific Inertia value.
        """
        # Override config
        sim_params = VP_DICT.copy()
        sim_params['Iz'] = test_Iz
        
        vehicle = MultiBodyVehicle(sim_params, TP_DICT)
        
        # Initial State from Log
        # [X, Y, psi, vx, vy, r, delta, Tf, Tr]
        v0 = df_segment['v'].iloc[0]
        r0 = df_segment[self.yaw_col].iloc[0] if self.yaw_col else 0.0
        delta0 = df_segment['delta'].iloc[0]
        
        x_curr = np.zeros(10)
        x_curr[3] = v0
        x_curr[5] = r0
        x_curr[6] = delta0
        x_curr[7] = 80.0 # Assume warm tires
        x_curr[8] = 80.0
        
        sim_yaw = []
        dt = 0.01 # Assume 100Hz or calc from time
        
        # Setup Params (Standard)
        # k_f, k_r, arb_f, arb_r, c_f, c_r
        setup = [30000, 30000, 500, 500, 3000, 3000] 

        for i in range(len(df_segment)):
            row = df_segment.iloc[i]
            
            # Input: Real Steering from Log
            u = [row['delta'], 0] # Throttle ignored for lateral dynamics
            
            # Step
            x_next = vehicle.simulate_step(x_curr, u, setup, dt=dt)
            
            sim_yaw.append(x_next[5])
            
            # Update state (Keep Speed locked to log to isolate Lat Dynamics)
            x_curr = x_next
            x_curr[3] = row['v'] # Overwrite V with real V
            
        return np.array(sim_yaw)

    def fit(self):
        if not self.yaw_col:
            print("[Error] Cannot fit Inertia without Yaw Rate log data.")
            return
            
        print("[Estimator] Identifying high-dynamic segment...")
        segment = self.select_dynamic_segment()
        real_yaw = segment[self.yaw_col].values
        
        def objective(Iz_guess):
            # Run Sim
            sim_yaw = self.run_simulation(segment, Iz_guess)
            
            # Calculate RMSE (Root Mean Square Error)
            # We weigh the error by steering rate? No, simple RMSE is usually fine.
            error = np.sqrt(np.mean((real_yaw - sim_yaw)**2))
            return error
        
        print(f"[Estimator] Optimizing Iz for {len(segment)} samples...")
        
        # Bounds: Go Kart (20) to Truck (4000)
        res = minimize_scalar(objective, bounds=(500, 3000), method='bounded')
        
        best_iz = res.x
        print(f"\n" + "="*40)
        print(f"CALCULATED YAW INERTIA (Iz): {best_iz:.2f} kg*m^2")
        print(f"="*40 + "\n")
        
        # Plot result
        sim_yaw_opt = self.run_simulation(segment, best_iz)
        
        plt.figure(figsize=(10, 6))
        plt.plot(real_yaw, label='Real Telemetry', color='black', alpha=0.7)
        plt.plot(sim_yaw_opt, label=f'Simulated (Iz={int(best_iz)})', color='cyan', linestyle='--')
        plt.title(f"Inertia Fitting: Real vs Sim ({len(segment)} samples)")
        plt.ylabel("Yaw Rate (rad/s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/inertia_fitting.py <path_to_log.csv>")
        # Fallback for testing
        print("Using default log...")
        default_log = os.path.join(project_root, 'data/logs/sample_log.csv')
        est = InertiaEstimator(default_log)
        est.fit()
    else:
        est = InertiaEstimator(sys.argv[1])
        est.fit()