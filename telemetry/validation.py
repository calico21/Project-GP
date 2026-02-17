import numpy as np
import scipy.interpolate as interp
from scipy.spatial.distance import cdist

class ModelValidator:
    """
    Validation Engine for the Digital Twin.
    
    Functions:
    1. Spatial Synchronization: Maps real Telemetry (Time-Domain) to Simulation (Space-Domain).
    2. Model Accuracy: Computes RMSE and R2 scores between Sim and Real.
    3. Driver Analysis: Compares Real Driver to Optimal Control (Ghost Car).
    """
    def __init__(self, track_model):
        """
        Args:
            track_model (dict): Output from TrackGenerator (contains s, x, y, k)
        """
        self.track = track_model
        
        # Create fast lookups for track path
        self.track_pts = np.vstack((self.track['x'], self.track['y'])).T
        self.s_vector = self.track['s']

    def sync_telemetry_to_track(self, real_data):
        """
        Maps real telemetry GPS (x,y) to track arc-length (s).
        
        Args:
            real_data (dict): {'time', 'x', 'y', 'velocity', ...} arrays
            
        Returns:
            synced_data (dict): Real data resampled to match the exact 's' steps of the track model.
        """
        print("[Validator] Synchronizing Telemetry to Track Path...")
        
        real_pts = np.vstack((real_data['x'], real_data['y'])).T
        
        # 1. Find nearest track node for every telemetry point
        # (Using cdist is expensive for large arrays, KDTree is better, but keeping deps minimal)
        # Optimization: process in chunks or assume continuity. 
        # Simple method: Distance to all track points
        dists = cdist(real_pts, self.track_pts)
        nearest_idx = np.argmin(dists, axis=1)
        
        # 2. Get 's' for every real point
        real_s = self.s_vector[nearest_idx]
        
        # 3. Handle Lap Rollover (if s jumps from max back to 0)
        # (Omitted for brevity, assuming single lap data for now)
        
        # 4. Remove duplicates/backtracking (Driver must move forward)
        # We sort by s to ensure monotonicity required for interpolation
        sort_mask = np.argsort(real_s)
        real_s_sorted = real_s[sort_mask]
        
        # 5. Interpolate Real Data onto Track 's' grid
        # We want to know: What was Real Velocity at s=100.0?
        # Target: self.s_vector
        synced = {}
        keys_to_sync = ['velocity', 'yaw_rate', 'g_lat', 'g_long', 'throttle', 'brake', 'steer']
        
        for k in keys_to_sync:
            if k in real_data:
                # Sort the source data
                vals_sorted = real_data[k][sort_mask]
                
                # Interpolate
                # fill_value="extrapolate" handles slight endpoint mismatches
                f = interp.interp1d(real_s_sorted, vals_sorted, kind='linear', bounds_error=False, fill_value="extrapolate")
                synced[k] = f(self.s_vector)
                
        synced['s'] = self.s_vector
        return synced

    def validate_model(self, real_data, sim_data):
        """
        Compares Simulation (Model) vs Reality.
        
        Args:
            real_data (dict): Output from log ingestion (Time domain)
            sim_data (dict): Output from Vehicle Simulator (NOT OCP, but the Fwd Integrator)
                             Using the same control inputs as real_data.
                             
        Returns:
            metrics (dict): RMSE, R2, etc.
        """
        # Note: Ideally we simulate the model using Real Inputs (Steer, Speed) 
        # to check dynamic validity. Here we assume sim_data is already spatially aligned.
        
        # Simple RMSE Calculation
        metrics = {}
        
        targets = ['velocity', 'yaw_rate', 'g_lat']
        
        for t in targets:
            if t in real_data and t in sim_data:
                real = real_data[t]
                sim = sim_data[t]
                
                # RMSE
                mse = np.mean((real - sim)**2)
                rmse = np.sqrt(mse)
                
                # Normalized RMSE (NRMSE) %
                range_val = np.max(real) - np.min(real)
                if range_val > 0:
                    nrmse = (rmse / range_val) * 100.0
                else:
                    nrmse = 0.0
                
                metrics[f"{t}_rmse"] = rmse
                metrics[f"{t}_accuracy"] = 100.0 - nrmse
                
        return metrics

    def analyze_driver(self, real_data, ideal_data):
        """
        Compares Real Driver vs. Ghost Car (OCP).
        
        Args:
            real_data (dict): Real telemetry
            ideal_data (dict): OCP Solver output (The "Ghost Car")
            
        Returns:
            analysis (dict): Time loss per sector, corner analysis.
        """
        # 1. Sync Real Data to OCP 's' vector (if not already)
        if len(real_data['velocity']) != len(ideal_data['s']):
            real_synced = self.sync_telemetry_to_track(real_data)
        else:
            real_synced = real_data
            
        # 2. Calculate Time Delta
        # Time = Integral(1/v ds)
        # We calculate cumulative time at each 's' for both
        ds = np.diff(ideal_data['s'], prepend=0)
        
        dt_ideal = ds / (ideal_data['v'] + 0.1)
        t_ideal = np.cumsum(dt_ideal)
        
        dt_real = ds / (real_synced['velocity'] + 0.1)
        t_real = np.cumsum(dt_real)
        
        delta_t = t_real - t_ideal # Positive means Real is slower
        
        # 3. Identify Top 3 Time Loss Areas
        # Calculate local time loss rate (slope of delta_t)
        loss_rate = np.gradient(delta_t)
        
        # Find peaks in loss rate
        # Simple heuristic: Split track into 100m sectors
        sectors = []
        L = ideal_data['s'][-1]
        sector_len = 100.0
        n_sectors = int(L / sector_len)
        
        for i in range(n_sectors):
            s_start = i * sector_len
            s_end = (i+1) * sector_len
            
            mask = (ideal_data['s'] >= s_start) & (ideal_data['s'] < s_end)
            if np.any(mask):
                loss = delta_t[mask][-1] - delta_t[mask][0]
                sectors.append({
                    'sector_id': i+1,
                    'start': s_start,
                    'loss': loss
                })
        
        # Sort by biggest loss
        sectors.sort(key=lambda x: x['loss'], reverse=True)
        
        return {
            'total_time_lost': delta_t[-1],
            'worst_sectors': sectors[:3],
            'delta_t_trace': delta_t,
            'real_v_trace': real_synced['velocity'],
            'ideal_v_trace': ideal_data['v']
        }

if __name__ == "__main__":
    # Test Stub
    # 1. Create Dummy Track
    s = np.linspace(0, 100, 100)
    track = {'s': s, 'x': s, 'y': np.zeros_like(s)}
    
    validator = ModelValidator(track)
    
    # 2. Create Dummy Data
    real = {
        'x': s + 0.1, 
        'y': np.zeros_like(s), 
        'velocity': np.sin(s/10) * 10 + 15
    }
    
    ideal = {
        's': s,
        'v': np.sin(s/10) * 10 + 16 # Ghost car is 1 m/s faster
    }
    
    # 3. Analyze
    report = validator.analyze_driver(real, ideal)
    print(f"Total Time Lost: {report['total_time_lost']:.2f}s")
    print("Worst Sector:", report['worst_sectors'][0])