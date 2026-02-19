import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks

class TrackGenerator:
    """
    State-of-the-Art Track Generator.
    Generates a Degree-5 Periodic Minimum-Curvature B-Spline from telemetry.
    Provides exact algebraic coordinates for rapid OCP/acados evaluation.
    """
    def __init__(self, df_log):
        self.df = df_log

    def generate(self, s_step=1.0, smoothing_s=2.0):
        """
        Input: DataFrame with ['x', 'y', 'v', 'time'].
        Output: Dictionary with analytical track definition.
        """
        # 1. Validation
        if 'x' not in self.df.columns or 'y' not in self.df.columns:
            if 'lat' in self.df.columns:
                print("[TrackGen] 'x'/'y' missing. Converting from 'lat'/'lon'...")
                self._convert_gps_to_xy()
            else:
                raise ValueError("[TrackGenerator] Input dataframe missing 'x'/'y' columns.")

        # 2. EXTRACT BEST LAP
        df_lap = self._extract_fastest_lap(self.df)
        
        # 3. GEOMETRY CORRECTION (The True Distance Fix)
        dt_lap = np.diff(df_lap['time'], prepend=df_lap['time'].iloc[0])
        dt_lap[0] = 0.02 
        
        true_length = np.sum(df_lap['v'] * dt_lap)
        
        x_raw = df_lap['x'].values
        y_raw = df_lap['y'].values
        gps_dist_segs = np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2)
        gps_length = np.sum(gps_dist_segs)
        
        print(f"[TrackGen] Correction Check:")
        print(f"   > GPS Measured Length: {gps_length:.1f} m")
        print(f"   > CAN Speed Integral:  {true_length:.1f} m")
        
        scale_factor = 1.0
        if gps_length > 0:
            scale_factor = true_length / gps_length
            
        if abs(scale_factor - 1.0) > 0.1:
            print(f"[TrackGen] WARNING: Significant GPS scaling error detected.")
            print(f"   > Applying Scale Factor: {scale_factor:.4f} to match physics.")
            x_raw = (x_raw - x_raw[0]) * scale_factor + x_raw[0]
            y_raw = (y_raw - y_raw[0]) * scale_factor + y_raw[0]

        # 4. SPATIAL DOWNSAMPLING
        x_filt, y_filt = [x_raw[0]], [y_raw[0]]
        min_dist = 2.0 
        
        for i in range(1, len(x_raw)):
            dist = np.sqrt((x_raw[i] - x_filt[-1])**2 + (y_raw[i] - y_filt[-1])**2)
            if dist >= min_dist:
                x_filt.append(x_raw[i])
                y_filt.append(y_raw[i])
        
        # 5. CLOSED-FORM HIGH-DEGREE B-SPLINE GENERATION (The SOTA Upgrade)
        # Degree 5 (k=5) ensures C4 continuity -> perfectly smooth Curvature/Jerk
        # per=True enforces perfect mathematical closure of the loop
        
        print(f"[TrackGen] Fitting Degree-5 Periodic B-Spline...")
        tck, u = splprep([x_filt, y_filt], s=smoothing_s * len(x_filt), k=5, per=True)
        
        # Determine number of nodes based on required spatial step
        total_estimated_length = len(x_filt) * min_dist
        n_nodes = int(total_estimated_length / s_step)
        if n_nodes < 10: n_nodes = 10
        
        # Parameterize over the standardized spline variable 'u' [0, 1]
        u_new = np.linspace(0, 1.0, n_nodes)
        
        # Evaluate exact X, Y coordinates
        x_new, y_new = splev(u_new, tck)
        
        # 6. EXACT ANALYTICAL KINEMATICS
        # Calculate precise derivatives instead of using noisy cumulative sums
        dx, dy = splev(u_new, tck, der=1)
        ddx, ddy = splev(u_new, tck, der=2)
        
        # Exact Heading
        psi_final = np.unwrap(np.arctan2(dy, dx))
        
        # Exact Analytical Curvature: k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        k_new = (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**(1.5) + 1e-8)
        
        # 7. TRUE ARCLENGTH (S) MAPPING
        ds_segs = np.sqrt(np.diff(x_new)**2 + np.diff(y_new)**2)
        s_new = np.concatenate(([0], np.cumsum(ds_segs)))
        
        w_left = np.full_like(s_new, 3.5)
        w_right = np.full_like(s_new, 3.5)

        print(f"[TrackGen] B-Spline Geometry Ready. True Length: {s_new[-1]:.1f}m, Nodes: {len(s_new)}")

        return {
            's': s_new,
            'x': x_new,
            'y': y_new,
            'psi': psi_final,
            'k': k_new,
            'w_left': w_left,
            'w_right': w_right,
            'total_length': s_new[-1]
        }
    
    def _extract_fastest_lap(self, df):
        """
        Analyzes the trajectory to find loops and selects the fastest valid lap.
        """
        print("[TrackGen] Analyzing session for laps...")
        
        df = df[df['v'] > 1.0].reset_index(drop=True)
        if len(df) < 100:
            return df
            
        points = df[['x', 'y']].values
        times = df['time'].values
        
        start_node = points[0]
        dists = np.sqrt(np.sum((points - start_node)**2, axis=1))
        
        peaks, _ = find_peaks(-dists, height=-20.0, distance=200) 
        
        if len(peaks) < 2:
            print("[TrackGen] No distinct laps found. Using full path.")
            return df
            
        print(f"[TrackGen] Found {len(peaks)-1} potential laps.")
        
        best_lap_idx = -1
        min_lap_time = float('inf')
        
        for i in range(len(peaks) - 1):
            idx_start = peaks[i]
            idx_end = peaks[i+1]
            
            lap_time = times[idx_end] - times[idx_start]
            
            if lap_time < 4.0: 
                continue
                
            if lap_time < min_lap_time:
                min_lap_time = lap_time
                best_lap_idx = i
                
        if best_lap_idx != -1:
            idx_s = peaks[best_lap_idx]
            idx_e = peaks[best_lap_idx+1]
            print(f"[TrackGen] Selected Lap {best_lap_idx+1} (Time: {min_lap_time:.2f}s) as Reference.")
            return df.iloc[idx_s:idx_e].copy()
        else:
            print("[TrackGen] No valid laps found. Using full path.")
            return df

    def _convert_gps_to_xy(self):
        lat0 = self.df['lat'].mean()
        lon0 = self.df['lon'].mean()
        R_earth = 6378137.0
        
        def to_xy(row):
            dlat = np.deg2rad(row['lat'] - lat0)
            dlon = np.deg2rad(row['lon'] - lon0)
            x = R_earth * dlon * np.cos(np.deg2rad(lat0))
            y = R_earth * dlat
            return x, y
            
        coords = self.df.apply(to_xy, axis=1)
        self.df['x'] = [c[0] for c in coords]
        self.df['y'] = [c[1] for c in coords]