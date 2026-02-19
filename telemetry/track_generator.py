import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

class TrackGenerator:
    """
    State-of-the-Art Track Generator.
    Generates a G^2 Continuous Clothoid Spline (Euler Spirals) from telemetry.
    Eliminates OCP solver jitter by enforcing piecewise-linear curvature.
    """
    def __init__(self, df_log):
        self.df = df_log

    def generate(self, s_step=1.0, smoothing_s=5.0):
        """
        Input: DataFrame with ['x', 'y', 'v', 'time'].
        Output: Dictionary with track definition.
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
        
        x_filt.append(x_filt[0])
        y_filt.append(y_filt[0])
        
        # 5. G^2 CLOTHOID SPLINE GENERATION (The Upgrade)
        # Instead of B-Splines, we enforce G2 continuity by calculating raw heading/curvature,
        # forcing curvature to be piecewise linear, and reintegrating (Fresnel approach).
        
        x_f, y_f = np.array(x_filt), np.array(y_filt)
        dx_f = np.diff(x_f, append=x_f[1])
        dy_f = np.diff(y_f, append=y_f[1])
        ds_f = np.sqrt(dx_f**2 + dy_f**2)
        s_f = np.cumsum(ds_f) - ds_f[0]
        
        # Raw Heading and Curvature
        psi_f = np.unwrap(np.arctan2(dy_f, dx_f))
        dpsi_f = np.diff(psi_f, append=psi_f[1])
        kappa_raw = dpsi_f / (ds_f + 1e-6)
        
        # Heavy Savitzky-Golay filter to smooth curvature and extract underlying Clothoid segments
        window_length = min(21, len(kappa_raw) // 2 * 2 + 1)
        kappa_smooth = savgol_filter(kappa_raw, window_length, polyorder=2)
        
        # 6. RESAMPLE & RE-INTEGRATE (Euler Spirals)
        n_nodes = int(s_f[-1] / s_step)
        if n_nodes < 10: n_nodes = 10
        s_new = np.linspace(0, s_f[-1], n_nodes)
        
        # Linear interpolation of smooth curvature creates explicit Clothoids (linear k(s))
        kappa_interp = interp1d(s_f, kappa_smooth, kind='linear', fill_value='extrapolate')
        k_new = kappa_interp(s_new)
        
        # Forward Euler Integration of Curvature -> Heading -> X/Y
        ds_step = s_new[1] - s_new[0]
        psi_new = psi_f[0] + np.cumsum(k_new) * ds_step
        
        x_new = np.zeros_like(s_new)
        y_new = np.zeros_like(s_new)
        x_new[0], y_new[0] = x_f[0], y_f[0]
        
        for i in range(1, n_nodes):
            x_new[i] = x_new[i-1] + np.cos(psi_new[i]) * ds_step
            y_new[i] = y_new[i-1] + np.sin(psi_new[i]) * ds_step
            
        # 7. CLOSED-LOOP DRIFT CORRECTION
        # Because we integrated an approximation, the start and end points might not perfectly align.
        # We apply a linear error correction distributed across the entire lap.
        err_x = x_f[-1] - x_new[-1]
        err_y = y_f[-1] - y_new[-1]
        
        correction_factor = s_new / s_new[-1]
        x_new = x_new + err_x * correction_factor
        y_new = y_new + err_y * correction_factor
        
        # Recalculate true heading after drift correction
        dx_new = np.gradient(x_new, ds_step)
        dy_new = np.gradient(y_new, ds_step)
        psi_final = np.unwrap(np.arctan2(dy_new, dx_new))
        
        w_left = np.full_like(s_new, 3.5)
        w_right = np.full_like(s_new, 3.5)

        print(f"[TrackGen] G2 Clothoid Track Generated. Length: {s_new[-1]:.1f}m, Nodes: {len(s_new)}")

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