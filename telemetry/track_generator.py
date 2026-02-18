import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

class TrackGenerator:
    """
    Converts raw GPS data (lat/lon) into a smooth path-coordinate track model.
    Critical for the OCP solver, which requires continuous curvature derivatives.
    """
    def __init__(self, df_log):
        self.df = df_log
        
    def generate(self, s_step=2.0, smoothing_s=1.0):
        """
        Generates track model [s, x, y, psi, k]
        
        Args:
            s_step: Distance between nodes [m] (Lower = more precision, slower OCP)
            smoothing_s: Spline smoothing factor (Higher = smoother but less accurate to GPS)
        """
        # 1. Extract and Center Coordinates
        # (Assuming df has 'x_m' and 'y_m' converted from Lat/Lon already)
        if 'x_m' not in self.df.columns:
            # Fallback if raw lat/lon
            # Simple equirectangular projection (sufficient for race tracks)
            R_earth = 6371000
            lat0 = self.df['lat'].mean()
            self.df['x_m'] = (self.df['lon'] - self.df['lon'].mean()) * (np.pi/180) * R_earth * np.cos(lat0 * np.pi/180)
            self.df['y_m'] = (self.df['lat'] - self.df['lat'].mean()) * (np.pi/180) * R_earth

        x_raw = self.df['x_m'].values
        y_raw = self.df['y_m'].values
        
        # Close the loop explicitly for the spline
        x_raw = np.append(x_raw, x_raw[0])
        y_raw = np.append(y_raw, y_raw[0])

        # 2. Fit Spline (B-Spline representation)
        # k=3 (Cubic spline) ensures continuous 2nd derivative (Curvature)
        # s=smoothing_s handles GPS noise
        try:
            tck, u = splprep([x_raw, y_raw], s=smoothing_s, k=3, per=True)
        except Exception as e:
            print(f"[TrackGen] Spline fit failed: {e}. Trying without periodic constraint...")
            tck, u = splprep([x_raw, y_raw], s=smoothing_s, k=3, per=False)

        # 3. Resample at fixed distance intervals
        # First, evaluate at high resolution to measure total length
        u_fine = np.linspace(0, 1, 10000)
        x_fine, y_fine = splev(u_fine, tck)
        
        # Calculate cumulative distance 's'
        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        ds = np.sqrt(dx**2 + dy**2)
        total_length = np.sum(ds)
        
        # Create evaluation points based on desired s_step
        n_nodes = int(total_length / s_step)
        u_new = np.linspace(0, 1, n_nodes)
        
        # 4. Evaluate Spline and Derivatives
        # der=0 -> Position (x, y)
        x_smooth, y_smooth = splev(u_new, tck)
        
        # der=1 -> Velocity vector (dx/du, dy/du) -> Heading
        dx_du, dy_du = splev(u_new, tck, der=1)
        psi = np.arctan2(dy_du, dx_du)
        
        # der=2 -> Acceleration vector -> Curvature
        ddx_du, ddy_du = splev(u_new, tck, der=2)
        
        # Curvature formula: k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        # This is the "Analytical Curvature" (Noise free!)
        num = dx_du * ddy_du - dy_du * ddx_du
        den = (dx_du**2 + dy_du**2)**1.5
        curvature = num / (den + 1e-6) # Avoid div/0

        # Calculate exact 's' for these nodes
        s_array = np.zeros(n_nodes)
        # We integrate the actual spline distance roughly
        # For OCP, s must be strictly increasing.
        # We approximate it by scaling u_new by total_length
        # (A simplified arc-length parameterization)
        s_array = u_new * total_length

        # 5. Track Width (Simple estimation)
        # In a real tool, you would extract this from left/right boundary GPS
        # Here we assume constant 10m width
        w_left = np.full(n_nodes, 5.0)
        w_right = np.full(n_nodes, 5.0)

        # 6. Format Output
        track_data = {
            's': s_array,
            'x': x_smooth,
            'y': y_smooth,
            'psi': psi,
            'k': curvature,
            'w_left': w_left,
            'w_right': w_right,
            'total_length': total_length
        }
        
        print(f"[TrackGen] Track generated. Length: {total_length:.1f}m, Nodes: {n_nodes}")
        return track_data

    def save_track(self, track_data, filename="track_model.csv"):
        df = pd.DataFrame({
            's': track_data['s'],
            'x': track_data['x'],
            'y': track_data['y'],
            'psi': track_data['psi'],
            'k': track_data['k'],
            'w_left': track_data['w_left'],
            'w_right': track_data['w_right']
        })
        df.to_csv(filename, index=False)
        print(f"[TrackGen] Saved to {filename}")