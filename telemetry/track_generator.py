import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

class TrackGenerator:
    """
    Advanced Track Generator for Formula Student.
    """
    def __init__(self, smoothing_factor=2.0):
        self.s_factor = smoothing_factor
        self.s = None
        self.x = None
        self.y = None
        self.k = None
        self.w_l = None
        self.w_r = None

    def from_centerline(self, x_raw, y_raw, track_width=3.0, closed_loop=True):
        # 1. Input Validation
        if len(x_raw) < 10:
            raise ValueError(f"Not enough GPS points (<10). Got {len(x_raw)}.")

        # 2. Filter Stationary Points
        dx = np.diff(x_raw)
        dy = np.diff(y_raw)
        dist = np.sqrt(dx**2 + dy**2)
        
        # DEBUG PRINT
        print(f"[TrackGen] Max step distance: {np.max(dist):.4f} m")
        print(f"[TrackGen] Avg step distance: {np.mean(dist):.4f} m")
        
        # LOWER THRESHOLD to 0.05m (5cm) to catch slow movement
        mask = np.insert(dist > 0.05, 0, True) 
        x = x_raw[mask]
        y = y_raw[mask]
        
        if len(x) < 4:
            # Fallback: If filtering removed everything, the car didn't move enough.
            # But maybe units are wrong? (e.g. x is in km not meters?)
            raise ValueError(f"Track too short. Points remaining: {len(x)}. Max movement was {np.max(dist):.4f}m. Check GPS units.")

        # 3. Check for Closed Loop Feasibility
        # If start and end are far apart (>15m), force open loop
        gap = np.sqrt((x[0]-x[-1])**2 + (y[0]-y[-1])**2)
        if gap > 15.0:
            print(f"[TrackGen] Warning: Loop gap is {gap:.1f}m. Treating as Open Sprint.")
            closed_loop = False
        
        # 4. Deduplicate exact matches (Floating point tolerance)
        points = np.vstack((x, y)).T
        _, idx = np.unique(np.round(points, 3), axis=0, return_index=True)
        idx = np.sort(idx)
        x = x[idx]
        y = y[idx]

        # 5. Close the loop manually if requested
        if closed_loop:
            # Append start point to end to ensure geometric closure
            if gap > 0.1: 
                x = np.append(x, x[0])
                y = np.append(y, y[0])

        # 6. Fit B-Spline
        try:
            # k=3 (Cubic), s is smoothing factor
            # quiet=1 suppresses warnings
            tck, u = interp.splprep([x, y], s=self.s_factor, k=3, per=closed_loop)
        except Exception as e:
            print(f"[TrackGen] Spline fit failed: {e}")
            print("[TrackGen] Retrying with higher smoothing...")
            tck, u = interp.splprep([x, y], s=self.s_factor*10, k=3, per=closed_loop)
        
        # 7. Resample to Spatial Domain
        # High res first to calculate arc length
        u_fine = np.linspace(0, 1.0, 2000)
        x_fine, y_fine = interp.splev(u_fine, tck)
        
        # Calculate Arc Length (s)
        dx_fine = np.gradient(x_fine)
        dy_fine = np.gradient(y_fine)
        ds = np.sqrt(dx_fine**2 + dy_fine**2)
        s_cumulative = np.cumsum(ds)
        total_length = s_cumulative[-1]
        
        # 8. Final Resampling (Constant ds = 1.0m)
        resolution = 1.0 
        N_nodes = int(total_length / resolution)
        
        self.s = np.linspace(0, total_length, N_nodes)
        
        # Map s back to u
        # Scale s to [0, 1] range of u (approx)
        # Better: Interpolate u from s_cumulative
        u_at_s = np.interp(self.s, s_cumulative, u_fine)
        
        # Evaluate Spline at uniform s
        self.x, self.y = interp.splev(u_at_s, tck)
        
        # 9. Derivatives
        dx_du, dy_du = interp.splev(u_at_s, tck, der=1)
        ddx_du, ddy_du = interp.splev(u_at_s, tck, der=2)
        
        # Heading
        self.psi = np.arctan2(dy_du, dx_du)
        self.psi = np.unwrap(self.psi)
        
        # Curvature
        num = dx_du * ddy_du - dy_du * ddx_du
        den = (dx_du**2 + dy_du**2)**1.5
        self.k = num / (den + 1e-6)
        
        # Boundaries
        self.w_l = np.ones_like(self.s) * (track_width / 2.0)
        self.w_r = np.ones_like(self.s) * (track_width / 2.0)
        
        print(f"[TrackGen] Success. L={total_length:.1f}m, Nodes={N_nodes}")
        return self

    def get_arrays(self):
        if self.s is None: raise ValueError("Track not generated.")
        return {'s': self.s, 'x': self.x, 'y': self.y, 'psi': self.psi, 'k': self.k, 'w_l': self.w_l, 'w_r': self.w_r}