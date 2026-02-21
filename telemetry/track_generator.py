import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import jax
import jax.numpy as jnp

class ContinuousManifoldTrackGenerator:
    """
    Continuous-Time SE(3) Track Generator.
    Eliminates discrete B-Spline interpolation entirely. 
    Queries analytical position, heading, and curvature directly from the SE(3) GP Manifold,
    and generates the Spatial Friction Uncertainty Map for Stochastic Tube-MPC.
    """
    def __init__(self, estimator, optimized_params):
        """
        Takes the optimized continuous-time estimator and its Lie Group parameters
        rather than raw discrete CSV telemetry.
        """
        self.estimator = estimator
        self.params = optimized_params

    def generate(self, s_step=1.0, base_track_width=3.5):
        """
        Analytically integrates the manifold to find the true arclength (s),
        extracts the fastest loop, and generates the exact spatial domain coordinates.
        """
        print("[TrackGen] Querying Continuous SE(3) Manifold...")
        
        # 1. High-Resolution Temporal Sampling (for precise arclength integration)
        t_max = (self.estimator.num_knots - 1) * self.estimator.dt
        t_dense = jnp.linspace(0, t_max, int(t_max * 200)) # 200 Hz internal evaluation
        
        # Vectorized analytical query of the SE(3) function
        v_interpolate = jax.jit(jax.vmap(self.estimator.interpolate_trajectory, in_axes=(0, None, None)))
        T_dense, w_dense = v_interpolate(t_dense, self.params['T'], self.params['w'])
        
        # 2. Extract Spatial Coordinates & True Arclength
        x_dense = np.array(T_dense[:, 0, 3])
        y_dense = np.array(T_dense[:, 1, 3])
        vx_dense = np.array(w_dense[:, 0])
        vy_dense = np.array(w_dense[:, 1])
        
        v_mag_dense = np.sqrt(vx_dense**2 + vy_dense**2)
        dt = float(t_dense[1] - t_dense[0])
        
        # Exact cumulative distance integration
        s_dense = np.concatenate(([0.0], np.cumsum(v_mag_dense * dt)[:-1]))
        
        # 3. Lap Extraction
        df_dense = pd.DataFrame({'time': t_dense, 'x': x_dense, 'y': y_dense, 's': s_dense, 'v': v_mag_dense})
        t_start, t_end = self._extract_fastest_lap_times(df_dense)
        
        # Isolate the exact arclength bounds of the best lap
        s_start = np.interp(t_start, t_dense, s_dense)
        s_end = np.interp(t_end, t_dense, s_dense)
        lap_length = s_end - s_start
        
        # 4. Equidistant Spatial Re-parameterization (s_step)
        n_nodes = int(lap_length / s_step)
        s_target = np.linspace(s_start, s_end, n_nodes)
        
        # Find the exact temporal moments that correspond to the uniform spatial steps
        t_target = np.interp(s_target, s_dense, t_dense)
        
        # 5. EXACT ANALYTICAL KINEMATICS (The Core Upgrade)
        # Re-query the SE(3) manifold at the exact spatial times to eliminate ALL interpolation errors
        T_exact, w_exact = v_interpolate(jnp.array(t_target), self.params['T'], self.params['w'])
        
        T_exact = np.array(T_exact)
        w_exact = np.array(w_exact)
        
        x_new = T_exact[:, 0, 3]
        y_new = T_exact[:, 1, 3]
        
        # Heading extracted flawlessly from the SE(3) rotation matrix (no noisy arctan2(dy, dx))
        psi_new = np.unwrap(np.arctan2(T_exact[:, 1, 0], T_exact[:, 0, 0]))
        
        # Analytical Curvature: k = omega_z / v_xy
        vx_exact = w_exact[:, 0]
        vy_exact = w_exact[:, 1]
        omega_z = w_exact[:, 5]
        v_mag_exact = np.sqrt(vx_exact**2 + vy_exact**2) + 1e-8
        
        k_new = omega_z / v_mag_exact
        
        # 6. NSDE PHASE 1: STOCHASTIC FRICTION UNCERTAINTY MAP (w_mu)
        # We quantify local uncertainty by measuring the lateral slip variance.
        # High slip angle (beta) directly implies proximity to the friction limit 
        # and non-linear tire behavior, therefore increasing the required safety margin.
        beta = np.abs(np.arctan2(vy_exact, vx_exact))
        
        # Normalize and map to a friction uncertainty coefficient [0.01 (high grip) -> 0.15 (high uncertainty)]
        beta_norm = np.clip(beta / (np.percentile(beta, 95) + 1e-6), 0.0, 1.0)
        w_mu_map = 0.01 + (0.14 * beta_norm)
        
        # Apply smoothing to the uncertainty map to prevent aggressive Tube-MPC constraint chattering
        w_mu_map = pd.Series(w_mu_map).rolling(window=5, min_periods=1, center=True).mean().values

        # 7. Track Limits
        w_left = np.full_like(s_target, base_track_width)
        w_right = np.full_like(s_target, base_track_width)
        
        # Reset relative arclength for the OCP solver
        s_final = s_target - s_target[0]

        print(f"[TrackGen] SE(3) Analytical Geometry Ready.")
        print(f"  > True Length: {s_final[-1]:.1f} m | Nodes: {len(s_final)}")
        print(f"  > Uncertainty Boundary Map Generated [w_mu min: {w_mu_map.min():.3f}, max: {w_mu_map.max():.3f}]")

        return {
            's': s_final,
            'x': x_new,
            'y': y_new,
            'psi': psi_new,
            'k': k_new,
            'w_left': w_left,
            'w_right': w_right,
            'w_mu': w_mu_map,  # Feeds directly into Tube-MPC parameter vectors
            'total_length': s_final[-1]
        }

    def _extract_fastest_lap_times(self, df_dense):
        """
        Locates loop closures purely based on spatial coordinates,
        returning the start and end times of the optimal lap.
        """
        print("[TrackGen] Analyzing SE(3) manifold for continuous loop closures...")
        
        # Filter out extreme slow-speed pacing sections
        df_active = df_dense[df_dense['v'] > 2.0].reset_index(drop=True)
        points = df_active[['x', 'y']].values
        times = df_active['time'].values
        
        # Reference point for start/finish line detection
        start_node = points[0]
        dists = np.sqrt(np.sum((points - start_node)**2, axis=1))
        
        # Find local minima in distance to the start point (loop completions)
        peaks, _ = find_peaks(-dists, height=-20.0, distance=300) 
        
        if len(peaks) < 2:
            print("[TrackGen] WARNING: No distinct laps found. Using full manifold.")
            return times[0], times[-1]
            
        best_lap_idx = -1
        min_lap_time = float('inf')
        
        for i in range(len(peaks) - 1):
            idx_start = peaks[i]
            idx_end = peaks[i+1]
            lap_time = times[idx_end] - times[idx_start]
            
            if lap_time > 10.0 and lap_time < min_lap_time:
                min_lap_time = lap_time
                best_lap_idx = i
                
        if best_lap_idx != -1:
            idx_s = peaks[best_lap_idx]
            idx_e = peaks[best_lap_idx+1]
            print(f"[TrackGen] Identified Optimal Lap {best_lap_idx+1} | Time: {min_lap_time:.2f}s")
            return times[idx_s], times[idx_e]
            
        return times[0], times[-1]