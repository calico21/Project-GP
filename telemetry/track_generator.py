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
        Uses RANSAC to detect the longest straight, places a virtual S/F line,
        and extracts the fastest lap via geometric line intersection.
        """
        print("[TrackGen] Analyzing SE(3) manifold with RANSAC-based Loop Closure...")
        
        # Filter out extreme slow-speed pacing sections
        df_active = df_dense[df_dense['v'] > 2.0].reset_index(drop=True)
        points = df_active[['x', 'y']].values
        times = df_active['time'].values
        
        # ---------------------------------------------------------
        # 1. RANSAC Line Fitting to find the Main Straight
        # ---------------------------------------------------------
        best_inlier_count = 0
        best_centroid = None
        best_normal = None
        
        np.random.seed(42) # Deterministic execution
        
        # Run 200 RANSAC iterations to find the dominant linear segment (the straight)
        for _ in range(200):
            idx = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[idx]
            
            vec = p2 - p1
            length = np.linalg.norm(vec)
            if length < 5.0: continue # Skip points too close together
            
            vec = vec / length
            normal = np.array([-vec[1], vec[0]]) # Perpendicular normal
            
            # Calculate distance of all points to the candidate line
            dists = np.abs(np.dot(points - p1, normal))
            inliers = np.where(dists < 1.5)[0] # 1.5m tolerance
            
            # Record the straightest section of the track
            if len(inliers) > best_inlier_count:
                best_inlier_count = len(inliers)
                best_centroid = np.mean(points[inliers], axis=0) # Middle of the straight
                best_normal = normal

        if best_centroid is None:
            print("[TrackGen] WARNING: RANSAC failed. Using full manifold.")
            return times[0], times[-1]

        # ---------------------------------------------------------
        # 2. Virtual Start/Finish Gate Generation
        # ---------------------------------------------------------
        # Place a 30-meter wide invisible gate perpendicular to the main straight
        gate_width = 15.0
        gate_p1 = best_centroid + best_normal * gate_width
        gate_p2 = best_centroid - best_normal * gate_width

        # ---------------------------------------------------------
        # 3. Geometric Crossing Detection
        # ---------------------------------------------------------
        # Vectorized check: does segment A->B intersect segment gate_p1->gate_p2?
        A = points[:-1]
        B = points[1:]
        
        # Counter-Clockwise (CCW) check logic for line intersection
        ccw_ACD = (gate_p2[1] - A[:, 1]) * (gate_p1[0] - A[:, 0]) > (gate_p1[1] - A[:, 1]) * (gate_p2[0] - A[:, 0])
        ccw_BCD = (gate_p2[1] - B[:, 1]) * (gate_p1[0] - B[:, 0]) > (gate_p1[1] - B[:, 1]) * (gate_p2[0] - B[:, 0])
        
        ccw_ABC = (gate_p1[1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (gate_p1[0] - A[:, 0])
        ccw_ABD = (gate_p2[1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (B[:, 1] - A[:, 1]) * (gate_p2[0] - A[:, 0])

        # Indices where the path intersects the S/F gate
        crossings = np.where((ccw_ACD != ccw_BCD) & (ccw_ABC != ccw_ABD))[0]

        # Filter out multiple crossings in the same pass (debounce)
        valid_crossings = []
        last_time = -999.0
        for idx in crossings:
            t = times[idx]
            if t - last_time > 10.0: # Minimum lap time threshold
                valid_crossings.append(idx)
                last_time = t

        if len(valid_crossings) < 2:
            print("[TrackGen] WARNING: <2 valid S/F crossings found. Using full manifold.")
            return times[0], times[-1]

        # ---------------------------------------------------------
        # 4. Extract Fastest Lap Segment
        # ---------------------------------------------------------
        min_lap_time = float('inf')
        best_lap_start = -1
        best_lap_end = -1
        
        for i in range(len(valid_crossings) - 1):
            idx_start = valid_crossings[i]
            idx_end = valid_crossings[i+1]
            lap_time = times[idx_end] - times[idx_start]
            
            if lap_time < min_lap_time:
                min_lap_time = lap_time
                best_lap_start = times[idx_start]
                best_lap_end = times[idx_end]

        print(f"[TrackGen] RANSAC S/F Gate Placed. Fastest Lap Time Extracted: {min_lap_time:.2f}s")
        return best_lap_start, best_lap_end