import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import sys
import os

# --- IMPORT PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the Master Config to ensure Physics Consistency
try:
    from data.configs.vehicle_params import vehicle_params as VP
except ImportError:
    # Fallback if config is missing
    VP = {'m': 1200.0, 'A': 2.2, 'Cd': 0.8}

class DriverCoach:
    """
    Advanced Driver Analysis Engine.
    
    Responsibilities:
    1. Segment Track into 'Corners' and 'Straights' based on curvature.
    2. Compare Real Telemetry vs. Ghost Telemetry (OCP).
    3. Generate textual advice (e.g., "Brake Later", "Missed Apex").
    """
    def __init__(self, track_data):
        self.track = track_data
        self.corners = self._detect_corners()
        
    def _detect_corners(self):
        """
        Splits the track into segments based on curvature (k).
        Returns a list of dicts: [{'id': 1, 'start': s1, 'end': s2, 'apex': s_apex}, ...]
        """
        k = np.abs(self.track['k'])
        s = self.track['s']
        
        # 1. Identify "High Curvature" regions
        # Threshold: 1/R. For R=200m -> k=0.005. For R=50m -> k=0.02.
        # Let's set a threshold for "Straight" vs "Corner"
        k_threshold = 0.005 
        is_corner = k > k_threshold
        
        corners = []
        in_corner = False
        start_idx = 0
        
        # 2. Iterate and group segments
        for i, val in enumerate(is_corner):
            if val and not in_corner:
                in_corner = True
                start_idx = i
            elif not val and in_corner:
                in_corner = False
                end_idx = i
                
                # Filter noise: A corner must be at least 10m long
                if (s[end_idx] - s[start_idx]) > 10.0:
                    # Find Apex (Max Curvature in this segment)
                    segment_k = k[start_idx:end_idx]
                    apex_local_idx = np.argmax(segment_k)
                    apex_idx = start_idx + apex_local_idx
                    
                    corners.append({
                        'id': len(corners) + 1,
                        'start_s': s[start_idx],
                        'end_s': s[end_idx],
                        'apex_s': s[apex_idx],
                        'length': s[end_idx] - s[start_idx]
                    })
        return corners

    def analyze_lap(self, df_real, df_ghost):
        """
        Generates the 'Coaching Report'.
        Comparing Real Driver (df_real) vs Optimal Driver (df_ghost)
        """
        report = []
        
        # Pre-interpolation functions for fast lookup
        # We map everything to distance 's'
        interp_v_real = lambda x: np.interp(x, df_real['s'], df_real['v'])
        interp_v_ghost = lambda x: np.interp(x, df_ghost['s'], df_ghost['v'])
        interp_t_real = lambda x: np.interp(x, df_real['s'], df_real['time'])
        interp_t_ghost = lambda x: np.interp(x, df_ghost['s'], df_ghost['time'])
        
        for c in self.corners:
            # 1. Get Speeds at critical points
            s_entry = c['start_s']
            s_apex = c['apex_s']
            s_exit = c['end_s']
            
            v_real_entry = interp_v_real(s_entry)
            v_ghost_entry = interp_v_ghost(s_entry)
            
            v_real_apex = interp_v_real(s_apex)
            v_ghost_apex = interp_v_ghost(s_apex)
            
            v_real_exit = interp_v_real(s_exit)
            v_ghost_exit = interp_v_ghost(s_exit)
            
            # 2. Calculate Time Loss in this corner
            t_entry_real = interp_t_real(s_entry)
            t_exit_real = interp_t_real(s_exit)
            dt_real = t_exit_real - t_entry_real
            
            t_entry_ghost = interp_t_ghost(s_entry)
            t_exit_ghost = interp_t_ghost(s_exit)
            dt_ghost = t_exit_ghost - t_entry_ghost
            
            time_lost = dt_real - dt_ghost
            
            # 3. Generate Advice
            advice = []
            
            # Entry Phase
            if v_real_entry > v_ghost_entry + 5.0:
                advice.append("Overshot Entry (Brake Earlier)")
            elif v_real_entry < v_ghost_entry - 5.0:
                advice.append("Braked Too Early")
                
            # Apex Phase
            if v_real_apex < v_ghost_apex - 3.0:
                advice.append("Overslowed Apex (Trust Grip)")
            elif abs(v_real_apex - v_ghost_apex) < 1.0:
                advice.append("Good Apex Speed")
                
            # Exit Phase
            if v_real_exit < v_ghost_exit - 5.0:
                advice.append("Poor Exit (Get on Throttle Earlier)")

            # Format primary advice
            primary_advice = advice[0] if advice else "Good Corner"
            
            report.append({
                'Corner': f"T{c['id']}",
                'S_Start': int(s_entry),
                'Time_Loss': round(time_lost, 3),
                'Apex_Speed_Real': round(v_real_apex * 3.6, 1), # kph
                'Apex_Speed_Ghost': round(v_ghost_apex * 3.6, 1),
                'Advice': primary_advice
            })
            
        return pd.DataFrame(report)

    def get_input_comparison(self, ghost_data):
        """
        Reverse-engineers the Ghost Car's pedal inputs from its motion.
        Used because the OCP output might not explicitly save throttle/brake %.
        """
        # Physics Constants from Config
        mass = VP['m']
        Cd = VP['Cd']
        A = VP['A']
        rho = 1.225
        
        s = ghost_data['s']
        v = ghost_data['v']
        
        # 1. Calculate Acceleration (a = v * dv/ds)
        # Gradient handles non-uniform spacing better
        dv_ds = np.gradient(v, s)
        accel = v * dv_ds
        
        # 2. Calculate Required Force (F_net = m*a)
        F_net = mass * accel
        
        # 3. Add Aerodynamic Drag (F_drag = 0.5 * rho * Cd * A * v^2)
        # We must overcome drag to accelerate, so Engine Force must be higher
        F_drag = 0.5 * rho * Cd * A * v**2
        F_total = F_net + F_drag
        
        # 4. Normalize to -100% (Brake) to +100% (Throttle)
        # We assume some max capabilities
        MAX_FORCE_DRIVE = VP.get('power_max', 400000) / (np.mean(v) + 1.0) # Approx mean force
        MAX_FORCE_BRAKE = VP['m'] * 9.81 * 1.5 # Approx 1.5G braking
        
        pedal_trace = np.zeros_like(F_total)
        
        # Throttle (Positive Force)
        mask_drive = F_total > 0
        pedal_trace[mask_drive] = (F_total[mask_drive] / 6000.0) * 100 # Approx 6000N max thrust
        
        # Brake (Negative Force)
        mask_brake = F_total < 0
        pedal_trace[mask_brake] = (F_total[mask_brake] / 15000.0) * 100 # Approx 15000N max brake
        
        # Clip
        pedal_trace = np.clip(pedal_trace, -100, 100)
        
        return pedal_trace