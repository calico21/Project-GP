import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class DriverCoach:
    """
    Advanced Driver Analysis Engine.
    
    Responsibilities:
    1. Segment Track into 'Corners' and 'Straights' based on curvature.
    2. Compare Inputs (Force -> Pedal %).
    3. Analyze Corner Phases (Braking Point, Apex Speed, Exit Speed).
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
        
        # Thresholds
        k_threshold = 0.05 # 1/m (approx 20m radius) - adjust for your track
        is_corner = k > k_threshold
        
        # Find transitions
        # This is a simplified logic. Robust segmentation merges close corners.
        corners = []
        in_corner = False
        start_idx = 0
        
        for i, val in enumerate(is_corner):
            if val and not in_corner:
                in_corner = True
                start_idx = i
            elif not val and in_corner:
                in_corner = False
                end_idx = i
                
                # Filter tiny blips (noise)
                if (s[end_idx] - s[start_idx]) > 5.0:
                    # Find Apex (Max Curvature)
                    segment_k = k[start_idx:end_idx]
                    apex_local_idx = np.argmax(segment_k)
                    apex_idx = start_idx + apex_local_idx
                    
                    corners.append({
                        'id': len(corners) + 1,
                        'start_s': s[start_idx],
                        'end_s': s[end_idx],
                        'apex_s': s[apex_idx],
                        'type': 'Corner'
                    })
        return corners

    def analyze_lap(self, real_data, ghost_data):
        """
        Generates the 'Coaching Report'.
        
        Args:
            real_data: Synced dictionary (s, velocity, etc.)
            ghost_data: Solver dictionary (s, v, delta, etc.)
        """
        report = []
        
        # Helper to get value at specific 's'
        def get_val_at(data, key, s_query):
            idx = np.searchsorted(data['s'], s_query)
            if idx >= len(data[key]): idx = len(data[key]) - 1
            return data[key][idx]
            
        for c in self.corners:
            # 1. Apex Speed Comparison (Mid-Corner Limit)
            v_real_apex = get_val_at(real_data, 'velocity', c['apex_s'])
            v_ghost_apex = get_val_at(ghost_data, 'v', c['apex_s'])
            
            # 2. Entry Speed (Speed at corner start)
            v_real_entry = get_val_at(real_data, 'velocity', c['start_s'])
            v_ghost_entry = get_val_at(ghost_data, 'v', c['start_s'])
            
            # 3. Braking Point Analysis (Look 50m before corner)
            # Find where Ghost switches from Accel to Brake
            # Simplified: Find max speed location before corner
            search_start = max(0, c['start_s'] - 50)
            
            # 4. Score this corner
            time_loss = 0 # Placeholder for integral calculation
            
            report.append({
                'Corner': f"T{c['id']}",
                'Apex_Speed_Real': v_real_apex,
                'Apex_Speed_Ghost': v_ghost_apex,
                'Delta_Apex': v_real_apex - v_ghost_apex,
                'Entry_Delta': v_real_entry - v_ghost_entry,
                'Advice': self._generate_advice(v_real_entry, v_real_apex, v_ghost_apex)
            })
            
        return pd.DataFrame(report)
        
    def _generate_advice(self, v_entry, v_apex, v_optimal):
        """Generates text feedback."""
        if v_apex < (v_optimal - 5.0):
            return "Overslowed Apex (Check Grip)"
        elif v_entry > (v_apex + 10.0) and v_apex < v_optimal:
            return "Overshot Entry (Brake Earlier)"
        elif abs(v_apex - v_optimal) < 1.0:
            return "Good Cornering"
        else:
            return "Review Line"

    def get_input_comparison(self, real_data, ghost_data):
        """
        Maps Ghost Forces to Pedal % for plotting.
        """
        # Ghost 'u' is Force [N]. We normalize to -1 (Brake) to +1 (Throttle)
        # Assuming Max Drive Force ~ 1000N, Max Brake ~ 2500N (from Solver constraints)
        MAX_DRIVE = 1000.0
        MAX_BRAKE = 2500.0
        
        # Calculate Ghost Pedals
        # OCP Fx is usually in ghost_data (check keys, sometimes it's U[1])
        # We need to re-extract or approximate from V_dot if inputs aren't saved directly
        # For this snippet, assuming we saved 'Fx' in ghost_data from solver
        
        # If Fx not in ghost data, calculate it: F = ma + drag
        # This is robust because it matches physics
        mass = 250
        v_ghost = ghost_data['v']
        dv = np.gradient(v_ghost, ghost_data['s']) * v_ghost # a = v * dv/ds
        drag = 0.5 * 1.225 * 1.1 * 1.2 * v_ghost**2
        fx_ghost = mass * dv + drag
        
        ghost_pedal = np.zeros_like(fx_ghost)
        ghost_pedal[fx_ghost > 0] = fx_ghost[fx_ghost > 0] / MAX_DRIVE
        ghost_pedal[fx_ghost < 0] = fx_ghost[fx_ghost < 0] / MAX_BRAKE
        
        # Real Pedals (assuming normalized 0-1)
        # Combine Throttle (0 to 1) and Brake (0 to 1) into single -1 to 1 trace
        real_pedal = real_data['throttle'] - real_data['brake']
        
        return {
            's': ghost_data['s'],
            'ghost_pedal': np.clip(ghost_pedal, -1, 1),
            'real_pedal': real_pedal
        }