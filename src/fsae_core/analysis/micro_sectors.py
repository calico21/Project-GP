import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter

class TelemetryAnalyzer:
    """
    Advanced Lap Analysis: Aligns driver data to reference track and 
    segments performance into 'Micro-Sectors'.
    """

    def __init__(self, track_centerline):
        """
        track_centerline: shape (N, 2) array of [x, y] coordinates
        """
        self.track_xy = track_centerline
        # Create a KD-Tree for fast nearest-neighbor lookup
        self.tree = cKDTree(self.track_xy)
        
        # Calculate track curvature (kappa)
        # kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
        dx = np.gradient(self.track_xy[:, 0])
        dy = np.gradient(self.track_xy[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        self.curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
        
        # Smooth curvature to remove noise for segmentation
        self.curvature = savgol_filter(self.curvature, window_length=11, polyorder=3)

    def align_data_to_track(self, driver_log):
        """
        Projects driver GPS/Path onto the centerline distance (s).
        
        Args:
            driver_log: DataFrame with 'x', 'y', 'speed', 'throttle', etc.
        """
        # 1. Find nearest point on centerline for every driver sample
        distances, indices = self.tree.query(driver_log[['x', 'y']].values)
        
        # 2. Assign 'Track Station' (s) to every log point
        # We assume the track points are spaced evenly or we calculate cumulative dist
        # simplified: just use index scaling
        driver_log['track_index'] = indices
        driver_log['lateral_deviation'] = distances
        
        return driver_log

    def segment_track(self, threshold=0.05):
        """
        Creates Micro-Sectors based on curvature.
        Returns a list of segments: {'type': 'straight'/'corner', 'start_idx': i, 'end_idx': j}
        """
        segments = []
        in_corner = False
        start_idx = 0
        
        for i, k in enumerate(np.abs(self.curvature)):
            is_turn = k > threshold
            
            if is_turn and not in_corner:
                # Transition: Straight -> Corner
                segments.append({'type': 'straight', 'start': start_idx, 'end': i-1})
                start_idx = i
                in_corner = True
                
            elif not is_turn and in_corner:
                # Transition: Corner -> Straight
                segments.append({'type': 'corner', 'start': start_idx, 'end': i-1})
                start_idx = i
                in_corner = False
        
        # Add final segment
        segments.append({'type': 'corner' if in_corner else 'straight', 'start': start_idx, 'end': len(self.curvature)-1})
        
        return pd.DataFrame(segments)

    def compare_laps(self, driver_df, ghost_df, segments):
        """
        Generates the 'Gain/Loss' report per corner.
        """
        report = []
        
        for _, row in segments.iterrows():
            if row['type'] == 'straight':
                continue # Skip straights for now
            
            # Extract data for this specific corner
            # We filter by 'track_index' which maps to centerline points
            d_data = driver_df[(driver_df['track_index'] >= row['start']) & 
                               (driver_df['track_index'] <= row['end'])]
            
            g_data = ghost_df[(ghost_df['track_index'] >= row['start']) & 
                              (ghost_df['track_index'] <= row['end'])]
            
            if d_data.empty or g_data.empty:
                continue
                
            # Key Metrics
            d_min_speed = d_data['speed'].min()
            g_min_speed = g_data['speed'].min()
            
            d_time = d_data['time'].max() - d_data['time'].min()
            g_time = g_data['time'].max() - g_data['time'].min()
            
            report.append({
                'Segment': f"Corner {row['start']}",
                'Driver Min Speed': d_min_speed,
                'Ghost Min Speed': g_min_speed,
                'Delta Speed': d_min_speed - g_min_speed, # Negative means driver is slower
                'Time Loss': d_time - g_time              # Positive means driver lost time
            })
            
        return pd.DataFrame(report)

# Example Usage
if __name__ == "__main__":
    # Create a dummy circle track
    theta = np.linspace(0, 2*np.pi, 1000)
    track = np.column_stack([100*np.cos(theta), 100*np.sin(theta)])
    
    analyzer = TelemetryAnalyzer(track)
    segs = analyzer.segment_track(threshold=0.01)
    print("Detected Segments:")
    print(segs)