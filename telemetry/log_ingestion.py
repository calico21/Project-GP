import pandas as pd
import numpy as np
import os

class LogIngestion:
    """
    Robust Data Ingestion for Formula Student Telemetry.
    Handles sparse GPS data (1Hz) vs fast sensors (100Hz).
    """
    
    # Standard names the Simulation expects
    STANDARD_SCHEMA = {
        'v_car': 'velocity',    # m/s
        'yaw_rate': 'yaw_rate', # rad/s
        'delta': 'steer',       # rad
        'acc_y': 'g_lat',       # m/s^2
        'acc_x': 'g_long',      # m/s^2
        'gps_lat': 'lat',
        'gps_lon': 'lon'
    }

    def __init__(self, dbc_config=None):
        self.dbc = dbc_config if dbc_config else {}
        self.df = None
        self.meta = {}

    def parse_asc(self, file_path, resample_freq='20ms'):
        """Parses Vector .asc file with Forward Fill strategy."""
        raw_data = []
        print(f"[Ingestor] Reading log file: {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6: continue
                    
                    try:
                        # 1. Parse Timestamp & ID
                        timestamp = float(parts[0])
                        can_id_str = parts[2].strip('x') # Handle '123x'
                        can_id = int(can_id_str, 16)
                        
                        if can_id in self.dbc:
                            # 2. Extract Data
                            if 'd' in parts:
                                d_idx = parts.index('d')
                                hex_bytes = parts[d_idx + 2:]
                                int_bytes = [int(b, 16) for b in hex_bytes]
                                
                                # 3. Decode
                                signals = self._decode_frame(can_id, int_bytes)
                                signals['Time'] = timestamp
                                raw_data.append(signals)
                                
                    except (ValueError, IndexError):
                        continue
        except FileNotFoundError:
            print(f"[Error] File not found: {file_path}")
            return pd.DataFrame()

        if not raw_data:
            print("[Warning] No matching CAN IDs found. Check DBC config.")
            return pd.DataFrame()

        # --- DATAFRAME CONSTRUCTION ---
        df = pd.DataFrame(raw_data)
        
        # Check raw capture before resampling
        if 'Latitude' in df.columns:
            print(f"[Ingestor] Raw GPS points found: {df['Latitude'].notna().sum()}")
        
        # Deduplicate and Sort
        df = df.sort_values('Time').drop_duplicates('Time', keep='last')
        df['Time'] = pd.to_timedelta(df['Time'], unit='s')
        df = df.set_index('Time')
        
        # --- ROBUST RESAMPLING ---
        # 1. Resample to grid (creates NaNs)
        df_resampled = df.resample(resample_freq).mean()
        
        # 2. Forward Fill (Propagate last known value) - Vital for GPS!
        df_resampled = df_resampled.ffill()
        
        # 3. Backward Fill (Fill start gaps)
        df_resampled = df_resampled.bfill()
        
        self.df = df_resampled
        
        # Create float time column
        self.df['time'] = self.df.index.total_seconds()
        self.df = self.df.reset_index(drop=True)
        
        print(f"[Ingestor] Final Grid: {len(self.df)} rows @ {resample_freq}")
        return self.df

    def _decode_frame(self, can_id, data_bytes):
        signals = {}
        msg_def = self.dbc[can_id]
        
        raw_payload = 0
        for i, byte in enumerate(data_bytes):
            if i < 8: raw_payload |= (byte << (i * 8))
            
        for sig in msg_def['signals']:
            mask = (1 << sig['length']) - 1
            raw_val = (raw_payload >> sig['start_bit']) & mask
            
            if sig.get('signed', False):
                if raw_val >= (1 << (sig['length'] - 1)):
                    raw_val -= (1 << sig['length'])
            
            phys_val = (raw_val * sig['factor']) + sig['offset']
            signals[sig['name']] = phys_val
            
        return signals

    def apply_schema(self, channel_map):
        if self.df is None: return
        self.df = self.df.rename(columns=channel_map)
        print("[Ingestor] Schema applied.")

    def process_units(self):
        if self.df is None: return
        
        # Auto-detect units based on column names
        if 'steer' in self.df.columns:
            # Assume degrees if max > 6.0 (2*PI is approx 6)
            if self.df['steer'].abs().max() > 10.0:
                 self.df['steer'] = np.deg2rad(self.df['steer'])
                 
        if 'velocity' in self.df.columns:
            # Assume kph if max > 40
            if self.df['velocity'].max() > 40.0:
                self.df['velocity'] = self.df['velocity'] / 3.6

    def project_gps_to_cartesian(self):
        required = ['lat', 'lon']
        if not all(col in self.df.columns for col in required):
            print(f"[Warning] Missing GPS columns: {[c for c in required if c not in self.df.columns]}")
            return

        # Check for NaNs
        if self.df['lat'].isna().all():
            print("[Error] GPS data exists but is all NaN. Check parsing.")
            return

        # Origin
        lat0 = self.df['lat'].iloc[0]
        lon0 = self.df['lon'].iloc[0]
        
        R = 6378137.0 
        lat_rad = np.deg2rad(self.df['lat'].values)
        lon_rad = np.deg2rad(self.df['lon'].values)
        lat0_rad = np.deg2rad(lat0)
        lon0_rad = np.deg2rad(lon0)
        
        dlon = lon_rad - lon0_rad
        dlat = lat_rad - lat0_rad
        
        x = R * dlon * np.cos(lat0_rad)
        y = R * dlat
        
        self.df['x'] = x
        self.df['y'] = y
        print(f"[Ingestor] GPS Projected. Track bounds: {np.min(x):.1f}m to {np.max(x):.1f}m")

    def export(self):
        return {c: self.df[c].values for c in self.df.columns}