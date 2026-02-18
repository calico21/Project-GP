import pandas as pd
import numpy as np
import os
import struct

class LogIngestion:
    """
    Robust telemetry importer.
    Handles:
    1. CSV files (Assetto Corsa, Motec CSV)
    2. ASC files (Vector CAN Logs) - *NEW*
    
    Standardizes output to SI Units:
    - Position: Lat/Lon [deg]
    - Speed: v [m/s]
    - Steering: delta [rad]
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
        # --- CAN CONFIGURATION (For ASC Files) ---
        # Based on your 'scan_logs_for_gps.py' and 'visualize_log.py'
        self.can_map = {
            # GPS: ID 0x119 (281)
            # Layout: [Lat (4 bytes)] [Lon (4 bytes)]
            0x119: {'name': 'gps', 'type': '<ii', 'factor': 1e-7, 'cols': ['lat', 'lon']},
            
            # Steering: ID 0x5
            # Layout: [Angle (2 bytes)] ...
            0x5:   {'name': 'steer', 'type': '<h', 'factor': 0.01, 'cols': ['delta_deg']},
            
            # Speed: ID 0x402 (Potential Speed)
            # Layout: [Speed (2 bytes)] ...
            # Assumption: 0.1 factor for kph (common standard)
            0x402: {'name': 'speed', 'type': '<h', 'factor': 0.1, 'cols': ['v_kph']},
        }

    def process(self):
        """
        Detects file type and processes it.
        Returns: DataFrame with ['lat', 'lon', 'v', 'delta', 'time']
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[LogIngestion] File not found: {self.file_path}")
            
        ext = os.path.splitext(self.file_path)[1].lower()
        
        print(f"[LogIngestion] Loading {self.file_path}...")
        
        if ext == '.asc':
            df = self._parse_asc_file()
        else:
            df = self._parse_csv_file()
            
        # --- POST-PROCESSING & VALIDATION ---
        
        # 1. Convert Units to SI
        if 'v_kph' in df.columns:
            df['v'] = df['v_kph'] / 3.6 # km/h -> m/s
        elif 'v' not in df.columns:
            # Fallback: Calc speed from GPS
            df = self._calc_speed_from_coords(df)

        if 'delta_deg' in df.columns:
            df['delta'] = np.deg2rad(df['delta_deg']) # deg -> rad
            
        # 2. Ensure Critical Columns Exist
        required = ['lat', 'lon', 'v', 'delta']
        for col in required:
            if col not in df.columns:
                print(f"[LogIngestion] Warning: Missing '{col}'. filling with 0.")
                df[col] = 0.0
                
        # 3. Clean NaN
        df = df.dropna(subset=['lat', 'lon'])
        df = df.reset_index(drop=True)
        
        print(f"[LogIngestion] Complete. {len(df)} samples loaded.")
        return df

    def _parse_asc_file(self):
        """
        Parses Vector ASC CAN logs line-by-line.
        Format: "  11.234 1  119  Rx   d 8 10 27 00 00 ..."
        """
        data_buffers = {0x119: [], 0x5: [], 0x402: []}
        
        print("[LogIngestion] Parsing ASC CAN data...")
        with open(self.file_path, 'r') as f:
            for line in f:
                # Basic filtering for data lines
                parts = line.strip().split()
                
                # Check structure: Timestamp, channel, ID, ... 'd', DLC, Data
                # Minimal check: at least 7 parts and 'd' indicates data payload
                if len(parts) > 6 and 'd' in parts:
                    try:
                        # Extract basic info
                        ts = float(parts[0])
                        
                        # ID handling: "119" or "119x" (extended)
                        can_id_str = parts[2].strip('x')
                        can_id = int(can_id_str, 16)
                        
                        if can_id in self.can_map:
                            # Locate data bytes (after 'd' and DLC)
                            d_idx = parts.index('d')
                            # parts[d_idx] = 'd'
                            # parts[d_idx+1] = DLC (e.g. '8')
                            # parts[d_idx+2:] = Data Bytes
                            hex_bytes = parts[d_idx+2:]
                            
                            # Convert hex list to byte string
                            # e.g. ['10', '27', ...] -> b'\x10\x27...'
                            raw_data = bytes([int(b, 16) for b in hex_bytes])
                            
                            cfg = self.can_map[can_id]
                            
                            # Unpack binary data
                            # Note: struct.unpack requires exact byte length
                            req_len = struct.calcsize(cfg['type'])
                            
                            if len(raw_data) >= req_len:
                                unpacked = struct.unpack(cfg['type'], raw_data[:req_len])
                                
                                # Apply factor and store
                                row = {'time': ts}
                                for i, col_name in enumerate(cfg['cols']):
                                    val = unpacked[i] * cfg['factor']
                                    row[col_name] = val
                                    
                                data_buffers[can_id].append(row)
                                
                    except (ValueError, struct.error, IndexError):
                        continue

        # Convert buffers to DataFrames
        df_gps = pd.DataFrame(data_buffers[0x119])
        df_steer = pd.DataFrame(data_buffers[0x5])
        df_speed = pd.DataFrame(data_buffers[0x402])
        
        # If no GPS found, crash early
        if df_gps.empty:
            raise ValueError("[LogIngestion] No GPS data (ID 0x119) found in .asc file!")
            
        # --- SYNCHRONIZATION ---
        # GPS is usually 10Hz, Steer 100Hz. We need to align them.
        # We use the GPS timestamps as the "Master Clock" (or a fixed 20Hz grid)
        
        # 1. Create a Master Time Grid (e.g. 20Hz / 50ms)
        t_start = df_gps['time'].min()
        t_end = df_gps['time'].max()
        t_grid = np.arange(t_start, t_end, 0.05)
        
        df_master = pd.DataFrame({'time': t_grid})
        
        # 2. Merge AsOf (Nearest neighbor interpolation)
        # GPS
        df_gps = df_gps.sort_values('time')
        df_master = pd.merge_asof(df_master, df_gps, on='time', direction='nearest', tolerance=0.2)
        
        # Steer
        if not df_steer.empty:
            df_steer = df_steer.sort_values('time')
            df_master = pd.merge_asof(df_master, df_steer, on='time', direction='nearest', tolerance=0.1)
        
        # Speed
        if not df_speed.empty:
            df_speed = df_speed.sort_values('time')
            df_master = pd.merge_asof(df_master, df_speed, on='time', direction='nearest', tolerance=0.1)
            
        return df_master

    def _parse_csv_file(self):
        """
        Legacy CSV parser (kept for compatibility)
        """
        try:
            df = pd.read_csv(self.file_path)
        except:
            df = pd.read_csv(self.file_path, sep=';')
            
        # Normalize columns (simple mapping)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Rename standard columns
        col_map = {
            'latitude': 'lat', 'pos_lat': 'lat',
            'longitude': 'lon', 'pos_lon': 'lon',
            'speed': 'v_kph', 'gps_speed': 'v_kph',
            'steer_angle': 'delta_deg', 'steer': 'delta_deg'
        }
        df = df.rename(columns=col_map)
        
        return df

    def _calc_speed_from_coords(self, df):
        """
        Calculates velocity if CAN Speed is missing.
        """
        print("[LogIngestion] Calculating speed from GPS coordinates...")
        R = 6371000
        dt = np.diff(df['time'], prepend=df['time'][0])
        dt[dt == 0] = 0.05 # Avoid div/0
        
        lat_rad = np.deg2rad(df['lat'])
        lon_rad = np.deg2rad(df['lon'])
        
        dlat = np.diff(lat_rad, prepend=lat_rad[0])
        dlon = np.diff(lon_rad, prepend=lon_rad[0])
        
        dx = dlon * np.cos(lat_rad) * R
        dy = dlat * R
        
        dist = np.sqrt(dx**2 + dy**2)
        df['v'] = dist / dt
        
        # Smooth noise
        df['v'] = df['v'].rolling(window=5, center=True, min_periods=1).mean()
        
        return df