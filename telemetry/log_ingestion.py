import os
import sys
import numpy as np
import pandas as pd
import cantools

class LogIngestion:
    """
    Continuous-Time Data Ingestor.
    Dynamically decodes raw CAN Vector ASC log files via a .dbc database.
    Outputs mathematically pristine, asynchronous event timelines formatted 
    strictly for Continuous-Time SE(3) Gaussian Process Optimization.
    """
    def __init__(self, file_path, dbc_path="TER.dbc"):
        self.file_path = file_path
        self.dbc_path = dbc_path
        
        # Load the DBC database dynamically
        if not os.path.exists(self.dbc_path):
            print(f"[LogIngestion] WARNING: DBC file not found at {self.dbc_path}. Will fallback to CSV if necessary.")
            self.db = None
        else:
            self.db = cantools.database.load_file(self.dbc_path)
            print(f"[LogIngestion] Loaded DBC database from {self.dbc_path}")

    def process(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[LogIngestion] File not found: {self.file_path}")
            
        print(f"[LogIngestion] Parsing raw telemetry from {self.file_path}...")
        ext = os.path.splitext(self.file_path)[1].lower()
        
        if ext == '.asc' and self.db is not None:
            raw_data = self._parse_asc_file()
        else:
            raw_data = self._parse_csv_file()
            
        print("[LogIngestion] Projecting Geodetic coordinates to Local Cartesian Manifold...")
        df_final = self._format_for_ct_gp(raw_data)
        
        print(f"[LogIngestion] Ingestion Complete. Extracted {len(df_final)} asynchronous spatial nodes.")
        return df_final

    def _parse_asc_file(self):
        """
        High-performance parser for Raw Vector ASC hexadecimal CAN dumps.
        Routes signals into asynchronous event buffers.
        """
        data_buffers = {'gps': [], 'steer': [], 'speed': [], 'imu': []}
        
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    
                    if len(parts) > 6 and 'd' in parts:
                        try:
                            ts = float(parts[0])
                            can_id = int(parts[2].strip('x'), 16)
                            
                            try:
                                msg = self.db.get_message_by_frame_id(can_id)
                            except KeyError:
                                continue # Message ID not defined in TER.dbc
                                
                            d_idx = parts.index('d')
                            hex_bytes = parts[d_idx+2 : d_idx+2+msg.length]
                            raw_data = bytes([int(b, 16) for b in hex_bytes])
                            
                            try:
                                decoded = msg.decode(raw_data)
                            except ValueError:
                                continue 
                            
                            # Dynamic Routing based on message names
                            # Using 'in' allows flexibility if the exact TER.dbc names shift slightly
                            msg_name = msg.name.upper()
                            
                            if 'GPS' in msg_name: 
                                # Extract matching keys dynamically
                                lat_key = next((k for k in decoded.keys() if 'LAT' in k.upper()), None)
                                lon_key = next((k for k in decoded.keys() if 'LON' in k.upper()), None)
                                if lat_key and lon_key:
                                    data_buffers['gps'].append({
                                        'time': ts, 
                                        'lat': decoded[lat_key], 
                                        'lon': decoded[lon_key]
                                    })
                                    
                            elif 'IMU' in msg_name or 'ACCEL' in msg_name:
                                ax_key = next((k for k in decoded.keys() if 'X' in k.upper() and 'ACC' in k.upper()), None)
                                ay_key = next((k for k in decoded.keys() if 'Y' in k.upper() and 'ACC' in k.upper()), None)
                                yaw_key = next((k for k in decoded.keys() if 'YAW' in k.upper()), None)
                                if ax_key and ay_key and yaw_key:
                                    data_buffers['imu'].append({
                                        'time': ts, 
                                        'ax': decoded[ax_key], 
                                        'ay': decoded[ay_key], 
                                        'yaw_rate': decoded[yaw_key]
                                    })
                                    
                            elif 'STEER' in msg_name:
                                steer_key = next((k for k in decoded.keys() if 'ANG' in k.upper()), None)
                                if steer_key:
                                    data_buffers['steer'].append({'time': ts, 'delta_deg': decoded[steer_key]})
                                    
                            elif 'SPEED' in msg_name or 'WHEEL' in msg_name:
                                spd_key = next((k for k in decoded.keys() if 'SPD' in k.upper() or 'VEL' in k.upper()), None)
                                if spd_key:
                                    data_buffers['speed'].append({'time': ts, 'v_kph': decoded[spd_key]})
                                
                        except Exception:
                            continue
        except Exception as e:
            print(f"[LogIngestion] Critical Error reading ASC file: {e}")

        return {k: pd.DataFrame(v) for k, v in data_buffers.items() if len(v) > 0}

    def _parse_csv_file(self):
        """Legacy CSV parser."""
        try:
            df = pd.read_csv(self.file_path)
        except:
            df = pd.read_csv(self.file_path, sep=';')
            
        df.columns = [c.lower().strip() for c in df.columns]
        col_map = {
            'latitude': 'lat', 'pos_lat': 'lat',
            'longitude': 'lon', 'pos_lon': 'lon',
            'speed': 'v_kph', 'gps_speed': 'v_kph',
            'steer_angle': 'delta_deg', 'steer': 'delta_deg',
            'accel_x': 'ax', 'accel_y': 'ay', 'yaw_rate': 'yaw_rate'
        }
        df = df.rename(columns=col_map)
        
        return {
            'gps': df[['time', 'lat', 'lon']].dropna() if 'lat' in df else pd.DataFrame(),
            'steer': df[['time', 'delta_deg']].dropna() if 'delta_deg' in df else pd.DataFrame(),
            'speed': df[['time', 'v_kph']].dropna() if 'v_kph' in df else pd.DataFrame(),
            'imu': df[['time', 'ax', 'ay', 'yaw_rate']].dropna() if 'ax' in df else pd.DataFrame()
        }

    def _format_for_ct_gp(self, raw_data):
        """
        Constructs the target array for the JAX Continuous-Time Trajectory Estimator.
        """
        gps_stream = raw_data.get('gps', pd.DataFrame())

        if gps_stream.empty:
            raise ValueError("[LogIngestion] Fatal: GPS stream is required to generate the SE(3) spatial manifold.")

        # 1. Coordinate Transformation (Geodetic Lat/Lon to Local Cartesian Euclidean Space)
        lat0, lon0 = gps_stream.iloc[0]['lat'], gps_stream.iloc[0]['lon']
        R_earth = 6378137.0
        
        def latlon_to_xy(lat, lon):
            dlat = np.deg2rad(lat - lat0)
            dlon = np.deg2rad(lon - lon0)
            x = R_earth * dlon * np.cos(np.deg2rad(lat0))
            y = R_earth * dlat
            return x, y

        coords = gps_stream.apply(lambda row: latlon_to_xy(row['lat'], row['lon']), axis=1)
        gps_stream['x'] = [c[0] for c in coords]
        gps_stream['y'] = [c[1] for c in coords]
        
        # We assume a planar track for the basic manifold (Z = 0)
        gps_stream['z'] = 0.0 
        
        # Sort chronologically to guarantee correct temporal flow for the Magnus Expansion
        gps_stream = gps_stream.sort_values('time').reset_index(drop=True)

        # Output the exact format expected by the `main.py` JAX orchestrator
        return gps_stream[['time', 'x', 'y', 'z']]