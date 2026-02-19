import os
import struct
import numpy as np
import pandas as pd
import sys

# --- SOTA FGO IMPORT ---
try:
    import gtsam
    from telemetry.filtering import FactorGraphSmoother
except ImportError:
    print("[Error] gtsam or FactorGraphSmoother missing. Ensure telemetry/filtering.py is updated.")
    sys.exit(1)

class LogIngestion:
    """
    State-of-the-Art Hybrid Ingestor:
    - Parser: Decodes raw CAN Vector ASC log files (struct/hex).
    - Fusion: Asynchronously feeds an iSAM2 Factor Graph Smoother (FGO) for trajectory reconstruction.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
        # --- CAN LOGGING CONFIGURATION ---
        self.can_map = {
            # GPS: ID 0x119 (Lat/Lon as integers)
            0x119: {'name': 'gps', 'type': '<ii', 'factor': 1e-7, 'cols': ['lat', 'lon']},
            # Steering: ID 0x5
            0x5:   {'name': 'steer', 'type': '<h', 'factor': 0.01, 'cols': ['delta_deg']},
            # Speed: ID 0x402
            0x402: {'name': 'speed', 'type': '<h', 'factor': 0.1, 'cols': ['v_kph']},
            # IMU: ID 0x120 (Ax, Ay, YawRate)
            0x120: {'name': 'imu', 'type': '<hhh', 'factor': 0.01, 'cols': ['ax', 'ay', 'yaw_rate']}
        }

    def process(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[LogIngestion] File not found: {self.file_path}")
            
        print(f"[LogIngestion] Loading {self.file_path}...")
        ext = os.path.splitext(self.file_path)[1].lower()
        
        if ext == '.asc':
            raw_data = self._parse_asc_file()
        else:
            raw_data = self._parse_csv_file()
            
        print("[LogIngestion] Feeding Asynchronous Event Stream to Factor Graph Smoother...")
        df_final = self._run_sensor_fusion(raw_data)
        
        print(f"[LogIngestion] Complete. {len(df_final)} trajectory poses optimized.")
        return df_final

    def _parse_asc_file(self):
        """Parses Raw Vector ASC hexadecimal CAN dumps."""
        data_buffers = {k: [] for k in self.can_map.keys()}
        
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 6 and 'd' in parts:
                        try:
                            ts = float(parts[0])
                            can_id_str = parts[2].strip('x')
                            can_id = int(can_id_str, 16)
                            
                            if can_id in self.can_map:
                                d_idx = parts.index('d')
                                hex_bytes = parts[d_idx+2:]
                                
                                raw_data = bytes([int(b, 16) for b in hex_bytes])
                                cfg = self.can_map[can_id]
                                req_len = struct.calcsize(cfg['type'])
                                
                                if len(raw_data) >= req_len:
                                    unpacked = struct.unpack(cfg['type'], raw_data[:req_len])
                                    row = {'time': ts}
                                    for i, col_name in enumerate(cfg['cols']):
                                        row[col_name] = unpacked[i] * cfg['factor']
                                    data_buffers[can_id].append(row)
                        except Exception:
                            continue
        except Exception as e:
            print(f"[LogIngestion] Error reading file: {e}")

        return {self.can_map[k]['name']: pd.DataFrame(data_buffers[k]) for k in self.can_map}

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
            'gps': df[['time', 'lat', 'lon']].dropna(),
            'steer': df[['time', 'delta_deg']].dropna() if 'delta_deg' in df else pd.DataFrame(),
            'speed': df[['time', 'v_kph']].dropna() if 'v_kph' in df else pd.DataFrame(),
            'imu': df[['time', 'ax', 'ay', 'yaw_rate']].dropna() if 'ax' in df else pd.DataFrame()
        }

    def _run_sensor_fusion(self, raw_data):
        """Asynchronous pipeline orchestrating the iSAM2 Factor Graph."""
        gps_stream = raw_data.get('gps', pd.DataFrame())
        imu_stream = raw_data.get('imu', pd.DataFrame())
        speed_stream = raw_data.get('speed', pd.DataFrame())

        if gps_stream.empty or imu_stream.empty:
            raise ValueError("[LogIngestion] Fatal: Both GPS and IMU streams are required for FGO.")

        # 1. Coordinate Transformation (Lat/Lon -> Local Cartesian X/Y)
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

        # 2. Unified Chronological Event Timeline
        gps_stream['event_type'] = 'gps'
        imu_stream['event_type'] = 'imu'
        speed_stream['event_type'] = 'speed'

        events = pd.concat([
            gps_stream[['time', 'event_type', 'x', 'y']],
            imu_stream[['time', 'event_type', 'ax', 'ay', 'yaw_rate']],
            speed_stream[['time', 'event_type', 'v_kph']]
        ]).sort_values('time').reset_index(drop=True)

        # 3. iSAM2 Initialization
        smoother = FactorGraphSmoother()
        
        # Set robust prior (Initial heading derived from first two GPS nodes)
        initial_yaw = 0.0
        if len(gps_stream) > 1:
            initial_yaw = np.arctan2(gps_stream.iloc[1]['y'] - gps_stream.iloc[0]['y'], 
                                     gps_stream.iloc[1]['x'] - gps_stream.iloc[0]['x'])
        
        initial_pose = gtsam.Pose3(gtsam.Rot3.Ypr(initial_yaw, 0, 0), gtsam.Point3(0, 0, 0))
        
        initial_v = speed_stream.iloc[0]['v_kph'] / 3.6 if not speed_stream.empty else 0.0
        initial_vel = np.array([initial_v * np.cos(initial_yaw), initial_v * np.sin(initial_yaw), 0.0])
        initial_bias = gtsam.imuBias.ConstantBias()
        
        smoother.initialize(initial_pose, initial_vel, initial_bias)

        # 4. Asynchronous Event Loop
        last_imu_time = events['time'].iloc[0]
        current_vel_vector = initial_vel
        node_times = []

        for _, row in events.iterrows():
            t = row['time']
            
            if row['event_type'] == 'imu':
                dt = max(t - last_imu_time, 1e-6) # Guard against duplicate timestamps
                accel = np.array([row['ax'], row['ay'], 9.81]) # Z axis gravity compensated
                gyro = np.array([0.0, 0.0, row['yaw_rate']])
                smoother.add_imu_measurement(accel, gyro, dt)
                last_imu_time = t
                
            elif row['event_type'] == 'speed':
                v = row['v_kph'] / 3.6
                # Transform wheel speed into navigation frame vector based on estimated heading
                current_vel_vector = np.array([v * np.cos(initial_yaw), v * np.sin(initial_yaw), 0.0])
                
            elif row['event_type'] == 'gps':
                gps_point = np.array([row['x'], row['y'], 0.0])
                smoother.add_gps_node(gps_point, can_velocity=current_vel_vector)
                node_times.append(t)

        # 5. Extraction & Downstream Formatting
        df_smoothed = smoother.extract_trajectory()
        
        # Attach the exact timestamps to the optimized poses and calculate absolute velocity
        df_smoothed['time'] = node_times
        df_smoothed['v'] = np.sqrt(df_smoothed['vx']**2 + df_smoothed['vy']**2)
        
        # Required format for `TrackGenerator`
        return df_smoothed[['time', 'x', 'y', 'v', 'yaw']]