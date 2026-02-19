import pandas as pd
import numpy as np
import os
import struct
from scipy.spatial.transform import Rotation as R_scipy

class ErrorStateEKF:
    """
    State-of-the-Art Error-State Extended Kalman Filter (ES-EKF).
    Tracks Nominal State non-linearly using Quaternions.
    Only linearizes the naturally small Error State, handling violent slip dynamics.
    """
    def __init__(self, dt, R_gps=2.0, Q_accel=0.1, Q_gyro=0.01):
        self.dt = dt
        
        # --- NOMINAL STATE ---
        # Position (x, y, z), Velocity (vx, vy, vz), Quaternion (qw, qx, qy, qz)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # Scalar-first quaternion
        
        # --- ERROR STATE COVARIANCE (9x9) ---
        # Error State: [dp_x, dp_y, dp_z, dv_x, dv_y, dv_z, dth_x, dth_y, dth_z]
        self.P = np.eye(9) * 1.0
        
        # Process Noise
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= Q_accel * dt**2
        self.Q[3:6, 3:6] *= Q_gyro * dt**2
        
        # Measurement Noise (GPS X, Y)
        self.R_gps = np.eye(2) * R_gps
        
        # Gravity Vector
        self.g = np.array([0, 0, -9.81])

    def _skew_symmetric(self, v):
        """Generates a skew-symmetric matrix for cross products."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def predict(self, accel_m, gyro_m):
        """
        1. Predicts Nominal State non-linearly.
        2. Propagates Error State Covariance linearly.
        """
        # --- 1. NOMINAL STATE PREDICTION ---
        rot = R_scipy.from_quat([self.q[1], self.q[2], self.q[3], self.q[0]]) # scipy uses x,y,z,w
        R_mat = rot.as_matrix()
        
        # Remove gravity from acceleration
        accel_world = R_mat @ accel_m + self.g
        
        # Integrate Position and Velocity
        self.p = self.p + self.v * self.dt + 0.5 * accel_world * self.dt**2
        self.v = self.v + accel_world * self.dt
        
        # Integrate Quaternion (Rotation)
        d_theta = gyro_m * self.dt
        d_q = R_scipy.from_rotvec(d_theta)
        self.q = (rot * d_q).as_quat()
        self.q = np.array([self.q[3], self.q[0], self.q[1], self.q[2]]) # back to w,x,y,z
        
        # --- 2. ERROR STATE COVARIANCE PROPAGATION ---
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * self.dt
        # The cross-product term for rotation error injection
        F[3:6, 6:9] = -R_mat @ self._skew_symmetric(accel_m) * self.dt
        
        # Jacobian for noise projection
        L = np.zeros((9, 6))
        L[3:6, 0:3] = np.eye(3)
        L[6:9, 3:6] = np.eye(3)
        
        self.P = F @ self.P @ F.T + L @ self.Q @ L.T

    def update_gps(self, gps_x, gps_y):
        """
        Corrects the Nominal State using GPS measurements.
        """
        # H maps the 9D Error State to the 2D GPS observation (only px, py)
        H = np.zeros((2, 9))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        
        # Residual (Measurement - Nominal State)
        z = np.array([gps_x, gps_y])
        h_x = np.array([self.p[0], self.p[1]])
        y = z - h_x
        
        # Kalman Gain
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Compute Error State
        delta_x = K @ y
        dp = delta_x[0:3]
        dv = delta_x[3:6]
        dth = delta_x[6:9]
        
        # --- 3. ERROR INJECTION (Update Nominal State) ---
        self.p += dp
        self.v += dv
        
        d_q = R_scipy.from_rotvec(dth)
        rot = R_scipy.from_quat([self.q[1], self.q[2], self.q[3], self.q[0]])
        self.q = (rot * d_q).as_quat()
        self.q = np.array([self.q[3], self.q[0], self.q[1], self.q[2]])
        
        # --- 4. ERROR RESET ---
        # The error state is pushed into the nominal state, so we reset error to zero.
        # We must also update the covariance matrix for the reset operation.
        I = np.eye(9)
        self.P = (I - K @ H) @ self.P


class LogIngestion:
    """
    Hybrid Ingestor:
    - Parser: Uses user-provided Vector ASC logic (struct/hex).
    - Fusion: Uses Error-State EKF with IMU support.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
        # --- USER CONFIGURATION (Preserved + IMU Added) ---
        self.can_map = {
            # GPS: ID 0x119 (Lat/Lon as integers)
            0x119: {'name': 'gps', 'type': '<ii', 'factor': 1e-7, 'cols': ['lat', 'lon']},
            # Steering: ID 0x5
            0x5:   {'name': 'steer', 'type': '<h', 'factor': 0.01, 'cols': ['delta_deg']},
            # Speed: ID 0x402
            0x402: {'name': 'speed', 'type': '<h', 'factor': 0.1, 'cols': ['v_kph']},
            # IMU (Added for ES-EKF Support): ID 0x120 (Ax, Ay, YawRate)
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
            
        print("[LogIngestion] Starting Sensor Fusion (Error-State EKF)...")
        df_final = self._run_sensor_fusion(raw_data)
        
        print(f"[LogIngestion] Complete. {len(df_final)} samples optimized.")
        return df_final

    def _parse_asc_file(self):
        data_buffers = {0x119: [], 0x5: [], 0x402: [], 0x120: []}
        
        print("[LogIngestion] Parsing ASC CAN data...")
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
                                        val = unpacked[i] * cfg['factor']
                                        row[col_name] = val
                                    
                                    data_buffers[can_id].append(row)
                        except Exception:
                            continue
        except Exception as e:
            print(f"[LogIngestion] Error reading file: {e}")

        return {
            'gps': pd.DataFrame(data_buffers[0x119]),
            'steer': pd.DataFrame(data_buffers[0x5]),
            'speed': pd.DataFrame(data_buffers[0x402]),
            'imu': pd.DataFrame(data_buffers[0x120])
        }

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
        gps_stream = raw_data.get('gps', pd.DataFrame())
        steer_stream = raw_data.get('steer', pd.DataFrame())
        speed_stream = raw_data.get('speed', pd.DataFrame())
        imu_stream = raw_data.get('imu', pd.DataFrame())

        if gps_stream.empty:
            raise ValueError("[LogIngestion] No GPS data found in log.")

        # 1. Convert GPS Lat/Lon to Meters (X/Y)
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

        # 2. Master Time Grid (50Hz = 0.02s)
        dt = 0.02
        t_start = gps_stream['time'].min()
        t_end = gps_stream['time'].max()
        t_grid = np.arange(t_start, t_end, dt)
        
        # 3. Interpolators
        def get_interp(df, col):
            if df.empty: return lambda t: 0.0
            return lambda t: np.interp(t, df['time'], df[col])
            
        get_v = get_interp(speed_stream, 'v_kph')
        get_steer = get_interp(steer_stream, 'delta_deg')
        get_ax = get_interp(imu_stream, 'ax')
        get_ay = get_interp(imu_stream, 'ay')
        get_yaw_rate = get_interp(imu_stream, 'yaw_rate')
        
        gps_times = gps_stream['time'].values
        gps_x = gps_stream['x'].values
        gps_y = gps_stream['y'].values
        
        # 4. Run ES-EKF
        ekf = ErrorStateEKF(dt=dt)
        
        # Initialize Yaw if moving
        if len(gps_x) > 1:
            initial_yaw = np.arctan2(gps_y[1] - gps_y[0], gps_x[1] - gps_x[0])
            init_q = R_scipy.from_euler('z', initial_yaw).as_quat()
            ekf.q = np.array([init_q[3], init_q[0], init_q[1], init_q[2]])
            
        results = []
        gps_idx = 0
        
        for t in t_grid:
            # Construct IMU inputs
            # Fallback to Kinematic Bicycle approximation if true IMU data is completely missing
            v_ms = get_v(t) / 3.6
            steer_rad = np.deg2rad(get_steer(t))
            
            if imu_stream.empty:
                beta = np.arctan(0.5 * np.tan(steer_rad))
                yaw_rate_val = (v_ms / 1.6) * np.sin(steer_rad) 
                ax_val = 0.0 # Ignore longitudinal accel for fallback
                ay_val = v_ms * yaw_rate_val 
            else:
                yaw_rate_val = get_yaw_rate(t)
                ax_val = get_ax(t)
                ay_val = get_ay(t)
                
            accel_m = np.array([ax_val, ay_val, 9.81]) # Z includes gravity compensation
            gyro_m = np.array([0.0, 0.0, yaw_rate_val])
            
            # Predict
            ekf.predict(accel_m, gyro_m)
            
            # Update (if GPS matched)
            while gps_idx < len(gps_times) and gps_times[gps_idx] < (t - dt/2):
                gps_idx += 1
            
            if gps_idx < len(gps_times) and abs(gps_times[gps_idx] - t) < dt/2:
                ekf.update_gps(gps_x[gps_idx], gps_y[gps_idx])
            
            # Extract current yaw from quaternion
            rot = R_scipy.from_quat([ekf.q[1], ekf.q[2], ekf.q[3], ekf.q[0]])
            current_yaw = rot.as_euler('xyz')[2]
            
            results.append({
                'time': t,
                'x': ekf.p[0],
                'y': ekf.p[1],
                'yaw': current_yaw,
                'v': v_ms, 
                'steer': steer_rad
            })
            
        return pd.DataFrame(results)