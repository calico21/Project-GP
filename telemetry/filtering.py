import numpy as np
import pandas as pd
import sys

# --- GTSAM IMPORTS (The SOTA Upgrade) ---
try:
    import gtsam
    from gtsam.symbol_shorthand import X, V, B  # X: Pose, V: Velocity, B: IMU Bias
except ImportError:
    print("[Error] gtsam is not installed. Please install gtsam for Factor Graph Optimization.")
    sys.exit(1)

class FactorGraphSmoother:
    """
    State-of-the-Art Factor Graph Optimization (FGO) using iSAM2.
    Replaces the Markovian ES-EKF by retaining a sliding window of historical states,
    allowing continuous re-linearization of past poses to handle GPS multipath and extreme slip.
    """
    def __init__(self):
        # 1. iSAM2 Parameters
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01) # Aggressively relinearize for high-slip accuracy
        parameters.setRelinearizeSkip(1)
        self.isam = gtsam.ISAM2(parameters)
        
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        
        # 2. IMU Preintegration Setup
        # Assuming a Z-up coordinate frame, gravity is -9.81 along Z
        imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        
        # Realistic noise models for a high-fidelity motorsport IMU
        imu_params.setAccelerometerCovariance(np.eye(3) * 1e-3)
        imu_params.setGyroscopeCovariance(np.eye(3) * 1e-4)
        imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)
        
        self.imu_preintegrated = gtsam.PreintegratedImuMeasurements(
            imu_params, 
            gtsam.imuBias.ConstantBias()
        )
        
        self.pose_idx = 0
        
        # 3. Sensor Noise Models
        self.gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.1])) # x, y, z variance
        self.can_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.05) # Wheel speed derived velocity

    def initialize(self, initial_pose: gtsam.Pose3, initial_vel: np.ndarray, initial_bias: gtsam.imuBias.ConstantBias):
        """Sets the exact initial priors for the factor graph origin."""
        prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01]*6)) # tight lock on origin
        prior_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        prior_bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
        
        self.graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, prior_pose_noise))
        self.graph.add(gtsam.PriorFactorVector(V(0), initial_vel, prior_vel_noise))
        self.graph.add(gtsam.PriorFactorConstantBias(B(0), initial_bias, prior_bias_noise))
        
        self.initial_estimates.insert(X(0), initial_pose)
        self.initial_estimates.insert(V(0), initial_vel)
        self.initial_estimates.insert(B(0), initial_bias)
        
        self.isam.update(self.graph, self.initial_estimates)
        self.graph.resize(0)
        self.initial_estimates.clear()

    def add_imu_measurement(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """Integrates high-frequency IMU data on the manifold between factor nodes."""
        self.imu_preintegrated.integrateMeasurement(accel, gyro, dt)

    def add_gps_node(self, gps_point: np.ndarray, can_velocity=None):
        """
        Triggered asynchronously when a low-frequency GPS measurement arrives.
        Fuses the preintegrated IMU factor, the GPS unary factor, and CAN speeds.
        """
        self.pose_idx += 1
        i = self.pose_idx
        
        # 1. Add IMU Preintegration Factor (Linking previous state to current)
        imu_factor = gtsam.ImuFactor(X(i-1), V(i-1), X(i), V(i), B(i-1), self.imu_preintegrated)
        self.graph.add(imu_factor)
        
        # 2. Add Bias Evolution Factor (Random Walk to prevent sensor drift)
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-5)
        self.graph.add(gtsam.BetweenFactorConstantBias(B(i-1), B(i), gtsam.imuBias.ConstantBias(), bias_noise))
        
        # 3. Add GPS Unary Factor
        gps_factor = gtsam.GPSFactor(X(i), gtsam.Point3(*gps_point), self.gps_noise)
        self.graph.add(gps_factor)
        
        # 4. Add CAN Velocity Factor (If wheel speeds are available)
        if can_velocity is not None:
            # Assumes velocity vector [vx, vy, vz] converted to navigation frame
            self.graph.add(gtsam.PriorFactorVector(V(i), can_velocity, self.can_vel_noise))
        
        # 5. Predict Initial Guesses for iSAM2 using Preintegration Kinematics
        prev_state = self.isam.calculateEstimate()
        prev_pose = prev_state.atPose3(X(i-1))
        prev_vel = prev_state.atVector(V(i-1))
        prev_bias = prev_state.atConstantBias(B(i-1))
        
        nav_state = gtsam.NavState(prev_pose, prev_vel)
        pred_nav_state = self.imu_preintegrated.predict(nav_state, prev_bias)
        
        self.initial_estimates.insert(X(i), pred_nav_state.pose())
        self.initial_estimates.insert(V(i), pred_nav_state.velocity())
        self.initial_estimates.insert(B(i), prev_bias)
        
        # 6. Execute iSAM2 Optimization Update
        self.isam.update(self.graph, self.initial_estimates)
        
        # 7. Clear graph and reset preintegration for the next temporal window
        self.graph.resize(0)
        self.initial_estimates.clear()
        
        # Get optimal bias from current run and apply it to next integration
        current_state = self.isam.calculateEstimate()
        optimized_bias = current_state.atConstantBias(B(i))
        self.imu_preintegrated.resetIntegrationAndSetBias(optimized_bias)

    def extract_trajectory(self) -> pd.DataFrame:
        """
        After the run is complete, extract the fully smoothed optimal trajectory.
        Because iSAM2 is a smoother, past nodes are corrected retroactively.
        """
        print(f"[FGO] Extracting {self.pose_idx} Smoothed Trajectory Nodes...")
        result = self.isam.calculateEstimate()
        
        trajectory = []
        for i in range(1, self.pose_idx + 1):
            pose = result.atPose3(X(i))
            vel = result.atVector(V(i))
            
            # Convert rotation matrix to Euler angles [Roll, Pitch, Yaw]
            rpy = pose.rotation().rpy()
            
            trajectory.append({
                'node': i,
                'x': pose.x(),
                'y': pose.y(),
                'z': pose.z(),
                'vx': vel[0],
                'vy': vel[1],
                'vz': vel[2],
                'roll': rpy[0],
                'pitch': rpy[1],
                'yaw': rpy[2] # True heading
            })
            
        return pd.DataFrame(trajectory)