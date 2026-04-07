"""
simulator/ros2_bridge.py
─────────────────────────────────────────────────────────────────────────────
Project-GP ROS 2 Bridge — UDP Telemetry → ROS 2 Topics

Publishes the JAX physics server's 46-DOF state as standard ROS 2 messages
so the driverless team's perception/planning/control stack can interface
with the digital twin identically to how it interfaces with the real car.

Published topics:
  /vehicle/odom         — nav_msgs/Odometry         (pose + twist)
  /vehicle/imu          — sensor_msgs/Imu           (accel + gyro)
  /vehicle/joint_states — sensor_msgs/JointState     (wheel angles + rates)
  /vehicle/wheel_loads  — std_msgs/Float32MultiArray (Fz per corner)
  /vehicle/slip         — std_msgs/Float32MultiArray (κ, α per corner)
  /vehicle/tire_temps   — std_msgs/Float32MultiArray (T per corner)
  /vehicle/energy       — std_msgs/Float32           (cumulative kJ)
  /tf                   — tf2_msgs/TFMessage          (world → base_link)

Subscribed topics:
  /cmd_vel              — geometry_msgs/Twist        (driverless stack control)
  /vehicle/reset        — std_msgs/Empty             (reset car to start)

Architecture:
  physics_server.py ──UDP 256B──→ ros2_bridge.py ──ROS 2──→ driverless stack
                                        ↑
                      driverless stack ──ROS 2──→ /cmd_vel ──UDP──→ physics_server.py

Usage:
  # In a ROS 2 workspace with this package:
  ros2 run project_gp ros2_bridge

  # Or standalone (if rclpy is on PYTHONPATH):
  python simulator/ros2_bridge.py

  # With custom ports:
  python simulator/ros2_bridge.py --udp-port 5001 --ctrl-port 5000

Dependencies:
  pip install rclpy  (or source your ROS 2 workspace)
  ROS 2 Humble/Iron/Jazzy

Note: If rclpy is not available, this module gracefully degrades to a
no-op that prints a helpful installation message. The rest of the
simulator is not affected.
"""

from __future__ import annotations

import os
import sys
import struct
import socket
import math
import time
import argparse
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.sim_config import (
    HOST, PORT_TELEM_ROS, PORT_CTRL_RECV, PHYSICS_HZ,
    S, WS_PORT,
)

# ─────────────────────────────────────────────────────────────────────────────
# §0  ROS 2 availability check
# ─────────────────────────────────────────────────────────────────────────────

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist, TransformStamped, Quaternion
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import Float32, Float32MultiArray, Empty, Header
    from tf2_ros import TransformBroadcaster
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# ─────────────────────────────────────────────────────────────────────────────
# §1  Telemetry packet parser (shared with ws_bridge.py)
# ─────────────────────────────────────────────────────────────────────────────

TX_FMT   = '<64f'
TX_BYTES = struct.calcsize(TX_FMT)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
    """Convert Euler angles (rad) to quaternion (x, y, z, w)."""
    cr, sr = math.cos(roll * 0.5),  math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5),   math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


# ─────────────────────────────────────────────────────────────────────────────
# §2  ROS 2 Bridge Node
# ─────────────────────────────────────────────────────────────────────────────

if HAS_ROS2:

    class ProjectGPBridge(Node):
        """
        ROS 2 node that bridges UDP telemetry from the JAX physics server
        to standard ROS 2 topics.
        """

        WHEEL_NAMES = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]

        def __init__(self,
                     udp_port:  int = PORT_TELEM_ROS,
                     ctrl_host: str = HOST,
                     ctrl_port: int = PORT_CTRL_RECV):
            super().__init__("projectgp_bridge")

            self.ctrl_host = ctrl_host
            self.ctrl_port = ctrl_port

            # QoS for real-time telemetry
            qos_rt = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )

            # Publishers
            self.pub_odom   = self.create_publisher(Odometry,            "/vehicle/odom",         qos_rt)
            self.pub_imu    = self.create_publisher(Imu,                 "/vehicle/imu",          qos_rt)
            self.pub_joints = self.create_publisher(JointState,          "/vehicle/joint_states",  qos_rt)
            self.pub_loads  = self.create_publisher(Float32MultiArray,   "/vehicle/wheel_loads",  qos_rt)
            self.pub_slip   = self.create_publisher(Float32MultiArray,   "/vehicle/slip",         qos_rt)
            self.pub_temps  = self.create_publisher(Float32MultiArray,   "/vehicle/tire_temps",   qos_rt)
            self.pub_energy = self.create_publisher(Float32,             "/vehicle/energy",       qos_rt)

            # TF broadcaster (world → base_link)
            self.tf_broadcaster = TransformBroadcaster(self)

            # Subscribers
            self.sub_cmd = self.create_subscription(
                Twist, "/cmd_vel", self._cmd_vel_callback, qos_rt)
            self.sub_reset = self.create_subscription(
                Empty, "/vehicle/reset", self._reset_callback, qos_rt)

            # UDP sockets
            self._udp_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._udp_recv.bind(("0.0.0.0", udp_port))
            self._udp_recv.setblocking(False)

            self._udp_ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            # Timer: poll UDP at physics rate
            timer_period = 1.0 / PHYSICS_HZ
            self.create_timer(timer_period, self._timer_callback)

            # State
            self._frame_count = 0
            self._last_log = time.monotonic()

            self.get_logger().info(
                f"ProjectGP ROS 2 Bridge started | "
                f"UDP:{udp_port} → ROS 2 topics | "
                f"/cmd_vel → UDP:{ctrl_host}:{ctrl_port}")

        def _timer_callback(self):
            """Poll UDP socket and publish to ROS 2."""
            try:
                data, _ = self._udp_recv.recvfrom(TX_BYTES + 16)
            except BlockingIOError:
                return
            except Exception:
                return

            if len(data) < TX_BYTES:
                return

            try:
                v = struct.unpack(TX_FMT, data[:TX_BYTES])
            except struct.error:
                return

            self._frame_count += 1
            now = self.get_clock().now().to_msg()

            # ── Odometry ──────────────────────────────────────────────
            odom = Odometry()
            odom.header.stamp    = now
            odom.header.frame_id = "world"
            odom.child_frame_id  = "base_link"

            odom.pose.pose.position.x = v[3]   # X
            odom.pose.pose.position.y = v[4]   # Y
            odom.pose.pose.position.z = v[5]   # Z

            qx, qy, qz, qw = euler_to_quaternion(v[6], v[7], v[8])
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw

            odom.twist.twist.linear.x  = v[9]    # vx
            odom.twist.twist.linear.y  = v[10]   # vy
            odom.twist.twist.linear.z  = v[11]   # vz
            odom.twist.twist.angular.z = v[15]   # yaw rate
            self.pub_odom.publish(odom)

            # ── TF: world → base_link ─────────────────────────────────
            t = TransformStamped()
            t.header.stamp    = now
            t.header.frame_id = "world"
            t.child_frame_id  = "base_link"
            t.transform.translation.x = v[3]
            t.transform.translation.y = v[4]
            t.transform.translation.z = v[5]
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)

            # ── IMU ───────────────────────────────────────────────────
            imu = Imu()
            imu.header.stamp    = now
            imu.header.frame_id = "base_link"
            imu.linear_acceleration.x = v[12]  # ax
            imu.linear_acceleration.y = v[13]  # ay
            imu.linear_acceleration.z = v[14]  # az
            imu.angular_velocity.z    = v[15]  # yaw rate
            self.pub_imu.publish(imu)

            # ── Joint states (wheels) ─────────────────────────────────
            js = JointState()
            js.header.stamp = now
            js.name     = self.WHEEL_NAMES
            js.position = [0.0, 0.0, 0.0, 0.0]      # not tracked in sim
            js.velocity = [v[34], v[35], v[36], v[37]]  # omega_fl/fr/rl/rr
            js.effort   = [v[20], v[21], v[22], v[23]]  # Fz as effort proxy
            self.pub_joints.publish(js)

            # ── Wheel loads ───────────────────────────────────────────
            loads = Float32MultiArray()
            loads.data = [v[20], v[21], v[22], v[23]]  # Fz_fl/fr/rl/rr
            self.pub_loads.publish(loads)

            # ── Slip angles + ratios ──────────────────────────────────
            slip = Float32MultiArray()
            slip.data = [
                v[28], v[29], v[30], v[31],  # α_fl/fr/rl/rr
                v[32], v[33],                # κ_rl/rr
            ]
            self.pub_slip.publish(slip)

            # ── Tire temps ────────────────────────────────────────────
            temps = Float32MultiArray()
            temps.data = [v[38], v[39], v[40], v[41]]  # T_fl/fr/rl/rr
            self.pub_temps.publish(temps)

            # ── Energy ────────────────────────────────────────────────
            energy = Float32()
            energy.data = v[50]  # energy_kj
            self.pub_energy.publish(energy)

            # ── Stats ─────────────────────────────────────────────────
            t_now = time.monotonic()
            if t_now - self._last_log > 5.0:
                hz = self._frame_count / (t_now - self._last_log)
                self.get_logger().info(
                    f"Publishing at {hz:.0f} Hz | "
                    f"v={v[9]*3.6:.1f} km/h | "
                    f"pos=({v[3]:.1f}, {v[4]:.1f})")
                self._frame_count = 0
                self._last_log = t_now

        def _cmd_vel_callback(self, msg: Twist):
            """
            Convert driverless stack's Twist command to physics server controls.

            Mapping:
              linear.x  → throttle/brake force (positive = throttle, negative = brake)
              angular.z → steering angle (rad, positive = left)
            """
            steer = float(msg.angular.z)
            lon   = float(msg.linear.x)

            throttle_f = max(0.0, lon)
            brake_f    = max(0.0, -lon)

            # Pack as sim_protocol RX format
            pkt = struct.pack('<8f', steer, throttle_f, brake_f,
                              0.0, 0.0, 0.0, 0.0, 0.0)
            self._udp_ctrl.sendto(pkt, (self.ctrl_host, self.ctrl_port))

        def _reset_callback(self, msg: Empty):
            """Reset car to start position."""
            pkt = struct.pack('<8f', 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
            self._udp_ctrl.sendto(pkt, (self.ctrl_host, self.ctrl_port))
            self.get_logger().info("Reset command sent to physics server")

        def destroy_node(self):
            self._udp_recv.close()
            self._udp_ctrl.close()
            super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
# §3  Standalone Fallback (no ROS 2)
# ─────────────────────────────────────────────────────────────────────────────

class StandaloneBridge:
    """
    Minimal UDP → stdout bridge when ROS 2 is not available.
    Useful for verifying the physics server is broadcasting correctly.
    """

    def __init__(self, udp_port: int = PORT_TELEM_ROS):
        self.udp_port = udp_port

    def run(self):
        print("=" * 60)
        print("  Project-GP ROS 2 Bridge (STANDALONE — rclpy not found)")
        print(f"  Listening on UDP :{self.udp_port}")
        print("  Printing telemetry to stdout for verification.")
        print("=" * 60)
        print()
        print("  To enable full ROS 2 integration:")
        print("    1. Source your ROS 2 workspace: source /opt/ros/humble/setup.bash")
        print("    2. pip install rclpy (if not in workspace)")
        print("    3. Re-run this script")
        print()

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.udp_port))
        sock.settimeout(2.0)

        count = 0
        t0 = time.monotonic()

        try:
            while True:
                try:
                    data, addr = sock.recvfrom(TX_BYTES + 16)
                except socket.timeout:
                    print("[Standalone] No data — is physics_server.py running?")
                    continue

                if len(data) < TX_BYTES:
                    continue

                v = struct.unpack(TX_FMT, data[:TX_BYTES])
                count += 1

                if count % 60 == 0:
                    elapsed = time.monotonic() - t0
                    hz = count / elapsed if elapsed > 0 else 0
                    print(f"  [{hz:.0f} Hz] v={v[9]*3.6:.1f} km/h | "
                          f"pos=({v[3]:.1f},{v[4]:.1f}) | "
                          f"ay={v[13]/9.81:+.2f}G | "
                          f"Fz=[{v[20]:.0f},{v[21]:.0f},{v[22]:.0f},{v[23]:.0f}]N")

        except KeyboardInterrupt:
            print("\n[Standalone] Done.")
        finally:
            sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# §4  CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Project-GP ROS 2 Bridge: UDP telemetry → ROS 2 topics")
    parser.add_argument("--udp-port", type=int, default=PORT_TELEM_ROS,
                        help=f"UDP port for telemetry (default: {PORT_TELEM_ROS})")
    parser.add_argument("--ctrl-port", type=int, default=PORT_CTRL_RECV,
                        help=f"UDP port for control forwarding (default: {PORT_CTRL_RECV})")
    args = parser.parse_args()

    if HAS_ROS2:
        rclpy.init()
        node = ProjectGPBridge(
            udp_port=args.udp_port,
            ctrl_port=args.ctrl_port,
        )
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        bridge = StandaloneBridge(udp_port=args.udp_port)
        bridge.run()


if __name__ == "__main__":
    main()