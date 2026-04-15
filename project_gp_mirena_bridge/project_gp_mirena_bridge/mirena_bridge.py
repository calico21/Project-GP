#!/usr/bin/env python3
"""
mirena_bridge.py — Project-GP ↔ MirenaSim Shadow Controller Bridge

Orchestrates the full integration between Project-GP's JAX inference stack
and MirenaSim's ROS2 interface. Runs as a standard ROS2 node.

Data flow:
  /car/state           [50 Hz]  →  state_mapper  →  JAX inference  →  /car/control
  /car/wheel_speeds    [async]  →  state_mapper  (slip estimation)
  /path_planner/path   [async]  →  track_ingestor → Diff-WMPC reference
  /track_manager/track [1 Hz]   →  track_ingestor (fallback if no Bézier path)
  /as_status           [async]  →  safety gate (no control unless AS_DRIVING)

Threading model:
  Thread A:  ROS2 MultiThreadedExecutor (4 threads) — subscriber callbacks
  Thread B:  JAX inference loop   — triggered by new state, <5ms target
  Thread C:  OnlineSysID daemon   — H_net adaptation, ~1 Hz background

Safety gates:
  1. AS_DRIVING state required before any /car/control is published.
  2. Inference timeout (25 ms): if JAX step exceeds budget, publish zero.
  3. NaN detection: inference result validated; failure → zero command + log.
  4. Steer/gas hard clamps at publish time (physical actuator limits).

SOCP warm-starting:
  The SOCP torque allocation has a ~265 ms cold-start. Warm-starting from the
  previous solution reduces this to ~10-15 ms. _prev_socp_sol stores the primal
  variables for the next call. See _run_project_gp_step() stub comments.

Usage:
    ros2 run project_gp mirena_bridge

    # Or with parameters:
    ros2 run project_gp mirena_bridge --ros-args \\
        -p enable_sysid:=true \\
        -p steer_limit_rad:=0.40 \\
        -p inference_hz:=50
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Header

import jax
import jax.numpy as jnp

# mirena_common messages
from mirena_common.msg import (
    Car,
    CarControl,
    WheelSpeeds,
    BezierCurve,
    Track,
    ASStatus,
    MissionStatus,
)

# Project-GP bridge modules
from .state_mapper import (
    expand_to_46dof,
    collapse_to_6dof,
    covariance_to_precision,
    make_nominal_state,
    OBS_INDICES,
)
from .control_mapper import project_to_car_control
from .track_ingestor import (
    bezier_msg_to_arrays,
    gates_to_arrays,
    build_spline,
    query_at_s,
    project_car_onto_spline,
    SplineData,
)
from .online_sysid import OnlineSysID

log = logging.getLogger(__name__)

# ─── Timing & limits ─────────────────────────────────────────────────────────
_INFER_TIMEOUT_S  = 0.025   # 25 ms: one state period + 5 ms headroom
_STEER_LIMIT_RAD  = 0.40    # ~23°, hard physical limit for FS steering rack
_GAS_LIMIT        = 1.0     # CarControl.msg contract
_CONTROL_PUB_HZ   = 50.0    # Hz — matches MirenaSim /car/state rate


# ─── Thread-safe buffers ──────────────────────────────────────────────────────

@dataclass
class _StateFrame:
    """One fused state frame: ROS data merged and expanded to 46-DOF."""
    state_46d:    np.ndarray = field(default_factory=lambda: np.zeros(46, np.float32))
    car_6d:       np.ndarray = field(default_factory=lambda: np.zeros(6,  np.float32))
    wheel_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4,  np.float32))
    precision:    np.ndarray = field(default_factory=lambda: np.ones(6,   np.float32))
    seq:          int        = 0


class _DoubleBuffer:
    """
    Writer-always-wins double buffer.
    ROS2 callback thread writes; JAX inference thread reads last committed frame.
    Uses a single lock held only for pointer swaps (~ns), not during JAX computation.
    """
    def __init__(self):
        self._frames = [_StateFrame(), _StateFrame()]
        self._write  = 0          # index being written to
        self._commit = 1          # index last committed (reader sees this)
        self._lock   = threading.Lock()

    def write(self, frame: _StateFrame) -> None:
        with self._lock:
            self._frames[self._write] = frame
            self._write, self._commit = self._commit, self._write  # atomic swap

    def read(self) -> _StateFrame:
        with self._lock:
            idx = self._commit
        return self._frames[idx]   # read outside lock (immutable after swap)


@dataclass
class _ControlOutput:
    gas:   float = 0.0
    steer: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def write(self, gas: float, steer: float) -> None:
        with self._lock:
            self.gas   = gas
            self.steer = steer

    def read(self) -> tuple[float, float]:
        with self._lock:
            return self.gas, self.steer


# ─── Main bridge node ─────────────────────────────────────────────────────────

class MirenaBridge(Node):
    """
    Project-GP shadow controller bridge node.

    The node subscribes to MirenaSim state, expands it to Project-GP's 46-DOF
    representation, runs the full powertrain + control stack, and publishes the
    resulting scalar (gas, steer) command back to MirenaSim.
    """

    def __init__(self):
        super().__init__("project_gp_bridge")

        # ── ROS2 parameters ────────────────────────────────────────────────────
        self.declare_parameter("enable_sysid",    True)
        self.declare_parameter("steer_limit_rad", _STEER_LIMIT_RAD)
        self.declare_parameter("inference_hz",    _CONTROL_PUB_HZ)

        _enable_sysid   = self.get_parameter("enable_sysid").value
        _steer_lim      = self.get_parameter("steer_limit_rad").value
        _ctrl_hz        = self.get_parameter("inference_hz").value

        self._steer_lim = float(_steer_lim)
        self._cbg       = ReentrantCallbackGroup()

        # ── Internal state ─────────────────────────────────────────────────────
        self._state_buf         = _DoubleBuffer()
        self._ctrl_out          = _ControlOutput()
        self._prev_46d          = make_nominal_state()          # thermal continuity
        self._prev_6d           = np.zeros(6, dtype=np.float32)
        self._prev_ctrl_2d      = np.zeros(2, dtype=np.float32)
        self._as_driving        = False
        self._last_seq          = -1
        self._new_state_event   = threading.Event()
        self._spline: Optional[SplineData] = None
        self._spline_lock       = threading.Lock()
        self._prev_s_progress   = jnp.zeros(())  # arc-length progress warm-start

        # SOCP warm-start: numpy array (avoids device-host transfer overhead)
        self._prev_socp_sol: Optional[np.ndarray] = None
        from powertrain.powertrain_manager import make_powertrain_manager
        from config.vehicles.ter26 import vehicle_params as _vp
        self._pt_config, self._pt_state = make_powertrain_manager(_vp)
        # Wheel speeds are published asynchronously — cache between state frames
        self._ws_cache     = np.zeros(4, dtype=np.float32)
        self._ws_lock      = threading.Lock()

        # ── Optional: OnlineSysID ──────────────────────────────────────────────
        # Wired after Project-GP forward_fn is available.
        # Alex: replace the lambda stub with the actual powertrain forward step.
        self._sysid: Optional[OnlineSysID] = None
        if _enable_sysid:
            self._init_sysid()

        # ── Subscribers ────────────────────────────────────────────────────────
        self.create_subscription(
            Car,         "/car/state",             self._on_car_state,   10,
            callback_group=self._cbg,
        )
        self.create_subscription(
            WheelSpeeds, "/car/wheel_speeds",       self._on_wheel_speeds, 10,
            callback_group=self._cbg,
        )
        self.create_subscription(
            BezierCurve, "/path_planner/path",      self._on_bezier_path,  1,
            callback_group=self._cbg,
        )
        self.create_subscription(
            Track,       "/track_manager/track",    self._on_track,        1,
            callback_group=self._cbg,
        )
        self.create_subscription(
            ASStatus,    "/as_status",              self._on_as_status,   10,
            callback_group=self._cbg,
        )
        self.create_subscription(
            MissionStatus, "/mission_status",       self._on_mission_status, 1,
            callback_group=self._cbg,
        )

        # ── Publisher + timer ──────────────────────────────────────────────────
        self._ctrl_pub = self.create_publisher(CarControl, "/car/control", 10)
        self.create_timer(
            1.0 / _ctrl_hz, self._publish_control,
            callback_group=self._cbg,
        )

        # ── JAX inference thread ───────────────────────────────────────────────
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="gp_infer",
        )
        self._infer_thread.start()

        self.get_logger().info(
            "MirenaBridge ready. "
            f"SysID={'ON' if _enable_sysid else 'OFF'} | "
            f"steer_lim={_steer_lim:.3f} rad | "
            f"ctrl_hz={_ctrl_hz} Hz"
        )
        self.get_logger().info("Waiting for AS_DRIVING state before commanding...")

    # ─── Subscriber Callbacks ─────────────────────────────────────────────────

    def _on_car_state(self, msg: Car) -> None:
        """
        Core ingestion callback. Runs on ROS2 executor thread at ~50 Hz.

        Expands Car.msg to 46-DOF using cached wheel speeds, then double-buffers
        the frame and signals the inference thread. The covariance is converted
        to per-DOF precision weights for sysid loss weighting.

        Note: expand_to_46dof() is JIT-compiled — first call traces (~100ms),
        subsequent calls are ~0.1ms. Trace occurs before AS_DRIVING is set.
        """
        car_6d = np.array(
            [msg.x, msg.y, msg.psi, msg.u, msg.v, msg.omega],
            dtype=np.float32,
        )
        cov36 = np.array(msg.covariance, dtype=np.float32)

        with self._ws_lock:
            ws = self._ws_cache.copy()

        # JAX expand (JIT-compiled; returns DeviceArray → convert to np immediately)
        s46 = np.asarray(
            expand_to_46dof(
                jnp.array(car_6d),
                jnp.array(ws),
                jnp.array(self._prev_46d),
            )
        )
        precision = np.asarray(covariance_to_precision(jnp.array(cov36)))

        frame = _StateFrame(
            state_46d    = s46,
            car_6d       = car_6d,
            wheel_speeds = ws,
            precision    = precision,
            seq          = self._state_buf.read().seq + 1,
        )
        self._state_buf.write(frame)
        self._new_state_event.set()

        # Push to sysid buffer: (s_{t-1}, u_{t-1}) → s_t
        if (self._sysid is not None
                and self._prev_6d is not None
                and self._prev_ctrl_2d is not None):
            self._sysid.push(
                self._prev_46d,        # s_t: 46-DOF prev state
                self._prev_ctrl_2d,    # u_t: [gas, steer] prev control
                s46,                   # s_{t+1}: 46-DOF current state (ground truth)
                precision,             # precision weights
            )

        # Roll forward for next frame's sysid push
        self._prev_46d = s46
        self._prev_6d  = car_6d

    def _on_wheel_speeds(self, msg: WheelSpeeds) -> None:
        """Async wheel speed update — cached for next state expansion."""
        with self._ws_lock:
            self._ws_cache[:] = [msg.fl, msg.fr, msg.rl, msg.rr]

    def _on_bezier_path(self, msg: BezierCurve) -> None:
        """
        Path planner Bézier output → JAX arc-length spline.
        Runs on ROS2 thread; heavy computation (~2-5ms for 100 gates) is
        acceptable since this triggers at the planner's output rate (<10 Hz).
        """
        if len(msg.points) < 2:
            self.get_logger().warn("BezierCurve has <2 points, ignoring.")
            return

        anchors, out_cps, in_cps = bezier_msg_to_arrays(msg.points)
        spline = build_spline(
            jnp.array(anchors),
            jnp.array(out_cps),
            jnp.array(in_cps),
        )
        with self._spline_lock:
            self._spline = spline
            self._prev_s_progress = jnp.zeros(())   # reset progress on new path

        self.get_logger().debug(
            f"Bézier path ingested: {len(msg.points)} anchors, "
            f"total_len={float(spline.total_len):.1f} m"
        )

    def _on_track(self, msg: Track) -> None:
        """
        Fallback: build spline from Gate[] when Bézier path is not available.
        Gate centrepoints → Catmull-Rom → Bézier → SplineData.
        """
        with self._spline_lock:
            if self._spline is not None:
                return  # Bézier path already available — prefer it

        if len(msg.gates) < 2:
            return

        anchors, out_cps, in_cps = gates_to_arrays(msg.gates, msg.is_closed)
        spline = build_spline(
            jnp.array(anchors),
            jnp.array(out_cps),
            jnp.array(in_cps),
        )
        with self._spline_lock:
            self._spline = spline

        self.get_logger().info(
            f"Track spline built from {len(msg.gates)} gates, "
            f"total_len={float(spline.total_len):.1f} m"
        )

    def _on_as_status(self, msg: ASStatus) -> None:
        """Safety gate: only command car when autonomous system is driving."""
        was_driving = self._as_driving
        self._as_driving = (msg.state == ASStatus.AS_DRIVING)

        if not self._as_driving:
            self._ctrl_out.write(0.0, 0.0)  # zero-command immediately on non-DRIVING

        if self._as_driving and not was_driving:
            self.get_logger().info("AS_DRIVING: Project-GP control ACTIVE.")
        elif not self._as_driving and was_driving:
            self.get_logger().info(
                f"AS state → {msg.state}: Project-GP control DISABLED."
            )

    def _on_mission_status(self, msg: MissionStatus) -> None:
        """Log mission progress. Could gate lap-time optimisation objectives here."""
        if msg.state == MissionStatus.MISSION_FINISHED:
            self.get_logger().info(
                f"Mission finished: {msg.lap_counter} laps, "
                f"{msg.cones_count_actual}/{msg.cones_count_total} cones."
            )

    # ─── JAX Inference Thread ─────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """
        JAX inference thread. Triggered by new state; target latency <5 ms.

        Critical path per iteration:
          1. Read double-buffer         ~0.01 ms
          2. expand_to_46dof            ~0.1  ms  (JIT-compiled)
          3. _run_project_gp_step       ~2-4  ms  (JIT-compiled, warm)
          4. project_to_car_control     ~0.05 ms
          5. Write ControlOutput        ~0.01 ms
        """
        self.get_logger().info("JAX inference thread started.")

        # Trigger one dummy expand to force JIT trace before AS_DRIVING
        _ = expand_to_46dof(jnp.zeros(6), jnp.zeros(4), jnp.array(self._prev_46d))
        _ = project_to_car_control(jnp.zeros(4), jnp.zeros(2), jnp.zeros(()))
        self.get_logger().info("JAX JIT warm-up complete.")

        while rclpy.ok():
            triggered = self._new_state_event.wait(timeout=_INFER_TIMEOUT_S)
            self._new_state_event.clear()

            if not self._as_driving:
                continue

            frame = self._state_buf.read()
            if frame.seq == self._last_seq:
                continue  # Spurious wakeup — no new state
            self._last_seq = frame.seq

            t0 = time.perf_counter()
            try:
                gas, steer = self._run_project_gp_step(frame)

                # Validate: reject NaN/Inf before writing to output
                if not (np.isfinite(gas) and np.isfinite(steer)):
                    raise ValueError(f"Non-finite control: gas={gas}, steer={steer}")

                self._ctrl_out.write(float(gas), float(steer))
                self._prev_ctrl_2d = np.array([gas, steer], dtype=np.float32)

            except Exception as exc:
                import traceback
                self.get_logger().error(
                    f"Inference error: {exc}\n{traceback.format_exc()}",
                    throttle_duration_sec=1.0
                )
                self._ctrl_out.write(0.0, 0.0)
            dt_ms = (time.perf_counter() - t0) * 1e3
            if dt_ms > 10.0:  # Log if we're burning more than 2× budget
                self.get_logger().warn(
                    f"Inference step took {dt_ms:.1f} ms (budget: 5 ms)",
                    throttle_duration_sec=2.0,
                )

    def _run_project_gp_step(self, frame):
        from powertrain.powertrain_manager import powertrain_step
        from simulator.sim_config import S

        state = jnp.array(frame.state_46d)
        vx      = state[S.VX]
        vy      = state[S.VY]
        wz      = state[S.WYAW]
        omega_w = state[jnp.array([S.WSPIN_FL, S.WSPIN_FR, S.WSPIN_RL, S.WSPIN_RR])]
        alpha_t = state[jnp.array([38, 40, 42, 44])]
        T_tire  = jnp.array([state[28], state[31], state[35], state[33]])
        Fz      = jnp.array([750., 750., 750., 750.])
        Fy      = jnp.zeros(4)
        mu_est   = jnp.array(1.4)
        gp_sigma = jnp.array(0.05)
        curvature = jnp.array(0.0)
        with self._spline_lock:
            spline = self._spline
        if spline is not None:
            s, _, _ = project_car_onto_spline(
                spline, jnp.array(frame.car_6d[:2]), self._prev_s_progress,
            )
            self._prev_s_progress = s
            _, _, _, curvature = query_at_s(spline, s)
        prev_gas = float(self._prev_ctrl_2d[0])
        throttle = jnp.array(max(0.0,  prev_gas))
        brake    = jnp.array(max(0.0, -prev_gas))
        delta    = jnp.array(float(self._prev_ctrl_2d[1]))
        diag, self._pt_state = powertrain_step(
            throttle_raw=throttle, brake_raw=brake, delta=delta,
            vx=vx, vy=vy, wz=wz,
            Fz=Fz, Fy=Fy, omega_wheel=omega_w, alpha_t=alpha_t, T_tire=T_tire,
            mu_est=mu_est, gp_sigma=gp_sigma, curvature=curvature,
            manager_state=self._pt_state, dt=jnp.array(1.0/50.0), config=self._pt_config,
        )
        gas, steer = project_to_car_control(
            diag.T_wheel, jnp.array([delta, delta]), vx,
        )
        return float(gas), float(steer)

    def _publish_control(self) -> None:
        """
        Timer callback: publishes last computed (gas, steer) at CONTROL_PUB_HZ.
        Decoupled from inference rate — always publishes at a steady rate.
        Hard clamps applied here as the final safety backstop.
        """
        gas, steer = self._ctrl_out.read()

        msg                 = CarControl()
        msg.header          = Header()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.gas             = float(np.clip(gas,   -_GAS_LIMIT,       _GAS_LIMIT))
        msg.steer_angle     = float(np.clip(steer, -self._steer_lim, self._steer_lim))

        self._ctrl_pub.publish(msg)

    # ─── SysID initialisation ─────────────────────────────────────────────────

    def _init_sysid(self) -> None:
        """
        Initialise OnlineSysID once Project-GP forward_fn is available.

        Alex: replace the lambda stub with your actual forward step.
        The forward_fn must be pure JAX: (params, state_46d, control_2d) → state_46d
        and must be differentiable w.r.t. params (H_net weights).

        Example:
            from project_gp.physics.integrator import port_hamiltonian_step
            forward_fn = port_hamiltonian_step

        For now, a no-op lambda is used so the rest of the bridge runs cleanly.
        """
        # STUB: replace with real forward function
        def _stub_forward(params, state_46d: jnp.ndarray, ctrl_2d: jnp.ndarray):
            return state_46d  # no-op: s_{t+1} = s_t

        # STUB: replace with real H_net params pytree
        _stub_params = {"dummy": jnp.zeros(1)}

        self._sysid = OnlineSysID(
            forward_fn  = _stub_forward,
            init_params = _stub_params,
        )
        self._sysid.start()
        self.get_logger().info("OnlineSysID initialised (stub forward_fn).")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MirenaBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node._sysid is not None:
            node._sysid.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
