"""
project_gp_mirena_bridge — Project-GP ↔ MirenaSim ROS2 integration package.

Module overview:
    mirena_bridge   — Main ROS2 node (entry point)
    state_mapper    — JAX bidirectional 6-DOF ↔ 46-DOF state expansion
    control_mapper  — 4WD torque allocation → scalar CarControl projection
    track_ingestor  — BezierCurve.msg → arc-length-parameterised JAX spline
    online_sysid    — Online H_net adaptation via jax.grad on MirenaSim ground truth

Wiring guide (3 steps to activate the full stack):
  1. state_mapper.py  — adjust _M_* / _T_* / _S_* / _E_* index constants
                        to match your actual project_gp/physics/state.py layout.

  2. mirena_bridge.py — _run_project_gp_step(): replace the STUB block with
                        your actual powertrain_step() + wmpc_step() calls.

  3. online_sysid.py  — MirenaBridge._init_sysid(): replace _stub_forward
                        with port_hamiltonian_step from your physics integrator,
                        and _stub_params with the actual H_net param pytree.
"""

from .state_mapper   import expand_to_46dof, collapse_to_6dof, covariance_to_precision
from .control_mapper import project_to_car_control, project_with_ackermann
from .track_ingestor import build_spline, query_at_s, project_car_onto_spline
from .online_sysid   import OnlineSysID
from .mirena_bridge  import MirenaBridge, main

__all__ = [
    "MirenaBridge",
    "main",
    "expand_to_46dof",
    "collapse_to_6dof",
    "covariance_to_precision",
    "project_to_car_control",
    "project_with_ackermann",
    "build_spline",
    "query_at_s",
    "project_car_onto_spline",
    "OnlineSysID",
]
