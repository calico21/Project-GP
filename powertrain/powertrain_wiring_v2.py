from __future__ import annotations
 
import jax
import jax.numpy as jnp
from typing import NamedTuple
 
# Import the new state estimator
from powertrain.state_estimator import (
    UKFState, UKFParams, ukf_step,
    extract_estimated_state, pack_measurement_from_reading,
    make_ukf_state,
)
 
 
class HubMotorCommand(NamedTuple):
    """
    Output command for the 4WD hub motor system.
 
    This is packed into the u-vector for vehicle_dynamics.simulate_step():
      u = [delta, T_fl, T_fr, T_rl, T_rr, F_brake_hyd]
    """
    delta:        jax.Array   # [rad]  steering angle
    T_fl:         jax.Array   # [Nm]   front-left hub motor torque
    T_fr:         jax.Array   # [Nm]   front-right hub motor torque
    T_rl:         jax.Array   # [Nm]   rear-left hub motor torque
    T_rr:         jax.Array   # [Nm]   rear-right hub motor torque
    F_brake_hyd:  jax.Array   # [N]    hydraulic brake force
 
    def to_u_vector(self) -> jax.Array:
        """Pack into the 6-dim input vector for vehicle_dynamics."""
        return jnp.array([
            self.delta,
            self.T_fl, self.T_fr,
            self.T_rl, self.T_rr,
            self.F_brake_hyd,
        ])
 
 
class PowertrainOutputV2(NamedTuple):
    """Extended powertrain output including UKF state and hub motor commands."""
    command:      HubMotorCommand   # packed motor + brake commands
    diagnostics:  object            # PowertrainDiagnostics (unchanged)
    ukf_state:    UKFState          # updated UKF state (carry forward)
    imu_bias:     jax.Array         # (6,) updated IMU bias
 
 
def pack_hub_motor_command(
    T_wheel: jax.Array,         # (4,) from powertrain allocator
    delta: jax.Array,           # [rad] steering command
    F_brake_hyd: jax.Array,     # [N] from regen_blend
) -> HubMotorCommand:
    """
    Pack allocator output into HubMotorCommand.
 
    T_wheel comes from the SOCP/mpQP allocator (Nm per wheel).
    F_brake_hyd comes from regen_blend.py (N total hydraulic).
    delta is the raw steering command.
    """
    return HubMotorCommand(
        delta=delta,
        T_fl=T_wheel[0],
        T_fr=T_wheel[1],
        T_rl=T_wheel[2],
        T_rr=T_wheel[3],
        F_brake_hyd=F_brake_hyd,
    )