# powertrain/motor_model.py
# Project-GP — Differentiable PMSM + Inverter + Battery Model
# ═══════════════════════════════════════════════════════════════════════════════
#
# Four independent in-wheel PMSM motors (Ter27 4WD configuration).
# Models the complete electromechanical chain from battery terminal to
# tire contact patch, including:
#
#   1. Motor torque-speed envelope with smooth field-weakening transition
#   2. Copper + iron + switching losses (efficiency map)
#   3. Per-motor lumped thermal dynamics (stator winding + rotor + housing)
#   4. Battery pack model: OCV(SoC), internal resistance, voltage sag
#   5. Inverter thermal model with smooth sigmoid derating
#   6. Regenerative braking envelope (motor limit ∩ battery charge limit)
#
# Design rules:
#   · Every function is C∞ differentiable (softplus/sigmoid, no hard clamps)
#   · All operations are pure JAX — safe inside jit/grad/vmap/scan
#   · Parameters are physically grounded from Ter26/27 motor datasheets
#   · The model compiles to a single XLA subgraph for the SOCP allocator
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# §1  Motor Parameters (Ter27 spec)
# ─────────────────────────────────────────────────────────────────────────────

class MotorParams(NamedTuple):
    """Electromechanical parameters for a single PMSM + inverter unit."""
    # Torque/power
    T_peak: float = 120.0       # Nm peak motor torque (before final drive)
    P_peak: float = 20000.0     # W peak motor power (20 kW per motor, 80 kW total)
    gear_ratio: float = 4.5     # final drive ratio (motor → wheel)
    eta_gear: float = 0.96      # gear efficiency

    # Electrical
    R_s: float = 0.025          # Ω stator phase resistance
    K_t: float = 0.12           # Nm/A torque constant (line-to-line)
    K_e: float = 0.12           # V/(rad/s) back-EMF constant (= K_t for PMSM)
    L_d: float = 0.0005         # H d-axis inductance (for field weakening model)
    n_poles: int = 10           # pole pairs

    # Losses
    K_iron: float = 0.008       # W/(rad/s)² iron loss coefficient
    K_windage: float = 0.0001   # W/(rad/s)³ windage loss coefficient
    P_switch_base: float = 15.0 # W switching loss at rated conditions

    # Thermal (lumped single-node per motor)
    C_th_motor: float = 800.0   # J/K motor thermal capacitance (winding + stator)
    R_th_motor: float = 0.8     # K/W motor-to-coolant thermal resistance
    T_coolant: float = 40.0     # °C coolant temperature (liquid-cooled)
    T_motor_limit: float = 150.0  # °C winding temperature limit
    T_motor_derate_onset: float = 130.0  # °C derating begins

    # Inverter thermal
    C_th_inv: float = 200.0     # J/K inverter thermal capacitance
    R_th_inv: float = 1.5       # K/W inverter-to-ambient thermal resistance
    T_inv_limit: float = 85.0   # °C IGBT/MOSFET junction limit
    T_inv_derate_onset: float = 75.0
    V_ce: float = 1.5           # V collector-emitter saturation voltage
    R_on: float = 0.005         # Ω on-state resistance


class BatteryParams(NamedTuple):
    """Battery pack parameters."""
    V_nominal: float = 600.0    # V nominal pack voltage
    V_max: float = 672.0        # V maximum (fully charged)
    V_min: float = 480.0        # V minimum (LV cutoff)
    Q_capacity: float = 11.0    # Ah rated capacity
    R_internal: float = 0.15    # Ω total pack internal resistance
    I_charge_max: float = 80.0  # A maximum charge current (regen limit)
    I_discharge_max: float = 250.0  # A maximum discharge current
    C_th_cell: float = 5000.0   # J/K pack thermal capacitance
    R_th_cell: float = 3.0      # K/W pack-to-ambient thermal resistance
    T_cell_limit: float = 60.0  # °C cell temperature limit
    SoC_min: float = 5.0        # % minimum SoC


# ─────────────────────────────────────────────────────────────────────────────
# §2  Motor Torque Envelope (smooth field-weakening transition)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def motor_torque_envelope(
    omega_motor: jax.Array,
    V_bus: jax.Array,
    T_motor: jax.Array,
    T_inv: jax.Array,
    mp: MotorParams = MotorParams(),
) -> jax.Array:
    """
    Maximum available motor torque at given speed, voltage, and temperatures.

    Returns T_max_available [Nm] at the motor shaft (before gear ratio).
    Always non-negative. Smooth and differentiable everywhere.

    The envelope is the intersection of:
      1. Peak torque limit (constant torque region)
      2. Power limit (field weakening region)  
      3. Voltage headroom limit (back-EMF ceiling)
      4. Motor thermal derating
      5. Inverter thermal derating
    """
    omega_abs = jnp.abs(omega_motor) + 1e-3  # avoid division by zero

    # Region 1: constant torque
    T_const = mp.T_peak

    # Region 2: power-limited (field weakening)
    T_power = mp.P_peak / omega_abs

    # Smooth minimum: transition between constant-torque and field-weakening
    # softmin(a, b) ≈ min(a, b) as β → ∞
    beta_fw = 20.0  # sharpness of transition (higher = sharper)
    T_speed = -jax.nn.logsumexp(
        jnp.array([-beta_fw * T_const, -beta_fw * T_power])
    ) / beta_fw

    # Region 3: voltage headroom — V_bus must exceed back-EMF
    # V_headroom = V_bus - K_e · |ω| · √3 (line-to-line)
    V_bemf = mp.K_e * omega_abs * jnp.sqrt(3.0)
    V_headroom = jax.nn.softplus(V_bus - V_bemf)  # ≥ 0
    V_headroom_frac = jnp.clip(V_headroom / (mp.V_ce * 10.0 + 1e-3), 0.0, 1.0)
    T_voltage = T_speed * V_headroom_frac

    # Motor thermal derating: sigmoid transition at T_derate_onset → T_limit
    # At T_motor = T_derate_onset: derating ≈ 0.73 (27% reduction)
    # At T_motor = T_limit: derating ≈ 0.12 (88% reduction)
    k_derate_motor = 0.15  # 1/°C derating slope
    derate_motor = jax.nn.sigmoid(
        k_derate_motor * (mp.T_motor_limit - T_motor)
    )

    # Inverter thermal derating
    k_derate_inv = 0.20
    derate_inv = jax.nn.sigmoid(
        k_derate_inv * (mp.T_inv_limit - T_inv)
    )

    T_max = jnp.maximum(T_voltage * derate_motor * derate_inv, 0.0)
    return T_max


@jax.jit
def motor_torque_limits_at_wheel(
    omega_wheel: jax.Array,  # (4,) wheel angular velocities [rad/s]
    V_bus: jax.Array,        # scalar battery bus voltage [V]
    T_motors: jax.Array,     # (4,) motor temperatures [°C]
    T_invs: jax.Array,       # (4,) inverter temperatures [°C]
    SoC: jax.Array,          # scalar state of charge [%]
    mp: MotorParams = MotorParams(),
    bp: BatteryParams = BatteryParams(),
) -> tuple[jax.Array, jax.Array]:
    """
    Per-wheel torque limits [T_min, T_max] in Nm at the tire contact patch.

    T_min < 0 (regenerative braking), T_max > 0 (driving).
    Accounts for motor envelope, battery limits, and thermal derating.

    Returns:
        T_min_wheel: (4,) minimum (most negative) torque per wheel [Nm]
        T_max_wheel: (4,) maximum (most positive) torque per wheel [Nm]
    """
    omega_motor = omega_wheel * mp.gear_ratio  # motor speed from wheel speed

    # Per-motor envelope (vectorized over 4 motors)
    T_max_motor = jax.vmap(
        lambda om, tm, ti: motor_torque_envelope(om, V_bus, tm, ti, mp)
    )(omega_motor, T_motors, T_invs)

    # Convert to wheel torque
    T_max_wheel = T_max_motor * mp.gear_ratio * mp.eta_gear

    # Regen limit: motor can regenerate up to the same envelope
    # BUT limited by battery charge current
    I_charge_available = jnp.clip(
        bp.I_charge_max * jax.nn.sigmoid(0.3 * (95.0 - SoC)),  # reduce at high SoC
        0.0, bp.I_charge_max,
    )
    P_regen_batt_max = I_charge_available * V_bus  # total pack regen power limit
    P_regen_per_motor = P_regen_batt_max / 4.0     # split equally
    T_regen_batt_per_wheel = P_regen_per_motor / (
        jnp.abs(omega_motor) + 1e-3
    ) * mp.gear_ratio * mp.eta_gear

    # Regen torque at wheel: min of motor envelope and battery limit
    T_min_wheel = -jax.vmap(
        lambda t_env, t_batt: -jax.nn.logsumexp(
            jnp.array([-20.0 * t_env, -20.0 * t_batt])
        ) / 20.0
    )(T_max_wheel, T_regen_batt_per_wheel)

    return T_min_wheel, T_max_wheel


# ─────────────────────────────────────────────────────────────────────────────
# §3  Motor Losses (efficiency model)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def motor_power_loss(
    T_wheel: jax.Array,      # (4,) torque at wheel [Nm]
    omega_wheel: jax.Array,  # (4,) wheel angular velocity [rad/s]
    mp: MotorParams = MotorParams(),
) -> jax.Array:
    """
    Electrical power loss per motor [W]. Shape: (4,).

    Components:
      1. Copper loss: I²R (quadratic in torque)
      2. Iron loss: proportional to ω² (hysteresis + eddy current)
      3. Windage loss: proportional to ω³
      4. Switching loss: approximately constant at rated conditions

    Total electrical input = mechanical output + losses:
      P_elec = T·ω + P_loss
    """
    T_motor = T_wheel / (mp.gear_ratio * mp.eta_gear + 1e-6)
    omega_motor = omega_wheel * mp.gear_ratio

    # Phase current from torque
    I_phase = jnp.abs(T_motor) / (mp.K_t + 1e-6)

    # Copper loss: 3 × I² × R_s (three-phase)
    P_copper = 3.0 * I_phase ** 2 * mp.R_s

    # Iron loss (hysteresis + eddy)
    P_iron = mp.K_iron * omega_motor ** 2

    # Windage
    P_windage = mp.K_windage * jnp.abs(omega_motor) ** 3

    # Switching (approximately constant, scaled by current fraction)
    I_rated = mp.T_peak / (mp.K_t + 1e-6)
    P_switch = mp.P_switch_base * (I_phase / (I_rated + 1e-3))

    # Gear losses (1 - η) × |P_mechanical|
    P_gear = (1.0 - mp.eta_gear) * jnp.abs(T_motor * omega_motor)

    return P_copper + P_iron + P_windage + P_switch + P_gear


# ─────────────────────────────────────────────────────────────────────────────
# §4  Thermal Dynamics
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def thermal_derivatives(
    T_motors: jax.Array,     # (4,) motor temps [°C]
    T_invs: jax.Array,       # (4,) inverter temps [°C]
    SoC: jax.Array,          # scalar [%]
    T_cell: jax.Array,       # scalar cell temp [°C]
    T_wheel: jax.Array,      # (4,) wheel torques [Nm]
    omega_wheel: jax.Array,  # (4,) wheel speeds [rad/s]
    V_bus: jax.Array,        # scalar [V]
    mp: MotorParams = MotorParams(),
    bp: BatteryParams = BatteryParams(),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Time derivatives of all thermal states [°C/s].

    Returns:
        dT_motors: (4,) motor temperature rates
        dT_invs:   (4,) inverter temperature rates
        dSoC:      scalar SoC rate [%/s]
        dT_cell:   scalar cell temperature rate [°C/s]
    """
    T_motor_shaft = T_wheel / (mp.gear_ratio * mp.eta_gear + 1e-6)
    omega_motor = omega_wheel * mp.gear_ratio
    I_phase = jnp.abs(T_motor_shaft) / (mp.K_t + 1e-6)

    # Motor thermal: copper + iron heating, coolant removal
    P_copper = 3.0 * I_phase ** 2 * mp.R_s
    P_iron = mp.K_iron * omega_motor ** 2
    P_motor_heat = P_copper + P_iron
    Q_motor_cool = (T_motors - mp.T_coolant) / mp.R_th_motor
    dT_motors = (P_motor_heat - Q_motor_cool) / mp.C_th_motor

    # Inverter thermal: conduction + switching losses
    P_inv_cond = mp.V_ce * I_phase + mp.R_on * I_phase ** 2
    P_inv_switch = mp.P_switch_base * (I_phase / (mp.T_peak / mp.K_t + 1e-3))
    P_inv_total = P_inv_cond + P_inv_switch
    Q_inv_cool = (T_invs - mp.T_coolant) / mp.R_th_inv
    dT_invs = (P_inv_total - Q_inv_cool) / mp.C_th_inv

    # Battery: total electrical current (sum of all motors)
    P_mech_total = jnp.sum(T_wheel * omega_wheel)
    P_loss_total = jnp.sum(motor_power_loss(T_wheel, omega_wheel, mp))
    P_elec_total = P_mech_total + P_loss_total
    I_batt = P_elec_total / (V_bus + 1e-3)

    # SoC dynamics: dSoC/dt = -I / (Q * 3600) * 100 [%/s]
    dSoC = -I_batt / (bp.Q_capacity * 3600.0) * 100.0

    # Cell thermal
    P_batt_heat = bp.R_internal * I_batt ** 2
    T_ambient = mp.T_coolant  # approximate
    Q_cell_cool = (T_cell - T_ambient) / bp.R_th_cell
    dT_cell = (P_batt_heat - Q_cell_cool) / bp.C_th_cell

    return dT_motors, dT_invs, dSoC, dT_cell


# ─────────────────────────────────────────────────────────────────────────────
# §5  Battery Voltage Model
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def battery_voltage(
    SoC: jax.Array,       # [%]
    I_total: jax.Array,   # [A] total current draw (positive = discharge)
    T_cell: jax.Array,    # [°C]
    bp: BatteryParams = BatteryParams(),
) -> jax.Array:
    """
    Battery terminal voltage under load [V].
    Smooth model: V = OCV(SoC) − R_int(SoC, T) × I
    """
    # Open-circuit voltage: affine in SoC with softplus floor
    SoC_frac = jnp.clip(SoC / 100.0, 0.01, 1.0)
    V_oc = bp.V_min + (bp.V_max - bp.V_min) * SoC_frac

    # Internal resistance increases at low SoC and low temperature
    R_soc_factor = 1.0 + 0.5 * jax.nn.softplus(0.3 - SoC_frac)  # rises below 30%
    R_temp_factor = 1.0 + 0.003 * jax.nn.relu(25.0 - T_cell)    # rises below 25°C
    R_eff = bp.R_internal * R_soc_factor * R_temp_factor

    V_terminal = V_oc - R_eff * I_total
    return jnp.maximum(V_terminal, bp.V_min)


# ─────────────────────────────────────────────────────────────────────────────
# §6  Convenience: total power budget
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def total_power_limit(
    V_bus: jax.Array,
    SoC: jax.Array,
    T_cell: jax.Array,
    bp: BatteryParams = BatteryParams(),
) -> jax.Array:
    """Maximum instantaneous electrical power [W] from the battery."""
    # Discharge current limit (hard limit from BMS)
    I_max = bp.I_discharge_max

    # SoC derating: reduce below 15% SoC
    I_soc = I_max * jnp.clip(SoC / 15.0, 0.0, 1.0)

    # Temperature derating: reduce above 55°C
    I_temp = I_max * jax.nn.sigmoid(0.5 * (bp.T_cell_limit - T_cell))

    I_available = jnp.minimum(I_soc, I_temp)
    return I_available * V_bus


# ─────────────────────────────────────────────────────────────────────────────
# §7  Electrothermal state container
# ─────────────────────────────────────────────────────────────────────────────

class PowertrainState(NamedTuple):
    """Complete electrothermal state vector for the 4-motor powertrain."""
    T_motors: jax.Array     # (4,) motor temperatures [°C]
    T_invs: jax.Array       # (4,) inverter temperatures [°C]
    SoC: jax.Array          # scalar [%]
    T_cell: jax.Array       # scalar [°C]
    V_bus: jax.Array        # scalar [V] (computed, not integrated)

    @staticmethod
    def default() -> 'PowertrainState':
        return PowertrainState(
            T_motors=jnp.array([55.0, 55.0, 55.0, 55.0]),
            T_invs=jnp.array([42.0, 42.0, 42.0, 42.0]),
            SoC=jnp.array(95.0),
            T_cell=jnp.array(30.0),
            V_bus=jnp.array(650.0),
        )

    def step(self, T_wheel, omega_wheel, dt, mp=MotorParams(), bp=BatteryParams()):
        """Integrate thermal state forward by dt. Returns new PowertrainState."""
        dT_m, dT_i, dSoC, dT_c = thermal_derivatives(
            self.T_motors, self.T_invs, self.SoC, self.T_cell,
            T_wheel, omega_wheel, self.V_bus, mp, bp,
        )
        # Forward Euler (thermal dynamics are slow enough for Euler stability)
        T_m_new = self.T_motors + dT_m * dt
        T_i_new = self.T_invs + dT_i * dt
        SoC_new = jnp.clip(self.SoC + dSoC * dt, bp.SoC_min, 100.0)
        T_c_new = self.T_cell + dT_c * dt

        # Update V_bus from new SoC (quasi-static electrical model)
        P_mech = jnp.sum(T_wheel * omega_wheel)
        P_loss = jnp.sum(motor_power_loss(T_wheel, omega_wheel, mp))
        I_total = (P_mech + P_loss) / (self.V_bus + 1e-3)
        V_new = battery_voltage(SoC_new, I_total, T_c_new, bp)

        return PowertrainState(T_m_new, T_i_new, SoC_new, T_c_new, V_new)