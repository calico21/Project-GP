# powertrain/virtual_impedance.py
# Project-GP — Virtual Impedance Controller (PIO Mitigation)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Electric motors respond in ~5 ms — 100× faster than the driver's
# neuromuscular system (~500 ms). When the car hits a bump at corner exit,
# the driver's foot bounces on the throttle at the pitch frequency (~3 Hz).
# Each bounce commands a torque pulse that couples through anti-squat
# geometry back into the suspension → positive feedback → PIO.
#
# The standard fix (5 Hz low-pass filter) kills PIO but adds 30 ms phase lag.
# The driver feels "sluggish" and compensates by being less aggressive.
#
# This module implements a VIRTUAL MECHANICAL IMPEDANCE — a simulated
# flywheel + damper attached to the throttle pedal. It provides:
#   · 55° phase lag at 3 Hz (breaks PIO loop — needs >90° for instability)
#   · Only 7% amplitude attenuation at 3 Hz (vs. 14% for LPF)
#   · Zero steady-state lag (vs. group delay of LPF)
#   · 50 ms step response (perceived as instantaneous by driver)
#
# The impedance parameters (J, C, K) enter the MORL optimization space.
# The same architecture applies to the brake pedal for ABS PIO prevention.
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# §1  Impedance Parameters (MORL-tunable)
# ─────────────────────────────────────────────────────────────────────────────

class ImpedanceParams(NamedTuple):
    """Virtual mechanical impedance parameters."""
    # Throttle channel
    J_throttle: float = 0.05    # kg·m² virtual inertia (≈50 ms response)
    C_throttle: float = 2.1     # N·m·s/rad virtual damping (ζ ≈ 0.7)
    K_throttle: float = 45.0    # N/m virtual centering stiffness (ωₙ ≈ 30 rad/s)
    # Brake channel
    J_brake: float = 0.03       # kg·m² virtual inertia (faster than throttle)
    C_brake: float = 1.5        # N·m·s/rad virtual damping
    K_brake: float = 60.0       # N/m virtual stiffness (ωₙ ≈ 45 rad/s — brakes need faster response)
    # Nonlinear enhancement
    asymmetry: float = 1.5      # brake applies faster than it releases (safety bias)
    deadband: float = 0.02      # pedal deadband [0–1] (rejects sub-2% noise)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Impedance State
# ─────────────────────────────────────────────────────────────────────────────

class ImpedanceState(NamedTuple):
    """Persistent state for the virtual impedance filters."""
    theta_throttle: jax.Array     # scalar: virtual throttle pedal position [0, 1]
    theta_dot_throttle: jax.Array # scalar: virtual throttle pedal velocity
    theta_brake: jax.Array        # scalar: virtual brake pedal position [0, 1]
    theta_dot_brake: jax.Array    # scalar: virtual brake pedal velocity

    @staticmethod
    def default() -> 'ImpedanceState':
        return ImpedanceState(
            theta_throttle=jnp.array(0.0),
            theta_dot_throttle=jnp.array(0.0),
            theta_brake=jnp.array(0.0),
            theta_dot_brake=jnp.array(0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3  Single-Channel Impedance Step
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def _impedance_channel_step(
    theta: jax.Array,          # current virtual position [0, 1]
    theta_dot: jax.Array,      # current virtual velocity
    pedal_raw: jax.Array,      # raw pedal input [0, 1]
    J: float,                  # virtual inertia
    C: float,                  # virtual damping
    K: float,                  # virtual stiffness
    dt: jax.Array,             # timestep [s]
    asymmetry: float = 1.0,    # >1 = faster application than release
    deadband: float = 0.02,    # pedal deadband
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Single impedance channel: J·θ̈ + C·θ̇ + K·(θ − u) = 0

    Reformulated as a spring pulling toward the raw pedal input:
      J·θ̈ + C_eff·θ̇ = K·(pedal − θ)

    where C_eff is asymmetric: higher damping on release than application
    (safety: applying brakes should be fast, releasing should be deliberate).

    Symplectic Euler integration (energy-consistent with the PH engine).

    Returns (theta_new, theta_dot_new, output_pedal)
    """
    # Deadband: suppress sub-threshold noise
    pedal = jnp.where(pedal_raw < deadband, 0.0, pedal_raw)

    # Direction-dependent damping (asymmetric impedance)
    # Positive theta_dot = pedal applying → lower damping (responsive)
    # Negative theta_dot = pedal releasing → higher damping (deliberate)
    direction = jnp.tanh(theta_dot * 20.0)  # smooth sign
    C_eff = C * (1.0 + (asymmetry - 1.0) * 0.5 * (1.0 - direction))

    # Virtual dynamics: J·θ̈ = K·(pedal − θ) − C·θ̇
    spring_force = K * (pedal - theta)
    damper_force = C_eff * theta_dot
    theta_ddot = (spring_force - damper_force) / J

    # Symplectic Euler: update velocity first, then position
    theta_dot_new = theta_dot + theta_ddot * dt
    theta_new = theta + theta_dot_new * dt

    # Virtual bump stops at [0, 1]: zero velocity at limits
    at_lower = theta_new <= 0.0
    at_upper = theta_new >= 1.0
    theta_new = jnp.clip(theta_new, 0.0, 1.0)
    # Kill velocity when hitting stops (inelastic collision with the virtual endstop)
    theta_dot_new = jnp.where(at_lower | at_upper, 0.0, theta_dot_new)

    return theta_new, theta_dot_new, theta_new


# ─────────────────────────────────────────────────────────────────────────────
# §4  Combined Throttle + Brake Step
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def impedance_step(
    state: ImpedanceState,
    throttle_raw: jax.Array,   # [0, 1] raw throttle pedal
    brake_raw: jax.Array,      # [0, 1] raw brake pedal
    dt: jax.Array,             # timestep [s]
    params: ImpedanceParams = ImpedanceParams(),
) -> tuple[ImpedanceState, jax.Array, jax.Array]:
    """
    Full impedance filter step for both pedals.

    Returns:
        new_state: updated ImpedanceState
        throttle_filtered: [0, 1] impedance-filtered throttle
        brake_filtered: [0, 1] impedance-filtered brake
    """
    # Throttle channel
    th_new, th_dot_new, throttle_out = _impedance_channel_step(
        state.theta_throttle, state.theta_dot_throttle,
        throttle_raw,
        params.J_throttle, params.C_throttle, params.K_throttle,
        dt, asymmetry=1.0, deadband=params.deadband,
    )

    # Brake channel (with asymmetry: apply fast, release slow)
    br_new, br_dot_new, brake_out = _impedance_channel_step(
        state.theta_brake, state.theta_dot_brake,
        brake_raw,
        params.J_brake, params.C_brake, params.K_brake,
        dt, asymmetry=params.asymmetry, deadband=params.deadband,
    )

    new_state = ImpedanceState(
        theta_throttle=th_new,
        theta_dot_throttle=th_dot_new,
        theta_brake=br_new,
        theta_dot_brake=br_dot_new,
    )

    return new_state, throttle_out, brake_out


# ─────────────────────────────────────────────────────────────────────────────
# §5  Frequency Response Analysis (offline diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def impedance_frequency_response(
    f_array,                        # Hz frequency array
    J: float = 0.05,
    C: float = 2.1,
    K: float = 45.0,
):
    """
    Analytical frequency response of the virtual impedance.

    Transfer function: G(s) = K / (J·s² + C·s + K)

    Returns:
        magnitude: |G(jω)| at each frequency
        phase_deg: ∠G(jω) in degrees at each frequency
        omega_n: natural frequency [rad/s]
        zeta: damping ratio
    """
    import numpy as np

    omega = 2.0 * np.pi * np.array(f_array)
    omega_n = np.sqrt(K / J)
    zeta = C / (2.0 * np.sqrt(K * J))

    # G(jω) = 1 / (1 − (ω/ωₙ)² + j·2ζ·ω/ωₙ)
    r = omega / omega_n
    denom_real = 1.0 - r ** 2
    denom_imag = 2.0 * zeta * r
    denom_mag = np.sqrt(denom_real ** 2 + denom_imag ** 2)

    magnitude = 1.0 / denom_mag
    phase_rad = -np.arctan2(denom_imag, denom_real)
    phase_deg = np.degrees(phase_rad)

    return {
        'frequency': np.array(f_array),
        'magnitude': magnitude,
        'phase_deg': phase_deg,
        'omega_n': omega_n,
        'zeta': zeta,
        'f_n': omega_n / (2.0 * np.pi),
    }