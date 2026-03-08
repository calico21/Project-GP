#!/usr/bin/env python3
"""
simulator/debug_physics.py
─────────────────────────────────────────────────────────────────────────────
Advanced diagnostics to isolate lateral mass scaling and brake failures.
"""

import sys, os, math
import numpy as np
import jax.numpy as jnp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP
from data.configs.tire_coeffs import tire_coeffs as TC

def main():
    print("=" * 60)
    print(" PROJECT-GP: ADVANCED PHYSICS DIAGNOSTICS")
    print("=" * 60)

    vd = DifferentiableMultiBodyVehicle(VP, TC)
    setup = jnp.array([40000., 40000., 500., 500., 3000., 3000., 0.30, 0.60])

    print("\n[1] MASS & INERTIA MATRIX CHECK")
    print(f"  Total Vehicle Mass (m) : {vd.m} kg")
    print(f"  Sprung Mass (m_s)      : {vd.m_s} kg")
    print(f"  M_diag[0] (vx mass)    : {vd.M_diag[0]} kg  <-- SHOULD BE {vd.m}")
    print(f"  M_diag[1] (vy mass)    : {vd.M_diag[1]} kg  <-- SHOULD BE {vd.m}")
    if vd.M_diag[1] < vd.m:
        print("  >>> ERROR: Lateral force (vy) is only accelerating the sprung mass!")
        print("  >>> Consequence: The car accelerates laterally ~22% too fast, snapping into slides.")

    print("\n[2] BRAKE FORCE ROUTING CHECK")
    ctrl_brake = jnp.array([0.0, -8000.0]) # 0 steer, Max brake
    state = jnp.zeros(46).at[14].set(20.0).at[2].set(0.0)
    dx = vd._compute_derivatives(state, ctrl_brake, setup)
    accel_x = dx[14]
    
    print(f"  Initial Speed   : 20.0 m/s")
    print(f"  Brake Command   : -8000 N")
    print(f"  Longitudinal Ax : {accel_x:.2f} m/s^2")
    if accel_x > -1.0:
        print("  >>> ERROR: The car is barely decelerating!")
        print("  >>> Consequence: Fx_brake_f/r are calculated but never added to F_ext_mech[14]. You have no brakes.")

    print("\n[3] STEERING ACKERMANN CHECK")
    print("  Look at lines 209-210 in vehicle_dynamics.py:")
    print("    delta_fl = delta - d_ack")
    print("    delta_fr = delta + d_ack")
    print("  >>> ERROR: Anti-Ackermann! The inside wheel turns LESS than the outside wheel.")
    print("  >>> Consequence: The front tires fight each other during cornering, breaking traction.")

    print("\n[4] DOUBLE-COUNTING UNSPRUNG MASS")
    print("  Look at lines 203-206 in vehicle_dynamics.py:")
    print("    Fz_fl = ... (W_total * self.lr / (L * 2) ... + W_us_f / 2)")
    print("  >>> ERROR: W_total already includes the unsprung mass. You are adding it twice.")

if __name__ == "__main__":
    main()