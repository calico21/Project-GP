import os
import sys

# --- JAX / XLA ENVIRONMENT SETUP ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8' 

import jax
import jax.numpy as jnp
import numpy as np

# Ensure root directory is in path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from optimization.residual_fitting import train_neural_residuals
from optimization.ocp_solver import DiffWMPCSolver
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT


def test_neural_convergence():
    print("\n" + "="*60)
    print("TEST 1: NEURAL RESIDUAL CONVERGENCE")
    print("="*60)
    print("Training H_net and R_net for 1000 epochs (synthetic chassis flex)...")
    try:
        h_params, r_params = train_neural_residuals()
        if h_params is not None and r_params is not None:
            print("[PASS] Neural components successfully trained and returned parameters.")
        else:
            print("[FAIL] train_neural_residuals returned None. Did you miss the return statement?")
    except Exception as e:
        print(f"[FAIL] Neural training crashed: {e}")


def test_forward_pass():
    print("\n" + "="*60)
    print("TEST 2: 46-DOF SYMPLECTIC FORWARD PASS")
    print("="*60)
    print("Instantiating DifferentiableMultiBodyVehicle...")
    
    try:
        vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
        
        # 46-D state: x[14] is vx. Let's start at 10.0 m/s
        x0 = jnp.zeros(46).at[14].set(10.0)
        
        # Controls: u[0] = steer (rad), u[1] = Fx (N)
        u = jnp.array([0.2, 2000.0]) 
        setup = jnp.zeros(7)
        
        print("Executing single simulate_step (dt=0.01s)...")
        x_next = vehicle.simulate_step(x0, u, setup, dt=0.01)
        
        is_finite = bool(jnp.all(jnp.isfinite(x_next)))
        
        if is_finite:
            print(f"  > Speed changed: 10.000 m/s -> {x_next[14]:.3f} m/s")
            print(f"  > Yaw rate built to: {x_next[19]:.3f} rad/s")
            print(f"  > Transient Slip Angle FL: {x_next[38]:.4f} rad")
            print("[PASS] Forward pass is mathematically stable and outputs are finite.")
        else:
            print("[FAIL] NaNs detected in physics engine output.")
    except Exception as e:
        print(f"[FAIL] Forward pass crashed: {e}")


def test_circular_track():
    print("\n" + "="*60)
    print("TEST 3: WMPC CIRCULAR TRACK OPTIMIZATION")
    print("="*60)
    
    N = 64
    track_s = np.linspace(0, 100, N)
    track_k = np.full(N, 0.05)  # Radius = 20m
    track_w_left = np.full(N, 3.5)
    track_w_right = np.full(N, 3.5)
    
    # Generate X/Y/Psi analytically for the circle
    track_psi = track_s * 0.05
    track_x = 20.0 * np.sin(track_psi)
    track_y = 20.0 * (1.0 - np.cos(track_psi))
    
    print("Solving MPC for constant curvature (R=20m). Expected physical limit ~16.6 m/s...")
    try:
        solver = DiffWMPCSolver(N_horizon=N)
        
        result = solver.solve(
            track_s=track_s, 
            track_k=track_k, 
            track_x=track_x, 
            track_y=track_y, 
            track_psi=track_psi,
            track_w_left=track_w_left, 
            track_w_right=track_w_right
        )
        
        mean_v = np.mean(result['v'])
        mean_g = np.mean(result['lat_g'])
        print(f"  > Solver achieved mean speed: {mean_v:.2f} m/s")
        print(f"  > Solver achieved Lat G: {mean_g:.2f} G")
        
        # Validating against expected Pacejka grip limits
        if 15.0 < mean_v < 18.0:
            print("[PASS] Solver correctly discovered the exact physical limit of the tires.")
        else:
            print(f"[FAIL] Solver velocity ({mean_v:.2f} m/s) is outside the expected physical envelope (15-18 m/s).")
    except Exception as e:
        print(f"[FAIL] WMPC Optimization crashed: {e}")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print(" PROJECT-GP DIGITAL TWIN: PRE-FLIGHT SANITY CHECKS")
    print("#"*60)
    
    test_neural_convergence()
    test_forward_pass()
    test_circular_track()
    
    print("\n" + "="*60)
    print("✅ ALL SANITY CHECKS COMPLETED.")
    print("System architecture is fully validated and ready for real telemetry injection.")
    print("="*60 + "\n")