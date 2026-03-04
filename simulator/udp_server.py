import os
import sys
import time
import socket
import struct
import numpy as np
import jax
import jax.numpy as jnp
from jax import remat

# 1. Add project root to sys.path so we can import Project-GP modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vehicle_dynamics import DifferentiableMultiBodyVehicle
from data.configs.vehicle_params import vehicle_params as VP_DICT
from data.configs.tire_coeffs import tire_coeffs as TP_DICT

# --- CONFIGURATION ---
HOST = '127.0.0.1'
PORT_RECEIVE = 5000  
PORT_SEND = 5001     
PHYSICS_HZ = 200
DT = 1.0 / PHYSICS_HZ
SUBSTEPS = 5          # Substep to maintain symplectic stability
DT_SUB = DT / SUBSTEPS

def main():
    print("="*50)
    print(" Project-GP: Headless JAX Physics Server ")
    print("="*50)

    print("[1/4] Initializing DifferentiableMultiBodyVehicle...")
    vehicle = DifferentiableMultiBodyVehicle(VP_DICT, TP_DICT)
    
    setup_params = jnp.array([40000.0, 40000.0, 500.0, 500.0, 3000.0, 3000.0, 0.3, 0.60])
    
    # --- WARM STATE INITIALIZATION ---
    current_state = jnp.zeros(46)
    START_SPEED = 5.0
    TIRE_RADIUS = VP_DICT.get('tire_radius', 0.23) 
    OMEGA_START = START_SPEED / TIRE_RADIUS 
    
    current_state = current_state.at[14].set(START_SPEED) # STATE_VX
    current_state = current_state.at[24:28].set(OMEGA_START) # Wheel RPM
    
    # FIX: Spawn the car at equilibrium ride height so it doesn't explode into the air
    current_state = current_state.at[2].set(0.30)     # Chassis Z (approx CG height)
    current_state = current_state.at[6:10].set(0.23)  # Wheel Z (resting on the ground)
    
    print("[2/4] AOT Compiling Physics Step via jax.jit (Please wait...)")
    
    @jax.jit
    def fast_step(state, controls):
        # Implement the same safe substep loop used in ocp_solver.py
        @remat
        def substep(x_s, _):
            return vehicle.simulate_step(x_s, controls, setup_params, DT_SUB), None

        next_state, _ = jax.lax.scan(substep, state, None, length=SUBSTEPS)
        
        # NaN Guard: If it explodes, return the previous safe state
        next_state = jnp.where(jnp.isfinite(next_state[14]), next_state, state)
        return next_state
    
    # Trigger compiler with a 2-element array [steer, force]
    dummy_controls = jnp.array([0.0, 0.0]) 
    _ = fast_step(current_state, dummy_controls)
    print("[2/4] Compilation Complete. JAX graph is hot.")

    print(f"[3/4] Binding UDP Sockets (Rx: {PORT_RECEIVE}, Tx: {PORT_SEND})")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT_RECEIVE))
    sock.setblocking(False)

    rx_fmt = '<3f'
    tx_fmt = '<11f'

    print(f"[4/4] Entering {PHYSICS_HZ}Hz Real-Time Loop. Waiting for UE5...")
    
    controls = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    frame_count = 0

    try:
        while True:
            loop_start = time.perf_counter()

            try:
                data, _ = sock.recvfrom(12)
                unpacked = struct.unpack(rx_fmt, data)
                controls[0] = unpacked[0] # Steer
                controls[1] = unpacked[1] # Throttle Force
                controls[2] = unpacked[2] # Brake Force
            except BlockingIOError:
                pass
            
            # Combine throttle and brake into net longitudinal force
            net_longitudinal_force = controls[1] - controls[2]
            jax_controls = jnp.array([controls[0], net_longitudinal_force])
            
            current_state = fast_step(current_state, jax_controls)
            
            x, y, z = current_state[0], current_state[1], current_state[2]
            roll, pitch, yaw = current_state[3], current_state[4], current_state[5]
            z_fl, z_fr, z_rl, z_rr = current_state[6], current_state[7], current_state[8], current_state[9]
            mz_aligning = 0.0 

            tx_data = struct.pack(tx_fmt, float(x), float(y), float(z), 
                                  float(roll), float(pitch), float(yaw), 
                                  float(z_fl), float(z_fr), float(z_rl), float(z_rr), 
                                  float(mz_aligning))
            sock.sendto(tx_data, (HOST, PORT_SEND))

            # Debug print every 100 frames to verify the server is updating
            if frame_count % 100 == 0:
                print(f"Server Internal -> Speed: {current_state[14]:.2f} m/s | X: {x:.2f} | Y: {y:.2f}")
            frame_count += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[System] Simulator Server safely terminated.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()