import os
import sys
import numpy as np
import pandas as pd
import argparse
import wandb
import flax.serialization

# --- JAX / XLA ENVIRONMENT SETUP ---
# Pre-allocate memory for the massive 46-DOF and Wavelet Collocation operations
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8' 

import jax
import jax.numpy as jnp

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- ADVANCED DIGITAL TWIN IMPORTS ---
try:
    # 1. Telemetry & Manifolds
    from telemetry.log_ingestion import LogIngestion
    from telemetry.filtering import ContinuousTimeTrajectoryEstimator, SE3Manifold
    from telemetry.track_generator import ContinuousManifoldTrackGenerator
    
    # 2. Optimal Control (46-DOF Native JAX Diff-WMPC)
    from optimization.ocp_solver import DiffWMPCSolver
    
    # 3. AI Setup & Coaching (Ghost Car AC & MORL-DB)
    from optimization.evolutionary import MORL_SB_TRPO_Optimizer
    from telemetry.driver_coaching import GhostCarEvaluator

    from optimization.residual_fitting import train_neural_residuals
except ImportError as e:
    print(f"[Critical Error] Architecture Import Failed: {e}")
    print("Ensure JAX, Flax, Optax, and WandB are properly installed in the hardware-accelerated environment.")
    sys.exit(1)


# =============================================================================
#  MASTER EXECUTION PIPELINE
# =============================================================================

def execute_continuous_telemetry_pipeline(log_path):
    print("\n" + "="*60)
    print("PHASE 1: CONTINUOUS SE(3) MANIFOLD & TRACK GENERATION")
    print("="*60)
    
    ingestor = LogIngestion(log_path)
    df_raw = ingestor.process()
    
    gps_times = jnp.array(df_raw['time'].values)
    gps_meas = jnp.array(df_raw[['x', 'y', 'z']].values)
    
    duration = float(gps_times[-1])
    dt_knot = 0.5 
    num_knots = int(duration / dt_knot) + 5
    
    estimator = ContinuousTimeTrajectoryEstimator(num_knots, dt_knot=dt_knot)
    
    initial_T = jnp.tile(jnp.eye(4), (num_knots, 1, 1))
    initial_w = jnp.zeros((num_knots, 6))
    
    # BUG 3 FIX: Gracefully handle IMU arguments with try/except
    try:
        if 'ax' in df_raw.columns and 'rx' in df_raw.columns:
            imu_times = jnp.array(df_raw['time'].values)
            accel = jnp.array(df_raw[['ax', 'ay', 'az']].values)
            gyro = jnp.array(df_raw[['rx', 'ry', 'rz']].values)
        else:
            imu_times = jnp.array([])
            accel = jnp.array([])
            gyro = jnp.array([])
            
        optimized_params = estimator.optimize_trajectory(
            initial_T, initial_w, 
            gps_times, gps_meas, 
            imu_times=imu_times, accel_measurements=accel, gyro_measurements=gyro,
            iterations=150
        )
    except TypeError:
        print("[TrackGen] Fallback: Estimator doesn't support IMU keyword arguments. Running GPS-only.")
        optimized_params = estimator.optimize_trajectory(
            initial_T, initial_w, 
            gps_times, gps_meas, 
            iterations=150
        )
    
    track_gen = ContinuousManifoldTrackGenerator(estimator, optimized_params)
    track_data = track_gen.generate(s_step=1.0) 
    
    return track_data, df_raw

def execute_stochastic_ghost_car(track_data, ai_cost_map=None):
    print("\n" + "="*60)
    print("PHASE 2: DIFF-WMPC STOCHASTIC GHOST CAR")
    print("="*60)
    
    # Horizon must be a power of 2 for the Haar Wavelet Basis Transform
    WAVELET_HORIZON = 128
    solver = DiffWMPCSolver(N_horizon=WAVELET_HORIZON)
    
    try:
        # The solver will now automatically handle padding/resizing internally
        # Passing exact XY coordinates to enable exact Frenet lateral deviation calculation
        result = solver.solve(
            track_s = track_data['s'], 
            track_k = track_data['k'],
            track_x = track_data['x'],
            track_y = track_data['y'],
            track_psi = track_data['psi'],
            track_w_left = track_data['w_left'], 
            track_w_right = track_data['w_right'], 
            friction_uncertainty_map = track_data.get('w_mu', None),
            ai_cost_map = ai_cost_map
        )
        
        print(f"[Diff-WMPC] Wavelet Manifold Optimization Complete. Nominal Lap Time: {result['time']:.3f} s")
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(current_dir, 'stochastic_ghost_car.csv'), index=False)
        return df
            
    except Exception as e:
        print(f"[Diff-WMPC] Critical XLA Compilation Failure: {e}")
        return None

def execute_ai_coaching(df_human, df_ghost):
    print("\n" + "="*60)
    print("PHASE 3: AC-MPC DRIVER & VEHICLE ADAPTATION")
    print("="*60)
    
    # Instantiating the upgraded JAX vmap Actor-Critic
    evaluator = GhostCarEvaluator()
    
    # The evaluator now returns both the human-readable report AND the active tensor weights
    report_df, active_cost_map = evaluator.evaluate_continuous_policy(df_human, df_ghost.to_dict(orient='list'))
    
    if not report_df.empty:
        print("\n[Ghost-Car AI] Critical Intervention Zones:")
        print(report_df.sort_values('Critic_Advantage', ascending=True).head(5).to_string(index=False))
        
        out_file = os.path.join(current_dir, 'ac_mpc_coaching_report.csv')
        report_df.to_csv(out_file, index=False)
        print(f"\n[Ghost-Car AI] Full compensation matrix saved to {out_file}")
        
        return active_cost_map
    else:
        print("\n[Ghost-Car AI] Driver policy aligns perfectly with Tube-MPC manifold. No active weight adjustments needed.")
        return None

def execute_morl_setup():
    print("\n" + "="*60)
    print("PHASE 4: DYNAMIC AI SETUP DISCOVERY (MORL-DB & SB-TRPO)")
    print("="*60)
    
    # Initialize W&B Experiment Tracking
    wandb.init(project="Project-GP-Digital-Twin", name="MORL_Pareto_Evolution", config={"ensemble_size": 20})

    # Evolves populations natively through the 46-DOF Port-Hamiltonian Engine
    optimizer = MORL_SB_TRPO_Optimizer(ensemble_size=20, dim=7)
    pareto_setups, pareto_grips, pareto_stabs = optimizer.run(iterations=100)
    
    # Log final metrics to W&B
    wandb.log({
        "Max_Grip_Found": np.max(pareto_grips), 
        "Max_Stability_Found": np.max(pareto_stabs)
    })
    wandb.finish()

    df = pd.DataFrame(pareto_setups, columns=optimizer.var_keys)
    df['Lat_G_Score'] = pareto_grips
    df['Stability_Overshoot'] = pareto_stabs
    
    print("\n[MORL-DB] 46-DOF Target-Fidelity Pareto Front Discovery Complete:")
    print(df.sort_values('Lat_G_Score', ascending=False).to_string(index=False))
    
    out_file = os.path.join(current_dir, 'morl_pareto_front.csv')
    df.to_csv(out_file, index=False)
    print(f"\n[Success] Full 46-DOF Pareto array saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Project-GP: End-to-End Differentiable Digital Twin")
    parser.add_argument('--mode', type=str, default='full', choices=['pretrain', 'telemetry', 'ghost', 'coach', 'setup', 'full', 'closed_loop'])
    parser.add_argument('--log', type=str, default=None, help="Path to raw telemetry ASC/CSV")
    args = parser.parse_args()

    # Pre-train the neural physics networks and serialize the weights
    if args.mode == 'pretrain':
        print("\n" + "="*60 + "\nPHASE 0: NEURAL PHYSICS PRE-TRAINING\n" + "="*60)
        h_params, r_params = train_neural_residuals()
        
        if h_params is not None and r_params is not None:
            models_dir = os.path.join(current_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            with open(os.path.join(models_dir, 'h_net.bytes'), 'wb') as f: 
                f.write(flax.serialization.to_bytes(h_params))
            with open(os.path.join(models_dir, 'r_net.bytes'), 'wb') as f: 
                f.write(flax.serialization.to_bytes(r_params))
            print("[System] Neural weights successfully serialized.")
        sys.exit(0)

    track_data = None
    df_human = None
    df_ghost = None
    active_cost_map = None

    if args.mode in ['telemetry', 'ghost', 'coach', 'full', 'closed_loop']:
        if not args.log:
            print("[Error] A telemetry log is required. Use --log <path>")
            sys.exit(1)
        track_data, df_human = execute_continuous_telemetry_pipeline(args.log)

    if args.mode in ['ghost', 'coach', 'full', 'closed_loop'] and track_data is not None:
        # Base solve using default MPC weights
        df_ghost = execute_stochastic_ghost_car(track_data)

    if args.mode in ['coach', 'full', 'closed_loop'] and df_human is not None and df_ghost is not None:
        # Ghost Car Actor-Critic evaluates human and outputs new active setup weights
        active_cost_map = execute_ai_coaching(df_human, df_ghost)

    if args.mode == 'closed_loop' and active_cost_map is not None:
        print("\n" + "="*60)
        print("PHASE 3.5: CLOSED-LOOP MPC RE-SOLVE WITH AI WEIGHTS")
        print("="*60)
        # Feed the AI's dynamically generated weights directly back into the Diff-WMPC solver
        df_ghost_adapted = execute_stochastic_ghost_car(track_data, ai_cost_map=active_cost_map)
        df_ghost_adapted.to_csv(os.path.join(current_dir, 'stochastic_ghost_car_adapted.csv'), index=False)

    if args.mode in ['setup', 'full', 'closed_loop']:
        execute_morl_setup()
        
    print("\n[System] Project-GP Digital Twin Execution Concluded Successfully.")

if __name__ == "__main__":
    main()