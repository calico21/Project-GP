import os
import sys
import numpy as np
import pandas as pd
import argparse

# --- JAX / XLA ENVIRONMENT SETUP ---
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7' 

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
    
    # 2. Optimal Control (The Final Diff-WMPC Upgrade)
    from optimization.ocp_solver import DiffWMPCSolver
    
    # 3. AI Setup & Coaching (The Final MORL-DB / SB-TRPO Upgrade)
    from optimization.evolutionary import MORL_SB_TRPO_Optimizer
    from telemetry.driver_coaching import ActorCriticPolicyEvaluator
except ImportError as e:
    print(f"[Critical Error] Architecture Import Failed: {e}")
    print("Ensure JAX, Flax, Optax, and Acados are properly installed in the WSL environment.")
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
    
    optimized_params = estimator.optimize_trajectory(initial_T, initial_w, gps_times, gps_meas, iterations=150)
    
    track_gen = ContinuousManifoldTrackGenerator(estimator, optimized_params)
    track_data = track_gen.generate(s_step=1.0) 
    
    return track_data, df_raw

def execute_stochastic_ghost_car(track_data):
    print("\n" + "="*60)
    print("PHASE 2: DIFF-WMPC STOCHASTIC GHOST CAR")
    print("="*60)
    
    # Updated to the Diff-WMPC Solver
    solver = DiffWMPCSolver()
    
    N_nodes = min(300, len(track_data['s']) - 1)
    idx = np.linspace(0, len(track_data['s'])-1, N_nodes+1).astype(int)
    
    try:
        result = solver.solve(
            track_s = track_data['s'][idx], 
            track_k = track_data['k'][idx], 
            track_w_left = track_data['w_left'][idx], 
            track_w_right = track_data['w_right'][idx], 
            friction_uncertainty_map = track_data['w_mu'][idx],
            N = N_nodes
        )
        
        if "error" in result:
            print(f"[Diff-WMPC] Solver Failed: {result['error']}")
            return None
        else:
            print(f"[Diff-WMPC] Nominal Lap Time: {result['time']:.3f} s")
            df = pd.DataFrame(result)
            df.to_csv(os.path.join(current_dir, 'stochastic_ghost_car.csv'), index=False)
            return df
            
    except Exception as e:
        print(f"[Diff-WMPC] Critical Failure: {e}")
        return None

def execute_ai_coaching(df_human, df_ghost):
    print("\n" + "="*60)
    print("PHASE 3: AC-MPC DRIVER & VEHICLE ADAPTATION")
    print("="*60)
    
    evaluator = ActorCriticPolicyEvaluator()
    report_df = evaluator.analyze_continuous_policy(df_human, df_ghost.to_dict(orient='list'))
    
    if not report_df.empty:
        print("\n[AC-MPC] Critical Intervention Zones:")
        print(report_df.sort_values('Critic_Advantage', ascending=True).head(5).to_string(index=False))
        
        out_file = os.path.join(current_dir, 'ac_mpc_coaching_report.csv')
        report_df.to_csv(out_file, index=False)
        print(f"\n[AC-MPC] Full compensation matrix saved to {out_file}")
    else:
        print("\n[AC-MPC] Driver policy aligns perfectly with Tube-MPC manifold.")

def execute_morl_setup():
    print("\n" + "="*60)
    print("PHASE 4: DYNAMIC AI SETUP DISCOVERY (MORL-DB & SB-TRPO)")
    print("="*60)
    
    # Updated to the new Reinforcement Learning Setup AI
    optimizer = MORL_SB_TRPO_Optimizer(ensemble_size=20, dim=7)
    pareto_setups, pareto_grips, pareto_stabs = optimizer.run(iterations=100)
    
    df = pd.DataFrame(pareto_setups, columns=optimizer.var_keys)
    df['Lat_G_Score'] = pareto_grips
    df['Stability_Overshoot'] = pareto_stabs
    
    print("\n[MORL-DB] Target-Fidelity Pareto Front Discovery Complete:")
    print(df.sort_values('Lat_G_Score', ascending=False).to_string(index=False))
    
    out_file = os.path.join(current_dir, 'morl_pareto_front.csv')
    df.to_csv(out_file, index=False)
    print(f"\n[Success] Full Pareto array saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Project-GP: End-to-End Differentiable Digital Twin")
    parser.add_argument('--mode', type=str, default='full', choices=['telemetry', 'ghost', 'coach', 'setup', 'full'])
    parser.add_argument('--log', type=str, default=None, help="Path to raw telemetry ASC/CSV")
    args = parser.parse_args()

    track_data = None
    df_human = None
    df_ghost = None

    if args.mode in ['telemetry', 'ghost', 'coach', 'full']:
        if not args.log:
            print("[Error] A telemetry log is required. Use --log <path>")
            sys.exit(1)
        track_data, df_human = execute_continuous_telemetry_pipeline(args.log)

    if args.mode in ['ghost', 'coach', 'full'] and track_data is not None:
        df_ghost = execute_stochastic_ghost_car(track_data)

    if args.mode in ['coach', 'full'] and df_human is not None and df_ghost is not None:
        execute_ai_coaching(df_human, df_ghost)

    if args.mode in ['setup', 'full']:
        execute_morl_setup()
        
    print("\n[System] Project-GP Digital Twin Execution Concluded Successfully.")

if __name__ == "__main__":
    main()