import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import argparse
import wandb
import flax.serialization

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

import jax
import jax.numpy as jnp

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.jax_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
jax.config.update('jax_compilation_cache_dir', _CACHE_DIR)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 5.0)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(PROJECT_ROOT, 'models')
H_NET_PATH   = os.path.join(MODEL_DIR, 'h_net.bytes')
R_NET_PATH   = os.path.join(MODEL_DIR, 'r_net.bytes')

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from telemetry.log_ingestion import LogIngestion
    from telemetry.filtering import ContinuousTimeTrajectoryEstimator, SE3Manifold
    from telemetry.track_generator import ContinuousManifoldTrackGenerator

    from optimization.ocp_solver import DiffWMPCSolver
    from optimization.evolutionary import MORL_SB_TRPO_Optimizer
    from telemetry.driver_coaching import GhostCarEvaluator

    from optimization.residual_fitting import train_neural_residuals
except ImportError as e:
    print(f"[Critical Error] Architecture import failed: {e}")
    print("[Traceback]:")
    traceback.print_exc()
    sys.exit(1)


# =============================================================================
#  PIPELINE PHASES
# =============================================================================

def execute_continuous_telemetry_pipeline(log_path):
    print("\n" + "=" * 60)
    print("PHASE 1: CONTINUOUS SE(3) MANIFOLD & TRACK GENERATION")
    print("=" * 60)

    ingestor = LogIngestion(log_path)
    df_raw   = ingestor.process()

    gps_times = jnp.array(df_raw['time'].values)
    gps_meas  = jnp.array(df_raw[['x', 'y', 'z']].values)

    duration  = float(gps_times[-1])
    dt_knot   = 0.5
    num_knots = int(duration / dt_knot) + 5

    estimator = ContinuousTimeTrajectoryEstimator(num_knots, dt_knot=dt_knot)
    initial_T = jnp.tile(jnp.eye(4), (num_knots, 1, 1))
    initial_w = jnp.zeros((num_knots, 6))

    try:
        if 'ax' in df_raw.columns and 'rx' in df_raw.columns:
            imu_times = jnp.array(df_raw['time'].values)
            accel     = jnp.array(df_raw[['ax', 'ay', 'az']].values)
            gyro      = jnp.array(df_raw[['rx', 'ry', 'rz']].values)
        else:
            imu_times = jnp.array([])
            accel     = jnp.array([])
            gyro      = jnp.array([])

        optimized_params = estimator.optimize_trajectory(
            initial_T, initial_w, gps_times, gps_meas,
            imu_times=imu_times, accel_measurements=accel,
            gyro_measurements=gyro, iterations=150,
        )
    except TypeError:
        print("[TrackGen] Fallback: running GPS-only trajectory estimation.")
        optimized_params = estimator.optimize_trajectory(
            initial_T, initial_w, gps_times, gps_meas, iterations=150,
        )

    track_gen  = ContinuousManifoldTrackGenerator(estimator, optimized_params)
    track_data = track_gen.generate(s_step=1.0)
    return track_data, df_raw


def execute_stochastic_ghost_car(track_data, ai_cost_map=None, wavelet_horizon=128):
    print("\n" + "=" * 60)
    print("PHASE 2: DIFF-WMPC STOCHASTIC GHOST CAR")
    print("=" * 60)

    solver = DiffWMPCSolver(N_horizon=wavelet_horizon)
    try:
        result = solver.solve(
            track_s=track_data['s'], track_k=track_data['k'],
            track_x=track_data['x'], track_y=track_data['y'],
            track_psi=track_data['psi'],
            track_w_left=track_data['w_left'],
            track_w_right=track_data['w_right'],
            friction_uncertainty_map=track_data.get('w_mu'),
            ai_cost_map=ai_cost_map,
        )
        print(f"[Diff-WMPC] Optimisation complete. Nominal lap time: {result['time']:.3f} s")
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(PROJECT_ROOT, 'stochastic_ghost_car.csv'), index=False)
        return df

    except Exception as e:
        is_jax_error = (
            'xla' in str(e).lower() or 'jax' in type(e).__name__.lower()
            or 'tracer' in str(e).lower()
        )
        if is_jax_error:
            print(f"[Diff-WMPC] XLA error: {e}")
            return None
        print(f"[Diff-WMPC] Unexpected error:")
        traceback.print_exc()
        raise


def execute_ai_coaching(df_human, df_ghost):
    print("\n" + "=" * 60)
    print("PHASE 3: AC-MPC DRIVER & VEHICLE ADAPTATION")
    print("=" * 60)

    evaluator = GhostCarEvaluator()
    report_df, active_cost_map = evaluator.evaluate_continuous_policy(
        df_human, df_ghost.to_dict(orient='list')
    )

    if not report_df.empty:
        print("\n[Ghost-Car AI] Critical intervention zones:")
        print(report_df.sort_values('Critic_Advantage', ascending=True)
                       .head(5).to_string(index=False))
        out_file = os.path.join(PROJECT_ROOT, 'ac_mpc_coaching_report.csv')
        report_df.to_csv(out_file, index=False)
        print(f"\n[Ghost-Car AI] Full report saved to {out_file}")
        return active_cost_map
    else:
        print("\n[Ghost-Car AI] Driver policy matches Tube-MPC manifold perfectly.")
        return None


def execute_morl_setup(wandb_run=None, iterations=1000):
    """
    P27: wandb_run is now passed into the optimizer's run() method so that
    every phase of logging (per-iteration, per-member, restart events, and
    final Pareto/sensitivity tables) is driven from a single run object.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: DYNAMIC AI SETUP DISCOVERY (MORL-SB-TRPO)")
    print("=" * 60)

    optimizer = MORL_SB_TRPO_Optimizer(ensemble_size=20, dim=8)

    # P27: pass wandb_run into run() so per-iter diagnostics are logged there
    pareto_setups, pareto_grips, pareto_stabs, pareto_gen = optimizer.run(
        iterations=iterations,
        wandb_run=wandb_run,
    )

    # P27: post-run summary metrics (final-state scalars for the run summary)
    if wandb_run is not None:
        wandb_run.summary["Max_Grip_Found"]      = float(np.max(pareto_grips))
        wandb_run.summary["Max_Stability_Found"] = float(np.max(pareto_stabs))
        wandb_run.summary["Pareto_Front_Count"]  = int(len(pareto_grips))

        # P27: Pareto front scatter as a W&B custom chart
        try:
            scatter_data = [[float(g), float(s), int(gen)]
                            for g, s, gen in zip(pareto_grips, pareto_stabs, pareto_gen)]
            scatter_table = wandb.Table(
                data=scatter_data,
                columns=["Grip_G", "Stability_Overshoot", "Generation"]
            )
            wandb_run.log({
                "Pareto_Scatter": wandb.plot.scatter(
                    scatter_table,
                    x="Grip_G",
                    y="Stability_Overshoot",
                    title="Pareto Front: Grip vs Stability (coloured by Generation)",
                )
            })
        except Exception as e:
            print(f"[P27] Pareto scatter plot failed: {e}")

    df = pd.DataFrame(pareto_setups, columns=optimizer.var_keys)
    df['Lat_G_Score']         = pareto_grips
    df['Stability_Overshoot'] = pareto_stabs
    df['Generation']          = pareto_gen

    print("\n[MORL-SB-TRPO] Pareto front discovery complete:")
    print(df.sort_values('Lat_G_Score', ascending=False).to_string(index=False))

    out_file = os.path.join(PROJECT_ROOT, 'morl_pareto_front.csv')
    df.to_csv(out_file, index=False)
    print(f"\n[Success] Pareto array saved to {out_file}")


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Project-GP: End-to-End Differentiable Digital Twin"
    )
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=['pretrain', 'telemetry', 'ghost', 'coach', 'setup',
                 'full', 'closed_loop'],
    )
    parser.add_argument('--log',        type=str, default=None,
                        help="Path to raw telemetry ASC/CSV")
    parser.add_argument('--horizon',    type=int, default=128,
                        help="Wavelet horizon (must be power of 2)")
    parser.add_argument('--iterations', type=int, default=1500,
                        help="MORL-SB-TRPO iteration count")
    args = parser.parse_args()

    # W&B initialisation
    wandb_run = None
    if args.mode in ['pretrain', 'setup', 'full', 'closed_loop']:
        # P27: pretrain mode now also gets a run so H_net metrics can be logged
        wandb_run = wandb.init(
            project="Project-GP-Digital-Twin",
            name=f"run_{args.mode}_{int(time.time())}",
            config={
                **vars(args),
                # P27: log architecture constants alongside run config
                "ensemble_size":    20,
                "stability_max":    5.0,
                "kl_lag_horizon":   10,
                "restart_interval": 200,
                "n_restart":        5,
                "bo_n_init":        5,
                "bo_n_iters":       25,
            },
        )

    # ── Pre-train neural physics residuals ───────────────────────────────────
    if args.mode == 'pretrain':
        print("\n" + "=" * 60 + "\nPHASE 0: NEURAL PHYSICS PRE-TRAINING\n" + "=" * 60)

        # P27: wrap train_neural_residuals to capture epoch-level metrics if available
        h_params, r_params = train_neural_residuals()

        if h_params is not None and r_params is not None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(H_NET_PATH, 'wb') as f:
                f.write(flax.serialization.to_bytes(h_params))
            with open(R_NET_PATH, 'wb') as f:
                f.write(flax.serialization.to_bytes(r_params))
            print(f"[System] Neural weights saved to {MODEL_DIR}")

            # P27: log training metadata to W&B
            if wandb_run is not None:
                try:
                    # Attempt to read the passivity scale saved by residual_fitting
                    scale_path = os.path.join(MODEL_DIR, 'h_net_scale.txt')
                    if os.path.exists(scale_path):
                        with open(scale_path) as f:
                            h_scale = float(f.read().strip())
                        wandb_run.log({"pretrain/h_net_passivity_scale_J": h_scale})

                    # Log the final MSE if residual_fitting exposes it
                    import optimization.residual_fitting as _rf
                    if hasattr(_rf, 'LAST_TRAIN_MSE'):
                        wandb_run.log({"pretrain/h_net_final_mse": _rf.LAST_TRAIN_MSE})
                    if hasattr(_rf, 'LAST_PHRATE'):
                        wandb_run.log({"pretrain/h_net_final_phrate": _rf.LAST_PHRATE})

                    wandb_run.summary["pretrain/complete"] = True
                    print("[P27] Pre-train metrics logged to W&B.")
                except Exception as e:
                    print(f"[P27] Pre-train W&B logging failed: {e}")

        if wandb_run:
            wandb_run.finish()
        sys.exit(0)

    # ── Telemetry & track generation ─────────────────────────────────────────
    track_data = None
    df_human   = None
    df_ghost   = None

    if args.mode in ['telemetry', 'ghost', 'coach', 'full', 'closed_loop']:
        if not args.log:
            print("[Error] A telemetry log is required. Use --log <path>")
            if wandb_run:
                wandb_run.finish()
            sys.exit(1)
        track_data, df_human = execute_continuous_telemetry_pipeline(args.log)

        # P27: log track summary to W&B
        if wandb_run is not None and track_data is not None:
            try:
                wandb_run.log({
                    "track/total_distance_m": float(track_data['s'][-1]),
                    "track/n_waypoints":      int(len(track_data['s'])),
                    "track/max_curvature":    float(np.max(np.abs(track_data['k']))),
                })
            except Exception:
                pass

    if args.mode in ['ghost', 'coach', 'full', 'closed_loop'] and track_data is not None:
        df_ghost = execute_stochastic_ghost_car(
            track_data, wavelet_horizon=args.horizon
        )
        # P27: log ghost car lap time to W&B
        if wandb_run is not None and df_ghost is not None:
            try:
                lap_time = float(df_ghost['time'].iloc[0]) if 'time' in df_ghost.columns else None
                if lap_time:
                    wandb_run.log({"ghost/nominal_lap_time_s": lap_time})
                wandb_run.log({
                    "ghost/mean_speed_mps": float(df_ghost['v'].mean()),
                    "ghost/max_lat_g":      float(df_ghost['lat_g'].max()),
                })
            except Exception:
                pass

    if (args.mode in ['coach', 'full', 'closed_loop']
            and df_human is not None and df_ghost is not None):
        active_cost_map = execute_ai_coaching(df_human, df_ghost)
    else:
        active_cost_map = None

    if args.mode == 'closed_loop' and active_cost_map is not None:
        print("\n" + "=" * 60)
        print("PHASE 3.5: CLOSED-LOOP MPC RE-SOLVE WITH AI WEIGHTS")
        print("=" * 60)
        df_ghost_adapted = execute_stochastic_ghost_car(
            track_data, ai_cost_map=active_cost_map,
            wavelet_horizon=args.horizon,
        )
        if df_ghost_adapted is not None:
            df_ghost_adapted.to_csv(
                os.path.join(PROJECT_ROOT, 'stochastic_ghost_car_adapted.csv'),
                index=False,
            )

    if args.mode in ['setup', 'full', 'closed_loop']:
        execute_morl_setup(wandb_run=wandb_run, iterations=args.iterations)

    if wandb_run:
        wandb_run.finish()

    print("\n[System] Project-GP execution concluded successfully.")


if __name__ == "__main__":
    main()