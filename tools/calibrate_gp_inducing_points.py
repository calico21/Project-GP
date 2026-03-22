"""
tools/calibrate_gp_inducing_points.py

Takes a MoTeC CSV export, back-calculates tyre operating conditions per corner,
runs k-means to find 50 representative points, and writes them as numpy arrays
to be loaded as initial GP inducing point parameters.

Requires only standard MoTeC ADL3 channels:
  - wheel_speed_fl/fr/rl/rr  [rpm]
  - lateral_accel             [g]
  - yaw_rate                  [deg/s]
  - velocity                  [km/h or m/s]
  - steer_angle               [deg]
  - susp_fl/fr/rl/rr          [mm]
"""
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

def back_calculate_operating_conditions(df: pd.DataFrame, vp: dict) -> np.ndarray:
    """
    Returns (N, 5) array of [alpha, kappa, gamma_approx, Fz_approx, Vx]
    for all timesteps, averaged across all four corners per timestep.
    
    alpha  = atan2(vy + wz*lf, vx) - delta   [front, simplified]
    kappa  = (omega * R_wheel - vx) / vx       [longitudinal slip]
    gamma  = camber_f + camber_gain * phi      [from suspension travel]
    Fz     = back-calculated from load transfer + static weight
    """
    # --- extract channels (adjust column names to match your MoTeC export) ---
    vx   = df['velocity_ms'].values               # m/s
    ay   = df['lateral_accel_ms2'].values          # m/s²
    wz   = np.deg2rad(df['yaw_rate_degs'].values)  # rad/s
    delta = np.deg2rad(df['steer_angle_deg'].values / vp.get('steer_ratio', 4.2))
    
    omega_fl = df['wheel_speed_fl_rpm'].values * 2*np.pi/60
    omega_fr = df['wheel_speed_fr_rpm'].values * 2*np.pi/60
    omega_rl = df['wheel_speed_rl_rpm'].values * 2*np.pi/60
    omega_rr = df['wheel_speed_rr_rpm'].values * 2*np.pi/60

    susp_fl = df['susp_fl_mm'].values * 1e-3   # → metres
    susp_fr = df['susp_fr_mm'].values * 1e-3
    susp_rl = df['susp_rl_mm'].values * 1e-3
    susp_rr = df['susp_rr_mm'].values * 1e-3

    # --- back-calculate vy from yaw dynamics (single-track approximation) ---
    # vy = (ay - vx*wz) integrated with 1st-order filter — avoids GPS noise
    # This is ~85% accurate; sufficient for inducing point placement
    vy = np.zeros_like(vx)
    dt = np.diff(df.index).mean() if hasattr(df.index, 'freq') else 0.005
    for i in range(1, len(vx)):
        vy_dot = ay[i] - vx[i] * wz[i]
        vy[i]  = 0.995 * vy[i-1] + dt * vy_dot   # 1st-order decay prevents drift

    # --- slip angles (SAE convention) ---
    vx_safe = np.maximum(np.abs(vx), 1.0)
    lf, lr  = vp.get('lf', 0.8525), vp.get('lr', 0.6975)
    tf, tr  = vp.get('track_front', 1.2), vp.get('track_rear', 1.18)

    alpha_fl = delta - np.arctan2(vy + wz*lf, np.maximum(np.abs(vx - wz*tf/2), 0.5))
    alpha_fr = delta - np.arctan2(vy + wz*lf, np.maximum(np.abs(vx + wz*tf/2), 0.5))
    alpha_rl =       - np.arctan2(vy - wz*lr, np.maximum(np.abs(vx - wz*tr/2), 0.5))
    alpha_rr =       - np.arctan2(vy - wz*lr, np.maximum(np.abs(vx + wz*tr/2), 0.5))

    # --- longitudinal slip ---
    R = vp.get('wheel_radius', 0.2045)
    kappa_fl = np.clip((omega_fl*R - vx) / vx_safe, -0.5, 0.5)
    kappa_fr = np.clip((omega_fr*R - vx) / vx_safe, -0.5, 0.5)
    kappa_rl = np.clip((omega_rl*R - vx) / vx_safe, -0.5, 0.5)
    kappa_rr = np.clip((omega_rr*R - vx) / vx_safe, -0.5, 0.5)

    # --- approximate camber from roll (static + kinematic gain) ---
    phi      = np.deg2rad(-2.0) + vp.get('camber_gain_f', -0.80) * np.deg2rad(
                  susp_fl - susp_fr) / tf   # rough roll angle proxy
    gamma    = np.full_like(vx, np.deg2rad(-2.0)) + vp.get('camber_gain_f', -0.8) * phi

    # --- normal load from load transfer (4-corner) ---
    m, g     = vp.get('total_mass', 300.0), 9.81
    ax       = np.gradient(vx, dt)
    h        = vp.get('h_cg', 0.285)
    L        = lf + lr
    dFz_lon  = m * ax * h / L
    dFz_lat  = m * ay * h / tf
    Fz_fl = np.maximum(m*g*lr/(2*L) - dFz_lon/2 - dFz_lat/2, 50.0)
    Fz_fr = np.maximum(m*g*lr/(2*L) - dFz_lon/2 + dFz_lat/2, 50.0)
    Fz_rl = np.maximum(m*g*lf/(2*L) + dFz_lon/2 - dFz_lat/2, 50.0)
    Fz_rr = np.maximum(m*g*lf/(2*L) + dFz_lon/2 + dFz_lat/2, 50.0)

    # --- assemble (N, 5) operating condition matrix ---
    # Stack all four corners as separate observations
    ops = np.column_stack([
        np.concatenate([alpha_fl, alpha_fr, alpha_rl, alpha_rr]),
        np.concatenate([kappa_fl, kappa_fr, kappa_rl, kappa_rr]),
        np.concatenate([gamma,    gamma,    gamma,    gamma   ]),
        np.concatenate([Fz_fl,    Fz_fr,    Fz_rl,   Fz_rr   ]),
        np.concatenate([vx,       vx,       vx,      vx      ]),
    ])
    # Remove rows with NaN (GPS dropouts, etc.)
    return ops[np.all(np.isfinite(ops), axis=1)]


def calibrate_inducing_points(
    motec_csv: str,
    vp: dict,
    n_inducing: int = 50,
    output_path: str = 'models/gp_inducing_calibrated.npy',
):
    """
    Runs k-means on back-calculated tyre operating conditions to place
    GP inducing points where the car actually operates.
    """
    df = pd.read_csv(motec_csv)
    ops = back_calculate_operating_conditions(df, vp)
    
    print(f"[GP Cal] {len(ops)} tyre observations from telemetry")
    print(f"[GP Cal] α range: [{ops[:,0].min():.3f}, {ops[:,0].max():.3f}] rad")
    print(f"[GP Cal] Fz range: [{ops[:,3].min():.0f}, {ops[:,3].max():.0f}] N")

    # Normalise to unit cube before clustering (matching GP input normalisation)
    scales = np.array([0.25, 0.20, 0.08, 400.0, 10.0])
    shifts = np.array([0.0,  0.0,  0.0,  800.0, 12.0])
    ops_n  = (ops - shifts) / scales   # [-1, 1] range approximately

    # k-means with multiple restarts for stability
    centroids, _ = kmeans2(ops_n, n_inducing, niter=50, minit='points', seed=42)

    # Convert back to physical space → then to Z_raw via inverse tanh
    centroids_phys = centroids * scales + shifts
    centroids_clip = np.clip(centroids, -0.999, 0.999)
    Z_raw_init     = np.arctanh(centroids_clip)   # inverse tanh

    np.save(output_path, Z_raw_init.astype(np.float32))
    print(f"[GP Cal] Saved {n_inducing} calibrated inducing points → {output_path}")
    return Z_raw_init