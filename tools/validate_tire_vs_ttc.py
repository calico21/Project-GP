"""
tools/validate_tire_vs_ttc.py
Project-GP — TTC Round 9 Tire Model Cross-Validation

ONE COMMAND:
  cd ~/FS_Driver_Setup_Optimizer
  python tools/validate_tire_vs_ttc.py

WHAT IT DOES:
  1. Loads B2356run4/5/6.mat from data/ttc_round9/
  2. Extracts steady-state cornering points (removes transients, spring tests)
  3. Splits 80% train / 20% test (stratified by Fz × IA × P)
  4. Evaluates YOUR PacejkaTire model from models/tire_model.py
  5. Prints R², RMSE, load sensitivity comparison
  6. Saves processed dataset to data/ttc_round9/processed.npz

WHAT YOU NEED BEFORE RUNNING:
  - data/ttc_round9/B2356run4.mat
  - data/ttc_round9/B2356run5.mat
  - data/ttc_round9/B2356run6.mat
  (Download from fsaettc.org → Round 9 → RunData_Cornering_Matlab_SI)
"""
from __future__ import annotations

import sys
import os
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Must import jax_config FIRST (XLA cache, memory, etc.)
try:
    import jax_config
except ImportError:
    pass

import jax
import jax.numpy as jnp
import scipy.io as sio


# ─────────────────────────────────────────────────────────────────────────────
# §1  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TTC_DIR = os.path.join(PROJECT_ROOT, 'data', 'ttc_round9')
RUN_FILES = ['B2356run4.mat', 'B2356run5.mat', 'B2356run6.mat']
OUTPUT_FILE = os.path.join(TTC_DIR, 'processed.npz')

CHANNELS = [
    'SA', 'FY', 'FX', 'FZ', 'MZ', 'IA', 'P', 'V',
    'TSTC', 'TSTI', 'TSTO', 'ET', 'N', 'MX', 'RL', 'SL'  # <-- Add 'SL' here
]


# ─────────────────────────────────────────────────────────────────────────────
# §2  LOAD RAW .MAT FILES
# ─────────────────────────────────────────────────────────────────────────────

def load_ttc_runs() -> dict:
    """Load and concatenate all cornering runs."""
    all_channels = {ch: [] for ch in CHANNELS}
    all_channels['RUN'] = []

    for fname in RUN_FILES:
        path = os.path.join(TTC_DIR, fname)
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            print(f"  Download from fsaettc.org → Round 9 → "
                  f"RunData_Cornering_Matlab_SI_Round9.zip")
            sys.exit(1)

        mat = sio.loadmat(path, squeeze_me=True)
        n = len(mat['SA'])
        run_num = int(mat['RUN'].flat[0]) if mat['RUN'].ndim > 0 else int(mat['RUN'])

        print(f"  Run {run_num:2d}: {n:6d} pts | {mat['tireid']}")

        for ch in CHANNELS:
            all_channels[ch].append(mat[ch].astype(np.float64))
        all_channels['RUN'].append(np.full(n, run_num, dtype=np.int8))

    # Concatenate
    merged = {ch: np.concatenate(all_channels[ch]) for ch in CHANNELS + ['RUN']}
    print(f"  Total raw: {len(merged['SA'])} points")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# §3  EXTRACT STEADY-STATE POINTS
# ─────────────────────────────────────────────────────────────────────────────

def extract_steady_state(raw: dict) -> dict:
    """Remove transients, spring rate tests, dwell periods."""
    SA = raw['SA']
    dSA_dt = np.gradient(SA, 0.01)  # 100 Hz sampling

    mask = (
        (raw['FZ'] < -100) &           # tire loaded
        (raw['V'] > 20) &              # belt running
        (np.abs(dSA_dt) < 30) &        # quasi-steady (sweep rate ~24°/s)
        (np.abs(SA) < 13) &            # within sweep range
        (np.abs(raw['FY']) < 5000)     # no spikes
    )

    ss = {ch: raw[ch][mask] for ch in CHANNELS + ['RUN']}
    print(f"  Steady-state: {mask.sum()} pts ({mask.sum()/len(SA)*100:.1f}%)")
    return ss


# ─────────────────────────────────────────────────────────────────────────────
# §4  STRATIFIED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(ss: dict, test_frac: float = 0.2, seed: int = 42):
    """Split by (Fz × IA × P) strata so both sets cover full envelope."""
    rng = np.random.default_rng(seed)
    Fz = -ss['FZ']  # positive = loaded

    fz_bins = np.digitize(Fz, [300, 500, 700, 900, 1100])
    ia_bins = np.digitize(ss['IA'], [1, 3])
    p_bins  = np.digitize(ss['P'], [60, 75, 90])
    strata  = fz_bins * 100 + ia_bins * 10 + p_bins

    n = len(Fz)
    test_mask = np.zeros(n, dtype=bool)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        n_test = max(1, int(len(idx) * test_frac))
        test_mask[rng.choice(idx, n_test, replace=False)] = True

    print(f"  Train: {(~test_mask).sum()} | Test: {test_mask.sum()}")
    return test_mask


# ─────────────────────────────────────────────────────────────────────────────
# §5  CONVERT TO PROJECT-GP CONVENTIONS & SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(ss: dict, test_mask: np.ndarray):
    """Save in Project-GP units: rad, N (positive Fz), m/s, °C."""
    out = {
        'alpha_rad':  np.deg2rad(ss['SA']),
        'kappa':      ss['SL'],                   # <-- ADD THIS LINE
        'Fy_N':       ss['FY'],                   # SAE convention
        'Fx_N':       ss['FX'],
        'Fz_N':       -ss['FZ'],                  # flip: positive = loaded
        'Mz_Nm':      ss['MZ'],
        'gamma_rad':  np.deg2rad(ss['IA']),
        'P_kPa':      ss['P'],
        'Vx_ms':      ss['V'] / 3.6,
        'T_center_C': ss['TSTC'],
        'T_inner_C':  ss['TSTI'],
        'T_outer_C':  ss['TSTO'],
        'is_test':    test_mask.astype(np.int8),
        'run':        ss['RUN'],
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez_compressed(OUTPUT_FILE, **out)
    print(f"  Saved → {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)//1024} KB)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# §6  VALIDATE AGAINST PROJECT-GP PacejkaTire
# ─────────────────────────────────────────────────────────────────────────────

def validate(dataset: dict):
    """Run PacejkaTire on test points using vectorized JAX. Report R², RMSE."""
    from config.tire_coeffs import tire_coeffs
    from models.tire_model import PacejkaTire

    tire = PacejkaTire(tire_coeffs)
    # Zero out PINN/GP for clean Pacejka baseline validation
    import jax.tree_util
    # Zero out PINN/GP for clean Pacejka baseline validation
    tire._pinn_params = jax.tree_util.tree_map(jnp.zeros_like, tire._pinn_params)
    # Force GP variance to ~0 so penalty = 0 (zeroed log_var gives exp(0)=1, wrong direction)
    tire._pinn_params = jax.tree_util.tree_map_with_path(
        lambda path, v: jnp.full_like(v, -30.0) if any('log_var' in str(p) for p in path)
        else jnp.full_like(v, 10.0) if any('log_ls' in str(p) for p in path)
        else jnp.zeros_like(v),
        tire._pinn_params,
    )
    is_test = dataset['is_test'].astype(bool)
    alpha_test = jnp.array(dataset['alpha_rad'][is_test], dtype=jnp.float32)
    Fz_test    = jnp.array(dataset['Fz_N'][is_test], dtype=jnp.float32)
    gamma_test = jnp.array(dataset['gamma_rad'][is_test], dtype=jnp.float32)
    Vx_test    = jnp.array(np.maximum(dataset['Vx_ms'][is_test], 1.0), dtype=jnp.float32)
    Fy_meas    = dataset['Fy_N'][is_test]

    T_ribs = jnp.array([90.0, 90.0, 90.0])
    T_gas  = jnp.float32(90.0)
    kappa  = jnp.float32(0.0)

    n = len(alpha_test)
    print(f"\n  Vectorized evaluation on {n} test points...")

    # Wrap compute_force for vmap: single-point function
    def eval_one(a, fz, g, vx):
        fx, fy = tire.compute_force(
            alpha=a, kappa=kappa, Fz=fz, gamma=g,
            T_ribs=T_ribs, T_gas=T_gas, Vx=vx,
        )
        return fy

    # vmap over all test points at once — single GPU kernel
    eval_batch = jax.jit(jax.vmap(eval_one))

    print(f"  JIT-compiling vectorized kernel (one-time cost)...")
    Fy_pred = np.array(eval_batch(alpha_test, Fz_test, gamma_test, Vx_test))
    print(f"  Done. {n} points evaluated.")

    # Check sign convention
    corr = np.corrcoef(Fy_meas, Fy_pred)[0, 1]
    sign_note = ""
    if corr < -0.5:
        Fy_pred = -Fy_pred
        sign_note = " [sign-flipped]"

    # Metrics
    err = Fy_meas - Fy_pred
    rmse_test = float(np.sqrt(np.mean(err**2)))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((Fy_meas - Fy_meas.mean())**2)
    r2_test = float(1 - ss_res / (ss_tot + 1e-12))
    nrmse = rmse_test / (np.max(np.abs(Fy_meas))) * 100

    print(f"\n{'='*70}")
    print(f"  CROSS-VALIDATION RESULTS{sign_note}")
    print(f"{'='*70}")
    print(f"\n  Fy (test):  R² = {r2_test:.6f}  |  RMSE = {rmse_test:.1f} N  |  NRMSE = {nrmse:.2f}%")

    # Breakdown by load
    Fz_np = np.array(Fz_test)
    gamma_np = np.array(gamma_test)
    P_test = dataset['P_kPa'][is_test]

    print(f"\n  Fy R² by normal load:")
    for lo, hi in [(100,350), (350,550), (550,750), (750,1000), (1000,1300)]:
        m = (Fz_np >= lo) & (Fz_np < hi)
        if m.sum() < 50: continue
        e = Fy_meas[m] - Fy_pred[m]
        r2 = 1 - np.sum(e**2) / (np.sum((Fy_meas[m] - Fy_meas[m].mean())**2) + 1e-12)
        print(f"    Fz {lo:4d}–{hi:4d} N: R²={r2:.4f}  RMSE={np.sqrt(np.mean(e**2)):.1f} N  (n={m.sum()})")

    print(f"\n  Fy R² by camber:")
    for ia_deg in [0, 2, 4]:
        m = np.abs(gamma_np - np.deg2rad(ia_deg)) < np.deg2rad(0.5)
        if m.sum() < 50: continue
        e = Fy_meas[m] - Fy_pred[m]
        r2 = 1 - np.sum(e**2) / (np.sum((Fy_meas[m] - Fy_meas[m].mean())**2) + 1e-12)
        print(f"    IA={ia_deg}°: R²={r2:.4f}  (n={m.sum()})")

    print(f"\n  Load sensitivity — model vs Calspan:")
    print(f"    {'Fz':>6} {'μ_data':>8} {'μ_model':>8} {'error':>8}")
    for fz_t in [250, 450, 650, 900, 1100]:
        m = (np.abs(Fz_np - fz_t) < 100) & (np.abs(gamma_np) < np.deg2rad(0.5))
        if m.sum() < 20: continue
        mu_d = np.max(np.abs(Fy_meas[m])) / fz_t
        mu_m = np.max(np.abs(Fy_pred[m])) / fz_t
        print(f"    {fz_t:6.0f} {mu_d:8.3f} {mu_m:8.3f} {mu_m-mu_d:+8.3f}")

    residual = Fy_meas - Fy_pred
    print(f"\n  Residual stats (PINN target):")
    print(f"    Std: {residual.std():.1f} N  |  95th: {np.percentile(np.abs(residual), 95):.1f} N")

    print(f"\n{'='*70}")
    print(f"  AWARD-READY SUMMARY")
    print(f"{'='*70}")
    print(f"  Fy R² (test) : {r2_test:.4f}")
    print(f"  Fy RMSE      : {rmse_test:.1f} N")
    print(f"  Test points  : {n}")
    print(f"{'='*70}\n")

    return r2_test, rmse_test


# ─────────────────────────────────────────────────────────────────────────────
# §7  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "#" * 70)
    print("  PROJECT-GP: TIRE MODEL vs TTC CALSPAN DATA")
    print("#" * 70)

    print("\n[1/4] Loading TTC Round 9 .mat files...")
    raw = load_ttc_runs()

    print("\n[2/4] Extracting steady-state points...")
    ss = extract_steady_state(raw)

    print("\n[3/4] Stratified train/test split...")
    test_mask = stratified_split(ss)

    print("\n[4/4] Saving processed dataset...")
    dataset = save_processed(ss, test_mask)

    print("\n[5/5] Cross-validating PacejkaTire model...")
    r2, rmse = validate(dataset)

    return r2, rmse


if __name__ == '__main__':
    main()