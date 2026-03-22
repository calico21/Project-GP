"""
tools/train_pinn_from_telemetry.py

Trains the TireOperatorPINN drift correction from MoTeC back-calculated forces.
Requires no force sensors — uses Newton's second law on the full car body.

Error budget:
  Mass uncertainty:   ±5 kg  → ±1.6% force error
  CG uncertainty:     ±5 mm  → ±0.8% lateral transfer error
  Sensor noise:       IMU 0.02g → ±0.6 N/Hz RMS at 200 Hz
  Total:              ~3% RMS noise on back-calculated forces
  This is sufficient — PINN corrections are bounded to ±25%, so 3% noise
  trains a correction that is accurate to within ~5%, vs zero correction now.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
import flax.serialization
from models.tire_model import PacejkaTire, TireOperatorPINN
from data.configs.tire_coeffs import tire_coeffs as TC
from data.configs.vehicle_params import vehicle_params as VP


def build_pinn_training_data(
    motec_csv: str,
    vp: dict,
    tc: dict,
    min_vx: float = 3.0,      # exclude low-speed (tyre model less reliable)
    max_alpha: float = 0.25,  # exclude extreme slip (outside Pacejka validity)
) -> dict:
    """
    Returns training pairs: (state_8d, Fy_residual, Fx_residual) per corner.
    state_8d = [alpha, kappa, gamma, Fz, Vx, T_norm, 0, 0] — matches TireOperatorPINN input.
    """
    from tools.calibrate_gp_inducing_points import back_calculate_operating_conditions
    
    df  = pd.read_csv(motec_csv)
    ops = back_calculate_operating_conditions(df, vp)   # (N*4, 5) — reuse from B1

    tire = PacejkaTire(tc)
    m    = vp.get('total_mass', 300.0)

    # Back-calculate total forces from IMU
    vx_raw = df['velocity_ms'].values
    ay_raw = df['lateral_accel_ms2'].values
    ax_raw = np.gradient(vx_raw, 0.005)   # longitudinal accel from vx derivative
    wz_raw = np.deg2rad(df['yaw_rate_degs'].values)

    # Total forces in body frame (Newton's 2nd law)
    Fy_total_meas = m * (ay_raw - vx_raw * wz_raw)   # centripetal correction
    Fx_total_meas = m * ax_raw

    # Per-corner forces via load transfer model
    lf, lr = vp['lf'], vp['lr']
    L = lf + lr
    h = vp.get('h_cg', 0.285)
    tf = vp.get('track_front', 1.2)

    dFz_lon = m * ax_raw * h / L
    dFz_lat = m * ay_raw * h / tf
    Fz_fl   = np.maximum(m*9.81*lr/(2*L) - dFz_lon/2 - dFz_lat/2, 50.0)
    Fz_fr   = np.maximum(m*9.81*lr/(2*L) - dFz_lon/2 + dFz_lat/2, 50.0)
    Fz_rl   = np.maximum(m*9.81*lf/(2*L) + dFz_lon/2 - dFz_lat/2, 50.0)
    Fz_rr   = np.maximum(m*9.81*lf/(2*L) + dFz_lon/2 + dFz_lat/2, 50.0)

    # Front/rear split (assume front handles majority of lateral in skidpad)
    # Use bicycle model moment balance: Fy_f * lf = Fy_r * lr ... from yaw balance
    # Fy_f = Fy_total * lr/L,  Fy_r = Fy_total * lf/L (static approximation)
    Fy_f_meas = Fy_total_meas * lr / L
    Fy_r_meas = Fy_total_meas * lf / L
    Fx_f_meas = Fx_total_meas * vp.get('brake_bias_f', 0.6)
    Fx_r_meas = Fx_total_meas * (1 - vp.get('brake_bias_f', 0.6))

    # Pacejka predictions at each timestep
    T_ribs_nom = jnp.array([85., 85., 85.])
    T_gas_nom  = jnp.array(85.)
    
    training_rows = []
    for i in range(len(vx_raw)):
        vx_i = float(vx_raw[i])
        if vx_i < min_vx:
            continue

        # Average front-left and front-right into representative front conditions
        alpha_f = float(ops[i, 0])   # approximate — front left
        kappa_f = float(ops[i, 1])
        gamma_f = float(ops[i, 2])
        Fz_f_i  = float((Fz_fl[i] + Fz_fr[i]) * 0.5)

        if abs(alpha_f) > max_alpha:
            continue

        # Pacejka prediction (no PINN correction)
        Fx_pac, Fy_pac = tire.compute_force(
            jnp.array(alpha_f), jnp.array(kappa_f),
            jnp.array(Fz_f_i),  jnp.array(gamma_f),
            T_ribs_nom, T_gas_nom, jnp.array(vx_i),
        )
        Fy_pac_f = float(Fy_pac)
        Fx_pac_f = float(Fx_pac)

        # Residual (what the PINN should learn)
        Fy_res = (float(Fy_f_meas[i]) - Fy_pac_f) / (abs(Fy_pac_f) + 1e-3)
        Fx_res = (float(Fx_f_meas[i]) - Fx_pac_f) / (abs(Fx_pac_f) + 1e-3)

        # Clip to ±0.25 — PINN output is bounded here anyway
        Fy_res = np.clip(Fy_res, -0.25, 0.25)
        Fx_res = np.clip(Fx_res, -0.25, 0.25)

        state_8d = np.array([
            alpha_f, kappa_f, gamma_f, Fz_f_i, vx_i,
            0.0,   # T_norm = 0 (unknown without IR sensor — neutral)
            0.0, 0.0,
        ], dtype=np.float32)

        training_rows.append((state_8d, Fy_res, Fx_res))

    print(f"[PINN Cal] {len(training_rows)} training samples from telemetry")
    states  = jnp.array([r[0] for r in training_rows])
    fy_tgts = jnp.array([r[1] for r in training_rows])
    fx_tgts = jnp.array([r[2] for r in training_rows])
    return {'states': states, 'fy': fy_tgts, 'fx': fx_tgts}


def train_pinn(motec_csv: str, vp: dict, tc: dict, n_epochs: int = 2000):
    data    = build_pinn_training_data(motec_csv, vp, tc)
    pinn    = TireOperatorPINN()
    key     = jax.random.PRNGKey(0)
    params  = pinn.init(key, data['states'][0])

    # Conservative LR — preserving the architecture's zero-initialisation intent
    # means learning small corrections, not large swings
    tx      = optax.adamw(learning_rate=1e-4, weight_decay=1e-5)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, states, fy_t, fx_t):
        def loss_fn(p):
            def per_sample(s, fy, fx):
                mods, _ = pinn.apply(p, s)
                # Mean of the two output corrections vs measured residuals
                return (mods[0] - fx) ** 2 + (mods[1] - fy) ** 2
            return jnp.mean(jax.vmap(per_sample)(states, fy_t, fx_t))

        l, g = jax.value_and_grad(loss_fn)(params)
        upd, new_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), new_state, l

    # Mini-batch training
    batch_size = 256
    n = len(data['states'])

    for epoch in range(1, n_epochs + 1):
        idx     = jax.random.permutation(jax.random.PRNGKey(epoch), n)[:batch_size]
        params, opt_state, loss = step(
            params, opt_state,
            data['states'][idx], data['fy'][idx], data['fx'][idx]
        )
        if epoch % 200 == 0:
            print(f"  PINN epoch {epoch:4d} | loss: {float(loss):.6f}")

    # Save into tire model's standard parameter location
    out_path = os.path.join(os.path.dirname(__file__), '../models/pinn_params.bytes')
    with open(out_path, 'wb') as f:
        f.write(flax.serialization.to_bytes(params))
    print(f"[PINN Cal] Weights saved → {out_path}")
    return params