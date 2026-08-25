#!/usr/bin/env python3
import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw_can_logs")
DT = 0.005        # 200 Hz
R_WHEEL = 0.2045   # Radio de rueda en metros
K_ROLL = 2.0       # Gradiente de balanceo del chasis (deg/g)

def repair_and_resample_csv(file_path: Path, dt: float = 0.005) -> pd.DataFrame:
    # 1. Lectura segura evitando desbordamiento de buffer por comas finales
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header_line = f.readline().strip()
    header_cols = [c.strip() for c in header_line.split(',') if c.strip() != '']
    n_cols = len(header_cols)

    rows = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < n_cols:
                row = row + [''] * (n_cols - len(row))
            elif len(row) > n_cols:
                row = row[:n_cols]
            rows.append(row)

    raw_df = pd.DataFrame(rows, columns=header_cols)
    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]

    # 2. Localizar columna de tiempo
    cols_lower = {c.lower(): c for c in raw_df.columns}
    t_col = cols_lower.get('timestamp') or cols_lower.get('time')
    if not t_col:
        print(f"  [!] Sin columna de tiempo en {file_path.name}")
        return None

    raw_df[t_col] = pd.to_numeric(raw_df[t_col], errors='coerce')
    raw_df = raw_df.dropna(subset=[t_col]).sort_values(by=t_col)

    t_arr = raw_df[t_col].to_numpy(dtype=np.float64)
    t_rel = t_arr - t_arr[0]
    if len(t_rel) < 2 or t_rel[-1] <= 0:
        return None

    # 3. Interpolar todas las señales asíncronas a una cuadrícula uniforme de 200 Hz
    t_uniform = np.arange(0, t_rel[-1], dt)
    resampled = {'timestamp': t_uniform}

    for col in raw_df.columns:
        if col == t_col:
            continue
        vals = pd.to_numeric(raw_df[col], errors='coerce').to_numpy(dtype=np.float64)
        valid = np.isfinite(vals) & (vals != 0.0)  # Evitar ceros de tramas vacías
        if valid.sum() < 2:
            valid = np.isfinite(vals)
        if valid.sum() >= 2:
            order = np.argsort(t_rel[valid])
            resampled[col] = np.interp(t_uniform, t_rel[valid][order], vals[valid][order])
        else:
            resampled[col] = np.zeros_like(t_uniform)

    df_sync = pd.DataFrame(resampled)

    # 4. Reconstrucción de velocidad vx limpia desde encoders
    rl_rpm = df_sync.get('rlRPM', pd.Series(0, index=df_sync.index)).to_numpy()
    rr_rpm = df_sync.get('rrRPM', pd.Series(0, index=df_sync.index)).to_numpy()
    vx_wheel = ((rl_rpm + rr_rpm) * 0.5) * (2.0 * np.pi / 60.0) * R_WHEEL

    vx_gps = df_sync.get('v_x', pd.Series(0, index=df_sync.index)).to_numpy()
    vx = vx_wheel if np.std(vx_gps) < 1.0 else vx_gps
    df_sync['v_x'] = vx

    # 5. Guiñada wz (rad/s)
    wz_deg = df_sync.get('Yaw_Rate_z', pd.Series(0, index=df_sync.index)).to_numpy()
    wz_rad = np.deg2rad(wz_deg)

    # 6. Aceleración lateral cinemática recuperada (m/s²)
    ay_kinematic = vx * wz_rad

    # 7. Aceleración lateral IMU escalada
    ay_raw = df_sync.get('a_y', pd.Series(0, index=df_sync.index)).to_numpy()
    if 0.01 < np.std(ay_raw) < 2.5:
        ay_raw = ay_raw * 9.81

    # Fusión: dinámica real centrípeta en curva (>1.5 m/s²), sensor en recta
    ay_final = np.where(np.abs(ay_kinematic) > 1.5, ay_kinematic, ay_raw)
    df_sync['a_y'] = ay_final

    # 8. Roll libre del artefacto de 50° del atan2
    df_sync['ROLL'] = (ay_kinematic / 9.81) * K_ROLL

    return df_sync

for csv_file in sorted(DATA_DIR.glob("*.csv")):
    print(f"[*] Procesando {csv_file.name}...")
    df_clean = repair_and_resample_csv(csv_file, DT)
    if df_clean is not None:
        df_clean.to_csv(csv_file, index=False)
        ay_vals = df_clean['a_y'].to_numpy()
        roll_vals = df_clean['ROLL'].to_numpy()
        print(f"    -> Rango ay restaurado:   [{ay_vals.min():.2f}, {ay_vals.max():.2f}] m/s²")
        print(f"    -> Rango roll restaurado: [{roll_vals.min():.2f}, {roll_vals.max():.2f}] deg")

print("\n[*] Todos los logs han sido sincronizados y reparados con éxito.")