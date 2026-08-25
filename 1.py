import numpy as np
import pandas as pd
from scripts.run_can_backtest import decode_can_csv_to_dataframe, _extract_1d
from pathlib import Path

# Carga la sesión 2 (la que tiene r=0.454)
df = decode_can_csv_to_dataframe(Path("data/raw_can_logs/2.csv"), Path("TER.dbc"))
vx = _extract_1d(df, 'vx_mps')
wz = np.deg2rad(_extract_1d(df, 'yaw_rate_deg_s'))
ay = _extract_1d(df, 'ay_mps2')

# Aceleración centrípeta cinemática esperada
ay_kin = vx * wz

mask = np.abs(ay_kin) > 2.0  # Zonas de curva clara (>0.2g)

print(f"--- Diagnóstico 2.csv (muestras en curva: {np.sum(mask)}) ---")
print(f"Rango a_y real:        [{ay[mask].min():.2f}, {ay[mask].max():.2f}] m/s²")
print(f"Rango a_y cinemático:  [{ay_kin[mask].min():.2f}, {ay_kin[mask].max():.2f}] m/s²")
print(f"Offset medio (bias):   {np.mean(ay[mask] - ay_kin[mask]):.3f} m/s²")
print(f"Ratio de amplitudes:   {np.std(ay[mask]) / np.std(ay_kin[mask]):.3f}")
print(f"Correlación ay vs kin: {np.corrcoef(ay[mask], ay_kin[mask])[0, 1]:.3f}")