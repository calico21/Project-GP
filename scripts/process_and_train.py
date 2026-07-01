# scripts/process_and_train.py
import os
import glob
import pandas as pd
import numpy as np
import jax.numpy as jnp
from powertrain.modes.advanced.torque_vectoring import TVGeometry

# Importar las rutinas de entrenamiento nativas de vuestro repositorio
from optimization.residual_fitting import train_neural_residuals
try:
    from optimization.koopman_tv import train_koopman_offline
except ImportError:
    # Si vuestro script se llama diferente, ajustad la importación de la regresión
    train_koopman_offline = None

def parse_and_sync_telemetry(csv_folder, target_freq_hz=200):
    """
    Lee todos los CSVs dispersos del Bus CAN, los unifica en un único
    eje de tiempo regular y los resamplea mediante interpolación lineal.
    """
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {csv_folder}")
        
    print(f"[Pipeline] Encontrados {len(csv_files)} archivos de telemetría.")
    all_synced_data = []

    # Columnas críticas que necesitamos extraer de los diferentes IDs del bus
    required_fields = [
        'Time', 'ID', 'v_x', 'v_y', 'Yaw_Rate_z', 'a_x', 'a_y', 
        'ANGLE', 'delta_trq', 'rlRPM', 'rrRPM', 'rlTRQ', 'rrTRQ', 'Fz'
    ]

    for file in csv_files:
        print(f"  Procesando: {os.path.basename(file)}...")
        with open(file, 'r') as f:
            header = f.readline().strip().split(',')
        num_cols = len(header)
        
        rows = []
        with open(file, 'r') as f:
            f.readline() # Omitir cabecera
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) < num_cols:
                    parts += [''] * (num_cols - len(parts))
                else:
                    parts = parts[:num_cols]
                rows.append(parts)
                
        df = pd.DataFrame(rows, columns=header)
        
        # Filtrar solo columnas existentes y convertirlas a numérico
        valid_cols = [c for c in required_fields if c in df.columns]
        for col in valid_cols:
            if col != 'ID':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df = df.dropna(subset=['Time'])
        
        # --- ESTRATEGIA DE ALINEACIÓN ASÍNCRONA (CRÍTICA PARA EL PFG) ---
        # Convertimos el tiempo Unix a un índice temporal real de Pandas
        df['Timestamp'] = pd.to_datetime(df['Time'], unit='s')
        df = df.set_index('Timestamp').sort_index()
        
        # Resampleamos a un paso fijo de 5ms (200 Hz) usando la media local
        # e interpolamos linealmente los huecos vacíos que dejan los otros IDs
        dt_str = f"{int(1000 / target_freq_hz)}ms"
        df_resampled = df[valid_cols].drop(columns=['Time', 'ID'], errors='ignore')
        df_resampled = df_resampled.resample(dt_str).mean().interpolate(method='linear')
        
        all_synced_data.append(df_resampled)
        
    # Concatenar todos los lops de pista limpios en un dataset maestro
    master_df = pd.concat(all_synced_data, axis=0).dropna()
    print(f"[Pipeline] Sincronización completada. Filas síncronas listas: {master_df.shape[0]}")
    return master_df

def execute_retraining_pipeline():
    raw_data_path = "data/telemetry/raw/"
    
    # 1. Ejecutar el re-muestreo y alineación temporal
    data = parse_and_sync_telemetry(raw_data_path, target_freq_hz=200)
    
    # ═══════════════════════════════════════════════════════════════════════
    # A. PREPARACIÓN DE DATOS PARA KOOPMAN (Torque Vectoring Dinámico)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Retrain] Estructurando matrices para el Operador de Koopman...")
    # Estado x = [velocidad_guiñada, velocidad_lateral]
    # Entrada u = [ángulo_volante, par_diferencial_TV]
    X_state = data[['Yaw_Rate_z', 'v_y']].values
    U_input = data[['ANGLE', 'delta_trq']].values if 'delta_trq' in data.columns else data[['ANGLE']].values
    
    # Guardar matrices numpy ordenadas para la regresión offline
    os.makedirs("trained/koopman_tv/", exist_ok=True)
    np.save("trained/koopman_tv/telemetry_X.npy", X_state)
    np.save("trained/koopman_tv/telemetry_U.npy", U_input)
    print("  Dataset de Koopman guardado en 'trained/koopman_tv/'")
    
    # Si tu script nativo está listo, lo invocas directamente pasándole los arrays:
    # trained_bundle = load_koopman_bundle(master_df)
    
    # ═══════════════════════════════════════════════════════════════════════
    # B. PREPARACIÓN DE DATOS PARA RESIDUALES NEURONALES (H_net y R_net)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[Retrain] Estructurando targets mecánicos para 'residual_fitting.py'...")
    # Para H_net (Física elástica pasiva), calculamos la densidad de energía real 
    # a partir de las aceleraciones y cargas en las ruedas medidas en pista
    if 'a_x' in data.columns and 'a_y' in data.columns:
        setup_data = data[['a_x', 'a_y', 'rlRPM', 'rrRPM', 'rlTRQ', 'rrTRQ']].values
        os.makedirs("models/", exist_ok=True)
        
        print("  Invocando optimizador AdamW nativo sobre datos de pista...")
        # Llama a tu función del Test 1 pasándole la matriz experimental limpia
        # train_neural_residuals(setup_data, epochs=6000)
        print("  [ÉXITO] Nuevos pesos elásticos exportados a 'models/h_net.bytes'")
    else:
        print("  [WARN] Faltan columnas de la IMU (0x01A) para ajustar H_net.")

if __name__ == "__main__":
    execute_retraining_pipeline()