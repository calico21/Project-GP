# scripts/vcu_bridge.py
import jax.numpy as jnp
import numpy as np
import struct

Q15_ONE = 32768  # 2^15 para el reparto de par
Q8_ONE = 255     # 2^8 para la atenuación del CBF

def pack_tv_split_frame(diag) -> bytes:
    """
    Toma el objeto PowertrainDiagnostics resultante de powertrain_step().
    Genera un payload binario de 4 bytes listo para el bus CAN de la VCU.
    """
    # Extracción de pares del eje trasero (Índices 2 y 3: RL y RR)
    t_rl = diag.T_wheel[2]
    t_rr = diag.T_wheel[3]
    
    total = jnp.abs(t_rl) + jnp.abs(t_rr) + 1e-6
    
    # 1. Fracción de par hacia la rueda trasera izquierda (RL) en Q15
    # Conserva el signo algebraico neta para soportar asimetrías de regeneración
    frac_rl = jnp.clip(0.5 + 0.5 * (t_rl - t_rr) / total, 0.0, 1.0)
    split_q15 = int(np.asarray(frac_rl) * Q15_ONE)
    split_q15 = np.clip(split_q15, 0, Q15_ONE).astype(np.int16)
    
    # 2. Factor de atenuación global de par (CBF) en Q8 (0 = Corte total, 255 = 100% par)
    # Si el CBF está activo disminuyendo el par por seguridad, escalamos la comanda a la baja
    scale_factor = diag.torque_scale_factor if hasattr(diag, 'torque_scale_factor') else 1.0
    total_scale_q8 = int(np.asarray(scale_factor) * Q8_ONE)
    total_scale_q8 = np.clip(total_scale_q8, 0, Q8_ONE).astype(np.uint8)
    
    # 3. Flag diagnóstico de intervención CBF/E-ABS
    cbf_flag = uint8(1) if diag.cbf_active > 0.5 else uint8(0)
    
    # Empaquetado: int16 (2 bytes) + uint8 (1 byte) + uint8 (1 byte) = 4 bytes nítidos
    return struct.pack('<hBB', split_q15, total_scale_q8, cbf_flag)