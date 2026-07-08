# scripts/vcu_bridge.py
import jax
import jax.numpy as jnp
import numpy as np
import struct

Q15_ONE = 32768
Q8_ONE = 255

@jax.jit
def compute_can_scalars(t_rl: jax.Array, t_rr: jax.Array, cbf_active: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Executes entirely on device (GPU/XLA). 
    Calculates torque split and scales directly in fixed-point space.
    """
    total = jnp.abs(t_rl) + jnp.abs(t_rr) + 1e-6
    frac_rl = jnp.clip(0.5 + 0.5 * (t_rl - t_rr) / total, 0.0, 1.0)
    
    split_q15 = jnp.clip(jnp.round(frac_rl * Q15_ONE), 0, Q15_ONE).astype(jnp.int16)
    
    # Scale factor default to 255 (100% torque) unless CBF scales it down
    total_scale_q8 = jnp.uint8(Q8_ONE)
    cbf_flag = jnp.where(cbf_active > 0.5, jnp.uint8(1), jnp.uint8(0))
    
    return split_q15, total_scale_q8, cbf_flag

def pack_tv_split_frame(diag) -> bytes:
    """
    Host-side packer. Pulls the pre-quantized 32-bit payload from device memory 
    in a single DMA transfer and packs it into the 0x300 CAN frame.
    """
    # Pull only the required scalars from device in one operation
    split_q15, scale_q8, cbf_flag = jax.device_get(
        compute_can_scalars(diag.T_wheel[2], diag.T_wheel[3], diag.cbf_active)
    )
    
    # Pack into Little-Endian int16 + uint8 + uint8 = 4 bytes
    return struct.pack('<hBB', int(split_q15), int(scale_q8), int(cbf_flag))