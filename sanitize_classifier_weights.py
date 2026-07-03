# sanitize_classifier_weights.py
import flax
import os
import numpy as np

def remap_and_pad_complete():
    print("=== SANITIZANDO Y ADECUANDO MATRICES DEL CLASIFICADOR V2 ===")
    file_path = "models/active_set_classifier.bytes"
    target_path = "models/active_set_classifier_v2.bytes"
    
    if not os.path.exists(file_path):
        print(f"[FAIL] No se encuentra el archivo base en {file_path}")
        return

    # 1. Restaurar el diccionario msgpack en bruto del archivo original
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    raw_state = flax.serialization.msgpack_restore(raw_bytes)
    
    # 2. Transformar claves y ajustar dimensiones (Entrada: 15->19 | Salida: 12->20)
    sanitized_state = {}
    for key, value in raw_state.items():
        if key.startswith("dense_"):
            new_key = key.replace("dense_", "Dense_")
            sanitized_state[new_key] = dict(value)
            
            # Ajuste de entrada en Dense_0: (15, 128) -> (19, 128)
            if new_key == "Dense_0" and "kernel" in sanitized_state[new_key]:
                orig_kernel = np.array(sanitized_state[new_key]["kernel"])
                padded_kernel = np.pad(orig_kernel, ((0, 4), (0, 0)), mode='constant')
                sanitized_state[new_key]["kernel"] = padded_kernel
                print(f"  > [INPUT] Dense_0 kernel ampliado de {orig_kernel.shape} a {padded_kernel.shape}")
                
        elif key == "out":
            new_key = "Dense_3"
            sanitized_state[new_key] = dict(value)
            
            # ✅ AJUSTE DE SALIDA EN Dense_3 KERNEL: (64, 12) -> (64, 20)
            if "kernel" in sanitized_state[new_key]:
                orig_kernel = np.array(sanitized_state[new_key]["kernel"])
                padded_kernel = np.pad(orig_kernel, ((0, 0), (0, 8)), mode='constant')
                sanitized_state[new_key]["kernel"] = padded_kernel
                print(f"  > [OUTPUT] Dense_3 kernel ampliado de {orig_kernel.shape} a {padded_kernel.shape}")
            
            # ✅ AJUSTE DE SESGO EN Dense_3 BIAS: (12,) -> (20,)
            if "bias" in sanitized_state[new_key]:
                orig_bias = np.array(sanitized_state[new_key]["bias"])
                padded_bias = np.pad(orig_bias, ((0, 8),), mode='constant')
                sanitized_state[new_key]["bias"] = padded_bias
                print(f"  > [BIAS] Dense_3 bias ampliado de {orig_bias.shape} a {padded_bias.shape}")
        else:
            sanitized_state[key] = value
            
    # 3. Serializar de nuevo a msgpack de Flax
    sanitized_bytes = flax.serialization.msgpack_serialize(sanitized_state)
    with open(target_path, "wb") as f:
        f.write(sanitized_bytes)
        
    print(f"\n[SUCCESS] ¡Clasificador adaptado a 19-in y 20-out con éxito en '{target_path}'!")

if __name__ == "__main__":
    remap_and_pad_complete()