# sanitize_classifier_weights.py
import flax
import os

def remap_keys():
    print("=== SANITIZANDO CLASIFICADOR DE RESTRICCIONES ACTIVE-SET ===")
    file_path = "models/active_set_classifier.bytes"
    target_path = "models/active_set_classifier_v2.bytes"
    
    if not os.path.exists(file_path):
        print(f"[FAIL] No se encuentra el archivo base en {file_path}")
        return

    # 1. Restaurar el diccionario msgpack en bruto del archivo original
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    raw_state = flax.serialization.msgpack_restore(raw_bytes)
    
    print("  > Estructura cargada del archivo (minúsculas):", list(raw_state.keys()))
    
    # 2. Transformar las claves para cumplir con el estándar moderno de Flax
    sanitized_state = {}
    for key, value in raw_state.items():
        if key.startswith("dense_"):
            new_key = key.replace("dense_", "Dense_")
            sanitized_state[new_key] = value
        elif key == "out":
            # ✅ LA PIEZA MAESTRA: 'out' mapea exactamente a la capa de salida 'Dense_3'
            sanitized_state["Dense_3"] = value
        else:
            sanitized_state[key] = value
            
    print("  > Estructura sanitizada final (mayúsculas):", list(sanitized_state.keys()))
    
    # 3. Serializar de nuevo a msgpack e inyectarlo en el archivo V2
    sanitized_bytes = flax.serialization.msgpack_serialize(sanitized_state)
    with open(target_path, "wb") as f:
        f.write(sanitized_bytes)
        
    print(f"[SUCCESS] ¡Archivo convertido y blindado con éxito en '{target_path}'!")

if __name__ == "__main__":
    remap_keys()