import time
import can 
import jax.numpy as jnp

# Importa tu control de tracción intermedio (configurado para RWD)
from powertrain.modes.intermediate.launch_control import intermediate_launch_step, IntermediateLCState, IntermediateLCParams

# Importa el archivo que me acabas de pasar
from vcu_bridge import pack_tv_split_frame

def main():
    print("Iniciando Project-GP: Modo Intermedio RWD...")
    
    # 1. Configuración obligatoria para RWD
    params = IntermediateLCParams(front_ratio_initial=0.0, front_ratio_final=0.0)
    lc_state = IntermediateLCState.default(params)
    
    # 2. Conexión al bus CAN de la Raspberry/Jetson
    # Asegúrate de que 'can0' es el nombre correcto en vuestro Linux
    bus = can.interface.Bus(channel='can0', bustype='socketcan')
    
    T_MAX_AXLE = 500.0 # Ajusta esto al límite máximo en Nm de tu eje trasero
    
    print("Sistema listo. Esperando a que el piloto pise el acelerador...")
    
    while True:
        start_time = time.perf_counter()
        
        # ---------------------------------------------------------
        # PASO 1: LEER SENSORES DEL COCHE (Deberías leerlos por CAN)
        # ---------------------------------------------------------
        throttle_norm = 1.0  # DUMMY: Reemplazar con lectura real del pedal [0 a 1]
        brake_norm = 0.0     # DUMMY: Reemplazar con lectura real del freno [0 a 1]
        vx = 5.0             # DUMMY: Reemplazar con velocidad del coche [m/s]
        wz = 0.0             # DUMMY: Reemplazar con Yaw Rate [rad/s]
        
        Fz_dummy = jnp.array([100.0, 100.0, 750.0, 750.0]) 
        T_tc_dummy = jnp.zeros(4)
        T_max_hw = jnp.full(4, 250.0) 
        
        # ---------------------------------------------------------
        # PASO 2: EJECUTAR EL CONTROL DE TRACCIÓN (JAX)
        # ---------------------------------------------------------
        output, lc_state = intermediate_launch_step(
            throttle=jnp.array(throttle_norm),
            brake=jnp.array(brake_norm),
            vx=jnp.array(vx),
            wz=jnp.array(wz),
            Fz=Fz_dummy,
            T_tc=T_tc_dummy,
            T_max_hw=T_max_hw,
            lc_state=lc_state,
            dt=jnp.array(0.005),
            params=params
        )
        
        T_RL = output.T_command[2]
        T_RR = output.T_command[3]
        
        # ---------------------------------------------------------
        # PASO 3: TRADUCIR CON vcu_bridge.py
        # ---------------------------------------------------------
        # Esto usa exactamente la función de tu archivo para empaquetar los datos
        can_data_bytes = pack_tv_split_frame(
            T_RL, 
            T_RR, 
            limit_axle=T_MAX_AXLE, 
            cbf_active=jnp.array(0.0)
        )
        
        # ---------------------------------------------------------
        # PASO 4: ENVIAR A LA STM32
        # ---------------------------------------------------------
        msg = can.Message(
            arbitration_id=0x300, # Cambia 0x300 por el ID que use tu VCU
            data=can_data_bytes,
            is_extended_id=False
        )
        bus.send(msg)
        
        # Mantener el loop a 200Hz (5ms)
        elapsed = time.perf_counter() - start_time
        time.sleep(max(0.0, 0.005 - elapsed))

if __name__ == "__main__":
    main()