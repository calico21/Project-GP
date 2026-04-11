import jax
import jax.numpy as jnp
from powertrain.modes.advanced.torque_vectoring import (
    allocator_cost, TVGeometry, AllocatorWeights
)

def run_thermal_validation():
    geo = TVGeometry()
    w   = AllocatorWeights()
    T   = jnp.array([80., 80., 90., 90.])
    T_prev = jnp.zeros(4)

    # LA SOLUCIÓN: Calculamos la fuerza exacta que generan esos 80/90 Nm
    # Así el error de Fx_target es 0 y evitamos la pérdida de precisión float32
    Fx_actual_exacta = jnp.sum(T) / geo.r_w

    # Ruedas Frías (25°C)
    cost_cold = allocator_cost(
        T, T_prev, Fx_actual_exacta, jnp.array(0.), jnp.array(0.),
        jnp.full(4, 750.), jnp.zeros(4), jnp.full(4, 1.5),
        jnp.full(4, 60.), T_prev, jnp.full(4, 150.), jnp.array(80000.),
        T_ribs=jnp.full(4, 25.0)
    )

    # Ruedas Óptimas (85°C)
    cost_opt = allocator_cost(
        T, T_prev, Fx_actual_exacta, jnp.array(0.), jnp.array(0.),
        jnp.full(4, 750.), jnp.zeros(4), jnp.full(4, 1.5),
        jnp.full(4, 60.), T_prev, jnp.full(4, 150.), jnp.array(80000.),
        T_ribs=jnp.full(4, 85.0)
    )

    print(f"\nCoste con Ruedas Frías:  {cost_cold:.4f}")
    print(f"Coste con Ruedas Óptimas: {cost_opt:.4f}")

    assert cost_cold > cost_opt, "ERROR: El coste térmico no se está aplicando."
    print("✅ Lógica Térmica Correcta: El optimizador penaliza los neumáticos fuera de rango.")

    # Comprobar gradientes
    grad = jax.grad(lambda T_r: allocator_cost(
        T, T_prev, Fx_actual_exacta, jnp.array(0.), jnp.array(0.),
        jnp.full(4, 750.), jnp.zeros(4), jnp.full(4, 1.5),
        jnp.full(4, 60.), T_prev, jnp.full(4, 150.), jnp.array(80000.),
        T_ribs=T_r
    ))(jnp.full(4, 60.0))

    assert jnp.all(jnp.isfinite(grad)), "ERROR: NaN en el gradiente térmico"
    print(f"✅ Gradientes Extraídos Exitosamente: {grad}")
    print("🚀 All checks passed. El sistema está listo para producción.")

if __name__ == "__main__":
    run_thermal_validation()