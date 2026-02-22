import jax
import jax.numpy as jnp

def compute_skidpad_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=1.5):
    # Ensure x_init has 46 elements in the caller!
    """
    Simulates a steady-state cornering maneuver.
    Returns the mean Lateral G-force (Grip) and Safety Margin.
    """
    steps = int(T_max / dt)
    
    def step_fn(x, t):
        steer = 0.16 * jnp.tanh(3.0 * jnp.sin(2.0 * jnp.pi * 0.25 * t))
        vx_error = 12.0 - x[14]
        throttle_brake = 2500.0 * jnp.tanh(vx_error)
        u = jnp.array([steer, throttle_brake])
        
        x_next = simulate_step_fn(x, u, params, dt)
        
        vx = x_next[14]
        yaw_rate = jnp.sqrt(x_next[19]**2 + 1e-6)
        lat_g = jnp.sqrt((vx * x_next[19] / 9.81)**2 + 1e-6)
        safety_margin = 3.0 - yaw_rate

        # Truncated BPTT: Stop gradients on state, keep gradients on physics outputs
        return jax.lax.stop_gradient(x_next), (lat_g, safety_margin)

    t_array = jnp.linspace(0, T_max, steps)
    _, (lat_gs, safety_margins) = jax.lax.scan(step_fn, x_init, t_array)
    
    # Calculate steady-state grip by averaging only the final 50 time steps
    obj_grip = jnp.mean(lat_gs[-50:]) 
    min_safety = jnp.min(safety_margins)
    return obj_grip, min_safety

def compute_frequency_response_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Simulates a swept-sine (chirp) steering input from 0.5Hz to 4.0Hz.
    Calculates the resonance/nervousness of the setup.
    """
    steps = int(T_max / dt)
    
    def step_fn(x, t):
        f0, f1 = 0.5, 4.0
        # Instantaneous phase for chirp signal
        phase = 2.0 * jnp.pi * (f0 * t + 0.5 * (f1 - f0) * (t**2) / T_max)
        steer = 0.05 * jnp.sin(phase)
        
        # Maintain 15 m/s for slalom testing
        vx_error = 15.0 - x[14] 
        throttle_brake = 2000.0 * jnp.tanh(vx_error)
        u = jnp.array([steer, throttle_brake])
        
        x_next = simulate_step_fn(x, u, params, dt)
        
        # We penalize aggressive yaw acceleration (snappy oversteer)
        yaw_accel = (x_next[19] - x[19]) / dt
        return jax.lax.stop_gradient(x_next), yaw_accel

    t_array = jnp.linspace(0, T_max, steps)
    _, yaw_accels = jax.lax.scan(step_fn, x_init, t_array)
    
    # High variance in yaw acceleration = highly resonant, unpredictable setup
    resonance_penalty = jnp.var(yaw_accels) 
    return resonance_penalty