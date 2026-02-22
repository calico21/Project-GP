import jax
import jax.numpy as jnp

def compute_skidpad_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Analytical steady-state cornering balance.
    Computes maximum lateral G where front AND rear tyres are simultaneously 
    at their grip limits — this is the definition of a balanced setup.
    """
    from data.configs.vehicle_params import vehicle_params as VP
    
    k_f   = params[0]
    k_r   = params[1]
    arb_f = params[2]
    arb_r = params[3]
    c_f   = params[4]   # dampers don't affect steady-state grip directly
    c_r   = params[5]
    h_cg  = params[6]
    
    mr_f = 1.2
    mr_r = 1.15
    wheel_rate_f = k_f  / (mr_f ** 2)
    wheel_rate_r = k_r  / (mr_r ** 2)
    arb_rate_f   = arb_f / (mr_f ** 2)
    arb_rate_r   = arb_r / (mr_r ** 2)
    
    Kroll_f = (wheel_rate_f + arb_rate_f) * (1.2 ** 2) * 0.5
    Kroll_r = (wheel_rate_r + arb_rate_r) * (1.15 ** 2) * 0.5
    Kroll_total = Kroll_f + Kroll_r + 1.0
    
    lltd_f = Kroll_f / Kroll_total   # Lateral Load Transfer Distribution to front
    lltd_r = Kroll_r / Kroll_total
    
    m   = VP.get('m', 300.0)
    lf  = VP.get('lf', 0.765)
    lr  = VP.get('lr', 0.765)
    t_w = 1.2
    g   = 9.81
    PDY1 = 2.218 * 0.65 # Adjusted for 1.5G target (was 2.218 for ~2.2G peak)
    PDY2 = -0.25
    Fz0  = 1000.0

    # Static loads
    Fz_f_static = m * g * lr / (lf + lr)
    Fz_r_static = m * g * lf / (lf + lr)
    
    # Sweep lateral G from 1.2G to 1.8G and compute balance metric
    ay_sweep = jnp.linspace(1.2, 1.8, 200)
    
    def compute_balance_at_ay(ay_g):
        ay = ay_g * g
        LLT_total = m * ay * h_cg / t_w
        
        LLT_f = LLT_total * lltd_f
        LLT_r = LLT_total * lltd_r
        
        Fz_fo = jnp.maximum(10.0, Fz_f_static/2 + LLT_f)
        Fz_fi = jnp.maximum(10.0, Fz_f_static/2 - LLT_f)
        Fz_ro = jnp.maximum(10.0, Fz_r_static/2 + LLT_r)
        Fz_ri = jnp.maximum(10.0, Fz_r_static/2 - LLT_r)
                
        def mu(Fz):
            dfz = (Fz - Fz0) / Fz0
            return PDY1 * (1.0 + PDY2 * dfz)
        
        # Lateral grip capacity per axle
        Fy_f_max = mu(Fz_fo) * Fz_fo + mu(Fz_fi) * Fz_fi
        Fy_r_max = mu(Fz_ro) * Fz_ro + mu(Fz_ri) * Fz_ri
        
        # Required lateral force from dynamics
        Fy_required = m * ay
        
        # Front/rear split from geometry
        Fy_f_req = Fy_required * lr / (lf + lr)
        Fy_r_req = Fy_required * lf / (lf + lr)
        
        # Utilisation — how close each axle is to its limit (1.0 = at limit)
        util_f = Fy_f_req / (Fy_f_max + 1e-3)
        util_r = Fy_r_req / (Fy_r_max + 1e-3)
        
        # Balance metric: 1.0 when perfectly balanced, <1 when imbalanced
        balance = 1.0 - jnp.abs(util_f - util_r)
        
        # Both axles must be below limit for this G to be achievable
        feasible = jnp.where((util_f <= 1.0) & (util_r <= 1.0), 1.0, 0.0)
        
        return ay_g * balance * feasible
    
    grip_scores = jax.vmap(compute_balance_at_ay)(ay_sweep)
    
    # Max achievable balanced grip
    obj_grip = jnp.max(grip_scores)
    
    # Safety: penalise if front saturates before rear (oversteer)
    util_at_1g_f = (m * g * lr / (lf + lr)) / (
        2.0 * PDY1 * (1.0 + PDY2 * ((Fz_f_static/2 - m*g*h_cg/t_w*lltd_f - 1000)/1000)) * 
        jnp.maximum(Fz_f_static/2, 10.0) + 1e-3
    )
    safety_margin = lltd_f - lltd_r   # positive = understeer bias = safe
    
    return obj_grip, safety_margin


def compute_frequency_response_objective(simulate_step_fn, params, x_init, dt=0.005, T_max=2.0):
    """
    Analytical damping ratio estimate from setup params.
    A well-damped car has damping ratio 0.6-0.7 on heave and roll modes.
    Penalise under/over-damped setups.
    """
    from data.configs.vehicle_params import vehicle_params as VP
    
    k_f, k_r = params[0], params[1]
    c_f, c_r = params[4], params[5]
    
    mr_f, mr_r = 1.2, 1.15
    wheel_rate_f = k_f / (mr_f ** 2)
    wheel_rate_r = k_r / (mr_r ** 2)
    damp_rate_f  = c_f / (mr_f ** 2)
    damp_rate_r  = c_r / (mr_r ** 2)
    
    m_s  = VP.get('m', 300.0) * 0.85
    m_us = VP.get('m', 300.0) * 0.0375
    
    # Heave natural frequency and damping ratio
    k_heave = wheel_rate_f * 2 + wheel_rate_r * 2
    c_heave  = damp_rate_f  * 2 + damp_rate_r  * 2
    omega_n_heave = jnp.sqrt(k_heave / (m_s + 1e-3))
    zeta_heave = c_heave / (2.0 * jnp.sqrt(k_heave * m_s) + 1e-3)
    
    # Unsprung mass natural frequency (wheel hop) — want this well-damped too
    k_us = wheel_rate_f + 95000.0  # wheel rate + tyre vertical stiffness
    zeta_us = damp_rate_f / (2.0 * jnp.sqrt(k_us * m_us) + 1e-3)
    
    # Target: zeta = 0.65 for both modes
    # Penalise deviation from ideal damping
    target_zeta = 0.65
    resonance = (zeta_heave - target_zeta)**2 + (zeta_us - target_zeta)**2
    
    return resonance