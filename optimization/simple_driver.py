# optimization/simple_driver.py — 30 lines, always works
import numpy as np, math

def simple_driver_step(x_car, s_driven, track_s, track_k, track_psi,
                        racing_line, mu=1.40, m=235.0):
    """
    Pure feedforward driver from racing line.
    No optimization, no iLQR, no L-BFGS-B.
    Just: steer toward racing line heading, brake to v_ref.
    """
    s = s_driven % float(track_s[-1])
    rl_s   = racing_line.s.astype(np.float64)
    
    v_tgt  = float(np.interp(s, rl_s, racing_line.v_ref.astype(np.float64)))
    psi_tgt = float(np.interp(s, rl_s, np.unwrap(racing_line.psi.astype(np.float64))))
    kap_tgt = float(np.interp(s, rl_s, racing_line.kappa.astype(np.float64)))

    # Ackermann feedforward + heading error feedback
    L      = 1.55
    vx     = max(float(x_car[14]), 1.0)
    yaw    = float(x_car[5])
    x_pos, y_pos = float(x_car[0]), float(x_car[1])

    # Nearest racing line point for lateral error
    rx_tgt = float(np.interp(s, rl_s, racing_line.rx.astype(np.float64)))
    ry_tgt = float(np.interp(s, rl_s, racing_line.ry.astype(np.float64)))
    
    n_err     = (-math.sin(psi_tgt)*(x_pos - rx_tgt) 
                 + math.cos(psi_tgt)*(y_pos - ry_tgt))
    psi_err   = math.atan2(math.sin(yaw - psi_tgt), math.cos(yaw - psi_tgt))
    
    steer = math.atan(L * kap_tgt) - 0.5*n_err/max(vx,1.0) - 1.5*psi_err
    steer = float(np.clip(steer, -0.45, 0.45))

    # Speed tracking with friction budget
    v_err  = vx - v_tgt
    mu_g   = mu * 9.81
    a_lat  = vx**2 * abs(kap_tgt)
    lon_bud = m * math.sqrt(max(mu_g**2 - min(a_lat, mu_g)**2, 0.0))
    force  = float(np.clip(-800.0 * v_err, -lon_bud, lon_bud))

    return steer, force