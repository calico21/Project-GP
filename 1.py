import numpy as np
d = np.load("reports/calib_window0_debug.npz")

vx = d["vx_sim"][0]      # (WINDOW_LEN,) del rollout calibrado
wz = d["wz_real"][0]     # yaw real del mismo tramo
ay_real = d["ay_real"][0]

ay_expected = vx * wz    # centrípeta física esperada, m/s²
ratio = np.median(ay_expected / (ay_real + 1e-6))
print("ratio esperado/real:", ratio)