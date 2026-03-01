import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
import os
import flax.serialization

from models.tire_model import PacejkaTire


class PhysicsNormalizer:
    q_mean  = jnp.array([0., 0., 0.3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    q_scale = jnp.array([100., 100., 0.1, 0.1, 0.05, jnp.pi,
                          0.05, 0.05, 0.05, 0.05,
                          500., 500., 500., 500.])

    v_mean  = jnp.zeros(14)
    v_scale = jnp.array([25., 5., 1., 1., 1., 1., 1., 1., 1., 1., 50., 50., 50., 50.])

    setup_mean  = jnp.array([40000., 40000., 500., 500., 3000., 3000., 0.3, 0.60])
    setup_scale = jnp.array([20000., 20000., 500., 500., 2000., 2000., 0.05, 0.10])

    @staticmethod
    def normalize_q(q):
        return (q - PhysicsNormalizer.q_mean) / (PhysicsNormalizer.q_scale + 1e-6)

    @staticmethod
    def normalize_v(v):
        return (v - PhysicsNormalizer.v_mean) / (PhysicsNormalizer.v_scale + 1e-6)

    @staticmethod
    def normalize_setup(setup):
        return (setup - PhysicsNormalizer.setup_mean) / (PhysicsNormalizer.setup_scale + 1e-6)


class DifferentiableAeroMap(nn.Module):
    base_A:   float
    base_Cl:  float
    base_Cd:  float
    lf:       float
    lr:       float

    @nn.compact
    def __call__(self, vx, pitch, roll, heave_f, heave_r):
        state = jnp.stack([vx / 100.0, pitch, roll, heave_f, heave_r])
        x = nn.Dense(32)(state)
        x = nn.swish(x)
        x = nn.Dense(32)(x)
        x = nn.swish(x)

        modifiers = nn.Dense(4, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)

        h_ref = 0.040
        h_f   = jnp.maximum(0.040 - heave_f, 0.015)
        h_r   = jnp.maximum(0.040 - heave_r, 0.015)

        Cl_f_base = (self.base_Cl * 0.40 * (1.0 + 0.30 * (h_ref / h_f - 1.0)) + 0.35 * pitch)
        Cl_r_base = self.base_Cl * 0.60 * (1.0 + 0.45 * (h_ref / h_r - 1.0))

        Cd_dynamic = self.base_Cd + modifiers[1]

        Cl_f_net = jnp.maximum(Cl_f_base + modifiers[0] * 0.40, 0.0)
        Cl_r_net = jnp.maximum(Cl_r_base + modifiers[0] * 0.60, 0.0)

        q_dyn = 0.5 * 1.225 * (vx ** 2)

        Fz_aero_f = q_dyn * Cl_f_net * self.base_A
        Fz_aero_r = q_dyn * Cl_r_net * self.base_A
        Fx_aero   = -q_dyn * Cd_dynamic * self.base_A

        My_aero = (Fz_aero_r * self.lr - Fz_aero_f * self.lf + (Fz_aero_f + Fz_aero_r) * modifiers[2])
        Mx_aero = (Fz_aero_f + Fz_aero_r) * modifiers[3]

        return Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero


class NeuralEnergyLandscape(nn.Module):
    M_diag:  jnp.ndarray
    h_scale: float = 1.0
    H_RESIDUAL_CAP: float = 5000.0   

    @nn.compact
    def __call__(self, q, p, setup_params):
        T_prior = 0.5 * jnp.sum((p ** 2) / self.M_diag)

        v          = p / self.M_diag
        q_norm     = PhysicsNormalizer.normalize_q(q)
        v_norm     = PhysicsNormalizer.normalize_v(v)
        setup_norm = PhysicsNormalizer.normalize_setup(setup_params)

        # ── H_NET MASK (Prevents Braking/Steering Interference) ──
        q_norm_blind = q_norm.at[0:2].set(0.0) 
        q_norm_blind = q_norm_blind.at[5].set(0.0)
        q_norm_blind = q_norm_blind.at[10:14].set(0.0) 
        
        v_norm_blind = v_norm.at[0:2].set(0.0) 
        v_norm_blind = v_norm_blind.at[5].set(0.0)   
        v_norm_blind = v_norm_blind.at[10:14].set(0.0) 
        # ─────────────────────────────────────────────────────────

        state = jnp.concatenate([q_norm_blind, v_norm_blind, setup_norm])
        x = nn.Dense(128)(state)
        x = nn.swish(x)
        x = nn.Dense(64)(x)
        x = nn.swish(x)

        H_residual_raw = nn.Dense(1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)[0]

        V_structural = 0.5 * jnp.sum(q[6:10] ** 2) * 30000.0

        # ── THE TRUE SILVER BULLET ────────────────────────────────
        # Multiply by the suspension displacements squared.
        # By the product rule, both energy AND forces are exactly 0.0 at equilibrium!
        susp_norm_sq = jnp.sum(q[6:10] ** 2)
        H_residual_physical = H_residual_raw * self.h_scale * susp_norm_sq
        # ──────────────────────────────────────────────────────────

        return T_prior + V_structural + H_residual_physical


class NeuralDissipationMatrix(nn.Module):
    dim: int = 14

    @nn.compact
    def __call__(self, q, p):
        q_norm  = PhysicsNormalizer.normalize_q(q)
        p_scale = PhysicsNormalizer.v_scale * 200.0
        p_norm  = p / (p_scale + 1e-6)

        state = jnp.concatenate([q_norm, p_norm])
        x = nn.Dense(128)(state)
        x = nn.swish(x)
        x = nn.Dense(64)(x)
        x = nn.swish(x)

        num_elements = self.dim * (self.dim + 1) // 2
        L_elements   = nn.Dense(num_elements, kernel_init=jax.nn.initializers.lecun_normal(), bias_init=jax.nn.initializers.zeros)(x)

        L       = jnp.zeros((self.dim, self.dim))
        indices = jnp.tril_indices(self.dim)
        L       = L.at[indices].set(L_elements)

        R_dense = jnp.dot(L, L.T)

        mask = jnp.array([0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.])
        mask_matrix = jnp.outer(mask, mask)
        
        return R_dense * mask_matrix


class DifferentiableMultiBodyVehicle:
    def __init__(self, vehicle_params, tire_coeffs, rng_seed=42):
        self.vp   = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)

        self.m_us_f    = self.vp.get('unsprung_mass_f', 10.0)
        self.m_us_r    = self.vp.get('unsprung_mass_r', 11.0)
        self.m_unsprung = 2.0 * self.m_us_f + 2.0 * self.m_us_r

        self.m   = self.vp.get('total_mass', 230.0)
        self.m_s = self.m - self.m_unsprung

        self.Ix  = self.vp.get('Ix', 45.0)
        self.Iy  = self.vp.get('Iy', 85.0)
        self.Iz  = self.vp.get('Iz', 125.0)
        self.Iw  = self.vp.get('Iw', 1.2)

        self.lf      = self.vp.get('lf', 0.680)
        self.lr      = self.vp.get('lr', 0.920)
        self.track_w = 1.2
        self.h_cg    = self.vp.get('h_cg', 0.285)
        self.g       = 9.81

        self.M_diag = jnp.array([
            self.m_s, self.m_s, self.m_s, self.Ix, self.Iy, self.Iz,
            self.m_us_f, self.m_us_f, self.m_us_r, self.m_us_r,
            self.Iw, self.Iw, self.Iw, self.Iw,
        ])

        current_dir  = os.path.dirname(os.path.abspath(__file__))
        h_scale_path = os.path.join(current_dir, 'h_net_scale.txt')
        
        if os.path.exists(h_scale_path):
            with open(h_scale_path) as f:
                _h_scale = float(f.read().strip())
            print(f"[VehicleDynamics] H_net scale loaded: {_h_scale:.2f} J")
        else:
            _h_scale = 1.0
            print("[VehicleDynamics] h_net_scale.txt not found — using h_scale=1.0.")

        self.H_net = NeuralEnergyLandscape(
            M_diag=self.M_diag,
            h_scale=_h_scale,
            H_RESIDUAL_CAP=5000.0,
        )
        self.R_net    = NeuralDissipationMatrix(dim=14)
        self.aero_map = DifferentiableAeroMap(
            base_A=self.vp.get('A_ref',  1.1),
            base_Cl=self.vp.get('Cl_ref', 3.0),
            base_Cd=self.vp.get('Cd_ref', 1.5),
            lf=self.lf,
            lr=self.lr,
        )

        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(rng_seed), 3)
        dummy_q, dummy_p = jnp.zeros(14), jnp.zeros(14)
        dummy_setup      = jnp.zeros(8)

        self.H_params    = self.H_net.init(key1, dummy_q, dummy_p, dummy_setup)
        self.R_params    = self.R_net.init(key2, dummy_q, dummy_p)
        self.Aero_params = self.aero_map.init(key3, 0.0, 0.0, 0.0, 0.0, 0.0)

        h_path = os.path.join(current_dir, 'h_net.bytes')
        r_path = os.path.join(current_dir, 'r_net.bytes')

        if os.path.exists(h_path):
            with open(h_path, 'rb') as f:
                self.H_params = flax.serialization.from_bytes(self.H_params, f.read())
        if os.path.exists(r_path):
            with open(r_path, 'rb') as f:
                self.R_params = flax.serialization.from_bytes(self.R_params, f.read())

    def _damper_force(self, vz, c_bump_low, c_bump_high, c_reb_low,  c_reb_high, v_transition=0.10):
        sharpness = 50.0
        w_reb  = 0.5 * (1.0 + jnp.tanh(sharpness * vz))
        c_low  = w_reb * c_reb_low  + (1.0 - w_reb) * c_bump_low
        c_high = w_reb * c_reb_high + (1.0 - w_reb) * c_bump_high
        abs_vz = jnp.abs(vz)
        excess = jax.nn.softplus(100.0 * (abs_vz - v_transition)) / 100.0
        f_mag  = c_low * jnp.minimum(abs_vz, v_transition) + c_high * excess
        return -jnp.tanh(sharpness * vz) * f_mag

    def compute_drive_force(self, throttle, vx):
        throttle_s = jnp.clip(throttle, 0.0, 1.0)
        T_peak     = self.vp.get('motor_peak_torque',    140.0)
        P_peak     = self.vp.get('motor_peak_power',  60000.0)
        R_wheel    = self.vp.get('wheel_radius',        0.2032)
        eta        = self.vp.get('drivetrain_efficiency', 0.92)
        fd         = self.vp.get('final_drive_ratio',      4.5)
        vx_s     = jnp.maximum(jnp.abs(vx), 0.1)
        F_torque = throttle_s * T_peak * fd * eta / R_wheel
        F_power  = throttle_s * P_peak * eta / vx_s
        return jnp.minimum(F_torque, F_power)

    def compute_brake_forces(self, brake_total, Fz_f, Fz_r, vx, brake_bias_f):
        eps        = 1e-6
        use_ideal  = self.vp.get('ideal_brake_balance', False)
        Fz_f_s     = jnp.maximum(Fz_f, eps)
        Fz_r_s     = jnp.maximum(Fz_r, eps)
        ideal_bias = Fz_f_s / (Fz_f_s + Fz_r_s)
        bias       = jnp.where(use_ideal, ideal_bias, brake_bias_f)
        return -brake_total * bias, -brake_total * (1.0 - bias)

    def compute_differential_forces(self, T_drive, vx, wz, Fz_rl, Fz_rr, alpha_t_rl, alpha_t_rr, gamma_r, T_ribs_r, T_gas_r):
        eps        = 1e-6
        R_wheel    = self.vp.get('wheel_radius',          0.2032)
        tr         = self.track_w
        lock_ratio = self.vp.get('diff_lock_ratio',          1.0)
        eta        = self.vp.get('drivetrain_efficiency',    0.92)
        vx_s   = jnp.maximum(jnp.abs(vx), 0.5)
        v_rl   = vx_s * (1.0 - wz * tr / (2.0 * vx_s))
        v_rr   = vx_s * (1.0 + wz * tr / (2.0 * vx_s))
        omega_diff = (v_rl + v_rr) / (2.0 * R_wheel)
        d_omega    = omega_diff - v_rl / R_wheel
        T_lock     = lock_ratio * 500.0 * d_omega
        T_rl_input = T_drive * 0.5 * eta - T_lock
        T_rr_input = T_drive * 0.5 * eta + T_lock
        omega_rl = v_rl / R_wheel + T_rl_input / (10.0 * R_wheel)
        omega_rr = v_rr / R_wheel + T_rr_input / (10.0 * R_wheel)
        kappa_rl = jnp.clip((omega_rl * R_wheel - v_rl) / jnp.maximum(v_rl, eps), -0.5, 0.5)
        kappa_rr = jnp.clip((omega_rr * R_wheel - v_rr) / jnp.maximum(v_rr, eps), -0.5, 0.5)
        Fx_rl, Fy_rl = self.tire.compute_force(alpha_t_rl, kappa_rl, Fz_rl, gamma_r, T_ribs_r, T_gas_r, vx_s)
        Fx_rr, Fy_rr = self.tire.compute_force(alpha_t_rr, kappa_rr, Fz_rr, gamma_r, T_ribs_r, T_gas_r, vx_s)
        return Fx_rl, Fx_rr, Fy_rl, Fy_rr, kappa_rl, kappa_rr

    @partial(jax.jit, static_argnums=(0,))
    def _compute_derivatives(self, x, u, setup_params):
        q, v = x[0:14], x[14:28]
        p    = self.M_diag * v

        grad_H_fn    = jax.grad(self.H_net.apply, argnums=(1, 2))
        dH_dq, dH_dp = grad_H_fn(self.H_params, q, p, setup_params)

        NEURAL_FORCE_CLAMP    = 12000.0
        NEURAL_VELOCITY_CLAMP =   150.0

        dH_dq = NEURAL_FORCE_CLAMP    * jnp.tanh(dH_dq / (NEURAL_FORCE_CLAMP    + 1e-8))
        dH_dp = NEURAL_VELOCITY_CLAMP * jnp.tanh(dH_dp / (NEURAL_VELOCITY_CLAMP + 1e-8))
        grad_H = jnp.concatenate([dH_dq, dH_dp])

        J = jnp.zeros((28, 28))
        J = J.at[0:14, 14:28].set( jnp.eye(14))
        J = J.at[14:28, 0:14].set(-jnp.eye(14))

        R = jnp.zeros((28, 28))
        R = R.at[14:28, 14:28].set(self.R_net.apply(self.R_params, q, p))

        X, Y, Z, phi_roll, theta_pitch, psi_yaw = q[0:6]
        vx, vy, vz, wx, wy, wz                  = v[0:6]

        vx          = jnp.clip(vx,          -80.0, 80.0)
        vy          = jnp.clip(vy,          -30.0, 30.0)
        wz          = jnp.clip(wz,            -8.0,  8.0)
        Z           = jnp.clip(Z,            -0.3,  1.5)
        phi_roll    = jnp.clip(phi_roll,    -0.3,  0.3)
        theta_pitch = jnp.clip(theta_pitch, -0.2,  0.2)

        z_fl = Z - (self.track_w / 2) * phi_roll - self.lf * theta_pitch
        z_fr = Z + (self.track_w / 2) * phi_roll - self.lf * theta_pitch
        z_rl = Z - (self.track_w / 2) * phi_roll + self.lr * theta_pitch
        z_rr = Z + (self.track_w / 2) * phi_roll + self.lr * theta_pitch

        heave_f = (z_fl + z_fr) / 2.0
        heave_r = (z_rl + z_rr) / 2.0

        vz_fl = vz - (self.track_w / 2) * wx - self.lf * wy
        vz_fr = vz + (self.track_w / 2) * wx - self.lf * wy
        vz_rl = vz - (self.track_w / 2) * wx + self.lr * wy
        vz_rr = vz + (self.track_w / 2) * wx + self.lr * wy

        k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg, brake_bias_f = setup_params

        mr_f_p = jnp.array(self.vp.get('motion_ratio_f_poly', [1.20, 0.0, 0.0]))
        mr_r_p = jnp.array(self.vp.get('motion_ratio_r_poly', [1.15, 0.0, 0.0]))
        mr_f   = jnp.clip(mr_f_p[0] + mr_f_p[1] * heave_f + mr_f_p[2] * (heave_f ** 2), 0.8, 2.5)
        mr_r   = jnp.clip(mr_r_p[0] + mr_r_p[1] * heave_r + mr_r_p[2] * (heave_r ** 2), 0.8, 2.5)

        wheel_rate_f = k_f  / (mr_f ** 2)
        wheel_rate_r = k_r  / (mr_r ** 2)
        arb_rate_f   = arb_f / (mr_f ** 2)
        arb_rate_r   = arb_r / (mr_r ** 2)
        damp_rate_f  = c_f  / (mr_f ** 2)
        damp_rate_r  = c_r  / (mr_r ** 2)

        delta       = u[0]
        throttle    = jnp.clip( u[1] / 2000.0,  0.0, 1.0)
        brake_force = jnp.clip(-u[1],            0.0, 10000.0)

        Fx_drive          = self.compute_drive_force(throttle, vx)
        Fz_f_static_base  = self.m_s * self.g * self.lr / (self.lf + self.lr)
        Fz_r_static_base  = self.m_s * self.g * self.lf / (self.lf + self.lr)

        Fx_brake_f, Fx_brake_r = self.compute_brake_forces(
            brake_force, Fz_f_static_base, Fz_r_static_base, vx, brake_bias_f
        )

        Fz_aero_f, Fz_aero_r, Fx_aero, My_aero, Mx_aero = self.aero_map.apply(
            self.Aero_params, vx, theta_pitch, phi_roll, heave_f, heave_r
        )

        ay = vx * wz
        ax = (Fx_drive + Fx_brake_f + Fx_brake_r + Fx_aero) / self.m

        h_rc_f = jnp.clip(self.vp.get('h_rc_f', 0.030) + self.vp.get('dh_rc_dz_f', 0.20) * heave_f, 0.0, 0.20)
        h_rc_r = jnp.clip(self.vp.get('h_rc_r', 0.050) + self.vp.get('dh_rc_dz_r', 0.30) * heave_r, 0.0, 0.25)

        W_s    = self.m_s * self.g
        W_us_f = self.m_us_f * self.g
        W_us_r = self.m_us_r * self.g
        L      = self.lf + self.lr
        tf, tr = self.track_w, self.track_w

        k_phi_f   = (wheel_rate_f * 2) * (tf / 2) ** 2 * 0.5 + arb_rate_f * (tf / 2) ** 2
        k_phi_r   = (wheel_rate_r * 2) * (tr / 2) ** 2 * 0.5 + arb_rate_r * (tr / 2) ** 2
        k_phi_tot = k_phi_f + k_phi_r + 1.0

        LLT_geom_f = W_s * ay * h_rc_f / tf
        LLT_geom_r = W_s * ay * h_rc_r / tr

        W_sf       = W_s * self.lr / L
        W_sr       = W_s * self.lf / L
        h_roll_arm = h_cg - (h_rc_f * W_sf + h_rc_r * W_sr) / jnp.maximum(W_s, 1.0)
        LLT_elast  = W_s * ay * h_roll_arm / k_phi_tot

        LLT_f = LLT_geom_f + LLT_elast * (k_phi_f / k_phi_tot)
        LLT_r = LLT_geom_r + LLT_elast * (k_phi_r / k_phi_tot)

        LLT_long   = self.m * ax * h_cg / L
        LLT_long_f = LLT_long * brake_bias_f
        LLT_long_r = LLT_long * (1.0 - brake_bias_f)

        Fx_r_accel = jnp.maximum(Fx_drive, 0.0)
        dFz_squat  = Fx_r_accel  * h_cg / L * self.vp.get('anti_squat',  0.30)
        dFz_lift   = Fx_r_accel  * h_cg / L * self.vp.get('anti_lift',   0.20)
        dFz_dive_f = -Fx_brake_f * h_cg / L * self.vp.get('anti_dive_f', 0.40)
        dFz_dive_r = -Fx_brake_r * h_cg / L * self.vp.get('anti_dive_r', 0.10)

        W_total = self.m * self.g
        _sp = lambda fz: jax.nn.softplus(fz * 200.0) / 200.0
        
        Fz_fl = _sp(W_total * self.lr / (L * 2) - LLT_f - LLT_long_f - dFz_dive_f / 2 + dFz_lift / 2 + Fz_aero_f / 2 + W_us_f / 2)
        Fz_fr = _sp(W_total * self.lr / (L * 2) + LLT_f - LLT_long_f - dFz_dive_f / 2 + dFz_lift / 2 + Fz_aero_f / 2 + W_us_f / 2)
        Fz_rl = _sp(W_total * self.lf / (L * 2) - LLT_r + LLT_long_r + dFz_squat / 2 - dFz_dive_r / 2 + Fz_aero_r / 2 + W_us_r / 2)
        Fz_rr = _sp(W_total * self.lf / (L * 2) + LLT_r + LLT_long_r + dFz_squat / 2 - dFz_dive_r / 2 + Fz_aero_r / 2 + W_us_r / 2)

        bs_f  = self.vp.get('bump_steer_f', 0.0)
        bs_r  = self.vp.get('bump_steer_r', 0.0)
        d_ack = delta * self.vp.get('ackermann_factor', 0.50) * (self.track_w / 2.0) / L

        delta_fl = delta - d_ack
        delta_fr = delta + d_ack

        vx_safe = jnp.maximum(vx, 1.0)
        beta    = jnp.arctan2(vy, vx_safe)

        alpha_kin_fl = delta_fl + bs_f * z_fl - (beta + self.lf * wz / vx_safe)
        alpha_kin_fr = delta_fr + bs_f * z_fr - (beta + self.lf * wz / vx_safe)
        alpha_kin_rl = bs_r * z_rl             - (beta - self.lr * wz / vx_safe)
        alpha_kin_rr = bs_r * z_rr             - (beta - self.lr * wz / vx_safe)

        alpha_t_fl, kappa_t_fl = x[38], x[39]
        alpha_t_fr, kappa_t_fr = x[40], x[41]
        alpha_t_rl, kappa_t_rl = x[42], x[43]
        alpha_t_rr, kappa_t_rr = x[44], x[45]

        C_cs_f = jnp.deg2rad(self.vp.get('compliance_steer_f', -0.15)) / 1000.0
        C_cs_r = jnp.deg2rad(self.vp.get('compliance_steer_r', -0.10)) / 1000.0

        _Fz0_tc = self.tire.coeffs.get('FNOMIN', 1000.0)
        _PKY1   = self.tire.coeffs.get('PKY1',   15.324)
        _PKY2   = self.tire.coeffs.get('PKY2',    1.715)
        _PKY4   = self.tire.coeffs.get('PKY4',    2.0)

        def _Ky(Fz_c):
            return _PKY1 * _Fz0_tc * jnp.sin(_PKY4 * jnp.arctan(Fz_c / jnp.maximum(_PKY2 * _Fz0_tc, 1e-6)))

        Fy_prev_fl = _Ky(Fz_fl) * alpha_t_fl
        Fy_prev_fr = _Ky(Fz_fr) * alpha_t_fr
        Fy_prev_rl = _Ky(Fz_rl) * alpha_t_rl
        Fy_prev_rr = _Ky(Fz_rr) * alpha_t_rr

        alpha_ss_fl = jnp.clip(alpha_kin_fl + C_cs_f * Fy_prev_fl, -1.5, 1.5)
        alpha_ss_fr = jnp.clip(alpha_kin_fr + C_cs_f * Fy_prev_fr, -1.5, 1.5)
        alpha_ss_rl = jnp.clip(alpha_kin_rl + C_cs_r * Fy_prev_rl, -1.5, 1.5)
        alpha_ss_rr = jnp.clip(alpha_kin_rr + C_cs_r * Fy_prev_rr, -1.5, 1.5)

        kappa_ss_f = 0.0
        d_alpha_fl, d_kappa_fl = self.tire.compute_transient_slip_derivatives(alpha_ss_fl, kappa_ss_f, alpha_t_fl, kappa_t_fl, Fz_fl, vx)
        d_alpha_fr, d_kappa_fr = self.tire.compute_transient_slip_derivatives(alpha_ss_fr, kappa_ss_f, alpha_t_fr, kappa_t_fr, Fz_fr, vx)
        d_alpha_rl, _ = self.tire.compute_transient_slip_derivatives(alpha_ss_rl, 0.0, alpha_t_rl, kappa_t_rl, Fz_rl, vx)
        d_alpha_rr, _ = self.tire.compute_transient_slip_derivatives(alpha_ss_rr, 0.0, alpha_t_rr, kappa_t_rr, Fz_rr, vx)

        gamma_f = jnp.clip(self.vp.get('static_camber_f', -2.0) + jnp.rad2deg(phi_roll) * self.vp.get('camber_gain_f', -0.8), -10.0, 5.0)
        gamma_r = jnp.clip(self.vp.get('static_camber_r', -1.5) + jnp.rad2deg(phi_roll) * self.vp.get('camber_gain_r', -0.6), -10.0, 5.0)

        T_core_f = jnp.maximum(x[28], 20.)
        T_ribs_f = jnp.clip(x[29:32], 20., 150.)
        T_gas_f  = jnp.maximum(x[32], 20.)
        T_core_r = jnp.maximum(x[33], 20.)
        T_ribs_r = jnp.clip(x[34:37], 20., 150.)
        T_gas_r  = jnp.maximum(x[37], 20.)

        Fx_fl, Fy_fl = self.tire.compute_force(alpha_t_fl, kappa_t_fl, Fz_fl, gamma_f, T_ribs_f, T_gas_f, vx)
        Fx_fr, Fy_fr = self.tire.compute_force(alpha_t_fr, kappa_t_fr, Fz_fr, gamma_f, T_ribs_f, T_gas_f, vx)

        T_drive_wheel = Fx_drive * self.vp.get('wheel_radius', 0.2032)
        Fx_rl, Fx_rr, Fy_rl, Fy_rr, k_rl_diff, k_rr_diff = self.compute_differential_forces(
            T_drive_wheel, vx, wz, Fz_rl, Fz_rr, alpha_t_rl, alpha_t_rr, gamma_r, T_ribs_r, T_gas_r
        )

        d_kappa_rl = (k_rl_diff - kappa_t_rl) / 0.005
        d_kappa_rr = (k_rr_diff - kappa_t_rr) / 0.005

        Fx_f, Fy_f = Fx_fl + Fx_fr, Fy_fl + Fy_fr
        Fx_r, Fy_r = Fx_rl + Fx_rr, Fy_rl + Fy_rr

        Mz_fl    = self.tire.compute_aligning_torque(alpha_t_fl, kappa_t_fl, Fz_fl, gamma_f, Fy_fl)
        Mz_fr    = self.tire.compute_aligning_torque(alpha_t_fr, kappa_t_fr, Fz_fr, gamma_f, Fy_fr)
        Mz_rl    = self.tire.compute_aligning_torque(alpha_t_rl, kappa_t_rl, Fz_rl, gamma_r, Fy_rl)
        Mz_rr    = self.tire.compute_aligning_torque(alpha_t_rr, kappa_t_rr, Fz_rr, gamma_r, Fy_rr)
        Mz_total = Mz_fl + Mz_fr + Mz_rl + Mz_rr
        M_diff   = (Fx_rr - Fx_rl) * (self.track_w / 2.0)

        c_bump_f  = damp_rate_f * 0.8;   c_reb_f  = damp_rate_f * 1.5
        c_bump_r  = damp_rate_r * 0.8;   c_reb_r  = damp_rate_r * 1.5
        c_hi_bf   = c_bump_f * 0.4;      c_hi_rf  = c_reb_f  * 0.4
        c_hi_br   = c_bump_r * 0.4;      c_hi_rr  = c_reb_r  * 0.4

        F_gas_f = self.vp.get('damper_gas_force_f', 120.0) / mr_f
        F_gas_r = self.vp.get('damper_gas_force_r', 120.0) / mr_r

        bump_fl = jnp.maximum(0.0, jnp.maximum(0.0, -z_fl) - 0.025)
        bump_fr = jnp.maximum(0.0, jnp.maximum(0.0, -z_fr) - 0.025)
        bump_rl = jnp.maximum(0.0, jnp.maximum(0.0, -z_rl) - 0.025)
        bump_rr = jnp.maximum(0.0, jnp.maximum(0.0, -z_rr) - 0.025)

        F_spring_fl = (-wheel_rate_f * z_fl + self._damper_force(vz_fl, c_bump_f, c_hi_bf, c_reb_f, c_hi_rf) - 50000.0 * bump_fl + F_gas_f)
        F_spring_fr = (-wheel_rate_f * z_fr + self._damper_force(vz_fr, c_bump_f, c_hi_bf, c_reb_f, c_hi_rf) - 50000.0 * bump_fr + F_gas_f)
        F_spring_rl = (-wheel_rate_r * z_rl + self._damper_force(vz_rl, c_bump_r, c_hi_br, c_reb_r, c_hi_rr) - 50000.0 * bump_rl + F_gas_r)
        F_spring_rr = (-wheel_rate_r * z_rr + self._damper_force(vz_rr, c_bump_r, c_hi_br, c_reb_r, c_hi_rr) - 50000.0 * bump_rr + F_gas_r)

        F_arb_f_force = -arb_rate_f * (z_fl - z_fr)
        F_arb_r_force = -arb_rate_r * (z_rl - z_rr)

        F_susp_fl = F_spring_fl + F_arb_f_force
        F_susp_fr = F_spring_fr - F_arb_f_force
        F_susp_rl = F_spring_rl + F_arb_r_force
        F_susp_rr = F_spring_rr - F_arb_r_force

        Fz_susp_total = F_susp_fl + F_susp_fr + F_susp_rl + F_susp_rr
        Mx_susp = (F_susp_fr + F_susp_rr - F_susp_fl - F_susp_rl) * (self.track_w / 2.0)
        My_susp = (F_susp_rl + F_susp_rr) * self.lr - (F_susp_fl + F_susp_fr) * self.lf

        omega_spin = vx_safe / self.vp.get('wheel_radius', 0.2032)
        Mz_gyro    = -4.0 * self.Iw * omega_spin * wx
        My_gyro    =  4.0 * self.Iw * omega_spin * wz

        F_ext_mech = jnp.zeros(28)
        F_ext_mech = F_ext_mech.at[14].set(Fx_aero + Fx_f * jnp.cos(delta) - Fy_f * jnp.sin(delta) + Fx_r)
        F_ext_mech = F_ext_mech.at[15].set(Fx_f * jnp.sin(delta) + Fy_f * jnp.cos(delta) + Fy_r)
        F_ext_mech = F_ext_mech.at[16].set(-self.m_s * self.g - (Fz_aero_f + Fz_aero_r) + Fz_susp_total)
        F_ext_mech = F_ext_mech.at[17].set(Mx_aero + Mx_susp)
        F_ext_mech = F_ext_mech.at[18].set(My_aero + My_susp + My_gyro)
        F_ext_mech = F_ext_mech.at[19].set(self.lf * Fy_f * jnp.cos(delta) - self.lr * Fy_r + M_diff + Mz_total + Mz_gyro)

        dx_mech = jnp.dot((J - R), grad_H) + F_ext_mech

        m_eff   = self.m + self.vp.get('m_drivetrain_eff', 12.0)
        accel_x = F_ext_mech[14] / m_eff

        v_slide_f = vx * jnp.sqrt(
            0.25 * (kappa_t_fl ** 2 + kappa_t_fr ** 2) +
            jnp.tanh(0.5 * (alpha_t_fl + alpha_t_fr)) ** 2 + 1e-6
        )
        v_slide_r = vx * jnp.sqrt(
            0.25 * (kappa_t_rl ** 2 + kappa_t_rr ** 2) +
            jnp.tanh(0.5 * (alpha_t_rl + alpha_t_rr)) ** 2 + 1e-6
        )

        T_state_f = jnp.array([T_ribs_f[0], T_ribs_f[1], T_ribs_f[2], T_core_f])
        dT_f = self.tire.compute_thermal_derivatives(T_state_f, T_gas_f, Fz_f_static_base / 2, v_slide_f)
        dt_core_f, dT_ribs_f, dt_gas_f = dT_f[3], dT_f[0:3], dT_f[4]

        T_state_r = jnp.array([T_ribs_r[0], T_ribs_r[1], T_ribs_r[2], T_core_r])
        dT_r = self.tire.compute_thermal_derivatives(T_state_r, T_gas_r, Fz_r_static_base / 2, v_slide_r)
        dt_core_r, dT_ribs_r, dt_gas_r = dT_r[3], dT_r[0:3], dT_r[4]

        dx_dt = jnp.concatenate([
            dx_mech[0:14],
            jnp.concatenate([
                jnp.array([accel_x]),
                dx_mech[15:28] / self.M_diag[1:],
            ]),
            jnp.array([dt_core_f]), dT_ribs_f, jnp.array([dt_gas_f]),
            jnp.array([dt_core_r]), dT_ribs_r, jnp.array([dt_gas_r]),
            jnp.array([d_alpha_fl, d_kappa_fl, d_alpha_fr, d_kappa_fr,
                       d_alpha_rl, d_kappa_rl, d_alpha_rr, d_kappa_rr]),
        ])
        return jnp.nan_to_num(dx_dt, nan=0.0, posinf=0.0, neginf=0.0)

    @partial(jax.jit, static_argnums=(0,))
    def simulate_step(self, x, u, setup_params, dt=0.005):
        def single_leapfrog(state, dt_step):
            dx = self._compute_derivatives(state, u, setup_params)
            v_half   = state[14:28] + 0.5 * dt_step * dx[14:28]
            x_half   = state.at[14:28].set(v_half)
            dx_half  = self._compute_derivatives(x_half, u, setup_params)
            q_next   = state[0:14] + dt_step * dx_half[0:14]
            x_q      = x_half.at[0:14].set(q_next)
            dx_final = self._compute_derivatives(x_q, u, setup_params)
            v_next = v_half + 0.5 * dt_step * dx_final[14:28]
            therm_next = state[28:38] + dt_step * dx[28:38]
            trans_next = state[38:46] + dt_step * dx[38:46]
            return (x_q.at[14:28].set(v_next)
                       .at[28:38].set(therm_next)
                       .at[38:46].set(trans_next))

        dt_sub = dt / 5.0
        def scan_fn(carry, _):
            return single_leapfrog(carry, dt_sub), None
        x_out, _ = jax.lax.scan(scan_fn, x, None, length=5)

        x_out = x_out.at[14].set(jnp.clip(x_out[14], -80.0, 80.0))
        x_out = x_out.at[15].set(jnp.clip(x_out[15], -30.0, 30.0))
        x_out = x_out.at[17:20].set(jnp.clip(x_out[17:20], -8.0, 8.0))
        x_out = x_out.at[2].set(jnp.clip(x_out[2], -0.5, 2.0))
        x_out = x_out.at[3:5].set(jnp.clip(x_out[3:5], -0.4, 0.4))
        x_out = x_out.at[38:46].set(jnp.clip(x_out[38:46], -0.5, 0.5))

        return x_out


class DynamicBicycleModel:
    def __init__(self, vehicle_params, tire_coeffs):
        self.vp   = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)

    def get_equations(self):
        try:
            import casadi as ca
        except ImportError:
            raise ImportError("CasADi is required for DynamicBicycleModel")

        n, alpha, v, delta, r = [ca.MX.sym(name) for name in ['n', 'alpha', 'v', 'delta', 'r']]
        x_mech    = ca.vertcat(n, alpha, v, delta, r)
        u_d_delta = ca.MX.sym('u_d_delta')
        u_fx      = ca.MX.sym('u_fx')
        u         = ca.vertcat(u_d_delta, u_fx)
        k_c       = ca.MX.sym('k_c')

        m, Iz    = self.vp['m'], self.vp['Iz']
        lf, lr   = self.vp['lf'], self.vp['lr']
        rho, A, Cl = 1.225, self.vp['A'], self.vp['Cl']

        Fz_aero = 0.5 * rho * Cl * A * v ** 2
        Fz_f    = m * 9.81 * lr / (lf + lr) + Fz_aero / 2
        Fz_r    = m * 9.81 * lf / (lf + lr) + Fz_aero / 2

        alpha_f = delta - (alpha + lf * r / v)
        alpha_r = -(alpha - lr * r / v)

        T_ribs_dummy = [90.0, 90.0, 90.0]
        Fx_f, Fy_f = self.tire.compute_force(alpha_f, 0, Fz_f, -1.5, T_ribs_dummy, 60.0, v)
        Fx_r, Fy_r = self.tire.compute_force(alpha_r, 0, Fz_r, -1.0, T_ribs_dummy, 60.0, v)

        n_dot     = v * ca.sin(alpha)
        alpha_dot = ((Fy_f * ca.cos(delta) + Fy_r) / m - v * r) / v
        v_dot     = u_fx / m
        r_dot     = (lf * Fy_f * ca.cos(delta) - lr * Fy_r) / Iz
        delta_dot = u_d_delta

        rhs = ca.vertcat(n_dot, alpha_dot, v_dot, delta_dot, r_dot)
        return ca.Function('f_dyn', [x_mech, u, k_c], [rhs])
