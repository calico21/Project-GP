import jax
import jax.numpy as jnp
from jax import jit, grad
import flax.linen as nn
from functools import partial
import casadi as ca
import os
import flax.serialization

from models.tire_model import PacejkaTire 

class PhysicsNormalizer:
    """
    Statically normalizes the 38-D physical state vector to approximately 
    zero-mean, unit-variance before feeding it to the Neural Components.
    Prevents gradient explosions from mixed-unit scales (meters vs Celsius).
    """
    # 14 Kinematic positions (q)
    # [X, Y, Z, Roll, Pitch, Yaw, z_fl, z_fr, z_rl, z_rr, w_fl, w_fr, w_rl, w_rr]
    q_mean = jnp.array([0., 0., 0.3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    q_scale = jnp.array([100., 100., 0.1, 0.1, 0.05, jnp.pi, 0.05, 0.05, 0.05, 0.05, 100., 100., 100., 100.])
    
    # 14 Kinematic velocities (v)
    v_mean = jnp.zeros(14)
    v_scale = jnp.array([25., 5., 1., 1., 1., 1., 1., 1., 1., 1., 50., 50., 50., 50.])
    
    # 7 Setup Parameters
    # [k_f, k_r, arb_f, arb_r, c_f, c_r, h_cg]
    setup_mean = jnp.array([40000., 40000., 500., 500., 3000., 3000., 0.3])
    setup_scale = jnp.array([20000., 20000., 500., 500., 2000., 2000., 0.05])

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
    base_A: float
    base_Cl: float
    base_Cd: float

    @nn.compact
    def __call__(self, vx, pitch, roll, z_chassis):
        state = jnp.stack([vx / 100.0, pitch, roll, z_chassis])
        
        x = nn.Dense(32)(state)
        x = nn.swish(x)
        x = nn.Dense(32)(x)
        x = nn.swish(x)
        
        modifiers = nn.Dense(4, kernel_init=jax.nn.initializers.zeros, 
                             bias_init=jax.nn.initializers.zeros)(x)
        
        Cl_dynamic = self.base_Cl + modifiers[0]
        Cd_dynamic = self.base_Cd + modifiers[1]
        
        q_dyn = 0.5 * 1.225 * (vx ** 2)
        Fz_aero = q_dyn * Cl_dynamic * self.base_A
        Fx_aero = -q_dyn * Cd_dynamic * self.base_A
        
        My_aero = Fz_aero * modifiers[2]
        Mx_aero = Fz_aero * modifiers[3]
        
        return Fz_aero, Fx_aero, My_aero, Mx_aero

class NeuralEnergyLandscape(nn.Module):
    M_diag: jnp.ndarray

    @nn.compact
    def __call__(self, q, p, setup_params):
        T_prior = 0.5 * jnp.sum((p ** 2) / self.M_diag)
        
        # --- NORMALIZATION FIX ---
        v = p / self.M_diag
        q_norm = PhysicsNormalizer.normalize_q(q)
        v_norm = PhysicsNormalizer.normalize_v(v)
        setup_norm = PhysicsNormalizer.normalize_setup(setup_params)
        
        # Neural Network sees normalized data
        state = jnp.concatenate([q_norm, v_norm, setup_norm])
        x = nn.Dense(128)(state)
        x = nn.swish(x)
        x = nn.Dense(64)(x)
        x = nn.swish(x)
        
        H_residual = nn.Dense(1, kernel_init=jax.nn.initializers.zeros, 
                              bias_init=jax.nn.initializers.zeros)(x)[0]
        
        V_structural = 0.5 * jnp.sum(q[6:10] ** 2) * 30000.0  
        
        return T_prior + V_structural + H_residual

class NeuralDissipationMatrix(nn.Module):
    dim: int = 14

    @nn.compact
    def __call__(self, q, p):
        # --- NORMALIZATION FIX ---
        # NOTE: R_net doesn't have access to M_diag directly in its current definition,
        # so we will normalize p using a generic velocity scale assumption for now,
        # or we can pass v directly to R_net from the caller. For simplicity:
        q_norm = PhysicsNormalizer.normalize_q(q)
        # Approximate p normalization (mass * velocity)
        p_scale = PhysicsNormalizer.v_scale * 200.0 
        p_norm = p / (p_scale + 1e-6)
        
        state = jnp.concatenate([q_norm, p_norm])
        x = nn.Dense(128)(state)
        x = nn.swish(x)
        x = nn.Dense(64)(x)
        x = nn.swish(x)
        
        num_elements = self.dim * (self.dim + 1) // 2
        L_elements = nn.Dense(num_elements, kernel_init=jax.nn.initializers.lecun_normal(), 
                      bias_init=jax.nn.initializers.zeros)(x)
        
        L = jnp.zeros((self.dim, self.dim))
        indices = jnp.tril_indices(self.dim)
        L = L.at[indices].set(L_elements)
        
        return jnp.dot(L, L.T)

class DifferentiableMultiBodyVehicle:
    def __init__(self, vehicle_params, tire_coeffs, rng_seed=42):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
        
        self.m_s = self.vp.get('m_s', self.vp['m'] * 0.85)
        self.m_us = self.vp.get('m_us', self.vp['m'] * 0.0375)
        self.Ix = self.vp.get('Ix', 200.0)
        self.Iy = self.vp.get('Iy', 800.0)
        self.Iz = self.vp['Iz']
        self.Iw = self.vp.get('Iw', 1.2)
        
        self.lf = self.vp['lf']
        self.lr = self.vp['lr']
        self.track_w = 1.2 
        self.h_cg = self.vp.get('h_cg', 0.3)
        self.g = 9.81

        self.M_diag = jnp.array([
            self.m_s, self.m_s, self.m_s, self.Ix, self.Iy, self.Iz,
            self.m_us, self.m_us, self.m_us, self.m_us,
            self.Iw, self.Iw, self.Iw, self.Iw
        ])

        self.H_net = NeuralEnergyLandscape(M_diag=self.M_diag)
        self.R_net = NeuralDissipationMatrix(dim=14)
        self.aero_map = DifferentiableAeroMap(
            base_A=self.vp['A'], 
            base_Cl=self.vp.get('Cl', 3.0), 
            base_Cd=self.vp.get('Cd', 1.0)
        )
        
        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(rng_seed), 3)
        
        dummy_q = jnp.zeros(14)
        dummy_p = jnp.zeros(14)
        dummy_setup = jnp.zeros(7)
        
        self.H_params = self.H_net.init(key1, dummy_q, dummy_p, dummy_setup)
        self.R_params = self.R_net.init(key2, dummy_q, dummy_p)
        self.Aero_params = self.aero_map.init(key3, 0.0, 0.0, 0.0, 0.0)
        
        # --- BUG 2 FIX: Load Pre-Trained Neural Weights ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        h_path = os.path.join(current_dir, 'h_net.bytes')
        r_path = os.path.join(current_dir, 'r_net.bytes')
        
        if os.path.exists(h_path):
            with open(h_path, 'rb') as f:
                self.H_params = flax.serialization.from_bytes(self.H_params, f.read())
            print("[Physics Engine] Successfully loaded pre-trained H_net weights.")
            
        if os.path.exists(r_path):
            with open(r_path, 'rb') as f:
                self.R_params = flax.serialization.from_bytes(self.R_params, f.read())
            print("[Physics Engine] Successfully loaded pre-trained R_net weights.")

    @partial(jax.jit, static_argnums=(0,)) 
    def _compute_derivatives(self, x, u, setup_params):
        q = x[0:14]
        v = x[14:28]
        p = self.M_diag * v
        
        grad_H_fn = jax.grad(self.H_net.apply, argnums=(1, 2))
        dH_dq, dH_dp = grad_H_fn(self.H_params, q, p, setup_params)
        grad_H = jnp.concatenate([dH_dq, dH_dp])
        
        J = jnp.zeros((28, 28))
        J = J.at[0:14, 14:28].set(jnp.eye(14))
        J = J.at[14:28, 0:14].set(-jnp.eye(14))
        
        D_14 = self.R_net.apply(self.R_params, q, p)
        R = jnp.zeros((28, 28))
        R = R.at[14:28, 14:28].set(D_14)
        
        X, Y, Z, phi_roll, theta_pitch, psi_yaw = q[0:6]
        vx, vy, vz, wx, wy, wz = v[0:6]
        
        Fz_aero, Fx_aero, My_aero, Mx_aero = self.aero_map.apply(
            self.Aero_params, vx, theta_pitch, phi_roll, Z
        )

        k_f, k_r = setup_params[0], setup_params[1]
        arb_f, arb_r = setup_params[2], setup_params[3]
        c_f, c_r = setup_params[4], setup_params[5] 
        h_cg = setup_params[6]
        
        mr_f = self.vp.get('motion_ratio_f', 1.2)
        mr_r = self.vp.get('motion_ratio_r', 1.15)
        
        wheel_rate_f = k_f / (mr_f ** 2)
        wheel_rate_r = k_r / (mr_r ** 2)
        arb_rate_f = arb_f / (mr_f ** 2)
        arb_rate_r = arb_r / (mr_r ** 2)
        damp_rate_f = c_f / (mr_f ** 2)
        damp_rate_r = c_r / (mr_r ** 2)

        total_stiff_f = wheel_rate_f + arb_rate_f
        total_stiff_r = wheel_rate_r + arb_rate_r
        total_stiff = total_stiff_f + total_stiff_r + 1.0
        
        ay = vx * wz 
        W_transfer = (self.m_s * ay * h_cg) / self.track_w
        
        phi_chassis = (self.m_s * ay * h_cg) / total_stiff 
        phi_deg = jnp.rad2deg(phi_chassis)
        
        gamma_f = -2.0 + (phi_deg * 0.6) 
        gamma_r = -1.5 + (phi_deg * 0.4)
        
        dFz_f = W_transfer * (total_stiff_f / total_stiff)
        dFz_r = W_transfer * (total_stiff_r / total_stiff)

        Fz_f_static = (self.m_s * self.g * self.lr) / (self.lf + self.lr) + (Fz_aero * 0.4)
        Fz_r_static = (self.m_s * self.g * self.lf) / (self.lf + self.lr) + (Fz_aero * 0.6)

        Fz_fl = jnp.maximum(0.0, Fz_f_static/2 - dFz_f)
        Fz_fr = jnp.maximum(0.0, Fz_f_static/2 + dFz_f)
        Fz_rl = jnp.maximum(0.0, Fz_r_static/2 - dFz_r)
        Fz_rr = jnp.maximum(0.0, Fz_r_static/2 + dFz_r)

        z_fl = Z - (self.track_w / 2) * phi_roll - self.lf * theta_pitch
        z_fr = Z + (self.track_w / 2) * phi_roll - self.lf * theta_pitch
        z_rl = Z - (self.track_w / 2) * phi_roll + self.lr * theta_pitch
        z_rr = Z + (self.track_w / 2) * phi_roll + self.lr * theta_pitch

        vz_fl = vz - (self.track_w / 2) * wx - self.lf * wy
        vz_fr = vz + (self.track_w / 2) * wx - self.lf * wy
        vz_rl = vz - (self.track_w / 2) * wx + self.lr * wy
        vz_rr = vz + (self.track_w / 2) * wx + self.lr * wy

        F_spring_fl = -wheel_rate_f * z_fl - damp_rate_f * vz_fl
        F_spring_fr = -wheel_rate_f * z_fr - damp_rate_f * vz_fr
        F_spring_rl = -wheel_rate_r * z_rl - damp_rate_r * vz_rl
        F_spring_rr = -wheel_rate_r * z_rr - damp_rate_r * vz_rr
        
        F_arb_f_force = -arb_rate_f * (z_fl - z_fr)
        F_arb_r_force = -arb_rate_r * (z_rl - z_rr)

        F_susp_fl = F_spring_fl + F_arb_f_force
        F_susp_fr = F_spring_fr - F_arb_f_force
        F_susp_rl = F_spring_rl + F_arb_r_force
        F_susp_rr = F_spring_rr - F_arb_r_force
        
        Fz_susp_total = F_susp_fl + F_susp_fr + F_susp_rl + F_susp_rr
        Mx_susp = (F_susp_fr + F_susp_rr - F_susp_fl - F_susp_rl) * (self.track_w / 2.0)
        My_susp = (F_susp_rl + F_susp_rr) * self.lr - (F_susp_fl + F_susp_fr) * self.lf

        T_core_f = jnp.where(x[28] > 10, x[28], 60.0)
        T_ribs_f = jnp.clip(x[29:32], 10.0, 150.0)
        T_gas_f  = jnp.where(x[32] > 10, x[32], 25.0)
        
        T_core_r = jnp.where(x[33] > 10, x[33], 60.0)
        T_ribs_r = jnp.clip(x[34:37], 10.0, 150.0)
        T_gas_r  = jnp.where(x[37] > 10, x[37], 25.0)

        # Extract the transient slip states (46-DOF)
        alpha_t_fl, kappa_t_fl = x[38], x[39]
        alpha_t_fr, kappa_t_fr = x[40], x[41]
        alpha_t_rl, kappa_t_rl = x[42], x[43]
        alpha_t_rr, kappa_t_rr = x[44], x[45]

        delta = u[0] 
        vx_safe = jnp.maximum(vx, 1.0)
        beta = jnp.arctan2(vy, vx_safe)
        
        # Kinematic steady-state slip
        alpha_ss_f = delta - (beta + self.lf * wz / vx_safe)
        alpha_ss_r = -(beta - self.lr * wz / vx_safe)
        alpha_ss_f = jnp.clip(alpha_ss_f, -1.5, 1.5)
        alpha_ss_r = jnp.clip(alpha_ss_r, -1.5, 1.5)

        Fz_r_ref = jnp.maximum(Fz_r_static, 100.0)
        kappa_ss_r = jnp.clip(u[1] / (Fz_r_ref * 10.0), -0.3, 0.3)
        kappa_ss_f = 0.0

        # Calculate Transient Derivatives
        d_alpha_fl, d_kappa_fl = self.tire.compute_transient_slip_derivatives(alpha_ss_f, kappa_ss_f, alpha_t_fl, kappa_t_fl, Fz_fl, vx)
        d_alpha_fr, d_kappa_fr = self.tire.compute_transient_slip_derivatives(alpha_ss_f, kappa_ss_f, alpha_t_fr, kappa_t_fr, Fz_fr, vx)
        d_alpha_rl, d_kappa_rl = self.tire.compute_transient_slip_derivatives(alpha_ss_r, kappa_ss_r, alpha_t_rl, kappa_t_rl, Fz_rl, vx)
        d_alpha_rr, d_kappa_rr = self.tire.compute_transient_slip_derivatives(alpha_ss_r, kappa_ss_r, alpha_t_rr, kappa_t_rr, Fz_rr, vx)

        # Compute forces using TRANSIENT slip, not kinematic slip
        Fx_fl, Fy_fl = self.tire.compute_force(alpha_t_fl, kappa_t_fl, Fz_fl, gamma_f, T_ribs_f, T_gas_f, vx)
        Fx_fr, Fy_fr = self.tire.compute_force(alpha_t_fr, kappa_t_fr, Fz_fr, gamma_f, T_ribs_f, T_gas_f, vx)
        Fx_rl, Fy_rl = self.tire.compute_force(alpha_t_rl, kappa_t_rl, Fz_rl, gamma_r, T_ribs_r, T_gas_r, vx)
        Fx_rr, Fy_rr = self.tire.compute_force(alpha_t_rr, kappa_t_rr, Fz_rr, gamma_r, T_ribs_r, T_gas_r, vx)

        Fx_f = Fx_fl + Fx_fr
        Fy_f = Fy_fl + Fy_fr
        Fx_r = Fx_rl + Fx_rr
        Fy_r = Fy_rl + Fy_rr

        F_ext_mech = jnp.zeros(28)
        F_ext_mech = F_ext_mech.at[14].set(Fx_aero + Fx_f*jnp.cos(delta) - Fy_f*jnp.sin(delta) + Fx_r)
        F_ext_mech = F_ext_mech.at[15].set(Fx_f*jnp.sin(delta) + Fy_f*jnp.cos(delta) + Fy_r)
        F_ext_mech = F_ext_mech.at[16].set(-self.m_s * self.g - Fz_aero + Fz_susp_total)
        F_ext_mech = F_ext_mech.at[17].set(Mx_aero + Mx_susp)
        F_ext_mech = F_ext_mech.at[18].set(My_aero + My_susp)
        F_ext_mech = F_ext_mech.at[19].set(self.lf * Fy_f * jnp.cos(delta) - self.lr * Fy_r)

        dx_mech = jnp.dot((J - R), grad_H) + F_ext_mech
        
        dT_ribs_f, dt_core_f, dt_gas_f = self.tire.compute_thermal_dynamics(
            0.0, Fy_f, Fz_f_static/2, gamma_f, alpha_t_fl, 0.0, vx, T_core_f, T_ribs_f, T_gas_f
        )
        dT_ribs_r, dt_core_r, dt_gas_r = self.tire.compute_thermal_dynamics(
            0.0, Fy_r, Fz_r_static/2, gamma_r, alpha_t_rl, kappa_t_rl, vx, T_core_r, T_ribs_r, T_gas_r
        )

        dx_dt = jnp.concatenate([
            dx_mech[0:14],
            dx_mech[14:28] / self.M_diag,
            jnp.array([dt_core_f]), dT_ribs_f, jnp.array([dt_gas_f]),
            jnp.array([dt_core_r]), dT_ribs_r, jnp.array([dt_gas_r]),
            # Append transient state derivatives
            jnp.array([d_alpha_fl, d_kappa_fl, d_alpha_fr, d_kappa_fr, 
                       d_alpha_rl, d_kappa_rl, d_alpha_rr, d_kappa_rr])
        ])
        return dx_dt

    @partial(jax.jit, static_argnums=(0,))
    def simulate_step(self, x, u, setup_params, dt=0.005):
        """
        Störmer-Verlet (Leapfrog) Symplectic Integrator.
        """
        dx_dt_initial = self._compute_derivatives(x, u, setup_params)
        v_half = x[14:28] + 0.5 * dt * dx_dt_initial[14:28]
        therm_half = x[28:38] + 0.5 * dt * dx_dt_initial[28:38]
        trans_half = x[38:46] + 0.5 * dt * dx_dt_initial[38:46]
        
        x_half = x.at[14:28].set(v_half).at[28:38].set(therm_half).at[38:46].set(trans_half)
        
        dx_dt_half = self._compute_derivatives(x_half, u, setup_params)
        q_next = x[0:14] + dt * dx_dt_half[0:14]
        
        x_next_q = x_half.at[0:14].set(q_next)
        
        dx_dt_final = self._compute_derivatives(x_next_q, u, setup_params)
        v_next = v_half + 0.5 * dt * dx_dt_final[14:28]
        therm_next = therm_half + 0.5 * dt * dx_dt_final[28:38]
        trans_next = trans_half + 0.5 * dt * dx_dt_final[38:46]
        
        return x_next_q.at[14:28].set(v_next).at[28:38].set(therm_next).at[38:46].set(trans_next)

class DynamicBicycleModel:
    """Retained for Acados/CasADi legacy compatibility."""
    def __init__(self, vehicle_params, tire_coeffs):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
        
    def get_equations(self):
        n, alpha, v, delta, r = [ca.MX.sym(name) for name in ['n', 'alpha', 'v', 'delta', 'r']]
        x_mech = ca.vertcat(n, alpha, v, delta, r)
        u_d_delta, u_fx = ca.MX.sym('u_d_delta'), ca.MX.sym('u_fx')
        u = ca.vertcat(u_d_delta, u_fx)
        k_c = ca.MX.sym('k_c') 
        m, Iz = self.vp['m'], self.vp['Iz']
        lf, lr = self.vp['lf'], self.vp['lr']
        rho, A, Cl = 1.225, self.vp['A'], self.vp['Cl']
        
        Fz_aero = 0.5 * rho * Cl * A * v**2
        Fz_f = m * 9.81 * lr / (lf + lr) + Fz_aero/2
        Fz_r = m * 9.81 * lf / (lf + lr) + Fz_aero/2
        
        alpha_f = delta - (alpha + lf*r/v)
        alpha_r = -(alpha - lr*r/v)
        
        T_ribs_dummy = [90.0, 90.0, 90.0]
        Fx_f, Fy_f = self.tire.compute_force(alpha_f, 0, Fz_f, -1.5, T_ribs_dummy, 60.0, v)
        Fx_r, Fy_r = self.tire.compute_force(alpha_r, 0, Fz_r, -1.0, T_ribs_dummy, 60.0, v)
        
        n_dot = v * ca.sin(alpha)
        alpha_dot = ((Fy_f * ca.cos(delta) + Fy_r) / m - v * r) / v
        v_dot = u_fx / m
        r_dot = (lf * Fy_f * ca.cos(delta) - lr * Fy_r) / Iz
        delta_dot = u_d_delta
        
        rhs = ca.vertcat(n_dot, alpha_dot, v_dot, delta_dot, r_dot)
        return ca.Function('f_dyn', [x_mech, u, k_c], [rhs])