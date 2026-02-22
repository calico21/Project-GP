import jax
import jax.numpy as jnp
from jax import jit, grad
import flax.linen as nn
from functools import partial
import casadi as ca

from models.tire_model import PacejkaTire 

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
        
        state = jnp.concatenate([q, p, setup_params])
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
        state = jnp.concatenate([q, p])
        x = nn.Dense(128)(state)
        x = nn.swish(x)
        x = nn.Dense(64)(x)
        x = nn.swish(x)
        
        num_elements = self.dim * (self.dim + 1) // 2
        L_elements = nn.Dense(num_elements, kernel_init=jax.nn.initializers.zeros, 
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
        
        total_stiff_f = k_f + arb_f
        total_stiff_r = k_r + arb_r
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

        F_susp_fl = -(k_f * z_fl + c_f * vz_fl)
        F_susp_fr = -(k_f * z_fr + c_f * vz_fr)
        F_susp_rl = -(k_r * z_rl + c_r * vz_rl)
        F_susp_rr = -(k_r * z_rr + c_r * vz_rr)
        
        Fz_susp_total = F_susp_fl + F_susp_fr + F_susp_rl + F_susp_rr
        Mx_susp = (F_susp_fr + F_susp_rr - F_susp_fl - F_susp_rl) * (self.track_w / 2.0)
        My_susp = (F_susp_rl + F_susp_rr) * self.lr - (F_susp_fl + F_susp_fr) * self.lf

        T_core_f = jnp.where(x[28] > 10, x[28], 60.0)
        T_ribs_f = jnp.clip(x[29:32], 10.0, 150.0)
        T_gas_f  = jnp.where(x[32] > 10, x[32], 25.0)
        
        T_core_r = jnp.where(x[33] > 10, x[33], 60.0)
        T_ribs_r = jnp.clip(x[34:37], 10.0, 150.0)
        T_gas_r  = jnp.where(x[37] > 10, x[37], 25.0)

        delta = u[0] 
        
        vx_safe = jnp.maximum(vx, 1.0)
        beta = jnp.arctan2(vy, vx_safe)
        alpha_f = delta - (beta + self.lf * wz / vx_safe)
        alpha_r = -(beta - self.lr * wz / vx_safe)
        alpha_f = jnp.clip(alpha_f, -1.5, 1.5)
        alpha_r = jnp.clip(alpha_r, -1.5, 1.5)

        # KAPPA FIX: u[1] is a drive force in Newtons (~0–2500 N), NOT a slip ratio.
        # Passing it raw as kappa caused P_fric = Fy * Vx * 1250 ≈ 7.5 MW/tire,
        # driving tire temps to ~10,000°C and producing NaN within the first RK4 step.
        # Normalize to a dimensionless slip ratio bounded to the physically valid range.
        # We use rear static load as the normalizer (stiffer load → less slip at same force).
        Fz_r_ref = jnp.maximum(Fz_r_static, 100.0)  # avoid div-by-zero on liftoff
        kappa_r = jnp.clip(u[1] / (Fz_r_ref * 10.0), -0.3, 0.3)

        Fx_fl, Fy_fl = self.tire.compute_force(alpha_f, 0.0, Fz_fl, gamma_f, T_ribs_f, T_gas_f, vx)
        Fx_fr, Fy_fr = self.tire.compute_force(alpha_f, 0.0, Fz_fr, gamma_f, T_ribs_f, T_gas_f, vx)
        Fx_rl, Fy_rl = self.tire.compute_force(alpha_r, kappa_r, Fz_rl, gamma_r, T_ribs_r, T_gas_r, vx)
        Fx_rr, Fy_rr = self.tire.compute_force(alpha_r, kappa_r, Fz_rr, gamma_r, T_ribs_r, T_gas_r, vx)

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
            0.0, Fy_f, Fz_f_static/2, gamma_f, alpha_f, 0.0, vx, T_core_f, T_ribs_f, T_gas_f
        )
        dT_ribs_r, dt_core_r, dt_gas_r = self.tire.compute_thermal_dynamics(
            0.0, Fy_r, Fz_r_static/2, gamma_r, alpha_r, kappa_r, vx, T_core_r, T_ribs_r, T_gas_r
        )

        dx_dt = jnp.concatenate([
            dx_mech[0:14],
            dx_mech[14:28] / self.M_diag,
            jnp.array([dt_core_f]), dT_ribs_f, jnp.array([dt_gas_f]),
            jnp.array([dt_core_r]), dT_ribs_r, jnp.array([dt_gas_r])
        ])
        return dx_dt

    @partial(jax.jit, static_argnums=(0,))
    def simulate_step(self, x, u, setup_params, dt=0.005):
        k1 = self._compute_derivatives(x, u, setup_params)
        k2 = self._compute_derivatives(x + 0.5 * dt * k1, u, setup_params)
        k3 = self._compute_derivatives(x + 0.5 * dt * k2, u, setup_params)
        k4 = self._compute_derivatives(x + dt * k3, u, setup_params)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

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