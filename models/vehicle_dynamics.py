import jax
import jax.numpy as jnp
from jax import jit, grad
import flax.linen as nn
from functools import partial
import casadi as ca

from models.tire_model import PacejkaTire 

class NeuralEnergyLandscape(nn.Module):
    m: float
    Iz: float

    @nn.compact
    def __call__(self, q, p, setup_params):
        T_prior = 0.5 * (p[0]**2 / self.m + p[1]**2 / self.m + p[2]**2 / self.Iz)
        state = jnp.concatenate([q, p, setup_params])
        x = nn.Dense(64)(state)
        x = nn.swish(x)
        x = nn.Dense(32)(x)
        x = nn.swish(x)
        H_residual = nn.Dense(1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)[0]
        return T_prior + H_residual

class NeuralDissipationMatrix(nn.Module):
    dim: int = 7

    @nn.compact
    def __call__(self, q, p):
        state = jnp.concatenate([q, p])
        x = nn.Dense(64)(state)
        x = nn.swish(x)
        num_elements = self.dim * (self.dim + 1) // 2
        L_elements = nn.Dense(num_elements, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)
        L = jnp.zeros((self.dim, self.dim))
        indices = jnp.tril_indices(self.dim)
        L = L.at[indices].set(L_elements)
        return jnp.dot(L, L.T)

class DifferentiableMultiBodyVehicle:
    def __init__(self, vehicle_params, tire_coeffs, rng_seed=42):
        self.vp = vehicle_params
        self.tire = PacejkaTire(tire_coeffs)
        
        self.m = self.vp['m']
        self.Iz = self.vp['Iz']
        self.lf = self.vp['lf']
        self.lr = self.vp['lr']
        self.h_cg = self.vp.get('h_cg', 0.3)
        self.track_w = 1.2 
        self.rho = 1.225
        self.A = self.vp['A']
        self.Cl = self.vp.get('Cl', 3.0)
        self.g = 9.81

        self.H_net = NeuralEnergyLandscape(m=self.m, Iz=self.Iz)
        self.R_net = NeuralDissipationMatrix(dim=7)
        
        key1, key2 = jax.random.split(jax.random.PRNGKey(rng_seed))
        
        dummy_q = jnp.zeros(4)
        dummy_p = jnp.zeros(3)
        dummy_setup = jnp.zeros(7)
        
        self.H_params = self.H_net.init(key1, dummy_q, dummy_p, dummy_setup)
        self.R_params = self.R_net.init(key2, dummy_q, dummy_p)

    @partial(jax.jit, static_argnums=(0,)) 
    def _compute_derivatives(self, x, u, setup_params):
        q = jnp.array([x[0], x[1], x[2], x[6]])
        p = jnp.array([self.m * x[3], self.m * x[4], self.Iz * x[5]])
        
        grad_H_fn = jax.grad(self.H_net.apply, argnums=(1, 2))
        dH_dq, dH_dp = grad_H_fn(self.H_params, q, p, setup_params)
        grad_H = jnp.concatenate([dH_dq, dH_dp])
        
        J = jnp.zeros((7, 7))
        J = J.at[0:4, 4:7].set(jnp.eye(4)[:, :3])
        J = J.at[4:7, 0:4].set(-jnp.eye(3, 4))
        R = self.R_net.apply(self.R_params, q, p)
        
        vx, vy, r, delta = x[3], x[4], x[5], x[6]
        
        # --- THE FIX: RESTORING THE PARAMETER GRADIENT GRAPH ---
        k_f, k_r = setup_params[0], setup_params[1]
        arb_f, arb_r = setup_params[2], setup_params[3]
        h_cg = setup_params[6]
        
        total_stiff_f = k_f + arb_f
        total_stiff_r = k_r + arb_r
        total_stiff = total_stiff_f + total_stiff_r + 1.0
        
        ay = vx * r
        W_transfer = (self.m * ay * h_cg) / self.track_w
        
        phi = (self.m * ay * h_cg) / total_stiff 
        phi_deg = jnp.rad2deg(phi)
        
        gamma_f = -2.0 + (phi_deg * 0.6) 
        gamma_r = -1.5 + (phi_deg * 0.4)
        
        dFz_f = W_transfer * (total_stiff_f / total_stiff)
        dFz_r = W_transfer * (total_stiff_r / total_stiff)

        Fz_aero = 0.5 * self.rho * self.Cl * self.A * vx**2
        Fz_f_static = (self.m * self.g * self.lr) / (self.lf + self.lr) + Fz_aero * 0.4
        Fz_r_static = (self.m * self.g * self.lf) / (self.lf + self.lr) + Fz_aero * 0.6
        
        Fz_f_l = jnp.maximum(0.0, Fz_f_static - dFz_f)
        Fz_f_r = jnp.maximum(0.0, Fz_f_static + dFz_f)
        Fz_r_l = jnp.maximum(0.0, Fz_r_static - dFz_r)
        Fz_r_r = jnp.maximum(0.0, Fz_r_static + dFz_r)

        beta = jnp.where(vx > 1.0, jnp.arctan2(vy, vx), 0.0)
        alpha_f = delta - (beta + self.lf * r / (vx + 1e-6))
        alpha_r = - (beta - self.lr * r / (vx + 1e-6))
        
        T_core_f = jnp.where(x[7] > 10, x[7], 60.0)
        T_in_f   = jnp.where(x[8] > 10, x[8], 60.0)
        T_mid_f  = jnp.where(x[9] > 10, x[9], 60.0)
        T_out_f  = jnp.where(x[10] > 10, x[10], 60.0)
        T_gas_f  = jnp.where(x[11] > 10, x[11], 25.0)
        
        T_core_r = jnp.where(x[12] > 10, x[12], 60.0)
        T_in_r   = jnp.where(x[13] > 10, x[13], 60.0)
        T_mid_r  = jnp.where(x[14] > 10, x[14], 60.0)
        T_out_r  = jnp.where(x[15] > 10, x[15], 60.0)
        T_gas_r  = jnp.where(x[16] > 10, x[16], 25.0)
        
        # Tire Physics now directly depend on the setup_params via Fz_l/r and gamma
        T_ribs_f = jnp.array([T_in_f, T_mid_f, T_out_f])
        Fx_fl, Fy_fl = self.tire.compute_force(alpha_f, 0.0, Fz_f_l, gamma_f, T_ribs_f, T_gas_f, vx)
        Fx_fr, Fy_fr = self.tire.compute_force(alpha_f, 0.0, Fz_f_r, gamma_f, T_ribs_f, T_gas_f, vx)
        Fy_f_tot = Fy_fl + Fy_fr
        
        T_ribs_r = jnp.array([T_in_r, T_mid_r, T_out_r])
        Fx_rl, Fy_rl = self.tire.compute_force(alpha_r, 0.0, Fz_r_l, gamma_r, T_ribs_r, T_gas_r, vx)
        Fx_rr, Fy_rr = self.tire.compute_force(alpha_r, 0.0, Fz_r_r, gamma_r, T_ribs_r, T_gas_r, vx)
        Fy_r_tot = Fy_rl + Fy_rr

        F_ext = jnp.array([
            0.0, 0.0, 0.0, (u[0] - delta) * 15.0, 
            self.m * vy * r, 
            Fy_f_tot * jnp.cos(delta) + Fy_r_tot - self.m * vx * r, 
            self.lf * Fy_f_tot * jnp.cos(delta) - self.lr * Fy_r_tot 
        ])

        dx_mech = jnp.dot((J - R), grad_H) + F_ext
        
        dT_ribs_f, dt_core_f, dt_gas_f = self.tire.compute_thermal_dynamics(
            0.0, Fy_f_tot/2, Fz_f_static, gamma_f, alpha_f, 0.0, vx, T_core_f, T_ribs_f, T_gas_f
        )
        dT_ribs_r, dt_core_r, dt_gas_r = self.tire.compute_thermal_dynamics(
            0.0, Fy_r_tot/2, Fz_r_static, gamma_r, alpha_r, 0.0, vx, T_core_r, T_ribs_r, T_gas_r
        )

        dx_dt = jnp.array([
            dx_mech[0], dx_mech[1], dx_mech[2], 
            dx_mech[4] / self.m,  
            dx_mech[5] / self.m,  
            dx_mech[6] / self.Iz, 
            dx_mech[3],           
            dt_core_f, dT_ribs_f[0], dT_ribs_f[1], dT_ribs_f[2], dt_gas_f,
            dt_core_r, dT_ribs_r[0], dT_ribs_r[1], dT_ribs_r[2], dt_gas_r
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
        alpha_r = - (alpha - lr*r/v)
        
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