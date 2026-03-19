"""
visualization/suspension_viz.py  —  Project-GP Suspension Dynamics Visualizer
===============================================================================
Physics-accurate, quasi-static suspension kinematics visualized as an animated
Plotly-in-Streamlit dashboard tab. Designed to drop directly into the existing
dashboard.py sidebar router as a new "Suspension Dynamics" module.

Physics sourced from vehicle_params.py:
  · Digressive (Horstman bilinear) damper model — identical to vehicle_dynamics.py
  · Wheel-rate-correct motion ratio (MR polynomial evaluated at design ride height)
  · Full lateral/longitudinal load transfer with geometric + elastic RC split
  · K&C kinematic maps: camber gain, bump steer (quadratic), compliance steer
  · ARB cross-coupling (torsional stiffness / MR²)
  · Softplus bump stop engagement (C^∞, matching vehicle_dynamics treatment)
  · Ground-effect aero downforce (Cl_ref, split_f/r, pitch sensitivity)

No JAX in this file. Pure numpy/scipy for the visualization layer.
"""

from __future__ import annotations
import os
import sys
from typing import Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Project root discovery ────────────────────────────────────────────────────
def _find_project_root() -> str:
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if (os.path.isdir(os.path.join(candidate, 'optimization')) and
                os.path.isdir(os.path.join(candidate, 'models'))):
            return candidate
        candidate = os.path.dirname(candidate)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ROOT = _find_project_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Theme constants — mirror dashboard.py CSS variables ──────────────────────
_BG_MAIN    = '#0E1117'
_BG_CARD    = '#1A1E2E'
_BG_SURFACE = '#1F2436'
_CYAN       = '#00C0F2'
_RED        = '#FF4B4B'
_GREEN      = '#23D160'
_AMBER      = '#FFB020'
_WHITE      = '#FFFFFF'
_TEXT_B     = '#D8DCE8'
_TEXT_M     = '#9BA3BC'
_BORDER_M   = 'rgba(0, 192, 242, 0.30)'

_PFONT = dict(family='Courier New', color=_TEXT_B, size=11)
_BASE_LAY = dict(
    paper_bgcolor=_BG_MAIN,
    plot_bgcolor=_BG_SURFACE,
    font=_PFONT,
    margin=dict(l=52, r=16, t=40, b=36),
    xaxis=dict(
        gridcolor='rgba(0,192,242,0.08)',
        zerolinecolor='rgba(0,192,242,0.20)',
        tickfont=dict(family='Courier New', size=10, color=_TEXT_M),
    ),
    yaxis=dict(
        gridcolor='rgba(0,192,242,0.08)',
        zerolinecolor='rgba(0,192,242,0.20)',
        tickfont=dict(family='Courier New', size=10, color=_TEXT_M),
    ),
)


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICS LAYER
# ═════════════════════════════════════════════════════════════════════════════

class SuspensionPhysics:
    """
    Self-contained quasi-static suspension model.
    Reads vehicle_params directly; falls back to hardcoded Ter26 defaults if
    the import fails (allows standalone testing outside project root).
    """

    # ── Hardcoded fallbacks (Ter26 FS2026 baseline) ───────────────────────────
    _DEFAULTS = {
        'total_mass': 300.0, 'sprung_mass': 268.9,
        'h_cg': 0.330, 'h_cg_sprung': 0.350,
        'lf': 0.8525, 'lr': 0.6975,
        'track_front': 1.200, 'track_rear': 1.180,
        'wheel_radius': 0.2032,
        'spring_rate_f': 35030.0, 'spring_rate_r': 52540.0,
        'arb_rate_f': 200.0, 'arb_rate_r': 150.0,
        'motion_ratio_f_poly': [1.14, 2.5, 0.0],
        'motion_ratio_r_poly': [1.16, 2.0, 0.0],
        'damper_c_low_f': 1800.0, 'damper_c_low_r': 1600.0,
        'damper_c_high_f': 720.0, 'damper_c_high_r': 640.0,
        'damper_v_knee_f': 0.10, 'damper_v_knee_r': 0.10,
        'rebound_ratio_f': 1.60, 'rebound_ratio_r': 1.60,
        'damper_gas_force_f': 120.0, 'damper_gas_force_r': 120.0,
        'h_rc_f': 0.040, 'h_rc_r': 0.060,
        'dh_rc_dz_f': 0.20, 'dh_rc_dz_r': 0.30,
        'static_camber_f': -2.0, 'static_camber_r': -1.5,
        'static_toe_f': -0.10, 'static_toe_r': -0.15,
        'camber_gain_f': -0.80, 'camber_gain_r': -0.65,
        'camber_per_m_travel_f': -25.0, 'camber_per_m_travel_r': -20.0,
        'bump_steer_f': 0.0, 'bump_steer_r': 0.0,
        'bump_steer_quad_f': 0.0, 'bump_steer_quad_r': 0.0,
        'compliance_steer_f': -0.15, 'compliance_steer_r': -0.10,
        'castor_f': 5.0,
        'bump_stop_engage': 0.025, 'bump_stop_rate': 50000.0,
        'Cl_ref': 4.14, 'Cd_ref': 2.50, 'A_ref': 1.10,
        'aero_split_f': 0.45, 'aero_split_r': 0.55,
        'dCl_f_dtheta': 0.35, 'rho_air': 1.225,
        'unsprung_mass_f': 7.74, 'unsprung_mass_r': 7.76,
        'h_ride_f': 0.030, 'h_ride_r': 0.030,
        'anti_dive_f': 0.40, 'anti_dive_r': 0.10,
        'anti_squat': 0.30,
    }

    def __init__(self):
        try:
            from data.configs.vehicle_params import vehicle_params as _VP
            self._vp = {**self._DEFAULTS, **_VP}
        except ImportError:
            self._vp = dict(self._DEFAULTS)
        self._extract()

    def _p(self, key, fallback=None):
        return self._vp.get(key, fallback if fallback is not None else self._DEFAULTS.get(key))

    def _extract(self):
        p = self._p
        self.g       = 9.81
        self.m       = p('total_mass')
        self.m_s     = p('sprung_mass')
        self.h_cg    = p('h_cg')
        self.lf      = p('lf');       self.lr = p('lr')
        self.wb      = self.lf + self.lr
        self.tw_f    = p('track_front')
        self.tw_r    = p('track_rear')
        self.R       = p('wheel_radius')
        self.m_us_f  = p('unsprung_mass_f')
        self.m_us_r  = p('unsprung_mass_r')

        # Springs at wheel (effective wheel rate = k_spring / MR²)
        self.k_f     = p('spring_rate_f')
        self.k_r     = p('spring_rate_r')
        mr_f = p('motion_ratio_f_poly');  mr_r = p('motion_ratio_r_poly')
        self.mr_f0   = mr_f[0];  self.mr_f1 = mr_f[1] if len(mr_f) > 1 else 0.0
        self.mr_r0   = mr_r[0];  self.mr_r1 = mr_r[1] if len(mr_r) > 1 else 0.0
        self.wr_f    = self.k_f / (self.mr_f0 ** 2)   # N/m at wheel
        self.wr_r    = self.k_r / (self.mr_r0 ** 2)

        # ARB wheel rates
        self.arb_f   = p('arb_rate_f')
        self.arb_r   = p('arb_rate_r')

        # Effective axle stiffness (spring + ARB) for load transfer
        self.k_axle_f = self.wr_f + self.arb_f
        self.k_axle_r = self.wr_r + self.arb_r

        # Roll stiffness distribution: K_phi = wr*(t/2)² + arb*(t/2)²
        self.K_phi_f = (self.wr_f + self.arb_f) * (self.tw_f / 2) ** 2
        self.K_phi_r = (self.wr_r + self.arb_r) * (self.tw_r / 2) ** 2
        self.K_phi   = self.K_phi_f + self.K_phi_r

        # Pitch stiffness: K_theta = wr_f*lf² + wr_r*lr²
        self.K_theta = self.wr_f * self.lf ** 2 + self.wr_r * self.lr ** 2

        # Damper params
        self.c_lo_f   = p('damper_c_low_f');  self.c_lo_r  = p('damper_c_low_r')
        self.c_hi_f   = p('damper_c_high_f'); self.c_hi_r  = p('damper_c_high_r')
        self.vk_f     = p('damper_v_knee_f'); self.vk_r    = p('damper_v_knee_r')
        self.rho_f    = p('rebound_ratio_f'); self.rho_r   = p('rebound_ratio_r')

        # Gas preload at wheel
        self.Fg_f    = p('damper_gas_force_f') / self.mr_f0
        self.Fg_r    = p('damper_gas_force_r') / self.mr_r0

        # Static equilibrium spring coordinate
        self.Z_eq    = (2*(self.Fg_f + self.Fg_r) - self.m_s * self.g) \
                       / (2*(self.wr_f + self.wr_r))

        # Roll centre geometry
        self.h_rc_f  = p('h_rc_f'); self.h_rc_r = p('h_rc_r')
        self.dh_rc_f = p('dh_rc_dz_f'); self.dh_rc_r = p('dh_rc_dz_r')

        # Kinematic coefficients
        self.camber_f0 = p('static_camber_f');  self.camber_r0 = p('static_camber_r')
        self.cg_f      = p('camber_gain_f');    self.cg_r      = p('camber_gain_r')
        self.cm_f      = p('camber_per_m_travel_f')
        self.cm_r      = p('camber_per_m_travel_r')
        self.bs_f      = p('bump_steer_f');     self.bs_r      = p('bump_steer_r')
        self.bsq_f     = p('bump_steer_quad_f');self.bsq_r     = p('bump_steer_quad_r')
        self.cs_f      = p('compliance_steer_f')
        self.cs_r      = p('compliance_steer_r')
        self.toe_f0    = p('static_toe_f');     self.toe_r0    = p('static_toe_r')
        self.castor    = p('castor_f')

        # Bump stop
        self.bs_gap    = p('bump_stop_engage')
        self.bs_k      = p('bump_stop_rate')

        # Aero
        self.Cl        = p('Cl_ref'); self.A = p('A_ref')
        self.split_f   = p('aero_split_f'); self.split_r = p('aero_split_r')
        self.rho_air   = p('rho_air')
        self.dCl_dpitch= p('dCl_f_dtheta')

    # ── Physics helpers ───────────────────────────────────────────────────────

    def motion_ratio(self, z: float | np.ndarray, axle: str = 'f') -> np.ndarray:
        """MR(z) = a0 + a1·z  (linear truncation of polynomial)."""
        a0, a1 = (self.mr_f0, self.mr_f1) if axle == 'f' else (self.mr_r0, self.mr_r1)
        return a0 + a1 * z

    def digressive_damper(
        self, v_damp: np.ndarray, axle: str = 'f'
    ) -> np.ndarray:
        """
        Horstman bilinear digressive model (numpy mirror of vehicle_dynamics.py).
        Bump:    F = c_lo·v/(1+v/vk)   + c_hi·v
        Rebound: F = ρ·c_lo·|v|/(1+|v|/vk) + c_hi·|v|  [sign preserved]
        """
        c_lo, c_hi, vk, rho = (
            (self.c_lo_f, self.c_hi_f, self.vk_f, self.rho_f) if axle == 'f'
            else (self.c_lo_r, self.c_hi_r, self.vk_r, self.rho_r)
        )
        v_abs = np.abs(v_damp)
        sgn   = np.sign(v_damp)
        # Bilinear saturation
        F_lo  = c_lo * v_abs / (1.0 + v_abs / (vk + 1e-9))
        F_hi  = c_hi * v_abs
        # Rebound multiplier: smoothly blended via sigmoid (avoids discontinuity)
        w_bump = 0.5 + 0.5 * np.tanh(200.0 * v_damp)   # ≈ Heaviside(v)
        rho_eff = w_bump * 1.0 + (1.0 - w_bump) * rho
        return sgn * rho_eff * (F_lo + F_hi)

    def bump_stop_force(self, z_wheel: np.ndarray) -> np.ndarray:
        """C^∞ softplus approximation: F = k_bs·softplus(β·(z-gap))/β."""
        beta = 200.0
        return self.bs_k * np.log1p(np.exp(beta * (z_wheel - self.bs_gap))) / beta

    def aero_fz(self, v_kmh: np.ndarray, pitch_deg: float = 0.0) -> np.ndarray:
        """Downforce split between axles, pitch-sensitive."""
        q = 0.5 * self.rho_air * (v_kmh / 3.6) ** 2
        Cl_total = self.Cl * (1.0 + self.dCl_dpitch * np.radians(pitch_deg))
        return q * Cl_total * self.A  # total downforce N

    def corner_loads_static(self) -> np.ndarray:
        """4-corner static loads [FL, FR, RL, RR] N."""
        Fz_f = self.m * self.g * self.lr / self.wb
        Fz_r = self.m * self.g * self.lf / self.wb
        return np.array([Fz_f / 2, Fz_f / 2, Fz_r / 2, Fz_r / 2])

    def corner_loads_dynamic(
        self, ax: float, ay: float, v_kmh: float = 0.0, pitch_deg: float = 0.0
    ) -> np.ndarray:
        """
        Quasi-static load transfer with geometric (RC) + elastic (spring) lateral split
        and longitudinal LT from CG height.

        Returns: Fz [FL, FR, RL, RR] N, clamped above zero.
        """
        g = self.g; m = self.m; h = self.h_cg
        lf = self.lf; lr = self.lr; wb = self.wb

        # Aero downforce split
        Fz_aero = self.aero_fz(v_kmh, pitch_deg)
        Fz_aero_f = Fz_aero * self.split_f
        Fz_aero_r = Fz_aero * self.split_r

        Fz_s_f = m * g * lr / wb + Fz_aero_f
        Fz_s_r = m * g * lf / wb + Fz_aero_r

        # Longitudinal LT (positive ax = forward acceleration)
        dFz_lon = m * ax * h / wb

        # Lateral LT — separate geometric and elastic contributions per axle
        # Geometric (unsprung + RC moment): F_geo = m*ay*h_rc / tw
        # Elastic (sprung above RC through roll stiffness): F_ela = m*ay*(h-h_rc)*K_phi_axle/(K_phi*tw)
        dFz_lat_geo_f = m * ay * self.h_rc_f / self.tw_f
        dFz_lat_geo_r = m * ay * self.h_rc_r / self.tw_r
        dFz_lat_ela_f = m * ay * (h - self.h_rc_f) * (self.K_phi_f / self.K_phi) / self.tw_f
        dFz_lat_ela_r = m * ay * (h - self.h_rc_r) * (self.K_phi_r / self.K_phi) / self.tw_r
        dFz_lat_f = dFz_lat_geo_f + dFz_lat_ela_f
        dFz_lat_r = dFz_lat_geo_r + dFz_lat_ela_r

        # Positive ay = left turn: right side loaded, left side unloaded
        Fz = np.array([
            Fz_s_f / 2 - dFz_lon / 2 - dFz_lat_f,   # FL
            Fz_s_f / 2 - dFz_lon / 2 + dFz_lat_f,   # FR
            Fz_s_r / 2 + dFz_lon / 2 - dFz_lat_r,   # RL
            Fz_s_r / 2 + dFz_lon / 2 + dFz_lat_r,   # RR
        ])
        return np.maximum(Fz, 10.0)   # minimum 10 N (wheel on ground)

    def wheel_travel(self, Fz: np.ndarray) -> np.ndarray:
        """
        Wheel travel from ΔFz relative to static.
        z = ΔFz / (wr + arb)  [m, positive = bump]
        ARB is included in the effective spring stiffness (it resists differential motion).
        """
        Fz_s = self.corner_loads_static()
        wr   = np.array([self.wr_f, self.wr_f, self.wr_r, self.wr_r])
        arb  = np.array([self.arb_f, self.arb_f, self.arb_r, self.arb_r])
        return (Fz - Fz_s) / (wr + arb)

    def body_roll(self, ay: float) -> float:
        """Body roll angle in degrees (+ve = roll right = left corner in bump)."""
        return np.degrees(self.m_s * ay * (self.h_cg - self.h_rc_f) / self.K_phi)

    def body_pitch(self, ax: float) -> float:
        """Body pitch angle in degrees (+ve = nose down = front in bump under braking)."""
        return np.degrees(-self.m_s * ax * self.h_cg / self.K_theta)

    def camber(self, z_travel: float | np.ndarray, roll_deg: float, axle: str = 'f') -> np.ndarray:
        """Instantaneous camber [deg]. roll_deg positive = outer side in bump."""
        c0  = self.camber_f0 if axle == 'f' else self.camber_r0
        cg  = self.cg_f      if axle == 'f' else self.cg_r
        cm  = self.cm_f      if axle == 'f' else self.cm_r
        return c0 + cg * roll_deg + cm * z_travel

    def toe(self, z_travel: float | np.ndarray, axle: str = 'f') -> np.ndarray:
        """Bump steer + static toe [deg]."""
        t0  = self.toe_f0 if axle == 'f' else self.toe_r0
        bs  = self.bs_f   if axle == 'f' else self.bs_r
        bsq = self.bsq_f  if axle == 'f' else self.bsq_r
        return t0 + np.degrees(bs * z_travel + bsq * z_travel ** 2)

    # ── Maneuver simulation ───────────────────────────────────────────────────

    def simulate(self, maneuver: str = 'combined_lap', dt: float = 0.02) -> Dict:
        """
        Quasi-static time-series simulation of a chosen maneuver.
        Returns a flat dict of numpy arrays, all aligned to time vector t.
        """
        # ── Maneuver profiles (smooth, physically credible) ───────────────────
        if maneuver == 'skidpad':
            t = np.arange(0.0, 9.0, dt)
            v = np.full_like(t, 42.0)  # ~12 m/s — FSAE skidpad speed
            # Lateral ramp: 0→1.5G in 1s, hold 6s, ramp out
            ay_g = np.where(t < 1.0,  1.5 * t / 1.0,
                   np.where(t < 7.0,  1.5,
                   np.where(t < 8.0,  1.5 * (8.0 - t) / 1.0, 0.0)))
            ax_g = np.zeros_like(t)

        elif maneuver == 'brake_straight':
            t = np.arange(0.0, 7.0, dt)
            v_prof = np.where(t < 0.5,  80.0,
                     np.where(t < 4.0,  80.0 - 22.0 * (t - 0.5) / 3.5,
                              80.0 - 22.0 * 0.99))
            v = np.clip(v_prof, 10.0, 85.0)
            ax_g = np.where(t < 0.4,  0.0,
                   np.where(t < 0.7, -2.5 * (t - 0.4) / 0.3,
                   np.where(t < 3.8, -2.5,
                   np.where(t < 4.2, -2.5 * (4.2 - t) / 0.4, 0.0))))
            ay_g = np.zeros_like(t)

        elif maneuver == 'acceleration':
            t = np.arange(0.0, 7.0, dt)
            v = np.clip(5.0 + 8.33 * t * 3.6, 5.0, 95.0)  # ~0-100 in 12s
            ax_g = np.where(t < 0.3,  1.8 * t / 0.3,
                   np.where(t < 5.0,  1.8,
                   np.where(t < 5.5,  1.8 * (5.5 - t) / 0.5, 0.0)))
            ay_g = np.zeros_like(t)

        elif maneuver == 'chicane':
            t = np.arange(0.0, 8.0, dt)
            v = 55.0 + 10.0 * np.cos(2 * np.pi * t / 8.0)
            # Double-apex: left-right
            ay_g = (1.3 * np.sin(2 * np.pi * t / 4.0)
                    * np.exp(-((t - 4.0) ** 2) / 8.0 + 0.5))
            ay_g = np.clip(ay_g, -1.5, 1.5)
            ax_g = -0.8 * np.abs(np.cos(2 * np.pi * t / 4.0)) + 0.3

        else:  # combined_lap — default
            t = np.arange(0.0, 14.0, dt)
            # Lap segment: straight → corner → braking → re-acceleration
            v = 50.0 + 18.0 * np.sin(2 * np.pi * t / 14.0 - 0.5)
            ax_g = (1.2 * np.sin(2 * np.pi * t / 14.0 + 0.8)
                    - 0.4 * np.sin(4 * np.pi * t / 14.0))
            ax_g = np.clip(ax_g, -2.5, 1.8)
            ay_g = (1.4 * np.sin(2 * np.pi * t / 14.0 + 2.2)
                    + 0.4 * np.cos(4 * np.pi * t / 14.0))
            ay_g = np.clip(ay_g, -1.6, 1.6)

        ax = ax_g * self.g
        ay = ay_g * self.g
        n  = len(t)

        # ── Compute pitch/roll for each timestep ──────────────────────────────
        pitch  = np.array([self.body_pitch(ax[i]) for i in range(n)])
        roll   = np.array([self.body_roll(ay[i])  for i in range(n)])

        # ── Corner loads → wheel travel ───────────────────────────────────────
        Fz_all = np.array([
            self.corner_loads_dynamic(ax[i], ay[i], v[i], pitch[i]) for i in range(n)
        ])   # (n, 4)
        z_all  = np.array([self.wheel_travel(Fz_all[i]) for i in range(n)])   # (n, 4)

        # ── Bump stop engagement ──────────────────────────────────────────────
        Fbs_all = np.vectorize(self.bump_stop_force)(z_all)   # (n, 4)

        # ── Camber (left = positive roll, right = negative roll wrt camber sign)
        camber_fl = np.array([self.camber(z_all[i, 0],  roll[i], 'f') for i in range(n)])
        camber_fr = np.array([self.camber(z_all[i, 1], -roll[i], 'f') for i in range(n)])
        camber_rl = np.array([self.camber(z_all[i, 2],  roll[i], 'r') for i in range(n)])
        camber_rr = np.array([self.camber(z_all[i, 3], -roll[i], 'r') for i in range(n)])

        # ── Toe ───────────────────────────────────────────────────────────────
        toe_fl = np.array([self.toe(z_all[i, 0], 'f') for i in range(n)])
        toe_fr = np.array([self.toe(z_all[i, 1], 'f') for i in range(n)])
        toe_rl = np.array([self.toe(z_all[i, 2], 'r') for i in range(n)])
        toe_rr = np.array([self.toe(z_all[i, 3], 'r') for i in range(n)])

        # ── Damper velocity (wheel space) → damper space ──────────────────────
        vd_wheel = np.gradient(z_all, dt, axis=0)          # wheel velocity, (n, 4)
        vd_f = vd_wheel[:, :2] * self.mr_f0               # damper velocity front
        vd_r = vd_wheel[:, 2:] * self.mr_r0               # damper velocity rear

        Fd_fl = self.digressive_damper(vd_f[:, 0], 'f')
        Fd_fr = self.digressive_damper(vd_f[:, 1], 'f')
        Fd_rl = self.digressive_damper(vd_r[:, 0], 'r')
        Fd_rr = self.digressive_damper(vd_r[:, 1], 'r')

        # ── ARB differential travel and coupling torque ───────────────────────
        dz_f_mm = (z_all[:, 0] - z_all[:, 1]) * 1000   # FL-FR differential [mm]
        dz_r_mm = (z_all[:, 2] - z_all[:, 3]) * 1000
        arb_torque_f = self.arb_f * (dz_f_mm / 1000) * (self.tw_f / 2)
        arb_torque_r = self.arb_r * (dz_r_mm / 1000) * (self.tw_r / 2)

        return {
            't': t, 'v_kmh': v,
            'ax_g': ax_g, 'ay_g': ay_g,
            'pitch': pitch, 'roll': roll,
            # Corner loads (N)
            'Fz_fl': Fz_all[:, 0], 'Fz_fr': Fz_all[:, 1],
            'Fz_rl': Fz_all[:, 2], 'Fz_rr': Fz_all[:, 3],
            # Wheel travel (mm)
            'z_fl': z_all[:, 0]*1000, 'z_fr': z_all[:, 1]*1000,
            'z_rl': z_all[:, 2]*1000, 'z_rr': z_all[:, 3]*1000,
            # Bump stop forces (N)
            'Fbs_fl': Fbs_all[:, 0], 'Fbs_fr': Fbs_all[:, 1],
            'Fbs_rl': Fbs_all[:, 2], 'Fbs_rr': Fbs_all[:, 3],
            # Alignment
            'camber_fl': camber_fl, 'camber_fr': camber_fr,
            'camber_rl': camber_rl, 'camber_rr': camber_rr,
            'toe_fl': toe_fl, 'toe_fr': toe_fr,
            'toe_rl': toe_rl, 'toe_rr': toe_rr,
            # Damper
            'vd_fl': vd_f[:, 0], 'vd_fr': vd_f[:, 1],
            'vd_rl': vd_r[:, 0], 'vd_rr': vd_r[:, 1],
            'Fd_fl': Fd_fl, 'Fd_fr': Fd_fr,
            'Fd_rl': Fd_rl, 'Fd_rr': Fd_rr,
            # ARB
            'arb_torque_f': arb_torque_f, 'arb_torque_r': arb_torque_r,
            'dz_diff_f': dz_f_mm, 'dz_diff_r': dz_r_mm,
        }


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMATIC BUILDER
# Generates the animated front/side-view suspension schematic as Plotly frames.
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# ANALYTICAL 4-BAR KINEMATIC SOLVER
# Topology: body(P1_lo, P2_up) → rigid wishbones → upright(A_hub, B_top)
#
# Ground truth constraints (enforced every frame):
#   1.  Hub A = (±tw/2, R)  — wheel always on ground, never floats
#   2.  |P1→A| = l_lw       — lower wishbone rigid
#   3.  |P2→B| = l_uw       — upper wishbone rigid
#   4.  |A→B|  = h_upright  — upright rigid
#   5.  P1, P2 body-fixed   — inner pivots rotate with body roll + heave
#
# B is solved analytically as the unique circle intersection:
#   B ∈ circle(A, h_upright) ∩ circle(P2, l_uw)
# ═════════════════════════════════════════════════════════════════════════════

class _KinSolver2D:
    """
    Closed-form double-wishbone solver for front-view (lateral/vertical plane).
    Coordinate system: x=lateral (right=+), y=vertical (up=+), ground y=0.
    """

    def __init__(self, phys: SuspensionPhysics):
        P  = phys
        h  = P.h_cg
        tw = P.tw_f
        R  = P.R

        # ── Body-frame pivot coords (CG at origin, right-side positive x) ──
        # Lower WB inboard: 23% of track width outboard, 17cm below CG
        self._lw_ib = np.array([ tw * 0.230, -(h - 0.162)])
        # Upper WB inboard: 18% outboard, 2cm below CG  (shorter arm, more inclined)
        self._uw_ib = np.array([ tw * 0.183, -(h - 0.308)])
        # Rocker/bellcrank: 48% outboard, 5.5cm below CG (body-fixed)
        self._rocker  = np.array([ tw * 0.480, -(h - 0.275)])
        # Spring top mount: 43% outboard, 10cm below CG (inboard of rocker)
        self._spr_top = np.array([ tw * 0.430, -(h - 0.190)])
        # Steering rack end: 28% outboard, at lower-WB height
        self._rack    = np.array([ tw * 0.280, -(h - 0.162)])

        # ── Fixed link lengths ──────────────────────────────────────────────
        self.h_upright = 0.258   # hub to upper ball-joint [m]
        self.pushrod_f = 0.55    # fraction up upright for pushrod attachment
        self.tierod_f  = 0.12    # fraction up upright for steering arm

        # Compute WB lengths at design (zero roll, zero travel)
        hub_b  = np.array([tw / 2, R - h])      # hub in body frame at design
        ub_b   = hub_b + np.array([0.0, self.h_upright])
        self.l_lw = float(np.linalg.norm(hub_b - self._lw_ib))
        self.l_uw = float(np.linalg.norm(ub_b  - self._uw_ib))

        # Design spring length (for compression tracking)
        self._phys = P
        body_cg_d  = np.array([0.0, h])
        sol0 = self._solve_side(0.0, body_cg_d, 0.0, +1)
        self.spr_len0 = float(np.linalg.norm(sol0['spr_top'] - sol0['rocker']))

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _circle_intersect(
        c1: np.ndarray, r1: float,
        c2: np.ndarray, r2: float,
        sign: int = 1,
    ) -> np.ndarray:
        """
        Two-circle intersection via analytical formula.
        Returns one solution selected by sign=±1.
        Gracefully handles near-degenerate cases (droop/bump limit).
        """
        diff = c2 - c1
        d    = float(np.linalg.norm(diff))
        if d < 1e-9:
            return c1 + np.array([0.0, r1])
        # Clamp for numerical safety at travel limits
        r1c = min(r1, d + r2 - 1e-4)
        r2c = min(r2, d + r1 - 1e-4)
        a   = (r1c*r1c - r2c*r2c + d*d) / (2.0 * d)
        h2  = max(r1c*r1c - a*a, 0.0)
        h_  = np.sqrt(h2)
        mid  = c1 + (a / d) * diff
        perp = np.array([-diff[1], diff[0]]) / d
        return mid + sign * h_ * perp

    def _body_to_world(
        self,
        local: np.ndarray,
        cg_w:  np.ndarray,
        roll:  float,
        side:  int,
    ) -> np.ndarray:
        """
        Rigid-body transform: body frame → world frame.
        side: +1=right, -1=left (mirrors x-component of local coord).
        """
        lx = local[0] * side   # mirror for left side
        ly = local[1]
        cr, sr = np.cos(roll), np.sin(roll)
        return cg_w + np.array([lx * cr - ly * sr,
                                 lx * sr + ly * cr])

    def _solve_side(
        self,
        z_m:    float,
        cg_w:   np.ndarray,
        roll:   float,
        side:   int,
    ) -> dict:
        P = self._phys
        b2w = lambda local: self._body_to_world(local, cg_w, roll, side)

        # ── Hub A: wheel centre ALWAYS on ground ──────────────────────────────
        A = np.array([side * P.tw_f / 2.0, P.R])

        # ── Inner pivots (body-fixed, move with body) ─────────────────────────
        P1 = b2w(self._lw_ib)   # lower WB inboard
        P2 = b2w(self._uw_ib)   # upper WB inboard

        # ── Upright (Driven by analytical camber map) ─────────────────────────
        # Fetch instantaneous camber from the physics model
        camber_deg = P.camber(z_m, np.degrees(roll), 'f')
        camber_rad = np.radians(camber_deg)
        
        # Local upright hardpoints relative to Hub (A)
        # LBJ: 3cm inboard, 10cm below hub
        lbj_local = np.array([side * -0.03, -0.10])
        # UBJ: 4cm inboard, 12cm above hub
        ubj_local = np.array([side * -0.04,  0.12])

        # Rotate upright by camber angle (negative camber leans top inboard)
        theta = -side * camber_rad
        c, s = np.cos(theta), np.sin(theta)
        R_mat = np.array([[c, -s], [s, c]])

        LBJ = A + R_mat @ lbj_local
        UBJ = A + R_mat @ ubj_local

        # Upright-fixed attachment points (interpolate between LBJ and UBJ)
        PR_ob     = LBJ + self.pushrod_f * (UBJ - LBJ)
        tierod_ob = LBJ + self.tierod_f  * (UBJ - LBJ)

        # Body-fixed attachment points
        rocker  = b2w(self._rocker)
        spr_top = b2w(self._spr_top)
        rack    = b2w(self._rack)

        return {
            'P1': P1, 'P2': P2,
            'A':  A,  'B':  UBJ,  # Keep 'B' as UBJ for compatibility
            'LBJ': LBJ, 'UBJ': UBJ,
            'PR_ob':      PR_ob,
            'tierod_ob': tierod_ob,
            'rocker':     rocker,
            'spr_top':    spr_top,
            'rack':       rack,
            'camber_rad': camber_rad,
        }

    def solve(
        self,
        z_fl_m: float,
        z_fr_m: float,
        roll_deg: float,
    ) -> tuple:
        """
        Solve both sides.
        Body CG height: at design = h_cg.  Sinks by average spring compression.
        (Wheels fixed to ground; body translates down as springs compress.)

        Returns (sol_left, sol_right, body_cg_world, roll_rad).
        """
        roll_rad = np.radians(roll_deg)
        # Body heave: body sinks when springs compress (z > 0 = bump)
        y_cg = self._phys.h_cg - (z_fl_m + z_fr_m) * 0.5
        cg_w = np.array([0.0, y_cg])
        sol_l = self._solve_side(z_fl_m, cg_w, roll_rad, side=-1)
        sol_r = self._solve_side(z_fr_m, cg_w, roll_rad, side=+1)
        return sol_l, sol_r, cg_w, roll_rad


# ═════════════════════════════════════════════════════════════════════════════
# PUSHROD-ROCKER KINEMATICS
# ─────────────────────────────────────────────────────────────────────────────
# Chain: upright_pt(PR_ob) ──pushrod──► rocker_arm_end(Kkpr)
#        rocker pivot (Rpiv, body-fixed)
#        spring_arm_end(Kksr) ──spring/damper──► chassis_mount(Fspr, body-fixed)
#
# Constraint: |PR_ob - Kkpr| = L_pr  (pushrod rigid)
# Solved analytically: find rocker angle α s.t. circle(Rpiv,rpr) ∩ circle(PR_ob,L_pr)
#
# Design geometry (right side, world coords at z=0, roll=0):
#   PR_ob  ≈ (0.600, 0.275) — 28% up the upright, near lower BJ area
#   Rpiv   = (0.340, 0.185) — lower than PR_ob, gives 30° pushrod angle from horiz
#   Kkpr   ≈ (0.393, 0.147) — rocker arm 65mm, 55° CW below D-vector at design
#   Kksr   ≈ (0.340, 0.250) — spring arm 65mm, pointing straight up → near-horiz spring
#   Fspr   = (0.155, 0.185) — spring fixed end, 185mm inboard at Rpiv height
#   L_spr0 ≈ 196mm, pushrod L_pr ≈ 243mm, MR ≈ 0.65
# ═════════════════════════════════════════════════════════════════════════════

class _PushroDRocker:
    """
    Closed-form pushrod-actuated bellcrank kinematics.
    Computes rocker angle and spring compression every frame.
    Numerically robust through ±50mm wheel travel via circle-intersection.
    """

    def __init__(self, kin: '_KinSolver2D'):
        P  = kin._phys
        h  = P.h_cg
        self._K = kin

        # Body-frame pivot/mount coords (right-side positive x, mirrored for left)
        # Rocker pivot: below PR_ob, inboard → gives diagonal pushrod at ~30° from horiz
        self._rpiv_b = np.array([+0.340, -(h - 0.185)])
        # Spring fixed chassis mount: same height as rocker pivot → near-horizontal spring
        self._fspr_b = np.array([+0.155, -(h - 0.185)])

        # Pushrod attachment fraction along upright A→B (near lower BJ = typical FS)
        self._pr_frac = 0.28

        # Rocker arm lengths
        self._rpr  = 0.065   # pushrod arm [m]
        self._rspr = 0.065   # spring arm [m]

        # Spring-arm angular offset from pushrod arm on the rocker body.
        # Right side +1: +125° puts spring arm pointing ~upward → runs diagonally to Fspr
        # Left  side -1: -125° by mirror symmetry → same result on other side
        self._phi_mag = np.radians(125.0)

        # Compute design-state geometry for both sides (needed for branch selection)
        cg_design = np.array([0.0, h])
        self._alpha0 = {}   # design rocker angle per side
        self._L_pr   = {}   # pushrod length per side (constant, computed at design)
        self._L_spr0 = {}   # spring length at design per side

        for side in (+1, -1):
            sol   = kin._solve_side(0.0, cg_design, 0.0, side)
            PR_ob = sol['PR_ob']    # <-- REPLACE this line
            Rpiv  = kin._body_to_world(self._rpiv_b, cg_design, 0.0, side)
            Fspr  = kin._body_to_world(self._fspr_b, cg_design, 0.0, side)

            D       = PR_ob - Rpiv
            d_angle = float(np.arctan2(D[1], D[0]))

            # Design rocker angle: arm points 55° CW from D-vector for right side
            # (arm points down-outboard, pushrod runs at ~30° above horizontal)
            alpha0 = d_angle + side * np.radians(-55.0)
            self._alpha0[side] = alpha0

            Kkpr0 = Rpiv + self._rpr * np.array([np.cos(alpha0), np.sin(alpha0)])
            self._L_pr[side] = float(np.linalg.norm(PR_ob - Kkpr0))

            phi   = side * self._phi_mag
            beta0 = alpha0 + phi
            Kksr0 = Rpiv + self._rspr * np.array([np.cos(beta0), np.sin(beta0)])
            self._L_spr0[side] = float(np.linalg.norm(Kksr0 - Fspr))

    @staticmethod
    def _angle_dist(a: float, b: float) -> float:
        d = abs(a - b) % (2.0 * np.pi)
        return min(d, 2.0 * np.pi - d)

    def solve(self, upright_sol: dict, cg_w: np.ndarray,
              roll: float, side: int) -> dict:
        b2w = lambda local: self._K._body_to_world(local, cg_w, roll, side)

        A     = upright_sol['A']
        B     = upright_sol['B']
        PR_ob = upright_sol['PR_ob']   # <-- REPLACE this line
        Rpiv  = b2w(self._rpiv_b)
        Fspr  = b2w(self._fspr_b)

        D       = PR_ob - Rpiv
        d_len   = max(float(np.linalg.norm(D)), 1e-6)
        d_angle = float(np.arctan2(D[1], D[0]))

        rpr  = self._rpr
        L_pr = self._L_pr[side]

        cos_in  = np.clip((rpr**2 + d_len**2 - L_pr**2) / (2.0 * rpr * d_len), -1.0, 1.0)
        acos_v  = float(np.arccos(cos_in))
        alpha_a = d_angle + acos_v
        alpha_b = d_angle - acos_v

        # Select physically continuous branch
        alpha0_ref = self._alpha0[side]
        alpha = (alpha_a if self._angle_dist(alpha_a, alpha0_ref) <=
                             self._angle_dist(alpha_b, alpha0_ref) else alpha_b)

        phi  = side * self._phi_mag
        beta = alpha + phi

        Kkpr = Rpiv + rpr            * np.array([np.cos(alpha), np.sin(alpha)])
        Kksr = Rpiv + self._rspr * np.array([np.cos(beta),  np.sin(beta)])

        L_spr      = float(np.linalg.norm(Kksr - Fspr))
        comp_ratio = (self._L_spr0[side] - L_spr) / (self._L_spr0[side] + 1e-9)

        return {
            'PR_ob':      PR_ob,
            'Rpiv':       Rpiv,
            'Kkpr':       Kkpr,
            'Kksr':       Kksr,
            'Fspr':       Fspr,
            'alpha':      alpha,
            'beta':       beta,
            'comp_ratio': comp_ratio,
        }


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMATIC BUILDER  — Formula Student pushrod inboard double-wishbone
# Every joint world-position is derived analytically (zero hard-coded coords).
# ═════════════════════════════════════════════════════════════════════════════

class _SchemBuilder:
    """
    Renders one animation frame as Plotly trace dicts.
    Uses _KinSolver2D (wishbones + upright, 4-bar exact) and
    _PushroDRocker (pushrod → bellcrank → horizontal spring/damper).
    """

    def __init__(self, phys: SuspensionPhysics):
        self._P = phys
        self._K = _KinSolver2D(phys)
        self._R = _PushroDRocker(self._K)

    # ── Primitive generators ─────────────────────────────────────────────────

    @staticmethod
    def _line(p0, p1, color=_TEXT_M, width=3, name='', dash='solid') -> dict:
        ht = f'{name}<extra></extra>' if name else None
        return dict(type='scatter', x=[p0[0], p1[0]], y=[p0[1], p1[1]],
                    mode='lines', line=dict(color=color, width=width, dash=dash),
                    showlegend=False,
                    hovertemplate=ht, hoverinfo='skip' if not name else 'text')

    @staticmethod
    def _circle_trace(cx, cy, r, n=80,
                      fill='rgba(60,60,70,0.85)', outline='#555566') -> dict:
        th = np.linspace(0, 2*np.pi, n)
        return dict(type='scatter',
                    x=(cx + r*np.cos(th)).tolist(),
                    y=(cy + r*np.sin(th)).tolist(),
                    mode='lines', fill='toself', fillcolor=fill,
                    line=dict(color=outline, width=2),
                    showlegend=False, hoverinfo='skip')

    @staticmethod
    def _spring_trace(p0: np.ndarray, p1: np.ndarray,
                      comp_ratio: float = 0.0, n_coils: int = 7) -> dict:
        """
        Coil spring between p0 and p1.
        comp_ratio ∈ [-0.4, +0.4]: +ve = compressed (bunched coils), -ve = extended.
        Width widens under compression (constant wire length conserved).
        """
        L    = float(np.linalg.norm(p1 - p0)) + 1e-9
        tang = (p1 - p0) / L
        norm = np.array([-tang[1], tang[0]])

        # Coil amplitude grows with compression (coils squash outward)
        w = 0.0185 * (1.0 + 1.2 * max(comp_ratio, 0.0))

        # End-taper zones: first and last 10% of spring length are straight
        margin = 0.10
        s = np.linspace(0, 1, n_coils * 20 + 4)
        env = np.clip(s / margin, 0, 1) * np.clip((1 - s) / margin, 0, 1)
        osc = w * env * np.sin(n_coils * 2 * np.pi * s)

        pts = p0[np.newaxis, :] + np.outer(s, p1 - p0) + np.outer(osc, norm)
        return dict(type='scatter', x=pts[:, 0].tolist(), y=pts[:, 1].tolist(),
                    mode='lines', line=dict(color=_CYAN, width=2.5),
                    showlegend=False, hoverinfo='skip')

    @staticmethod
    def _damper_traces(p0: np.ndarray, p1: np.ndarray,
                       comp_ratio: float = 0.0) -> list:
        """
        Damper: outer cylinder (wheel side) + sliding rod (body side) + piston cap.
        comp_ratio ∈ [-1, 1]: +1 = fully bumped (rod pushed into cylinder).
        Piston fraction tracks compression: mid = 0.5 + 0.2*comp_ratio.
        """
        diff = p1 - p0
        L    = float(np.linalg.norm(diff)) + 1e-9
        tang = diff / L
        norm = np.array([-tang[1], tang[0]])
        gap  = 0.013   # half-width of damper symbol [m]

        # Piston at fraction `mid` along the total damper length
        mid = np.clip(0.50 + 0.22 * comp_ratio, 0.28, 0.72)

        traces = []
        # Cylinder walls (p0 side = wheel): p0 → piston
        for sgn in (+1, -1):
            o = sgn * gap * norm
            # Cylinder (thick, wheel to piston)
            traces.append(dict(type='scatter',
                               x=[p0[0]+o[0], p0[0]+mid*diff[0]+o[0]],
                               y=[p0[1]+o[1], p0[1]+mid*diff[1]+o[1]],
                               mode='lines', line=dict(color=_AMBER, width=3),
                               showlegend=False, hoverinfo='skip'))
            # Rod (thin, piston to body)
            traces.append(dict(type='scatter',
                               x=[p0[0]+mid*diff[0]+o[0]*0.55, p1[0]+o[0]*0.55],
                               y=[p0[1]+mid*diff[1]+o[1]*0.55, p1[1]+o[1]*0.55],
                               mode='lines', line=dict(color=_AMBER, width=1.8),
                               showlegend=False, hoverinfo='skip'))
        # Piston cap bar
        pc = p0 + mid * diff
        traces.append(dict(type='scatter',
                           x=[pc[0]-gap*1.5*norm[0], pc[0]+gap*1.5*norm[0]],
                           y=[pc[1]-gap*1.5*norm[1], pc[1]+gap*1.5*norm[1]],
                           mode='lines', line=dict(color=_AMBER, width=4),
                           showlegend=False, hoverinfo='skip'))
        return traces

    @staticmethod
    def _wheel_trace(cx: float, cy: float, R: float, n_spokes: int = 6) -> list:
        """
        Full wheel: rubber tyre, sidewall, rim, hub, spokes.
        Contact patch is a flat segment at y=0 (tyre deformation).
        """
        traces = []
        th     = np.linspace(0, 2*np.pi, 128)
        # Tyre outer (with contact-patch flattening: slight squish at bottom)
        r_mod  = R * (1.0 - 0.018 * np.exp(-8.0 * (np.sin(th/2)**2 + 0.001)))
        x_tyre = cx + r_mod * np.cos(th)
        y_tyre = np.maximum(cy + r_mod * np.sin(th), 0.0)  # never below ground
        traces.append(dict(type='scatter', x=x_tyre.tolist(), y=y_tyre.tolist(),
                           mode='lines', fill='toself',
                           fillcolor='rgba(35,35,40,0.92)',
                           line=dict(color='#3A3A45', width=2),
                           showlegend=False, hoverinfo='skip'))
        # Rim
        rim_r = R * 0.68
        traces.append(dict(type='scatter',
                           x=(cx + rim_r*np.cos(th)).tolist(),
                           y=(cy + rim_r*np.sin(th)).tolist(),
                           mode='lines', fill='toself',
                           fillcolor='rgba(80,90,110,0.6)',
                           line=dict(color=_CYAN, width=1.8),
                           showlegend=False, hoverinfo='skip'))
        # Spokes
        for k in range(n_spokes):
            a = 2*np.pi * k / n_spokes
            traces.append(dict(type='scatter',
                               x=[cx, cx + rim_r*np.cos(a)],
                               y=[cy, cy + rim_r*np.sin(a)],
                               mode='lines',
                               line=dict(color='rgba(100,120,150,0.7)', width=1.5),
                               showlegend=False, hoverinfo='skip'))
        # Hub dot
        traces.append(dict(type='scatter', x=[cx], y=[cy], mode='markers',
                           marker=dict(size=7, color=_CYAN), showlegend=False,
                           hoverinfo='skip'))
        return traces

    @staticmethod
    def _camber_indicator(A: np.ndarray, B: np.ndarray, R: float,
                          camber_rad: float) -> dict:
        """Green dashed line through wheel centre at camber angle from vertical."""
        r_in = R * 0.85
        sa, ca = np.sin(camber_rad), np.cos(camber_rad)
        return dict(type='scatter',
                    x=[A[0] - r_in*sa, A[0] + r_in*sa],
                    y=[A[1] - r_in*ca, A[1] + r_in*ca],
                    mode='lines',
                    line=dict(color=_GREEN, width=2, dash='dash'),
                    showlegend=False,
                    hovertemplate=f'Camber: {np.degrees(camber_rad):.2f}°<extra></extra>')

    @staticmethod
    def _rocker_trace(Rpiv: np.ndarray, Kkpr: np.ndarray,
                      Kksr: np.ndarray) -> list:
        """
        Bellcrank / rocker: solid filled triangle (Rpiv, Kkpr, Kksr) +
        a small hub circle at the pivot pin.
        """
        traces = []
        # Filled triangle
        xs = [Rpiv[0], Kkpr[0], Kksr[0], Rpiv[0]]
        ys = [Rpiv[1], Kkpr[1], Kksr[1], Rpiv[1]]
        traces.append(dict(type='scatter', x=xs, y=ys,
                           mode='lines', fill='toself',
                           fillcolor='rgba(220,160,40,0.45)',
                           line=dict(color='rgba(240,200,60,0.9)', width=2),
                           showlegend=False, hoverinfo='skip'))
        # Pivot pin circle
        th = np.linspace(0, 2*np.pi, 32)
        r  = 0.010
        traces.append(dict(type='scatter',
                           x=(Rpiv[0] + r*np.cos(th)).tolist(),
                           y=(Rpiv[1] + r*np.sin(th)).tolist(),
                           mode='lines', fill='toself',
                           fillcolor='rgba(60,60,70,1.0)',
                           line=dict(color='#C8C8D8', width=1.5),
                           showlegend=False, hoverinfo='skip'))
        # Arm-end joints
        for pt in [Kkpr, Kksr]:
            traces.append(dict(type='scatter', x=[pt[0]], y=[pt[1]],
                               mode='markers',
                               marker=dict(size=5, color='#E8E060',
                                           symbol='circle'),
                               showlegend=False, hoverinfo='skip'))
        return traces

    # ── One-side assembly — full FS pushrod-rocker chain ──────────────────────

    def _draw_corner(self, sol: dict, cg_w: np.ndarray,
                     roll_rad: float, side: int,
                     side_label: str) -> tuple:
        P       = self._P
        A       = sol['A']           # hub
        LBJ     = sol['LBJ']         # lower ball joint
        UBJ     = sol['UBJ']         # upper ball joint
        P1      = sol['P1']          # lower WB inboard
        P2      = sol['P2']          # upper WB inboard
        tierod  = sol['tierod_ob']   # steering arm on upright
        rack    = sol['rack']        # steering rack end (body-fixed)
        camber  = sol['camber_rad']

        # ── Pushrod-rocker solution ──────────────────────────────────────────
        rk      = self._R.solve(sol, cg_w, roll_rad, side)
        PR_ob   = rk['PR_ob']        # pushrod on upright
        Rpiv    = rk['Rpiv']         # rocker pivot (body-fixed)
        Kkpr    = rk['Kkpr']         # pushrod arm tip
        Kksr    = rk['Kksr']         # spring arm tip
        Fspr    = rk['Fspr']         # spring chassis mount (body-fixed)
        comp    = rk['comp_ratio']
        dcomp   = float(np.clip(comp / 0.25, -1.0, 1.0))

        traces = []

        # ── Tyre + wheel ──────────────────────────────────────────────────────
        traces += self._wheel_trace(A[0], A[1], P.R)

        # ── A-arm wishbones ───────────────────────────────────────────────────
        # Lower WB: inboard pivot → lower ball joint
        traces.append(self._line(P1, LBJ, color='#4A80D0', width=4,
                                  name=f'Lower WB {side_label}'))
        # Upper WB: inboard pivot → upper ball joint
        traces.append(self._line(P2, UBJ, color='#2A5898', width=4,
                                  name=f'Upper WB {side_label}'))

        # ── Upright ───────────────────────────────────────────────────────────
        traces.append(self._line(LBJ, UBJ, color='#C0C0D0', width=6,
                                  name=f'Upright {side_label}'))
        # Brake disc outline behind upright (cosmetic — dark circle at hub)
        th  = np.linspace(0, 2*np.pi, 48)
        br  = P.R * 0.42
        traces.append(dict(type='scatter',
                           x=(A[0] + br*np.cos(th)).tolist(),
                           y=(A[1] + br*np.sin(th)).tolist(),
                           mode='lines',
                           line=dict(color='rgba(140,50,50,0.65)', width=2.5),
                           showlegend=False, hoverinfo='skip'))

        # ── Joint dots ────────────────────────────────────────────────────────
        for pt, sz, col in [
            (A,   9, '#E0E0F0'),   # hub
            (LBJ, 7, '#C0C0D8'),   # lower BJ
            (UBJ, 7, '#C0C0D8'),   # upper BJ
            (P1,  6, '#8090B8'),   # lower WB inboard
            (P2,  6, '#8090B8'),   # upper WB inboard
        ]:
            traces.append(dict(type='scatter', x=[pt[0]], y=[pt[1]],
                               mode='markers',
                               marker=dict(size=sz, color=col, symbol='circle'),
                               showlegend=False, hoverinfo='skip'))

        # ── Joint dots ────────────────────────────────────────────────────────
        for pt, sz, col in [
            (A,  9, '#E0E0F0'),   # hub / lower BJ
            (B,  7, '#C0C0D8'),   # upper BJ
            (P1, 6, '#8090B8'),   # lower WB inboard
            (P2, 6, '#8090B8'),   # upper WB inboard
        ]:
            traces.append(dict(type='scatter', x=[pt[0]], y=[pt[1]],
                               mode='markers',
                               marker=dict(size=sz, color=col, symbol='circle'),
                               showlegend=False, hoverinfo='skip'))

        # ── Pushrod ───────────────────────────────────────────────────────────
        # Thin diagonal strut: upright attachment → rocker arm tip (Kkpr)
        traces.append(self._line(PR_ob, Kkpr,
                                  color='rgba(210,200,60,0.90)', width=2.5,
                                  name=f'Pushrod {side_label}'))
        # Pushrod end joint
        traces.append(dict(type='scatter', x=[PR_ob[0]], y=[PR_ob[1]],
                           mode='markers',
                           marker=dict(size=5, color='#D8CC40', symbol='circle'),
                           showlegend=False, hoverinfo='skip'))

        # ── Bellcrank / rocker ────────────────────────────────────────────────
        traces += self._rocker_trace(Rpiv, Kkpr, Kksr)

        # ── Spring (near-horizontal inboard) ──────────────────────────────────
        traces.append(self._spring_trace(Kksr, Fspr,
                                          comp_ratio=comp, n_coils=7))

        # ── Damper (parallel to spring, offset perpendicular) ─────────────────
        diff  = Fspr - Kksr
        Ld    = np.linalg.norm(diff) + 1e-9
        perp  = np.array([-diff[1], diff[0]]) / Ld * 0.022
        traces += self._damper_traces(Kksr + perp, Fspr + perp, dcomp)

        # Spring chassis mount pin
        traces.append(dict(type='scatter', x=[Fspr[0]], y=[Fspr[1]],
                           mode='markers',
                           marker=dict(size=7, color='#A0A0B0',
                                       symbol='square'),
                           showlegend=False,
                           hovertemplate=f'Spring mount<extra></extra>'))

        # ── Steering tie rod ──────────────────────────────────────────────────
        traces.append(self._line(rack, tierod,
                                  color='rgba(220,120,50,0.80)', width=2,
                                  name=f'Tie rod {side_label}'))

        # ── Camber indicator ──────────────────────────────────────────────────
        traces.append(self._camber_indicator(A, B, P.R, camber))

        return traces, Rpiv

    # ── Full frame builders ───────────────────────────────────────────────────

    def build_traces(self, z_fl_m: float, z_fr_m: float,
                     roll_deg: float, view: str = 'front') -> list:
        if view == 'front':
            return self._front_traces(z_fl_m, z_fr_m, roll_deg)
        return self._side_traces(z_fl_m, z_fr_m, roll_deg)

    def _front_traces(self, z_fl: float, z_fr: float, roll_deg: float) -> list:
        K        = self._K
        P        = self._P
        roll_rad = np.radians(roll_deg)
        sol_l, sol_r, cg_w, roll_rad = K.solve(z_fl, z_fr, roll_deg)

        traces = []

        # ── Ground strip ──────────────────────────────────────────────────────
        xg = P.tw_f / 2 + 0.30
        traces.append(dict(type='scatter', x=[-xg, xg], y=[0, 0],
                           mode='lines',
                           line=dict(color='rgba(255,255,255,0.20)', width=2),
                           showlegend=False, hoverinfo='skip'))
        for xh in np.linspace(-xg + 0.06, xg - 0.06, 16):
            traces.append(dict(type='scatter',
                               x=[xh, xh + 0.08], y=[0, -0.045],
                               mode='lines',
                               line=dict(color='rgba(255,255,255,0.07)', width=1),
                               showlegend=False, hoverinfo='skip'))

        # ── Body chassis cross-section (drawn first = behind suspension) ───────
        bx, by  = float(cg_w[0]), float(cg_w[1])
        cr, sr  = np.cos(roll_rad), np.sin(roll_rad)
        body_hw = P.tw_f * 0.33
        body_hh = 0.078
        bxs = np.array([-body_hw, body_hw, body_hw, -body_hw, -body_hw])
        bys = np.array([-body_hh, -body_hh,  body_hh,  body_hh, -body_hh])
        bxr = bx + bxs * cr - bys * sr
        byr = by + bxs * sr + bys * cr
        traces.append(dict(type='scatter', x=bxr.tolist(), y=byr.tolist(),
                           mode='lines', fill='toself',
                           fillcolor='rgba(180,30,30,0.18)',
                           line=dict(color=_RED, width=2),
                           showlegend=False, hoverinfo='skip'))

        # ── Both suspension corners ───────────────────────────────────────────
        tr_l, Rpiv_l = self._draw_corner(sol_l, cg_w, roll_rad, side=-1, side_label='L')
        tr_r, Rpiv_r = self._draw_corner(sol_r, cg_w, roll_rad, side=+1, side_label='R')
        traces += tr_l
        traces += tr_r

        # ── ARB cross-link ────────────────────────────────────────────────────
        # Torsion bar runs between rocker pivots, slight mid-arc for clarity.
        mid_arb = np.array([0.0, (Rpiv_l[1] + Rpiv_r[1]) * 0.5 + 0.018])
        for seg in [(Rpiv_l, mid_arb), (mid_arb, Rpiv_r)]:
            traces.append(self._line(*seg,
                                      color='rgba(160,80,220,0.85)', width=3,
                                      name='ARB'))
        for pt in [Rpiv_l, Rpiv_r]:
            traces.append(dict(type='scatter', x=[pt[0]], y=[pt[1]],
                               mode='markers',
                               marker=dict(size=8, color='rgba(180,100,240,0.9)',
                                           symbol='diamond'),
                               showlegend=False, hoverinfo='skip'))

        # ── CG marker ─────────────────────────────────────────────────────────
        traces.append(dict(type='scatter', x=[bx], y=[by], mode='markers',
                           marker=dict(size=12, color=_AMBER, symbol='x-thin',
                                       line=dict(width=3, color=_AMBER)),
                           showlegend=False, hovertemplate='CG<extra></extra>'))

        # ── Roll-centre indicator ─────────────────────────────────────────────
        avg_z     = (z_fl + z_fr) * 0.5
        h_rc_curr = P.h_rc_f + P.dh_rc_f * avg_z
        traces.append(dict(type='scatter', x=[0], y=[h_rc_curr], mode='markers',
                           marker=dict(size=10, color=_GREEN, symbol='circle-open',
                                       line=dict(width=2.5)),
                           showlegend=False,
                           hovertemplate=f'Roll Centre: {h_rc_curr*1000:.0f} mm<extra></extra>'))

        return traces

    def _side_traces(self, z_fl: float, z_fr: float, roll_deg: float) -> list:
        """
        Side view (longitudinal/vertical plane).
        Both wheels grounded. Body pitches about CG. Wishbones shown as
        schematic A-arm silhouettes. Pushrod/spring/damper per axle.
        Anti-dive and anti-squat force lines computed from vehicle_params.
        """
        P  = self._P
        R  = P.R
        wb = P.wb
        lf = P.lf
        h  = P.h_cg

        # ── Quasi-static pitch from differential heave ────────────────────────
        # Front in more bump than rear → nose dives
        z_front = z_fl                          # front representative
        z_rear  = (z_fl + z_fr) * 0.35         # rear responds less than front
        pitch_deg = float(np.degrees((z_front - z_rear) / wb))
        pitch_rad = np.radians(pitch_deg)

        # Body CG sinks with average front compression
        y_cg = h - (z_fl + z_fr) * 0.5

        # ── Wheel centres (always on ground) ──────────────────────────────────
        Wf = np.array([0.0, R])     # front wheel centre
        Wr = np.array([wb,  R])     # rear  wheel centre

        # ── Body-fixed → world (pitch about CG) ───────────────────────────────
        def b2w_side(local_x, local_y):
            dlx = local_x - lf
            dly = local_y - y_cg
            wx  = lf  + dlx * np.cos(pitch_rad) - dly * np.sin(pitch_rad)
            wy  = y_cg + dlx * np.sin(pitch_rad) + dly * np.cos(pitch_rad)
            return np.array([wx, wy])

        # ── Body rectangle ─────────────────────────────────────────────────────
        body_hw = wb * 0.46
        body_hh = 0.082
        bxs = np.array([-body_hw, body_hw, body_hw, -body_hw, -body_hw])
        bys = np.array([-body_hh, -body_hh, body_hh, body_hh, -body_hh])
        cr, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        bxr = lf + bxs * cr - bys * sp
        byr = y_cg + bxs * sp + bys * cr

        # ── Wishbone inboard attachment points (side view) ────────────────────
        f_lo_ib = b2w_side(lf * 0.18,  h - 0.170)
        f_up_ib = b2w_side(lf * 0.14,  h - 0.022)
        r_lo_ib = b2w_side(wb - lf * 0.20, h - 0.168)
        r_up_ib = b2w_side(wb - lf * 0.14, h - 0.020)

        # ── Pushrod / spring attachment (side view) ────────────────────────────
        # Front: pushrod from upright mid-height to rocker
        f_pr_ob   = np.array([Wf[0] + 0.115, R + 0.130])
        f_rocker  = b2w_side(lf * 0.30, h - 0.055)
        f_spr_top = b2w_side(lf * 0.28, h - 0.190)

        # Rear: pullrod (runs downward from upright to rocker below)
        r_pr_ob   = np.array([Wr[0] - 0.115, R + 0.108])
        r_rocker  = b2w_side(wb - lf * 0.32, h - 0.058)
        r_spr_top = b2w_side(wb - lf * 0.30, h - 0.185)

        # Spring compression for front/rear
        spr_len0  = self._K.spr_len0
        comp_f    = (spr_len0 - np.linalg.norm(f_spr_top - f_rocker)) / spr_len0
        comp_r    = (spr_len0 - np.linalg.norm(r_spr_top - r_rocker)) / spr_len0

        # ── Anti-geometry force lines ──────────────────────────────────────────
        # Front anti-dive: contact patch → instant centre on body
        ad_frac   = float(P._p('anti_dive_f'))
        anti_dive_ib = b2w_side(0.0, h - h * (1.0 - ad_frac))
        # Rear anti-squat: contact patch → instant centre
        as_frac   = float(P._p('anti_squat'))
        anti_squat_ib = b2w_side(wb, h - h * (1.0 - as_frac))

        traces = []

        # Ground
        traces.append(dict(type='scatter', x=[-0.32, wb+0.32], y=[0, 0],
                           mode='lines',
                           line=dict(color='rgba(255,255,255,0.18)', width=2),
                           showlegend=False, hoverinfo='skip'))
        for xh in np.linspace(-0.25, wb+0.25, 18):
            traces.append(dict(type='scatter', x=[xh, xh+0.06], y=[0, -0.04],
                               mode='lines',
                               line=dict(color='rgba(255,255,255,0.07)', width=1),
                               showlegend=False, hoverinfo='skip'))

        # Body
        traces.append(dict(type='scatter', x=bxr.tolist(), y=byr.tolist(),
                           mode='lines', fill='toself',
                           fillcolor='rgba(200,40,40,0.22)',
                           line=dict(color=_RED, width=2),
                           showlegend=False, hoverinfo='skip'))

        # CG
        traces.append(dict(type='scatter', x=[lf], y=[y_cg], mode='markers',
                           marker=dict(size=11, color=_AMBER, symbol='x-thin',
                                       line=dict(width=3, color=_AMBER)),
                           showlegend=False, hovertemplate='CG<extra></extra>'))

        # Tyres (side view: circles)
        for wc in [Wf, Wr]:
            traces += self._wheel_trace(wc[0], wc[1], R, n_spokes=6)

        # Wishbones (front)
        traces.append(self._line(Wf, f_lo_ib, color='#4878C8', width=3.5))
        traces.append(self._line(np.array([Wf[0], Wf[1]+0.235]), f_up_ib,
                                  color='#2C5890', width=3.5))
        # Wishbones (rear)
        traces.append(self._line(Wr, r_lo_ib, color='#4878C8', width=3.5))
        traces.append(self._line(np.array([Wr[0], Wr[1]+0.235]), r_up_ib,
                                  color='#2C5890', width=3.5))

        # Pushrods
        traces.append(self._line(f_pr_ob, f_rocker,
                                  color='rgba(200,190,60,0.85)', width=2))
        traces.append(self._line(r_pr_ob, r_rocker,
                                  color='rgba(200,190,60,0.85)', width=2))

        # Springs + dampers
        for (lo, hi, comp) in [(f_rocker, f_spr_top, comp_f),
                                (r_rocker, r_spr_top, comp_r)]:
            traces.append(self._spring_trace(lo, hi, comp_ratio=comp))
            diff = hi - lo
            L    = np.linalg.norm(diff) + 1e-9
            perp = np.array([-diff[1], diff[0]]) / L * 0.026
            traces += self._damper_traces(lo + perp, hi + perp,
                                           np.clip(comp / 0.30, -1, 1))

        # Anti-geometry lines (dashed)
        traces.append(dict(type='scatter',
                           x=[Wf[0], anti_dive_ib[0]], y=[0, anti_dive_ib[1]],
                           mode='lines',
                           line=dict(color='rgba(0,192,242,0.55)',
                                      width=1.5, dash='dot'),
                           showlegend=False,
                           hovertemplate='Anti-Dive<extra></extra>'))
        traces.append(dict(type='scatter',
                           x=[Wr[0], anti_squat_ib[0]], y=[0, anti_squat_ib[1]],
                           mode='lines',
                           line=dict(color='rgba(35,209,96,0.55)',
                                      width=1.5, dash='dot'),
                           showlegend=False,
                           hovertemplate='Anti-Squat<extra></extra>'))

        return traces


# ═════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def _layout(height: int = 300, title: str = '') -> dict:
    lay = dict(**_BASE_LAY)
    lay['height'] = height
    lay['title']  = dict(text=title,
                          font=dict(color=_CYAN, size=12, family='Courier New'),
                          x=0.0, xanchor='left')
    return lay


def _legend() -> dict:
    return dict(bgcolor='rgba(26,30,46,0.85)', bordercolor=_BORDER_M,
                borderwidth=1, font=dict(family='Courier New', size=10, color=_TEXT_B))


def _axis_titles(xlabel: str, ylabel: str) -> dict:
    tf = dict(color=_TEXT_M, size=10, family='Courier New')
    return dict(
        xaxis_title=dict(text=xlabel, font=tf),
        yaxis_title=dict(text=ylabel, font=tf),
    )


def chart_corner_loads(sim: dict) -> go.Figure:
    t = sim['t']
    fig = go.Figure()
    for key, col, lbl in [
        ('Fz_fl', _CYAN,  'FL'), ('Fz_fr', _GREEN, 'FR'),
        ('Fz_rl', _AMBER, 'RL'), ('Fz_rr', _RED,   'RR'),
    ]:
        fig.add_trace(go.Scatter(x=t, y=sim[key], mode='lines', name=lbl,
                                  line=dict(color=col, width=1.8)))
    lay = _layout(260, 'CORNER VERTICAL LOADS [N]')
    lay.update(_axis_titles('Time [s]', 'Fz [N]'))
    lay['legend'] = dict(**_legend(), orientation='h', x=0, y=1.12)
    fig.update_layout(**lay)
    return fig


def chart_wheel_travel(sim: dict) -> go.Figure:
    t = sim['t']
    fig = go.Figure()
    for key, col, lbl in [
        ('z_fl', _CYAN, 'FL'), ('z_fr', _GREEN, 'FR'),
        ('z_rl', _AMBER,'RL'), ('z_rr', _RED,   'RR'),
    ]:
        fig.add_trace(go.Scatter(x=t, y=sim[key], mode='lines', name=lbl,
                                  line=dict(color=col, width=1.8)))
    # Bump stop engagement threshold (instantiate locally, no circular import)
    bs = SuspensionPhysics().bs_gap * 1000
    fig.add_hline(y=bs, line=dict(color='rgba(255,75,75,0.60)', dash='dot', width=1.5),
                  annotation_text='BUMP STOP',
                  annotation_font=dict(color=_RED, size=9, family='Courier New'))
    lay = _layout(260, 'WHEEL TRAVEL [mm]  (+ve = BUMP)')
    lay.update(_axis_titles('Time [s]', 'Travel [mm]'))
    lay['legend'] = dict(**_legend(), orientation='h', x=0, y=1.12)
    fig.update_layout(**lay)
    return fig


def chart_damper_fv(sim: dict, phys: SuspensionPhysics, corner: str = 'fl') -> go.Figure:
    axle = 'f' if corner in ('fl', 'fr') else 'r'
    v_range = np.linspace(-0.55, 0.55, 600)
    F_char  = phys.digressive_damper(v_range, axle)
    vk = phys.vk_f if axle == 'f' else phys.vk_r

    fig = go.Figure()
    bump_m = v_range >= 0
    fig.add_trace(go.Scatter(x=v_range[bump_m], y=F_char[bump_m],
                              mode='lines', name='Bump',
                              line=dict(color=_CYAN, width=2.5)))
    fig.add_trace(go.Scatter(x=v_range[~bump_m], y=F_char[~bump_m],
                              mode='lines', name='Rebound',
                              line=dict(color=_AMBER, width=2.5)))
    # Knee markers
    Fk_b = phys.digressive_damper(np.array([vk]),  axle)[0]
    Fk_r = phys.digressive_damper(np.array([-vk]), axle)[0]
    fig.add_trace(go.Scatter(x=[vk, -vk], y=[Fk_b, Fk_r], mode='markers',
                              marker=dict(size=9, color=_GREEN, symbol='diamond'),
                              name=f'v_knee={vk:.2f}m/s'))
    # Operating point cloud
    fig.add_trace(go.Scatter(x=sim[f'vd_{corner}'], y=sim[f'Fd_{corner}'],
                              mode='markers', name='Op. Cloud',
                              marker=dict(size=3, color=_RED, opacity=0.45)))

    cn = corner.upper()
    lay = _layout(290, f'DAMPER F-v CHARACTERISTIC  [{cn}]')
    lay.update(_axis_titles('Damper Velocity [m/s]', 'Force [N]'))
    lay['legend'] = _legend()
    fig.update_layout(**lay)
    return fig


def chart_camber_map(phys: SuspensionPhysics) -> go.Figure:
    z_mm  = np.linspace(-60, 60, 300)
    z_m   = z_mm / 1000
    fig   = go.Figure()
    for roll, opacity in [(0, 1.0), (2, 0.55), (4, 0.30)]:
        lbl_f = f'Front (roll={roll}°)'
        lbl_r = f'Rear (roll={roll}°)'
        cf = np.array([phys.camber(z, roll, 'f') for z in z_m])
        cr = np.array([phys.camber(z, roll, 'r') for z in z_m])
        dash = 'solid' if roll == 0 else 'dash'
        fig.add_trace(go.Scatter(x=z_mm, y=cf, mode='lines', name=lbl_f,
                                  line=dict(color=_CYAN, width=2, dash=dash),
                                  opacity=opacity))
        fig.add_trace(go.Scatter(x=z_mm, y=cr, mode='lines', name=lbl_r,
                                  line=dict(color=_AMBER, width=2, dash=dash),
                                  opacity=opacity))
    # Hoosier R20 optimal window
    fig.add_hrect(y0=-3.0, y1=-1.5,
                  fillcolor='rgba(35,209,96,0.07)',
                  line=dict(color='rgba(35,209,96,0.35)', width=1),
                  annotation_text='TTC Optimal [-1.5°, -3.0°]',
                  annotation_font=dict(color=_GREEN, size=9, family='Courier New'))
    lay = _layout(290, 'CAMBER vs WHEEL TRAVEL  [K&C MAP]')
    lay.update(_axis_titles('Wheel Travel [mm]  (+bump)', 'Camber [°]'))
    lay['legend'] = _legend()
    fig.update_layout(**lay)
    return fig


def chart_camber_time(sim: dict) -> go.Figure:
    t = sim['t']
    fig = go.Figure()
    for key, col, lbl in [
        ('camber_fl', _CYAN,  'FL'), ('camber_fr', _GREEN, 'FR'),
        ('camber_rl', _AMBER, 'RL'), ('camber_rr', _RED,   'RR'),
    ]:
        fig.add_trace(go.Scatter(x=t, y=sim[key], mode='lines', name=lbl,
                                  line=dict(color=col, width=1.8)))
    fig.add_hrect(y0=-3.0, y1=-1.5,
                  fillcolor='rgba(35,209,96,0.06)',
                  line=dict(color='rgba(35,209,96,0.25)', width=1))
    lay = _layout(260, 'CAMBER EVOLUTION [°]')
    lay.update(_axis_titles('Time [s]', 'Camber [°]'))
    lay['legend'] = dict(**_legend(), orientation='h', x=0, y=1.12)
    fig.update_layout(**lay)
    return fig


def chart_arb_coupling(sim: dict) -> go.Figure:
    t = sim['t']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=sim['arb_torque_f'], mode='lines',
                              name='Front ARB', line=dict(color=_CYAN, width=2)))
    fig.add_trace(go.Scatter(x=t, y=sim['arb_torque_r'], mode='lines',
                              name='Rear ARB', line=dict(color=_AMBER, width=2)))
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.12)', width=1))
    lay = _layout(250, 'ARB COUPLING TORQUE [N·m]')
    lay.update(_axis_titles('Time [s]', 'Torque [N·m]'))
    lay['legend'] = dict(**_legend(), orientation='h', x=0, y=1.12)
    fig.update_layout(**lay)
    return fig


def chart_roll_gradient(sim: dict, phys: SuspensionPhysics) -> go.Figure:
    ay_g = sim['ay_g']
    roll = sim['roll']
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=ay_g, y=roll, mode='markers',
                              name='Sim data',
                              marker=dict(size=3.5, color=_CYAN, opacity=0.55)))
    ay_line  = np.linspace(-2.0, 2.0, 200)
    roll_lin = np.array([phys.body_roll(a * phys.g) for a in ay_line])
    fig.add_trace(go.Scatter(x=ay_line, y=roll_lin, mode='lines',
                              name='K-model',
                              line=dict(color=_AMBER, width=2, dash='dash')))
    grad = np.degrees(phys.m_s * phys.g * (phys.h_cg - phys.h_rc_f) / phys.K_phi)
    fig.add_annotation(x=1.3, y=roll_lin[np.argmin(np.abs(ay_line - 1.3))],
                        text=f'∂φ/∂ay = {grad:.2f} °/G',
                        showarrow=True, arrowhead=2, arrowcolor=_AMBER,
                        font=dict(color=_AMBER, size=10, family='Courier New'))
    lay = _layout(260, 'ROLL GRADIENT [°/G]')
    lay.update(_axis_titles('Lateral Accel [G]', 'Body Roll [°]'))
    lay['legend'] = _legend()
    fig.update_layout(**lay)
    return fig


def chart_g_g(sim: dict) -> go.Figure:
    """g-g diagram coloured by wheel travel spread (setup load sensitivity)."""
    ax_g = sim['ax_g']
    ay_g = sim['ay_g']
    # Colour by max corner Fz spread (N) — indicates load sensitivity
    spread = (np.maximum(sim['Fz_fl'], sim['Fz_fr']) -
              np.minimum(sim['Fz_fl'], sim['Fz_fr']))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ay_g, y=ax_g, mode='markers',
        marker=dict(size=4, color=spread, colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title=dict(text='ΔFz [N]',
                                             font=dict(color=_TEXT_M, size=10)),
                                  tickfont=dict(color=_TEXT_M, size=9))),
        showlegend=False,
    ))
    lay = _layout(270, 'g-g DIAGRAM  [COLOURED: FRONT ΔFz]')
    lay.update(_axis_titles('Lateral [G]', 'Longitudinal [G]'))
    fig.update_layout(**lay)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ANIMATED SCHEMATIC
# ═════════════════════════════════════════════════════════════════════════════

def build_animated_schematic(
    sim: dict,
    phys: SuspensionPhysics,
    view: str = 'front',
    frame_step: int = 5,
) -> go.Figure:
    """
    Build a Plotly figure with frames for client-side animation.
    Each frame recomputes wishbone positions, spring coils, and damper piston
    from the quasi-static suspension state at that timestep.
    """
    builder = _SchemBuilder(phys)
    t   = sim['t']
    idx = np.arange(0, len(t), frame_step)
    P   = phys

    def _state_at(i):
        z_fl = sim['z_fl'][i] / 1000
        z_fr = sim['z_fr'][i] / 1000
        roll = sim['roll'][i]
        return z_fl, z_fr, roll

    # ── Build initial figure from first frame ─────────────────────────────────
    z_fl0, z_fr0, roll0 = _state_at(0)
    traces0 = builder.build_traces(z_fl0, z_fr0, roll0, view=view)

    # Convert trace dicts to go objects for initial data
    def _dict_to_trace(d: dict) -> go.Scatter:
        kw = {k: v for k, v in d.items() if k != 'type'}
        return go.Scatter(**kw)

    fig = go.Figure(data=[_dict_to_trace(d) for d in traces0])

    # ── Frames ────────────────────────────────────────────────────────────────
    frames = []
    for i in idx:
        z_fl, z_fr, roll = _state_at(i)
        td = builder.build_traces(z_fl, z_fr, roll, view=view)
        # Overlay: annotation with live telemetry
        camber_fl = P.camber(z_fl,  roll, 'f')
        camber_fr = P.camber(z_fr, -roll, 'f')
        annots = [
            dict(x=-P.tw_f / 2, y=-0.065, text=f'FL {z_fl*1000:+.1f}mm',
                 showarrow=False, font=dict(color=_CYAN, size=10, family='Courier New')),
            dict(x=+P.tw_f / 2, y=-0.065, text=f'FR {z_fr*1000:+.1f}mm',
                 showarrow=False, font=dict(color=_CYAN, size=10, family='Courier New')),
            dict(x=0, y=P.h_cg + 0.30,
                 text=f'φ = {roll:+.2f}°  |  C_FL = {camber_fl:.2f}°  |  t = {t[i]:.1f}s',
                 showarrow=False, font=dict(color=_AMBER, size=11, family='Courier New')),
        ]
        frames.append(go.Frame(
            data=[_dict_to_trace(d) for d in td],
            layout=go.Layout(annotations=annots),
            name=str(i),
        ))

    fig.frames = frames

    # ── Layout & controls ─────────────────────────────────────────────────────
    tw = P.tw_f if view == 'front' else P.wb
    x_range = [-tw / 2 - 0.22, tw / 2 + 0.22] if view == 'front' else [-0.32, tw + 0.32]
    y_range = [-0.10, P.h_cg + 0.40]

    ttl = ('FRONT AXLE — SUSPENSION KINEMATICS  (ANIMATED)'
           if view == 'front' else
           'SIDE VIEW — PITCH & ANTI GEOMETRY  (ANIMATED)')

    lay = dict(**_BASE_LAY)
    lay.update(dict(
        height=490,
        title=dict(text=ttl, font=dict(color=_CYAN, size=12, family='Courier New'), x=0.0),
        showlegend=False,
        xaxis=dict(**_BASE_LAY['xaxis'],
                   range=x_range, showticklabels=False, zeroline=False,
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(**_BASE_LAY['yaxis'],
                   range=y_range, showticklabels=False, zeroline=False),
        updatemenus=[dict(
            type='buttons', showactive=False,
            bgcolor=_BG_CARD, bordercolor=_BORDER_M,
            font=dict(color=_CYAN, size=11, family='Courier New'),
            x=0.0, y=1.10, xanchor='left',
            buttons=[
                dict(label='▶  PLAY',
                     method='animate',
                     args=[None, dict(frame=dict(duration=35, redraw=True),
                                      fromcurrent=True, mode='immediate')]),
                dict(label='⏸  PAUSE',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')]),
            ],
        )],
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[str(i)], dict(mode='immediate',
                                      frame=dict(duration=0, redraw=True))],
                label=f'{t[i]:.1f}s',
            ) for i in idx],
            x=0.0, y=0.0, len=1.0,
            bgcolor=_BG_SURFACE,
            font=dict(color=_TEXT_M, size=9, family='Courier New'),
            currentvalue=dict(prefix='t = ', suffix=' s',
                               font=dict(color=_CYAN, size=11, family='Courier New')),
            pad=dict(t=32),
            tickcolor=_BORDER_M,
        )],
    ))
    fig.update_layout(**lay)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# MAIN STREAMLIT RENDER CLASS
# Drop-in: from visualization.suspension_viz import SuspensionVisualizer
#          SuspensionVisualizer().render()
# ═════════════════════════════════════════════════════════════════════════════

class SuspensionVisualizer:

    def __init__(self):
        self._phys = SuspensionPhysics()

    def render(self):
        P = self._phys

        st.title("Suspension Dynamics")
        st.markdown(
            """
            <p style="color:#9BA3BC;font-size:12px;font-family:'Courier New';">
            Quasi-static kinematics ...
            </p>
            """,
            unsafe_allow_html=True,
        )
 
        # ── 2D / 3D toggle ───────────────────────────────────────
        view_mode = st.radio(
            'VISUALIZATION MODE',
            ['2D  ·  Plotly Schematic', '3D  ·  Interactive Three.js'],
            horizontal=True,
            help='3D mode: orbit (drag), zoom (scroll), pan (shift+drag)',
        )
        if '3D' in view_mode:
            from visualization.suspension_3d_embed import render_3d_suspension
            render_3d_suspension(height=700)
            # Still show the 2D charts below for telemetry data
            st.markdown('<div class="section-label">2D TELEMETRY CHARTS</div>',
                        unsafe_allow_html=True)
 
        # ── Controls ──────
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        with c1:
            maneuver = st.selectbox(
                'MANEUVER PROFILE',
                ['combined_lap', 'skidpad', 'brake_straight', 'acceleration', 'chicane'],
                format_func=lambda x: {
                    'combined_lap':   '⚡  Combined Lap Segment',
                    'skidpad':        '⭕  Skidpad  1.5G Lateral',
                    'brake_straight': '🛑  Braking Straight  2.5G',
                    'acceleration':   '🚀  Acceleration  1.8G',
                    'chicane':        '〽  Chicane  Double-Apex',
                }[x],
            )
        with c2:
            view = st.radio('SCHEMATIC VIEW', ['front', 'side'],
                             format_func=lambda x: {'front': '⬛ Front Axle',
                                                     'side':  '↔ Side View'}[x],
                             horizontal=True)
        with c3:
            damper_corner = st.selectbox(
                'DAMPER DETAIL',
                ['fl', 'fr', 'rl', 'rr'],
                format_func=lambda x: {'fl': 'Front Left', 'fr': 'Front Right',
                                        'rl': 'Rear Left',  'rr': 'Rear Right'}[x],
            )
        with c4:
            frame_step = st.select_slider(
                'ANIMATION STEP',
                options=[1, 2, 4, 8, 16],
                value=4,
                format_func=lambda x: f'×{x}',
            )

        # ── Cache simulation result ───────────────────────────────────────────
        sim_key = f'susp_sim_{maneuver}'
        if sim_key not in st.session_state:
            with st.spinner('Running suspension simulation…'):
                st.session_state[sim_key] = P.simulate(maneuver=maneuver)
        sim = st.session_state[sim_key]

        # ── KPI ribbon (computed from params, not simulation) ─────────────────
        nat_f   = np.sqrt(P.wr_f / (P.m_s / 2)) / (2 * np.pi)
        nat_r   = np.sqrt(P.wr_r / (P.m_s / 2)) / (2 * np.pi)
        cc_f    = 2 * np.sqrt(P.wr_f * (P.m_s / 2))
        cc_r    = 2 * np.sqrt(P.wr_r * (P.m_s / 2))
        dr_f    = P.c_lo_f / cc_f
        dr_r    = P.c_lo_r / cc_r
        rg      = np.degrees(P.m_s * (P.h_cg - P.h_rc_f) * P.g / P.K_phi)
        arb_pct = P.K_phi_f / P.K_phi * 100

        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        with k1: st.metric('Nat Freq F',   f'{nat_f:.2f} Hz')
        with k2: st.metric('Nat Freq R',   f'{nat_r:.2f} Hz')
        with k3: st.metric('Damp Ratio F', f'{dr_f:.3f}')
        with k4: st.metric('Damp Ratio R', f'{dr_r:.3f}')
        with k5: st.metric('Roll Grad',    f'{rg:.2f} °/G')
        with k6: st.metric('ARB Split F',  f'{arb_pct:.1f}%')
        with k7: st.metric('Z_eq',         f'{P.Z_eq*1000:.1f} mm')

        st.markdown('---')

        # ── Section 1: Animated Schematic ─────────────────────────────────────
        st.markdown('<div class="section-label">ANIMATED SUSPENSION SCHEMATIC</div>',
                    unsafe_allow_html=True)
        anim_key = f'susp_anim_{maneuver}_{view}_{frame_step}'
        if anim_key not in st.session_state:
            with st.spinner('Building animation frames…'):
                st.session_state[anim_key] = build_animated_schematic(
                    sim, P, view=view, frame_step=frame_step)
        st.plotly_chart(st.session_state[anim_key], use_container_width=True)

        # ── Section 2: Forces & Travel ────────────────────────────────────────
        st.markdown('<div class="section-label">FORCE & TRAVEL TIME SERIES</div>',
                    unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(chart_corner_loads(sim), use_container_width=True)
        with cb:
            st.plotly_chart(chart_wheel_travel(sim), use_container_width=True)

        # ── Section 3: Damper & Alignment ─────────────────────────────────────
        st.markdown('<div class="section-label">DAMPER DYNAMICS & ALIGNMENT</div>',
                    unsafe_allow_html=True)
        cc, cd = st.columns(2)
        with cc:
            st.plotly_chart(chart_damper_fv(sim, P, corner=damper_corner),
                            use_container_width=True)
        with cd:
            st.plotly_chart(chart_camber_time(sim), use_container_width=True)

        # ── Section 4: Kinematic Maps & Coupling ──────────────────────────────
        st.markdown('<div class="section-label">KINEMATIC MAPS & COUPLING</div>',
                    unsafe_allow_html=True)
        ce, cf_, cg = st.columns(3)
        with ce:
            st.plotly_chart(chart_camber_map(P), use_container_width=True)
        with cf_:
            st.plotly_chart(chart_arb_coupling(sim), use_container_width=True)
        with cg:
            st.plotly_chart(chart_roll_gradient(sim, P), use_container_width=True)

        # ── Section 5: g-g diagram ────────────────────────────────────────────
        st.markdown('<div class="section-label">g-g ENVELOPE & LOAD SENSITIVITY</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_g_g(sim), use_container_width=True)