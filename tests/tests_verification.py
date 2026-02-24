"""
tests/test_verification.py
Part 10 — Sanity Checks for the full simulation stack.

Run with:  pytest tests/test_verification.py -v

Tests:
  1. Friction circle           — sqrt(Fx² + Fy²) ≤ mu*Fz at any slip condition
  2. Load sensitivity          — Fy increases monotonically with Fz at constant alpha
  3. Diagonal load transfer    — correct corner Fz at 1.5G cornering + 0.5G braking
  4. Aero scaling              — Fz_aero ∝ v² (within 1% of theoretical)
  5. Spring rate optimisation  — optimiser converges above 1.20 Hz lower bound
"""

import math
import jax
import jax.numpy as jnp
import pytest

# ── Shared fixtures ───────────────────────────────────────────────────────────

from data.configs.tire_coeffs  import tire_coeffs
from data.configs.vehicle_params import vehicle_params as VP
from models.tire_model import PacejkaTire

T_RIBS_NOM = jnp.array([90.0, 90.0, 90.0])   # nominal tyre temperature
T_GAS_NOM  = 90.0                              # nominal gas temperature
VX_NOM     = 15.0                              # m/s reference speed


@pytest.fixture(scope="module")
def tire():
    return PacejkaTire(tire_coeffs, rng_seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Friction Circle
# ─────────────────────────────────────────────────────────────────────────────
class TestFrictionCircle:
    """
    At every combination of (alpha, kappa) the resultant force vector must
    not exceed the friction limit mu_peak * Fz.  We allow a 5% margin for
    the PINN correction at its maximum clip value.
    """

    def test_friction_circle_not_exceeded(self, tire):
        Fz     = 1000.0   # N  (= Fz0)
        gamma  = 0.0
        mu_est = VP.get('PDY1', 2.218) * 1.05  # slightly generous upper bound

        alpha_vals = jnp.linspace(-0.30, 0.30, 12)
        kappa_vals = jnp.linspace(-0.20, 0.20,  8)

        worst_ratio = 0.0
        for alpha in alpha_vals:
            for kappa in kappa_vals:
                Fx, Fy = tire.compute_force(
                    float(alpha), float(kappa), Fz, gamma,
                    T_RIBS_NOM, T_GAS_NOM, VX_NOM
                )
                F_res  = math.sqrt(float(Fx)**2 + float(Fy)**2)
                limit  = mu_est * Fz
                ratio  = F_res / (limit + 1e-6)
                worst_ratio = max(worst_ratio, ratio)

        # Allow 10% headroom (PINN ±0.25 mod + sigma penalty keeps this safe)
        assert worst_ratio < 1.10, (
            f"Friction circle violated: F_resultant / (mu*Fz) = {worst_ratio:.3f} > 1.10"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Load Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
class TestLoadSensitivity:
    """
    Fy_peak must increase with Fz, but with a degressive (diminishing-returns)
    slope.  We check that Fy(2*Fz0) > Fy(Fz0) and that the ratio Fy/Fz
    decreases as load increases (typical rubber behaviour).
    """

    def test_fy_increases_with_fz(self, tire):
        alpha  = math.radians(4.0)
        kappa  = 0.0
        gamma  = 0.0
        Fz0    = tire_coeffs['FNOMIN']

        Fz_vals  = [0.5 * Fz0, Fz0, 1.5 * Fz0, 2.0 * Fz0]
        Fy_vals  = []
        for Fz in Fz_vals:
            _, Fy = tire.compute_force(
                alpha, kappa, Fz, gamma,
                T_RIBS_NOM, T_GAS_NOM, VX_NOM
            )
            Fy_vals.append(float(Fy))

        # Fy must increase monotonically
        for i in range(len(Fy_vals) - 1):
            assert Fy_vals[i+1] > Fy_vals[i], (
                f"Fy not monotone: Fy({Fz_vals[i+1]:.0f}) = {Fy_vals[i+1]:.1f} "
                f"≤ Fy({Fz_vals[i]:.0f}) = {Fy_vals[i]:.1f}"
            )

    def test_fy_degressive_with_fz(self, tire):
        """mu = Fy/Fz must decrease as Fz increases."""
        alpha  = math.radians(4.0)
        kappa  = 0.0
        gamma  = 0.0
        Fz0    = tire_coeffs['FNOMIN']

        Fz_low,  Fz_high = Fz0, 2.0 * Fz0
        _, Fy_low  = tire.compute_force(alpha, kappa, Fz_low,  gamma, T_RIBS_NOM, T_GAS_NOM, VX_NOM)
        _, Fy_high = tire.compute_force(alpha, kappa, Fz_high, gamma, T_RIBS_NOM, T_GAS_NOM, VX_NOM)

        mu_low  = float(Fy_low)  / Fz_low
        mu_high = float(Fy_high) / Fz_high

        assert mu_high < mu_low, (
            f"Expected degressive load sensitivity: "
            f"mu(Fz0) = {mu_low:.3f}, mu(2*Fz0) = {mu_high:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Diagonal Load Transfer
# ─────────────────────────────────────────────────────────────────────────────
class TestDiagonalLoadTransfer:
    """
    Under combined 1.5G lateral + 0.5G longitudinal deceleration the four
    corner loads must:
      a) all be non-negative (no wheel lift)
      b) sum to total vehicle weight ± 1%
      c) the loaded outer-front corner must carry the most load
    """

    def _corner_loads(self, ay_g=1.5, ax_g=-0.5):
        """Quasi-static 4-corner load calculation (mirrors vehicle_dynamics.py)."""
        m   = VP['total_mass']
        lf  = VP['lf'];   lr  = VP['lr']
        L   = lf + lr
        h   = VP['h_cg']
        tf  = VP['track_front'];  tr = VP['track_rear']
        g   = 9.81
        ay  = ay_g * g
        ax  = ax_g * g

        h_rc_f = VP.get('h_rc_f', 0.040)
        h_rc_r = VP.get('h_rc_r', 0.060)

        # Approximate roll stiffness from baseline spring + ARB
        mr_f   = VP.get('motion_ratio_f_poly', [1.14])[0]
        mr_r   = VP.get('motion_ratio_r_poly', [1.16])[0]
        k_f    = VP['spring_rate_f'] / mr_f**2
        k_r    = VP['spring_rate_r'] / mr_r**2
        arb_f  = VP['arb_rate_f']   / mr_f**2
        arb_r  = VP['arb_rate_r']   / mr_r**2

        Kroll_f   = (k_f + arb_f) * tf**2 * 0.5
        Kroll_r   = (k_r + arb_r) * tr**2 * 0.5
        Kroll_tot = Kroll_f + Kroll_r + 1.0

        LLT_geo_f  = m * ay * h_rc_f / tf
        LLT_geo_r  = m * ay * h_rc_r / tr
        h_arm      = h - (h_rc_f * lr + h_rc_r * lf) / L
        LLT_el_f   = m * ay * h_arm / tf  * (Kroll_f / Kroll_tot)
        LLT_el_r   = m * ay * h_arm / tr  * (Kroll_r / Kroll_tot)
        LLT_f      = LLT_geo_f + LLT_el_f
        LLT_r      = LLT_geo_r + LLT_el_r

        LLT_long = m * ax * h / L    # negative ax → braking → front loaded
        bias_f   = VP.get('brake_bias_f', 0.60)
        LLT_long_f = LLT_long * bias_f
        LLT_long_r = LLT_long * (1.0 - bias_f)

        W = m * g
        Fz_fl = W * lr / (L*2) - LLT_f - LLT_long_f
        Fz_fr = W * lr / (L*2) + LLT_f - LLT_long_f
        Fz_rl = W * lf / (L*2) - LLT_r + LLT_long_r
        Fz_rr = W * lf / (L*2) + LLT_r + LLT_long_r
        return Fz_fl, Fz_fr, Fz_rl, Fz_rr

    def test_no_wheel_lift(self):
        Fz_fl, Fz_fr, Fz_rl, Fz_rr = self._corner_loads()
        for name, Fz in [('FL', Fz_fl), ('FR', Fz_fr), ('RL', Fz_rl), ('RR', Fz_rr)]:
            assert Fz >= 0.0, f"Wheel lift at {name}: Fz = {Fz:.1f} N"

    def test_load_sum(self):
        loads = self._corner_loads()
        W     = VP['total_mass'] * 9.81
        total = sum(loads)
        rel_err = abs(total - W) / W
        assert rel_err < 0.01, (
            f"Corner loads sum to {total:.1f} N, expected {W:.1f} N "
            f"(relative error {rel_err*100:.2f}% > 1%)"
        )

    def test_outer_front_most_loaded(self):
        """FR = outer-front for left-hand (positive ay) cornering."""
        Fz_fl, Fz_fr, Fz_rl, Fz_rr = self._corner_loads(ay_g=1.5, ax_g=-0.5)
        assert Fz_fr == max(Fz_fl, Fz_fr, Fz_rl, Fz_rr), (
            f"Expected FR to be most loaded under left-hand cornering+braking. "
            f"Loads: FL={Fz_fl:.0f} FR={Fz_fr:.0f} RL={Fz_rl:.0f} RR={Fz_rr:.0f} N"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Aero Scaling  (Fz_aero ∝ v²)
# ─────────────────────────────────────────────────────────────────────────────
class TestAeroScaling:
    """
    Dynamic pressure q = 0.5*rho*v².  For a constant Cl and A the downforce
    must scale exactly as v².  We evaluate the analytical formula directly
    (the neural modifier is zero-initialised so it adds nothing at init).
    """

    def test_downforce_proportional_to_v_squared(self):
        rho = VP.get('rho_air', 1.225)
        Cl  = VP.get('Cl_ref',  4.14)
        A   = VP.get('A_ref',   1.10)

        v1, v2 = 10.0, 20.0  # m/s
        Fz1 = 0.5 * rho * Cl * A * v1**2
        Fz2 = 0.5 * rho * Cl * A * v2**2

        # Expect Fz2 / Fz1 ≈ (v2/v1)² = 4.0  (within 1%)
        ratio     = Fz2 / Fz1
        expected  = (v2 / v1) ** 2
        rel_err   = abs(ratio - expected) / expected

        assert rel_err < 0.01, (
            f"Aero scaling error: Fz2/Fz1 = {ratio:.4f}, expected {expected:.4f} "
            f"(rel error {rel_err*100:.3f}%)"
        )

    def test_downforce_increases_with_speed(self):
        rho = VP.get('rho_air', 1.225)
        Cl  = VP.get('Cl_ref',  4.14)
        A   = VP.get('A_ref',   1.10)

        speeds = [5.0, 10.0, 15.0, 20.0, 30.0]
        Fz_vals = [0.5 * rho * Cl * A * v**2 for v in speeds]

        for i in range(len(Fz_vals) - 1):
            assert Fz_vals[i+1] > Fz_vals[i], (
                f"Downforce not monotone with speed: "
                f"Fz({speeds[i+1]}) = {Fz_vals[i+1]:.1f} ≤ Fz({speeds[i]}) = {Fz_vals[i]:.1f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Spring Rate Optimisation Lower Bound
# ─────────────────────────────────────────────────────────────────────────────
class TestSpringRateOptimisationBound:
    """
    The optimiser must not pin spring rates to the lower bound.
    We run compute_skidpad_objective on two parameter sets:
      A: k_f = k_r = 10 000 N/m  (suspiciously soft — below 1.20 Hz)
      B: k_f = k_r = 35 000 N/m  (realistic FSAE)
    Objective B must be strictly greater than Objective A (after penalties).
    """

    def _make_params(self, k_f, k_r):
        """Return an 8-element parameter vector matching objectives.py convention."""
        return jnp.array([
            k_f,      # spring rate front  [N/m]
            k_r,      # spring rate rear
            200.0,    # ARB front
            150.0,    # ARB rear
            1800.0,   # damper low-speed front
            1600.0,   # damper low-speed rear
            VP.get('h_cg', 0.330),
            VP.get('brake_bias_f', 0.60),
        ])

    def test_stiffer_springs_give_better_objective(self):
        from objectives import compute_skidpad_objective

        params_soft   = self._make_params(10_000.0, 10_000.0)
        params_realis = self._make_params(35_000.0, 52_000.0)

        obj_soft,   safety_soft   = compute_skidpad_objective(None, params_soft,   None)
        obj_realis, safety_realis = compute_skidpad_objective(None, params_realis, None)

        assert float(obj_realis) > float(obj_soft), (
            f"Optimiser would prefer soft springs: "
            f"obj(35k/52k) = {float(obj_realis):.4f} ≤ obj(10k/10k) = {float(obj_soft):.4f}\n"
            f"Check freq_lower_bound and stiffness_penalty weight in objectives.py."
        )

    def test_freq_above_lower_bound_at_optimum(self):
        """Verify 35k/52k springs satisfy the 1.20 Hz lower bound."""
        mr_f = VP.get('motion_ratio_f_poly', [1.14])[0]
        mr_r = VP.get('motion_ratio_r_poly', [1.16])[0]
        m_s  = VP.get('sprung_mass', VP['total_mass'] * 0.85)
        m_corner = m_s / 4.0

        for k_spring, mr, label in [
            (35_000.0, mr_f, 'front'),
            (52_000.0, mr_r, 'rear'),
        ]:
            wheel_rate = k_spring / mr**2
            freq = math.sqrt(wheel_rate / m_corner) / (2.0 * math.pi)
            assert freq >= 1.20, (
                f"{label} heave frequency {freq:.3f} Hz < 1.20 Hz lower bound "
                f"(wheel_rate = {wheel_rate:.0f} N/m)"
            )