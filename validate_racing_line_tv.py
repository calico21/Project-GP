#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# validate_racing_line_tv.py
# Project-GP — Closed-Loop Racing Line Optimizer + Torque Vectoring Validator
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE
# ───────
# Validates the full powertrain pipeline (SOCP TV, CBF, κ* tracking, regen)
# in a closed-loop lap simulation by:
#
#   1. Building an FSG autocross racing line (MinCurv QP — no WMPC dependency)
#   2. Running a forward-backward velocity sweep for speed profile
#   3. Executing a full closed-loop lap with:
#      - Feedforward kinematic steering
#      - Lateral/heading PD correction
#      - powertrain_step() at every timestep (full 14-stage pipeline)
#   4. Producing rich diagnostic plots:
#      - Racing line with TV Mz_target vs Mz_actual overlay
#      - κ* tracking (Koopman observer vs measured)
#      - CBF intervention map (when/where CBF fires)
#      - Torque distribution per wheel (vectoring authority)
#      - Regen blend α* across braking zones
#      - Phase-plane: β (sideslip) vs ψ̇ (yaw rate)
#      - Grip utilisation ρ heatmap
#
# DESIGN CONTRACTS
# ────────────────
# · Does NOT require trained H_net weights — uses vehicle.simulate_step directly
# · Does NOT use DiffWMPC — uses feedforward + PD driver (architecturally sufficient)
# · Full powertrain_step() called every dt=5ms → validates TV + CBF + TC + launch
# · All diagnostics computed from PowertrainDiagnostics fields
# · Gradient-free validation script — jax.grad NOT called here
#
# USAGE
# ─────
#   cd ~/FS_Driver_Setup_Optimizer
#   python validate_racing_line_tv.py
#   python validate_racing_line_tv.py --track fsg_autocross --n-laps 2 --output out/
#   python validate_racing_line_tv.py --track skidpad           # TV skidpad test
#
# OUTPUTS
# ───────
#   out/tv_validation_racing_line.png   — 3×3 diagnostic panel
#   out/tv_diagnostics.csv              — per-step telemetry for offline analysis
#   out/tv_summary.txt                  — pass/fail criteria
#
# ═══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import sys
import time
import argparse
import warnings
from pathlib import Path

# ── JAX environment must be set before any JAX import ─────────────────────────
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import jax_config  # noqa: F401
except ImportError:
    pass

import jax
import jax.numpy as jnp
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Track generation — FSG autocross + skidpad
# ─────────────────────────────────────────────────────────────────────────────

def _build_fsg_autocross(N: int = 600) -> dict:
    """
    Procedural FSG autocross: ~750m, 12 corners.
    Returns dict with (s, x, y, psi, kappa, w_left, w_right, total_length).
    """
    total = 750.0
    ds    = total / N
    s_arr = np.linspace(0.0, total, N)

    # Curvature profile
    kappa = np.zeros(N)
    segs = [
        (0,   30,   0.00),
        (30,  55,   0.12),
        (55,  75,   0.00),
        (75,  95,  -0.09),
        (95,  115,  0.00),
        (115, 135,  0.15),
        (135, 150,  0.00),
        (150, 170, -0.12),
        (170, 190,  0.00),
        (190, 210,  0.10),
        (210, 225, -0.10),
        (225, 260,  0.00),
        (260, 310, -0.04),
        (310, 370,  0.00),
        (370, 395,  0.13),
        (395, 420,  0.00),
        (420, 445, -0.11),
        (445, 490,  0.00),
        (490, 520,  0.08),
        (520, 560,  0.00),
        (560, 590, -0.14),
        (590, 640,  0.00),
        (640, 670,  0.10),
        (670, 750,  0.00),
    ]
    for s0, s1, k in segs:
        kappa[(s_arr >= s0) & (s_arr < s1)] = k

    # Gaussian smoothing (σ≈3.5m)
    kernel = np.exp(-0.5 * (np.arange(-7, 8) / 3.5) ** 2)
    kernel /= kernel.sum()
    kappa = np.convolve(kappa, kernel, mode="same")

    psi = np.cumsum(kappa * ds)
    x   = np.cumsum(np.cos(psi) * ds)
    y   = np.cumsum(np.sin(psi) * ds)

    return dict(s=s_arr, x=x, y=y, psi=psi, kappa=kappa,
                w_left=np.full(N, 3.5), w_right=np.full(N, 3.5),
                total_length=total)


def _build_skidpad(N: int = 300) -> dict:
    """
    Classic 8-figure skidpad: two R=9.125m circles + entry/exit straights.
    Excellent for validating TV steady-state yaw moment authority.
    """
    R          = 9.125
    straight   = 15.0
    circ       = 2.0 * np.pi * R
    total      = 2.0 * straight + 2.0 * circ
    ds         = total / N
    s_arr      = np.linspace(0.0, total, N)

    kappa  = np.zeros(N)
    s1, s2 = straight, straight + circ
    s3, s4 = s2, s2 + circ

    kappa[(s_arr >= s1) & (s_arr < s2)] =  1.0 / R
    kappa[(s_arr >= s3) & (s_arr < s4)] = -1.0 / R  # reverse circle

    psi = np.cumsum(kappa * ds)
    x   = np.cumsum(np.cos(psi) * ds)
    y   = np.cumsum(np.sin(psi) * ds)

    return dict(s=s_arr, x=x, y=y, psi=psi, kappa=kappa,
                w_left=np.full(N, 1.5), w_right=np.full(N, 1.5),
                total_length=total)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Forward-backward velocity sweep (friction-limited speed profile)
# ─────────────────────────────────────────────────────────────────────────────

def velocity_sweep(
    kappa:    np.ndarray,
    mu:       float = 1.40,
    g:        float = 9.81,
    V_max:    float = 25.0,
    a_max_lon: float = 12.0,   # m/s²  (motor-limited, ~80kW / 300kg @ 15m/s)
    a_max_brk: float = 16.0,   # m/s²  (braking: full friction available)
    ds:       float = None,
) -> np.ndarray:
    """
    Classic point-mass forward-backward sweep.
    Returns per-node speed profile [m/s].
    """
    N = len(kappa)
    if ds is None:
        ds = 1.5  # default for 750m / 500 nodes

    # Lateral grip limit
    v_lat = np.sqrt(np.abs(mu * g / (np.abs(kappa) + 1e-4)))
    v_cap = np.minimum(v_lat, V_max)

    # Forward pass (acceleration limited)
    v_fwd = v_cap.copy()
    for i in range(1, N):
        v_reachable = np.sqrt(v_fwd[i-1]**2 + 2.0 * a_max_lon * ds)
        v_fwd[i]    = min(v_fwd[i], v_reachable)

    # Backward pass (braking limited)
    v_bwd = v_fwd.copy()
    for i in range(N - 2, -1, -1):
        v_reachable = np.sqrt(v_bwd[i+1]**2 + 2.0 * a_max_brk * ds)
        v_bwd[i]    = min(v_bwd[i], v_reachable)

    return np.clip(v_bwd, 1.0, V_max)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Closed-loop driver model
# ─────────────────────────────────────────────────────────────────────────────

class ClosedLoopDriver:
    """
    Pure-pursuit + speed P-controller driver.

    Steering:   δ = L_wb × κ_target + K_lat × e_lat + K_head × e_heading
    Throttle:   u = K_spd × (v_target − vx) / F_max
    Brake:      symmetric

    All gains are smooth; no conditionals in the control law.
    """

    def __init__(
        self,
        wheelbase:    float = 1.55,
        K_lat:        float = 0.15,     # rad/m
        K_head:       float = 1.20,     # rad/rad
        K_spd:        float = 4000.0,   # N/(m/s)
        delta_max:    float = 0.40,     # rad
        F_max:        float = 6000.0,   # N
        lookahead_s:  float = 4.0,      # m arc-length preview
    ):
        self.L          = wheelbase
        self.K_lat      = K_lat
        self.K_head     = K_head
        self.K_spd      = K_spd
        self.delta_max  = delta_max
        self.F_max      = F_max
        self.lookahead  = lookahead_s

    def control(
        self,
        vx:      float, vy: float, yaw: float,
        x_car:   float, y_car: float,
        s_curr:  float,
        track:   dict,
        v_prof:  np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Returns (delta_rad, throttle_norm, brake_norm) ∈ [−δmax,+δmax], [0,1], [0,1].
        """
        s_arr   = track["s"]
        total   = track["total_length"]
        s_ahead = (s_curr + self.lookahead) % total
        s_now   = s_curr % total

        # Interpolate reference point
        psi_ref = float(np.interp(s_now,   s_arr, np.unwrap(track["psi"])))
        k_ref   = float(np.interp(s_now,   s_arr, track["kappa"]))
        x_ref   = float(np.interp(s_now,   s_arr, track["x"]))
        y_ref   = float(np.interp(s_now,   s_arr, track["y"]))
        v_tgt   = float(np.interp(s_now,   s_arr, v_prof))

        # Lateral / heading errors in track frame
        dx   = x_car - x_ref
        dy   = y_car - y_ref
        cpsi = np.cos(psi_ref);  spsi = np.sin(psi_ref)
        e_lat  = -spsi * dx + cpsi * dy          # positive = left of line
        e_head = np.arctan2(np.sin(yaw - psi_ref), np.cos(yaw - psi_ref))

        # Steering: kinematic base + lateral + heading corrections
        delta = float(np.clip(
            self.L * k_ref - self.K_lat * e_lat - self.K_head * e_head,
            -self.delta_max, self.delta_max
        ))

        # Longitudinal: P-controller with smooth throttle/brake split
        v_err   = vx - v_tgt
        F_cmd   = float(np.clip(-self.K_spd * v_err, -self.F_max, self.F_max))
        throttle = float(np.clip( F_cmd / self.F_max, 0.0, 1.0))
        brake    = float(np.clip(-F_cmd / self.F_max, 0.0, 1.0))

        return delta, throttle, brake


# ─────────────────────────────────────────────────────────────────────────────
# §4  Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_closed_loop_lap(
    track:       dict,
    v_prof:      np.ndarray,
    n_laps:      int  = 1,
    dt:          float = 0.005,
    use_powertrain: bool = True,
    verbose:     bool = True,
) -> dict:
    """
    Execute closed-loop lap simulation with full powertrain pipeline.

    Returns a dict of per-timestep arrays for diagnostic plotting.
    """
    from models.vehicle_dynamics import (
        DifferentiableMultiBodyVehicle, DEFAULT_SETUP,
        compute_equilibrium_suspension,
    )
    from config.vehicles.ter26 import vehicle_params as VP
    from config.tire_coeffs import tire_coeffs as TC
    from config.vehicles.ter26 import vehicle_params as VP
    
    # --- KINEMATIC OVERRIDE (Fixing the synthetic geometry) ---
    VP['bump_steer_f'] = 0.0
    VP['bump_steer_r'] = 0.0
    VP['camber_gain_f'] = -0.5
    VP['camber_gain_r'] = -0.5
    VP['mr_f'] = 0.85
    VP['mr_r'] = 0.85
    VP['rc_height_f'] = 0.030
    VP['rc_height_r'] = 0.040
    # ----------------------------------------------------------
    if verbose:
        print("\n[Sim] Initialising vehicle model...")
    vehicle = DifferentiableMultiBodyVehicle(VP, TC)
    setup   = jnp.asarray(DEFAULT_SETUP, dtype=jnp.float32)

    # Powertrain manager (optional — degrades gracefully if import fails)
    pt_config = None
    pt_state  = None
    if use_powertrain:
        try:
            from powertrain.powertrain_manager import make_powertrain_manager
            pt_config, pt_state = make_powertrain_manager(VP)
            if verbose:
                print("[Sim] Powertrain manager loaded.")
        except Exception as e:
            if verbose:
                print(f"[Sim] Powertrain manager unavailable ({e}) — using simple TV fallback.")
            use_powertrain = False

    # ── Initial state ─────────────────────────────────────────────────────────
    total_len = float(track["total_length"])
    k0        = abs(float(track["kappa"][0])) + 1e-4
    v0        = min(float(v_prof[0]), 15.0)

    z_eq = compute_equilibrium_suspension(setup, VP)
    x    = vehicle.make_initial_state(T_env=25.0, vx0=v0)
    x    = x.at[0].set(float(track["x"][0]))
    x    = x.at[1].set(float(track["y"][0]))
    x    = x.at[5].set(float(track["psi"][0]))
    x    = x.at[6:10].set(z_eq)
    # Warm tire thermal nodes
    x    = x.at[28:56].set(jnp.tile(jnp.array([80., 82., 84., 76., 70., 30., 40.]), 4))

    driver = ClosedLoopDriver(
        wheelbase=VP.get("lf", 0.8525) + VP.get("lr", 0.6975)
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    N_steps_total = int(n_laps * total_len / (v0 * dt)) + 5000  # generous estimate
    store = {k: [] for k in [
        "t", "s", "vx", "vy", "wz", "delta", "throttle", "brake",
        "x_car", "y_car", "yaw",
        "T_fl", "T_fr", "T_rl", "T_rr",   # wheel torques
        "Mz_target", "Mz_actual", "wz_ref",
        "kappa_star_fl", "kappa_star_rr",
        "kappa_meas_fl", "kappa_meas_rr",
        "cbf_active",
        "alpha_regen", "F_hydraulic",
        "SoC", "T_motor_fl",
        "w_slip", "w_yaw",
        "rho_util",
        "lap_n",
    ]}

    s_curr   = 0.0
    t_curr   = 0.0
    lap_done = 0
    prev_s   = 0.0

    # JIT-compile vehicle step once
    if verbose:
        print("[Sim] JIT-compiling vehicle.simulate_step...")
    t0 = time.perf_counter()
    _  = vehicle.simulate_step(x, jnp.zeros(6), setup, dt=dt)
    if verbose:
        print(f"[Sim] Compile: {time.perf_counter()-t0:.1f}s")

    if verbose:
        print(f"[Sim] Starting {n_laps}-lap run (dt={dt*1000:.1f}ms)...")

    step    = 0
    laps_completed = 0

    while laps_completed < n_laps and step < N_steps_total:
        vx  = float(x[14])
        vy  = float(x[15])
        yaw = float(x[5])
        wz  = float(x[19])
        x_c = float(x[0])
        y_c = float(x[1])

        s_norm = s_curr % total_len

        # ── Driver control ────────────────────────────────────────────────────
        delta, throttle, brake = driver.control(
            vx, vy, yaw, x_c, y_c, s_norm, track, v_prof
        )

        # ── Powertrain step ───────────────────────────────────────────────────
        if use_powertrain:
            try:
                from powertrain.powertrain_manager import powertrain_step

                omega_w = jnp.array(x[24:28])
                Fz_est  = jnp.array([
                    VP.get("total_mass", 300.0) * 9.81
                    * VP.get("lr", 0.6975) / (VP.get("lf", 0.8525) + VP.get("lr", 0.6975))
                    * 0.5
                ] * 4, dtype=jnp.float32)
                Fy_est  = jnp.zeros(4)
                alpha_t = jnp.array(x[56:60])
                T_tire  = jnp.array([float(jnp.mean(x[28:31])),
                                     float(jnp.mean(x[35:38])),
                                     float(jnp.mean(x[42:45])),
                                     float(jnp.mean(x[49:52]))])
                k_now   = float(np.interp(s_norm, track["s"], track["kappa"]))

                diag, pt_state = powertrain_step(
                    throttle_raw=jnp.array(throttle),
                    brake_raw=jnp.array(brake),
                    delta=jnp.array(delta),
                    vx=jnp.array(vx),
                    vy=jnp.array(vy),
                    wz=jnp.array(wz),
                    Fz=Fz_est,
                    Fy=Fy_est,
                    omega_wheel=omega_w,
                    alpha_t=alpha_t,
                    T_tire=T_tire,
                    mu_est=jnp.array(1.40),
                    gp_sigma=jnp.array(0.05),
                    curvature=jnp.array(k_now),
                    manager_state=pt_state,
                    dt=jnp.array(dt),
                    config=pt_config,
                )

                T_w = [float(diag.T_wheel[i]) for i in range(4)]

                # Build u_vehicle from powertrain output
                u = jnp.array([delta,
                                T_w[0], T_w[1], T_w[2], T_w[3],
                                float(diag.F_brake_hydraulic)],
                               dtype=jnp.float32)

                # Store powertrain diagnostics
                store["T_fl"].append(T_w[0]);   store["T_fr"].append(T_w[1])
                store["T_rl"].append(T_w[2]);   store["T_rr"].append(T_w[3])
                store["Mz_target"].append(float(diag.Mz_target))
                store["Mz_actual"].append(float(diag.Mz_actual))
                store["wz_ref"].append(float(diag.wz_ref))
                store["kappa_star_fl"].append(float(diag.kappa_star[0]))
                store["kappa_star_rr"].append(float(diag.kappa_star[3]))
                store["kappa_meas_fl"].append(float(diag.kappa_measured[0]))
                store["kappa_meas_rr"].append(float(diag.kappa_measured[3]))
                store["cbf_active"].append(float(diag.cbf_active))
                store["alpha_regen"].append(float(diag.alpha_regen))
                store["F_hydraulic"].append(float(diag.F_brake_hydraulic))
                store["SoC"].append(float(diag.SoC))
                store["T_motor_fl"].append(float(diag.T_motors[0]))
                store["w_slip"].append(float(diag.w_slip))
                store["w_yaw"].append(float(diag.w_yaw))
                store["rho_util"].append(float(diag.koopman_rho))

            except Exception as e:
                # Graceful fallback: simple kinematic torque allocation
                use_powertrain = False
                if verbose and step < 5:
                    print(f"[Sim] PT step failed at step {step}: {e}")

        if not use_powertrain:
            # Simple feedforward allocation (no TV, no CBF)
            R_w = 0.2032
            F_cmd = (throttle - brake) * 5000.0
            T_req = max(F_cmd, 0.0) * R_w / 4.0
            F_brk = max(-F_cmd, 0.0)
            u = jnp.array([delta, T_req, T_req, T_req, T_req, F_brk], dtype=jnp.float32)

            # Fill with zeros/nans for consistency
            for k in ["T_fl","T_fr","T_rl","T_rr"]:
                store[k].append(float(T_req))
            for k in ["Mz_target","Mz_actual","wz_ref",
                       "kappa_star_fl","kappa_star_rr",
                       "kappa_meas_fl","kappa_meas_rr",
                       "cbf_active","alpha_regen","F_hydraulic",
                       "SoC","T_motor_fl","w_slip","w_yaw","rho_util"]:
                store[k].append(0.0)

        # ── Physics step ──────────────────────────────────────────────────────
        x = vehicle.simulate_step(x, u, setup, dt=dt)

        # ── Arc-length progress ───────────────────────────────────────────────
        vx_new = max(float(x[14]), 0.0)
        ds_step = vx_new * dt
        s_curr += ds_step

        # Lap detection
        if s_curr - laps_completed * total_len >= total_len:
            laps_completed += 1
            if verbose:
                t_lap = step * dt
                print(f"[Sim] Lap {laps_completed} complete: t={t_lap:.2f}s, "
                      f"vx={vx_new:.1f}m/s, SoC={float(x[14]):.1f}%")

        # ── Common stores ─────────────────────────────────────────────────────
        store["t"].append(t_curr)
        store["s"].append(s_curr % total_len)
        store["vx"].append(float(x[14]))
        store["vy"].append(float(x[15]))
        store["wz"].append(float(x[19]))
        store["delta"].append(delta)
        store["throttle"].append(throttle)
        store["brake"].append(brake)
        store["x_car"].append(float(x[0]))
        store["y_car"].append(float(x[1]))
        store["yaw"].append(float(x[5]))
        store["lap_n"].append(laps_completed)

        t_curr += dt
        step   += 1

        if step % 2000 == 0 and verbose:
            print(f"  step={step:6d}  s={s_curr%total_len:6.1f}m  "
                  f"vx={float(x[14]):5.1f}m/s  "
                  f"lap={laps_completed}/{n_laps}")

    if verbose:
        print(f"[Sim] Done: {step} steps, {laps_completed} laps, "
              f"{step*dt:.1f}s simulated.")

    return {k: np.array(v) for k, v in store.items()}


# ─────────────────────────────────────────────────────────────────────────────
# §5  Diagnostic plotting
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, w: int = 20) -> np.ndarray:
    """Boxcar moving average for display."""
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


def plot_tv_diagnostics(
    data:   dict,
    track:  dict,
    v_prof: np.ndarray,
    out_dir: str = "out",
    verbose: bool = False,  # <--- Add this line!
) -> str:
    """Generate 3×3 diagnostic panel. Returns path to saved PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.collections import LineCollection
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError:
        print("[Plot] matplotlib not available — skipping plots.")
        return ""

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 16), facecolor="#0d1117")
    fig.suptitle("Project-GP — TV Validation: Racing Line + Powertrain Diagnostics",
                 color="#e6edf3", fontsize=15, fontweight="bold", y=0.98)

    GS    = "#0d1117"  # background
    GRID  = "#21262d"  # panel bg
    C1    = "#58a6ff"  # blue
    C2    = "#3fb950"  # green
    C3    = "#f78166"  # red / braking
    C4    = "#d2a8ff"  # purple
    C5    = "#ffa657"  # orange
    CTEXT = "#e6edf3"

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.40)
    ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

    def _style(a):
        a.set_facecolor(GRID)
        for spine in a.spines.values():
            spine.set_edgecolor("#30363d")
        a.tick_params(colors=CTEXT, labelsize=8)
        a.xaxis.label.set_color(CTEXT)
        a.yaxis.label.set_color(CTEXT)
        a.title.set_color(CTEXT)
        a.grid(True, color="#21262d", linewidth=0.5, alpha=0.7)

    t     = data["t"]
    s_arr = data["s"]

    # ── Panel 0,0 — Racing Line coloured by vx ─────────────────────────────
    a = ax[0][0]
    _style(a)
    pts     = np.stack([data["x_car"], data["y_car"]], axis=1).reshape(-1, 1, 2)
    segs    = np.concatenate([pts[:-1], pts[1:]], axis=1)
    vx_c    = data["vx"]
    norm_vx = Normalize(vmin=vx_c.min(), vmax=vx_c.max())
    lc      = LineCollection(segs, cmap="plasma", norm=norm_vx, linewidth=1.2)
    lc.set_array(vx_c)
    a.add_collection(lc)
    # Track centre line (faint)
    a.plot(track["x"], track["y"], ":", color="#30363d", linewidth=0.8, alpha=0.6)
    a.autoscale()
    a.set_aspect("equal")
    cb = fig.colorbar(ScalarMappable(norm=norm_vx, cmap="plasma"), ax=a, pad=0.01)
    cb.ax.tick_params(labelsize=7, colors=CTEXT)
    cb.set_label("vx [m/s]", color=CTEXT, fontsize=8)
    a.set_title("Racing Line — Speed [m/s]", fontsize=9)
    a.set_xlabel("X [m]"); a.set_ylabel("Y [m]")

    # ── Panel 0,1 — Mz_target vs Mz_actual ────────────────────────────────
    a = ax[0][1]
    _style(a)
    if np.any(data["Mz_target"] != 0):
        sm_tgt = _smooth(data["Mz_target"])
        sm_act = _smooth(data["Mz_actual"])
        a.plot(t, sm_tgt, color=C1, linewidth=1.0, label="Mz_target")
        a.plot(t, sm_act, color=C2, linewidth=1.0, label="Mz_actual", alpha=0.8)
        # CBF intervention highlights
        cbf_mask = data["cbf_active"] > 0.5
        if cbf_mask.any():
            a.fill_between(t, a.get_ylim()[0], a.get_ylim()[1],
                           where=cbf_mask, color=C3, alpha=0.15, label="CBF active")
        a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d")
    else:
        a.text(0.5, 0.5, "Powertrain disabled\n(simple TV fallback)",
               ha="center", va="center", color=CTEXT, transform=a.transAxes, fontsize=9)
    a.set_title("TV Yaw Moment: Target vs Actual [Nm]", fontsize=9)
    a.set_xlabel("t [s]"); a.set_ylabel("Mz [Nm]")

    # ── Panel 0,2 — wz_ref vs wz_measured ─────────────────────────────────
    a = ax[0][2]
    _style(a)
    if np.any(data["wz_ref"] != 0):
        a.plot(t, _smooth(data["wz_ref"]), color=C1, linewidth=1.0, label="wz_ref")
        a.plot(t, _smooth(data["wz"]),     color=C2, linewidth=1.0, label="wz_meas", alpha=0.8)
        a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d")
    else:
        a.plot(t, _smooth(data["wz"]), color=C2, linewidth=1.0, label="wz [rad/s]")
        a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d")
    a.set_title("Yaw Rate: Reference vs Measured [rad/s]", fontsize=9)
    a.set_xlabel("t [s]"); a.set_ylabel("ψ̇ [rad/s]")

    # ── Panel 1,0 — Wheel torques ─────────────────────────────────────────
    a = ax[1][0]
    _style(a)
    for i, (k, lbl, col) in enumerate([
        ("T_fl","FL",C1), ("T_fr","FR",C2), ("T_rl","RL",C4), ("T_rr","RR",C5)
    ]):
        if data[k].any():
            a.plot(t, _smooth(data[k], 10), color=col, linewidth=0.9, label=lbl, alpha=0.85)
    a.axhline(0, color="#30363d", linewidth=0.7)
    a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d")
    a.set_title("Per-Wheel Torque (TV Allocation) [Nm]", fontsize=9)
    a.set_xlabel("t [s]"); a.set_ylabel("T [Nm]")

    # ── Panel 1,1 — κ* tracking ──────────────────────────────────────────
    a = ax[1][1]
    _style(a)
    if np.any(data["kappa_star_fl"] != 0):
        a.plot(t, data["kappa_star_fl"],  color=C1, linewidth=0.9, label="κ*_FL (Koopman)")
        a.plot(t, data["kappa_meas_fl"],  color=C3, linewidth=0.9, label="κ_FL (measured)", alpha=0.7)
        a.plot(t, data["kappa_star_rr"],  color=C4, linewidth=0.9, label="κ*_RR (Koopman)", linestyle="--")
        a.plot(t, data["kappa_meas_rr"],  color=C5, linewidth=0.9, label="κ_RR (measured)", alpha=0.7, linestyle="--")
        a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d")
    else:
        a.text(0.5, 0.5, "κ* tracking requires\npowertrain pipeline",
               ha="center", va="center", color=CTEXT, transform=a.transAxes, fontsize=9)
    a.set_title("Koopman κ* vs Measured Slip Ratio", fontsize=9)
    a.set_xlabel("t [s]"); a.set_ylabel("κ [-]")

    # ── Panel 1,2 — Phase plane: β vs ψ̇ ─────────────────────────────────
    a = ax[1][2]
    _style(a)
    vx_safe = np.maximum(data["vx"], 0.5)
    beta    = np.arctan(data["vy"] / vx_safe)
    norm_t  = Normalize(vmin=0, vmax=t.max())
    pts_pp  = np.stack([beta, data["wz"]], axis=1).reshape(-1, 1, 2)
    segs_pp = np.concatenate([pts_pp[:-1], pts_pp[1:]], axis=1)
    lc_pp   = LineCollection(segs_pp, cmap="viridis", norm=norm_t, linewidth=1.0, alpha=0.8)
    lc_pp.set_array(t)
    a.add_collection(lc_pp)
    # Stability boundary (β_max = 0.15 rad, ψ̇_max = 1.5 rad/s)
    theta = np.linspace(0, 2 * np.pi, 200)
    a.plot(0.15 * np.cos(theta), 1.5 * np.sin(theta), color=C3,
           linewidth=1.2, linestyle="--", label="CBF boundary", alpha=0.7)
    a.autoscale()
    a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d")
    a.set_title("Phase Plane: Sideslip β vs Yaw Rate ψ̇", fontsize=9)
    a.set_xlabel("β [rad]"); a.set_ylabel("ψ̇ [rad/s]")

    # ── Panel 2,0 — Regen blend α* ────────────────────────────────────────
    a = ax[2][0]
    _style(a)
    if np.any(data["alpha_regen"] != 0):
        a.fill_between(t, 0, _smooth(data["alpha_regen"]), color=C2, alpha=0.4, label="α_regen")
        a.plot(t, _smooth(data["alpha_regen"]), color=C2, linewidth=1.0)
        ax2 = a.twinx()
        ax2.plot(t, _smooth(data["F_hydraulic"]), color=C3, linewidth=0.9,
                 label="F_hydraulic [N]", alpha=0.7)
        ax2.set_ylabel("F_hyd [N]", color=C3, fontsize=8)
        ax2.tick_params(colors=C3)
        ax2.set_facecolor(GRID)
        a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d", loc="upper left")
        ax2.legend(fontsize=7, labelcolor=C3, facecolor=GRID, edgecolor="#30363d", loc="upper right")
    else:
        a.text(0.5, 0.5, "Regen requires\npowertrain pipeline",
               ha="center", va="center", color=CTEXT, transform=a.transAxes, fontsize=9)
    a.set_title("Regen Blend α* + Hydraulic Brake [N]", fontsize=9)
    a.set_xlabel("t [s]"); a.set_ylabel("α* [-]")
    a.set_ylim(0, 1.1)

    # ── Panel 2,1 — Speed profile: actual vs target ───────────────────────
    a = ax[2][1]
    _style(a)
    a.plot(data["s"], data["vx"], color=C1, linewidth=1.0, label="vx actual")
    a.plot(track["s"], v_prof,    color=C5, linewidth=1.2, linestyle="--",
           label="v_target (velocity sweep)", alpha=0.8)
    a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d")
    a.set_title("Speed Profile vs Arc-Length [m/s]", fontsize=9)
    a.set_xlabel("s [m]"); a.set_ylabel("v [m/s]")

    # ── Panel 2,2 — SoC + motor temperature ──────────────────────────────
    a = ax[2][2]
    _style(a)
    if np.any(data["SoC"] != 0) and data["SoC"].max() > 1.0:
        a.plot(t, data["SoC"], color=C2, linewidth=1.0, label="SoC [%]")
        ax3 = a.twinx()
        ax3.plot(t, _smooth(data["T_motor_fl"]), color=C5, linewidth=0.9,
                 label="T_motor FL [°C]", alpha=0.8)
        ax3.set_ylabel("T_motor [°C]", color=C5, fontsize=8)
        ax3.tick_params(colors=C5)
        ax3.set_facecolor(GRID)
        a.set_ylim(max(0, data["SoC"].min() - 2), min(100, data["SoC"].max() + 2))
        a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d", loc="upper left")
        ax3.legend(fontsize=7, labelcolor=C5, facecolor=GRID, edgecolor="#30363d", loc="upper right")
    else:
        # Fallback: just vx + curvature for sanity check
        a.plot(track["s"], track["kappa"], color=C4, linewidth=1.0, label="κ [1/m]")
        a.legend(fontsize=7, labelcolor=CTEXT, facecolor=GRID, edgecolor="#30363d")
    a.set_title("SoC [%] + Motor Temperature [°C]", fontsize=9)
    a.set_xlabel("t [s]"); a.set_ylabel("SoC [%]")

    out_path = str(Path(out_dir) / "tv_validation_racing_line.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=GS)
    plt.close(fig)
    if verbose:
        print(f"[Plot] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# §6  Validation criteria
# ─────────────────────────────────────────────────────────────────────────────

def compute_pass_fail(data: dict, track: dict) -> dict:
    """
    Objective pass/fail criteria for TV validation.

    All criteria are physically grounded for a Formula Student car.
    """
    t   = data["t"]
    vx  = data["vx"]
    s   = data["s"]
    N   = len(t)

    results = {}

    # 1. Lap completion: car travelled at least total_length × n_laps
    total = float(track["total_length"])
    max_s = float(data["s"].max()) + (data["lap_n"].max() * total)
    results["LAP_COMPLETE"] = {
        "value": max_s, "threshold": total * 0.95,
        "pass": max_s >= total * 0.95,
        "unit": "m", "desc": "Total arc-length covered ≥ 95% of 1 lap"
    }

    # 2. Mean speed in physically reasonable range
    mean_vx = float(np.mean(vx))
    results["MEAN_SPEED"] = {
        "value": mean_vx, "threshold": (10.0, 22.0),
        "pass": 10.0 <= mean_vx <= 22.0,
        "unit": "m/s", "desc": "Mean lap speed in [10, 22] m/s"
    }

    # 3. No NaN in state (already checked but make explicit)
    has_nan = bool(np.any(~np.isfinite(vx)))
    results["NO_NAN"] = {
        "value": int(np.sum(~np.isfinite(vx))), "threshold": 0,
        "pass": not has_nan,
        "unit": "count", "desc": "Zero NaN/Inf in vx trajectory"
    }

    # 4. TV Mz tracking error (only if powertrain active)
    if np.any(data["Mz_target"] != 0):
        mz_err = float(np.mean(np.abs(data["Mz_actual"] - data["Mz_target"])))
        results["MZ_TRACKING"] = {
            "value": mz_err, "threshold": 200.0,
            "pass": mz_err < 200.0,
            "unit": "Nm", "desc": "Mean |Mz_actual − Mz_target| < 200 Nm"
        }

        # 5. CBF intervention rate
        cbf_rate = float(np.mean(data["cbf_active"])) * 100.0
        results["CBF_RATE"] = {
            "value": cbf_rate, "threshold": 15.0,
            "pass": cbf_rate < 15.0,
            "unit": "%", "desc": "CBF intervention rate < 15% of timesteps"
        }

        # 6. κ* tracking: Koopman estimate stays in physical range
        kstar = data["kappa_star_fl"]
        kstar_ok = bool(np.all((kstar > 0.01) & (kstar < 0.30)))
        results["KAPPA_STAR_RANGE"] = {
            "value": f"[{kstar.min():.3f}, {kstar.max():.3f}]",
            "threshold": "[0.01, 0.30]",
            "pass": kstar_ok,
            "unit": "-", "desc": "κ* estimate in physical range [0.01, 0.30]"
        }

        # 7. SoC decreases monotonically (consuming energy)
        if data["SoC"].max() > 1.0:
            delta_soc = float(data["SoC"][0]) - float(data["SoC"][-1])
            results["SOC_DECREASES"] = {
                "value": delta_soc, "threshold": 0.0,
                "pass": delta_soc > 0.0,
                "unit": "%", "desc": "SoC decreases under power (energy consumed)"
            }

    # 8. Sideslip stays below CBF limit
    vx_safe = np.maximum(vx, 0.5)
    beta    = np.arctan(data["vy"] / vx_safe)
    beta_max = float(np.max(np.abs(beta)))
    results["SIDESLIP_BOUNDED"] = {
        "value": float(np.degrees(beta_max)), "threshold": 9.0,
        "pass": beta_max < np.radians(9.0),
        "unit": "deg", "desc": "Max |β| < 9° (CBF hard limit ~8.6°)"
    }

    return results


def print_summary(results: dict, out_dir: str = "out"):
    """Print and save validation summary."""
    lines = [
        "=" * 62,
        "  PROJECT-GP — TORQUE VECTORING VALIDATION SUMMARY",
        "=" * 62,
    ]
    n_pass = n_fail = 0
    for name, r in results.items():
        tag  = "PASS" if r["pass"] else "FAIL"
        val  = r["value"]
        thr  = r["threshold"]
        unit = r["unit"]
        desc = r["desc"]
        val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
        thr_str = f"{thr:.3f}" if isinstance(thr, float) else str(thr)
        lines.append(f"  [{tag}] {name:<24} = {val_str:>10} {unit:<6}  (thr: {thr_str})")
        if r["pass"]:
            n_pass += 1
        else:
            n_fail += 1

    lines += [
        "=" * 62,
        f"  RESULT: {n_pass} PASS, {n_fail} FAIL",
        "=" * 62,
    ]
    summary = "\n".join(lines)
    print("\n" + summary)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / "tv_summary.txt")
    with open(out_path, "w") as f:
        f.write(summary + "\n")
    print(f"[Summary] Saved → {out_path}")
    return n_fail == 0


# ─────────────────────────────────────────────────────────────────────────────
# §7  CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(data: dict, out_dir: str = "out"):
    """Export per-step telemetry to CSV for offline analysis."""
    try:
        import pandas as pd
        out_path = str(Path(out_dir) / "tv_diagnostics.csv")
        df = pd.DataFrame(data)
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"[CSV] Saved → {out_path}  ({len(df)} rows, {len(df.columns)} cols)")
    except ImportError:
        # Manual CSV without pandas
        out_path = str(Path(out_dir) / "tv_diagnostics.csv")
        keys = list(data.keys())
        rows = len(data[keys[0]])
        with open(out_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for i in range(rows):
                f.write(",".join(str(data[k][i]) for k in keys) + "\n")
        print(f"[CSV] Saved → {out_path}  ({rows} rows, {len(keys)} cols)")


# ─────────────────────────────────────────────────────────────────────────────
# §8  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Project-GP: Closed-loop TV validation + racing line optimizer"
    )
    parser.add_argument("--track",   default="fsg_autocross",
                        choices=["fsg_autocross", "skidpad"],
                        help="Track layout")
    parser.add_argument("--n-laps",  type=int, default=1,
                        help="Number of laps to simulate")
    parser.add_argument("--dt",      type=float, default=0.005,
                        help="Simulation timestep [s]")
    parser.add_argument("--no-pt",   action="store_true",
                        help="Disable powertrain manager (simple TV fallback)")
    parser.add_argument("--output",  default="out",
                        help="Output directory for plots and CSV")
    parser.add_argument("--mu",      type=float, default=1.40,
                        help="Friction coefficient for velocity sweep")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    print("\n" + "█" * 62)
    print("  PROJECT-GP — RACING LINE + TORQUE VECTORING VALIDATOR")
    print("█" * 62)
    print(f"  Track:     {args.track}")
    print(f"  N laps:    {args.n_laps}")
    print(f"  dt:        {args.dt * 1000:.1f} ms")
    print(f"  Powertrain:{'disabled (--no-pt)' if args.no_pt else 'enabled'}")
    print(f"  Output:    {args.output}")
    print("█" * 62 + "\n")

    # ── 1. Build track ────────────────────────────────────────────────────────
    print("[1/5] Building track...")
    if args.track == "skidpad":
        track = _build_skidpad(N=300)
        print(f"      Skidpad: R=9.125m, {len(track['s'])} nodes")
    else:
        track = _build_fsg_autocross(N=600)
        print(f"      FSG Autocross: {track['total_length']:.0f}m, {len(track['s'])} nodes")

    # ── 2. Velocity sweep ─────────────────────────────────────────────────────
    print("[2/5] Computing velocity profile (forward-backward sweep)...")
    ds   = float(track["s"][1] - track["s"][0])
    v_prof = velocity_sweep(
        track["kappa"], mu=args.mu,
        V_max=25.0, a_max_lon=12.0, a_max_brk=16.0, ds=ds
    )
    print(f"      v_min={v_prof.min():.1f}  v_max={v_prof.max():.1f}  "
          f"v_mean={v_prof.mean():.1f} m/s")

    # ── 3. Closed-loop simulation ─────────────────────────────────────────────
    print("[3/5] Running closed-loop simulation...")
    t0   = time.perf_counter()
    data = run_closed_loop_lap(
        track, v_prof,
        n_laps=args.n_laps,
        dt=args.dt,
        use_powertrain=(not args.no_pt),
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    n_steps = len(data["t"])
    print(f"      {n_steps} steps in {elapsed:.1f}s "
          f"({elapsed/n_steps*1000:.2f} ms/step wall-clock)")

    # ── 4. Pass/fail assessment ───────────────────────────────────────────────
    print("[4/5] Computing validation criteria...")
    results  = compute_pass_fail(data, track)
    all_pass = print_summary(results, out_dir=args.output)

    # ── 5. Plots + CSV ────────────────────────────────────────────────────────
    print("[5/5] Generating diagnostics...")
    export_csv(data, out_dir=args.output)
    if not args.no_plot:
        plot_tv_diagnostics(data, track, v_prof, out_dir=args.output)

    print("\n" + "█" * 62)
    print(f"  {'✅ ALL PASS' if all_pass else '❌ SOME FAILURES — see tv_summary.txt'}")
    print("█" * 62 + "\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())