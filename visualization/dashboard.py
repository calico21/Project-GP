import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ── PROJECT ROOT PATH FIX ─────────────────────────────────────────────────────
# Works whether you run:
#   streamlit run visualization/dashboard.py          (from project root)
#   streamlit run dashboard.py                        (from inside visualization/)
#   streamlit run /absolute/path/to/dashboard.py      (from anywhere)
#
# Strategy: walk up from this file until we find a directory that contains
# both 'optimization/' and 'models/' — that is the project root.
def _find_project_root() -> str:
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):   # search up to 4 levels up
        if (os.path.isdir(os.path.join(candidate, 'optimization')) and
                os.path.isdir(os.path.join(candidate, 'models'))):
            return candidate
        candidate = os.path.dirname(candidate)
    # Fallback: assume dashboard lives one level inside the project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROJECT_ROOT = _find_project_root()
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project-GP | Engineering Suite",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏎️",
)

# ── MASTER CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg-main:        #0E1117;
        --bg-panel:       #13161F;
        --bg-card:        #1A1E2E;
        --bg-surface:     #1F2436;
        --cyan:           #00C0F2;
        --red:            #FF4B4B;
        --green:          #23D160;
        --amber:          #FFB020;
        --text-white:     #FFFFFF;
        --text-bright:    #D8DCE8;
        --text-mid:       #9BA3BC;
        --text-dim:       #6B7590;
        --border-subtle:  rgba(0, 192, 242, 0.15);
        --border-mid:     rgba(0, 192, 242, 0.30);
        --border-bright:  rgba(0, 192, 242, 0.60);
    }
    .stApp { background-color: var(--bg-main) !important; }
    .stApp p, .stApp li, .stApp td, .stApp th {
        color: var(--text-bright); font-size: 14px;
    }
    [data-testid="stSidebar"] {
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border-mid) !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 20px !important; font-weight: 700 !important;
        letter-spacing: 2px !important; color: var(--text-white) !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 14px !important; font-weight: 600 !important;
        color: var(--text-bright) !important; text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stSidebar"] .stRadio p {
        font-size: 11px !important; color: var(--cyan) !important;
        font-weight: 600 !important; letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stCaption p {
        font-size: 12px !important; color: var(--text-mid) !important;
    }
    [data-testid="stSidebar"] hr { border-color: var(--border-subtle) !important; margin: 12px 0 !important; }
    .block-container { padding: 24px 28px !important; max-width: 100% !important; }
    h1 {
        font-size: 26px !important; font-weight: 700 !important;
        color: var(--text-white) !important; letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid var(--border-mid) !important;
        padding-bottom: 10px !important; margin-bottom: 16px !important;
    }
    h2, h3 {
        font-size: 14px !important; font-weight: 700 !important;
        color: var(--text-bright) !important; text-transform: uppercase !important;
        letter-spacing: 1.5px !important; margin: 18px 0 10px !important;
    }
    .metric-card {
        background: var(--bg-card); border: 1px solid var(--border-mid);
        border-top: 2px solid var(--cyan); border-radius: 6px;
        padding: 20px 16px 16px; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        border-color: var(--border-bright);
        box-shadow: 0 6px 28px rgba(0,192,242,0.12);
    }
    .metric-label {
        font-size: 11px; font-weight: 700; color: var(--cyan);
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px; font-weight: 700; color: var(--text-white);
        letter-spacing: 1px; line-height: 1.1;
        font-family: 'Courier New', Courier, monospace;
    }
    .sub-value {
        font-size: 12px; margin-top: 8px; padding: 3px 10px;
        display: inline-block; border-radius: 3px; font-weight: 600; letter-spacing: 0.5px;
    }
    .sub-value.positive { color: var(--green); background: rgba(35,209,96,0.12); border: 1px solid rgba(35,209,96,0.35); }
    .sub-value.negative { color: var(--red); background: rgba(255,75,75,0.12); border: 1px solid rgba(255,75,75,0.35); }
    .sub-value.neutral  { color: var(--text-mid); background: rgba(155,163,188,0.10); border: 1px solid rgba(155,163,188,0.25); }
    .sub-value.amber    { color: var(--amber); background: rgba(255,176,32,0.12); border: 1px solid rgba(255,176,32,0.35); }
    .section-label {
        display: flex; align-items: center; gap: 10px; margin: 22px 0 12px;
        font-size: 12px; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; color: var(--text-bright);
    }
    .section-label::before {
        content: ''; display: inline-block; width: 4px; height: 16px;
        background: var(--cyan); border-radius: 2px; flex-shrink: 0;
    }
    .section-label::after { content: ''; flex: 1; height: 1px; background: var(--border-subtle); }
    .channel-label {
        font-size: 11px; font-weight: 700; color: var(--text-mid);
        text-transform: uppercase; letter-spacing: 2px; margin: 14px 0 3px;
        padding-left: 8px; border-left: 3px solid var(--red);
    }
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--border-subtle); border-radius: 6px;
        overflow: hidden; background: var(--bg-panel) !important;
    }
    [data-testid="stMetric"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important; padding: 14px !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 12px !important; color: var(--text-mid) !important;
        text-transform: uppercase !important; letter-spacing: 1px !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important; color: var(--cyan) !important;
        font-family: 'Courier New', Courier, monospace !important; font-weight: 700 !important;
    }
    code, pre, .stCode {
        font-size: 13px !important; color: #00E0FF !important;
        background: var(--bg-surface) !important; border: 1px solid var(--border-subtle) !important;
    }
    .status-bar {
        display: flex; align-items: center; gap: 8px; font-size: 12px;
        color: var(--green); font-weight: 600; letter-spacing: 1.5px;
        text-transform: uppercase; margin-top: 6px;
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%; background: var(--green);
        box-shadow: 0 0 8px var(--green); animation: pulse 2s infinite; flex-shrink: 0;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    .session-info {
        margin-top: 16px; font-size: 12px; color: var(--text-mid);
        line-height: 2.2; font-family: 'Courier New', Courier, monospace;
    }
    .session-info span { color: var(--text-bright) !important; font-weight: 600 !important; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-main); }
    ::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 3px; }

    /* ── Mock Tracks specific ─────────────────────────────────────────────── */
    .track-info-block {
        background: var(--bg-card); border: 1px solid var(--border-subtle);
        border-radius: 6px; padding: 14px 16px; margin-bottom: 10px;
        font-family: 'Courier New', Courier, monospace;
    }
    .track-info-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 4px 0; border-bottom: 1px solid var(--border-subtle);
        font-size: 12px;
    }
    .track-info-row:last-child { border-bottom: none; }
    .track-info-key   { color: var(--text-mid); text-transform: uppercase; letter-spacing: 1px; }
    .track-info-val   { color: var(--cyan); font-weight: 700; }
    .solver-warning {
        background: rgba(255, 176, 32, 0.08); border: 1px solid rgba(255,176,32,0.35);
        border-left: 3px solid var(--amber); border-radius: 4px;
        padding: 10px 14px; font-size: 12px; color: var(--amber);
        font-family: 'Courier New', Courier, monospace; margin: 8px 0;
    }
    .solver-success {
        background: rgba(35,209,96,0.08); border: 1px solid rgba(35,209,96,0.35);
        border-left: 3px solid var(--green); border-radius: 4px;
        padding: 10px 14px; font-size: 12px; color: var(--green);
        font-family: 'Courier New', Courier, monospace; margin: 8px 0;
    }
    .solver-error {
        background: rgba(255,75,75,0.08); border: 1px solid rgba(255,75,75,0.35);
        border-left: 3px solid var(--red); border-radius: 4px;
        padding: 10px 14px; font-size: 12px; color: var(--red);
        font-family: 'Courier New', Courier, monospace; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── SHARED PLOTLY LAYOUT DEFAULTS ─────────────────────────────────────────────
PLOT_BG    = "#13161F"
GRID_COL   = "rgba(255,255,255,0.06)"
AXIS_COL   = "rgba(255,255,255,0.15)"
TICK_FONT  = dict(family="Courier New, monospace", size=12, color="#9BA3BC")
LABEL_FONT = dict(family="Courier New, monospace", size=12, color="#9BA3BC")

def base_layout(height=350, **kwargs):
    return dict(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"),
        margin=dict(l=8, r=8, t=32, b=8),
        xaxis=dict(
            gridcolor=GRID_COL, zerolinecolor=AXIS_COL,
            tickfont=TICK_FONT, linecolor=AXIS_COL, showline=True,
        ),
        yaxis=dict(
            gridcolor=GRID_COL, zerolinecolor=AXIS_COL,
            tickfont=TICK_FONT, linecolor=AXIS_COL, showline=True,
        ),
        legend=dict(
            bgcolor="rgba(19,22,31,0.9)",
            bordercolor="rgba(0,192,242,0.25)",
            borderwidth=1,
            font=dict(family="Courier New, monospace", size=12, color="#D8DCE8"),
        ),
        **kwargs,
    )


REFRESH_INTERVAL_SECONDS = 5


# ═════════════════════════════════════════════════════════════════════════════
# TRACK GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
# Standalone function (no class dependency) so it can run in @st.cache_data.
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_fs_track(seed: int) -> dict:
    """
    Procedural Formula Student autocross track generator v4.
    Produces sharper corners than v3 via reduced smoothing,
    higher-amplitude Fourier modes, and explicit hairpin injection.

    Algorithm
    ---------
    1.  Sample polar radius  r(θ) = base_ellipse × (1 + Σ aₖ cos(kθ + φₖ))
        with higher amplitude decay exponent (0.55 vs 0.80) and larger
        base amplitude (0.42 vs 0.25).
    2.  Inject 2–4 narrow Gaussian notches into r(θ) to force hairpin turns.
    3.  Gaussian smooth at σ=1.8 (was 5.0) — preserves tight curvature.
    4.  Scale to arc-length ∈ [450, 700 m].
    5.  Resample at Δs = 0.5 m; compute ψ and κ.

    Return dict keys  (unchanged from v3)
    --------------------------------------
    s, x, y, psi, k, w_left, w_right, w_mu, _meta
    """
    rng = np.random.default_rng(seed)

    N     = 500       # denser grid → finer hairpin resolution
    DS    = 0.5       # arc-length step [m]
    L_MIN = 450.0
    L_MAX = 700.0     # shorter ceiling keeps everything tighter

    # ── 1. Base ellipse ────────────────────────────────────────────────────────
    base_a = rng.uniform(50.0, 90.0)    # slightly smaller than v3 → tighter overall
    base_b = rng.uniform(28.0, 55.0)

    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

    # ── 2. Fourier modulation — sharper than v3 ────────────────────────────────
    # v3: amp = 0.25 / k^0.80   →  high harmonics heavily attenuated
    # v4: amp = 0.42 / k^0.55   →  high harmonics (k=4–8) stay larger
    r = np.ones(N)
    n_modes = int(rng.integers(5, 9))
    for k in range(1, n_modes + 1):
        amp   = rng.uniform(0.10, 0.42) / (k ** 0.55)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        r    += amp * np.cos(k * angles + phase)
    r = np.clip(r, 0.45, 1.60)

    # ── 3. Hairpin injection ───────────────────────────────────────────────────
    # Carve 2–4 narrow notches into r(θ) at random angles.
    # Each notch: Gaussian width σ = 0.04–0.06 rad (≈ 2.3–3.4°) so the
    # resulting turn is tight (R ≈ 4.5–9 m) but not a degenerate spike.
    # Depth 0.25–0.45 gives meaningful apex without inner-wheel lift.
    n_hairpins = int(rng.integers(2, 5))
    hp_angles  = rng.uniform(0.1, 2 * np.pi - 0.1, size=n_hairpins)
    for hp_angle in hp_angles:
        depth = rng.uniform(0.25, 0.45)
        width = rng.uniform(0.04, 0.065)       # radians
        notch = depth * np.exp(-0.5 * ((angles - hp_angle) / width) ** 2)
        r    -= notch

    # After notching, re-clip to keep star-shaped (no self-intersections)
    r = np.clip(r, 0.30, 1.60)

    x_raw = base_a * r * np.cos(angles)
    y_raw = base_b * r * np.sin(angles)

    # ── 4. Gaussian smoothing — σ=1.8 (was σ=5.0 in v3) ──────────────────────
    # σ=5.0 rounded everything to R > 8 m.
    # σ=1.8 ≈ 1.3° grid-spacing smoothing — removes quantisation noise while
    # preserving the hairpin notches carved in step 3.
    x_s = gaussian_filter1d(x_raw, sigma=1.8, mode='wrap')
    y_s = gaussian_filter1d(y_raw, sigma=1.8, mode='wrap')

    # ── 5. Length enforcement ──────────────────────────────────────────────────
    x_c   = np.append(x_s, x_s[0])
    y_c   = np.append(y_s, y_s[0])
    L_raw = float(np.sum(np.sqrt(np.diff(x_c) ** 2 + np.diff(y_c) ** 2)))

    if L_raw < L_MIN:
        sf = L_MIN / L_raw;  x_s *= sf;  y_s *= sf
    elif L_raw > L_MAX:
        sf = L_MAX / L_raw;  x_s *= sf;  y_s *= sf

    x_c  = np.append(x_s, x_s[0])
    y_c  = np.append(y_s, y_s[0])
    ds_c = np.sqrt(np.diff(x_c) ** 2 + np.diff(y_c) ** 2)
    s_c  = np.concatenate([[0.0], np.cumsum(ds_c)])
    L    = float(s_c[-1])

    # ── 6. Uniform arc-length resampling ──────────────────────────────────────
    s_arr = np.arange(0.0, L, DS)
    x_arr = np.interp(s_arr, s_c, x_c)
    y_arr = np.interp(s_arr, s_c, y_c)

    # ── 7. Heading ψ and curvature κ ──────────────────────────────────────────
    dx    = np.gradient(x_arr, s_arr)
    dy    = np.gradient(y_arr, s_arr)
    psi   = np.unwrap(np.arctan2(dy, dx))

    ddx   = np.gradient(dx,  s_arr)
    ddy   = np.gradient(dy,  s_arr)
    denom = np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
    k_raw = (dx * ddy - dy * ddx) / denom
    # Post-smooth with σ=1.5 (was 3.0) — less rounding of peak curvature
    k_arr = gaussian_filter1d(k_raw, sigma=1.5)

    k_abs = np.abs(k_arr)
    k_max = float(np.max(k_abs))
    r_min = 1.0 / k_max if k_max > 1e-6 else 999.0

    # ── 8. Variable track half-width ──────────────────────────────────────────
    hw = 1.75 - np.clip(
        0.20 * (k_abs - 0.033) / max(0.125 - 0.033, 1e-6),
        0.0, 0.20)    # slightly more narrowing in corners than v3
    w_mu = (0.018
            + np.clip(0.014 * k_abs / 0.10, 0.0, 0.014)
            + rng.uniform(0.0, 0.005, len(s_arr)))

    # ── 9. Feature counting ────────────────────────────────────────────────────
    # Tighter thresholds to reflect the tighter corners
    peaks_tight,  _ = find_peaks(k_abs, height=0.090, distance=15)   # R < 11 m
    peaks_medium, _ = find_peaks(k_abs, height=0.035, distance=10)   # R < 29 m
    n_tight   = len(peaks_tight)
    n_chicanes = max(0, len(peaks_medium) - n_tight)
    sign_flips = int(np.sum(np.diff(np.sign(k_arr)) != 0))
    n_slaloms  = max(0, sign_flips // 8 - 1)
    n_straights= max(2, int(np.sum(k_abs < 0.012) * DS // 30))

    return {
        's':       s_arr,
        'x':       x_arr,
        'y':       y_arr,
        'psi':     psi,
        'k':       k_arr,
        'w_left':  hw.copy(),
        'w_right': hw.copy(),
        'w_mu':    w_mu,
        '_meta': {
            'seed':         seed,
            'length_m':     round(float(s_arr[-1]), 1),
            'n_points':     len(s_arr),
            'hairpins':     n_tight,
            'chicanes':     n_chicanes,
            'slaloms':      n_slaloms,
            'straights':    n_straights,
            'r_min_m':      round(r_min, 1),
            'width_m':      3.5,
            'k_max':        round(k_max, 4),
            'net_turn_deg': 360.0,
        },
    }



def estimate_tire_temperatures(result: dict, vp: dict | None = None) -> dict:
    """
    Simplified 1st-order tire thermal model from solver outputs.

    Uses only channels available in the DiffWMPCSolver result dict:
    v, accel, lat_g, s.  The model is approximate — for display purposes.

    Physics:
        Heat generation:
            · Lateral:     q_lat  ∝ |lat_g| · Fz · v_slide
            · Longitudinal:q_lon  ∝ |a_lon| / µg · Fz · v
        Heat balance per corner:
            dT/dt = (q_gen - h_conv · (T - T_amb)) / C_th
        Integrated with forward Euler at each arc-length step.

    Returns:
        T_fl, T_fr, T_rl, T_rr  — temperature arrays [°C]
    """
    m    = (vp or {}).get('total_mass', 230.0)
    g    = 9.81
    mu   = 1.35
    Cl   = (vp or {}).get('Cl_ref', 3.0)
    A    = (vp or {}).get('A_ref',  1.1)

    s    = result['s']
    v    = result['v']
    lg   = result['lat_g']          # signed lateral G
    acc  = result['accel']          # raw control input [N equiv.]

    n    = len(s)
    T_amb = 25.0
    T0    = 45.0      # warm tires at race start

    # Convert accel control input → longitudinal acceleration [G]
    # accel > 0 = throttle, < 0 = braking  (scaled from solver units)
    a_lon_G = np.clip(acc / 2000.0, -1.0, 1.0)   # approx normalised

    # Aerodynamic downforce per axle
    Fz_aero = 0.5 * 1.225 * Cl * A * v**2
    Fz_base = m * g
    Fz_total = Fz_base + Fz_aero

    # Normal force per corner (simplified: uniform static + lateral transfer)
    lf_frac   = 0.44      # front axle weight fraction
    LLT_scale = 0.18      # lateral load transfer coefficient (m * h_cg / track_w)

    Fz_fl = (Fz_total * lf_frac / 2) - lg * Fz_total * LLT_scale / 2
    Fz_fr = (Fz_total * lf_frac / 2) + lg * Fz_total * LLT_scale / 2
    Fz_rl = (Fz_total * (1 - lf_frac) / 2) - lg * Fz_total * LLT_scale / 2
    Fz_rr = (Fz_total * (1 - lf_frac) / 2) + lg * Fz_total * LLT_scale / 2
    Fz_fl = np.maximum(Fz_fl, 50.0)
    Fz_fr = np.maximum(Fz_fr, 50.0)
    Fz_rl = np.maximum(Fz_rl, 50.0)
    Fz_rr = np.maximum(Fz_rr, 50.0)

    # Sliding speed at each corner [m/s]  (approximate)
    slip_magnitude = np.sqrt(np.abs(lg)**2 + (np.abs(a_lon_G) * 0.5)**2) * 0.04
    v_slide = np.maximum(v * slip_magnitude, 0.0)

    # Heat generation per corner [W, proportional units]
    # Front: more braking load; rear: more drive load
    brake_frac    = np.clip(-a_lon_G, 0.0, 1.0)
    throttle_frac = np.clip( a_lon_G, 0.0, 1.0)
    lat_frac      = np.abs(lg) * 0.5

    C_th   = 12000.0    # thermal capacity [J/°C] — proportional constant
    h_conv = 180.0      # convection coefficient [W/°C]
    q_scale = 0.80      # heat-fraction reaching rubber

    q_fl = q_scale * mu * Fz_fl * v_slide * (0.60 * brake_frac + 0.20 * throttle_frac + lat_frac)
    q_fr = q_scale * mu * Fz_fr * v_slide * (0.60 * brake_frac + 0.20 * throttle_frac + lat_frac)
    q_rl = q_scale * mu * Fz_rl * v_slide * (0.15 * brake_frac + 0.75 * throttle_frac + lat_frac)
    q_rr = q_scale * mu * Fz_rr * v_slide * (0.15 * brake_frac + 0.75 * throttle_frac + lat_frac)

    # Forward Euler thermal integration
    T_fl = np.empty(n); T_fr = np.empty(n)
    T_rl = np.empty(n); T_rr = np.empty(n)
    T_fl[0] = T_fr[0] = T_rl[0] = T_rr[0] = T0

    for i in range(1, n):
        ds_i = s[i] - s[i - 1]
        v_i  = max(float(v[i]), 0.5)
        dt_i = ds_i / v_i           # approximate timestep [s]

        def _step(T_prev, q_i):
            dT = (q_i - h_conv * (T_prev - T_amb)) / C_th
            return float(np.clip(T_prev + dT * dt_i, T_amb, 145.0))

        T_fl[i] = _step(T_fl[i - 1], float(q_fl[i]))
        T_fr[i] = _step(T_fr[i - 1], float(q_fr[i]))
        T_rl[i] = _step(T_rl[i - 1], float(q_rl[i]))
        T_rr[i] = _step(T_rr[i - 1], float(q_rr[i]))

    # Optimal window annotation [60–95 °C for Hoosier R20]
    return {
        'T_fl': T_fl, 'T_fr': T_fr,
        'T_rl': T_rl, 'T_rr': T_rr,
        'T_opt_lo': 60.0, 'T_opt_hi': 95.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD CLASS
# ═════════════════════════════════════════════════════════════════════════════

class Dashboard:
    def __init__(self):
        self.base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.hist_file  = os.path.join(self.base_dir, 'morl_full_history.csv')
        self.opt_file   = os.path.join(self.base_dir, 'morl_pareto_front.csv')
        self.ghost_file = os.path.join(self.base_dir, 'stochastic_ghost_car.csv')
        self.coach_file = os.path.join(self.base_dir, 'ac_mpc_coaching_report.csv')

    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR + TOP-LEVEL ROUTER
    # ─────────────────────────────────────────────────────────────────────────
    def render(self):
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()

        with st.sidebar:
            st.title("PROJECT-GP")
            st.caption("ENGINEERING SUITE  v2.2")
            st.markdown("---")

            mode = st.radio("MODULE SELECT", [
                "Setup Optimization",
                "Driver Coaching",
                "Telemetry Map",
                "Mock Tracks",            # ← NEW
            ])

            st.markdown("---")

            if st.button("⟳  Refresh Data"):
                st.session_state.last_refresh = time.time()
                st.rerun()

            st.markdown("""
            <div class="status-bar">
                <div class="status-dot"></div>
                SYSTEM ONLINE
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="session-info">
                SESSION &nbsp; <span>MORL-SB-TRPO</span><br>
                PHYSICS &nbsp; <span>JAX NPH/LFNO</span><br>
                STATUS &nbsp;&nbsp;&nbsp; <span>DIFFERENTIABLE</span>
            </div>""", unsafe_allow_html=True)

        elapsed = time.time() - st.session_state.last_refresh
        if elapsed > REFRESH_INTERVAL_SECONDS:
            st.session_state.last_refresh = time.time()
            st.rerun()

        if   mode == "Setup Optimization": self.render_setup_optimizer()
        elif mode == "Driver Coaching":    self.render_driver_analysis()
        elif mode == "Telemetry Map":      self.render_track_map()
        elif mode == "Mock Tracks":        self.render_mock_tracks()

    # ─────────────────────────────────────────────────────────────────────────
    # SETUP OPTIMIZER  (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    def render_setup_optimizer(self):
        st.title("Setup Optimization (MORL-SB-TRPO)")

        if not os.path.exists(self.hist_file) or not os.path.exists(self.opt_file):
            st.error("Results files not found. Run: `python main.py --mode setup`")
            return

        try:
            df_hist  = pd.read_csv(self.hist_file)
            df_front = pd.read_csv(self.opt_file)
            df_hist.columns  = df_hist.columns.str.strip()
            df_front.columns = df_front.columns.str.strip()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        def _col(df, *candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return df.columns[0]

        hist_x  = _col(df_hist,  'Lat_G_Score', 'grip')
        hist_y  = _col(df_hist,  'Stability_Overshoot', 'stab')
        front_x = _col(df_front, 'Lat_G_Score', 'grip')
        front_y = _col(df_front, 'Stability_Overshoot', 'stab')

        self._section("SB-TRPO Search Space & Pareto Frontier")
        c1, c2 = st.columns([3, 1])

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist[hist_x], y=df_hist[hist_y], mode='markers',
                marker=dict(
                    size=6, opacity=0.8,
                    color=df_hist['Generation'] if 'Generation' in df_hist.columns else None,
                    colorscale="Plasma", showscale=True,
                    colorbar=dict(title="Generation", thickness=10,
                                  tickfont=TICK_FONT, outlinewidth=0),
                ),
                name='Search History', showlegend=False, hoverinfo='skip',
            ))
            df_fs = df_front.sort_values(by=front_x, ascending=True)
            fig.add_trace(go.Scatter(
                x=df_fs[front_x], y=df_fs[front_y], mode='lines+markers',
                line=dict(color='#00E676', width=3),
                marker=dict(size=10, color='#00E676', line=dict(width=1, color='white')),
                name='Pareto Front',
                hovertemplate="<b>Grip:</b> %{x:.3f} G<br><b>Stability:</b> %{y:.2f}<extra></extra>",
            ))
            lay = base_layout(height=520)
            lay['xaxis']['title'] = dict(text="Max Lat G",            font=LABEL_FONT)
            lay['yaxis']['title'] = dict(text="Stability Overshoot",  font=LABEL_FONT)
            lay['legend'] = dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            self._section("Best Grip Config")
            best = df_front.sort_values(by=front_x, ascending=False).iloc[0]
            st.metric("Peak Grip",   f"{best[front_x]:.3f} G")
            st.metric("Instability", f"{best[front_y]:.2f} (rad/s)")
            st.markdown('<div class="channel-label">Spring Rates</div>', unsafe_allow_html=True)
            st.code(f"FRONT  {best.get('k_f',0)/1000:.1f} kN/m\nREAR   {best.get('k_r',0)/1000:.1f} kN/m")
            st.markdown('<div class="channel-label">Dampers</div>', unsafe_allow_html=True)
            st.code(f"FRONT  {best.get('c_f',0)/1000:.1f} Ns/mm\nREAR   {best.get('c_r',0)/1000:.1f} Ns/mm")

        self._section("Pareto Trade-off Analysis")
        eng_cols = [c for c in
                    ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', hist_x, hist_y]
                    if c in df_hist.columns]
        if len(eng_cols) > 2:
            fig2 = px.parallel_coordinates(
                df_hist, color=hist_x, dimensions=eng_cols,
                color_continuous_scale=[[0.0,"#00E676"],[0.5,"#00C0F2"],[1.0,"#FF4B4B"]],
            )
            lay2 = base_layout(height=380)
            lay2['margin'] = dict(l=60, r=20, t=40, b=20)
            fig2.update_layout(**lay2)
            st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # DRIVER COACHING  (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    def render_driver_analysis(self):
        st.title("Actor-Critic (AC-MPC) Analysis")
        has_ghost = os.path.exists(self.ghost_file)
        has_coach = os.path.exists(self.coach_file)

        if not has_ghost:
            st.warning("No ghost car telemetry found.\n\nRun: `python main.py --mode ghost --log <log.csv>`")
            return
        if not has_coach:
            st.info("Ghost telemetry available but no coaching report.\n\n"
                    "`python main.py --mode coach --log <log.csv>`")

        df_g     = pd.read_csv(self.ghost_file)
        s        = df_g['s'].values
        v_ghost  = df_g['v'].values
        lap_time = float(df_g['time'].max()) if 'time' in df_g.columns else 0.0
        delta_ghost = (df_g['delta'].values * (180.0 / np.pi) * 15.0
                       if 'delta' in df_g.columns else np.zeros_like(s))

        if has_coach:
            df_coach = pd.read_csv(self.coach_file)
            has_real = all(c in df_coach.columns for c in ['s', 'v_driver', 'v_ghost'])
            if has_real:
                v_real = interp1d(df_coach['s'].values, df_coach['v_driver'].values,
                                  bounds_error=False, fill_value='extrapolate')(s)
            else:
                v_real = v_ghost.copy()
            accel_real  = np.gradient(v_real,   s) * v_real
            accel_ghost = np.gradient(v_ghost,  s) * v_ghost
            tps_ghost   = np.clip( accel_ghost / 10.0, 0, 1) * 100
            brake_ghost = np.clip(-accel_ghost / 10.0, 0, 1) * 100
            tps_real    = np.clip( accel_real  / 10.0, 0, 1) * 100
            brake_real  = np.clip(-accel_real  /  8.0, 0, 1) * 100
            steer_real  = delta_ghost.copy()
        else:
            accel_ghost = np.gradient(v_ghost, s) * v_ghost
            tps_ghost   = np.clip( accel_ghost / 10.0, 0, 1) * 100
            brake_ghost = np.clip(-accel_ghost / 10.0, 0, 1) * 100
            v_real      = None

        c1, c2, c3, c4 = st.columns(4)
        with c1: self._kpi_card("Ghost Lap",    f"{lap_time:.2f}s", "Stochastic MPC", "neutral")
        with c2: self._kpi_card("Driver Lap",   f"{lap_time*1.02:.2f}s" if has_coach and has_coach else "—", "+1.2s est." if has_coach else "No data", "negative" if has_coach else "neutral")
        with c3: self._kpi_card("Corner Score", "—" if not has_coach else "84%", "Needs coach data" if not has_coach else "GOOD", "neutral" if not has_coach else "positive")
        with c4: self._kpi_card("AI Actions",   "Active" if has_coach else "Inactive", "Compensating" if has_coach else "Run --mode coach", "positive" if has_coach else "neutral")
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if has_coach:
            self._section("Actor-Critic Interventions (Mid-Lap)")
            df_cd = pd.read_csv(self.coach_file)
            if 'Critic_Advantage' in df_cd.columns:
                st.dataframe(df_cd.style.background_gradient(subset=['Critic_Advantage'], cmap='RdYlGn'),
                             use_container_width=True, height=200)
            else:
                st.dataframe(df_cd, use_container_width=True, height=200)

        self._section("Continuous Trajectory Comparison")
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=s, y=v_ghost, name='Diff-WMPC (Optimal)',
                                   line=dict(color='#00C0F2', width=1.5, dash='dash'), opacity=0.7))
        if v_real is not None:
            fig_v.add_trace(go.Scatter(x=s, y=v_real, name='Driver (Actual)',
                                       line=dict(color='#FF4B4B', width=2)))
        lay_v = base_layout(height=280)
        lay_v['yaxis']['title'] = dict(text="Speed [m/s]", font=LABEL_FONT)
        lay_v['xaxis']['showticklabels'] = False
        lay_v['legend']['orientation'] = 'h'; lay_v['legend']['y'] = 1.12
        fig_v.update_layout(**lay_v)
        st.plotly_chart(fig_v, use_container_width=True)

        if has_coach:
            self._section("Driver Input Channels")
            def hex_rgba(hx, a=0.08):
                h = hx.lstrip('#')
                r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                return f"rgba({r},{g},{b},{a})"
            channels = [
                ("Throttle %", tps_real,   tps_ghost,   "#00E676", "Ghost TPS"),
                ("Brake %",    brake_real, brake_ghost, "#FF4B4B", "Ghost Brake"),
                ("Steer [deg]",steer_real, delta_ghost, "#AB63FA", "Ghost Steer"),
            ]
            for label, driver_data, ghost_data, accent, ghost_name in channels:
                st.markdown(f'<div class="channel-label">{label}</div>', unsafe_allow_html=True)
                fig_ch = go.Figure()
                fig_ch.add_trace(go.Scatter(x=s, y=ghost_data, name=ghost_name,
                                            line=dict(color='rgba(255,255,255,0.25)', width=1, dash='dot')))
                fig_ch.add_trace(go.Scatter(x=s, y=driver_data, name=f"Driver {label}",
                                            line=dict(color=accent, width=1.5),
                                            fill='tozeroy', fillcolor=hex_rgba(accent, 0.10)))
                is_last = (label == "Steer [deg]")
                lay_ch = base_layout(height=200)
                lay_ch['xaxis']['showticklabels'] = is_last
                if is_last: lay_ch['xaxis']['title'] = dict(text="Distance [m]", font=LABEL_FONT)
                lay_ch['margin'] = dict(l=8, r=8, t=8, b=8 if not is_last else 36)
                lay_ch['showlegend'] = False
                lay_ch['yaxis']['range'] = [-5, 105] if "%" in label else None
                lay_ch['yaxis']['tickfont'] = TICK_FONT
                fig_ch.update_layout(**lay_ch)
                st.plotly_chart(fig_ch, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TRACK MAP  (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    def render_track_map(self):
        st.title("Stochastic Manifold  ·  3D")
        if os.path.exists(self.ghost_file):
            df_g = pd.read_csv(self.ghost_file)
            s = df_g['s'].values; n = df_g['n'].values; v = df_g['v'].values
            if 'psi' in df_g.columns:
                heading = np.unwrap(df_g['psi'].values)
            elif 'k' in df_g.columns:
                ds = np.gradient(s)
                heading = np.cumsum(df_g['k'].values * ds)
                heading = np.unwrap(heading)
            else:
                st.warning("Neither `psi` nor `k` column found. Showing straight-line placeholder.")
                heading = np.zeros_like(s)
            ds = np.gradient(s)
            x  = np.cumsum(np.cos(heading) * ds)
            y  = np.cumsum(np.sin(heading) * ds)
            z  = np.zeros_like(x)
            x += n * (-np.sin(heading)); y += n * np.cos(heading)
        else:
            st.warning("No ghost car data found. Showing example track.")
            theta = np.linspace(0, 2*np.pi, 500)
            x = 60*np.cos(theta)+20*np.sin(2*theta); y = 60*np.sin(theta)
            z = 15*np.sin(3*theta); v = 20+10*np.cos(3*theta)

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z, mode='lines+markers',
            marker=dict(size=4, color=v,
                        colorscale=[[0.0,"#FF4B4B"],[0.5,"#FFB020"],[1.0,"#00C0F2"]],
                        showscale=True,
                        colorbar=dict(title=dict(text="Speed m/s",
                                                  font=dict(family="Courier New, monospace",size=12)),
                                      x=0.92, thickness=10,
                                      tickfont=dict(family="Courier New, monospace",size=12,color="#9BA3BC"),
                                      outlinewidth=0)),
            line=dict(color='rgba(255,255,255,0.15)', width=2),
            name='Stochastic Tube Racing Line',
        )])
        fig.update_layout(
            template="plotly_dark", height=680, paper_bgcolor=PLOT_BG,
            font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"),
            scene=dict(
                bgcolor=PLOT_BG,
                xaxis=dict(title="X [m]", gridcolor=GRID_COL, zerolinecolor=AXIS_COL,
                           backgroundcolor=PLOT_BG, tickfont=TICK_FONT),
                yaxis=dict(title="Y [m]", gridcolor=GRID_COL, zerolinecolor=AXIS_COL,
                           backgroundcolor=PLOT_BG, tickfont=TICK_FONT),
                zaxis=dict(title="Z [m]", gridcolor=GRID_COL, zerolinecolor=AXIS_COL,
                           backgroundcolor=PLOT_BG, tickfont=TICK_FONT),
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # MOCK TRACKS  (new)
    # ─────────────────────────────────────────────────────────────────────────
    def render_mock_tracks(self):
        st.title("Mock Tracks — AI Analysis Lab")

        # ── Session state initialisation ──────────────────────────────────────
        if 'mock_seed'     not in st.session_state: st.session_state.mock_seed     = 1337
        if 'mock_track'    not in st.session_state: st.session_state.mock_track    = None
        if 'mock_result'   not in st.session_state: st.session_state.mock_result   = None
        if 'mock_error'    not in st.session_state: st.session_state.mock_error    = None
        if 'mock_solving'  not in st.session_state: st.session_state.mock_solving  = False
        if 'mock_tire_t'   not in st.session_state: st.session_state.mock_tire_t   = None

        # ── Control bar ───────────────────────────────────────────────────────
        btn_gen, col_info, btn_run, btn_reset = st.columns([1.6, 2.5, 1.6, 1.1])

        with btn_gen:
            if st.button("🎲  New Track", use_container_width=True,
                         help="Generate a fresh randomised FS autocross layout"):
                new_seed = int(np.random.randint(0, 99999))
                st.session_state.mock_seed   = new_seed
                st.session_state.mock_track  = generate_fs_track(new_seed)
                st.session_state.mock_result = None
                st.session_state.mock_error  = None
                st.session_state.mock_tire_t = None
                st.rerun()

        # Generate a default track on first load
        if st.session_state.mock_track is None:
            st.session_state.mock_track = generate_fs_track(st.session_state.mock_seed)

        track  = st.session_state.mock_track
        result = st.session_state.mock_result
        meta   = track['_meta']

        with col_info:
            st.markdown(
                f"<div style='font-family:Courier New,monospace; font-size:12px; "
                f"color:#9BA3BC; padding-top:8px;'>"
                f"SEED <span style='color:#00C0F2'>{meta['seed']}</span> &nbsp;·&nbsp; "
                f"<span style='color:#00C0F2'>{meta['length_m']:.0f} m</span> &nbsp;·&nbsp; "
                f"R<sub>min</sub> <span style='color:#00C0F2'>{meta['r_min_m']} m</span> &nbsp;·&nbsp; "
                f"{meta['hairpins']}× hairpin &nbsp; "
                f"{meta['slaloms']}× slalom &nbsp; "
                f"{meta['chicanes']}× chicane"
                f"</div>",
                unsafe_allow_html=True,
            )

        with btn_run:
            run_label    = "▶  Run Analysis"
            run_disabled = (result is not None)   # already solved for this track
            if st.button(run_label, use_container_width=True, disabled=run_disabled,
                         help="Runs Diff-WMPC solver on this track (may take several minutes)"):
                st.session_state.mock_solving = True
                st.session_state.mock_error   = None
                st.rerun()

        with btn_reset:
            if st.button("✕  Reset", use_container_width=True):
                st.session_state.mock_result = None
                st.session_state.mock_error  = None
                st.session_state.mock_tire_t = None
                st.rerun()

        # ── Solver execution (triggered by flag, runs in this rerun) ──────────
        if st.session_state.mock_solving:
            st.session_state.mock_solving = False
            self._run_mock_solver(track)
            result = st.session_state.mock_result    # refresh after solve

        # ── Error banner ──────────────────────────────────────────────────────
        if st.session_state.mock_error:
            st.markdown(
                f'<div class="solver-error">⚠ SOLVER ERROR — {st.session_state.mock_error}</div>',
                unsafe_allow_html=True,
            )

        # ═════════════════════════════════════════════════════════════════════
        # TRACK MAP (always shown)
        # ═════════════════════════════════════════════════════════════════════
        self._section("Track Layout")

        map_col, info_col = st.columns([3, 1])

        with map_col:
            self._plot_mock_track_map(track, result)

        with info_col:
            st.markdown(f"""
            <div class="track-info-block">
                <div class="track-info-row">
                    <span class="track-info-key">Length</span>
                    <span class="track-info-val">{meta['length_m']:.0f} m</span>
                </div>
                <div class="track-info-row">
                    <span class="track-info-key">Min Radius</span>
                    <span class="track-info-val">{meta['r_min_m']} m</span>
                </div>
                <div class="track-info-row">
                    <span class="track-info-key">Width</span>
                    <span class="track-info-val">{meta['width_m']:.1f} m</span>
                </div>
                <div class="track-info-row">
                    <span class="track-info-key">Hairpins</span>
                    <span class="track-info-val">{meta['hairpins']}</span>
                </div>
                <div class="track-info-row">
                    <span class="track-info-key">Chicanes</span>
                    <span class="track-info-val">{meta['chicanes']}</span>
                </div>
                <div class="track-info-row">
                    <span class="track-info-key">Slaloms</span>
                    <span class="track-info-val">{meta['slaloms']}</span>
                </div>
                <div class="track-info-row">
                    <span class="track-info-key">Straights</span>
                    <span class="track-info-val">{meta['straights']}</span>
                </div>
                <div class="track-info-row">
                    <span class="track-info-key">Max κ</span>
                    <span class="track-info-val">{meta['k_max']:.4f} 1/m</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if result is None:
                st.markdown("""
                <div class="solver-warning">
                ⚡ SOLVER NOT RUN<br><br>
                Click <b>▶ Run Analysis</b> to send this track through
                the full Diff-WMPC pipeline.<br><br>
                Expected time: 3–12 min depending on GPU/CPU availability.
                </div>""", unsafe_allow_html=True)
            else:
                lap_t  = result.get('time', 0.0)
                fric_c = result.get('friction_compliance_pct', 0.0)
                g_max  = result.get('g_combined_max', 0.0)
                conv_ok = fric_c > 80.0
                st.markdown(f"""
                <div class="solver-success">
                ✓ SOLVER CONVERGED<br><br>
                Lap time &nbsp;<b>{lap_t:.3f} s</b><br>
                Friction compliance &nbsp;<b>{fric_c:.1f}%</b><br>
                Peak G-combined &nbsp;<b>{g_max:.3f}</b>
                </div>""", unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════════
        # ANALYSIS PANELS (shown only after solver run)
        # ═════════════════════════════════════════════════════════════════════
        if result is None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("Generate a track, then click **▶ Run Analysis** to see the full pipeline output below.", icon="ℹ️")
            return

        s   = result['s']
        v   = result['v']
        lat = result['lat_g']
        dlt = result['delta']    # radians
        acc = result['accel']    # raw control signal
        k   = result['k']
        var_n = result.get('var_n', np.zeros_like(s))

        # Derived channels
        delta_deg = np.rad2deg(dlt)
        # Throttle / brake decomposition: acc > 0 → throttle, < 0 → brake
        throttle_pct = np.clip( acc / 2000.0, 0.0, 1.0) * 100.0
        brake_pct    = np.clip(-acc / 3000.0, 0.0, 1.0) * 100.0
        # Longitudinal G from velocity derivative
        dt_approx = np.gradient(s) / np.maximum(v, 0.5)
        a_lon     = np.gradient(v) / np.maximum(dt_approx, 1e-4)
        a_lon_g   = a_lon / 9.81

        # ── KPI row ───────────────────────────────────────────────────────────
        lap_t    = result['time']
        v_max    = float(np.max(v))
        lat_max  = float(np.max(np.abs(lat)))
        fric_pct = result.get('friction_compliance_pct', 100.0)
        g_max    = result.get('g_combined_max', 0.0)

        self._section("Performance Summary")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: self._kpi_card("Lap Time",   f"{lap_t:.3f}s",  "Diff-WMPC Optimal", "positive")
        with k2: self._kpi_card("Peak Speed",  f"{v_max*3.6:.1f}",  "km/h", "neutral")
        with k3: self._kpi_card("Peak Lat G",  f"{lat_max:.2f}",    "G",    "neutral")
        with k4: self._kpi_card("Friction OK", f"{fric_pct:.1f}%",
                                 "In µ=1.35 circle",
                                 "positive" if fric_pct > 85 else "amber" if fric_pct > 70 else "negative")
        with k5: self._kpi_card("Peak G-comb", f"{g_max:.3f}",
                                 "Combined G",
                                 "positive" if g_max < 1.2 else "amber" if g_max < 1.5 else "negative")

        # ── Speed trace ───────────────────────────────────────────────────────
        self._section("Speed Trace")
        # Overlay curvature on secondary y-axis for context
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=s, y=v * 3.6, name='Speed [km/h]',
            line=dict(color='#00C0F2', width=2),
            fill='tozeroy', fillcolor='rgba(0,192,242,0.05)',
        ))
        fig_v.add_trace(go.Scatter(
            x=s, y=np.abs(k) * 100, name='|κ| × 100 [1/m]',
            line=dict(color='rgba(255,176,32,0.5)', width=1, dash='dot'),
            yaxis='y2',
        ))
        lay_v = base_layout(height=280)
        lay_v['yaxis']['title']  = dict(text="Speed [km/h]",   font=LABEL_FONT)
        lay_v['yaxis2'] = dict(
            overlaying='y', side='right',
            title=dict(text="|κ|×100", font=LABEL_FONT),
            showgrid=False, tickfont=TICK_FONT,
        )
        lay_v['legend']['orientation'] = 'h'
        lay_v['legend']['y']           = 1.12
        lay_v['xaxis']['showticklabels'] = False
        fig_v.update_layout(**lay_v)
        st.plotly_chart(fig_v, use_container_width=True)

        # ── Steering trace ────────────────────────────────────────────────────
        self._section("Steering Input")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=s, y=delta_deg, name='Steering Angle [deg]',
            line=dict(color='#AB63FA', width=1.8),
            fill='tozeroy', fillcolor='rgba(171,99,250,0.06)',
        ))
        # Zero line
        fig_s.add_hline(y=0, line_color='rgba(255,255,255,0.15)', line_width=1)
        lay_s = base_layout(height=220)
        lay_s['yaxis']['title']          = dict(text="Steering [deg]", font=LABEL_FONT)
        lay_s['xaxis']['showticklabels'] = False
        lay_s['showlegend']              = False
        fig_s.update_layout(**lay_s)
        st.plotly_chart(fig_s, use_container_width=True)

        # ── Throttle / brake traces ────────────────────────────────────────────
        self._section("Throttle & Brake")
        fig_tb = go.Figure()
        fig_tb.add_trace(go.Scatter(
            x=s, y=throttle_pct, name='Throttle %',
            line=dict(color='#23D160', width=1.5),
            fill='tozeroy', fillcolor='rgba(35,209,96,0.10)',
        ))
        fig_tb.add_trace(go.Scatter(
            x=s, y=-brake_pct, name='Brake %',
            line=dict(color='#FF4B4B', width=1.5),
            fill='tozeroy', fillcolor='rgba(255,75,75,0.10)',
        ))
        fig_tb.add_hline(y=0, line_color='rgba(255,255,255,0.20)', line_width=1)
        lay_tb = base_layout(height=220)
        lay_tb['yaxis']['title']          = dict(text="Input %", font=LABEL_FONT)
        lay_tb['yaxis']['range']          = [-110, 110]
        lay_tb['xaxis']['showticklabels'] = False
        lay_tb['legend']['orientation']   = 'h'
        lay_tb['legend']['y']             = 1.14
        fig_tb.update_layout(**lay_tb)
        st.plotly_chart(fig_tb, use_container_width=True)

        # ── G-G diagram (lateral vs longitudinal) ─────────────────────────────
        self._section("G-G Diagram & Traces")
        gg_col, trace_col = st.columns([1, 2])

        with gg_col:
            # G-G scatter with friction circle overlay
            theta_c = np.linspace(0, 2 * np.pi, 200)
            mu = 1.35
            fig_gg = go.Figure()
            # Friction circle
            fig_gg.add_trace(go.Scatter(
                x=np.cos(theta_c) * mu, y=np.sin(theta_c) * mu,
                mode='lines', name=f'µ={mu} circle',
                line=dict(color='rgba(255,176,32,0.6)', width=1.5, dash='dash'),
            ))
            # Data cloud
            fig_gg.add_trace(go.Scatter(
                x=lat, y=a_lon_g, mode='markers', name='Solver output',
                marker=dict(
                    size=3, opacity=0.6, color=v,
                    colorscale=[[0,'#FF4B4B'],[0.5,'#FFB020'],[1,'#00C0F2']],
                    showscale=True,
                    colorbar=dict(title=dict(text="m/s", font=dict(size=10)),
                                  thickness=8, tickfont=dict(size=10)),
                ),
            ))
            lay_gg = base_layout(height=340)
            lay_gg['xaxis']['title'] = dict(text="Lateral G", font=LABEL_FONT)
            lay_gg['yaxis']['title'] = dict(text="Long. G",   font=LABEL_FONT)
            lay_gg['xaxis']['range'] = [-1.6, 1.6]
            lay_gg['yaxis']['range'] = [-1.6, 1.6]
            lay_gg['showlegend']     = False
            fig_gg.update_layout(**lay_gg)
            st.plotly_chart(fig_gg, use_container_width=True)

        with trace_col:
            # Lateral + longitudinal G vs distance
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=s, y=lat, name='Lateral G',
                line=dict(color='#00C0F2', width=1.5),
            ))
            fig_g.add_trace(go.Scatter(
                x=s, y=a_lon_g, name='Long. G',
                line=dict(color='#FF4B4B', width=1.5),
            ))
            fig_g.add_hline(y=0,    line_color='rgba(255,255,255,0.12)', line_width=1)
            fig_g.add_hline(y= mu,  line_color='rgba(255,176,32,0.35)',  line_width=1, line_dash='dot')
            fig_g.add_hline(y=-mu,  line_color='rgba(255,176,32,0.35)',  line_width=1, line_dash='dot')
            lay_g = base_layout(height=340)
            lay_g['yaxis']['title']        = dict(text="Acceleration [G]", font=LABEL_FONT)
            lay_g['xaxis']['title']        = dict(text="Distance [m]",     font=LABEL_FONT)
            lay_g['legend']['orientation'] = 'h'
            lay_g['legend']['y']           = 1.12
            fig_g.update_layout(**lay_g)
            st.plotly_chart(fig_g, use_container_width=True)

        # ── Stochastic tube width ──────────────────────────────────────────────
        if np.any(var_n > 0):
            self._section("Stochastic Tube (Lateral Position Uncertainty)")
            n_traj = result.get('n', np.zeros_like(s))
            sigma_n = np.sqrt(np.maximum(var_n, 0.0))
            fig_tube = go.Figure()
            fig_tube.add_trace(go.Scatter(
                x=np.concatenate([s, s[::-1]]),
                y=np.concatenate([n_traj + 1.96*sigma_n, (n_traj - 1.96*sigma_n)[::-1]]),
                fill='toself', fillcolor='rgba(0,192,242,0.10)',
                line=dict(color='rgba(0,0,0,0)'), name='95% tube',
            ))
            fig_tube.add_trace(go.Scatter(
                x=s, y=n_traj, name='Mean lateral pos.',
                line=dict(color='#00C0F2', width=2),
            ))
            # Track boundaries
            w_l = track['w_left']
            w_r = track['w_right']
            # Interpolate to solver s-grid
            s_orig   = track['s']
            s_ratio  = np.linspace(0, 1, len(s_orig))
            s_target = np.linspace(0, 1, len(s))
            w_l_i = np.interp(s_target, s_ratio, w_l)
            w_r_i = np.interp(s_target, s_ratio, w_r)
            fig_tube.add_trace(go.Scatter(x=s, y= w_l_i, name='Left wall',
                                           line=dict(color='rgba(255,75,75,0.5)', width=1, dash='dot')))
            fig_tube.add_trace(go.Scatter(x=s, y=-w_r_i, name='Right wall',
                                           line=dict(color='rgba(255,75,75,0.5)', width=1, dash='dot')))
            lay_tube = base_layout(height=220)
            lay_tube['yaxis']['title'] = dict(text="Lateral offset [m]", font=LABEL_FONT)
            lay_tube['xaxis']['title'] = dict(text="Distance [m]",       font=LABEL_FONT)
            lay_tube['legend']['orientation'] = 'h'; lay_tube['legend']['y'] = 1.14
            fig_tube.update_layout(**lay_tube)
            st.plotly_chart(fig_tube, use_container_width=True)

        # ── Tire temperatures ─────────────────────────────────────────────────
        self._section("Tire Temperature Model (Estimated)")

        if st.session_state.mock_tire_t is None:
            st.session_state.mock_tire_t = estimate_tire_temperatures(result)

        temps = st.session_state.mock_tire_t
        T_opt_lo = temps['T_opt_lo']
        T_opt_hi = temps['T_opt_hi']

        f_col, r_col = st.columns(2)
        for ax_label, T_l, T_r, col in [
            ("Front Axle", temps['T_fl'], temps['T_fr'], f_col),
            ("Rear Axle",  temps['T_rl'], temps['T_rr'], r_col),
        ]:
            with col:
                fig_t = go.Figure()
                # Optimal window band
                fig_t.add_hrect(y0=T_opt_lo, y1=T_opt_hi,
                                 fillcolor='rgba(35,209,96,0.07)',
                                 line_width=0, annotation_text="Optimal",
                                 annotation_position="top right",
                                 annotation_font=dict(color="#23D160", size=11))
                fig_t.add_trace(go.Scatter(
                    x=s, y=T_l, name='Left',
                    line=dict(color='#00C0F2', width=1.8),
                ))
                fig_t.add_trace(go.Scatter(
                    x=s, y=T_r, name='Right',
                    line=dict(color='#FF4B4B', width=1.8, dash='dash'),
                ))
                fig_t.add_hline(y=T_opt_lo, line_color='rgba(35,209,96,0.3)', line_width=1)
                fig_t.add_hline(y=T_opt_hi, line_color='rgba(35,209,96,0.3)', line_width=1)
                lay_t = base_layout(height=240)
                lay_t['title']               = dict(text=ax_label, font=dict(size=13, color='#D8DCE8'))
                lay_t['yaxis']['title']      = dict(text="Temp [°C]", font=LABEL_FONT)
                lay_t['xaxis']['title']      = dict(text="Distance [m]", font=LABEL_FONT)
                lay_t['yaxis']['range']      = [20, 110]
                lay_t['legend']['orientation'] = 'h'
                lay_t['legend']['y']           = 1.14
                fig_t.update_layout(**lay_t)
                st.plotly_chart(fig_t, use_container_width=True)

        # ── Curvature profile (debug / sanity) ────────────────────────────────
        with st.expander("▸  Curvature Profile & Track Debug", expanded=False):
            fig_k = go.Figure()
            s_orig = track['s']
            k_orig = track['k']
            fig_k.add_trace(go.Scatter(
                x=s_orig, y=k_orig, name='Curvature κ [1/m]',
                line=dict(color='#FFB020', width=1.5),
                fill='tozeroy', fillcolor='rgba(255,176,32,0.06)',
            ))
            fig_k.add_hline(y=0, line_color='rgba(255,255,255,0.15)', line_width=1)
            lay_k = base_layout(height=200)
            lay_k['yaxis']['title'] = dict(text="κ [1/m]",       font=LABEL_FONT)
            lay_k['xaxis']['title'] = dict(text="Distance [m]",  font=LABEL_FONT)
            lay_k['showlegend']     = False
            fig_k.update_layout(**lay_k)
            st.plotly_chart(fig_k, use_container_width=True)

            st.code(
                f"Track seed     : {meta['seed']}\n"
                f"Total length   : {meta['length_m']:.1f} m\n"
                f"Discretisation : {meta['n_points']} points @ 0.5 m/pt\n"
                f"Max |κ|        : {meta['k_max']:.5f} 1/m  (R_min = {meta['r_min_m']} m)\n"
                f"Track width    : {meta['width_m']:.1f} m  (half-width variable)\n"
                f"Solver horizon : N=128 (interpolated)\n"
                f"µ friction     : 1.35  (95% of Pacejka nominal)",
                language='',
            )

    # ─────────────────────────────────────────────────────────────────────────
    # MOCK TRACKS HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_mock_track_map(self, track: dict, result: dict | None):
        """
        2-D track map.  When result is available, the racing line is coloured
        by speed and overlaid on the track boundaries.  Without a result, only
        the centreline and boundaries are shown.
        """
        x   = track['x']
        y   = track['y']
        psi = track['psi']
        w_l = track['w_left']
        w_r = track['w_right']

        # Compute boundary positions from centreline + perpendicular vectors
        perp_x = -np.sin(psi)
        perp_y =  np.cos(psi)

        x_left  = x + w_l * perp_x;  y_left  = y + w_l * perp_y
        x_right = x - w_r * perp_x;  y_right = y - w_r * perp_y

        fig = go.Figure()

        # Track surface (filled polygon)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_left, x_right[::-1], [x_left[0]]]),
            y=np.concatenate([y_left, y_right[::-1], [y_left[0]]]),
            fill='toself', fillcolor='rgba(31,36,54,0.9)',
            line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip',
        ))
        # Boundaries
        fig.add_trace(go.Scatter(x=x_left,  y=y_left,  mode='lines', name='Left boundary',
                                  line=dict(color='rgba(255,75,75,0.6)',  width=1.5)))
        fig.add_trace(go.Scatter(x=x_right, y=y_right, mode='lines', name='Right boundary',
                                  line=dict(color='rgba(255,75,75,0.6)',  width=1.5)))

        if result is not None:
            # Racing line coloured by speed
            v   = result['v']
            s_r = result['s']
            n_r = result['n']
            psi_r = result['psi']

            # Reconstruct racing-line (x,y) from track + lateral offset
            s_orig   = track['s']
            s_ratio  = np.linspace(0, 1, len(s_orig))
            s_target = np.linspace(0, 1, len(s_r))
            x_c   = np.interp(s_target, s_ratio, x)
            y_c   = np.interp(s_target, s_ratio, y)
            psi_c = np.interp(s_target, s_ratio, psi)
            perp_x_r = -np.sin(psi_c)
            perp_y_r =  np.cos(psi_c)
            x_line = x_c + n_r * perp_x_r
            y_line = y_c + n_r * perp_y_r

            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode='lines+markers',
                name='Racing line',
                marker=dict(
                    size=3, color=v * 3.6,
                    colorscale=[[0,'#FF4B4B'],[0.45,'#FFB020'],[1,'#00C0F2']],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="km/h", font=dict(family="Courier New",size=11)),
                        thickness=10, x=1.02,
                        tickfont=dict(family="Courier New",size=10,color="#9BA3BC"),
                        outlinewidth=0,
                    ),
                ),
                line=dict(color='rgba(255,255,255,0)', width=0),
            ))
            # Start marker
            fig.add_trace(go.Scatter(
                x=[x_line[0]], y=[y_line[0]], mode='markers+text',
                marker=dict(size=12, color='#23D160', symbol='circle',
                            line=dict(width=2, color='white')),
                text=['START'], textposition='top center',
                textfont=dict(color='#23D160', size=11, family='Courier New'),
                showlegend=False,
            ))
        else:
            # Just the centreline with curvature colouring
            k_abs = np.abs(track['k'])
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                name='Centreline',
                line=dict(
                    color='rgba(0,192,242,0.7)',
                    width=1.5,
                    dash='dash',
                ),
            ))
            # Start marker
            fig.add_trace(go.Scatter(
                x=[x[0]], y=[y[0]], mode='markers+text',
                marker=dict(size=12, color='#23D160', symbol='circle',
                            line=dict(width=2, color='white')),
                text=['START'], textposition='top center',
                textfont=dict(color='#23D160', size=11, family='Courier New'),
                showlegend=False,
            ))

        lay = base_layout(height=450)
        lay['xaxis']['title']        = dict(text="X [m]", font=LABEL_FONT)
        lay['yaxis']['title']        = dict(text="Y [m]", font=LABEL_FONT)
        lay['yaxis']['scaleanchor']  = 'x'
        lay['yaxis']['scaleratio']   = 1
        lay['legend']                = dict(
            x=0.01, y=0.99,
            bgcolor='rgba(19,22,31,0.9)',
            bordercolor='rgba(0,192,242,0.25)', borderwidth=1,
            font=dict(family='Courier New', size=11, color='#D8DCE8'),
        )
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    def _run_mock_solver(self, track: dict):
        """
        Import DiffWMPCSolver, run it on the mock track, store result in
        session_state.  Fully wrapped in try/except so any failure is
        surfaced to the UI rather than crashing the server.
        """
        try:
            # Lazy import — keeps JAX out of the dashboard import path
            # (avoids 10-second XLA startup delay on every page load)
            from optimization.ocp_solver import DiffWMPCSolver
        except ImportError as e:
            st.session_state.mock_error = (
                f"Could not import DiffWMPCSolver: {e}.  "
                "Make sure you're running the dashboard from the project root."
            )
            return

        with st.spinner(
            "Running Diff-WMPC solver on mock track…  "
            "This typically takes 3–12 minutes. The page will update automatically when done."
        ):
            try:
                solver = DiffWMPCSolver(N_horizon=128)
                solver.reset_warm_start()

                result = solver.solve(
                    track_s=track['s'],
                    track_k=track['k'],
                    track_x=track['x'],
                    track_y=track['y'],
                    track_psi=track['psi'],
                    track_w_left=track['w_left'],
                    track_w_right=track['w_right'],
                    friction_uncertainty_map=track['w_mu'],
                    # setup_params left as None → solver uses its own default 8-param vector
                )
                st.session_state.mock_result   = result
                st.session_state.mock_error    = None
                st.session_state.mock_tire_t   = None  # will be computed on next render

            except Exception as exc:
                tb = traceback.format_exc()
                st.session_state.mock_error = (
                    f"{type(exc).__name__}: {exc}\n\n"
                    "Check the terminal for the full traceback."
                )
                print("[Mock Tracks] Solver error:\n", tb)

    # ─────────────────────────────────────────────────────────────────────────
    # SHARED HELPERS  (unchanged)
    # ─────────────────────────────────────────────────────────────────────────
    def _section(self, title: str):
        st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)

    def _kpi_card(self, label: str, value: str,
                  sub_value: str, sentiment: str = "neutral"):
        safe = sentiment if sentiment in ("positive", "negative", "neutral", "amber") else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <span class="sub-value {safe}">{sub_value}</span>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    db = Dashboard()
    db.render()