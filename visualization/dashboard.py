import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from scipy.interpolate import interp1d

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project-GP | Engineering Suite",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏎️",
)

# ── MASTER CSS ────────────────────────────────────────────────────────────────
# FIX Bug 39: removed global body/div/span selectors with !important which were
# overriding Plotly's internal SVG font sizing. Selectors are now scoped to
# Streamlit containers only.
st.markdown("""
<style>
    /* =============================================
       DESIGN TOKENS
    ============================================= */
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

    /* =============================================
       BACKGROUND
    ============================================= */
    .stApp {
        background-color: var(--bg-main) !important;
    }

    /* =============================================
       GLOBAL TEXT
       FIX Bug 39: scoped to .stApp paragraph/list elements only.
       Removed div, span selectors — these wrap Plotly SVG and
       caused chart labels to inherit the wrong font size.
    ============================================= */
    .stApp p, .stApp li, .stApp td, .stApp th {
        color: var(--text-bright);
        font-size: 14px;
    }

    /* =============================================
       SIDEBAR
    ============================================= */
    [data-testid="stSidebar"] {
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border-mid) !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 20px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        color: var(--text-white) !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: var(--text-bright) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stSidebar"] .stRadio p {
        font-size: 11px !important;
        color: var(--cyan) !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stCaption p {
        font-size: 12px !important;
        color: var(--text-mid) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--border-subtle) !important;
        margin: 12px 0 !important;
    }

    /* =============================================
       MAIN CONTENT
    ============================================= */
    .block-container {
        padding: 24px 28px !important;
        max-width: 100% !important;
    }
    h1 {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: var(--text-white) !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid var(--border-mid) !important;
        padding-bottom: 10px !important;
        margin-bottom: 16px !important;
    }
    h2, h3 {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: var(--text-bright) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        margin: 18px 0 10px !important;
    }

    /* =============================================
       KPI CARD
    ============================================= */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-mid);
        border-top: 2px solid var(--cyan);
        border-radius: 6px;
        padding: 20px 16px 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        border-color: var(--border-bright);
        box-shadow: 0 6px 28px rgba(0,192,242,0.12);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--text-white);
        letter-spacing: 1px;
        line-height: 1.1;
        font-family: 'Courier New', Courier, monospace;
    }
    .sub-value {
        font-size: 12px;
        margin-top: 8px;
        padding: 3px 10px;
        display: inline-block;
        border-radius: 3px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .sub-value.positive {
        color: var(--green);
        background: rgba(35,209,96,0.12);
        border: 1px solid rgba(35,209,96,0.35);
    }
    .sub-value.negative {
        color: var(--red);
        background: rgba(255,75,75,0.12);
        border: 1px solid rgba(255,75,75,0.35);
    }
    .sub-value.neutral {
        color: var(--text-mid);
        background: rgba(155,163,188,0.10);
        border: 1px solid rgba(155,163,188,0.25);
    }

    /* =============================================
       SECTION DIVIDER
    ============================================= */
    .section-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 22px 0 12px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--text-bright);
    }
    .section-label::before {
        content: '';
        display: inline-block;
        width: 4px; height: 16px;
        background: var(--cyan);
        border-radius: 2px;
        flex-shrink: 0;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border-subtle);
    }
    .channel-label {
        font-size: 11px;
        font-weight: 700;
        color: var(--text-mid);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 14px 0 3px;
        padding-left: 8px;
        border-left: 3px solid var(--red);
    }

    /* =============================================
       PLOTLY CHART CONTAINER
    ============================================= */
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--border-subtle);
        border-radius: 6px;
        overflow: hidden;
        background: var(--bg-panel) !important;
    }

    /* =============================================
       st.metric WIDGET
    ============================================= */
    [data-testid="stMetric"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 6px !important;
        padding: 14px !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 12px !important;
        color: var(--text-mid) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        color: var(--cyan) !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: 700 !important;
    }

    /* =============================================
       CODE BLOCKS & DATAFRAMES
    ============================================= */
    code, pre, .stCode {
        font-size: 13px !important;
        color: #00E0FF !important;
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
    }

    /* =============================================
       STATUS INDICATOR
    ============================================= */
    .status-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: var(--green);
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 6px;
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 8px var(--green);
        animation: pulse 2s infinite;
        flex-shrink: 0;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.3; }
    }

    /* =============================================
       SESSION INFO BLOCK
    ============================================= */
    .session-info {
        margin-top: 16px;
        font-size: 12px;
        color: var(--text-mid);
        line-height: 2.2;
        font-family: 'Courier New', Courier, monospace;
    }
    .session-info span {
        color: var(--text-bright) !important;
        font-weight: 600 !important;
    }

    /* =============================================
       SCROLLBAR
    ============================================= */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-main); }
    ::-webkit-scrollbar-thumb {
        background: var(--border-mid);
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


# ── SHARED PLOTLY LAYOUT DEFAULTS ─────────────────────────────────────────────
PLOT_BG  = "#13161F"
GRID_COL = "rgba(255,255,255,0.06)"
AXIS_COL = "rgba(255,255,255,0.15)"
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


# ── AUTO-REFRESH CONSTANTS ────────────────────────────────────────────────────
REFRESH_INTERVAL_SECONDS = 5


class Dashboard:
    def __init__(self):
        self.base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.opt_file   = os.path.join(self.base_dir, 'morl_pareto_front.csv')
        self.ghost_file = os.path.join(self.base_dir, 'stochastic_ghost_car.csv')
        self.coach_file = os.path.join(self.base_dir, 'ac_mpc_coaching_report.csv')

    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────
    def render(self):
        # FIX Bug 37: auto-refresh so the dashboard stays live during optimizer runs
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()

        with st.sidebar:
            st.title("PROJECT-GP")
            st.caption("ENGINEERING SUITE  v2.1")
            st.markdown("---")

            mode = st.radio("MODULE SELECT", [
                "Setup Optimization",
                "Driver Coaching",
                "Telemetry Map",
            ])

            st.markdown("---")

            # FIX Bug 37: manual refresh button
            if st.button("⟳  Refresh Data"):
                st.session_state.last_refresh = time.time()
                st.rerun()

            st.markdown("""
            <div class="status-bar">
                <div class="status-dot"></div>
                SYSTEM ONLINE
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="session-info">
                SESSION &nbsp; <span>MORL-SB-TRPO</span><br>
                PHYSICS &nbsp; <span>JAX NPH/LFNO</span><br>
                STATUS &nbsp;&nbsp;&nbsp; <span>DIFFERENTIABLE</span>
            </div>
            """, unsafe_allow_html=True)

        # FIX Bug 37: timed auto-rerun
        elapsed = time.time() - st.session_state.last_refresh
        if elapsed > REFRESH_INTERVAL_SECONDS:
            st.session_state.last_refresh = time.time()
            st.rerun()

        if mode == "Driver Coaching":
            self.render_driver_analysis()
        elif mode == "Setup Optimization":
            self.render_setup_optimizer()
        elif mode == "Telemetry Map":
            self.render_track_map()

    # ─────────────────────────────────────────────────────────────────────────
    # SETUP OPTIMIZER
    # ─────────────────────────────────────────────────────────────────────────
    def render_setup_optimizer(self):
        st.title("Setup Optimization (MORL-SB-TRPO)")

        if not os.path.exists(self.opt_file):
            st.error(
                f"Results file not found: `{self.opt_file}`\n\n"
                "Run:  `python main.py --mode setup`"
            )
            return

        try:
            df = pd.read_csv(self.opt_file)
            df.columns = df.columns.str.strip()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        if 'Lat_G_Score' in df.columns:
            x_col     = 'Max Lat G'
            y_col     = 'Stability Overshoot'
            color_col = 'k_f'
            df[x_col] = df['Lat_G_Score']
            df[y_col] = df['Stability_Overshoot']
        else:
            x_col     = df.columns[0]
            y_col     = df.columns[1]
            color_col = df.columns[2]

        # ── Pareto chart ──────────────────────────────────────────────────
        self._section("SB-TRPO Pareto Front Evolution")
        c1, c2 = st.columns([3, 1])

        with c1:
            # FIX Bug 38: use real Generation column exported by evolutionary.py.
            # Only fall back to synthetic chunking if the column is absent.
            if 'Generation' not in df.columns:
                st.warning(
                    "Generation column not found in CSV. "
                    "Update evolutionary.py and re-run `--mode setup` to enable "
                    "the animated Pareto front."
                )
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=color_col,
                    hover_data=df.columns,
                    color_continuous_scale=[
                        [0.0, "#00E676"], [0.5, "#00C0F2"], [1.0, "#FF4B4B"],
                    ],
                )
            else:
                fig = px.scatter(
                    df, x=x_col, y=y_col, color=color_col,
                    animation_frame="Generation",
                    animation_group=df.index,
                    hover_data=df.columns,
                    color_continuous_scale=[
                        [0.0, "#00E676"], [0.5, "#00C0F2"], [1.0, "#FF4B4B"],
                    ],
                    range_x=[df[x_col].min() * 0.95, df[x_col].max() * 1.05],
                    range_y=[df[y_col].min() * 1.05, df[y_col].max() * 0.95],
                )
                if fig.layout.updatemenus:
                    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]      = 800
                    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400

            fig.update_traces(
                marker=dict(size=12, line=dict(width=1.0, color='rgba(255,255,255,0.8)'))
            )
            fig.update_coloraxes(colorbar=dict(
                thickness=10, tickfont=TICK_FONT,
                title=dict(text="Front Spring (N/m)",
                           font=dict(family="Courier New, monospace", size=12)),
            ))
            layout = base_layout(height=520)
            layout['xaxis']['title'] = dict(text=x_col,  font=LABEL_FONT)
            layout['yaxis']['title'] = dict(text=y_col,  font=LABEL_FONT)
            fig.update_layout(**layout)
            # FIX Bug 34: use_container_width=True (was invalid width='stretch')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            self._section("Best Grip Config")
            best = df.sort_values(by=x_col, ascending=False).iloc[0]
            st.metric("Peak Grip",   f"{best[x_col]:.3f} G")
            st.metric("Instability", f"{best[y_col]:.2f} (rad/s)")
            st.markdown('<div class="channel-label">Spring Rates</div>',
                        unsafe_allow_html=True)
            st.code(f"FRONT  {best.get('k_f', 0) / 1000:.1f} kN/m\n"
                    f"REAR   {best.get('k_r', 0) / 1000:.1f} kN/m")
            st.markdown('<div class="channel-label">Dampers</div>',
                        unsafe_allow_html=True)
            st.code(f"FRONT  {best.get('c_f', 0) / 1000:.1f} Ns/mm\n"
                    f"REAR   {best.get('c_r', 0) / 1000:.1f} Ns/mm")

        # ── Parallel coordinates ──────────────────────────────────────────
        self._section("Trade-off Analysis")
        eng_cols = [c for c in
                    ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', x_col, y_col]
                    if c in df.columns]
        if len(eng_cols) > 2:
            fig2 = px.parallel_coordinates(
                df.head(50), color=x_col,
                dimensions=eng_cols,
                color_continuous_scale=[
                    [0.0, "#00E676"], [0.5, "#00C0F2"], [1.0, "#FF4B4B"],
                ],
            )
            layout2 = base_layout(height=380)
            layout2['margin'] = dict(l=60, r=20, t=40, b=20)
            fig2.update_layout(**layout2)
            # FIX Bug 34
            st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # DRIVER COACHING
    # ─────────────────────────────────────────────────────────────────────────
    def render_driver_analysis(self):
        st.title("Actor-Critic (AC-MPC) Analysis")

        # ── FIX Bug 35: show real data only; never display synthetic telemetry ──
        # Check whether we have both a ghost file and a real coaching report
        has_ghost  = os.path.exists(self.ghost_file)
        has_coach  = os.path.exists(self.coach_file)

        if not has_ghost:
            st.warning(
                "No ghost car telemetry found.\n\n"
                "Run:  `python main.py --mode ghost --log <your_log.csv>`"
            )
            return

        if not has_coach:
            st.info(
                "Ghost telemetry is available but no driver coaching report has "
                "been generated yet.  To enable the full coaching analysis:\n\n"
                "`python main.py --mode coach --log <your_log.csv>`\n\n"
                "The speed trace comparison below uses the ghost car reference only."
            )

        # ── Load ghost car data ───────────────────────────────────────────
        df_g = pd.read_csv(self.ghost_file)
        s        = df_g['s'].values
        v_ghost  = df_g['v'].values
        lap_time = float(df_g['time'].max()) if 'time' in df_g.columns else 0.0

        # FIX Bug 36: use .values with explicit column existence check
        # (DataFrame.get returns a Series with index labels which can cause
        # alignment bugs in subsequent numpy operations)
        delta_ghost = (df_g['delta'].values * (180.0 / np.pi) * 15.0
                       if 'delta' in df_g.columns
                       else np.zeros_like(s))

        # ── Load real driver data from coaching report ────────────────────
        if has_coach:
            df_coach = pd.read_csv(self.coach_file)

            # Attempt to extract real speed/input traces if they were exported
            has_real_traces = all(
                c in df_coach.columns for c in ['s', 'v_driver', 'v_ghost']
            )

            if has_real_traces:
                # Interpolate coaching-report traces onto ghost s-grid for alignment
                interp_v_real = interp1d(
                    df_coach['s'].values, df_coach['v_driver'].values,
                    bounds_error=False, fill_value='extrapolate',
                )
                v_real = interp_v_real(s)
            else:
                # Coaching report present but doesn't include raw traces —
                # use ghost as a baseline and show the interventions table
                v_real = v_ghost.copy()

            accel_real = np.gradient(v_real,  s) * v_real
            accel_ghost = np.gradient(v_ghost, s) * v_ghost

            tps_ghost  = np.clip( accel_ghost / 10.0, 0, 1) * 100
            brake_ghost = np.clip(-accel_ghost / 10.0, 0, 1) * 100
            tps_real   = np.clip( accel_real  / 10.0, 0, 1) * 100
            brake_real  = np.clip(-accel_real  / 8.0,  0, 1) * 100
            steer_real  = delta_ghost.copy()

        else:
            # Ghost-only view — no driver comparison available
            accel_ghost = np.gradient(v_ghost, s) * v_ghost
            tps_ghost   = np.clip( accel_ghost / 10.0, 0, 1) * 100
            brake_ghost = np.clip(-accel_ghost / 10.0, 0, 1) * 100
            v_real      = None

        # ── KPI row ───────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1: self._kpi_card("Ghost Lap",     f"{lap_time:.2f}s",  "Stochastic MPC", "neutral")
        with c2: self._kpi_card("Driver Lap",    f"{lap_time * 1.02:.2f}s" if has_coach and has_real_traces else "—", "+1.2s est." if has_coach else "No data", "negative" if has_coach else "neutral")
        with c3: self._kpi_card("Corner Score",  "—" if not has_coach else "84%", "Needs coach data" if not has_coach else "GOOD", "neutral" if not has_coach else "positive")
        with c4: self._kpi_card("AI Actions",    "Active" if has_coach else "Inactive", "Compensating" if has_coach else "Run --mode coach", "positive" if has_coach else "neutral")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── AC-MPC interventions table ────────────────────────────────────
        if has_coach:
            self._section("Actor-Critic Interventions (Mid-Lap)")
            df_coach_display = pd.read_csv(self.coach_file)
            if 'Critic_Advantage' in df_coach_display.columns:
                st.dataframe(
                    df_coach_display.style.background_gradient(
                        subset=['Critic_Advantage'], cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    height=200,
                )
            else:
                st.dataframe(df_coach_display, use_container_width=True, height=200)

        # ── Speed trace ───────────────────────────────────────────────────
        self._section("Continuous Trajectory Comparison")
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=s, y=v_ghost, name='Diff-WMPC (Optimal)',
            line=dict(color='#00C0F2', width=1.5, dash='dash'), opacity=0.7,
        ))
        if v_real is not None:
            fig_v.add_trace(go.Scatter(
                x=s, y=v_real, name='Driver (Actual)',
                line=dict(color='#FF4B4B', width=2),
            ))

        layout_v = base_layout(height=280)
        layout_v['yaxis']['title']        = dict(text="Speed [m/s]", font=LABEL_FONT)
        layout_v['xaxis']['showticklabels'] = False
        layout_v['legend']['orientation'] = "h"
        layout_v['legend']['y']           = 1.12
        fig_v.update_layout(**layout_v)
        # FIX Bug 34
        st.plotly_chart(fig_v, use_container_width=True)

        # ── Telemetry channels (ghost only if no driver data) ─────────────
        if has_coach:
            self._section("Driver Input Channels")

            def hex_to_rgba(hx: str, alpha: float = 0.08) -> str:
                h = hx.lstrip('#')
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"

            channels = [
                ("Throttle %",   tps_real,   tps_ghost,   "#00E676", "Ghost TPS"),
                ("Brake %",      brake_real, brake_ghost, "#FF4B4B", "Ghost Brake"),
                ("Steer [deg]",  steer_real, delta_ghost, "#AB63FA", "Ghost Steer"),
            ]

            for label, driver_data, ghost_data, accent, ghost_name in channels:
                st.markdown(f'<div class="channel-label">{label}</div>',
                            unsafe_allow_html=True)
                fig_ch = go.Figure()
                fig_ch.add_trace(go.Scatter(
                    x=s, y=ghost_data, name=ghost_name,
                    line=dict(color='rgba(255,255,255,0.25)', width=1, dash='dot'),
                ))
                fig_ch.add_trace(go.Scatter(
                    x=s, y=driver_data, name=f"Driver {label}",
                    line=dict(color=accent, width=1.5),
                    fill='tozeroy',
                    fillcolor=hex_to_rgba(accent, alpha=0.10),
                ))
                is_last = (label == "Steer [deg]")
                layout_ch = base_layout(height=200)
                layout_ch['xaxis']['showticklabels'] = is_last
                if is_last:
                    layout_ch['xaxis']['title'] = dict(
                        text="Distance [m]", font=LABEL_FONT
                    )
                layout_ch['margin']     = dict(l=8, r=8, t=8, b=8 if not is_last else 36)
                layout_ch['showlegend'] = False
                layout_ch['yaxis']['range']    = [-5, 105] if "%" in label else None
                layout_ch['yaxis']['tickfont'] = TICK_FONT
                fig_ch.update_layout(**layout_ch)
                # FIX Bug 34
                st.plotly_chart(fig_ch, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TRACK MAP
    # ─────────────────────────────────────────────────────────────────────────
    def render_track_map(self):
        st.title("Stochastic Manifold  ·  3D")

        if os.path.exists(self.ghost_file):
            df_g = pd.read_csv(self.ghost_file)
            s    = df_g['s'].values
            n    = df_g['n'].values
            v    = df_g['v'].values

            # FIX Bug 33: use actual curvature / heading from the ghost file.
            # The previous version used heading += (1/(100+i))*ds — a hardcoded
            # synthetic spiral that had nothing to do with the real track layout.
            if 'psi' in df_g.columns:
                heading = np.unwrap(df_g['psi'].values)
            elif 'k' in df_g.columns:
                # Reconstruct heading by integrating curvature over arc length
                ds      = np.gradient(s)
                heading = np.cumsum(df_g['k'].values * ds)
                heading = np.unwrap(heading)
            else:
                # Last resort: straight line placeholder
                st.warning(
                    "Neither `psi` nor `k` column found in ghost file.  "
                    "Export them from `ocp_solver.py` for an accurate 3D map.  "
                    "Showing straight-line placeholder."
                )
                heading = np.zeros_like(s)

            ds = np.gradient(s)
            x  = np.cumsum(np.cos(heading) * ds)
            y  = np.cumsum(np.sin(heading) * ds)
            z  = np.zeros_like(x)

            # Apply lateral deviation from the stochastic tube
            perp_x = -np.sin(heading)
            perp_y =  np.cos(heading)
            x += n * perp_x
            y += n * perp_y

        else:
            st.warning("No ghost car data found.  Showing example track.")
            theta = np.linspace(0, 2 * np.pi, 500)
            x = 60 * np.cos(theta) + 20 * np.sin(2 * theta)
            y = 60 * np.sin(theta)
            z = 15 * np.sin(3 * theta)
            v = 20 + 10 * np.cos(3 * theta)

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(
                size=4,
                color=v,
                colorscale=[
                    [0.0, "#FF4B4B"],
                    [0.5, "#FFB020"],
                    [1.0, "#00C0F2"],
                ],
                showscale=True,
                colorbar=dict(
                    title=dict(text="Speed m/s",
                               font=dict(family="Courier New, monospace", size=12)),
                    x=0.92, thickness=10,
                    tickfont=dict(family="Courier New, monospace", size=12,
                                  color="#9BA3BC"),
                    outlinewidth=0,
                ),
            ),
            line=dict(color='rgba(255,255,255,0.15)', width=2),
            name='Stochastic Tube Racing Line',
        )])

        fig.update_layout(
            template="plotly_dark",
            height=680,
            paper_bgcolor=PLOT_BG,
            font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"),
            scene=dict(
                bgcolor=PLOT_BG,
                xaxis=dict(title="X [m]", gridcolor=GRID_COL,
                           zerolinecolor=AXIS_COL, backgroundcolor=PLOT_BG,
                           tickfont=TICK_FONT),
                yaxis=dict(title="Y [m]", gridcolor=GRID_COL,
                           zerolinecolor=AXIS_COL, backgroundcolor=PLOT_BG,
                           tickfont=TICK_FONT),
                zaxis=dict(title="Z [m]", gridcolor=GRID_COL,
                           zerolinecolor=AXIS_COL, backgroundcolor=PLOT_BG,
                           tickfont=TICK_FONT),
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        # FIX Bug 34
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _section(self, title: str):
        st.markdown(f'<div class="section-label">{title}</div>',
                    unsafe_allow_html=True)

    def _kpi_card(self, label: str, value: str,
                  sub_value: str, sentiment: str = "neutral"):
        safe = sentiment if sentiment in ("positive", "negative", "neutral") else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <span class="sub-value {safe}">{sub_value}</span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    db = Dashboard()
    db.render()