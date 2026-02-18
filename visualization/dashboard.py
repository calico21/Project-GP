import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from scipy.interpolate import interp1d

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Project-GP | Engineering Suite",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏎️"
)

# --- MASTER CSS: F1 Telemetry — High Contrast, Readable ---
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
       GLOBAL TEXT — ensure everything is readable
    ============================================= */
    body, p, span, div, li, td, th {
        color: var(--text-bright) !important;
        font-size: 14px !important;
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
    /* Sidebar nav labels */
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

    /* Page title */
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

    /* Section headers */
    h2, h3 {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: var(--text-bright) !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        margin: 18px 0 10px !important;
    }

    /* =============================================
       KPI CARD — glass with VISIBLE text
    ============================================= */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-mid);
        border-top: 2px solid var(--cyan);
        border-radius: 6px;
        padding: 20px 16px 16px;
        text-align: center;
        position: relative;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        border-color: var(--border-bright);
        box-shadow: 0 6px 28px rgba(0,192,242,0.12);
    }

    .metric-label {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: var(--cyan) !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        margin-bottom: 10px !important;
    }

    .metric-value {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: var(--text-white) !important;
        letter-spacing: 1px !important;
        line-height: 1.1 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }

    .sub-value {
        font-size: 12px !important;
        margin-top: 8px !important;
        padding: 3px 10px !important;
        display: inline-block !important;
        border-radius: 3px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }
    .sub-value.positive {
        color: var(--green) !important;
        background: rgba(35, 209, 96, 0.12) !important;
        border: 1px solid rgba(35, 209, 96, 0.35) !important;
    }
    .sub-value.negative {
        color: var(--red) !important;
        background: rgba(255, 75, 75, 0.12) !important;
        border: 1px solid rgba(255, 75, 75, 0.35) !important;
    }
    .sub-value.neutral {
        color: var(--text-mid) !important;
        background: rgba(155, 163, 188, 0.10) !important;
        border: 1px solid rgba(155, 163, 188, 0.25) !important;
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
        width: 4px;
        height: 16px;
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

    /* Channel sub-label above stacked charts */
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
       CODE BLOCKS
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


# =============================================
#  SHARED PLOTLY LAYOUT DEFAULTS
# =============================================
PLOT_BG  = "#13161F"
GRID_COL = "rgba(255,255,255,0.06)"
AXIS_COL = "rgba(255,255,255,0.15)"

# All chart text uses a size and color that is actually readable
TICK_FONT  = dict(family="Courier New, monospace", size=12, color="#9BA3BC")
LABEL_FONT = dict(family="Courier New, monospace", size=12, color="#9BA3BC")

def base_layout(height=350, **kwargs):
    """Returns a plotly layout dict — clean, high-contrast, readable."""
    return dict(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"),
        margin=dict(l=8, r=8, t=32, b=8),
        xaxis=dict(
            gridcolor=GRID_COL,
            zerolinecolor=AXIS_COL,
            tickfont=TICK_FONT,
            linecolor=AXIS_COL,
            showline=True,
        ),
        yaxis=dict(
            gridcolor=GRID_COL,
            zerolinecolor=AXIS_COL,
            tickfont=TICK_FONT,
            linecolor=AXIS_COL,
            showline=True,
        ),
        legend=dict(
            bgcolor="rgba(19,22,31,0.9)",
            bordercolor="rgba(0,192,242,0.25)",
            borderwidth=1,
            font=dict(family="Courier New, monospace", size=12, color="#D8DCE8"),
        ),
        **kwargs
    )


class Dashboard:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.opt_file = os.path.join(self.base_dir, 'optimization_results.csv')
        self.ghost_file = os.path.join(self.base_dir, 'ghost_car_telemetry.csv')
        self.track_file = os.path.join(self.base_dir, 'track_model.csv')

    # --------------------------------------------------
    #  SIDEBAR
    # --------------------------------------------------
    def render(self):
        with st.sidebar:
            st.title("PROJECT-GP")
            st.caption("ENGINEERING SUITE  v2.0")
            st.markdown("---")
            
            mode = st.radio("MODULE SELECT", [
                "Setup Optimization",
                "Driver Coaching",
                "Telemetry Map"
            ])
            
            st.markdown("---")
            
            # Live status indicator
            st.markdown("""
            <div class="status-bar">
                <div class="status-dot"></div>
                SYSTEM ONLINE
            </div>
            """, unsafe_allow_html=True)
            
            # Session info panel
            st.markdown("""
            <div class="session-info">
                SESSION &nbsp; <span>R12 / SIM</span><br>
                CIRCUIT &nbsp; <span>GHOST-01</span><br>
                STATUS &nbsp;&nbsp;&nbsp; <span>OPTIMIZED</span>
            </div>
            """, unsafe_allow_html=True)

        if mode == "Driver Coaching":
            self.render_driver_analysis()
        elif mode == "Setup Optimization":
            self.render_setup_optimizer()
        elif mode == "Telemetry Map":
            self.render_track_map()

    # --------------------------------------------------
    #  SETUP OPTIMIZER
    # --------------------------------------------------
    def render_setup_optimizer(self):
        st.title("Setup Optimization")
        
        if not os.path.exists(self.opt_file):
            st.error(f"Results file not found: {self.opt_file}. Please run 'main.py' first.")
            return

        try:
            df = pd.read_csv(self.opt_file)
            df.columns = df.columns.str.strip()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        # --- DATA PREP ---
        # Map physics engine outputs to visualization labels
        if 'Lat_G_Score' in df.columns:
            x_col = 'Max Lat G'
            y_col = 'Stability %'
            df[x_col] = -df['Lat_G_Score'] # Convert back to positive G
            df[y_col] = df['Stability_Overshoot'] * 100
            color_col = 'k_f'
        else:
            # Fallback for older formats
            x_col = df.columns[0]
            y_col = df.columns[1]
            color_col = df.columns[2]

        # --- PARETO CHART ---
        self._section("Pareto Front")
        c1, c2 = st.columns([3, 1])
        with c1:
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                hover_data=df.columns,
                color_continuous_scale=[
                    [0.0, "#FF4B4B"],
                    [0.5, "#00C0F2"], 
                    [1.0, "#00E676"],
                ],
            )
            fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color='rgba(0,0,0,0.5)')))
            fig.update_coloraxes(colorbar=dict(
                thickness=10, tickfont=TICK_FONT,
                title=dict(text="Front Spring", font=dict(family="Courier New, monospace", size=12))
            ))
            layout = base_layout(height=480)
            layout['xaxis']['title'] = dict(text=x_col, font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"))
            layout['yaxis']['title'] = dict(text=y_col, font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"))
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            self._section("Best Config")
            # Sort by Grip (Maximize X)
            best = df.sort_values(by=x_col, ascending=False).iloc[0]
            
            st.metric("Peak Grip", f"{best[x_col]:.3f} G")
            st.metric("Stability",   f"{best[y_col]:.1f} %")
            st.markdown("""
            <div class="channel-label">Spring Rates</div>
            """, unsafe_allow_html=True)
            st.code(f"FRONT  {best.get('k_f',0)/1000:.1f} kN/m\nREAR   {best.get('k_r',0)/1000:.1f} kN/m")
            
            st.markdown("""
            <div class="channel-label">Dampers</div>
            """, unsafe_allow_html=True)
            st.code(f"FRONT  {best.get('c_f',0)/1000:.1f} Ns/mm\nREAR   {best.get('c_r',0)/1000:.1f} Ns/mm")

        # --- PARALLEL COORDS ---
        self._section("Trade-off Analysis")
        eng_cols = [c for c in ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', x_col, y_col] if c in df.columns]
        if len(eng_cols) > 2:
            fig2 = px.parallel_coordinates(
                df.head(50), 
                color=x_col,
                dimensions=eng_cols,
                color_continuous_scale=[
                    [0.0, "#FF4B4B"],
                    [0.5, "#00C0F2"], 
                    [1.0, "#00E676"],
                ],
            )
            layout2 = base_layout(height=380)
            layout2['margin'] = dict(l=60, r=20, t=40, b=20)
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------
    #  DRIVER COACHING
    # --------------------------------------------------
    def render_driver_analysis(self):
        st.title("Driver Coaching Analysis")
        
        # Load Real Ghost Data if available
        if os.path.exists(self.ghost_file):
            df_g = pd.read_csv(self.ghost_file)
            
            # --- REAL GHOST DATA ---
            s = df_g['s'].values
            v_ghost = df_g['v'].values # m/s
            
            # Derive Controls from Physics (since OCP outputs states)
            # Throttle/Brake approximation using diff(v)
            accel = np.gradient(v_ghost, s) * v_ghost # a = v * dv/ds
            tps_ghost = np.clip(accel / 10.0, 0, 1) * 100 # Normalize roughly
            brake_ghost = np.clip(-accel / 10.0, 0, 1) * 100
            
            # Steer (rad -> deg)
            steer_ghost = df_g['delta'].values * (180/np.pi) * 15.0 # 15:1 Steering Ratio
            
            # Create "Driver" trace (Simulated 'bad' driver for comparison)
            # In a real app, you would load a second CSV here.
            v_real = v_ghost * 0.95 + np.random.normal(0, 0.5, len(s))
            tps_real = tps_ghost * 0.9 + np.random.normal(0, 5, len(s))
            brake_real = brake_ghost * 0.9 + np.random.normal(0, 5, len(s))
            steer_real = steer_ghost + np.random.normal(0, 5, len(s))
            
            lap_time = df_g['time'].max() if 'time' in df_g.columns else 0.0
            
        else:
            st.warning("Ghost Telemetry not found. Using Mock Data.")
            # Fallback to Mock Data
            s = np.linspace(0, 1000, 500)
            v_ghost = 25 - 15 * np.abs(np.sin(s / 60.0))
            v_real = v_ghost * 0.95
            tps_ghost = np.abs(np.sin(s/50))*100
            brake_ghost = np.abs(np.cos(s/50))*100
            tps_real = tps_ghost
            brake_real = brake_ghost
            steer_ghost = np.sin(s/100)*100
            steer_real = steer_ghost
            lap_time = 60.0

        # --- KPI ROW ---
        c1, c2, c3, c4 = st.columns(4)
        with c1: self._kpi_card("Lap Time",     f"{lap_time*1.02:.2f}s",  "+1.2s vs Ghost",   "negative")
        with c2: self._kpi_card("Optimal Time", f"{lap_time:.2f}s",  "Theoretical Best", "neutral")
        with c3: self._kpi_card("Corner Score", "84%",     "GOOD",             "positive")
        with c4: self._kpi_card("Braking Eff.", "92%",     "EXCELLENT",        "positive")
        
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # --- SPEED TRACE ---
        self._section("Speed Comparison")
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=s, y=v_ghost, name='Ghost (Optimal)',
            line=dict(color='#00C0F2', width=1.5, dash='dash'), opacity=0.7
        ))
        fig_v.add_trace(go.Scatter(
            x=s, y=v_real, name='Driver (You)',
            line=dict(color='#FF4B4B', width=2)
        ))
        
        layout_v = base_layout(height=280)
        layout_v['yaxis']['title'] = dict(text="Speed [m/s]", font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"))
        layout_v['xaxis']['showticklabels'] = False
        layout_v['legend']['orientation'] = "h"
        layout_v['legend']['y'] = 1.12
        fig_v.update_layout(**layout_v)
        st.plotly_chart(fig_v, use_container_width=True)

        # --- TELEMETRY CHANNELS ---
        self._section("Driver Input Channels")
        
        def hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
            h = hex_color.lstrip('#')
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        channels = [
            ("Throttle %",    tps_real,    tps_ghost,    "#00E676", "Ghost TPS"),
            ("Brake %",       brake_real,  brake_ghost,  "#FF4B4B", "Ghost Brake"),
            ("Steer [deg]",   steer_real,  steer_ghost,  "#AB63FA", "Ghost Steer"),
        ]

        for label, driver_data, ghost_data, accent, ghost_name in channels:
            st.markdown(f'<div class="channel-label">{label}</div>', unsafe_allow_html=True)
            fig_ch = go.Figure()
            fig_ch.add_trace(go.Scatter(
                x=s, y=ghost_data, name=ghost_name,
                line=dict(color='rgba(255,255,255,0.25)', width=1, dash='dot')
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
                layout_ch['xaxis']['title'] = dict(text="Distance [m]", font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"))
            layout_ch['margin'] = dict(l=8, r=8, t=8, b=8 if not is_last else 36)
            layout_ch['showlegend'] = False
            layout_ch['yaxis']['range'] = [-5, 105] if "%" in label else None
            layout_ch['yaxis']['tickfont'] = dict(family="Courier New, monospace", size=12, color="#9BA3BC")
            fig_ch.update_layout(**layout_ch)
            st.plotly_chart(fig_ch, use_container_width=True)

    # --------------------------------------------------
    #  TRACK MAP
    # --------------------------------------------------
    def render_track_map(self):
        st.title("Telemetry Map  ·  3D")
        
        # Try to load actual track model
        if os.path.exists(self.track_file):
            df_track = pd.read_csv(self.track_file)
            x = df_track['x'].values
            y = df_track['y'].values
            z = np.zeros_like(x) # Flat track assumption if no elevation data
            
            # Map speed from ghost telemetry to track position
            v = np.zeros_like(x) + 20.0
            if os.path.exists(self.ghost_file):
                df_g = pd.read_csv(self.ghost_file)
                # Simple interpolation of speed onto track nodes
                # Assuming s matches roughly
                if len(df_g) > 0:
                    v_interp = interp1d(df_g['s'], df_g['v'], fill_value="extrapolate")
                    v = v_interp(df_track['s'])
        else:
            # Fallback to Circle/Spiral
            theta = np.linspace(0, 2 * np.pi, 500)
            x = 60 * np.cos(theta) + 20 * np.sin(2 * theta)
            y = 60 * np.sin(theta)
            z = 15 * np.sin(3 * theta)
            v = 20 + 10 * np.cos(3 * theta)

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(
                size=3,
                color=v,
                colorscale=[
                    [0.0, "#FF4B4B"],
                    [0.5, "#FFB020"], 
                    [1.0, "#00C0F2"],
                ],
                showscale=True,
                colorbar=dict(
                    title=dict(text="Speed m/s", font=dict(family="Courier New, monospace", size=12)),
                    x=0.92, thickness=10,
                    tickfont=dict(family="Courier New, monospace", size=12, color="#9BA3BC"),
                    outlinewidth=0,
                )
            ),
            line=dict(color='rgba(255,255,255,0.15)', width=2),
            name='Racing Line'
        )])

        fig.update_layout(
            template="plotly_dark",
            height=680,
            paper_bgcolor=PLOT_BG,
            font=dict(family="Courier New, monospace", size=12, color="#9BA3BC"),
            scene=dict(
                bgcolor=PLOT_BG,
                xaxis=dict(title="X [m]", gridcolor=GRID_COL, zerolinecolor=AXIS_COL, backgroundcolor=PLOT_BG, tickfont=TICK_FONT),
                yaxis=dict(title="Y [m]", gridcolor=GRID_COL, zerolinecolor=AXIS_COL, backgroundcolor=PLOT_BG, tickfont=TICK_FONT),
                zaxis=dict(title="Z [m]", gridcolor=GRID_COL, zerolinecolor=AXIS_COL, backgroundcolor=PLOT_BG, tickfont=TICK_FONT),
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    #  HELPERS
    # --------------------------------------------------
    def _section(self, title: str):
        st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)

    def _kpi_card(self, label: str, value: str, sub_value: str, sentiment: str = "neutral"):
        safe = sentiment if sentiment in ("positive", "negative", "neutral") else "neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-corner-br"></div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <span class="sub-value {safe}">{sub_value}</span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    db = Dashboard()
    db.render()