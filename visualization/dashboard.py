import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Project-GP | Engineering Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Professional Dark Mode) ---
st.markdown("""
<style>
    /* Card Styling */
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 4px; 
        color: white;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 600;
        color: #00C0F2;
        font-family: 'Roboto Mono', monospace; /* Monospace for numbers */
    }
    .metric-label {
        font-size: 12px;
        color: #B0B0B0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .sub-value {
        font-size: 11px;
        color: #00FF00;
        margin-top: 4px;
    }
    
    /* Global App Styling */
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
        letter-spacing: -0.5px;
    }
    .stRadio > label {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.opt_file = os.path.join(self.base_dir, 'optimization_results.csv')

    def render(self):
        with st.sidebar:
            st.title("Project-GP")
            st.caption("Engineering Suite v1.0")
            st.markdown("---")
            
            mode = st.radio("MODULE SELECT", [
                "Setup Optimization", 
                "Driver Coaching", 
                "Telemetry Map"
            ])
            
            st.markdown("---")
            st.caption("System Status: ONLINE")

        if mode == "Driver Coaching":
            self.render_driver_analysis()
        elif mode == "Setup Optimization":
            self.render_setup_optimizer()
        elif mode == "Telemetry Map":
            self.render_track_map()

    def render_setup_optimizer(self):
        st.title("Setup Optimization")
        
        if not os.path.exists(self.opt_file):
            st.error(f"Results file not found at: {self.opt_file}")
            st.info("Run 'python main.py' to generate data.")
            return

        try:
            df = pd.read_csv(self.opt_file)
            df.columns = df.columns.str.strip() 
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        # --- DATA PREP ---
        if 'Lat_G_Score' in df.columns:
            x_col = 'Max Lat G'
            y_col = 'Stability %'
            df[x_col] = -df['Lat_G_Score'] 
            df[y_col] = df['Stability_Overshoot'] * 100
            color_col = 'k_f'
            title = "Pareto Front: Grip vs. Stability"
        elif 'Lap_Time' in df.columns:
            x_col = 'Lap Time [s]'
            y_col = 'Stability Index'
            df[x_col] = df['Lap_Time']
            df[y_col] = df['Stability']
            color_col = 'k_f'
            title = "Pareto Front: Lap Time vs. Stability"
        else:
            st.error("Unknown CSV format.")
            st.dataframe(df.head())
            return

        # --- PARETO CHART ---
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### {title}")
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                color=color_col,
                hover_data=df.columns,
                color_continuous_scale="Viridis",
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("### Top Setup Configuration")
            if 'Lat_G_Score' in df.columns:
                best = df.sort_values(by=x_col, ascending=False).iloc[0]
            else:
                best = df.sort_values(by=x_col, ascending=True).iloc[0]
            
            st.metric("Performance Index", f"{best[x_col]:.3f}")
            st.metric("Stability Index", f"{best[y_col]:.1f}")
            
            st.markdown("**Spring Rates (N/mm)**")
            st.code(f"Front: {best.get('k_f',0)/1000:.1f}\nRear:  {best.get('k_r',0)/1000:.1f}")

        # --- PARALLEL COORDINATES ---
        st.markdown("### Engineering Trade-off Analysis")
        eng_cols = [c for c in ['k_f', 'k_r', 'arb_f', 'arb_r', 'c_f', 'c_r', x_col, y_col] if c in df.columns]
        if len(eng_cols) > 2:
            fig2 = px.parallel_coordinates(
                df.head(50), 
                color=x_col,
                dimensions=eng_cols,
                color_continuous_scale=px.colors.diverging.Tealrose,
            )
            fig2.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    def render_driver_analysis(self):
        st.title("Driver Coaching Analysis")
        
        # --- GENERATE REALISTIC MOCK DATA ---
        s = np.linspace(0, 1000, 500)
        v_ghost = 25 - 15 * np.abs(np.sin(s/60.0))
        v_ghost = np.clip(v_ghost, 10, 30)
        v_real = v_ghost * 0.95 + np.random.normal(0, 0.2, 500)
        v_real[100:150] -= 5.0
        
        dt = (s[1] - s[0])
        time_ghost = np.cumsum(dt / v_ghost)
        time_real = np.cumsum(dt / v_real)
        time_loss = time_real - time_ghost
        
        # --- KPI CARDS ---
        c1, c2, c3, c4 = st.columns(4)
        with c1: self._kpi_card("Lap Time", "58.42s", "+1.2s")
        with c2: self._kpi_card("Optimal Time", "57.22s", "Theoretical")
        with c3: self._kpi_card("Corner Score", "84%", "Good")
        with c4: self._kpi_card("Braking Eff.", "92%", "Excellent")

        # --- SPEED TRACE ---
        st.markdown("### Speed Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s, y=v_real, mode='lines', name='Real Driver', line=dict(color='#FF4B4B', width=2)))
        fig.add_trace(go.Scatter(x=s, y=v_ghost, mode='lines', name='Ghost Car', line=dict(color='#00C0F2', dash='dash')))
        fig.update_layout(
            template="plotly_dark", 
            height=400, 
            xaxis_title="Distance [m]", 
            yaxis_title="Speed [m/s]",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- TIME LOSS & INPUTS ---
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("### Time Loss Accumulation")
            fig2 = px.area(x=s, y=time_loss, labels={'x':'Distance', 'y':'Delta T [s]'})
            fig2.update_traces(line_color='#FFA500')
            fig2.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
        with c_right:
            st.markdown("### Control Inputs")
            throttle = np.clip(np.sin(s/20), 0, 1)
            brake = np.clip(-np.sin(s/20), 0, 1)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=s, y=throttle, name='Throttle', line=dict(color='#00FF00')))
            fig3.add_trace(go.Scatter(x=s, y=brake, name='Brake', line=dict(color='#FF0000')))
            fig3.update_layout(template="plotly_dark", height=300, yaxis_title="Pedal %", margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig3, use_container_width=True)

    def render_track_map(self):
        st.title("Track Telemetry Map")
        theta = np.linspace(0, 2*np.pi, 300)
        x = 50 * np.cos(theta) + 20 * np.sin(2*theta)
        y = 50 * np.sin(theta)
        v = 20 + 10 * np.cos(3*theta)
        
        fig = px.scatter(
            x=x, y=y, color=v, 
            color_continuous_scale="Turbo",
            labels={'color': 'Speed [m/s]'},
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(
            template="plotly_dark", 
            height=600,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=0,r=0,t=0,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    def _kpi_card(self, label, value, sub_value):
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="sub-value">{sub_value}</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    db = Dashboard()
    db.render()