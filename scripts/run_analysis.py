import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- FIX: Add 'src' directory to Python Path ---
# This ensures we can import 'fsae_core' directly
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

# Now we can import directly from fsae_core (without 'src.' prefix)
from fsae_core.utils.config_loader import load_vehicle_config
from fsae_core.optimal_control.solver import OptimalLapSolver

def load_real_track(filename):
    """
    Loads a track CSV (x, y, z) and calculates distance.
    """
    if not os.path.exists(filename):
        print(f"⚠️  Track file not found: {filename}")
        print("   Falling back to generated 'Peanut' track.")
        return generate_track_data()

    df = pd.read_csv(filename) 
    
    # Calculate path distance (s)
    dx = np.diff(df['x'], prepend=0)
    dy = np.diff(df['y'], prepend=0)
    dist = np.cumsum(np.sqrt(dx**2 + dy**2))
    
    return {
        'total_length': dist[-1],
        'x_center': df['x'].values,
        'y_center': df['y'].values,
        'width': np.ones_like(dist) * 5.0, 
        'segment_length': dist[-1] / len(df)
    }

def generate_track_data():
    """
    Generates a simple 'Peanut' shape track for testing 
    """
    theta = np.linspace(0, 2*np.pi, 200)
    # Parametric equations for a peanut/figure-8 shape
    x = 100 * np.cos(theta)
    y = 60 * np.sin(theta) * np.cos(theta/2)
    
    # Calculate simple track width and curvature
    dx = np.gradient(x)
    dy = np.gradient(y)
    dist = np.cumsum(np.sqrt(dx**2 + dy**2))
    
    return {
        'total_length': dist[-1],
        'x_center': x,
        'y_center': y,
        'width': np.ones_like(x) * 6.0, 
        'segment_length': dist[-1] / len(x)
    }

def main():
    print("--- FSAE Optimum Dynamics Dashboard ---")
    
    # 1. Load Configs
    try:
        # Paths relative to where the script is running
        project_root = os.path.join(current_dir, '..')
        config_path = os.path.join(project_root, 'config', 'vehicle_params')
        params = load_vehicle_config(config_path)
        print("✅ Vehicle parameters loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return

    # 2. Load Track
    # For now, we use the generator. Change this line when you have a CSV.
    # track = load_real_track(os.path.join(project_root, "config/track_maps/fs_spain.csv"))
    track = generate_track_data()
    print("✅ Track map generated (Peanut Shape).")

    # 3. Solve for Optimal Lap
    print("🏎️  Computing Optimal Ghost Lap... (This may take 30s)")
    solver = OptimalLapSolver(track, params)
    
    # Solve with lower resolution for speed
    states, controls = solver.solve(N_segments=199)
    
    # 4. Visualize Results
    plot_telemetry(track, states, controls)

def plot_telemetry(track, states, controls):
    velocity = states[14, :-1]
    lat_accel = velocity * states[19, :-1] # v * yaw_rate
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2)

    # Map View (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(states[0, :], states[1, :], c=states[14, :], cmap='inferno', s=5)
    ax1.plot(track['x_center'], track['y_center'], 'k--', alpha=0.3, label='Centerline')
    ax1.set_title("Optimal Trajectory (Color = Speed)")
    ax1.axis('equal')
    plt.colorbar(sc, ax=ax1, label='Speed (m/s)')

    # Speed Trace (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(velocity, 'b-', label='Ghost Speed')
    ax2.set_title("Speed Trace")
    ax2.set_xlabel("Track Distance (nodes)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.grid(True)

    # GG Diagram (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    long_accel = np.diff(velocity) / 0.1 # approximate
    long_accel = np.append(long_accel, 0)
    
    ax3.scatter(lat_accel, long_accel, alpha=0.5, s=2)
    ax3.set_title("G-G Diagram")
    ax3.set_xlabel("Lateral Accel (m/s^2)")
    ax3.set_ylabel("Longitudinal Accel (m/s^2)")
    ax3.axis('equal')
    ax3.grid(True)

    # Controls (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(controls[0, :], 'r', label='Steer')
    ax4.plot(controls[1, :], 'g', label='Throttle')
    ax4.plot(controls[2, :], 'k', label='Brake')
    ax4.set_title("Control Inputs")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()