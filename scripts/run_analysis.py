import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from fsae_core.utils.config_loader import load_vehicle_config
from fsae_core.optimal_control.solver import OptimalLapSolver

def load_real_track(filename):
    if not os.path.exists(filename):
        print(f"⚠️  Track file not found: {filename}")
        return generate_track_data()

    df = pd.read_csv(filename) 
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
    """ Generates a 'Peanut' (Figure-8) shape for testing """
    theta = np.linspace(0, 2*np.pi, 200)
    # Parametric equations
    x = 100 * np.cos(theta)
    y = 60 * np.sin(theta) * np.cos(theta/2)
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    dist = np.cumsum(np.sqrt(dx**2 + dy**2))
    
    return {
        'total_length': dist[-1],
        'x_center': x,
        'y_center': y,
        'width': np.ones_like(x) * 8.0, # 8 meters wide
        'segment_length': dist[-1] / len(x)
    }

def main():
    print("--- FSAE Optimum Dynamics Dashboard ---")
    
    # 1. Config
    try:
        project_root = os.path.join(current_dir, '..')
        config_path = os.path.join(project_root, 'config', 'vehicle_params')
        params = load_vehicle_config(config_path)
        print("✅ Vehicle parameters loaded.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 2. Track
    track = generate_track_data()
    print("✅ Track generated (Peanut Shape).")

    # 3. Solve
    print("🏎️  Computing Optimal Lap...")
    solver = OptimalLapSolver(track, params)
    states, controls = solver.solve(N_segments=199)
    
    # 4. Visualize
    plot_track_map(track, states)

def plot_track_map(track, states):
    """
    Plots the track borders and the optimal racing line.
    """
    x_center = track['x_center']
    y_center = track['y_center']
    width    = track['width']
    
    # --- 1. Calculate Track Borders ---
    # Gradient (Tangent)
    dx = np.gradient(x_center)
    dy = np.gradient(y_center)
    
    # Normal Vector (Perpendicular to Tangent)
    # Rotate tangent 90 degrees: (dx, dy) -> (-dy, dx)
    norm = np.sqrt(dx**2 + dy**2)
    nx = -dy / norm
    ny = dx / norm
    
    # Inner and Outer coordinates
    half_width = width / 2
    x_inner = x_center + nx * half_width
    y_inner = y_center + ny * half_width
    x_outer = x_center - nx * half_width
    y_outer = y_center - ny * half_width

    # --- 2. Setup Plot ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#2e2e2e') # Dark Tarmac background
    fig.patch.set_facecolor('#1e1e1e')
    
    # Plot Borders
    ax.plot(x_inner, y_inner, color='white', linewidth=2, linestyle='-')
    ax.plot(x_outer, y_outer, color='white', linewidth=2, linestyle='-')
    ax.plot(x_center, y_center, color='white', linewidth=1, linestyle='--', alpha=0.3)

    # --- 3. Plot Racing Line (Colored by Speed) ---
    x_line = states[0, :]
    y_line = states[1, :]
    speed  = states[14, :]
    
    # Create segments for colormap
    points = np.array([x_line, y_line]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm_plt = plt.Normalize(speed.min(), speed.max())
    lc = LineCollection(segments, cmap='turbo', norm=norm_plt)
    lc.set_array(speed)
    lc.set_linewidth(4)
    
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax, label='Speed (m/s)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    # Formatting
    ax.set_title(f"Optimal Racing Line - Lap Time: {np.sum(track['segment_length']/(speed[:-1]+1e-3)):.2f}s", color='white')
    ax.axis('equal')
    ax.grid(False)
    ax.tick_params(colors='white')
    
    plt.show()

if __name__ == "__main__":
    main()