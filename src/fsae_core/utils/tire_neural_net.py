import scipy.io
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os

# ==========================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==========================================
class TireGripNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Normal Load (Fz), Camber (IA), Pressure (P)]
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1) # Output: Peak Lateral Force (Fy)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. DATA LOADING (Smart Unwrap)
# ==========================================
def load_mat_data(bins_file):
    """
    Scans the .mat file for the data matrix, unwrapping structs if necessary.
    """
    print(f"📂 Loading {bins_file}...")
    
    if not os.path.exists(bins_file):
        print(f"❌ Error: File not found: {bins_file}")
        return None

    try:
        mat = scipy.io.loadmat(bins_file)
    except Exception as e:
        print(f"❌ Error reading .mat file: {e}")
        return None
    
    # --- Step 1: Find the Best Candidate Variable ---
    best_key = None
    max_score = 0
    data_raw = None

    print("   Scanning variables inside .mat file:")
    for key, val in mat.items():
        if key.startswith('__'): continue
        
        # Calculate a score to find the "main" data
        score = 0
        if isinstance(val, np.ndarray):
            score = val.size
            # Bonus score if it looks like a struct containing 'arr'
            if val.dtype.names and 'arr' in val.dtype.names:
                score += 100000 

        print(f"    - Found '{key}': Shape={val.shape}, Score={score}")
        
        if score > max_score:
            max_score = score
            best_key = key
            data_raw = val

    if data_raw is None:
        print("❌ Error: No valid data variables found.")
        return None

    print(f"   ✅ Selected variable: '{best_key}'")

    # --- Step 2: Unwrap 'arr' if present ---
    # This fixes the specific error you saw (fields 's0', 's1', 'arr')
    if data_raw.dtype.names and 'arr' in data_raw.dtype.names:
        print("   📦 Unwrapping 'arr' field from struct...")
        data_raw = data_raw['arr']
        # Unwrap if nested in (1,1) array
        if data_raw.shape == (1, 1):
            data_raw = data_raw[0, 0]

    # --- Step 3: Convert to DataFrame ---
    df = pd.DataFrame()
    
    # Case A: Structured Array (Named Columns)
    if data_raw.dtype.names:
        print("   ℹ️ Detected Table format (named columns).")
        target_len = 0
        field_data_map = {}
        for col in data_raw.dtype.names:
            try:
                item = data_raw[col]
                while isinstance(item, np.ndarray) and item.shape == (1, 1):
                    item = item[0, 0]
                val_arr = np.array(item).reshape(-1)
                field_data_map[col] = val_arr
                if len(val_arr) > target_len: target_len = len(val_arr)
            except: continue
        
        for col, arr in field_data_map.items():
            if len(arr) == target_len: df[col] = arr
            elif len(arr) == 1: df[col] = np.full(target_len, arr[0])
    
    # Case B: Raw Matrix (No Names) - The most likely case for 'arr'
    else:
        print("   ℹ️ Detected Matrix format (no column names). Applying index mapping...")
        df = pd.DataFrame(data_raw)
        
        # Map indices based on your text format:
        # Segment(0), Fz_Bin(1), FyMax_Th(2), ... IA_Mean(9), P_Mean(10)
        col_map = {1: 'Fz_Bin', 2: 'FyMax_Th', 9: 'IA_Mean', 10: 'P_Mean'}
        df = df.rename(columns=col_map)

    # --- Step 4: Validate & Clean ---
    required = ['Fz_Bin', 'FyMax_Th', 'IA_Mean', 'P_Mean']
    
    # Check for missing columns
    available = list(df.columns)
    missing = [c for c in required if c not in available]
    
    if missing:
        print(f"❌ Error: Missing columns {missing}")
        print(f"   Found columns: {available}")
        print("   (If indices shifted, adjust 'col_map' in the script)")
        return None

    df = df[required].dropna()
    df = df.astype(float)
    
    # Filter physical rows (Load > 50N to avoid noise)
    df = df[df['Fz_Bin'] > 50]
    
    print(f"✅ Successfully extracted {len(df)} rows of tire data.")
    return df

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train_grip_model(df):
    X = df[['Fz_Bin', 'IA_Mean', 'P_Mean']].values.astype(np.float32)
    y = df[['FyMax_Th']].values.astype(np.float32)

    # Normalization
    stats = {
        'X_mean': X.mean(axis=0), 'X_std': X.std(axis=0),
        'y_mean': y.mean(),       'y_std': y.std()
    }
    stats['X_std'][stats['X_std'] == 0] = 1.0
    if stats['y_std'] == 0: stats['y_std'] = 1.0
    
    X_norm = (X - stats['X_mean']) / stats['X_std']
    y_norm = (y - stats['y_mean']) / stats['y_std']
    
    X_tensor = torch.tensor(X_norm)
    y_tensor = torch.tensor(y_norm)
    
    model = TireGripNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("🧠 Training Neural Network...")
    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"   Epoch {epoch}: MSE Loss {loss.item():.5f}")

    return model, stats

# ==========================================
# 4. POLYNOMIAL FITTING
# ==========================================
def generate_solver_equation(df):
    """ Fits: Fy = c0*Fz + c1*Fz^2 + c2*IA + c3*IA^2 + c4*Fz*IA + c5 """
    # Filter for nominal pressure (median +/- 5)
    p_nom = df['P_Mean'].median()
    df_sub = df[(df['P_Mean'] > p_nom - 5) & (df['P_Mean'] < p_nom + 5)]
    if len(df_sub) < 20: df_sub = df
    
    Fz = df_sub['Fz_Bin'].values
    IA = df_sub['IA_Mean'].values
    Fy = df_sub['FyMax_Th'].values
    
    # Design Matrix
    A = np.column_stack([Fz, Fz**2, IA, IA**2, Fz*IA, np.ones_like(Fz)])
    
    # Least Squares
    coeffs, _, _, _ = np.linalg.lstsq(A, Fy, rcond=None)
    return coeffs

# ==========================================
# 5. VISUALIZATION
# ==========================================
def plot_results(model, stats, df):
    fz_sweep = np.linspace(df['Fz_Bin'].min(), df['Fz_Bin'].max(), 100)
    ia_levels = [0, 2, 4]
    p_nom = df['P_Mean'].median()
    
    plt.figure(figsize=(10, 6))
    
    # Plot Raw Data (near nom pressure)
    mask = np.abs(df['P_Mean'] - p_nom) < 5
    plt.scatter(df[mask]['Fz_Bin'], df[mask]['FyMax_Th'], c=df[mask]['IA_Mean'], cmap='viridis', alpha=0.5, label='Raw Data')
    plt.colorbar(label='Camber (deg)')
    
    # NN Predictions
    for ia in ia_levels:
        inputs = []
        for fz in fz_sweep:
            inputs.append([fz, ia, p_nom])
        inputs = np.array(inputs, dtype=np.float32)
        inputs_norm = (inputs - stats['X_mean']) / stats['X_std']
        
        with torch.no_grad():
            pred_norm = model(torch.tensor(inputs_norm)).numpy()
            
        pred = pred_norm * stats['y_std'] + stats['y_mean']
        plt.plot(fz_sweep, pred, linewidth=2, linestyle='--', label=f'NN (IA={ia}°)')
        
    plt.title(f"Tire Grip Model (Pressure ~{p_nom:.1f} kPa)")
    plt.xlabel("Normal Load Fz (N)")
    plt.ylabel("Peak Lateral Force Fy (N)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    bins_file = os.path.join(project_root, 'data', 'tire_data', 'master_all_bins.mat')
    
    try:
        df = load_mat_data(bins_file)
        if df is not None:
            model, stats = train_grip_model(df)
            c = generate_solver_equation(df)
            
            print("\n" + "="*60)
            print("✅ COMPLETED. COPY THIS INTO src/fsae_core/dynamics/vehicle_14dof.py")
            print("="*60)
            print(f"    def get_peak_grip(self, Fz, IA):")
            print(f"        # Derived from master_all_bins.mat")
            print(f"        # Fy = c0*Fz + c1*Fz^2 + c2*IA + c3*IA^2 + c4*Fz*IA + Bias")
            print(f"        return ({c[0]:.4f}*Fz) + ({c[1]:.4e}*Fz**2) + ({c[2]:.4f}*IA) + \\")
            print(f"               ({c[3]:.4f}*IA**2) + ({c[4]:.4e}*Fz*IA) + ({c[5]:.4f})")
            print("="*60)
            
            plot_results(model, stats, df)
            
    except Exception as e:
        print(f"\n❌ Script Failed: {e}")