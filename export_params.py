import sys
import os
import numpy as np

# Ensure project root directory is available in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the V2 loader primitive directly from your powertrain codebase
from powertrain.modes.advanced.active_set_classifier import load_classifier_v2

def format_1d_array(arr):
    """Format single dimensional bias/threshold vectors to standard C format."""
    return "{" + ", ".join(f"{x:.6f}f" for x in arr) + "}"

def format_2d_array(arr):
    """Format multi-dimensional weight matrices into nested row-major C blocks."""
    rows = []
    for row in arr:
        rows.append("    {" + ", ".join(f"{x:.6f}f" for x in row) + "}")
    return "{\n" + ",\n".join(rows) + "\n}"

def main():
    params_path = "models/active_set_classifier_v2.bytes"
    thresh_path = "models/active_set_thresholds_v2.npy"
    output_header = "gp_tv_weights.h"

    if not os.path.exists(params_path) or not os.path.exists(thresh_path):
        print(f"[ERROR] Missing weight binaries! Ensure you have generated your V2 files first.")
        return

    print("[Export] Loading JAX/Flax V2 Classifier Bundle...")
    # Properly initialize model architecture and restore parameter arrays
    clf_bundle = load_classifier_v2(params_path=params_path, thresh_path=thresh_path)
    params = clf_bundle.params
    thresholds = np.array(clf_bundle.thresholds)

    print("[Export] Generating formatted C headers and matrix arrays...")
    with open(output_header, "w") as f:
        f.write("#ifndef GP_TV_WEIGHTS_H\n")
        f.write("#define GP_TV_WEIGHTS_H\n\n")
        f.write("/* Automatically generated from Project-GP V2 Active-Set Classifier weights */\n\n")

        # Loop through each layer explicitly to structure the network hierarchy
        # V2 uses default Flax compact names: Dense_0, Dense_1, Dense_2, Dense_3
        layers = ["Dense_0", "Dense_1", "Dense_2", "Dense_3"]
        
        for idx, layer in enumerate(layers):
            w = np.array(params[layer]["kernel"])
            b = np.array(params[layer]["bias"])
            
            f.write(f"/* Layer {idx}: {layer} Matrix Configurations */\n")
            f.write(f"#define CLF_W{idx}_ROWS {w.shape[0]}\n")
            f.write(f"#define CLF_W{idx}_COLS {w.shape[1]}\n")
            f.write(f"static const float CLF_W{idx}[{w.shape[0]}][{w.shape[1]}] = {format_2d_array(w)};\n\n")
            f.write(f"static const float CLF_B{idx}[{b.shape[0]}] = {format_1d_array(b)};\n\n")

        # Calibrated Active-Set Threshold vectors
        f.write("/* Calibrated Sigmoid Activation Thresholds */\n")
        f.write(f"#define CLF_N_CONSTRAINTS {thresholds.shape[0]}\n")
        f.write(f"static const float CLF_THRESHOLDS[{thresholds.shape[0]}] = {format_1d_array(thresholds)};\n\n")

        f.write("#endif /* GP_TV_WEIGHTS_H */\n")

    print(f"[SUCCESS] Exported full explicit network params file to: {output_header}")

if __name__ == "__main__":
    main()