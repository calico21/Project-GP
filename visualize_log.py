import pandas as pd
import matplotlib.pyplot as plt
import struct
import os

# --- CONFIG ---
LOG_FILE = "data/logs/20250816_185811.asc"

# IDs to investigate (Based on your scan)
CANDIDATES = {
    0x5:   "Potential Steering (0x5)",
    0x402: "Potential Speed (0x402)",
    0x403: "Potential Speed Alt (0x403)",
}

def parse_and_plot():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, LOG_FILE)
    
    data = {cid: {'t': [], 'val': []} for cid in CANDIDATES}
    gps_candidates = {} # Store low-frequency IDs to find GPS

    print(f"Reading {LOG_FILE}...")
    
    with open(full_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 6 and 'd' in parts:
                try:
                    ts = float(parts[0])
                    can_id = int(parts[2].strip('x'), 16)
                    d_idx = parts.index('d')
                    hex_bytes = parts[d_idx+2:]
                    
                    # 1. Parse Specific Candidates
                    if can_id in CANDIDATES:
                        # Parse first 2 bytes as 16-bit Integer (Standard for Speed/Steer)
                        # Little Endian (<h)
                        if len(hex_bytes) >= 2:
                            raw_bytes = bytes([int(b, 16) for b in hex_bytes[:2]])
                            val = struct.unpack('<h', raw_bytes)[0] # Signed short
                            data[can_id]['t'].append(ts)
                            data[can_id]['val'].append(val)

                    # 2. Search for GPS (Look for counts between 1k and 50k)
                    # We store just the count for now
                    if can_id not in data:
                        gps_candidates[can_id] = gps_candidates.get(can_id, 0) + 1
                        
                except ValueError:
                    continue

    # --- PLOT 1: Speed & Steering candidates ---
    fig, axes = plt.subplots(len(CANDIDATES), 1, figsize=(10, 8), sharex=True)
    
    for i, (cid, label) in enumerate(CANDIDATES.items()):
        ax = axes[i] if len(CANDIDATES) > 1 else axes
        if len(data[cid]['t']) > 0:
            ax.plot(data[cid]['t'], data[cid]['val'])
            ax.set_title(f"{label} - Raw Value")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center')

    print("\n--- GPS SEARCH ---")
    print("Looking for IDs with count between 1000 and 60,000 (Typical for GPS 10Hz-20Hz)")
    print(f"{'ID (HEX)':<10} | {'COUNT':<10}")
    print("-" * 30)
    for cid, count in gps_candidates.items():
        if 1000 < count < 60000:
             print(f"{hex(cid):<10} | {count:<10}")

    print("\nCheck the popup graphs:")
    print("1. Does 0x5 look like Steering? (Should go + / - around 0)")
    print("2. Does 0x402 look like Speed? (Should only be positive, goes up to ~200-500)")
    plt.show()

if __name__ == "__main__":
    parse_and_plot()