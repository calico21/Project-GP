import matplotlib.pyplot as plt
import struct
import os

LOG_FILE = "data/logs/20250816_185811.asc"

def check_gps_and_speed():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, LOG_FILE)
    
    # Store data vectors
    t_gps = []
    lat_vals = [] # 0x573
    lon_vals = [] # 0x574
    
    t_speed = []
    speed_vals = [] # 0x400 (Engine RPM often) or 0x402 (Speed?)
    
    print(f"Analyzing {LOG_FILE}...")
    
    with open(full_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 6 and 'd' in parts:
                try:
                    ts = float(parts[0])
                    can_id = int(parts[2].strip('x'), 16)
                    d_idx = parts.index('d')
                    hex_bytes = parts[d_idx+2:]
                    
                    # Skip if not enough data
                    if len(hex_bytes) < 4: continue
                    
                    # Convert hex to bytes
                    data_raw = bytes([int(b, 16) for b in hex_bytes])
                    
                    # --- GPS CHECK (0x573 / 0x574) ---
                    # GPS is often sent as a 32-bit signed integer (scaled by 1e-7)
                    if can_id == 0x573:
                        val = struct.unpack('<i', data_raw[:4])[0] * 1e-7
                        # Sanity check: Lat should be -90 to 90
                        if -90 < val < 90 and val != 0:
                            lat_vals.append(val)
                            t_gps.append(ts)
                            
                    elif can_id == 0x574:
                        val = struct.unpack('<i', data_raw[:4])[0] * 1e-7
                        if -180 < val < 180 and val != 0:
                            lon_vals.append(val)

                    # --- SPEED CHECK (0x402) ---
                    # Let's try decoding as 16-bit UNSIGNED (Big Endian maybe?)
                    elif can_id == 0x402:
                        # Try Standard Little Endian Unsigned
                        val = struct.unpack('<H', data_raw[:2])[0] * 0.1
                        t_speed.append(ts)
                        speed_vals.append(val)

                except Exception:
                    continue

    # --- PLOTTING ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot Latitude
    if lat_vals:
        ax[0].plot(t_gps, lat_vals, label="Lat (0x573)")
        ax[0].set_title(f"GPS Latitude (0x573) - Avg: {sum(lat_vals)/len(lat_vals):.4f}")
        ax[0].grid(True)
    else:
        ax[0].text(0.5, 0.5, "No Valid Latitude Data Found", ha='center')

    # Plot Longitude
    if lon_vals:
        # Note: Time axis might not match perfectly len-wise if packets drop, 
        # but simplistic plotting is fine for visual check
        ax[1].plot(lon_vals, label="Lon (0x574)")
        ax[1].set_title(f"GPS Longitude (0x574) - Avg: {sum(lon_vals)/len(lon_vals):.4f}")
        ax[1].grid(True)

    # Plot Speed
    if speed_vals:
        ax[2].plot(t_speed, speed_vals, color='orange')
        ax[2].set_title("Potential Speed (0x402) - Decoded as Unsigned")
        ax[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_gps_and_speed()