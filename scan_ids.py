import os
import collections

# POINT THIS TO YOUR EXACT LOG FILE
LOG_FILE = "data/logs/20250816_185811.asc"

def scan():
    # Fix path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, LOG_FILE)

    if not os.path.exists(full_path):
        print(f"Error: File not found at {full_path}")
        return

    print(f"Scanning {LOG_FILE} for CAN IDs...")
    
    id_counts = collections.Counter()
    id_samples = {} # Store one raw line per ID to help identify it

    with open(full_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Vector ASC format check: Needs timestamp, channel, ID, direction, 'd', len, data...
            # Example: "0.001 1 100 Rx d 8 01 02..."
            if len(parts) > 6 and 'd' in parts:
                try:
                    # ID is usually index 2
                    can_id_str = parts[2]
                    # Convert Hex string to Int
                    can_id = int(can_id_str.strip('x'), 16)
                    
                    id_counts[can_id] += 1
                    
                    # Save a sample of the data bytes if we haven't seen this ID yet
                    if can_id not in id_samples:
                        # Find where 'd' is, data starts 2 after 'd'
                        d_idx = parts.index('d')
                        data_bytes = parts[d_idx+2:]
                        id_samples[can_id] = " ".join(data_bytes)
                        
                except ValueError:
                    continue

    print(f"\n--- FOUND {len(id_counts)} UNIQUE IDs ---")
    print(f"{'HEX ID':<10} | {'DEC ID':<10} | {'COUNT':<8} | {'SAMPLE DATA (Hex)'}")
    print("-" * 60)
    
    # Sort by most frequent
    for can_id, count in id_counts.most_common(15):
        hex_id = hex(can_id)
        sample = id_samples.get(can_id, "")[:25] # Truncate if long
        print(f"{hex_id:<10} | {can_id:<10} | {count:<8} | {sample}...")

    print("\n[ACTION REQUIRED]")
    print("1. Identify which ID above is Wheel Speed, Steering, and GPS.")
    print("2. Update DBC_CONFIG in main.py with these HEX IDs.")

if __name__ == "__main__":
    scan()