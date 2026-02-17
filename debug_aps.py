# debug_gps_scan.py
import struct

LOG_FILE = "data/logs/20250816_185811.asc"
TARGET_ID = 0x119 

print(f"Scanning {LOG_FILE} for NON-ZERO ID {hex(TARGET_ID)}...")

with open(LOG_FILE, 'r') as f:
    found_valid = False
    for line in f:
        parts = line.strip().split()
        if len(parts) < 6: continue
        
        try:
            can_id = int(parts[2].strip('x'), 16)
        except: continue
            
        if can_id == TARGET_ID and 'd' in parts:
            d_idx = parts.index('d')
            hex_bytes = parts[d_idx+2:]
            
            # Check if ANY byte is non-zero
            if any(b != '00' for b in hex_bytes):
                print(f"FOUND DATA: {hex_bytes}")
                found_valid = True
                break # Found one, stop.

    if not found_valid:
        print("Scanned entire file. ALL GPS DATA IS ZERO.")