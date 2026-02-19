"""
GPS Data Scanner for Formula Student Telemetry Logs.
Updates: Uses the new LogIngestion class (Project-GP v2.0).
"""

import os
import sys
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from telemetry.log_ingestion import LogIngestion

def analyze_gps_in_file(file_path):
    """
    Analyze a single file for GPS data quality using the new Ingestor.
    """
    try:
        # NEW API CALL:
        ingestor = LogIngestion(file_path)
        df = ingestor.process()
        
        if df.empty:
            return {'has_gps': False, 'is_valid': False, 'error': 'Empty dataframe'}
        
        # Check if GPS columns exist (LogIngestion normalizes them to 'lat', 'lon')
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return {'has_gps': False, 'is_valid': False}
        
        # Analyze GPS data quality
        # Filter out 0.0 coordinates
        valid_mask = (df['lat'] != 0.0) & (df['lon'] != 0.0)
        valid_points = valid_mask.sum()
        total_points = len(df)
        
        if valid_points < 10:
            return {
                'has_gps': True,
                'valid_points': valid_points,
                'total_points': total_points,
                'is_valid': False
            }
        
        # Calculate ranges
        lat_vals = df.loc[valid_mask, 'lat']
        lon_vals = df.loc[valid_mask, 'lon']
        
        lat_range = lat_vals.max() - lat_vals.min()
        lon_range = lon_vals.max() - lon_vals.min()
        
        # Validity Threshold: Moving at least ~10m
        is_valid = (lat_range > 0.0001 and lon_range > 0.0001)
        
        return {
            'has_gps': True,
            'valid_points': valid_points,
            'total_points': total_points,
            'lat_range': lat_range,
            'lon_range': lon_range,
            'is_valid': is_valid
        }
        
    except Exception as e:
        return {'has_gps': False, 'is_valid': False, 'error': str(e)}

def scan_logs_directory(logs_dir):
    logs_path = Path(logs_dir)
    
    # Scan both .asc and .csv
    files = list(logs_path.glob("*.asc")) + list(logs_path.glob("*.csv"))
    
    if not files:
        print(f"No log files found in {logs_dir}")
        return {}
    
    print(f"\nScanning {len(files)} log files...")
    results = {}
    
    for i, file_path in enumerate(sorted(files), 1):
        filename = file_path.name
        print(f"[{i}/{len(files)}] {filename}...", end=' ')
        
        result = analyze_gps_in_file(str(file_path))
        results[filename] = result
        
        if result.get('is_valid'):
            print("✓ VALID")
        elif result.get('has_gps'):
            print("✗ INVALID (Stationary)")
        else:
            print("✗ NO DATA")
    
    return results

if __name__ == "__main__":
    # Auto-detect logs folder relative to this script
    default_logs = os.path.join(script_dir, 'data', 'logs')
    
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = default_logs
        
    scan_logs_directory(target_dir)