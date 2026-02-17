"""
GPS Data Scanner for Formula Student Telemetry Logs
Scans all .asc files in the logs directory and identifies which have valid GPS data.
"""

import os
import sys
from pathlib import Path

# Add project root to path so we can import the log ingestion module
# Adjust this path based on where you save this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from telemetry.log_ingestion import LogIngestion

# DBC Configuration for GPS parsing
DBC_CONFIG = {
    # GPS Position (Log: "119" → Parser: 0x119 = 281 dec)
    0x119: {'name': 'GPS_Lat_Long', 'signals': [
        {'name': 'Latitude',  'start_bit': 0,  'length': 32, 'factor': 1e-07, 'offset': 0, 'signed': True},
        {'name': 'Longitude', 'start_bit': 32, 'length': 32, 'factor': 1e-07, 'offset': 0, 'signed': True}
    ]},
    # Also parse steering to verify it's a real log
    0x5: {'name': 'STEER', 'signals': [
        {'name': 'ANGLE', 'start_bit': 0, 'length': 16, 'factor': 0.01, 'offset': 0, 'signed': True}
    ]},
}

def analyze_gps_in_file(file_path):
    """
    Analyze a single .asc file for GPS data quality.
    
    Returns:
        dict: {
            'has_gps': bool,
            'valid_points': int,
            'total_points': int,
            'lat_range': float,
            'lon_range': float,
            'lat_min': float,
            'lat_max': float,
            'lon_min': float,
            'lon_max': float,
            'is_valid': bool  # True if GPS data is usable
        }
    """
    try:
        # Parse the log file
        ingestor = LogIngestion(DBC_CONFIG)
        df = ingestor.parse_asc(file_path, resample_freq='20ms')
        
        if df.empty:
            return {'has_gps': False, 'is_valid': False, 'error': 'Empty dataframe'}
        
        # Check if GPS columns exist
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return {'has_gps': False, 'is_valid': False}
        
        # Analyze GPS data quality
        lat_valid = df['Latitude'].notna()
        lon_valid = df['Longitude'].notna()
        
        valid_points = (lat_valid & lon_valid).sum()
        total_points = len(df)
        
        if valid_points == 0:
            return {
                'has_gps': True,
                'valid_points': 0,
                'total_points': total_points,
                'is_valid': False
            }
        
        # Calculate ranges
        lat_vals = df.loc[lat_valid, 'Latitude']
        lon_vals = df.loc[lon_valid, 'Longitude']
        
        lat_min, lat_max = lat_vals.min(), lat_vals.max()
        lon_min, lon_max = lon_vals.min(), lon_vals.max()
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        # GPS is "valid" if coordinates are changing (not stationary)
        # Threshold: at least 0.0001 degrees (~11 meters) of movement
        is_valid = (lat_range > 0.0001 and lon_range > 0.0001 and valid_points > 100)
        
        return {
            'has_gps': True,
            'valid_points': valid_points,
            'total_points': total_points,
            'lat_range': lat_range,
            'lon_range': lon_range,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max,
            'is_valid': is_valid
        }
        
    except Exception as e:
        return {'has_gps': False, 'is_valid': False, 'error': str(e)}


def scan_logs_directory(logs_dir):
    """
    Scan all .asc files in the logs directory.
    
    Args:
        logs_dir: Path to the logs directory
        
    Returns:
        dict: Mapping of filename to GPS analysis results
    """
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"Error: Directory not found: {logs_dir}")
        return {}
    
    # Find all .asc files
    asc_files = list(logs_path.glob("*.asc"))
    
    if not asc_files:
        print(f"No .asc files found in {logs_dir}")
        return {}
    
    print(f"\n{'='*80}")
    print(f"Scanning {len(asc_files)} log files for GPS data...")
    print(f"{'='*80}\n")
    
    results = {}
    
    for i, file_path in enumerate(sorted(asc_files), 1):
        filename = file_path.name
        print(f"[{i}/{len(asc_files)}] Analyzing {filename}...", end=' ')
        
        result = analyze_gps_in_file(str(file_path))
        results[filename] = result
        
        if result['is_valid']:
            print("✓ VALID GPS")
        elif result['has_gps']:
            print("✗ GPS data present but invalid (stationary or insufficient)")
        else:
            print("✗ No GPS data")
    
    return results


def print_summary(results):
    """Print a summary of the scan results."""
    
    # Separate files by GPS status
    valid_gps = {k: v for k, v in results.items() if v['is_valid']}
    invalid_gps = {k: v for k, v in results.items() if v['has_gps'] and not v['is_valid']}
    no_gps = {k: v for k, v in results.items() if not v['has_gps']}
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total files scanned: {len(results)}")
    print(f"  ✓ Files with VALID GPS:   {len(valid_gps)}")
    print(f"  ✗ Files with invalid GPS: {len(invalid_gps)}")
    print(f"  ✗ Files without GPS:      {len(no_gps)}")
    
    # Detailed results for valid GPS files
    if valid_gps:
        print(f"\n{'='*80}")
        print("FILES WITH VALID GPS DATA:")
        print(f"{'='*80}\n")
        
        for filename, data in sorted(valid_gps.items()):
            print(f"\n📍 {filename}")
            print(f"   Valid GPS points: {data['valid_points']:,} / {data['total_points']:,}")
            print(f"   Latitude range:   {data['lat_range']:.6f}° ({data['lat_min']:.6f}° to {data['lat_max']:.6f}°)")
            print(f"   Longitude range:  {data['lon_range']:.6f}° ({data['lon_min']:.6f}° to {data['lon_max']:.6f}°)")
            
            # Estimate distance traveled (very rough)
            # 1 degree latitude ≈ 111 km
            # 1 degree longitude ≈ 111 km * cos(latitude) 
            lat_dist = data['lat_range'] * 111000  # meters
            lon_dist = data['lon_range'] * 111000  # meters (approximation)
            approx_dist = (lat_dist**2 + lon_dist**2)**0.5
            print(f"   Estimated distance: ~{approx_dist:.0f}m")
    else:
        print("\n⚠️  No files with valid GPS data found.")
        print("   Make sure GPS had satellite lock during recording.")
    
    # Show invalid GPS files with details
    if invalid_gps:
        print(f"\n{'='*80}")
        print("FILES WITH GPS DATA BUT INVALID (Likely Stationary):")
        print(f"{'='*80}\n")
        
        for filename, data in sorted(invalid_gps.items()):
            print(f"   {filename}")
            if 'lat_range' in data:
                print(f"      Range: Lat {data['lat_range']:.6f}°, Lon {data['lon_range']:.6f}°")


if __name__ == "__main__":
    # Default path - adjust if needed
    logs_directory = r"C:\Users\alexr\Desktop\Ter26\Ter26 Carmaker\FS_Driver_Setup_Optimizer\data\logs"
    
    # Allow command line argument for custom path
    if len(sys.argv) > 1:
        logs_directory = sys.argv[1]
    
    results = scan_logs_directory(logs_directory)
    
    if results:
        print_summary(results)
        
        # Suggest best files
        valid_files = [f for f, r in results.items() if r['is_valid']]
        if valid_files:
            print(f"\n{'='*80}")
            print("RECOMMENDED FILES FOR ANALYSIS:")
            print(f"{'='*80}")
            for f in valid_files:
                print(f"  • {f}")
            print()