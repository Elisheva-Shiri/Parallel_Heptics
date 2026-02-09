import sys
import re
import argparse
from datetime import datetime
from typing import List, Optional
import statistics


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    return datetime.strptime(timestamp_str, "[%Y-%m-%d %H:%M:%S.%f]")


def extract_serial_out_timestamps(log_file_path: str) -> List[datetime]:
    """Extract all SERIAL OUT timestamps from the log file."""
    timestamps = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            if '[SERIAL OUT]' in line:
                # Extract timestamp (everything between first '[' and ']')
                match = re.match(r'(\[[\d\-: .]+\])', line)
                if match:
                    timestamp = parse_timestamp(match.group(1))
                    timestamps.append(timestamp)
    
    return timestamps


def calculate_time_differences(timestamps: List[datetime]) -> List[float]:
    """Calculate time differences in milliseconds between consecutive timestamps."""
    differences = []
    
    for i in range(1, len(timestamps)):
        diff = (timestamps[i] - timestamps[i-1]).total_seconds() * 1000  # Convert to ms
        differences.append(diff)
    
    return differences


def percentile(data: List[float], p: int) -> float:
    """Calculate percentile using linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = (p / 100) * (n - 1)
    
    if index.is_integer():
        return sorted_data[int(index)]
    else:
        lower = sorted_data[int(index)]
        upper = sorted_data[int(index) + 1]
        fraction = index - int(index)
        return lower + fraction * (upper - lower)


def analyze_serial_out_intervals(log_file_path: str, max_outlier_seconds: Optional[float] = None):
    """Analyze time intervals between SERIAL OUT logs."""
    timestamps = extract_serial_out_timestamps(log_file_path)
    
    if len(timestamps) < 2:
        print("Not enough SERIAL OUT entries found in the log file.")
        return
    
    differences = calculate_time_differences(timestamps)
    
    print(f"Found {len(timestamps)} SERIAL OUT entries")
    print(f"Total intervals: {len(differences)}\n")
    
    # Filter outliers if specified
    if max_outlier_seconds is not None:
        max_outlier_ms = max_outlier_seconds * 1000
        original_count = len(differences)
        differences = [d for d in differences if d <= max_outlier_ms]
        filtered_count = original_count - len(differences)
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} outlier(s) > {max_outlier_seconds}s ({max_outlier_ms}ms)")
            print(f"Analyzing {len(differences)} intervals after filtering\n")
    
    if len(differences) == 0:
        print("No intervals remaining after filtering.")
        return
    
    # Calculate statistics
    mean_val = statistics.mean(differences)
    p50_val = percentile(differences, 50)
    p70_val = percentile(differences, 70)
    p90_val = percentile(differences, 90)
    p99_val = percentile(differences, 99)
    
    # P99 excluding first and last intervals
    if len(differences) > 2:
        differences_trimmed = differences[1:-1]
        p99_trimmed_val = percentile(differences_trimmed, 99)
    else:
        p99_trimmed_val = None
    
    # Print results
    print("Time between consecutive SERIAL OUT logs (in milliseconds):")
    print("=" * 60)
    print(f"Mean:                                    {mean_val:>10.3f} ms")
    print(f"P50 (Median):                            {p50_val:>10.3f} ms")
    print(f"P70:                                     {p70_val:>10.3f} ms")
    print(f"P90:                                     {p90_val:>10.3f} ms")
    print(f"P99:                                     {p99_val:>10.3f} ms")
    if p99_trimmed_val is not None:
        print(f"P99 (excluding 1st and last intervals):  {p99_trimmed_val:>10.3f} ms")
    else:
        print(f"P99 (excluding 1st and last intervals):  N/A (not enough data)")
    
    print("\n" + "=" * 60)
    print(f"Min interval:                            {min(differences):>10.3f} ms")
    print(f"Max interval:                            {max(differences):>10.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze time intervals between SERIAL OUT log entries.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py logfile.log
  python script.py logfile.log --max-outlier 1.5
  python script.py logfile.log -m 2.0
        """
    )
    
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument(
        '-m', '--max-outlier',
        type=float,
        metavar='SECONDS',
        help='Filter out intervals larger than this many seconds (e.g., 1.5 for 1.5 seconds)'
    )
    
    args = parser.parse_args()
    
    analyze_serial_out_intervals(args.log_file, args.max_outlier)