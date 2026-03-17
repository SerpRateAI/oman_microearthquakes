"""
Extract waveform snippets based on the STA/LTA trigger windows
"""
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv

from utils_basic import (
    GEO_STATIONS as stations,
    DETECTION_DIR as dirpath_detections,
    get_freq_limits_string
)
from utils_sta_lta import Snippets

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Input arguments
    parser = ArgumentParser()
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--buffer_start", type=float, default=0.005)
    parser.add_argument("--buffer_end", type=float, default=0.005)

    args = parser.parse_args()
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    buffer_start = args.buffer_start
    buffer_end = args.buffer_end

    # Process each station
    for station in stations:
        print(f"Processing station {station}...")
        
        # Read the STA/LTA triggers
        freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
        filename = f"sta_lta_detections_{freq_str}_{station}.csv"
        filename = join(dirpath_detections, filename)
        det_df = read_csv(filename)

        # Get the triggers
        triggers = detdf.loc[detdf["station"] == station, "trigger_time"].values

        # Load the snippets

        # Filter the snippets
        snippets = snippets.filter(min_freq_filter, max_freq_filter)
    