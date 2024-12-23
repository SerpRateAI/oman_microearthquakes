# Compute the whole deployment STFT including both power and phase for the hydrophone data
# The results for each station is stored in a single HDF5 file, with the spectrograms of all locations of each day saved in a separate block for easy reading and writing

# Imports
from os import makedirs
from os.path import join
from argparse import ArgumentParser
from time import time

from utils_basic import SPECTROGRAM_DIR as outdir, GEO_STATIONS as stations, GEO_COMPONENTS as components
from utils_basic import get_geophone_days, get_unique_locations
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_spec import create_geo_stft_file, write_geo_stft_block, finish_geo_stft_file
from utils_torch import get_daily_stft

# Inputs
# Command line arguments
parser = ArgumentParser(description = "Compute the whole deployment power spectrograms for the hydrophone data")
parser.add_argument("--window_length", type = float, help = "Window length in seconds")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds")
parser.add_argument("--num_process", type = int, default = 1, help = "Number of processes for computing the spectrograms and resampling")

# Parse the arguments
args = parser.parse_args()
window_length = args.window_length
overlap = args.overlap
num_process = args.num_process

# Create the output directory
makedirs(outdir, exist_ok=True)

# Get the hydrophone deployment days
days = get_geophone_days()

# Compute the frequency intervals
freq_interval = 1.0 / window_length

# Print the parameters
print("### Parameters ###")
print(f"Window length: {window_length}")
print(f"Overlap: {overlap}")
print(f"Number of processes: {num_process}")

# Process each station
for station in stations:
    print("######")
    print(f"Processing {station}...")
    print("######")
    print("")

    # Create the HDF5 files
    file = create_geo_stft_file(station, 
                                window_length = window_length, overlap = overlap, freq_interval = freq_interval,
                                outdir = outdir)

    # Process each day
    time_labels = []
    for day in days:
        clock1 = time()
        print(f"### Processing {day} for {station}...")

        # Read and process the waveforms
        stream_day = read_and_process_day_long_geo_waveforms(day, stations = station)

        if stream_day is None:
            continue

        # Get the spectrograms
        print(f"Computing the STFT...")
        stream_stft = get_daily_stft(stream_day, window_length = window_length, overlap = overlap)
        # print(stream_stft[0].times[0], stream_stft[0].times[-1])
        time_labels.append(stream_stft[0].time_label)

        # Write the spectrogram blocks
        print(f"Writing the STFT block...")
        write_geo_stft_block(file, stream_stft)

        # Stop the clock
        clock2 = time()
        elapsed = clock2 - clock1
        print(f"Time taken: {elapsed:.1f} seconds")

    # Finish the file
    finish_geo_stft_file(file, time_labels)

    print("")