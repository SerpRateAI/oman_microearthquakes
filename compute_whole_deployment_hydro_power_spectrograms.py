# Compute the whole deployment spectrograms for the hydrophone data
# The results for each station is stored in a single HDF5 file, with the spectrograms of all locations of each day saved in a separate block for easy reading and writing

# Imports
from os import makedirs
from os.path import join
from argparse import ArgumentParser
from time import time

from utils_basic import SPECTROGRAM_DIR as outdir, HYDRO_LOCATIONS as loc_dict
from utils_basic import get_hydrophone_days, get_unique_locations
from utils_preproc import read_and_process_day_long_hydro_waveforms
from utils_spec import create_hydro_power_spectrogram_file, write_hydro_power_spectrogram_block, finish_hydro_power_spectrogram_file
from utils_torch import get_daily_hydro_stft_psd

# Inputs
# Command line arguments
parser = ArgumentParser(description = "Compute the whole deployment power spectrograms for the hydrophone data")
parser.add_argument("--stations", nargs = "+", help = "Stations to compute the spectrograms for")
parser.add_argument("--window_length", type = float, help = "Window length in seconds")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds")
parser.add_argument("--num_process", type = int, default = 1, help = "Number of processes for computing the spectrograms and resampling")

# Parse the arguments
args = parser.parse_args()
stations_to_compute = args.stations
window_length = args.window_length
overlap = args.overlap
num_process = args.num_process

# Create the output directory
makedirs(outdir, exist_ok=True)

# Get the hydrophone deployment days
days = get_hydrophone_days()

# Compute the frequency intervals
freq_interval = 1.0 / window_length

# Print the parameters
print("### Parameters ###")
print(f"Stations to compute: {stations_to_compute}")
print(f"Window length: {window_length}")
print(f"Overlap: {overlap}")
print(f"Number of processes: {num_process}")

# Process each station
for station in stations_to_compute:
    print("######")
    print(f"Processing {station}...")
    print("######")
    print("")

    locations = loc_dict[station]

    # Create the HDF5 files
    file = create_hydro_power_spectrogram_file(station, locations,
                                             window_length = window_length, overlap = overlap,
                                             freq_interval = freq_interval, outdir = outdir)

    # Process each day
    time_labels = []
    for day in days:
        clock1 = time()
        print(f"### Processing {day} for {station}...")

        # Read and process the waveforms
        stream_day = read_and_process_day_long_hydro_waveforms(day, stations = station)

        # Determine if all locations are present
        locations_in = get_unique_locations(stream_day)
        if not all([loc in locations_in for loc in locations]):
            print("Not all locations are present!")
            continue

        # Get the spectrograms
        stream_spec = get_daily_hydro_stft_psd(stream_day, 
                                              window_length = window_length, overlap = overlap,
                                              num_process = num_process)
        # print(type(stream_spec[0].psd_mat[0, 0]))
        time_labels.append(stream_spec[0].time_label)

        # Write the spectrogram blocks
        write_hydro_power_spectrogram_block(file, stream_spec, locations, close_file = False)

        # Stop the clock
        clock2 = time()
        elapsed = clock2 - clock1
        print(f"Time taken: {elapsed:.1f} seconds")

    # Finish the file
    finish_hydro_power_spectrogram_file(file, time_labels)

    print("")