# Compute the whole deployment spectrograms for the hydrophone data
# The results for each station is stored in a single HDF5 file, with the spectrograms of all locations of each day saved in a separate block for easy reading and writing

# Imports
from os import makedirs
from os.path import join
from time import time

from utils_basic import SPECTROGRAM_DIR as outdir, HYDRO_LOCATIONS as loc_dict
from utils_basic import get_hydrophone_days, get_unique_locations
from utils_preproc import read_and_process_day_long_hydro_waveforms
from utils_spec import create_hydro_spectrogram_file, write_hydro_spectrogram_block, finish_hydro_spectrogram_file
from utils_torch import get_daily_hydro_spectrograms

# Inputs
stations_to_compute = ["A00", "B00"]
window_length = 300.0
overlap = 0.0
downsample = False # Downsample along the frequency axis
downsample_factor = 60 # Downsample factor for the frequency axis
resample_in_parallel = True # Resample along the time axis in parallel
num_process_resample = 32 # Number of processes while resampling along the time axis in parallel
save_ds_only = False # Whether to save only the downsampled spectrograms

if not downsample and save_ds_only:
    raise ValueError("Conflicting options for downsampling!")

# Create the output directory
makedirs(outdir, exist_ok=True)

# Get the hydrophone deployment days
days = get_hydrophone_days()

# Compute the frequency intervals
freq_interval = 1.0 / window_length
freq_interval_ds = freq_interval * downsample_factor

# Print the parameters
print("### Parameters ###")
print(f"Stations to compute: {stations_to_compute}")
print(f"Window length: {window_length}")
print(f"Overlap: {overlap}")
print(f"Downsample: {downsample}")
print(f"Downsample factor: {downsample_factor}")
print(f"Resample in parallel: {resample_in_parallel}")
print(f"Number of processes for resampling: {num_process_resample}")
print(f"Save downsampled only: {save_ds_only}")

# Process each station
for station in stations_to_compute:
    print("######")
    print(f"Processing {station}...")
    print("######")
    print("")

    locations = loc_dict[station]

    # Create the HDF5 files
    if not save_ds_only:
        file = create_hydro_spectrogram_file(station, locations,
                                             window_length = window_length, overlap = overlap,
                                             freq_interval = freq_interval, downsample = False, outdir = outdir)
    
    if downsample:
        file_ds = create_hydro_spectrogram_file(station, locations,
                                                window_length = window_length, overlap = overlap,
                                                freq_interval = freq_interval_ds, downsample = True, downsample_factor = downsample_factor, outdir = outdir)

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
        stream_spec, stream_spec_ds = get_daily_hydro_spectrograms(stream_day, 
                                                        window_length = window_length, overlap = overlap,
                                                        resample_in_parallel = resample_in_parallel, num_process_resample = num_process_resample,
                                                        downsample = downsample, downsample_factor = downsample_factor)
        time_labels.append(stream_spec[0].time_label)

        # Write the spectrogram blocks
        if not save_ds_only:
            write_hydro_spectrogram_block(file, stream_spec, locations, close_file = False)

        if downsample:
            write_hydro_spectrogram_block(file_ds, stream_spec_ds, locations, close_file = False)

        # Stop the clock
        clock2 = time()
        elapsed = clock2 - clock1
        print(f"Time taken: {elapsed:.1f} seconds")

    # Finish the file
    if not save_ds_only:
        finish_hydro_spectrogram_file(file, time_labels)

    if downsample:
        finish_hydro_spectrogram_file(file_ds, time_labels)

    print("")