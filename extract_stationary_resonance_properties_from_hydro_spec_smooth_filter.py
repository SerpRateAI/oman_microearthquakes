"""
Extract the properties of a stationary resonance from the hydrophone spectral peaks using a smoothing filter
"""
###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import mean, abs    
from pandas import concat
from pandas import read_csv, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

# Function for selecting the peak with the frequency closest to the reference frequency
def select_peak_with_reference(peak_df, freq_ref, min_num_loc, max_freq_diff = 0.05):
    # Count how many rows are associated with each frequency
    freq_counts = peak_df["frequency"].value_counts()

    # Keep only frequencies with enough detections
    valid_freqs = freq_counts[freq_counts >= min_num_loc].index

    if len(valid_freqs) == 0:
        return None
    
    # Find the smallest frequency difference
    min_freq_diff = abs(valid_freqs - freq_ref).min()

    if min_freq_diff > max_freq_diff:
        return None

    # Find the frequency closest to the reference
    freq_closest = valid_freqs[abs(valid_freqs - freq_ref).argmin()]

    # Get the rows in the original DataFrame that match the selected frequency
    peak_df = peak_df[peak_df["frequency"] == freq_closest].copy()

    # Return the rows
    return peak_df

# Inputs
# Command-line arguments
parser = ArgumentParser(description="Extract the properties of a stationary resonance from the hydrophone spectrograms")
parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--filter_length", type=int, help="Filter length", default=10)
parser.add_argument("--station", type=str, help="Station name", default="A00")
parser.add_argument("--min_num_loc", type=int, default=3, help="Minimum number of locations that detect the peak")
parser.add_argument("--max_mean_db_hydro", type=float, default=-15.0, help="Maximum mean dB for excluding noisy windows")
parser.add_argument("--max_mean_db_geo", type=float, default=15.0, help="Maximum mean dB for excluding noisy windows")
parser.add_argument("--max_freq_diff", type=float, default=0.075, help="Maximum frequency difference for selecting the peak")

parser.add_argument("--window_length", type=float, default=300.0, help="Spectrogram window length in seconds")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction between adjacent windows")
parser.add_argument("--min_prom", type=float, default=15.0, help="Prominence threshold for peak detection")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Reverse bandwidth threshold for peak detection")

# Parse the command-line arguments
args = parser.parse_args()
mode_name = args.mode_name
filter_length = args.filter_length

station = args.station
min_num_loc = args.min_num_loc
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db_hydro = args.max_mean_db_hydro
max_mean_db_geo = args.max_mean_db_geo
max_freq_diff = args.max_freq_diff
# Print the inputs
print(f"Mode name: {mode_name}")
print(f"Station: {station}")
print(f"Minimum number of locations that detect the peak: {min_num_loc}")
print(f"Filter length: {filter_length}")
print(f"Maximum mean dB for excluding noisy windows on hydrophone data: {max_mean_db_hydro:.2f} dB")
print(f"Maximum mean dB for excluding noisy windows on geophone data: {max_mean_db_geo:.2f} dB")
print(f"Spectrogram window length: {window_length:.2f} s")
print(f"Overlap fraction: {overlap:.2f}")
print(f"Prominence threshold: {min_prom:.2f}")
print(f"Reverse bandwidth threshold: {min_rbw:.2f}")

# Read the resonance frequency range for extracting the stationary resonance properties
print("Reading the resonance frequency range...")
filename = f"stationary_resonance_freq_ranges_hydro.csv"
filepath = join(indir, filename)
freq_range_df = read_csv(filepath)

# Select the resonance frequency range 
min_freq_reson = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_extract"].values[0]
max_freq_reson = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_extract"].values[0]

# Print the resonance frequency range
print(f"Resonance frequency range: {min_freq_reson:.2f} - {max_freq_reson:.2f} Hz")

# Read the corresponding properties of the stationary resonance on the geophone
print("Reading the corresponding properties of the stationary resonance on the geophone...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db_geo)

filename = f"stationary_resonance_profile_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
geo_df = read_hdf(filepath, key = "profile")

start_time_geo = geo_df["time"].min()

# Read the hydrophone noise frequencies
print("Reading the hydrophone noise frequencies...")

filename = f"hydro_noise_to_mute_{mode_name}_{station}.csv"
filepath = join(indir, filename)
noise_df = read_csv(filepath)

# First find the peaks for Borehole A
peaks_reson_a_dfs = []

print("############################################")
print(f"Working on {station}")
print("############################################")

suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db_hydro)
filename = f"hydro_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)

# Read the block-timing data and reverse the order
block_timing_df = read_hdf(filepath, key="block_timing")
freq_interval = 1 / window_length

# Sort the block-timing data in descending order
block_timing_df = block_timing_df.sort_values(by = "start_time", ascending = False)

# Initialize the list of resonance peaks
property_dfs = []

# Initialize the list of frequencies for the smoothing filter
freqs_filter = []

# Loop over the time labels
for _, row in block_timing_df.iterrows():
    time_label = row["time_label"]
    start_time = row["start_time"]
    end_time = row["end_time"]

    print(f"Working on time label: {time_label}")

    # Read the peak data
    peak_df = read_hdf(filepath, key = time_label)

    # Select the peaks within the frequency range
    peak_df = peak_df[(peak_df["frequency"] >= min_freq_reson) & (peak_df["frequency"] <= max_freq_reson)]

    # Remove the peak if the frequency is different from the nosie frequency by less than the frequency interval
    peak_df = peak_df[peak_df["frequency"].apply(lambda x: all(abs(x - noise_df["frequency"]) > 2 *freq_interval))]

    # Sort the time groups in descending order
    peak_time_group = sorted(peak_df.groupby("time"), key=lambda x: x[0], reverse=True)

    # Process each time window
    for time_window, peak_time_df in peak_time_group:
        # Determine if the time window is within the geophone deployment
        if time_window >= start_time_geo:
            geo_time_win_df = geo_df.loc[ geo_df["time"] == time_window ]

            if len(geo_time_win_df) == 0:
                print(f"No geophone data is recorded for the time window {time_window.strftime('%Y-%m-%d %H:%M:%S')}. The window is skipped.")
                continue

            freq_geo = geo_time_win_df["frequency"].values[0]

            # Select the peak with the frequency closest to the geophone resonance frequency
            peak_selected_df = select_peak_with_reference(peak_time_df, freq_geo, min_num_loc, max_freq_diff = max_freq_diff)

        else:
            # Get the average of the frequency values in the filter
            freq_filter = mean(freqs_filter)

            # Select the peak with the frequency closest to the average frequency
            peak_selected_df = select_peak_with_reference(peak_time_df, freq_filter, min_num_loc, max_freq_diff = max_freq_diff)
            
        if peak_selected_df is None:
            print(f"No valid peak is found for the time window {time_window.strftime('%Y-%m-%d %H:%M:%S')}. The window is skipped.")
            continue

        # Add the time window to the peak properties
        peak_selected_df["station"] = station
        freq_hydro = peak_selected_df["frequency"].values[0]
        property_dfs.append(peak_selected_df)
        
        # Update the frequency filter
        freqs_filter.append(freq_hydro)

        if len(freqs_filter) > filter_length:
            freqs_filter.pop(0)

# Concatenate the peak properties
property_df = concat(property_dfs)
property_df.sort_values(["time", "location"], inplace = True)
property_df.reset_index(drop = True, inplace = True)

# Group the frequency values by time
profile_df = property_df.groupby("time").agg({"frequency": "mean", "power": "mean"}).reset_index()

# # Work on Borehole B
# peaks_reson_b_dfs = []

# print("############################################")
# print(f"Working on B00")
# print("############################################")

# filename = f"hydro_spectral_peaks_B00_{suffix_spec}_{suffix_peak}.h5"
# filepath = join(indir, filename)

# # Read the block-timing data
# block_timing_df = read_hdf(filepath, key="block_timing")

# # Process each time lable
# for time_label in block_timing_df["time_label"]:
#     print(f"Working on time label: {time_label}")

#     # Read the peak data
#     peak_df = read_hdf(filepath, key = time_label)

#     # Select the peaks within the frequency range
#     peak_df = peak_df[(peak_df["frequency"] >= min_freq_reson) & (peak_df["frequency"] <= max_freq_reson)]

#     # Group the peaks by time
#     peak_time_group = peak_df.groupby("time")

#     # Loop over the time groups
#     for time_window, peak_time_df in peak_time_group:
#         # Get the peaks recorded in the same time window in Borehole A
#         peaks_time_ref_df = peaks_reson_a_df[peaks_reson_a_df["time"] == time_window]

#         if len(peaks_time_ref_df) == 0:
#             continue

#         freq_ref = peaks_time_ref_df["frequency"].values[0]

#         # Select the resonance peak for one time window
#         min_num_loc = min_num_loc_dict["B00"]
#         peaks_reson_window_df = select_resonance_peak(peak_time_df, min_num_loc, freq_ref = freq_ref)

#         # If there is no peak, continue to the next time window
#         if peaks_reson_window_df is None:
#             continue

#         # print(peak_df[["location", "frequency", "power"]])

#         # Add the time window to the peak properties
#         peaks_reson_window_df["station"] = "B00"
#         freq_reson = peaks_reson_window_df["frequency"].values[0]
#         #print(f"Found {len(peaks_reson_window_df)} locations recording a peak at {freq_reson:.2f} Hz for the time window {time_window.strftime('%Y-%m-%d %H:%M:%S')}")
#         peaks_reson_b_dfs.append(peaks_reson_window_df)

# # Concatenate the peak properties
# peaks_reson_b_df = concat(peaks_reson_b_dfs)

# # Save the peak properties to CSV and HDF5 files
# peaks_reson_df = concat([peaks_reson_a_df, peaks_reson_b_df])
# peaks_reson_df.sort_values(["station", "location", "time"], inplace = True)
# peaks_reson_df.reset_index(drop = True, inplace = True)

# Save the peak properties to CSV and HDF5 files
filename = f"stationary_resonance_properties_hydro_smooth_filter_{mode_name}_{suffix_spec}_{suffix_peak}.csv"
filepath = join(indir, filename)
property_df.to_csv(filepath, index=False)
print(f"Saved the peak properties to {filepath}")

filename = f"stationary_resonance_properties_hydro_smooth_filter_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
property_df.to_hdf(filepath, key="properties", mode="w")
print(f"Saved the peak properties to {filepath}")

# Save the profile data
filename = f"stationary_resonance_profile_hydro_smooth_filter_{mode_name}_{suffix_spec}_{suffix_peak}.csv"
filepath = join(indir, filename)
profile_df.to_csv(filepath, index=False)
print(f"Saved the profile data to {filepath}")

filename = f"stationary_resonance_profile_hydro_smooth_filter_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
profile_df.to_hdf(filepath, key="profile", mode="w")
print(f"Saved the profile data to {filepath}")






