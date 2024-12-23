# Extract the properties of a stationary resonance from the hydrophone spectrograms
# Due to the higher noise levels in Borehole B, we first find the peaks for Borehole A and then use them as the reference for finding the peaks in Borehole B
# The spectral peaks caused by electrical noise are excluded from the analysis using a preselected list of frequencies


# Import the necessary libraries
from os.path import join
from argparse import ArgumentParser
from pandas import concat
from pandas import read_csv, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

# Function for selecting the resonance peak for one time window
def select_resonance_peak(peak_df, min_num_loc, freq_ref = None):

    # Group the peaks by frequency and count the number of locations in each group   
    peak_freq_group = peak_df.groupby("frequency")

    # Select the spectral peaks with number of locations greater than the threshold
    peak_freq_group = peak_freq_group.filter(lambda x: len(x) >= min_num_loc)

    # If there are no peaks, return None
    if len(peak_freq_group) == 0:
        return None

    # Determine if only one unique frequency is detected
    unique_freqs = peak_freq_group["frequency"].unique()

    if len(unique_freqs) == 1:
        peak_df = peak_freq_group
        
        # If the reference peaks are provided, check if the frequency is the same
        if freq_ref is not None:
            freq = peak_df["frequency"].values[0]

            # if abs(freq - freq_ref) > max_freq_diff:
            #     return None
            if freq != freq_ref:
                return None
    else:
        # If the reference peaks are provided, find the peaks with the same frequency
        if freq_ref is not None:
            # peak_freq_group["freq_diff"] = abs(peak_freq_group["frequency"] - freq_ref)
            # closest_freq_group = peak_freq_group.loc[peak_freq_group["freq_diff"].idxmin()]
            # freq = closest_freq_group["frequency"]
            # peak_df = peak_freq_group[peak_freq_group["frequency"] == freq]

            # if abs(freq - freq_ref) > max_freq_diff:
            #     return None

            if freq_ref not in unique_freqs:
                return None
            
            peak_df = peak_freq_group[peak_freq_group["frequency"] == freq_ref]
        # Find the frequency with the highest mean power
        else:
            mean_powers = peak_freq_group.groupby("frequency")["power"].mean()
            i_max = mean_powers.idxmax()  
            peak_df = peak_freq_group[peak_freq_group["frequency"] == i_max]

    return peak_df

# Inputs
# Command-line arguments
parser = ArgumentParser(description="Extract the properties of a stationary resonance from the hydrophone spectrograms")
parser.add_argument("--mode_name", type=str, help="Mode name")
parser.add_argument("--min_num_loc_dict", type=dict, default={"A00": 3, "B00": 4}, help="Minimum number of locations that detect the peak")
parser.add_argument("--max_mean_db", type=float, default=-10.0, help="Maximum mean dB for excluding noisy windows")

parser.add_argument("--window_length", type=float, default=60.0, help="Spectrogram window length in seconds")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction between adjacent windows")
parser.add_argument("--min_prom", type=float, default=15.0, help="Prominence threshold for peak detection")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Reverse bandwidth threshold for peak detection")
parser.add_argument("--min_freq_peak", type=float, default=0.0, help="Minimum frequency in Hz for peak detection")
parser.add_argument("--max_freq_peak", type=float, default=200.0, help="Maximum frequency in Hz for peak detection")

# Parse the command-line arguments
args = parser.parse_args()
mode_name = args.mode_name
min_num_loc_dict = args.min_num_loc_dict

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
min_freq_peak = args.min_freq_peak
max_freq_peak = args.max_freq_peak
max_mean_db = args.max_mean_db

# Print the inputs
print(f"Mode name: {mode_name}")
print(f"Minimum number of locations for peak detection: {min_num_loc_dict}")
print(f"Maximum mean dB for excluding noisy windows: {max_mean_db:.2f} dB")
print(f"Spectrogram window length: {window_length:.2f} s")
print(f"Overlap fraction: {overlap:.2f}")
print(f"Prominence threshold: {min_prom:.2f}")
print(f"Reverse bandwidth threshold: {min_rbw:.2f}")
print(f"Minimum frequency for peak detection: {min_freq_peak:.2f} Hz")
print(f"Maximum frequency for peak detection: {max_freq_peak:.2f} Hz")
print("")

# Read the resonance frequency range
print("Reading the resonance frequency range...")
filename = f"stationary_resonance_freq_ranges_hydro.csv"
filepath = join(indir, filename)
freq_range_df = read_csv(filepath)

# Select the resonance frequency range
min_freq_reson = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_extract"].values[0]
max_freq_reson = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_extract"].values[0]

# Print the resonance frequency range
print(f"Resonance frequency range: {min_freq_reson:.2f} - {max_freq_reson:.2f} Hz")

# Assemble the file suffices
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

# Read the hydrophone noise frequencies
print("Reading the hydrophone noise frequencies...")
filename = f"hydro_noise_to_mute_{mode_name}_A00.csv"
filepath = join(indir, filename)
noise_df = read_csv(filepath)

# First find the peaks for Borehole A
peaks_reson_a_dfs = []

print("############################################")
print(f"Working on A00")
print("############################################")

filename = f"hydro_spectral_peaks_A00_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
min_num_loc = min_num_loc_dict["A00"]

# Read the block-timing data
block_timing_df = read_hdf(filepath, key="block_timing")
freq_interval = 1 / window_length

# Process each time lable
for time_label in block_timing_df["time_label"]:
    print(f"Working on time label: {time_label}")

    # Read the peak data
    peak_df = read_hdf(filepath, key = time_label)

    # Select the peaks within the frequency range
    peak_df = peak_df[(peak_df["frequency"] >= min_freq_reson) & (peak_df["frequency"] <= max_freq_reson)]

    # Remove the peak if the frequency is different from the nosie frequency by less than the frequency interval
    peak_df = peak_df[peak_df["frequency"].apply(lambda x: all(abs(x - noise_df["frequency"]) > freq_interval / 2))]

    # Group the peaks by time
    peak_time_group = peak_df.groupby("time")

    # Loop over the time groups
    for time_window, peak_time_df in peak_time_group:
        # Select the resonance peak for one time window   
        peaks_reson_window_df = select_resonance_peak(peak_time_df, min_num_loc)

        # If there is no peak, continue to the next time window
        if peaks_reson_window_df is None:
            continue

        # print(peak_df[["location", "frequency", "power"]])

        # Add the time window to the peak properties
        peaks_reson_window_df["station"] = "A00"
        freq_reson = peaks_reson_window_df["frequency"].values[0]
        #print(f"Found {len(peaks_reson_window_df)} locations recording a peak at {freq_reson:.2f} Hz for the time window {time_window.strftime('%Y-%m-%d %H:%M:%S')}")
        peaks_reson_a_dfs.append(peaks_reson_window_df)

# Concatenate the peak properties
peaks_reson_a_df = concat(peaks_reson_a_dfs)


# Work on Borehole B
peaks_reson_b_dfs = []

print("############################################")
print(f"Working on B00")
print("############################################")

filename = f"hydro_spectral_peaks_B00_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)

# Read the block-timing data
block_timing_df = read_hdf(filepath, key="block_timing")

# Process each time lable
for time_label in block_timing_df["time_label"]:
    print(f"Working on time label: {time_label}")

    # Read the peak data
    peak_df = read_hdf(filepath, key = time_label)

    # Select the peaks within the frequency range
    peak_df = peak_df[(peak_df["frequency"] >= min_freq_reson) & (peak_df["frequency"] <= max_freq_reson)]

    # Group the peaks by time
    peak_time_group = peak_df.groupby("time")

    # Loop over the time groups
    for time_window, peak_time_df in peak_time_group:
        # Get the peaks recorded in the same time window in Borehole A
        peaks_time_ref_df = peaks_reson_a_df[peaks_reson_a_df["time"] == time_window]

        if len(peaks_time_ref_df) == 0:
            continue

        freq_ref = peaks_time_ref_df["frequency"].values[0]

        # Select the resonance peak for one time window
        min_num_loc = min_num_loc_dict["B00"]
        peaks_reson_window_df = select_resonance_peak(peak_time_df, min_num_loc, freq_ref = freq_ref)

        # If there is no peak, continue to the next time window
        if peaks_reson_window_df is None:
            continue

        # print(peak_df[["location", "frequency", "power"]])

        # Add the time window to the peak properties
        peaks_reson_window_df["station"] = "B00"
        freq_reson = peaks_reson_window_df["frequency"].values[0]
        #print(f"Found {len(peaks_reson_window_df)} locations recording a peak at {freq_reson:.2f} Hz for the time window {time_window.strftime('%Y-%m-%d %H:%M:%S')}")
        peaks_reson_b_dfs.append(peaks_reson_window_df)

# Concatenate the peak properties
peaks_reson_b_df = concat(peaks_reson_b_dfs)

# Save the peak properties to CSV and HDF5 files
peaks_reson_df = concat([peaks_reson_a_df, peaks_reson_b_df])
peaks_reson_df.sort_values(["station", "location", "time"], inplace = True)
peaks_reson_df.reset_index(drop = True, inplace = True)

filename = f"stationary_resonance_properties_{mode_name}_hydro_{suffix_spec}_{suffix_peak}.csv"
filepath = join(indir, filename)
peaks_reson_df.to_csv(filepath, index=True)
print(f"Saved the peak properties to {filepath}")

filename = f"stationary_resonance_properties_{mode_name}_hydro_{suffix_spec}_{suffix_peak}.h5"
filepath = join(indir, filename)
peaks_reson_df.to_hdf(filepath, key="properties", mode="w")
print(f"Saved the peak properties to {filepath}")



