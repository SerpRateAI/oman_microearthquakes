# Detect instrument noise spectral peaks from the hydrophone spectral-peak time cumulative array counts

# Import the required libraries
from os.path import join
from time import time
from scipy.signal import find_peaks
from argparse import ArgumentParser
from pandas import Timedelta
from pandas import concat
from pandas import read_csv, read_hdf
from matplotlib.pyplot import subplots

from utils_basic import HYDRO_LOCATIONS as location_dict, SPECTROGRAM_DIR as indir, STARTTIME_HYDRO as starttime, ENDTIME_HYDRO as endtime
from utils_basic import str2timestamp, time2suffix
from utils_spec import get_hydro_noise_file_suffix, get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import save_figure

# Inputs
# Command-line arguments
parser = ArgumentParser(description="Compute the time-cumulative counts of the spectral peaks detected in the hydrophone data")
parser.add_argument("--min_height", type=float, default=0.05, help="Minimum height in fraction for peak detection")
parser.add_argument("--min_threshold", type=float, default=0.03, help="Minimum threshold in fraction for peak detection")
parser.add_argument("--window_length", type=float, default=60.0, help="Spectrogram window length in seconds")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction between adjacent windows")
parser.add_argument("--min_prom", type=float, default=10.0, help="Minimum prominence in dB for peak detection")
parser.add_argument("--min_rbw", type=float, default=3.0, help="Minimum reverse bandwidth in 1/Hz for peak detection")
parser.add_argument("--max_mean_db", type=float, default=-10.0, help="Maximum mean dB for excluding noisy windows")

# Parse the command-line arguments
args = parser.parse_args()
min_height = args.min_height
min_threshold = args.min_threshold
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Constants
panel_width = 10
panel_height = 5.0

# Print the command-line arguments
print("### Computing the time-cumulative counts of the hydrophone spectral-peak array detections ###")
print("")
print("Command-line arguments:")
print(f"Minimum height: {min_height:.2f}")
print(f"Minimum threshold: {min_threshold:.2f}")
print(f"Window length: {window_length:.0f} s")
print(f"Overlap: {overlap:.0%}")
print(f"Minimum prominence: {min_prom:.0f} dB")
print(f"Minimum reverse bandwidth: {min_rbw:.0f} 1/Hz")
print("Maximum mean dB: ", max_mean_db)

# Read the cumulative counts
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
suffix_noise = get_hydro_noise_file_suffix(min_height, min_threshold)

cum_count_dfs = []
freq_peaks_dfs = []
for station in location_dict.keys():
    print(f"Working on {station}...")
    filename = f"hydro_spectral_peak_time_cum_freq_counts_{station}_{suffix_spec}_{suffix_peak}.csv"
    filepath = join(indir, filename)

    print(f"Reading the cumulative counts from {filepath:s}...")
    cum_count_df = read_csv(filepath)
    cum_count_df["station"] = station
    cum_count_dfs.append(cum_count_df)

    # Find the peaks in the cumulative counts
    print("Finding the noise peaks in the cumulative counts...")
    i_peaks, _ = find_peaks(cum_count_df["fraction"], height = min_height, threshold = min_threshold)
    print(f"Detected {len(i_peaks)} noise peaks.")

    freq_peaks = cum_count_df["frequency"][i_peaks]
    fraction_peaks = cum_count_df["fraction"][i_peaks]
    freq_peaks_df = freq_peaks.to_frame()
    freq_peaks_df["fraction"] = fraction_peaks
    freq_peaks_df.reset_index(drop = True, inplace = True)

    # Save the detected peaks 
    print("Saving the detected peaks...")
    outpath = join(indir, f"hydro_noise_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}_{suffix_noise}.csv")
    freq_peaks_df.to_csv(outpath, index =True)
    print(f"Saved the detected peaks to {outpath:s}.")

    freq_peaks_df["station"] = station
    freq_peaks_dfs.append(freq_peaks_df)

# Concatenate the cumulative counts and the peaks
cum_count_df = concat(cum_count_dfs, axis = 0)
freq_peaks_df = concat(freq_peaks_dfs, axis = 0)

# Plot the cumulative counts and the peaks
print("Plotting the cumulative counts and the peaks...")
fig, axes = subplots(nrows = 1, ncols = 2, figsize = (2 * panel_width, panel_height))

for i, station in enumerate(location_dict.keys()):
    cum_count_sta_df = cum_count_df[cum_count_df["station"] == station]
    freq_peaks_sta_df = freq_peaks_df[freq_peaks_df["station"] == station]

    ax = axes[i]
    ax.stem(cum_count_sta_df["frequency"], cum_count_sta_df["fraction"], linefmt = "black", markerfmt = "o", basefmt = "black")
    ax.scatter(freq_peaks_sta_df["frequency"], freq_peaks_sta_df["fraction"], color = "red", marker = "x", zorder = 2)

    ax.set_title(f"{station}", fontsize = 12, fontweight = "bold")
    ax.set_xlabel("Frequency [Hz]", fontsize = 12)
    ax.set_ylabel("Fraction", fontsize = 12)
    
fig.suptitle("Hydrophone noise spectral peaks", fontsize = 12, fontweight = "bold", y = 0.95)

# Save the figure
print("Saving the figure...")
figname = f"hydro_noise_spectral_peaks_{suffix_spec}_{suffix_peak}_{suffix_noise}.png"
save_figure(fig, figname)






