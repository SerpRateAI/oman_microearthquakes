# Find the spectral peaks due to hydrodynamic instrumental noise

# Import the necessary libraries
from os.path import join
from time import time
from argparse import ArgumentParser
from pandas import concat
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import HYDRO_LOCATIONS as location_dict, SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
# Command-line arguments
parser = ArgumentParser(description="Computing the spectral peak array detections for hydrophone data")
parser.add_argument("--window_length", type=float, default=300.0, help="Spectrogram window length in seconds")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction between adjacent windows")
parser.add_argument("--min_prom", type=float, default=15.0, help="Prominence threshold for peak detection")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Reverse bandwidth threshold for peak detection")
parser.add_argument("--min_freq", type=float, default=0.0, help="Minimum frequency in Hz for peak detection")
parser.add_argument("--max_freq", type=float, default=200.0, help="Maximum frequency in Hz for peak detection")
parser.add_argument("--max_mean_db", type=float, default=-15.0, help="Maximum mean dB for excluding noisy windows")

parser.add_argument("--time_label_to_plot", type=str, default="day_20191201", help="Time label to plot")

# Parse the command-line arguments
args = parser.parse_args()
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
min_freq = args.min_freq
max_freq = args.max_freq
max_mean_db = args.max_mean_db

time_label_to_plot = args.time_label_to_plot

# Constants
panel_width = 7.5
panel_height = 4.0

marker_size = 0.1

# Print the command-line arguments
print("### Finding the spectral peaks due to hydrodynamic instrumental noise ###")
print(f"Window length: {window_length:.0f} s")
print(f"Overlap: {overlap:.0%}")
print(f"Minimum prominence {min_prom:.0f} dB")
print(f"Minimum reverse bandwidth: {min_rbw:.0f} 1/Hz")
print(f"Minimum frequency: {min_freq:.0f} Hz")
print(f"Maximum frequency: {max_freq:.0f} Hz")
print(f"Time label to plot: {time_label_to_plot}")
print("")

# Read the data
# Assemble the file suffices
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

# Loop over the hydrophone stations
array_detect_to_plot_dfs = []
for station, locations in location_dict.items():
    print("################################################################")
    print(f"# Working on station {station}...")
    print("################################################################")

    filename = f"hydro_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
    inpath = join(indir, filename)

    # Read the block timing data
    print(f"Reading the block timing data from {inpath}...")
    block_timing_df = read_hdf(inpath, key = "block_timing")

    num_loc = len(locations)

    # Save the block timing data
    print("Saving the block timing data...")
    filename_out = f"hydro_spectral_peak_array_counts_{station}_{suffix_spec}_{suffix_peak}.h5"
    outpath = join(indir, filename_out)
    block_timing_df.to_hdf(outpath, key = "block_timing", mode = "w")

    # Loop over the time labels
    for time_label in block_timing_df["time_label"]:
        clock1 = time()
        # Read the spectral peaks of the current time label
        print(f"Reading the spectral peaks for time label {time_label}...")
        peak_df = read_hdf(inpath, key = time_label)
        print(f"Read {len(peak_df)} peaks.")

        # Group the peaks by frequency and time
        print("Grouping the peaks by frequency and time...")
        peak_group = peak_df.groupby(["frequency", "time"])

        # Count the number of locations for each group
        print("Counting the number of locations for each group...")
        peak_count = peak_group.size().reset_index(name = "count")

        # Filter the groups by the number of locations
        print("Filtering the groups by the number of locations...")
        peak_count = peak_count[peak_count["count"] == num_loc]
        print(f"Found {len(peak_count)} groups recorded by all {num_loc} locations.")

        # Record the time and frequency of the groups
        print("Recording the time and frequency of the groups...")
        array_detect_df = peak_count[["time", "frequency"]]
        array_detect_df["station"] = station

        # Save the array detection results for plotting
        if time_label == time_label_to_plot:
            print("Saving the array detection results for plotting...")
            array_detect_to_plot_dfs.append(array_detect_df)

        # Save the array detection results to file
        print("Saving the array detection results...")
        outpath = join(indir, filename_out)
        array_detect_df.to_hdf(outpath, key = time_label, mode = "a")
        print(f"Array detection results saved to {outpath}") 

        clock2 = time()
        elapse = clock2 - clock1
        print(f"Elapsed time: {elapse} s")
        print("")

# Concatenate the dataframes for plotting
print("Concatenating the dataframes for plotting...")
array_detect_to_plot_df = concat(array_detect_to_plot_dfs, ignore_index = True)

# Plot the array detections
print("Plotting the array detections...")
fig, axes = subplots(1, 2, figsize = (panel_width * 2, panel_height), sharex = True, sharey = True)

for i, station in enumerate(location_dict.keys()):
    ax = axes[i]
    station_array_detect_df = array_detect_to_plot_df[array_detect_to_plot_df["station"] == station]
    ax.scatter(station_array_detect_df["time"], station_array_detect_df["frequency"], s = marker_size, marker = "o", facecolors = "black", edgecolors = "none")
    ax.set_title(f"{station}", fontsize = 12, fontweight = "bold")

    if i == 0:
        format_datetime_xlabels(ax,
                                major_tick_spacing = "6h", num_minor_ticks = 6,
                                date_format = "%Y-%m-%d %H:%M:%S",
                                va = "top", ha = "right",
                                rotation = 15.0)
    else:
        format_datetime_xlabels(ax,
                                plot_axis_label = False,
                                plot_tick_label = True,
                                major_tick_spacing = "6h", num_minor_ticks = 6,
                                date_format = "%Y-%m-%d %H:%M:%S",
                                va = "top", ha = "right",
                                rotation = 15.0)

    format_freq_ylabels(ax, major_tick_spacing = 50.0, num_minor_ticks = 5)

fig.suptitle(f"Hydrophone spectral-peak array detections for {time_label_to_plot}", y = 0.97, fontsize = 12, fontweight = "bold")

# Save the plot
print("Saving the plot...")
figname = f"hydro_spectral_peak_array_detections_{suffix_spec}_{suffix_peak}_{time_label_to_plot}.png"
save_figure(fig, figname)

