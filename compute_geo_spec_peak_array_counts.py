# Group the spectral peaks of geophone data by time and frequency and count the number of peaks in each group

# Imports
from os.path import join
from argparse import ArgumentParser
from time import time
from pandas import Series
from pandas import concat, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo
from utils_basic import time2suffix
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_geo_spec_peaks, update_spectral_peak_group_counts
from utils_plot import plot_array_spec_peak_counts, save_figure

# Inputs
# Command line arguments
parser = ArgumentParser(description = "Tally the spectral peaks of geophone data at each time and frequency point")
parser.add_argument("--window_length", type = float, default = 60.0, help = "Window length in seconds")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence in dB")
parser.add_argument("--min_rbw", type = float, default = 15.0, help = "Minimum reverse bandwidth in 1/Hz")
parser.add_argument("--max_mean_db", type = float, default = 10.0, help = "Maximum mean dB for excluding noisy windows")

parser.add_argument("--min_count", type = int, default = 9, help = "Minimum count of peaks in a group")
parser.add_argument("--plot_results", action = "store_true", help = "Plot the results")

# Parse the arguments
args = parser.parse_args()
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

min_count = args.min_count

plot_results = args.plot_results

# Constants
starttime_to_plot = starttime_geo
endtime_to_plot = endtime_geo

min_freq_plot = 37.8
max_freq_plot = 38.8

date_format = "%Y-%m-%d"
major_time_spacing = "1d"
num_minor_time_ticks = 4

example_counts = [10, 20, 30]

size_scale = 30

# Process the detections of each station
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

# Print the inputs
print(f"### Tallying the spectral peaks of geophone data by time and frequency ###")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")

print(f"Minimum reverse-bandwidth threshold: {min_rbw} 1/Hz")
print(f"Minimum prominence threshold: {min_prom} dB")
print("")

print(f"Minimum count to include in the output {min_count}")
print("")

if plot_results:
    print(f"Plot the counts from {starttime_to_plot} to {endtime_to_plot} in the frequency range {min_freq_plot} to {max_freq_plot} Hz")

# Get the common time labels
print("Getting the common time labels...")
block_timing_dfs = []
for i, station in enumerate(stations):
    filename_in =  f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
    inpath = join(indir, filename_in)
    block_timing_df = read_hdf(inpath, "block_timing")
    block_timing_dfs.append(block_timing_df)

block_timing_df = concat(block_timing_dfs, axis = 0)
block_timing_in_df = block_timing_df.drop_duplicates()

filename_out = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}.h5"
outpath = join(indir, filename_out)

# Iterate over the time labels
if plot_results:
    count_dfs_to_plot = []

time_labels_out = []
for i, row in block_timing_in_df.iterrows():
    time_label = row["time_label"]
    print(f"Working on time label {time_label}...")

    # Iterate over the stations
    i = 0
    for station in stations:
        # Read the spectral peaks
        clock1 = time()

        print(f"Working on {station}...")
        filename_in =  f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
        inpath = join(indir, filename_in)

        # Read the spectral peaks
        peak_df = read_geo_spec_peaks(inpath, time_labels = time_label)

        if peak_df is None:
            print(f"No peak is found in {station} at {time_label}.")
            continue

        time_labels_out.append(time_label)

        num_peaks = len(peak_df)
        print(f"{num_peaks} peaks are read.")
        
        # Update the spectral-peak group count
        print("Updating the group counts...")
        if i == 0:
            cum_count_df = update_spectral_peak_group_counts(peak_df)
        else:
            cum_count_df = update_spectral_peak_group_counts(peak_df, counts_to_update = cum_count_df)
        print("Update finished")

        i += 1

        clock2 = time()
        elapse = clock2 - clock1
        print(f"Finished processing {station}.")
        print(f"Elapsed time: {elapse} s")
        print("")

    # Keep only the groups with counts over the threshold
    cum_count_df["count"] = cum_count_df["count"].astype(int)
    cum_count_df = cum_count_df[cum_count_df["count"] >= min_count]
    cum_count_df.reset_index(drop = True, inplace = True)
    print(f"Number of groups with counts over the threshold: {len(cum_count_df)}")

    # Save the counts
    cum_count_df.to_hdf(outpath, key = time_label, mode = "a")

    if plot_results:
        count_dfs_to_plot.append(cum_count_df.loc[(cum_count_df["time"] >= starttime_to_plot) & (cum_count_df["time"] <= endtime_to_plot) & (cum_count_df["frequency"] >= min_freq_plot) & (cum_count_df["frequency"] <= max_freq_plot)])

print("Done.")

# Save the block timings
print("Saving the block timings...")
block_timing_out_df = block_timing_in_df.loc[block_timing_in_df["time_label"].isin(time_labels_out)]
block_timing_out_df.to_hdf(outpath, key = "block_timing", mode = "a")

# Plot the bin counts in the example time range
print("Plotting the counts...")
if plot_results:
    count_df_to_plot = concat(count_dfs_to_plot, axis = 0)

    fig, ax = plot_array_spec_peak_counts(count_df_to_plot,
                                        size_scale = size_scale, 
                                        example_counts = example_counts,
                                        starttime = starttime_to_plot, endtime = endtime_to_plot, min_freq = min_freq_plot, max_freq = max_freq_plot,
                                        date_format = date_format, major_time_spacing = major_time_spacing, num_minor_time_ticks = num_minor_time_ticks)

    # Save the plot
    suffix_start = time2suffix(starttime_to_plot)
    suffix_end = time2suffix(endtime_to_plot)
    figname = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}_{suffix_start}to{suffix_end}_freq{min_freq_plot:.2f}to{max_freq_plot:.2f}hz.png"

    save_figure(fig, figname)

