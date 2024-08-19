# Group the spectral peaks of geophone data by time and frequency and count the number of peaks in each group

# Imports
from os.path import join
from time import time
from pandas import Series
from pandas import concat, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo
from utils_basic import time2suffix
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_geo_spec_peaks, update_spectral_peak_group_counts
from utils_plot import plot_array_spec_peak_counts, save_figure

# Inputs
# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral peak detection
prom_threshold = 15.0
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200.0

# Grouping
count_threshold = 9

# Plotting
starttime_to_plot = starttime_geo
endtime_to_plot = endtime_geo

min_freq_plot = 0.0
max_freq_plot = 5.0

date_format = "%Y-%m-%d"
major_time_spacing = "1d"
num_minor_time_ticks = 4

example_counts = [10, 20, 30]

size_scale = 30

# Plotting
plot_results = True

# Process the detections of each station
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

# Print the inputs
print(f"### Tallying the spectral peaks of geophone data by time and frequency ###")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print(f"Downsample: {downsample}")

if downsample:
    print(f"Downsample factor: {downsample_factor}")

print(f"Reverse-bandwidth threshold: {rbw_threshold} 1/Hz")
print(f"Prominence threshold: {prom_threshold} dB")
print(f"Frequency range: {min_freq_peak} - {max_freq_peak} Hz")
print("")

print(f"Count threshold: {count_threshold}")
print("")

if plot_results:
    print(f"Plot the counts from {starttime_to_plot} to {endtime_to_plot} in the frequency range {min_freq_plot} to {max_freq_plot} Hz")

# Get the common time labels
print("Getting the common time labels...")
block_timing_dfs = []
for i, station in enumerate(stations):
    filename_in =  f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
    inpath = join(indir, filename_in)
    block_timing_df = read_hdf(inpath, "block_timings")
    block_timing_dfs.append(block_timing_df)

block_timing_df = concat(block_timing_dfs, axis = 0)
block_timing_df = block_timing_df.drop_duplicates()


# Save the time labels
filename_out = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}_count{count_threshold}.h5"
outpath = join(indir, filename_out)
block_timing_df.to_hdf(outpath, key = "block_timings", mode = "w")

# Iterate over the time labels
if plot_results:
    count_dfs_to_plot = []

for i, row in block_timing_df.iterrows():
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
        peak_df = read_geo_spec_peaks(inpath, time_labels = [time_label])

        if peak_df is None:
            print(f"No peak is found in {station} at {time_label}.")
            continue

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
    cum_count_df = cum_count_df[cum_count_df["count"] >= count_threshold]
    cum_count_df.reset_index(drop = True, inplace = True)
    print(f"Number of groups with counts over the threshold: {len(cum_count_df)}")

    # Save the counts
    cum_count_df.to_hdf(outpath, key = time_label, mode = "a")

    if plot_results:
        count_dfs_to_plot.append(cum_count_df.loc[(cum_count_df["time"] >= starttime_to_plot) & (cum_count_df["time"] <= endtime_to_plot) & (cum_count_df["frequency"] >= min_freq_plot) & (cum_count_df["frequency"] <= max_freq_plot)])

print("Done.")



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
    figname = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}_count{count_threshold}_{suffix_start}to{suffix_end}_freq{min_freq_plot:.2f}to{max_freq_plot:.2f}hz.png"

    save_figure(fig, figname)

