# Group the spectral peaks of hydrophone data by time and frequency and count the number of peaks in each group

# Imports
from os.path import join
from time import time
from pandas import concat, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir, HYDRO_LOCATIONS as location_dict, STARTTIME_HYDRO as starttime, ENDTIME_HYDRO as endtime
from utils_basic import time2suffix
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_hydro_spec_peaks, save_spectral_peak_counts, update_spectral_peak_group_counts
from utils_plot import plot_array_spec_peak_times_and_freqs_in_multi_rows, save_figure

# Inputs
# Data
station = "A00"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60
prom_threshold = 10
rbw_threshold = 3.0

min_freq = None
max_freq = 200.0

# Grouping
count_thr_dict = {"A00": 2, "B00": 3}

# Plotting
starttime_plot = starttime
endtime_plot = endtime

min_freq_plot = 25.40
max_freq_plot = 25.60

date_format = "%Y-%m-%d %H:%M:%S"
major_time_spacing = "6h"
minor_time_spacing = "15h"

# Process the detections of each station
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)

# Initialize the spectral-peak group count
print(f"Working on {station}...")


# Read the time labels and locations
print("Reading the time labels...")
filename_in =  f"hydro_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename_in)
time_label_sr = read_hdf(inpath, "time_label")
time_labels = time_label_sr.values

# Process every time label
locations = location_dict[station]
count_threshold = count_thr_dict[station]
filename_out = f"hydro_spectral_peak_counts_{station}_{suffix_spec}_{suffix_peak}_count{count_threshold}.h5"
outpath = join(indir, filename_out)
time_label_sr.to_hdf(outpath, key = "time_label", mode = "w")

count_dfs_to_plot = []
for time_label in time_labels:
    clock1 = time()
    print(f"Processing the time label: {time_label}")

    for i, location in enumerate(locations):
        print(f"Processing the location: {location}")
        peak_df = read_hydro_spec_peaks(inpath, time_labels = [time_label], locations = [location])
        print(f"Number of peaks: {len(peak_df)}")
    
        # Update the spectral-peak group count
        print("Updating the group counts...")
        if i == 0:
            cum_count_df = update_spectral_peak_group_counts(peak_df)
        else:
            cum_count_df = update_spectral_peak_group_counts(peak_df, counts_to_update = cum_count_df)
        print("Update finished")

    clock2 = time()
    elapse = clock2 - clock1
    print(f"Finished processing {station}.")
    print(f"Elapsed time: {elapse} s")

    # Keep only the groups with counts over the threshold
    cum_count_df["count"] = cum_count_df["count"].astype(int)
    cum_count_df = cum_count_df[cum_count_df["count"] >= count_threshold]
    cum_count_df.reset_index(drop = True, inplace = True)
    print(f"Number of groups with counts over the threshold: {len(cum_count_df)}")

    # Select the time-frequency pairs to plot
    count_df_to_plot = cum_count_df[(cum_count_df["frequency"] >= min_freq_plot) & (cum_count_df["frequency"] <= max_freq_plot) & (cum_count_df["time"] >= starttime_plot) & (cum_count_df["time"] <= endtime_plot)]
    count_dfs_to_plot.append(count_df_to_plot)

    # Save the counts
    cum_count_df.to_hdf(outpath, key = time_label, mode = "a")
    print("")


# Plot the bin counts in the example time range
print("Plotting the counts...")
count_df_to_plot = concat(count_dfs_to_plot, axis = 0)
fig, ax = plot_array_spec_peak_times_and_freqs_in_multi_rows(count_df_to_plot)

# Save the plot
suffix_start = time2suffix(starttime_plot)
suffix_end = time2suffix(endtime_plot)
figname = f"hydro_spectral_array_peaks_{station}_{suffix_spec}_{suffix_peak}_count{count_threshold}_freq{min_freq:.2f}to{max_freq:.2f}hz.png"

save_figure(fig, figname)

