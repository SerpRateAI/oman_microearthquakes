# Group the spectral peaks of geophone data by time and frequency and count the number of peaks in each group

# Imports
from os.path import join
from time import time
from pandas import Series
from pandas import concat, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import time2suffix
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_geo_spec_peaks, update_spectral_peak_group_counts
from utils_plot import plot_array_spec_peak_counts, save_figure

# Inputs
# Data
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60
prom_threshold = 10
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200.0

# Grouping
count_threshold = 9

# Plotting
starttime_plot = starttime
endtime_plot = endtime

min_freq_plot = 0.0
max_freq_plot = 5.0

date_format = "%Y-%m-%d"
major_time_spacing = "1d"
num_minor_time_ticks = 4

example_counts = [10, 20, 30]

size_scale = 30

# Process the detections of each station
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

# Get the common time labels
print("Getting the common time labels...")
time_label_sr_list = []
for i, station in enumerate(stations):
    filename_in =  f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
    inpath = join(indir, filename_in)
    time_label_sr = read_hdf(inpath, "time_label")
    time_label_sr_list.append(time_label_sr)

time_label_sr = concat(time_label_sr_list, axis = 0)
time_label_sr = Series(time_label_sr.unique())
time_label_sr.reset_index(drop = True, inplace = True)
time_label_sr.sort_values(inplace = True)

# Save the time labels
filename_out = f"geo_spectral_peak_counts_{suffix_spec}_{suffix_peak}_count{count_threshold}.h5"
outpath = join(indir, filename_out)
time_label_sr.to_hdf(outpath, key = "time_label", mode = "w")

# Iterate over the time labels
count_dfs_to_plot = []
for time_label in time_label_sr:
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

    count_dfs_to_plot.append(cum_count_df.loc[(cum_count_df["time"] >= starttime) & (cum_count_df["time"] <= endtime) & (cum_count_df["frequency"] >= min_freq_plot) & (cum_count_df["frequency"] <= max_freq_plot)])

count_df_to_plot = concat(count_dfs_to_plot, axis = 0)

# Plot the bin counts in the example time range
print("Plotting the counts...")
fig, ax = plot_array_spec_peak_counts(count_df_to_plot,
                                      size_scale = size_scale, 
                                      example_counts = example_counts,
                                      starttime = starttime_plot, endtime = endtime_plot, freq_lim = (min_freq_plot, max_freq_plot),
                                      date_format = date_format,
                                      major_time_spacing = major_time_spacing, num_minor_time_ticks = num_minor_time_ticks)

# Save the plot
suffix_start = time2suffix(starttime_plot)
suffix_end = time2suffix(endtime_plot)
figname = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}_count{count_threshold}_{suffix_start}to{suffix_end}_freq{min_freq_plot:.2f}to{max_freq_plot:.2f}hz.png"

save_figure(fig, figname)

