# Group the spectral peaks of geophone data by time and frequency and count the number of peaks in each group

# Imports
from os.path import join
from time import time

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_basic import time2suffix
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_spectral_peaks, save_spectral_peak_counts, update_spectral_peak_group_counts
from utils_plot import plot_array_spec_peak_counts, save_figure

# Inputs
# Data
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60
prom_threshold = 10
rbw_threshold = 3.0

min_freq = None
max_freq = 200.0

file_ext_in = "h5"

# Grouping
count_threshold = 9

# Saving the results
file_format_out = "hdf"

# Plotting
starttime_plot = "2020-01-13T00:00:00"
endtime_plot = "2020-01-14T00:00:00"

date_format = "%Y-%m-%d %H:%M:%S"
major_time_spacing = "6h"
minor_time_spacing = "15h"

example_counts = [10, 20, 30]

size_scale = 30

# Process the detections of each station
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq, max_freq = max_freq)


# Initialize the spectral-peak group count
#stations = ["A01", "A02", "A03"]
for i, station in enumerate(stations):
    print(f"Working on {station}...")
    clock1 = time()

    # Read the spectral peaks
    print("Reading the spectral peaks...")
    filename = f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.{file_ext_in}"
    inpath = join(indir, filename)
    print(inpath)

    peak_df = read_spectral_peaks(inpath)
    num_peaks = len(peak_df)
    print(f"{num_peaks} peaks are read.")

    # Update the spectral-peak group count
    print("Updating the group counts...")
    if i == 0:
        cum_count_df = update_spectral_peak_group_counts(peak_df)
    else:
        cum_count_df = update_spectral_peak_group_counts(peak_df, counts_to_update = cum_count_df)
    print("Update finished")

    cum_num_peaks = len(cum_count_df)
    max_count = cum_count_df["count"].max()
    print(f"Cummulative number of peaks: {cum_num_peaks}")
    print(f"Maximum count: {max_count}")
    
    clock2 = time()
    elapse = clock2 - clock1
    print(f"Finished processing {station}.")
    print(f"Elapsed time: {elapse} s")

# Keep only the groups with counts over the threshold
cum_count_df["count"] = cum_count_df["count"].astype(int)
cum_count_df = cum_count_df[cum_count_df["count"] >= count_threshold]
cum_count_df.reset_index(drop = True, inplace = True)

# Save the counts
file_stem = f"geo_spectral_peak_counts_{suffix_spec}_{suffix_peak}_count{count_threshold}"
save_spectral_peak_counts(cum_count_df, file_stem, file_format_out)

# Plot the bin counts in the example time range
print("Plotting the counts...")
fig, ax = plot_array_spec_peak_counts(cum_count_df,
                                        size_scale = size_scale, 
                                        example_counts = example_counts,
                                        starttime = starttime_plot, endtime = endtime_plot, freq_lim = (min_freq, max_freq),
                                        date_format = date_format,
                                        major_time_spacing = major_time_spacing, num_minor_time_ticks = 3)

# Save the plot
suffix_start = time2suffix(starttime_plot)
suffix_end = time2suffix(endtime_plot)
figname = f"geo_spectral_peak_counts_{suffix_spec}_{suffix_peak}_count{count_threshold}_{suffix_start}_{suffix_end}.png"

save_figure(fig, figname)

