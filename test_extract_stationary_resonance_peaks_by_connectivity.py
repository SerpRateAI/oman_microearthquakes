# Test extracting stationary resonance peaks from the spectral-peak data using the connectivity and array-count criteria

# Imports
from os.path import join
from pandas import concat
from matplotlib.pyplot import subplots
from skimage.morphology import remove_small_objects
from numpy import array, zeros

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime_plot, ENDTIME_GEO as endtime_plot
from utils_spec import get_peak_freq_w_max_detection, get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_geo_spectrograms, read_spectral_peaks, read_spectral_peak_counts
from utils_plot import format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
station = "A01"

# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral peaks
prom_threshold = 10.0
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200

peak_file_ext = "h5"

# Array counts
count_threshold = 9
count_fille_ext = "h5"

# Resonance extracting
# Frequency range
min_freq_res = 99.9
max_freq_res = 100.1

# Isolated peak removal
min_patch_size = 3

# Plotting
fig_width = 15.0
row_height = 4.0

min_freq_plot = 25.0
max_freq_plot = 26.0

min_db = -10.0
max_db = 20.0

marker_size = 0.2

major_time_tick_spacing = "24h"
num_minor_time_ticks = 4

major_freq_tick_spacing = 0.5
num_minor_freq_ticks = 5

# Read the spectrograms
print("Reading the spectrograms...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5")

stream_spec = read_geo_spectrograms(inpath, min_freq = min_freq_plot, max_freq = max_freq_plot)
stream_spec.stitch()
trace_spec = stream_spec.get_total_power()
trace_spec.to_db()

# Read the spectral peaks
print("Reading the spectral peaks...")
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)
inpath = join(indir, f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.{peak_file_ext}")

peak_df = read_spectral_peaks(inpath)
peak_filt_df = peak_df.loc[(peak_df["frequency"] >= min_freq_res) & (peak_df["frequency"] <= max_freq_res)]

# Read the spectral peak counts
print("Reading the spectral peak array counts...")
filename_in = f"geo_spectral_peak_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.{count_fille_ext}"
inpath = join(indir, filename_in)
count_df = read_spectral_peak_counts(inpath)
count_filt_df = count_df.loc[(count_df["frequency"] >= min_freq_res) & (count_df["frequency"] <= max_freq_res)]

# Create the boolean array with each peak as a True value
print("Creating the boolean array representing the peaks")
data = trace_spec.data
timeax = trace_spec.times
freqax = trace_spec.freqs
freq_interval = freqax[1] - freqax[0]
peak_array = array([[False] * len(timeax)] * len(freqax))
peak_row_ind_array = zeros(data.shape, dtype = int)

# print(data.shape)
# print(peak_array.shape)

for index, row in peak_filt_df.iterrows():
    time = row["time"]
    freq = row["frequency"]
    power = row["power"]

    time_index = timeax.searchsorted(time)
    freq_index = freqax.searchsorted(freq)
    peak_array[freq_index, time_index] = True
    peak_row_ind_array[freq_index, time_index] = index

# Remove isolated peaks
print("Removing isolated peaks...")
peak_array_clean = remove_small_objects(peak_array, min_size = min_patch_size, connectivity = 2)

# Recover the peaks from the cleaned array
print("Recovering the peaks from the cleaned array...")
peak_clean_indices = peak_array_clean.nonzero()
peak_clean_row_inds = peak_row_ind_array[peak_clean_indices]

peak_clean_df = peak_filt_df.loc[peak_clean_row_inds]
print(f"{peak_clean_df.shape[0]:d} peaks are left after removing isolated peaks.")

# Group the peaks by time
print("Grouping the peaks by time...")
peak_group_by_time = peak_clean_df.groupby("time")
print(f"In total {len(peak_group_by_time):d} time groups of peaks.")

# Extract the peaks using the array counts as a guide
print("Extracting the stationary resonance peaks...")
resonance_peaks = []
for time, group in peak_group_by_time:
    if len(group) == 1:
        resonance_peaks.append(group)
    else:
        freq_peak = get_peak_freq_w_max_detection(count_filt_df, time)
        if freq_peak is None:
            continue
        else:
            # print(f"Time: {time}, Peak frequency: {freq_peak}")
            max_detect_peak = group.loc[group["frequency"] == freq_peak]
            if len(max_detect_peak) == 0:
                # print("No peak is found!")
                continue
            else:
                # print(type(max_detect_peak))
                resonance_peaks.append(max_detect_peak)
        
# Construct a data frame using the elements of resonance_peaks
resonance_df = concat(resonance_peaks, ignore_index=True)
print(f"Found {len(resonance_peaks):d} stationary resonance peaks.")

# Sort the data frame by time
resonance_df.sort_values("time", inplace = True)
resonance_df.set_index("time", inplace = True)

# Save the results to a CSV file
print("Saving the results to a CSV file...")
filename_out = f"test_stationary_resonance_extraction.csv"
outpath = join(indir, filename_out)
resonance_df.to_csv(outpath, index = True)
print(f"Results are saved to {outpath}")

# Plotting
print("Plotting the results...")
fig, axes = subplots(nrows = 5, ncols = 1, sharex = True, sharey = True, figsize = (fig_width, row_height * 5))

ax = axes[0]
timeax = trace_spec.times
freqax = trace_spec.freqs
data = trace_spec.data
ax.pcolormesh(timeax, freqax, data, cmap = "inferno", vmin = -10, vmax = 10)

ax.set_ylim(min_freq_plot, max_freq_plot)
format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_tick_spacing,
                    num_minor_ticks = num_minor_freq_ticks)

# Plot the spectral peaks
ax = axes[1]
times = peak_df["time"]
freqs = peak_df["frequency"]
powers = peak_df["power"]
ax.scatter(times, freqs, c = powers, cmap = cmap, s = marker_size, norm = norm)

ax.set_facecolor("darkgray")

format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_tick_spacing,
                    num_minor_ticks = num_minor_freq_ticks)

# Plot the boolean array representing the peaks
ax = axes[2]
ax.pcolormesh(timeax, freqax, peak_array, cmap = "gray")
ax.set_facecolor("blue")

format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_tick_spacing,
                    num_minor_ticks = num_minor_freq_ticks)

# Plot the boolean array with isolated peaks removed
ax = axes[3]
ax.pcolormesh(timeax, freqax, peak_array_clean, cmap = "gray")
ax.set_facecolor("blue")

format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_tick_spacing,
                    num_minor_ticks = num_minor_freq_ticks)

# Plot the extracted stationary resonance peaks
ax = axes[4]
times = resonance_df.index
freqs = resonance_df["frequency"]
powers = resonance_df["power"]
ax.scatter(times, freqs, c = powers, cmap = cmap, s = marker_size, norm = norm)
ax.set_facecolor("darkgray")

format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_tick_spacing,
                    num_minor_ticks = num_minor_freq_ticks)

# Format the x-axis
ax.set_xlim(starttime_plot, endtime_plot)

format_datetime_xlabels(ax,
                        date_format = "%Y-%m-%d",
                        va = "top", ha = "right",
                        major_tick_spacing = major_time_tick_spacing,
                        num_minor_ticks = num_minor_time_ticks,
                        rotation = 15)

# Savet the figure
print("Saving the figure...")
figname = f"test_extract_stationary_resonance_peaks_connectivity.png"
save_figure(fig, figname)
