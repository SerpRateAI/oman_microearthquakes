# Plot the frequency histogram of all cepstrum peaks 

# Import the necessary libraries
from os.path import join
from numpy import linspace
from pandas import cut, read_hdf
from matplotlib.pyplot import subplots

from utils_spec import get_spectrogram_file_suffix
from utils_basic import SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, SAMPLING_RATE as sampling_rate
from utils_plot import format_freq_xlabels, save_figure

# Input parameters
# Station
station = "A19"

# Spectrogram computation
window_length = 60.0
overlap = 0.0
downsample = False

# Frequency binning
min_freq = 3.0
max_freq = 30.0
num_bins = 200

# Plotting
figwidth = 9
figheight = 6

major_freq_tick_spacing = 5.0
num_minor_freq_ticks = 5

# Read the cepstrum peaks
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)

# Read the cepstrum peaks
filename = f"geo_station_cepstrum_peaks_{station}_{suffix_spec}.h5"
inpath = join(indir, filename)

peak_df = read_hdf(inpath)

# Group the peaks by frequency bins
bin_edges = linspace(min_freq, max_freq, num_bins + 1)
peak_df["bin"] = cut(peak_df["frequency"], bin_edges, right=False)
bin_count_sr = peak_df.groupby("bin").size()

# Plot the histogram
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
fig, ax = subplots(1, 1, figsize=(9, 6))
ax.bar(bin_centers, bin_count_sr, width=bin_centers[1] - bin_centers[0], color="tab:blue", edgecolor="black")

# Format the x-axis
ax.set_xlim(min_freq, max_freq)
format_freq_xlabels(ax, major_tick_spacing = major_freq_tick_spacing, num_minor_ticks = num_minor_freq_ticks)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Number of cepstrum peaks")

# Save the figure
filename_out = f"cepstrum_peak_freq_histogram_{station}_{suffix_spec}.png"
save_figure(fig, filename_out)


