# Compute the cumulative time-window counts and fractions of the total time of all geophone-spectral peaks

# Imports
from os.path import join
from pandas import read_csv, Timedelta
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_spectral_peak_counts
from utils_plot import plot_cum_freq_fractions, save_figure
from multiprocessing import Pool

# Inputs
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

file_ext_in = "h5"

# Grouping
count_threshold = 9

# Plotting
min_freq_main_plot = 0.0
max_freq_main_plot = 200.0

linewidth_main = 0.1
marker_size_main = 1.0

major_freq_spacing_main = 20.0
num_minor_freq_ticks_main = 10

min_freq_inset1_plot = 22.0
max_freq_inset1_plot = 27.0

min_freq_inset2_plot = 35.0
max_freq_inset2_plot = 40.0

linewidth_inset = 1.0
marker_size_inset = 10.0

major_freq_spacing_inset = 1.0
num_minor_freq_ticks_inset = 10

linewidth_box = 4.0

# Read the spectral-peak cummulative counts
print("Reading the spectral peak counts...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

filename_in = f"geo_spec_peak_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.csv"
inpath = join(indir, filename_in)
cum_count_df = read_csv(inpath, index_col = 0)
cum_count_df.sort_values('frequency', inplace = True)

# Plot the results
print("Plotting the cumulative spectral-peak fractions in two panels...")
fig, axes = subplots(3, 1, figsize = (10, 10))

ax = axes[0]
ax = plot_cum_freq_fractions(cum_count_df,
                            xtick_labels = False,
                            min_freq = min_freq_main_plot, max_freq = max_freq_main_plot,
                            linewidth = linewidth_main, marker_size = marker_size_main,
                            major_freq_spacing = major_freq_spacing_main, num_minor_freq_ticks = num_minor_freq_ticks_main,
                            axis = ax)

min_frac = ax.get_ylim()[0]
max_frac = ax.get_ylim()[1]

# Add a rectangle to highlight the insets
rect1 = Rectangle((min_freq_inset1_plot, min_frac), max_freq_inset1_plot - min_freq_inset1_plot, max_frac - min_frac,
                  edgecolor = 'skyblue', facecolor = 'none', linewidth = linewidth_box, zorder = 10)

rect2 = Rectangle((min_freq_inset2_plot, min_frac), max_freq_inset2_plot - min_freq_inset2_plot, max_frac - min_frac,
                    edgecolor = 'skyblue', facecolor = 'none', linewidth = linewidth_box, zorder = 10)

ax.add_patch(rect1)
ax.add_patch(rect2)

ax = axes[1]
ax = plot_cum_freq_fractions(cum_count_df,
                            xtick_labels = False,
                            min_freq = min_freq_inset1_plot, max_freq = max_freq_inset1_plot,
                            linewidth = linewidth_inset, marker_size = marker_size_inset,
                            major_freq_spacing = major_freq_spacing_inset, num_minor_freq_ticks = num_minor_freq_ticks_inset,
                            axis = ax)


ax = axes[2]
ax = plot_cum_freq_fractions(cum_count_df, 
                            min_freq = min_freq_inset2_plot, max_freq = max_freq_inset2_plot,
                            linewidth = linewidth_inset, marker_size = marker_size_inset,
                            major_freq_spacing = major_freq_spacing_inset, num_minor_freq_ticks = num_minor_freq_ticks_inset,
                            axis = ax)

fig.suptitle(f"Fraction of total time recorded by at least {count_threshold:d} geophones", fontsize = 12, fontweight = 'bold', y = 0.90)  
print("Done.")

# Save the figure
print("Saving the figure...")
filename_fig = f"geo_spec_peak_cum_freq_fracs_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.png"
save_figure(fig, filename_fig)
