# Compute the cumulative time-window counts and fractions of the total time of all geophone-spectral peaks

# Imports
from os.path import join
from pandas import read_csv, Timedelta
from numpy import empty, interp, isnan
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_harmonic_data_table
from utils_plot import get_interp_cat_colors, plot_cum_freq_fractions, save_figure
from multiprocessing import Pool

# Inputs
# Harmonic-series information
base_name = "PR00600_base1"

cmap_base = "tab20c"
begin_color_ind = 8
end_color_ind = 11

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
column_width = 5
row_height = 2

num_rows = 6
num_cols = 1

min_freq_main_plot = 0.0
max_freq_main_plot = 200.0

freq_window = 5.0

linewidth_stem_main = 0.01
marker_size_main = 0.2

major_freq_spacing_main = 20.0
num_minor_freq_ticks_main = 10

void_color = "gray"

linewidth_box = 3.0

harmo_label_x = 0.01
harmo_label_y = 0.98
harmo_label_size = 10

linewidth_stem_inset = 0.05
marker_size_inset = 0.5

major_freq_spacing_inset = 1.0
num_minor_freq_ticks_inset = 10

tria_size = 40.0
tria_offset = 1.2
linewidth_vline = 1.0
linewidth_tria = 1.0

# Read the list of stationary resonances
filename_harmo = f"stationary_harmonic_series_{base_name}.csv"
inpath = join(indir, filename_harmo)
harmo_df = read_harmonic_data_table(inpath)
exist_df = harmo_df.loc[harmo_df['detected'] == True].copy()
nonexist_df = harmo_df.loc[harmo_df['detected'] == False].copy()

# Read the spectral-peak cummulative counts
print("Reading the spectral peak counts...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

filename_in = f"geo_spec_peak_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_freq_sorted.csv"
inpath = join(indir, filename_in)
cum_count_df = read_csv(inpath, index_col = 0)

# Plot the results
print("Generating the plot...")
num_harmo = len(harmo_df)
num_plots = num_harmo + 1

if (num_rows - 1) * num_cols + 1 != num_plots:
    raise ValueError("The number of rows and columns does not match the number of plots!")

# Create a figure
fig = figure(figsize=(column_width * num_cols, row_height * num_rows))

# Create a GridSpec with specific dimensions
gs = GridSpec(num_rows, num_cols, figure = fig)

# Plot the whole frequency range
ax = fig.add_subplot(gs[0, :])
ax = plot_cum_freq_fractions(cum_count_df,
                            xtick_labels = False,
                            min_freq = min_freq_main_plot, max_freq = max_freq_main_plot,
                            linewidth = linewidth_stem_main, marker_size = marker_size_main,
                            major_freq_spacing = major_freq_spacing_main, num_minor_freq_ticks = num_minor_freq_ticks_main,
                            axis = ax)

min_frac = ax.get_ylim()[0]
max_frac = ax.get_ylim()[1]

# Add the rectangles marking the insets
# Existing harmonics
num_exist = len(exist_df)
harmo_colors = get_interp_cat_colors(cmap_base, begin_color_ind, end_color_ind, num_exist)

i = 0   
for _, row in exist_df.iterrows():
    color = harmo_colors[i]
    min_freq = row["predicted_freq"] - freq_window / 2
    max_freq = row["predicted_freq"] + freq_window / 2
    rect = Rectangle((min_freq, min_frac), max_freq - min_freq, max_frac - min_frac,
                     edgecolor = color, facecolor = 'none', linewidth = linewidth_box, zorder = 10)
    ax.add_patch(rect)
    i += 1

# Non-existing harmonics
for _, row in nonexist_df.iterrows():
    min_freq = row["predicted_freq"] - freq_window / 2
    max_freq = row["predicted_freq"] + freq_window / 2
    rect = Rectangle((min_freq, min_frac), max_freq - min_freq, max_frac - min_frac,
                     edgecolor = void_color, facecolor = 'none', linewidth = linewidth_box, linestyle = "--", zorder = 10)
    ax.add_patch(rect)

# Set the y label of the main plot as "Fraction"
ax.set_ylabel("Fraction")

# Add the remaining 15 subplots and store them in the array
axes_inset = empty((num_rows - 1, num_cols), dtype = object)
for i in range(1, num_rows):
    for j in range(num_cols):
        axes_inset[i-1, j] = fig.add_subplot(gs[i, j])

# Plot the inset plots of frequency ranges containing the harmonics
harmo_df.sort_values("predicted_freq", inplace = True)
i_panel = 0
i_exist = 0
for name, row in harmo_df.iterrows():
    ax = axes_inset.flatten()[i_panel]
    min_freq = row["predicted_freq"] - freq_window / 2
    max_freq = row["predicted_freq"] + freq_window / 2
    ax = plot_cum_freq_fractions(cum_count_df,
                                xtick_labels = False,
                                ytick_labels = False,
                                min_freq = min_freq, max_freq = max_freq,
                                linewidth = linewidth_stem_inset, marker_size = marker_size_inset,
                                major_freq_spacing = major_freq_spacing_inset, num_minor_freq_ticks = num_minor_freq_ticks_inset,
                                axis = ax)
    
    exist = row["detected"]
    pred_freq = row["predicted_freq"]
    # If it is a harmonic, add the peak label, plot the frequency range for extracting the data, and plot the predicted frequency
    if exist: 
        color = harmo_colors[i_exist]
        ax.text(harmo_label_x, harmo_label_y, name, color = color, fontsize = harmo_label_size, fontweight = "bold", transform = ax.transAxes, ha = 'left', va = 'top')
        
        obs_freq = row["observed_freq"]
        min_freq = row["extract_min_freq"]
        max_freq = row["extract_max_freq"]

        if not isnan(min_freq):
            ax.axvline(min_freq, color = color, linestyle = ':', linewidth = linewidth_vline)

        if not isnan(max_freq):
            ax.axvline(max_freq, color = color, linestyle = ':', linewidth = linewidth_vline)

        frac = cum_count_df.loc[cum_count_df["frequency"] == obs_freq, "fraction"].values[0]
        ax.scatter(obs_freq, frac * tria_offset, s = tria_size, marker = 'v', facecolors = color, edgecolors = color, linewidth = linewidth_tria, zorder = 10)
        ax.scatter(pred_freq, frac * tria_offset, s = tria_size, marker = 'v', facecolors = 'none', edgecolors = color, linewidth = linewidth_tria, zorder = 10)

        i_exist += 1
    # If it is not a harmonic, add the peak label and plot the predicted frequency
    else:
        ax.text(harmo_label_x, harmo_label_y, name, color = void_color, fontsize = harmo_label_size, fontweight = "bold", transform = ax.transAxes, ha = 'left', va = 'top')
        
        frac = interp(pred_freq, cum_count_df["frequency"], cum_count_df["fraction"])
        ax.scatter(pred_freq, frac * tria_offset, s = tria_size, marker = 'v', facecolors = 'none', edgecolors = void_color, linewidth = linewidth_tria, zorder = 10)

    i_panel += 1

# Set the x labels of the outer axes as "Frequency (Hz)"
for i in range(num_cols):
    axes_inset[-1, i].set_xlabel("Frequency (Hz)")

# Set the y labels of the outer axes as "Fraction"
for i in range(num_rows - 1):
    axes_inset[i, 0].set_ylabel("Fraction")

# Save the figure
print("Saving the figure...")
filename_fig = f"stationary_harmonics_from_cum_freq_fracs_{base_name}.png"
save_figure(fig, filename_fig)
