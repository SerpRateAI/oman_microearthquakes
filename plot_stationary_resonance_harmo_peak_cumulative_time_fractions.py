# Compute the cumulative time-window counts and fractions of the total time of all geophone-spectral peaks

# Imports
from os.path import join
from pandas import read_csv, Timedelta
from matplotlib.pyplot import subplots, get_cmap
from matplotlib.patches import Rectangle

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_harmonic_data_table
from utils_plot import get_interp_cat_colors, plot_cum_freq_fractions, save_figure
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

base_name = "SR25a"

# Grouping
count_threshold = 9

# Plotting
width = 20
height = 8

num_rows = 4
num_cols = 4

min_freq_main_plot = 0.0
max_freq_main_plot = 200.0

linewidth_stem_main = 0.02
marker_size_main = 0.1

major_freq_spacing_main = 20.0
num_minor_freq_ticks_main = 10

cmap_base = "tab20c"
begin_color_ind = 0 
end_color_ind = 3

void_color = "gray"

linewidth_box = 1.5

harmo_label_x = 0.01
harmo_label_y = 0.98
harmo_label_size = 12

linewidth_stem_inset = 0.1
marker_size_inset = 0.5

major_freq_spacing_inset = 1.0
num_minor_freq_ticks_inset = 10

linewidth_extract = 1.0
linewidth_pred = 1.0

# Read the list of stationary resonances
filename_harmo = f"stationary_resonance_params_harmo_{base_name}.csv"
inpath = join(indir, filename_harmo)
harmo_df = read_harmonic_data_table(inpath)
exist_df = harmo_df.loc[harmo_df['exist'] == True].copy()
nonexist_df = harmo_df.loc[harmo_df['exist'] == False].copy()

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
num_harmo = len(harmo_df)
num_plots = num_harmo + 1

if num_rows * num_cols != num_plots:
    raise ValueError("The number of rows and columns does not match the number of plots!")

fig, axes = subplots(num_rows, num_cols, figsize = (width, height))

# Plot the whole frequency range
ax = axes[0, 0]
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

# print(len(harmo_colors))
# print(len(exist_df))

i = 0   
for _, row in exist_df.iterrows():
    color = harmo_colors[i]
    min_freq = row['demo_min_freq']
    max_freq = row['demo_max_freq']
    rect = Rectangle((min_freq, min_frac), max_freq - min_freq, max_frac - min_frac,
                     edgecolor = color, facecolor = 'none', linewidth = linewidth_box, zorder = 10)
    ax.add_patch(rect)
    i += 1

# Non-existing harmonics
for _, row in nonexist_df.iterrows():
    min_freq = row['demo_min_freq']
    max_freq = row['demo_max_freq']
    rect = Rectangle((min_freq, min_frac), max_freq - min_freq, max_frac - min_frac,
                     edgecolor = void_color, facecolor = 'none', linewidth = linewidth_box, linestyle = "--", zorder = 10)
    ax.add_patch(rect)

# Plot the frequency ranges containing the harmonics
harmo_df.sort_values('demo_min_freq', inplace = True)
i_panel = 0
i_exist = 0
for name, row in harmo_df.iterrows():
    ax = axes.flatten()[i_panel + 1]
    min_freq = row['demo_min_freq']
    max_freq = row['demo_max_freq']
    ax = plot_cum_freq_fractions(cum_count_df,
                                xtick_labels = False,
                                ytick_labels = False,
                                min_freq = min_freq, max_freq = max_freq,
                                linewidth = linewidth_stem_inset, marker_size = marker_size_inset,
                                major_freq_spacing = major_freq_spacing_inset, num_minor_freq_ticks = num_minor_freq_ticks_inset,
                                axis = ax)
    
    # If it is a harmonic, add the peak label
    exist = row['exist']
    if exist: 
        color = harmo_colors[i_exist]
        ax.text(harmo_label_x, harmo_label_y, name, color = color, fontsize = harmo_label_size, fontweight = "bold", transform = ax.transAxes, ha = 'left', va = 'top')
        i_exist += 1
    else:
        ax.text(harmo_label_x, harmo_label_y, name, color = void_color, fontsize = harmo_label_size, fontweight = "bold", transform = ax.transAxes, ha = 'left', va = 'top')

    i_panel += 1

    # Plot the frequency range for extracting the data
    min_freq = row["extract_min_freq"]
    max_freq = row["extract_max_freq"]
    ax.axvline(min_freq, color = color, linestyle = ':', linewidth = linewidth_extract)
    ax.axvline(max_freq, color = color, linestyle = ':', linewidth = linewidth_extract)

    # Plot the predicted frequency
    pred_freq = row["pred_freq"]
    if exist:
        ax.axvline(pred_freq, color = color, linestyle = '--', linewidth = linewidth_pred)
    else:
        ax.axvline(pred_freq, color = void_color, linestyle = '--', linewidth = linewidth_pred)

# Set the x labels of the outer axes as "Frequency (Hz)"
for i in range(num_cols):
    axes[num_rows - 1, i].set_xlabel("Frequency (Hz)")

# Set the y labels of the outer axes as "Fraction"
for i in range(num_rows):
    axes[i, 0].set_ylabel("Fraction")

# Save the figure
print("Saving the figure...")
filename_fig = f"stationary_resoances_from_cum_freq_fracs_{base_name}_count{count_threshold:d}.png"
save_figure(fig, filename_fig)
