# Plot the figure in Liu et al., 2025a showing the harmonic relation of the stationary resonances

### Import the necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, read_hdf
from matplotlib.pyplot import figure, subplots
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import colorcet as cc

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_plot import add_colorbar, add_vertical_scalebar, add_day_night_shading, format_datetime_xlabels, save_figure
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the harmonic relations between the stationary resonances.")
parser.add_argument("--base_mode", type = str, default = "PR02549", help = "Base mode name.")
parser.add_argument("--base_order", type = int, default = 2, help = "Base mode number.")
parser.add_argument("--scale_factor", type = float, default = 2.0, help = "Scale factor for the frequency values.")

parser.add_argument("--station_to_plot", type = str, help = "Station whose peak powers are to be plotted as examples.")
parser.add_argument("--mode_name_plot1_x", type = str, default = "PR02549", help = "Mode name for the x-axis of the first plot.")
parser.add_argument("--mode_name_plot1_y", type = str, default = "PR03822", help = "Mode name for the y-axis of the first plot.")
parser.add_argument("--mode_name_plot2_x", type = str, default = "PR05097", help = "Mode name for the x-axis of the second plot.")
parser.add_argument("--mode_name_plot2_y", type = str, default = "PR03822", help = "Mode name for the y-axis of the second plot.")

parser.add_argument("--window_length", type = float, default = 60.0, help = "Window length in seconds for computing the spectrogram.")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap between consecutive windows for computing the spectrogram.")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type = float, default = 3.0, help = "Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type = float, default = 10.0, help = "Maximum mean dB value for excluding noise windows.")

# Parse the arguments
args = parser.parse_args()
base_mode = args.base_mode
base_order = args.base_order
scale_factor = args.scale_factor

station_to_plot = args.station_to_plot
mode_name_plot1_x = args.mode_name_plot1_x
mode_name_plot1_y = args.mode_name_plot1_y
mode_name_plot2_x = args.mode_name_plot2_x
mode_name_plot2_y = args.mode_name_plot2_y

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Constants
max_freq = 200.0
max_cmap_portion = 0.8

cell_dim = 5.0

marker_size_freq = 5
marker_size_corr = 10

marker_alpha = 0.5

linewidth = 0.5

db_range = 30.0

scalebar_x = 0.02
scalebar_y = 0.95
scalebar_length = 0.2

label_offset_x = 0.01
label_offset_y = 0.00

axis_label_size = 12
corr_label_size = 12

corr_label_offset_x = 0.03
corr_label_offset_y = 0.03

panel_label_size = 14

panel_label1_offset_x = -0.02
panel_label1_offset_y = 0.05

panel_label2_offset_x = -0.05
panel_label2_offset_y = 0.05

panel_label3_offset_x = -0.05
panel_label3_offset_y = 0.05

colorbar_offset = 0.02
colorbar_width = 0.01

### Generate the figure ###
print("Generating the figure...")
fig = figure(figsize=( 3 * cell_dim, 3 * cell_dim))
gs_top = GridSpec(2, 3, figure=fig, hspace=0.1, bottom = 0.4)

### Plot the mode orders and frequency ratios of the stationary resonances ###

# Read the harmonic series information
print(f"Reading the information of the harmonic series with {base_mode} as the base mode {base_order}...")
filename = f"stationary_harmonic_series_{base_mode}_base{base_order:d}.csv"
inpath = join(indir, filename)
harmonic_df = read_csv(inpath)

suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

# Plotting
print("Generating the subplots...")

ax_freq = fig.add_subplot(gs_top[0, 0:3])
ax_num = fig.add_subplot(gs_top[1, 0:3])
cmap = cc.cm.isolum

print("Plotting each mode...")
for i, row in harmonic_df.iterrows():
    mode_name = row["mode_name"]
    if not row["detected"]:
        print(f"Skipping the undetected mode {mode_name}...")
        continue

    # Read the frequencies of the mode
    print(f"Plotting the mode {mode_name}...")
    filename = f"stationary_resonance_profile_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
    inpath = join(indir, filename)
    current_mode_df = read_hdf(inpath, key = "profile")
    current_mode_df.set_index("time", inplace = True)

    mode_order = row["mode_order"]
    mean_freq = current_mode_df["frequency"].mean()

    if mode_order == base_order:
        norm = Normalize(vmin = mean_freq, vmax = max_freq)
        base_mode_df = current_mode_df
        freq_ratios = [1] * len(current_mode_df)
        timeax_common = current_mode_df.index
    else:
        # Get the common time axis between the base mode and the current mode
        merged_df = base_mode_df.merge(current_mode_df, how = "inner", left_index = True, right_index = True, suffixes = ("_base", "_current"))
        merged_df["freq_ratio"] = merged_df["frequency_current"] / merged_df["frequency_base"]
        freq_ratios = merged_df["freq_ratio"]
        timeax_common = merged_df.index

        # print((len(timeax), len(freq_ratios)))

    color = cmap(norm(mean_freq))
    ax_freq.scatter(current_mode_df.index, (current_mode_df["frequency"] - mean_freq) * scale_factor + mode_order, color = color, s = marker_size_freq, label = mode_name)
    ax_num.scatter(timeax_common, freq_ratios, color = color, s = marker_size_freq, label = mode_name)

    ax_num.axhline(y = mode_order / base_order, color = "crimson", linestyle = "--", linewidth = linewidth)

# Add the day-night shading
print("Adding the day-night shading...")
ax_freq = add_day_night_shading(ax_freq)
ax_num = add_day_night_shading(ax_num)

# Set the axis limits
ax_freq.set_xlim(starttime, endtime)
ax_num.set_xlim(starttime, endtime)

# Add the frequency scalebar
print("Adding the frequency scalebar...")
ax_freq = add_vertical_scalebar(ax_freq, (scalebar_x, scalebar_y), scalebar_length, scale_factor, (label_offset_x, label_offset_y), 
                                label_unit = "Hz")

# Format the x-axis labels
print("Formatting the x-axis labels...")

format_datetime_xlabels(ax_freq,
                        plot_axis_label=False, plot_tick_label=False,
                        major_tick_spacing="5d", num_minor_ticks=5,
                        date_format="%Y-%m-%d",
                        rotation=0, ha="center", va="top", 
                        axis_label_size=axis_label_size)

format_datetime_xlabels(ax_num,
                        plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing="5d", num_minor_ticks=5,
                        date_format="%Y-%m-%d",
                        rotation=0, ha="center", va="top", 
                        axis_label_size=axis_label_size)

# Format the y-axis labels
ax_freq.set_ylabel("Mode order", fontsize = axis_label_size)
ax_num.set_ylabel(f"Frequency ratio to Mode {base_order}", fontsize = axis_label_size)

# Plot the colorbar
print("Plotting the colorbar...")
bbox = ax_freq.get_position()
cbar_pos = [bbox.x1 + colorbar_offset, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, cbar_pos,  "Mean frequency (Hz)", cmap = cmap, norm = norm)

# Add the panel label
ax_freq.text(panel_label1_offset_x, 1.0 + panel_label1_offset_y, "(a)", ha = "right", va = "bottom", transform = ax_freq.transAxes, fontsize = panel_label_size, fontweight = "bold")

### Plot the first cross plot of peak powers between mode_name_plot1_x and mode_name_plot1_y ###
# Add the subplots for the correlation plots
gs_bottom = GridSpec(1, 3, figure=fig, hspace=0.1, top = 0.35)

# Read the station-level power-correlation data
filename = f"stationary_harmonic_station_power_corr_{base_mode}_base{base_order:d}_{suffix_spec}_{suffix_peak}.csv"
inpath = join(indir, filename)
corr_df = read_csv(inpath)

# Read the data
filename = f"stationary_resonance_properties_geo_{mode_name_plot1_x}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename)
mode_x_df = read_hdf(inpath, key = "properties")
mode_x_df = mode_x_df[mode_x_df["station"] == station_to_plot]
mode_order_x = harmonic_df[harmonic_df["mode_name"] == mode_name_plot1_x]["mode_order"].values[0]

filename = f"stationary_resonance_properties_geo_{mode_name_plot1_y}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename)
mode_y_df = read_hdf(inpath, key = "properties")
mode_y_df = mode_y_df[mode_y_df["station"] == station_to_plot]
mode_order_y = harmonic_df[harmonic_df["mode_name"] == mode_name_plot1_y]["mode_order"].values[0]

# Get the correlation value between the two modes
if mode_order_x < mode_order_y:
    corr = corr_df[(corr_df["mode_i_name"] == mode_name_plot1_x) & (corr_df["mode_j_name"] == mode_name_plot1_y)]["correlation"].values[0]
else:
    corr = corr_df[(corr_df["mode_i_name"] == mode_name_plot1_y) & (corr_df["mode_j_name"] == mode_name_plot1_x)]["correlation"].values[0]

# Build the correlation color scale
cmap = cc.cm.bmw
norm = Normalize(vmin = 0.0, vmax = 1.0)

# Plot the peak powers
print("Plotting the peak powers...")

mode_merged_df = mode_x_df.merge(mode_y_df, how = "inner", on = "time", suffixes = ("_x", "_y"))

ax = fig.add_subplot(gs_bottom[0])
ax.scatter(mode_merged_df["total_power_x"], mode_merged_df["total_power_y"], color = cmap(norm(corr)), s = marker_size_corr, alpha = marker_alpha, edgecolors = "none")

min_x = mode_merged_df["total_power_x"].mean() - db_range / 2
max_x = mode_merged_df["total_power_x"].mean() + db_range / 2

min_y = mode_merged_df["total_power_y"].mean() - db_range / 2
max_y = mode_merged_df["total_power_y"].mean() + db_range / 2

# Set the axis limits
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

# Set the axis labels
ax.set_xlabel(f"Mode {mode_order_x} power (dB)", fontsize = axis_label_size)
ax.set_ylabel(f"Mode {mode_order_y} power (dB)", fontsize = axis_label_size)

# Set the aspect ratio to be equal
ax.set_aspect("equal")

# Add the label
ax.text(corr_label_offset_x, 1.0 - corr_label_offset_y, f"{station_to_plot}\nMode {mode_order_x} vs. {mode_order_y}", ha = "left", va = "top", transform = ax.transAxes, fontsize = corr_label_size, fontweight = "bold")

# Add the panel label
ax.text(panel_label2_offset_x, 1.0 + panel_label2_offset_y, "(b)", ha = "right", va = "bottom", transform = ax.transAxes, fontsize = panel_label_size, fontweight = "bold")

### Plot the second cross plot of peak powers between mode_name_plot2_x and mode_name_plot2_y ###
# Read the data
filename = f"stationary_resonance_properties_geo_{mode_name_plot2_x}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename)
mode_x_df = read_hdf(inpath, key = "properties")
mode_x_df = mode_x_df[mode_x_df["station"] == station_to_plot]
mode_order_x = harmonic_df[harmonic_df["mode_name"] == mode_name_plot2_x]["mode_order"].values[0]

filename = f"stationary_resonance_properties_geo_{mode_name_plot2_y}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(indir, filename)
mode_y_df = read_hdf(inpath, key = "properties")
mode_y_df = mode_y_df[mode_y_df["station"] == station_to_plot]
mode_order_y = harmonic_df[harmonic_df["mode_name"] == mode_name_plot2_y]["mode_order"].values[0]

# Get the correlation value between the two modes
if mode_order_x < mode_order_y:
    corr = corr_df[(corr_df["mode_i_name"] == mode_name_plot2_x) & (corr_df["mode_j_name"] == mode_name_plot2_y)]["correlation"].values[0]
else:
    corr = corr_df[(corr_df["mode_i_name"] == mode_name_plot2_y) & (corr_df["mode_j_name"] == mode_name_plot2_x)]["correlation"].values[0]

# Plot the peak powers
print("Plotting the peak powers...")
ax = fig.add_subplot(gs_bottom[1])

mode_merged_df = mode_x_df.merge(mode_y_df, how = "inner", on = "time", suffixes = ("_x", "_y"))
ax.scatter(mode_merged_df["total_power_x"], mode_merged_df["total_power_y"], color = cmap(norm(corr)), s = marker_size_corr, alpha = marker_alpha, edgecolors = "none")

min_x = mode_merged_df["total_power_x"].mean() - db_range / 2
max_x = mode_merged_df["total_power_x"].mean() + db_range / 2

min_y = mode_merged_df["total_power_y"].mean() - db_range / 2
max_y = mode_merged_df["total_power_y"].mean() + db_range / 2

# Set the axis limits
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

# Set the axis labels
ax.set_xlabel(f"Mode {mode_order_x} power (dB)", fontsize = axis_label_size)
ax.set_ylabel(f"Mode {mode_order_y} power (dB)", fontsize = axis_label_size)

# Set the aspect ratio to be equal
ax.set_aspect("equal")

# Add the label
ax.text(corr_label_offset_x, 1.0 - corr_label_offset_y, f"{station_to_plot}\nMode {mode_order_x} vs. {mode_order_y}", ha = "left", va = "top", transform = ax.transAxes, fontsize = corr_label_size, fontweight = "bold")

### Plot the station-averaged power correlation matrix ###

# Read the data
filename = f"stationary_harmonic_avg_power_corr_PR02549_base2_{suffix_spec}_{suffix_peak}.csv"
inpath = join(indir, filename)
corr_df = read_csv(inpath)

# Assemble the correlation matrix
corr_matrix = corr_df.pivot(index = "mode_i_index", columns = "mode_j_index", values = "correlation")
corr_matrix = corr_matrix.transpose()

# Add the plot to the right of the frequency plot
print("Plotting the correlation matrix...")

ax_corr = fig.add_subplot(gs_bottom[2])
im = ax_corr.imshow(corr_matrix, cmap = cmap, norm = norm)

# Set the axis labels
ax_corr.set_xticklabels([])
ax_corr.set_yticklabels([])

# Label the mode orders
unique_rows = corr_df.drop_duplicates(subset=["mode_i_index", "mode_i_order"])

ax_corr.set_xticks(range(len(unique_rows)))
ax_corr.set_xticklabels(unique_rows["mode_i_order"])
ax_corr.set_xlabel("Mode order", fontsize = axis_label_size)

unique_rows = corr_df.drop_duplicates(subset=["mode_j_index", "mode_j_order"])

ax_corr.set_yticks(range(len(unique_rows)))
ax_corr.set_yticklabels(unique_rows["mode_j_order"])
ax_corr.set_ylabel("Mode order", fontsize = axis_label_size)

# Set the background color to transparent
ax_corr.patch.set_alpha(0.0)

# Remove the top and right spines
ax_corr.spines['top'].set_visible(False)
ax_corr.spines['right'].set_visible(False)

# Add the label
ax_corr.text(1 - corr_label_offset_x, 1 - corr_label_offset_y, "Station average", ha = "right", va = "top", transform = ax_corr.transAxes, fontsize = corr_label_size, fontweight = "bold")

# Add the colorbar
bbox = ax_corr.get_position()
cbar_pos = [bbox.x1 + colorbar_offset, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, cbar_pos,  "Correlation", cmap = cmap, norm = norm)

# Add the panel label
ax_corr.text(panel_label3_offset_x, 1.0 + panel_label3_offset_y, "(c)", ha = "right", va = "bottom", transform = ax_corr.transAxes, fontsize = panel_label_size, fontweight = "bold")

# Save the figure
print("Saving the figure...")
filename = f"liu_2025a_harmonic_relation.png"
save_figure(fig, filename)