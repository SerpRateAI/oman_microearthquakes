# Plot the figure in the AGU 2024 iPoster showing the harmonic relations between the stationary resonances

### Import the necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, read_hdf
from matplotlib.pyplot import figure, subplots
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import colorcet as cc

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import str2timestamp
from utils_plot import add_colorbar, add_vertical_scalebar, add_day_night_shading, format_datetime_xlabels, save_figure
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the harmonic relations between the stationary resonances.")
parser.add_argument("--base_mode", type = str, default = "PR02549", help = "Base mode name.")
parser.add_argument("--base_order", type = int, default = 2, help = "Base mode number.")
parser.add_argument("--scale_factor", type = float, default = 2.0, help = "Scale factor for the frequency values.")

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

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Constants
max_freq = 200.0
max_cmap_portion = 0.8

figwidth = 15.0
figheight = 5.0

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
legend_label_size = 10
title_size = 14
day_night_label_size = 12

corr_label_offset_x = 0.03
corr_label_offset_y = 0.03

panel_label_size = 14

panel_label1_offset_x = -0.02
panel_label1_offset_y = 0.05

panel_label2_offset_x = -0.05
panel_label2_offset_y = 0.05

panel_label3_offset_x = -0.05
panel_label3_offset_y = 0.05

colorbar_offset = 0.15
colorbar_width = 0.02

day_night_label_y = 10.0

day_label_x = str2timestamp("2020-01-16T08:00:00Z")
night_label_x = str2timestamp("2020-01-16T20:00:00Z")

### Generate the figure ###
print("Generating the figure...")
fig, ax = subplots(1, 1, figsize = (figwidth, figheight))

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
cmap = cc.cm.isolum
norm = Normalize(vmin = 0.0, vmax = max_freq)

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

    color = cmap(norm(mean_freq))
    ax.scatter(current_mode_df.index, (current_mode_df["frequency"] - mean_freq) * scale_factor + mode_order, color = color, s = marker_size_freq, label = mode_name)

# Add the day-night shading
print("Adding the day-night shading...")
add_day_night_shading(ax)

ax.text(day_label_x, day_night_label_y, "Day", 
    fontsize = axis_label_size, ha = "center", va = "center", fontweight = "bold", rotation = 90)

ax.text(night_label_x, day_night_label_y, "Night",
    fontsize = axis_label_size, ha = "center", va = "center", fontweight = "bold", rotation = 90)

# Set the axis limits
ax.set_xlim(starttime, endtime)

# Add the frequency scalebar
print("Adding the frequency scalebar...")
ax = add_vertical_scalebar(ax, (scalebar_x, scalebar_y), scalebar_length, scale_factor, (label_offset_x, label_offset_y), 
                                label_unit = "Hz")

# Format the x-axis labels
print("Formatting the x-axis labels...")

format_datetime_xlabels(ax,
                        plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing="5d", num_minor_ticks=5,
                        date_format="%Y-%m-%d",
                        rotation=0, ha="center", va="top", 
                        axis_label_size=axis_label_size)

# Format the y-axis labels
ax.set_ylabel("Mode order", fontsize = axis_label_size)

# Plot the colorbar
print("Plotting the colorbar...")
bbox = ax.get_position()
cbar_pos = [bbox.x0, bbox.y0 - colorbar_offset, bbox.width, colorbar_width]
add_colorbar(fig, cbar_pos,  "Mean frequency (Hz)", cmap = cmap, norm = norm, orientation = "horizontal")

# Add the title
print("Adding the title...")
title = "Frequency of all modes in the harmonic series as a function of time"
ax.set_title(title, fontsize = title_size, fontweight = "bold")

# Save the figure
print("Saving the figure...")
filename = "agu_2024_harmonic_relations.png"
save_figure(fig, filename)