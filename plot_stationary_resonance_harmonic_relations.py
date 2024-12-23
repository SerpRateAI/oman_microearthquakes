# Plot the mean frequency vs time plots of a set of stationary resonances and their harmonic relations 

# Import the necessary librariess
from os.path import join
from pandas import read_csv, read_hdf
from matplotlib.pyplot import subplots, get_cmap
from matplotlib.colors import Normalize

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_plot import add_colorbar, add_vertical_scalebar, add_day_night_shading, format_datetime_xlabels, get_cmap_segment, save_figure

# Inputs
# Data
base_number = 2
base_mode = "PR02549"

# Plotting
cmap_name = "Blues_r"
max_freq = 200.0
max_cmap_portion = 0.8

panel_height = 5.0
panel_width = 15.0
factor = 2.0

marker_size = 5

linewidth = 0.5

scalebar_x = 0.02
scalebar_y = 0.95
scalebar_length = 0.2

label_offset_x = 0.01
label_offset_y = 0.00

# Read the harmonic series information
print(f"Reading the information of the harmonic series with {base_mode} as the base mode {base_number}...")
filename = f"stationary_harmonic_series_{base_mode}_base{base_number:d}.csv"
inpath = join(indir, filename)
harmonic_series_df = read_csv(inpath)

# Plotting
print("Generating the subplots...")
fig, axes = subplots(nrows = 2, ncols = 1, figsize = (panel_width, 2 * panel_height), sharex = True)

ax_freq = axes[0]
ax_num = axes[1]

print("Plotting each mode...")
cmap = get_cmap_segment(cmap_name, 0.0, max_cmap_portion)
for i, row in harmonic_series_df.iterrows():
    mode_name = row["name"]
    if not row["detected"]:
        print(f"Skipping the undetected mode {mode_name}...")
        continue

    # Read the frequencies of the mode
    print(f"Plotting the mode {mode_name}...")
    filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
    inpath = join(indir, filename)
    current_mode_df = read_hdf(inpath, key = "properties")

    mode_number = row["harmonic_number"]
    mean_freq = current_mode_df["frequency"].mean()

    if mode_number == base_number:
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
    ax_freq.scatter(current_mode_df.index, (current_mode_df["frequency"] - mean_freq) * factor + mode_number, color = color, s = marker_size, label = mode_name)
    ax_num.scatter(timeax_common, freq_ratios, color = color, s = marker_size, label = mode_name)

    ax_num.axhline(y = mode_number / base_number, color = "crimson", linestyle = "--", linewidth = linewidth)

# Add the day-night shading
print("Adding the day-night shading...")
ax_freq = add_day_night_shading(ax_freq)
ax_num = add_day_night_shading(ax_num)

# Set the axis limits
ax_freq.set_xlim(starttime, endtime)

# Add the frequency scalebar
print("Adding the frequency scalebar...")
ax_freq = add_vertical_scalebar(ax_freq, (scalebar_x, scalebar_y), scalebar_length, factor, (label_offset_x, label_offset_y), 
                                label_unit = "Hz")

# Format the x-axis labels
print("Formatting the x-axis labels...")

format_datetime_xlabels(ax_num,
                        major_tick_spacing="1d", num_minor_ticks=4,
                        date_format="%Y-%m-%d",
                        rotation=30, ha="right", va="top")

# Format the y-axis labels
ax_freq.set_ylabel("Mode number")
ax_num.set_ylabel(f"Frequency ratio to {base_mode}")


# Plot the colorbar
print("Plotting the colorbar...")
bbox = ax_freq.get_position()
cbar_pos = [bbox.x1 + 0.02, bbox.y0, 0.01, bbox.height]
add_colorbar(fig, cbar_pos,  "Mean frequency (Hz)", cmap = cmap, norm = norm)

# Set the title
print("Setting the title...")
title = f"Harmonic relations between the stationary resonances with {base_mode} as Mode {base_number}"
fig.suptitle(title, fontsize = 14, fontweight = "bold", y = 0.92)

# Save the figure
print("Saving the figure...")
filename = f"stationary_resonance_harmonic_relations_{base_mode}_base{base_number}.png"
save_figure(fig, filename)

    


    