# Plot the frequency ratios between the stationary resonances with harmonic relations

# Imports
from os.path import join
from pandas import read_csv, DataFrame, Timedelta
from matplotlib.pyplot import subplots, get_cmap
from matplotlib.ticker import MultipleLocator


from utils_basic import SPECTROGRAM_DIR as indir
from utils_plot import format_datetime_xlabels, get_interp_cat_colors, save_figure

# Inputs
# Data
fund_name = "SR25a"
over_names = ["SR38a", "SR50a", "SR63a", "SR76a", "SR102a", "SR114a", "SR127a", "SR140a", "SR152a", "SR165a", "SR178a", "SR191a"]

min_num_sta = 9

# Plotting
width = 12
height = 8

cmap = "tab20c"
begin_color_ind = 16
end_color_ind = 19

harmonics = [1.5, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]

marker_size = 5

ymin = 1.0

title_size = 15

label_offset_x = "18h"
label_offset_y = 0.1
label_size = 10

# Read the average frequencies of the stationary resonances
all_names = [fund_name] + over_names

# Build the colormap
num_colors = len(over_names)
colors = get_interp_cat_colors(cmap, begin_color_ind, end_color_ind, num_colors)

# Read the data 
mean_freq_dict = {}

for i, name in enumerate(all_names):
    filename = f"stationary_resonance_mean_freq_{name}_geo_num{min_num_sta}.csv"
    inpath = join(indir, filename)

    mean_freq_df = read_csv(inpath, index_col = "time", parse_dates = True)
    mean_freq_dict[name] = mean_freq_df

# Compute the ratio between the higher frequencies and the fundamental one
ratio_dict = {}
color_dict = {}
for i, name in enumerate(over_names):
    mean_freq_over_df = mean_freq_dict[name]
    mean_freq_fund_df = mean_freq_dict[fund_name]

    # Resample the overtone frquency to the fundamental frequency time grid
    mean_freq_over_df = mean_freq_over_df.reindex(mean_freq_fund_df.index, method = "nearest", limit = 1)

    ratio = mean_freq_over_df["frequency"] / mean_freq_fund_df["frequency"]
    ratio_df = DataFrame({"time": mean_freq_over_df.index, "ratio": ratio})
    ratio_df.set_index("time", inplace = True)

    ratio_dict[name] = ratio_df
    color_dict[name] = colors[i]

# Plot the data
fig, ax = subplots(1, 1, figsize=(width, height))
ymax = harmonics[-1] + 0.5

for i, name in enumerate(over_names):
    # Plot the data
    ratio_df = ratio_dict[name]
    color = color_dict[name]
    ax.scatter(ratio_df.index, ratio_df["ratio"], s = marker_size, color = color, label = name)

    # Plot the harmonic relations
    h = harmonics[i]
    ax.axhline(h, color = "crimson", linestyle = "--", linewidth = 1.0)

    # Add the label
    label_x = ratio_df.index[0] - Timedelta(label_offset_x)
    label_y = h + label_offset_y
    ax.text(label_x, label_y, name, va = "bottom", ha = "left", fontsize = label_size, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))
    
ax.set_ylim(ymin, ymax)
ax.yaxis.set_major_locator(MultipleLocator(0.5))

ax.set_ylabel("Frequency ratio")

format_datetime_xlabels(ax,
                        date_format = "%Y-%m-%d",
                        rotation = 30, ha = "right", va = "top",
                        major_tick_spacing = "1d", num_minor_ticks = 4)

ax.set_title(f"Frequency ratios to {fund_name}", fontsize = title_size, fontweight = "bold")

# Save the figure
save_figure(fig, f"stationary_resonance_harmonic_relations_{fund_name}.png")