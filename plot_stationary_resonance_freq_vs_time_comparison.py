# Plot the dmeaned frequency vs time curves for different stationary resonances
# Imports
from os.path import join
from pandas import Timedelta
from pandas import read_csv
from matplotlib.pyplot import subplots, get_cmap

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_plot import add_day_night_shading, add_vertical_scalebar, format_datetime_xlabels, get_interp_cat_colors, save_figure

# Inputs
# Data
names = ["SR25a", "SR38a", "SR50a", "SR63a", "SR76a", "SR102a", "SR114a", "SR127a", "SR140a", "SR152a", "SR165a", "SR178a", "SR191a"]

min_num_sta = 9

# Plotting
cmap = "tab20c"
begin_color_ind = 0
end_color_ind = 3

fig_width = 15
column_height = 0.75

ymin = -0.5
ymax = len(names) - 0.5

scale = 2.0

signal_label_offset_x = Timedelta("6h")
signal_label_offset_y = -0.2

scalebar_x = 0.9
scalebar_y = 0.02
scalebar_len = 0.1

scalebar_label_offset = 0.01

# Build the colormap
num_colors = len(names)
colors = get_interp_cat_colors(cmap, begin_color_ind, end_color_ind, num_colors)

# Read the data and subtract the mean
mean_freq_dict = {}
color_dict = {}
cmap = get_cmap("tab20")

for i, name in enumerate(names):
    filename = f"stationary_resonance_mean_freq_{name}_geo_num{min_num_sta}.csv"
    inpath = join(indir, filename)

    mean_freq_by_time = read_csv(inpath, index_col = "time", parse_dates = True)
    mean_freq_by_time["frequency"] = mean_freq_by_time["frequency"] - mean_freq_by_time["frequency"].mean()
    mean_freq_dict[name] = mean_freq_by_time

    color_dict[name] = colors[i]

# Plot the data
num_res = len(names)
fig, ax = subplots(1, 1, figsize=(fig_width, column_height * num_res))

for i, name in enumerate(names):
    mean_freq_by_time = mean_freq_dict[name] * scale + i
    color = color_dict[name]
    ax.scatter(mean_freq_by_time.index, mean_freq_by_time["frequency"], s = 1, color = color, label = name)

    ax.text(starttime + signal_label_offset_x, i + signal_label_offset_y, name, va = "center", ha = "left", fontsize = 12, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

ax.set_xlim(starttime, endtime)
ax.set_ylim(ymin, ymax)
format_datetime_xlabels(ax,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing = "1d", num_minor_ticks = 4,
                        rotation = 30, va = "top", ha = "right")

ax.yaxis.set_ticks([])

# Add a vertical scalebar
coords = (scalebar_x, scalebar_y)
label_offsets = (scalebar_label_offset, 0)
add_vertical_scalebar(ax, coords, scalebar_len, scale, label_offsets, label_unit = "Hz")

# Add the day-night shading
add_day_night_shading(ax)
                      
# Save the figure
figname = f"stationary_resonance_mean_freq_vs_time_comparison_geo.png"
save_figure(fig, figname)