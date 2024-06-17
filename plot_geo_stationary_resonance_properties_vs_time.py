# Plot the variation of stationary harmonics as a function of time for all geophone stations
# The stations are plotted in the order of increasing north coordinate
# Imports
from os.path import join
from pandas import Timedelta
from pandas import read_csv, read_hdf
from matplotlib.pyplot import figure, subplots, get_cmap
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_geo_sunrise_sunset_times
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import add_day_night_shading, add_colorbar, add_station_map, format_datetime_xlabels, format_north_ylabels, save_figure

# Inputs
# Name of the stationary resonance
name = "SR38a"

# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral peaks
prom_threshold = 10.0
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200.0

# Grouping
count_threshold = 9

# Plotting
# Curves
factor = 7 # Multiplicative factor for the curves

min_freq_plot = 38.05
max_freq_plot = 38.35

min_db = -5.0
max_db = 25.0

min_qf = 600.0
max_qf = 900.0

curve_plot_width = 7.0
curve_plot_hw_ratio = 0.8

panel_gap = 0.5

marker_size = 1.0

linecolor = "lightgray"
linewidth = 0.1

starttime_offset = "12h"
starttime = starttime - Timedelta(starttime_offset)

station_label_offset = "2h"
station_label_time = starttime + Timedelta(station_label_offset)

station_label_size = 6

direction_label_offset = 0.1
y_lim_offset = 0.5

cbar_offset = 0.01
freq_tick_spacing = 0.1
power_tick_spacing = 5.0
qf_tick_spacing = 100.0

# map_ylabel_offset = 0.6

axis_label_size = 10
tick_label_size = 8

# Station map
stations_highlight = {"A01": (-3, 3), "A16": (3, -3), "A19": (3, -3), "B01":(-3, 3), "B19": (3, 3), "B20": (3, 3)}

# Read the sunrise and sunset times
print("Reading the sunrise and sunset times...")
sun_df = get_geo_sunrise_sunset_times()

# Read the stationary resonance properties
print("Reading the stationary resonance properties...")
filename_in = f"geo_stationary_resonance_properties_{name}.h5"
inpath = join(indir, filename_in)
resonance_df = read_hdf(inpath, key = "properties")

# Plot the frequency vs time for each station
coord_df = get_geophone_coords()
coord_df.sort_values(by = "north", inplace = True)

# Generate the figure and axes
print("Plotting the frequency vs time for each station...")
print("Generating the figure and axes...")
map_hw_ratio = (max_north - min_north) / (max_east - min_east)
width_ratios = [1, curve_plot_hw_ratio / map_hw_ratio]

gs = GridSpec(1, 2, width_ratios=width_ratios)

fig_width = curve_plot_width * (1 + curve_plot_hw_ratio / map_hw_ratio + panel_gap)
fig_height = curve_plot_width * curve_plot_hw_ratio
fig = figure(figsize = (fig_width, fig_height))

### Plot the frequency vs time for each station ###
curve_ax = fig.add_subplot(gs[0, 0])

mean_freq_plot = (min_freq_plot + max_freq_plot) / 2
max_freq_perturb = max_freq_plot - mean_freq_plot
min_freq_perturb = min_freq_plot - mean_freq_plot
i = 0
for index, row in coord_df.iterrows():
    station = index
    print(f"Plotting {station}...")

    # Get the power of all time points and those that are peaks
    resonance_sta_df = resonance_df.loc[resonance_df["station"] == station]
    times = resonance_sta_df.index
    freqs = resonance_sta_df["frequency"]

    # Plot the frequency vs time curves
    freqs_to_plot = (freqs - mean_freq_plot) * factor + i
    #curve_ax.plot(times, freqs_to_plot, color = linecolor, linewidth = linewidth, zorder = 1)

    # Plot the frequency vs time dots
    freq_color = curve_ax.scatter(times, freqs_to_plot, 
                                  c = freqs, s = marker_size,
                                  cmap = "coolwarm", vmin = min_freq_plot, vmax = max_freq_plot,
                                  edgecolors = None, linewidths = linewidth, 
                                  zorder = 2)

    # Plot the station labels
    curve_ax.text(station_label_time, i, station, color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "left")

    # Update the counter
    i += 1

# Add the day-night shading
print("Adding the day-night shading...")
curve_ax = add_day_night_shading(curve_ax, sun_df)

# Plot the direction labels
min_y = -1 - y_lim_offset
max_y = len(coord_df) + y_lim_offset

direction_label_time = station_label_time
north_label_y = max_y - direction_label_offset
south_label_y = min_y + direction_label_offset

curve_ax.text(direction_label_time, north_label_y, "North", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "top", horizontalalignment = "left")
curve_ax.text(direction_label_time, south_label_y, "South", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "bottom", horizontalalignment = "left")

# Turn off the y ticks
curve_ax.set_yticks([])

# Set the axis labels and limits
curve_ax.set_xlim(starttime, endtime)
format_datetime_xlabels(curve_ax, 
                        date_format = "%Y-%m-%d", major_tick_spacing = "1d", num_minor_ticks = 4, 
                        axis_label_size=10, tick_label_size=8,
                        rotation = 30, va = "top", ha = "right")

curve_ax.set_ylim(min_y, max_y)

# Add a vertical colorbar on the right
bbox = curve_ax.get_position()
position = [bbox.x1 + cbar_offset, bbox.y0, cbar_offset / 2, bbox.height]
cbar = add_colorbar(fig, freq_color, "Frequency (Hz)", position,
                        orientation = "vertical", axis_label_size = 8, tick_label_size = 6,
                        major_tick_spacing = freq_tick_spacing)

# Add the title
title = f"{name}, frequency"
curve_ax.set_title(title, fontsize = 12, fontweight = "bold")

# Add a station map on the right
map_ax = fig.add_subplot(gs[0, 1])
add_station_map(map_ax,
                stations_highlight = stations_highlight,
                axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# map_ax.set_ylabel(map_ax.yaxis.get_label().get_text(), labelpad=0)

# Save the figure
print("Saving the figure...")
figname = f"geo_stationary_resonance_freq_vs_time_{name}.png"
save_figure(fig, figname, dpi = 600)
print("")

### Plot the power vs time for each station ###
print("Plotting the power vs time for each station...")
print("Generating the figure and axes...")

fig = figure(figsize = (fig_width, fig_height))

# Plot the power vs time for each station
curve_ax = fig.add_subplot(gs[0, 0])

i = 0
for index, row in coord_df.iterrows():
    station = index
    print(f"Plotting {station}...")

    # Get the power of all time points and those that are peaks
    resonance_sta_df = resonance_df.loc[resonance_df["station"] == station]
    times = resonance_sta_df.index
    freqs = resonance_sta_df["frequency"]
    powers = resonance_sta_df["power"]

    # Plot the power vs time curves
    freqs_to_plot = (freqs - mean_freq_plot) * factor + i
    #curve_ax.plot(times, powers_to_plot, color = linecolor, linewidth = linewidth, zorder = 1)

    # Plot the power vs time dots
    power_color = curve_ax.scatter(times, freqs_to_plot, 
                                  c = powers, s = marker_size,
                                  cmap = "inferno", vmin = min_db, vmax = max_db,
                                  edgecolors = None, linewidths = linewidth, 
                                  zorder = 2)

    # Plot the station labels
    curve_ax.text(station_label_time, i, station, color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "left")

    # Update the counter
    i += 1

# Add the day-night shading
print("Adding the day-night shading...")
curve_ax = add_day_night_shading(curve_ax, sun_df)

# Plot the direction labels
min_y = -1 - y_lim_offset
max_y = len(coord_df) + y_lim_offset

direction_label_time = station_label_time
north_label_y = max_y - direction_label_offset
south_label_y = min_y + direction_label_offset

curve_ax.text(direction_label_time, north_label_y, "North", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "top", horizontalalignment = "left")
curve_ax.text(direction_label_time, south_label_y, "South", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "bottom", horizontalalignment = "left")

# Turn off the y ticks
curve_ax.set_yticks([])
curve_ax.set_yticklabels([])
curve_ax.set_ylabel("")
curve_ax.yaxis.label.set_visible(False)

# Set the axis labels and limits
curve_ax.set_xlim(starttime, endtime)
format_datetime_xlabels(curve_ax, 
                        date_format = "%Y-%m-%d", major_tick_spacing = "1d", num_minor_ticks = 4, 
                        axis_label_size=10, tick_label_size=8,
                        rotation = 30, va = "top", ha = "right")

curve_ax.set_ylim(min_y, max_y)

# Add a vertical colorbar on the right
bbox = curve_ax.get_position()
position = [bbox.x1 + cbar_offset, bbox.y0, cbar_offset / 2, bbox.height]
cbar = add_colorbar(fig, power_color, "Power (db)", position,
                        orientation = "vertical", axis_label_size = 8, tick_label_size = 6,
                        major_tick_spacing = power_tick_spacing)

# Add the title
title = f"{name}, power"
curve_ax.set_title(title, fontsize = 12, fontweight = "bold")

# Add a station map on the right
map_ax = fig.add_subplot(gs[0, 1])
add_station_map(map_ax,
                stations_highlight = stations_highlight,
                axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Save the figure
print("Saving the figure...")
figname = f"geo_stationary_resonance_power_vs_time_{name}.png"
save_figure(fig, figname, dpi = 600)
print("")

### Plot the quality factor vs time for each station ###
print("Plotting the quality factor vs time for each station...")
print("Generating the figure and axes...")
fig = figure(figsize = (fig_width, fig_height))

# Plot the quality factor vs time for each station
curve_ax = fig.add_subplot(gs[0, 0])

i = 0
for index, row in coord_df.iterrows():
    station = index
    print(f"Plotting {station}...")

    # Get the power of all time points and those that are peaks
    resonance_sta_df = resonance_df.loc[resonance_df["station"] == station]
    times = resonance_sta_df.index
    freqs = resonance_sta_df["frequency"]
    qualities = resonance_sta_df["quality_factor"]

    # Plot the quality factor vs time curves
    freqs_to_plot = (freqs - mean_freq_plot) * factor + i
    #curve_ax.plot(times, qualities_to_plot, color = linecolor, linewidth = linewidth, zorder = 1)

    # Plot the quality factor vs time dots
    quality_color = curve_ax.scatter(times, freqs_to_plot, 
                                     c = qualities, s = marker_size,
                                     cmap = "viridis", vmin = min_qf, vmax = max_qf,
                                     edgecolors = None, linewidths = linewidth, 
                                     zorder = 2)

    # Plot the station labels
    curve_ax.text(station_label_time, i, station, color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "left")

    # Update the counter
    i += 1

# Add the day-night shading
print("Adding the day-night shading...")
curve_ax = add_day_night_shading(curve_ax, sun_df)

# Plot the direction labels
min_y = -1 - y_lim_offset
max_y = len(coord_df) + y_lim_offset

direction_label_time = station_label_time
north_label_y = max_y - direction_label_offset
south_label_y = min_y + direction_label_offset

curve_ax.text(direction_label_time, north_label_y, "North", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "top", horizontalalignment = "left")
curve_ax.text(direction_label_time, south_label_y, "South", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "bottom", horizontalalignment = "left")

# Turn off the y ticks
curve_ax.set_yticks([])
curve_ax.set_yticklabels([])
curve_ax.set_ylabel("")
curve_ax.yaxis.label.set_visible(False)

# Set the axis labels and limits
curve_ax.set_xlim(starttime, endtime)
format_datetime_xlabels(curve_ax, 
                        date_format = "%Y-%m-%d", major_tick_spacing = "1d", num_minor_ticks = 4, 
                        axis_label_size=10, tick_label_size=8,
                        rotation = 30, va = "top", ha = "right")

curve_ax.set_ylim(min_y, max_y)

# Add a vertical colorbar on the right
bbox = curve_ax.get_position()
position = [bbox.x1 + cbar_offset, bbox.y0, cbar_offset / 2, bbox.height]
cbar = add_colorbar(fig, quality_color, "Quality factor", position,
                        orientation = "vertical", axis_label_size = 8, tick_label_size = 6,
                        major_tick_spacing = qf_tick_spacing)

# Add the title
title = f"{name}, quality factor"
curve_ax.set_title(title, fontsize = 12, fontweight = "bold")

# Add a station map on the right
map_ax = fig.add_subplot(gs[0, 1])
add_station_map(map_ax, 
                stations_highlight = stations_highlight,
                axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Save the figure
print("Saving the figure...")
figname = f"geo_stationary_resonance_qf_vs_time_{name}.png"

save_figure(fig, figname, dpi = 600)
print("")



