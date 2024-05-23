# Plot the variation of a specific geophone spectral vs time
# Each station is color-coded by its west-east and north-south coordinates
# Imports
from os.path import join
from pandas import Timedelta
from pandas import read_csv, read_hdf
from matplotlib.pyplot import subplots, get_cmap
from matplotlib.colors import Normalize

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_geo_sunrise_sunset_times
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import add_day_night_shading, add_power_colorbar, add_station_map, format_datetime_xlabels, save_figure

# Inputs
# Cumulative frequency count index
freq_count_ind = 0

# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = True
downsample_factor = 60

# Spectral peaks
prom_threshold = 5.0
rbw_threshold = 0.2

min_freq_peak = None
max_freq_peak = None

# Grouping
count_threshold = 4

# Plotting
factor = 0.03 # Multiplicative factor for the curves

width = 12
height = 8

marker_size = 3.0

min_db = -10.0
max_db = 10.0

linecolor = "lightgray"
linewidth = 0.1

starttime_offset = "12h"
starttime = starttime - Timedelta(starttime_offset)

station_label_offset = "2h"
station_label_time = starttime + Timedelta(station_label_offset)

direction_label_offset = 0.1
y_lim_offset = 0.5

cbar_offset = 0.01

map_offset = 0.03

# Read the sunrise and sunset times
print("Reading the sunrise and sunset times...")
sun_df = get_geo_sunrise_sunset_times()

# Read the cumulative frequency counts
print("Reading the cumulative frequency counts...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

filename_in = f"geo_spec_peak_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.csv"

inpath = join(indir, filename_in)
cum_count_df = read_csv(inpath)

freq_out =  cum_count_df.iloc[freq_count_ind]['frequency']

print(f"Plotting the power at {freq_out:.1f} Hz as a function of time...")

# Read the spectral power vs time
print("Reading the spectral power vs time...")
filename_in = f"geo_spec_power_vs_time_{suffix_spec}_{suffix_peak}_freq{freq_out:.1f}hz.h5"
inpath = join(indir, filename_in)
power_df = read_hdf(inpath, key = "power")

# Plot the power vs time for each station
coord_df = get_geophone_coords()
coord_df.sort_values(by = "north", inplace = True)
coord_df.reset_index(drop = True, inplace = True)

# Generate the figure and axes
print("Generating the figure and axes...")
fig, power_ax = subplots(1, 1, figsize = (width, height))

# Loop over all stations
print("Plotting the power vs time for each station...")

for i, row in coord_df.iterrows():
    station = row["name"]

    # Get the power of all time points and those that are peaks
    power_df_station = power_df.loc[station]
    times = power_df_station.index.get_level_values("time")
    powers = power_df_station["power"].values
    is_peak = power_df_station["is_peak"].values
    power_peaks = powers[is_peak]
    time_peaks = times[is_peak]

    # Plot the power vs time
    powers_to_plot = factor * powers + i
    power_ax.plot(times, powers_to_plot, color = linecolor, linewidth = linewidth, zorder = 1)

    # Plot the peaks
    power_peaks_to_plot = factor * power_peaks + i
    power_color = power_ax.scatter(time_peaks, power_peaks_to_plot, 
                            c = power_peaks, s = marker_size, 
                            cmap = "inferno", vmin = min_db, vmax = max_db,
                            edgecolors = linecolor, linewidths = linewidth, 
                            zorder = 2)

    # Plot the station labels
    power_ax.text(station_label_time, i, station, color = "black", fontsize = 8, fontweight = "bold", verticalalignment = "center", horizontalalignment = "left")

# Add the day-night shading
print("Adding the day-night shading...")
power_ax = add_day_night_shading(power_ax, sun_df)

# Plot the direction labels
min_y = -1 - y_lim_offset
max_y = len(coord_df) + y_lim_offset

direction_label_time = station_label_time
north_label_y = max_y - direction_label_offset
south_label_y = min_y + direction_label_offset

power_ax.text(direction_label_time, north_label_y, "North", color = "black", fontsize = 8, fontweight = "bold", verticalalignment = "top", horizontalalignment = "left")
power_ax.text(direction_label_time, south_label_y, "South", color = "black", fontsize = 8, fontweight = "bold", verticalalignment = "bottom", horizontalalignment = "left")

# Turn off the y ticks
power_ax.set_yticks([])

# Set the axis labels and limits
power_ax.set_xlim(starttime, endtime)
format_datetime_xlabels(power_ax, 
                        date_format = "%Y-%m-%d", major_tick_spacing = "1d", minor_tick_spacing = "6h", 
                        axis_label_size=10, tick_label_size=8,
                        rotation = 15, vertical_align = "top", horizontal_align = "right")

power_ax.set_ylim(min_y, max_y)

# Add a vertical colorbar on the right
bbox = power_ax.get_position()
position = [bbox.x1 + cbar_offset, bbox.y0, cbar_offset / 2, bbox.height]
cbar = add_power_colorbar(fig, power_color, position, 
                   orientation = "vertical", axis_label_size = 10, tick_label_size = 8)

# Add the title
title = f"Geophone Spectral Power at {freq_out:.1f} Hz"
power_ax.set_title(title, fontsize = 12, fontweight = "bold")

# Add a station map on the right
print("Adding the station map...")
map_height = bbox.height
map_width = bbox.height / (max_north - min_north) * (max_east - min_east)

position = [bbox.x1 + map_offset, bbox.y0, map_width, map_height]
map_ax = fig.add_axes(position)
add_station_map(map_ax, coord_df)

# Save the figure
print("Saving the figure...")
figname = f"geo_spec_power_vs_time_{suffix_spec}_{suffix_peak}_freq{freq_out:.1f}hz.png"
save_figure(fig, figname, dpi = 300)
