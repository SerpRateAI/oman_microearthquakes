# Plot close-in spectra of the stationary resonance at a few stations

# Import
from os.path import join
from numpy import ones
from pandas import Timedelta
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec

from utils_basic import SPECTROGRAM_DIR as indir, EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import time2suffix
from utils_spec import StreamSTFTPSD
from utils_spec import get_spectrogram_file_suffix, read_geo_spectrograms, read_hydro_spectrograms
from utils_plot import add_power_colorbar, add_station_map, format_datetime_xlabels, format_freq_ylabels, get_power_colormap, save_figure

# Inputs
# Data
geo_to_plot = ["A01", "A16", "A19", "B01", "B19", "B20"]
hydro_a_to_plot = "06"
hydro_b_to_plot = "06"
starttime = "2020-01-13T19:30:00"
endtime = "2020-01-13T20:30:00"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

min_freq = 37.0
max_freq = 40.0

# Plotting
# Spectrograms
spec_width = 10.0
spec_height = 2.0

dbmin_geo = -10.0
dbmax_geo = 10.0

dbmin_hydro = -70.0
dbmax_hydro = -50.0

major_time_spacing = "15min"
num_minor_ticks = 3

major_freq_spacing = 1.0
num_minor_ticks_freq = 5

rotation = 5

spec_label_x = 0.01
spec_label_y = 0.9
spec_label_size = 12

cbar_offset = 0.01
cbar_width = 0.005

# Station map
borehole_size = 60.0
borehole_label_size = 15.0

station_size = 60.0
station_label_size = 15.0

edge_width = 1.0
axis_label_size = 12.0
tick_label_size = 10.0

# Read the spectrograms
# Geophones
print("Reading the geophone spectrograms...")
stream_spec_geo = StreamSTFTPSD()
suffix = get_spectrogram_file_suffix(window_length, overlap, downsample = downsample, downsample_factor = downsample_factor)
for station in geo_to_plot:
    inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station}_{suffix}.h5")
    stream_spec_sta = read_geo_spectrograms(inpath, 
                                        starttime = starttime, endtime = endtime, 
                                        min_freq = min_freq, max_freq = max_freq)
    trace_spec_sta = stream_spec_sta.get_total_power()
    stream_spec_geo.append(trace_spec_sta)

stream_spec_geo.to_db()
num_geo = len(stream_spec_geo)

# Hydrophones
print("Reading the hydrophone spectrograms...")
stream_spec_hydro = StreamSTFTPSD()
suffix = get_spectrogram_file_suffix(window_length, overlap, downsample = downsample, downsample_factor = downsample_factor)

location = hydro_a_to_plot
inpath = join(indir, f"whole_deployment_daily_hydro_spectrograms_A00_{suffix}.h5")
stream_spec_hydro_a = read_hydro_spectrograms(inpath,
                                              locations = hydro_a_to_plot,
                                              starttime = starttime, endtime = endtime, 
                                              min_freq = min_freq, max_freq = max_freq)
stream_spec_hydro_a.to_db()

location = hydro_b_to_plot
inpath = join(indir, f"whole_deployment_daily_hydro_spectrograms_B00_{suffix}.h5")
stream_spec_hydro_b = read_hydro_spectrograms(inpath,
                                              locations = hydro_b_to_plot,
                                              starttime = starttime, endtime = endtime, 
                                              min_freq = min_freq, max_freq = max_freq)
stream_spec_hydro_b.to_db()

# Plotting
print("Plotting...")
print("Setting up the figure...")
hw_ratio_spec = spec_height / spec_width
hw_ratio_map = (max_north - min_north) / (max_east - min_east)
map_height = spec_height * (num_geo + 2)
map_width = map_height / hw_ratio_map

fig_width = spec_width + map_width
fig_height = spec_height * (num_geo + 2)
fig = figure(figsize=(fig_width, fig_height))

# Create a GridSpec with 8 rows and 2 columns
gs = GridSpec(num_geo + 2, 2, width_ratios=[1, map_width / spec_width])

print("Plotting the geophone spectrograms...")
spec_axes = []
for i, station in enumerate(geo_to_plot):
    ax = fig.add_subplot(gs[i, 0])
    spec_axes.append(ax)

    trace_spec_sta = stream_spec_geo[i]

    cmap = get_power_colormap()
    timeax = trace_spec_sta.times
    freqax = trace_spec_sta.freqs
    data = trace_spec_sta.data

    color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_geo, vmax = dbmax_geo)

    format_freq_ylabels(ax,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_ticks_freq)
    
    ax.text(spec_label_x, spec_label_y, station, 
            va = "top", ha = "left", transform = ax.transAxes, 
            fontsize = spec_label_size, fontweight = "bold",
            bbox = dict(facecolor = 'white', alpha = 1.0))

    # Add the colorbar
    if i == 0:
        bbox = ax.get_position()
        position = [bbox.x1 + cbar_offset, bbox.y0, cbar_width, bbox.height]
        add_power_colorbar(fig, color, position, orientation = "vertical")

# Plot the hydrophone spectrogram in Hole A
print("Plotting the hydrophone spectrograms...")
ax = fig.add_subplot(gs[num_geo, 0])
spec_axes.append(ax)
trace_spec = stream_spec_hydro_a[0]

cmap = get_power_colormap()
timeax = trace_spec.times
freqax = trace_spec.freqs
data = trace_spec.data

color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_hydro, vmax = dbmax_hydro)

format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_ticks_freq)

ax.text(spec_label_x, spec_label_y, f"A00.{hydro_a_to_plot}",
        va = "top", ha = "left", transform = ax.transAxes, 
        fontsize = spec_label_size, fontweight = "bold",
        bbox = dict(facecolor = 'white', alpha = 1.0))

# Add the colorbar
bbox = ax.get_position()
position = [bbox.x1 + cbar_offset, bbox.y0, cbar_width, bbox.height]
add_power_colorbar(fig, color, position, orientation = "vertical")

# Plot the hydrophone spectrogram in Hole B
ax = fig.add_subplot(gs[num_geo + 1, 0])
spec_axes.append(ax)
trace_spec = stream_spec_hydro_b[0]

cmap = get_power_colormap()
timeax = trace_spec.times
freqax = trace_spec.freqs
data = trace_spec.data

color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_hydro, vmax = dbmax_hydro)

format_datetime_xlabels(ax,
                        major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_ticks,
                        rotation = rotation, va = "top", ha = "right")

format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_ticks_freq)

ax.text(spec_label_x, spec_label_y, f"B00.{hydro_b_to_plot}",
        va = "top", ha = "left", transform = ax.transAxes, 
        fontsize = spec_label_size, fontweight = "bold",
        bbox = dict(facecolor = 'white', alpha = 1.0))

# Make all spectrograms share the same time axis
for i in range(num_geo + 1):
    ax = spec_axes[i]
    ax.sharex(spec_axes[-1])
    ax.label_outer()

# Add the station map
print("Adding the station map...")
station_ax = fig.add_subplot(gs[:, 1])
add_station_map(station_ax, 
                stations_highlight = geo_to_plot,
                label_boreholes = True,
                station_size = station_size, station_label_size = station_label_size,
                borehole_size = borehole_size, borehole_label_size = borehole_label_size,
                edge_width = edge_width,
                axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Save the figure
print("Saving the figure...")
suffix_start = time2suffix(starttime)
suffix_end = time2suffix(endtime)
figname = f"stationary_resonance_close_in_spec_{suffix_start}_{suffix_end}_{min_freq:.0f}to{max_freq:.0f}hz.png"
save_figure(fig, figname)