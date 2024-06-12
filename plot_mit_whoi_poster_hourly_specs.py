# Plot the hourly spectrograms for the MIT-WHOI retreat poster


from os.path import join
from pandas import Timestamp
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import StreamSTFTPSD
from utils_spec import get_spectrogram_file_suffix, read_geo_spectrograms, read_hydro_spectrograms, string_to_time_label
from utils_plot import add_power_colorbar, format_datetime_xlabels, format_freq_ylabels, get_power_colormap, save_figure

# Inputs

# Data
geo_stations_a = ["A01", "A16"]
geo_stations_b = ["B01", "B19"]
hydro_location_a = "06"
hydro_location_b = "06"

starttime = "2020-01-13T20:00:00"
endtime = "2020-01-13T21:00:00"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

min_freq = 25.0
max_freq = 85.0

# Plotting
fig_width = 18.0
fig_height = 15.0

dbmin_geo = -20.0
dbmax_geo = 10.0

dbmin_hydro = -80.0
dbmax_hydro = -60.0

title_size = 30.0
axis_label_size = 25.0
tick_label_size = 20.0

station_label_size = 25.0
station_label_x = 0.02
station_label_y = 0.96

param_label_size = 20.0
param_label_x = 0.99
param_label_y = 0.02

cbar_offset = 0.02
cbar_width = 0.01

date_format = "%H:%M:%S"

major_time_spacing = "15min"
minor_time_spacing = "5min"

major_freq_spacing = 10.0
minor_freq_spacing = 2.0

major_tick_length = 10.0
minor_tick_length = 5.0

spine_width_spec = 3.0
spine_width_cbar = 2.0
tick_width = 2.0
box_width = 3.0

spine_color = "cyan"

box_x1 = Timestamp("2020-01-13 20:00:00")
box_x2 = Timestamp("2020-01-13 20:02:30")
box_y1 = min_freq
box_y2 = max_freq
box_color = "lime"

suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)

# Read the geophone spectrograms
print(f"Reading the example geophone spectrograms of Array A...")
geo_spec_a = StreamSTFTPSD()
for station in geo_stations_a:
    filename = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
    inpath = join(indir, filename)

    stream_spec = read_geo_spectrograms(inpath, starttime = starttime, endtime = endtime, min_freq = min_freq, max_freq = max_freq)
    trace_spec = stream_spec.get_total_power()
    trace_spec.to_db()
    geo_spec_a.append(trace_spec)

print(f"Reading the example geophone spectrograms of Array B...")
geo_spec_b = StreamSTFTPSD()
for station in geo_stations_b:
    filename = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
    inpath = join(indir, filename)

    stream_spec = read_geo_spectrograms(inpath, starttime = starttime, endtime = endtime, min_freq = min_freq, max_freq = max_freq)
    trace_spec = stream_spec.get_total_power()
    trace_spec.to_db()
    geo_spec_b.append(trace_spec)

# Read the hydrophone spectrograms
print(f"Reading the example hydrophone spectrograms of Borehole A...")
filename = f"whole_deployment_daily_hydro_spectrograms_A00_{suffix_spec}.h5"
inpath = join(indir, filename)

hydro_spec_a = read_hydro_spectrograms(inpath, starttime = starttime, endtime = endtime, min_freq = min_freq, max_freq = max_freq)
hydro_spec_a.to_db()

print(f"Reading the example hydrophone spectrograms of Borehole B...")
filename = f"whole_deployment_daily_hydro_spectrograms_B00_{suffix_spec}.h5"
inpath = join(indir, filename)

hydro_spec_b = read_hydro_spectrograms(inpath, starttime = starttime, endtime = endtime, min_freq = min_freq, max_freq = max_freq)
hydro_spec_b.to_db()

# Plotting
# Generate the figure and axes
fig, axes = subplots(3, 2, figsize = (fig_width, fig_height), sharex = True, sharey = True)
cmap = get_power_colormap()

# Plot the geophone spectrograms
# First A station
print(f"Plotting the geophone spectrograms...")
ax = axes[0, 0]
trace_spec = geo_spec_a[0]
timeax = trace_spec.times
freqax = trace_spec.freqs
freq_interval = freqax[1] - freqax[0]
data = trace_spec.data
station = geo_stations_a[0]

power_color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_geo, vmax = dbmax_geo)
ax.text(station_label_x, station_label_y, station, 
        transform = ax.transAxes, va = "top", ha = "left",
        fontsize = station_label_size, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))

ax.text(param_label_x, param_label_y, f"Window length: {window_length:.0f} s\nFreq. sampling: {freq_interval:.3f} Hz",
        color = "white",
        transform = ax.transAxes, va = "bottom", ha = "right",
        fontsize = param_label_size)

# Add the boxes
rect = Rectangle((box_x1, box_y1), box_x2 - box_x1, box_y2 - box_y1,
                    edgecolor = box_color, facecolor = "none", lw = box_width)
ax.add_patch(rect)

format_datetime_xlabels(ax,
                        label = False,
                        date_format = date_format,
                        major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing,
                        vertical_align = "top", horizontal_align = "right", rotation = 5.0,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                        major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

format_freq_ylabels(ax, 
                    major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, 
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                    major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

for spine in ax.spines.values():
    spine.set_linewidth(spine_width_spec)
    spine.set_color(spine_color)

# First B station
ax = axes[0, 1]
trace_spec = geo_spec_b[0]
timeax = trace_spec.times
freqax = trace_spec.freqs
freq_interval = freqax[1] - freqax[0]
data = trace_spec.data
station = geo_stations_b[0]

power_color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_geo, vmax = dbmax_geo)
ax.text(station_label_x, station_label_y, station, 
        transform = ax.transAxes, va = "top", ha = "left",
        fontsize = station_label_size, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))

# Add the boxes
rect = Rectangle((box_x1, box_y1), box_x2 - box_x1, box_y2 - box_y1,
                    edgecolor = box_color, facecolor = "none", lw = box_width)
ax.add_patch(rect)

format_datetime_xlabels(ax,
                        label = False,
                        date_format = date_format,
                        major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing,
                        vertical_align = "top", horizontal_align = "right", rotation = 5.0,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                        major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

format_freq_ylabels(ax, label = False,
                    major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, 
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                    major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

for spine in ax.spines.values():
    spine.set_linewidth(spine_width_spec)
    spine.set_color(spine_color)

# Add the colorbar
bbox = ax.get_position()
position = [bbox.x1 + cbar_offset, bbox.y0, cbar_width, bbox.height]
cbar = add_power_colorbar(fig, power_color, position, 
                          orientation = "vertical",
                          tick_spacing = 10.0,
                          axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                          tick_length = major_tick_length, tick_width = tick_width)

for spine in cbar.ax.spines.values():
    spine.set_linewidth(spine_width_spec)

# Second A station
ax = axes[1, 0]
trace_spec = geo_spec_a[1]
timeax = trace_spec.times
freqax = trace_spec.freqs
freq_interval = freqax[1] - freqax[0]
data = trace_spec.data
station = geo_stations_a[1]

power_color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_geo, vmax = dbmax_geo)
ax.text(station_label_x, station_label_y, station,
        transform = ax.transAxes, va = "top", ha = "left",
        fontsize = station_label_size, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))

# Add the boxes
rect = Rectangle((box_x1, box_y1), box_x2 - box_x1, box_y2 - box_y1,
                edgecolor = box_color, facecolor = "none", lw = box_width)
ax.add_patch(rect)

format_datetime_xlabels(ax,
                        label = False,
                        date_format = date_format,
                        major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing,
                        vertical_align = "top", horizontal_align = "right", rotation = 5.0,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                        major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing,
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                    major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

for spine in ax.spines.values():
    spine.set_linewidth(spine_width_spec)
    spine.set_color(spine_color)

# Second B station
ax = axes[1, 1]
trace_spec = geo_spec_b[1]
timeax = trace_spec.times
freqax = trace_spec.freqs
freq_interval = freqax[1] - freqax[0]
data = trace_spec.data
station = geo_stations_b[1]

power_color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_geo, vmax = dbmax_geo)
ax.text(station_label_x, station_label_y, station,
        transform = ax.transAxes, va = "top", ha = "left",
        fontsize = station_label_size, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))

# Add the boxes
rect = Rectangle((box_x1, box_y1), box_x2 - box_x1, box_y2 - box_y1,
                edgecolor = box_color, facecolor = "none", lw = box_width)
ax.add_patch(rect)

format_datetime_xlabels(ax,
                        label = False,
                        date_format = date_format,
                        major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing,
                        vertical_align = "top", horizontal_align = "right", rotation = 5.0,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                        major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

format_freq_ylabels(ax, label = False,
                    major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing,
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                    major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

for spine in ax.spines.values():
    spine.set_linewidth(spine_width_spec)
    spine.set_color(spine_color)

# Plot the hydrophone spectrograms
print(f"Plotting the hydrophone spectrograms...")

# Borehole A
ax = axes[2, 0]

hydro_spec = hydro_spec_a.select(location = hydro_location_a)[0]
timeax = hydro_spec.times
freqax = hydro_spec.freqs
data = hydro_spec.data

power_color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_hydro, vmax = dbmax_hydro)
ax.text(station_label_x, station_label_y, f"A00.{hydro_location_a}",
        transform = ax.transAxes, va = "top", ha = "left",
        fontsize = station_label_size, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))

# Add the boxes
rect = Rectangle((box_x1, box_y1), box_x2 - box_x1, box_y2 - box_y1,
                    edgecolor = box_color, facecolor = "none", lw = box_width)
ax.add_patch(rect)

format_datetime_xlabels(ax,
                    major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing,
                    vertical_align = "top", horizontal_align = "right", rotation = 5.0,
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                    major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)


format_freq_ylabels(ax, 
                    major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, 
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                    major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

for spine in ax.spines.values():
    spine.set_linewidth(spine_width_cbar)
    spine.set_color(spine_color)

# Borehole B
ax = axes[2, 1]

hydro_spec = hydro_spec_b.select(location = hydro_location_b)[0]
timeax = hydro_spec.times
freqax = hydro_spec.freqs
data = hydro_spec.data

power_color = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin_hydro, vmax = dbmax_hydro)
ax.text(station_label_x, station_label_y, f"B00.{hydro_location_b}",
        transform = ax.transAxes, va = "top", ha = "left",
        fontsize = station_label_size, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))

# Add the boxes
rect = Rectangle((box_x1, box_y1), box_x2 - box_x1, box_y2 - box_y1,
                    edgecolor = box_color, facecolor = "none", lw = box_width)
ax.add_patch(rect)

format_datetime_xlabels(ax,
                        date_format = date_format,
                        major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing,
                        vertical_align = "top", horizontal_align = "right", rotation = 5.0,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                        major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

format_freq_ylabels(ax, label = False,
                    major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, 
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                    major_tick_length = major_tick_length, minor_tick_length = minor_tick_length, tick_width = tick_width)

for spine in ax.spines.values():
    spine.set_linewidth(spine_width_spec)
    spine.set_color(spine_color)

# Add the colorbar
bbox = ax.get_position()
position = [bbox.x1 + cbar_offset, bbox.y0, cbar_width, bbox.height]
cbar = add_power_colorbar(fig, power_color, position, 
                          orientation = "vertical",
                          tick_spacing = 10.0,
                          axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                          tick_length = major_tick_length, tick_width = tick_width)

for spine in cbar.ax.spines.values():
    spine.set_linewidth(spine_width_spec)

# Save the figure
figname = f"mit_whoi_poster_hourly_specs.png"
save_figure(fig, figname)

# figname = f"mit_whoi_poster_whole_day_specs.pdf"
# save_figure(fig, figname)
