# Plot the horizontal particle motions of two stationary resonances on a map.

# Imports 
from os.path import join
from pandas import Timedelta, Timestamp
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplots
from matplotlib.patches import Rectangle

from utils_basic import GEO_STATIONS, SPECTROGRAM_DIR as indir, EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, time2suffix, get_datetime_axis_from_trace
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_spec import get_spectrogram_file_suffix, read_geo_spectrograms
from utils_pol import get_horizontal_particle_motions
from utils_plot import add_power_colorbar, add_vertical_scalebar, format_datetime_xlabels, format_freq_ylabels, format_east_xlabels, format_north_ylabels, get_label_alignments, get_stft_param_labels, get_power_colormap, save_figure

# Inputs
# Data
station_spec = "A01"

decimate = True
decimate_factor = 10

# Signal 1
name1 = "SR38a"

starttime1_spec = "2020-01-13T19:30:00"
endtime1_spec = "2020-01-13T20:30:00"

window_length1 = 60.0
overlap1 = 0.0
downsample1 = False
downsample_factor1= 60

starttime_wf1 = "2020-01-13T20:00:00"
dur_wf1 = 60.0

min_freq1_filt = 38.0
max_freq1_filt = 38.4

# Signal 2
name2 = "SR25a"

starttime2_spec = "2020-01-13T19:30:00"
endtime2_spec = "2020-01-13T20:30:00"

window_length2 = 60.0
overlap2 = 0.0
downsample2 = False
downsample_factor2 = 60

starttime_wf2 = "2020-01-13T20:00:00"
dur_wf2 = 60.0

min_freq2_filt = 25.4
max_freq2_filt = 25.6

# Plotting
fig_width = 16.0
column_height = 12.0

# Spectrograms
min_freq1_plot = 37.0
max_freq1_plot = 40.0

min_freq2_plot = 24.0
max_freq2_plot = 27.0

dbmin = -10.0   
dbmax = 10.0

axis_label_size = 12.0
tick_label_size = 10.0
title_size = 12.0

rotation = 5
va = "top"
ha = "center"

cbar_offset_x = 0.02
cbar_offset_y = 0.0
cbar_width = 0.01

major_time_spacing = "15min"
num_minor_time_ticks = 3

major_dist_spacing = 20.0
minor_dist_spacing = 5.0

major_freq_spacing = 1.0
num_minor_freq_ticks = 5

param_label_x = 0.99
param_label_y = 0.03
param_label_size = 10.0

station_label_x = 0.015
station_label_y = 0.94
spec_station_label_size = 12.0

linewidth_box = 1.5

# Map
station_label_offset_x = -2.0
station_label_offset_y = 2.0
map_station_label_size = 12.0

linewidth_pm = 0.5

marker_size = 7.5

scale1 = 6.5
scale2 = 3.5

scale_bar_x = 0.1
scale_bar_y = 0.1
label_offsets = (0.05, 0.0)

amplitude1 = 1.0
amplitude2 = 2.0

linewidth_scale = 1.0

# Read the waveforms
print("Reading and processing the waveforms...")
stream_wf1 = read_and_process_windowed_geo_waveforms(starttime_wf1, dur = dur_wf1, 
                                                            min_freq = min_freq1_filt, max_freq = max_freq1_filt,
                                                            decimate = decimate, decimate_factor = decimate_factor)

stream_wf2 = read_and_process_windowed_geo_waveforms(starttime_wf1, dur = dur_wf2, 
                                                            min_freq = min_freq2_filt, max_freq = max_freq2_filt,
                                                            decimate = decimate, decimate_factor = decimate_factor)


# Read the spectrograms
suffix_spec = get_spectrogram_file_suffix(window_length1, overlap1, downsample1, downsample_factor = downsample_factor1)
print("Reading the spectrograms of signal 1...")
inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station_spec}_{suffix_spec}.h5")
stream1_spec = read_geo_spectrograms(inpath, starttime = starttime1_spec, endtime = endtime1_spec, min_freq = min_freq1_plot, max_freq = max_freq1_plot)
trace1_spec = stream1_spec.get_total_power()
trace1_spec.to_db()

suffix_spec = get_spectrogram_file_suffix(window_length2, overlap2, downsample2, downsample_factor = downsample_factor2)
print("Reading the spectrograms of signal 2...")
inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station_spec}_{suffix_spec}.h5")
stream2_spec = read_geo_spectrograms(inpath, starttime = starttime2_spec, endtime = endtime2_spec, min_freq = min_freq2_plot, max_freq = max_freq2_plot)
trace2_spec = stream2_spec.get_total_power()
trace2_spec.to_db()

# Get the station horizontal particle motions
station_pms1 = get_horizontal_particle_motions(stream_wf1)
station_pms2 = get_horizontal_particle_motions(stream_wf2)

# Get the station coordinates
coord_df = get_geophone_coords()

# Plotting
print("Plotting...")
fig = figure(figsize = (fig_width, column_height))
gs = GridSpec(nrows = 2, ncols = 2, height_ratios = [1, 4], figure = fig)

# Plot the spectrogram
# Plot Signal 1
print("Plotting the spectrogram of signal 1...")
spec1_ax = fig.add_subplot(gs[0, 0])
cmap = get_power_colormap()
timeax = trace1_spec.times
freqax = trace1_spec.freqs
freq_int = freqax[1] - freqax[0]
data = trace1_spec.data
color = spec1_ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin, vmax = dbmax)

box = Rectangle((Timestamp(starttime_wf1), min_freq1_filt), Timedelta(seconds = dur_wf1), max_freq1_filt - min_freq1_filt, edgecolor = "lime", facecolor = "none", linewidth = linewidth_box)
spec1_ax.add_patch(box)

# Plot the station label
spec1_ax.text(station_label_x, station_label_y, station_spec, transform = spec1_ax.transAxes, fontsize = spec_station_label_size, fontweight = "bold", va = "top", ha = "left", color = "black", bbox = dict(facecolor = "white", edgecolor = "black"))

# Plot the STFT parameter label
label = get_stft_param_labels(window_length1, freq_int)
spec1_ax.text(param_label_x, param_label_y, label, transform = spec1_ax.transAxes, fontsize = param_label_size, va = "bottom", ha = "right", color = "white")

spec1_ax.set_xlim(timeax[0], timeax[-1])
spec1_ax.set_ylim(min_freq1_plot, max_freq1_plot)

format_datetime_xlabels(spec1_ax, 
                        major_tick_spacing = major_time_spacing,
                        num_minor_ticks = num_minor_time_ticks,
                        rotation = rotation, 
                        va = "top", 
                        ha = "right",
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

format_freq_ylabels(spec1_ax,
                    major_tick_spacing = major_freq_spacing,
                    num_minor_ticks = num_minor_freq_ticks,
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size)

spec1_ax.set_title(f"{name1}", fontsize = title_size, fontweight = "bold")

# Plot Signal 2
print("Plotting the spectrogram of signal 2...")
spec2_ax = fig.add_subplot(gs[0, 1])
cmap = get_power_colormap()
timeax = trace2_spec.times
freqax = trace2_spec.freqs
freq_int = freqax[1] - freqax[0]
data = trace2_spec.data
spec2_ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin, vmax = dbmax)

box = Rectangle((Timestamp(starttime_wf2), min_freq2_filt), Timedelta(seconds = dur_wf2), max_freq2_filt - min_freq2_filt, edgecolor = "lime", facecolor = "none", linewidth = linewidth_box)
spec2_ax.add_patch(box)

# Plot the station label
spec2_ax.text(station_label_x, station_label_y, station_spec, transform = spec2_ax.transAxes, fontsize = spec_station_label_size, fontweight = "bold", va = "top", ha = "left", color = "black", bbox = dict(facecolor = "white", edgecolor = "black"))

# Plot the STFT parameter label
label = get_stft_param_labels(window_length2, freq_int)
spec2_ax.text(param_label_x, param_label_y, label, transform = spec2_ax.transAxes, fontsize = param_label_size, va = "bottom", ha = "right", color = "white")

spec2_ax.set_xlim(timeax[0], timeax[-1])
spec2_ax.set_ylim(min_freq2_plot, max_freq2_plot)

format_datetime_xlabels(spec2_ax,
                        major_tick_spacing = major_time_spacing,
                        num_minor_ticks = num_minor_time_ticks,
                        rotation = rotation, 
                        va = "top",
                        ha = "right",
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

format_freq_ylabels(spec2_ax,
                    label = False,
                    major_tick_spacing = major_freq_spacing,
                    num_minor_ticks = num_minor_freq_ticks,
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size)

spec2_ax.set_title(f"{name2}", fontsize = title_size, fontweight = "bold")

# Add the colorbar                  
bbox = spec2_ax.get_position()
position = [bbox.x1 + cbar_offset_x, bbox.y0 + cbar_offset_y, cbar_width, bbox.height]
add_power_colorbar(fig, color, position, orientation = "vertical")

# Plot the station horizontal particle motions
# Signal 1
print("Plotting the station horizontal particle motions of signal 1...")
map1_ax = fig.add_subplot(gs[1, 0])

for station in station_pms1.keys():
    partical_motion = station_pms1[station]
    east_motion = partical_motion[:, 0]
    north_motion = partical_motion[:, 1]

    east = coord_df.loc[station, "east"]
    north = coord_df.loc[station, "north"]

    east_motion_plot = east + east_motion * scale1
    north_motion_plot = north + north_motion * scale1

    map1_ax.plot(east_motion_plot, north_motion_plot, color = "lightgray", linewidth = linewidth_pm, zorder = 0)
    map1_ax.scatter(east, north, marker = "^", color = "crimson", s = marker_size, zorder = 1)

    # Plot the station label
    if station == station_spec:
        ha, va = get_label_alignments(station_label_offset_x, station_label_offset_y)
        map1_ax.annotate(station, (east, north), xytext = (station_label_offset_x, station_label_offset_y), textcoords = "offset points", fontsize = map_station_label_size, ha = ha, va = va, color = "crimson")

# Add the scale bar
add_vertical_scalebar(map1_ax, (scale_bar_x, scale_bar_y), amplitude1, scale1, label_offsets, linewidth = linewidth_scale)

# Set the axis limits
map1_ax.set_xlim(min_east, max_east)
map1_ax.set_ylim(min_north, max_north)
map1_ax.set_aspect("equal")

# Format the axis labels
format_east_xlabels(map1_ax, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
format_north_ylabels(map1_ax, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Signal 2
print("Plotting the station horizontal particle motions of signal 2...")
map2_ax = fig.add_subplot(gs[1, 1])

for station in station_pms2.keys():
    partical_motion = station_pms2[station]
    east_motion = partical_motion[:, 0]
    north_motion = partical_motion[:, 1]

    east = coord_df.loc[station, "east"]
    north = coord_df.loc[station, "north"]

    east_motion_plot = east + east_motion * scale2
    north_motion_plot = north + north_motion * scale2

    map2_ax.plot(east_motion_plot, north_motion_plot, color = "lightgray", linewidth = linewidth_pm, zorder = 0)
    map2_ax.scatter(east, north, marker = "^", color = "crimson", s = marker_size, zorder = 1)

    # Plot the station label
    if station == station_spec:
        ha, va = get_label_alignments(station_label_offset_x, station_label_offset_y)
        map2_ax.annotate(station, (east, north), xytext = (station_label_offset_x, station_label_offset_y), textcoords = "offset points", fontsize = map_station_label_size, ha = ha, va = va, color = "crimson")

# Add the scale bar
add_vertical_scalebar(map2_ax, (scale_bar_x, scale_bar_y), amplitude2, scale2, label_offsets, linewidth = linewidth_scale)

# Set the axis limits
map2_ax.set_xlim(min_east, max_east)
map2_ax.set_ylim(min_north, max_north)
map2_ax.set_aspect("equal")

# Format the axis labels
format_east_xlabels(map2_ax, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
format_north_ylabels(map2_ax, label = False, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Save the figure
print("Saving the figure...")
suffix_time1 = time2suffix(starttime_wf1)
suffix_time2 = time2suffix(starttime_wf2)
figname = f"stationary_resonances_pm_on_map_{min_freq1_filt:.0f}to{max_freq1_filt:.0f}hz_{suffix_time1}_{dur_wf1:.0f}s_{min_freq2_filt:.0f}to{max_freq2_filt:.0f}hz_{suffix_time2}_{dur_wf2:.0f}s.png"
save_figure(fig, figname)

