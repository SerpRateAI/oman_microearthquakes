# Plot the horizontal particle motions of stationary and gliding resonances on the geophone station map

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
from utils_plot import add_power_colorbar, add_scalebar, format_datetime_xlabels, format_freq_ylabels, format_east_xlabels, format_north_ylabels, get_label_alignments, get_stft_param_labels, get_power_colormap, save_figure

# Inputs
# Data
station_spec = "A01"

# Stationary
starttime_stable_spec = "2020-01-13T19:30:00"
endtime_stable_spec = "2020-01-13T20:30:00"

window_length_stable = 60.0
overlap_stable = 0.0
downsample_stable = False
downsample_factor_stable = 60

starttime_stable_wf = "2020-01-13T20:00:00"
dur_stable_wf = 60.0

min_freq_stable_filt = 38.0
max_freq_stable_filt = 38.4

decimate_stable = True
decimate_factor_stable = 10

# Gliding
starttime_glide_spec = "2020-01-13T20:00:00"
endtime_glide_spec = "2020-01-13T20:02:30"

window_length_glide = 1.0
overlap_glide = 0.0
downsample_glide = False
downsample_factor_glide = 60

starttime_glide_wf = "2020-01-13T20:00:45"
dur_glide_wf = 25.0

min_freq_glide_filt = 40.0
max_freq_glide_filt = 75.0

decimate_glide = True
decimate_factor_glide = 5

# Plotting
fig_width = 16.0
column_height = 12.0

# Spectrograms
min_freq_stable_plot = 37.0
max_freq_stable_plot = 40.0

min_freq_glide_plot = 30.0
max_freq_glide_plot = 80.0

dbmin = -10.0   
dbmax = 10.0

axis_label_size = 12.0
tick_label_size = 10.0

rotation = 5
va = "top"
ha = "center"

cbar_offset_x = 0.02
cbar_offset_y = 0.0
cbar_width = 0.01

major_time_spacing_stable = "15min"
num_minor_time_ticks_stable = 3

major_time_spacing_glide = "1min"
num_minor_time_ticks_glide = 4

major_dist_spacing = 20.0
minor_dist_spacing = 5.0

major_freq_spacing_stable = 1.0
num_minor_freq_ticks_stable = 5

major_freq_spacing_glide = 10.0
num_minor_freq_ticks_glide = 5

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

scale_stable = 6.5
scale_glide = 0.2

scale_bar_x = -110.0
scale_bar_y = -90.0
label_offset = (5.0, 0.0)
amplitude_stable = 1.0
amplitude_glide = 50.0
linewidth_scale = 1.0

# Read the waveforms
print("Reading and processing the waveforms...")
stream_stable_wf = read_and_process_windowed_geo_waveforms(starttime_stable_wf, dur = dur_stable_wf, 
                                                            min_freq = min_freq_stable_filt, max_freq = max_freq_stable_filt,
                                                            decimate = decimate_stable, decimate_factor = decimate_factor_stable)

stream_glide_wf = read_and_process_windowed_geo_waveforms(starttime_glide_wf, dur = dur_glide_wf, 
                                                            min_freq = min_freq_glide_filt, max_freq = max_freq_glide_filt,
                                                            decimate = decimate_glide, decimate_factor = decimate_factor_glide)


# Read the spectrograms
suffix_spec = get_spectrogram_file_suffix(window_length_stable, overlap_stable, downsample_stable, downsample_factor = downsample_factor_stable)
print("Reading the stationary-resonance spectrograms...")
inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station_spec}_{suffix_spec}.h5")
stream_stable_spec = read_geo_spectrograms(inpath, starttime = starttime_stable_spec, endtime = endtime_stable_spec, min_freq = min_freq_stable_plot, max_freq = max_freq_stable_plot)
trace_stable_spec = stream_stable_spec.get_total_power()
trace_stable_spec.to_db()

suffix_spec = get_spectrogram_file_suffix(window_length_glide, overlap_stable, downsample_glide, downsample_factor = downsample_factor_glide)
print("Reading the stationary-resonance spectrograms...")
inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station_spec}_{suffix_spec}.h5")
stream_glide_spec = read_geo_spectrograms(inpath, starttime = starttime_glide_spec, endtime = endtime_glide_spec, min_freq = min_freq_glide_plot, max_freq = max_freq_glide_plot)
trace_glide_spec = stream_glide_spec.get_total_power()
trace_glide_spec.to_db()

# Get the station horizontal particle motions
station_pms_stable = get_horizontal_particle_motions(stream_stable_wf)
station_pms_glide = get_horizontal_particle_motions(stream_glide_wf)

# Get the station coordinates
coord_df = get_geophone_coords()

# Plotting
print("Plotting...")
fig = figure(figsize = (fig_width, column_height))
gs = GridSpec(nrows = 2, ncols = 2, height_ratios = [1, 4], figure = fig)

# Plot the spectrogram
# Plot the stationary-resonance spectrogram
print("Plotting the stationary-resonance spectrogram...")
stable_spec_ax = fig.add_subplot(gs[0, 0])
cmap = get_power_colormap()
timeax = trace_stable_spec.times
freqax = trace_stable_spec.freqs
freq_int = freqax[1] - freqax[0]
data = trace_stable_spec.data
color = stable_spec_ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin, vmax = dbmax)

box = Rectangle((Timestamp(starttime_stable_wf), min_freq_stable_filt), Timedelta(seconds = dur_stable_wf), max_freq_stable_filt - min_freq_stable_filt, edgecolor = "lime", facecolor = "none", linewidth = linewidth_box)
stable_spec_ax.add_patch(box)

# Plot the station label
stable_spec_ax.text(station_label_x, station_label_y, station_spec, transform = stable_spec_ax.transAxes, fontsize = spec_station_label_size, fontweight = "bold", va = "top", ha = "left", color = "black", bbox = dict(facecolor = "white", edgecolor = "black"))

# Plot the STFT parameter label
label = get_stft_param_labels(window_length_stable, freq_int)
stable_spec_ax.text(param_label_x, param_label_y, label, transform = stable_spec_ax.transAxes, fontsize = param_label_size, va = "bottom", ha = "right", color = "white")

stable_spec_ax.set_xlim(timeax[0], timeax[-1])
stable_spec_ax.set_ylim(min_freq_stable_plot, max_freq_stable_plot)

format_datetime_xlabels(stable_spec_ax, 
                        major_tick_spacing = major_time_spacing_stable,
                        num_minor_ticks = num_minor_time_ticks_stable,
                        rotation = rotation, 
                        va = "top", 
                        ha = "right",
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

format_freq_ylabels(stable_spec_ax,
                    major_tick_spacing = major_freq_spacing_stable,
                    num_minor_ticks = num_minor_freq_ticks_stable,
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Plot the gliding-resonance spectrogram
print("Plotting the gliding-resonance spectrogram...")
glide_spec_ax = fig.add_subplot(gs[0, 1])
cmap = get_power_colormap()
timeax = trace_glide_spec.times
freqax = trace_glide_spec.freqs
freq_int = freqax[1] - freqax[0]
data = trace_glide_spec.data
glide_spec_ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin, vmax = dbmax)

box = Rectangle((Timestamp(starttime_glide_wf), min_freq_glide_filt), Timedelta(seconds = dur_glide_wf), max_freq_glide_filt - min_freq_glide_filt, edgecolor = "lime", facecolor = "none", linewidth = linewidth_box)
glide_spec_ax.add_patch(box)

# Plot the station label
glide_spec_ax.text(station_label_x, station_label_y, station_spec, transform = glide_spec_ax.transAxes, fontsize = spec_station_label_size, fontweight = "bold", va = "top", ha = "left", color = "black", bbox = dict(facecolor = "white", edgecolor = "black"))

# Plot the STFT parameter label
label = get_stft_param_labels(window_length_glide, freq_int)
glide_spec_ax.text(param_label_x, param_label_y, label, transform = glide_spec_ax.transAxes, fontsize = param_label_size, va = "bottom", ha = "right", color = "white")

glide_spec_ax.set_xlim(timeax[0], timeax[-1])
glide_spec_ax.set_ylim(min_freq_glide_plot, max_freq_glide_plot)

format_datetime_xlabels(glide_spec_ax,
                        major_tick_spacing = major_time_spacing_glide,
                        num_minor_ticks = num_minor_time_ticks_glide,
                        rotation = rotation, 
                        va = "top",
                        ha = "right",
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

format_freq_ylabels(glide_spec_ax,
                    label = False,
                    major_tick_spacing = major_freq_spacing_glide,
                    num_minor_ticks = num_minor_freq_ticks_glide,
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Add the colorbar                  
bbox = glide_spec_ax.get_position()
position = [bbox.x1 + cbar_offset_x, bbox.y0 + cbar_offset_y, cbar_width, bbox.height]
add_power_colorbar(fig, color, position, orientation = "vertical")

# Plot the station horizontal particle motions
# Stationary resonance
print("Plotting the station horizontal particle motions of the stationary resonance...")
stable_map_ax = fig.add_subplot(gs[1, 0])

for station in station_pms_stable.keys():
    partical_motion = station_pms_stable[station]
    east_motion = partical_motion[:, 0]
    north_motion = partical_motion[:, 1]

    east = coord_df.loc[station, "east"]
    north = coord_df.loc[station, "north"]

    east_motion_plot = east + east_motion * scale_stable
    north_motion_plot = north + north_motion * scale_stable

    stable_map_ax.plot(east_motion_plot, north_motion_plot, color = "lightgray", linewidth = linewidth_pm, zorder = 0)
    stable_map_ax.scatter(east, north, marker = "^", color = "crimson", s = marker_size, zorder = 1)

    # Plot the station label
    if station == station_spec:
        ha, va = get_label_alignments(station_label_offset_x, station_label_offset_y)
        stable_map_ax.annotate(station, (east, north), xytext = (station_label_offset_x, station_label_offset_y), textcoords = "offset points", fontsize = map_station_label_size, ha = ha, va = va, color = "crimson")

# Add the scale bar
add_scalebar(stable_map_ax, (scale_bar_x, scale_bar_y), amplitude_stable, scale_stable, label_offset, linewidth = linewidth_scale)

# Set the axis limits
stable_map_ax.set_xlim(min_east, max_east)
stable_map_ax.set_ylim(min_north, max_north)
stable_map_ax.set_aspect("equal")

# Format the axis labels
format_east_xlabels(stable_map_ax, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
format_north_ylabels(stable_map_ax, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Gliding resonance
print("Plotting the station horizontal particle motions of the gliding resonance...")
glide_map_ax = fig.add_subplot(gs[1, 1])

for station in station_pms_glide.keys():
    partical_motion = station_pms_glide[station]
    east_motion = partical_motion[:, 0]
    north_motion = partical_motion[:, 1]

    east = coord_df.loc[station, "east"]
    north = coord_df.loc[station, "north"]

    east_motion_plot = east + east_motion * scale_glide
    north_motion_plot = north + north_motion * scale_glide

    glide_map_ax.plot(east_motion_plot, north_motion_plot, color = "lightgray", linewidth = linewidth_pm, zorder = 0)
    glide_map_ax.scatter(east, north, marker = "^", color = "crimson", s = marker_size, zorder = 1)

    # Plot the station label
    if station == station_spec:
        ha, va = get_label_alignments(station_label_offset_x, station_label_offset_y)
        glide_map_ax.annotate(station, (east, north), xytext = (station_label_offset_x, station_label_offset_y), textcoords = "offset points", fontsize = map_station_label_size, ha = ha, va = va, color = "crimson")

# Add the scale bar
add_scalebar(glide_map_ax, (scale_bar_x, scale_bar_y), amplitude_glide, scale_glide, label_offset, linewidth = linewidth_scale)

# Set the axis limits
glide_map_ax.set_xlim(min_east, max_east)
glide_map_ax.set_ylim(min_north, max_north)
glide_map_ax.set_aspect("equal")

# Format the axis labels
format_east_xlabels(glide_map_ax, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
format_north_ylabels(glide_map_ax, label = False, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Save the figure
print("Saving the figure...")
suffix_time_stable = time2suffix(starttime_stable_wf)
suffix_time_glide = time2suffix(starttime_glide_wf)
figname = f"stationary_and_gliding_resonance_pm_on_map_{min_freq_stable_filt:.0f}to{max_freq_stable_filt:.0f}hz_{suffix_time_stable}_{dur_stable_wf:.0f}s_{min_freq_glide_filt:.0f}to{max_freq_glide_filt:.0f}hz_{suffix_time_glide}_{dur_glide_wf:.0f}s.png"
save_figure(fig, figname)

