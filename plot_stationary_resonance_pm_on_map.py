# Plot the horizontal particle motions of stationary resonances on the geophone station map

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
from utils_plot import format_datetime_xlabels, format_freq_ylabels, format_east_xlabels, format_north_ylabels, get_label_alignments, get_stft_param_labels, get_power_colormap, save_figure

# Inputs
# Data
station_spec = "A01"
starttime_spec = "2020-01-13T19:30:00"
endtime_spec = "2020-01-13T20:30:00"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

starttime_wf = "2020-01-13T20:00:00"
dur_wf = 60.0

min_freq_filt = 37.0
max_freq_filt = 39.0

decimate = True
decimate_factor = 10

# Plotting
# Waveform
min_freq_plot = 30.0
max_freq_plot = 50.0

scale = 6.5

dbmin = -10.0   
dbmax = 10.0

axis_label_size = 12.0
tick_label_size = 10.0

major_time_spacing = "15min"
minor_time_spacing = "5min"

major_dist_spacing = 20.0
minor_dist_spacing = 5.0

rotation = 5
va = "top"
ha = "center"

major_freq_spacing = 5.0
minor_freq_spacing = 1.0

param_label_x = 0.99
param_label_y = 0.03
param_label_size = 10.0

station_label_x = 0.015
station_label_y = 0.94
spec_station_label_size = 12.0

linewidth_box = 1.0

# Map
station_label_offset_x = -2.0
station_label_offset_y = 2.0
map_station_label_size = 12.0

linewidth_pm = 0.5

marker_size = 7.5


# Read the waveforms
print("Reading and processing the waveforms...")
stream_wf = read_and_process_windowed_geo_waveforms(starttime_wf, dur = dur_wf, 
                                                    min_freq = min_freq_filt, max_freq = max_freq_filt,
                                                    decimate = decimate, decimate_factor = decimate_factor)


# Read the spectrograms
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
print("Reading the spectrograms...")
inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station_spec}_{suffix_spec}.h5")
stream_spec = read_geo_spectrograms(inpath, starttime = starttime_spec, endtime = endtime_spec, min_freq = min_freq_plot, max_freq = max_freq_plot)
trace_spec_total = stream_spec.get_total_power()
trace_spec_total.to_db()

# Get the station horizontal particle motions
station_pms = get_horizontal_particle_motions(stream_wf)

# Get the station coordinates
coord_df = get_geophone_coords()

# Plotting
print("Plotting...")
fig = figure(figsize = (8, 12))
gs = GridSpec(nrows = 2, ncols = 1, height_ratios = [1, 4], figure = fig)

# Plot the spectrogram
print("Plotting the spectrogram...")
spec_ax = fig.add_subplot(gs[0])
cmap = get_power_colormap()
timeax = trace_spec_total.times
freqax = trace_spec_total.freqs
freq_int = freqax[1] - freqax[0]
data = trace_spec_total.data
spec_ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin, vmax = dbmax)

box = Rectangle((Timestamp(starttime_wf), min_freq_filt), Timedelta(seconds = dur_wf), max_freq_filt - min_freq_filt, edgecolor = "lime", facecolor = "none", linewidth = linewidth_box)
spec_ax.add_patch(box)

# Plot the station label
spec_ax.text(station_label_x, station_label_y, station_spec, transform = spec_ax.transAxes, fontsize = spec_station_label_size, fontweight = "bold", va = "top", ha = "left", color = "black", bbox = dict(facecolor = "white", edgecolor = "black"))

# Plot the STFT parameter label
label = get_stft_param_labels(window_length, freq_int)
spec_ax.text(param_label_x, param_label_y, label, transform = spec_ax.transAxes, fontsize = param_label_size, va = "bottom", ha = "right", color = "white")



spec_ax.set_xlim(timeax[0], timeax[-1])
spec_ax.set_ylim(min_freq_plot, max_freq_plot)

format_datetime_xlabels(spec_ax, 
                        major_tick_spacing = major_time_spacing, 
                        minor_tick_spacing = minor_time_spacing, 
                        rotation = rotation, 
                        va = va, 
                        ha = ha,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

format_freq_ylabels(spec_ax,
                        major_tick_spacing = major_freq_spacing,
                        minor_tick_spacing = minor_freq_spacing,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Plot the station horizontal particle motions
print("Plotting the station horizontal particle motions...")
# bbox = spec_ax.get_position()
# map_width = bbox.width
# map_height = map_width * (max_north - min_north) / (max_east - min_east)
# map_x0 = bbox.x0
# map_y0 = bbox.y0 - axis_offset - map_height
# position = [map_x0, map_y0, map_width, map_height]
spec_map = fig.add_subplot(gs[1])

for station in station_pms.keys():
    partical_motion = station_pms[station]
    east_motion = partical_motion[:, 0]
    north_motion = partical_motion[:, 1]

    east = coord_df.loc[station, "east"]
    north = coord_df.loc[station, "north"]

    east_motion_plot = east + east_motion * scale
    north_motion_plot = north + north_motion * scale

    spec_map.plot(east_motion_plot, north_motion_plot, color = "lightgray", linewidth = linewidth_pm, zorder = 0)
    spec_map.scatter(east, north, marker = "^", color = "crimson", s = marker_size, zorder = 1)

    # Plot the station label
    if station == station_spec:
        ha, va = get_label_alignments(station_label_offset_x, station_label_offset_y)
        spec_map.annotate(station, (east, north), xytext = (station_label_offset_x, station_label_offset_y), textcoords = "offset points", fontsize = map_station_label_size, ha = ha, va = va, color = "crimson")

# Set the axis limits
spec_map.set_xlim(min_east, max_east)
spec_map.set_ylim(min_north, max_north)
spec_map.set_aspect("equal")

# Format the axis labels
format_east_xlabels(spec_map, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
format_north_ylabels(spec_map, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Save the figure
print("Saving the figure...")
suffix_time = time2suffix(starttime_wf)
figname = f"stationary_resonance_pm_on_map_{min_freq_filt:.0f}to{max_freq_filt:.0f}hz_{suffix_time}_{dur_wf:.0f}s.png"
save_figure(fig, figname)

