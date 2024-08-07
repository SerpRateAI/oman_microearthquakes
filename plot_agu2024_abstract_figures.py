# Plot the two figures in the AGU 2024 abstract

# Imports
from os.path import join
from pandas import date_range
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro
from utils_spec import get_spectrogram_file_suffix, read_geo_spectrograms, read_hydro_spectrograms
from utils_plot import POWER_LABEL as colorbar_label
from utils_plot import add_colorbar, add_horizontal_scalebar, colormaps, format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
# Data
hydr_station_to_plot = "A00"
location_to_plot= "03"

geo_station_to_plot = "A01"
name = "PR2548"

window_length_long = 60.0
overlap_long = 0.0
downsample_long = False
downsample_factor_long = 60

window_length_short = 1.0
overlap_short = 0.0
downsample_short = False
downsample_factor_short = 60

starttime_long = starttime_hydro
endtime_long = endtime_hydro

starttime_short = "2020-01-13 19:35:30"
endtime_short = "2020-01-13 19:37:00"

min_freq_long = 25.00
max_freq_long = 26.00

min_freq_short = 30.0
max_freq_short = 175.0

pred_freq = 12.75

min_db_long = -80.0
max_db_long = -60.0

min_db_short = -15.0
max_db_short = 15.0

# Plotting
num_rows = 4
column_width = 10.0
row_height = 2.0

colorbar_width = 0.01
colorbar_gap = 0.02

major_time_spacing_long = "15d"
num_minor_time_ticks_long = 3

major_time_spacing_short = "15s"
num_minor_time_ticks_short = 3

major_freq_spacing_long = 0.5
num_minor_freq_ticks_long = 5

major_freq_spacing_short = 25.0
num_minor_freq_ticks_short = 5

color_ref = "aqua"
# linewidth_freq = 1.5
linewidth_time = 3.0

scalebar_coord_long = (0.03, 0.9)
scalebar_label_offsets_long = (0.02, 0.0)

scalebar_coord_short = (0.05, 0.9)
scalebar_label_offsets_short = (0.04, 0.0)

# Read the long-window spectrograms
print("Reading the long-window spectrograms...")
suffix = get_spectrogram_file_suffix(window_length_long, overlap_long, downsample_long)
filename = f"whole_deployment_daily_hydro_spectrograms_{hydr_station_to_plot}_{suffix}.h5"
inpath = join(indir, filename)

stream_spec_long = read_hydro_spectrograms(inpath,
                                      locations = location_to_plot,
                                      starttime = starttime_long, endtime = endtime_long,
                                      min_freq = min_freq_long, max_freq = max_freq_long)

# Read the short-window spectrograms
print("Reading the short-window spectrograms...")
suffix = get_spectrogram_file_suffix(window_length_short, overlap_short, downsample_short)
filename = f"whole_deployment_daily_geo_spectrograms_{geo_station_to_plot}_{suffix}.h5"
inpath = join(indir, filename)

stream_spec_short = read_geo_spectrograms(inpath,
                                        starttime = starttime_short, endtime = endtime_short,
                                        min_freq = min_freq_short, max_freq = max_freq_short)

trace_spec_short = stream_spec_short.get_total_power()

# Plotting
# Plot the long-window spectrograms
print("Plotting the long-window spectrograms...")
fig, axes = subplots(num_rows, 1, figsize=(column_width, row_height * num_rows), sharey = True)

# Plot each time window
windows = date_range(starttime_long, endtime_long, periods = num_rows + 1)

for i in range(num_rows):
    starttime = windows[i]
    endtime = windows[i + 1]
    stream_spec_window = stream_spec_long.slice_time(starttime = starttime, endtime = endtime)
    
    ax = axes[i]
    trace_spec = stream_spec_window[0]
    trace_spec.to_db()
    data = trace_spec.data
    freqax = trace_spec.freqs
    timeax = trace_spec.times

    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color='darkgray')

    quadmesh = ax.pcolormesh(timeax, freqax, data, shading = "auto", cmap = cmap, vmin = min_db_long, vmax = max_db_long)

    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing_long, num_minor_ticks = num_minor_time_ticks_long, date_format = "%Y-%m-%d")

    if i < num_rows - 1:       
        add_horizontal_scalebar(ax, scalebar_coord_long, "1d", 1.0, color = color_ref, plot_label = False)
    else:
        add_horizontal_scalebar(ax, scalebar_coord_long, "1d", 1.0, color = color_ref, plot_label = True,
                                label = "1 d", label_offsets = scalebar_label_offsets_long, fontsize = 10, fontweight = "bold")

format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing_long, num_minor_ticks = num_minor_freq_ticks_long)

fig.suptitle(f"Persistent tremor {name} on Hydrophone {hydr_station_to_plot}.{location_to_plot}", y = 0.92, fontsize = 14, fontweight = "bold")

# Add colorbar
ax = axes[-1]
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, quadmesh, colorbar_label, position, orientation = "vertical")

# Save the figure
figname = f"agu2024_abatract_fig1.png"
save_figure(fig, figname)

# Plot the short-window spectrograms
print("Plotting the short-window spectrograms...")
fig, ax = subplots(1, 1, figsize=(column_width, row_height))

trace_spec_short.to_db()
data = trace_spec_short.data
freqax = trace_spec_short.freqs
timeax = trace_spec_short.times

cmap = colormaps["inferno"].copy()
cmap.set_bad(color='darkgray')

quadmesh = ax.pcolormesh(timeax, freqax, data, shading = "auto", cmap = cmap, vmin = min_db_short, vmax = max_db_short)

add_horizontal_scalebar(ax, scalebar_coord_short, "5s", 1.0, color = color_ref, plot_label = True,
                                label = "5 s", label_offsets = scalebar_label_offsets_short, fontsize = 10, fontweight = "bold")

format_datetime_xlabels(ax, 
                        major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d %H:%M:%S",
                        va = "top", ha = "right", rotation = 15)

format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing_short, num_minor_ticks = num_minor_freq_ticks_short)

ax.set_title(f"Transient tremors on Geophone {geo_station_to_plot}", fontsize = 14, fontweight = "bold")

# Add colorbar
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, quadmesh, colorbar_label, position, orientation = "vertical")

# Save the figure
figname = f"agu2024_abatract_fig2.png"
save_figure(fig, figname)