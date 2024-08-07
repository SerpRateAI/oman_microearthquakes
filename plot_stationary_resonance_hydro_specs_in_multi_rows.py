# Plot the zoom-in view of the spectrograms of a stationary resoncance recorded on one hydrophone location in multiple rows

# Plot the comparison of the hydrophone spectrograms of two stationary resonances 

# Imports
from os.path import join
from pandas import date_range
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro
from utils_spec import get_spectrogram_file_suffix, read_hydro_spectrograms
from utils_plot import POWER_LABEL as colorbar_label
from utils_plot import add_colorbar, add_horizontal_scalebar, colormaps, format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
# Data
station_to_plot = "A00"
location_to_plot= "06"

name = "MH12"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

starttime = starttime_hydro
endtime = endtime_hydro

min_freq = 12.25
max_freq = 13.25

pred_freq = 12.75

min_db = -80.0
max_db = -60.0

# Plotting
num_rows = 4
column_width = 10.0
row_height = 2.0

colorbar_width = 0.01
colorbar_gap = 0.02

major_time_spacing = "15d"
num_minor_time_ticks = 3

major_freq_spacing = 0.5
num_minor_freq_ticks = 5

color_ref = "aqua"
linewidth_freq = 1.5
linewidth_time = 3.0

scalebar_coord = (0.03, 0.9)
scalebar_label_offsets = (0.02, 0.0)


# Read the spectrograms
print("Reading the spectrograms...")
suffix = get_spectrogram_file_suffix(window_length, overlap, downsample)
filename = f"whole_deployment_daily_hydro_spectrograms_{station_to_plot}_{suffix}.h5"
inpath = join(indir, filename)

stream_spec = read_hydro_spectrograms(inpath,
                                      locations = location_to_plot,
                                      starttime = starttime, endtime = endtime,
                                      min_freq = min_freq, max_freq = max_freq)

# Plotting
print("Plotting the spectrograms...")
fig, axes = subplots(num_rows, 1, figsize=(column_width, row_height * num_rows), sharey = True)

# Plot each time window
windows = date_range(starttime, endtime, periods = num_rows + 1)

for i in range(num_rows):
    starttime = windows[i]
    endtime = windows[i + 1]
    stream_spec_window = stream_spec.slice_time(starttime = starttime, endtime = endtime)
    
    ax = axes[i]
    trace_spec = stream_spec_window[0]
    trace_spec.to_db()
    data = trace_spec.data
    freqax = trace_spec.freqs
    timeax = trace_spec.times

    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color='darkgray')

    quadmesh = ax.pcolormesh(timeax, freqax, data, shading = "auto", cmap = cmap, vmin = min_db, vmax = max_db)
    ax.axhline(y = pred_freq, color = "aqua", linestyle = ":", linewidth = 1.0)

    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, date_format = "%Y-%m-%d")

    if i < num_rows - 1:       
        add_horizontal_scalebar(ax, scalebar_coord, "1d", 1.0, color = color_ref, plot_label = False)
    else:
        add_horizontal_scalebar(ax, scalebar_coord, "1d", 1.0, color = color_ref, plot_label = True,
                                label = "1d", label_offsets = scalebar_label_offsets, fontsize = 10, fontweight = "bold")

format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

fig.suptitle(f"Stationary resonance {name} on {station_to_plot}.{location_to_plot}", y = 0.92, fontsize = 14, fontweight = "bold")

# Add colorbar
ax = axes[-1]
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, quadmesh, colorbar_label, position, orientation = "vertical")

# Save the figure
figname = f"stationary_resonances_hydro_specs_{suffix}_{name}_{station_to_plot}.{location_to_plot}_multi_rows.png"
save_figure(fig, figname)
