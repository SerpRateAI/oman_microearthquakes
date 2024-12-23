# Plot the total power spectrograms of a few example geophone stations
# Imports
from os.path import join
from numpy import unravel_index

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo
from utils_spec import StreamSTFTPSD
from utils_spec import get_spectrogram_file_suffix, read_geo_spectrograms
from utils_plot import POWER_LABEL as colorbar_label
from utils_plot import add_colorbar, add_horizontal_scalebar, colormaps, format_datetime_xlabels, format_freq_ylabels, save_figure, subplots

# Spectrogram
window_length = 300.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Plotting
num_cols = 2
num_rows = 3

stations_to_plot = ["A16", "A01", "A19", "B01", "B19", "B20"]

min_freq = 0.0
max_freq = 10.0

min_db = -15.0
max_db = 15.0

col_width = 10.0
row_height= 3.0

colorbar_gap = 0.01
colorbar_width = 0.005

label_x = 0.015
label_y = 0.95

major_frequency_tick_spacing = 2.0
minor_frequency_ticks = 5

# Read the spectrograms and compute the total power
print("Reading the spectrograms...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)

stream_spec_total_power = StreamSTFTPSD()
for station in stations_to_plot:
    filename = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
    inpath = join(indir, filename)

    stream_spec = read_geo_spectrograms(inpath,
                                        starttime = starttime_geo, endtime = endtime_geo,
                                        min_freq = min_freq, max_freq = max_freq)
    trace_total_power = stream_spec.get_total_power()
    trace_total_power.to_db()

    stream_spec_total_power.append(trace_total_power)

# Plot the total power spectrograms
num_sta = len(stations_to_plot)
if num_sta != num_cols * num_rows:
    raise ValueError(f"num_sta = {num_sta} is not equal to num_cols * num_rows = {num_cols * num_rows}.")

fig, axes = subplots(num_rows, num_cols, figsize = (col_width * num_cols, row_height * num_rows), sharex = True, sharey = True)

for i, station in enumerate(stations_to_plot):
    trace = stream_spec_total_power.select(stations = station)[0]
    i_row, i_col = unravel_index(i, (num_rows, num_cols))

    ax = axes[i_row, i_col]
    quadmesh = trace.plot(ax, min_db, max_db)

    if i_row == num_rows - 1:
        ax.set_xlabel("Time (UTC)")
        format_datetime_xlabels(ax, date_format = "%Y-%m-%d", major_tick_spacing = "24h", num_minor_ticks = 4, rotation = 45, va = "top", ha = "right")
    else:
        format_datetime_xlabels(ax, label = False, date_format = "%Y-%m-%d", major_tick_spacing = "24h", num_minor_ticks = 4, rotation = 45, va = "top", ha = "right")

    if i_col == 0:
        ax.set_ylabel("Frequency (Hz)")
        format_freq_ylabels(ax, major_tick_spacing = major_frequency_tick_spacing, num_minor_ticks = minor_frequency_ticks)
    else:
        format_freq_ylabels(ax, label = False, major_tick_spacing = major_frequency_tick_spacing, num_minor_ticks = minor_frequency_ticks)
        
    # Plot the station label
    ax.text(label_x, label_y, station, va = "top", ha = "left", transform = ax.transAxes, fontsize = 12, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))

# Add a colorbar
ax = axes[-1, -1]
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
cbar = add_colorbar(fig, quadmesh, "Power (dB)", position, orientation = "vertical")

# Save the figure
figname = f"example_geo_total_power_spectrograms_{suffix_spec}_freq{min_freq:.1f}to{max_freq:.1f}hz.png"
save_figure(fig, figname)
