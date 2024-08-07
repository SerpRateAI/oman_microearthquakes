# Plot the total power spectrograms of the inner-circle stations of Array A or B
# Imports
from os.path import join
from utils_basic import SPECTROGRAM_DIR as indir, INNER_STATIONS_A as stations_a, INNER_STATIONS_B as stations_b, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_spec import StreamSTFTPSD
from utils_spec import get_spectrogram_file_suffix, read_geo_spectrograms
from utils_plot import POWER_LABEL as colorbar_label
from utils_plot import add_colorbar, add_horizontal_scalebar, colormaps, format_datetime_xlabels, format_freq_ylabels, save_figure, subplots

# Inputs
array = "B"  # "A" or "B"

# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Plotting
if array == "A":
    stations_to_plot = stations_a
else:
    stations_to_plot = stations_b

min_freq = 0.0
max_freq = 5.0

min_db = -15.0
max_db = 15.0

width = 10.0
row_height= 3.0

colorbar_gap = 0.02
colorbar_width = 0.01

label_x = 0.015
label_y = 0.95

# Read the spectrograms and compute the total power
print("Reading the spectrograms...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)

stream_spec_total_power = StreamSTFTPSD()
for station in stations_to_plot:
    filename = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
    inpath = join(indir, filename)

    stream_spec = read_geo_spectrograms(inpath,
                                        starttime = starttime, endtime = endtime,
                                        min_freq = min_freq, max_freq = max_freq)
    trace_total_power = stream_spec.get_total_power()
    trace_total_power.to_db()

    stream_spec_total_power.append(trace_total_power)

# Plot the total power spectrograms
num_sta = len(stations_to_plot)
fig, axes = subplots(num_sta, 1, figsize = (width, row_height * num_sta), sharex = True, sharey = True)

for i, station in enumerate(stations_to_plot):
    trace = stream_spec_total_power.select(station = station)[0]
    ax = axes[i]
    quadmesh = trace.plot(ax, min_db, max_db)

    if i == num_sta - 1:
        ax.set_xlabel("Time (UTC)")

    if i == num_sta // 2:
        ax.set_ylabel("Frequency (Hz)")

    if i < num_sta - 1:
        format_datetime_xlabels(ax, label = False, major_tick_spacing = "24h", num_minor_ticks = 4)
    else:
        format_datetime_xlabels(ax, 
                                date_format = "%Y-%m-%d", major_tick_spacing = "24h", num_minor_ticks = 4, 
                                rotation = 45, va = "top", ha = "right")
        
    format_freq_ylabels(ax, major_tick_spacing = 0.5, num_minor_ticks = 5)

    # Plot the station label
    ax.text(label_x, label_y, station, va = "top", ha = "left", transform = ax.transAxes, fontsize = 12, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))

# Add a colorbar
ax = axes[-1]
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
cbar = add_colorbar(fig, quadmesh, "Power (dB)", position, orientation = "vertical")

# Save the figure
figname = f"inner_{array.lower()}_geo_total_power_spectrograms_{suffix_spec}_freq{min_freq:.1f}to{max_freq:.1f}hz.png"
save_figure(fig, figname)
