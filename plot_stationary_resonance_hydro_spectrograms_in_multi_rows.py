# Plot the zoom-in view of the spectrograms of a stationary resoncance recorded on one hydrophone location and the spectral peaks in multiple rows

# Plot the comparison of the hydrophone spectrograms of two stationary resonances 

# Imports
from os.path import join
from argparse import ArgumentParser
from pandas import date_range, read_csv
from matplotlib.pyplot import subplots, colormaps

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro
from utils_spec import get_spectrogram_file_suffix, read_hydro_stft
from utils_plot import POWER_LABEL as colorbar_label
from utils_plot import add_colorbar, add_horizontal_scalebar, format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
# Parse the arguments
parser = ArgumentParser(description = "Plot the zoom-in view of the spectrograms of a stationary resoncance recorded on one hydrophone location and the spectral peaks in multiple rows")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--station", type = str, help = "Station name")
parser.add_argument("--location", type = str, help = "Location name")
parser.add_argument("--min_db", type = float, help = "Minimum dB")
parser.add_argument("--max_db", type = float, help = "Maximum dB")

parser.add_argument("--window_length", type = float, help = "Window length in seconds")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds")

starttime = starttime_hydro
endtime = endtime_hydro

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name
station_to_plot = args.station
location_to_plot = args.location
min_db = args.min_db
max_db = args.max_db

window_length = args.window_length
overlap = args.overlap


# Print the parameters
print(f"### Plotting the zoom-in view of the spectrograms of a stationary resoncance recorded on {station_to_plot}.{location_to_plot} ###")
print(f"Mode name: {mode_name}")
print("Station: ", station_to_plot)
print("Location: ", location_to_plot)
print(f"dB range: {min_db} - {max_db} dB")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print("")

# Constants
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

marker_size = 0.2

# Read the plotting frequency range
filename = f"stationary_resonance_freq_ranges_hydro.csv"
inpath = join(indir, filename)
freq_range_df = read_csv(inpath)

min_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_plot"].values[0]
max_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_plot"].values[0]

print(f"Plotting frequency range: {min_freq} - {max_freq} Hz")

# Read the spectrograms
print("Reading the spectrograms...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

filename = f"whole_deployment_daily_hydro_stft_{station_to_plot}_{suffix_spec}.h5"
inpath = join(indir, filename)

stream_stft = read_hydro_stft(inpath,
                            locations = location_to_plot,
                            starttime = starttime, endtime = endtime,
                            min_freq = min_freq, max_freq = max_freq,
                            psd_only = True)

# for trace_stft in stream_stft:
#     print(trace_stft.times[0], trace_stft.times[-1])

# Generate the figure
print("Generating the figure...")
fig, axes = subplots(num_rows, 1, figsize=(column_width, row_height * num_rows), sharey = True)

# Plotting
print("Plotting the spectrograms and the spectral peaks...")

# Plot each time window
windows = date_range(starttime, endtime, periods = num_rows + 1)
cmap = colormaps["inferno"].copy()
cmap.set_bad(color='darkgray')

for i in range(num_rows):
    starttime = windows[i]
    endtime = windows[i + 1]

    print("Plotting the time window: ", starttime, " - ", endtime)
    stream_stft_window = stream_stft.slice_time(starttime = starttime, endtime = endtime)
    
    print(f"Plotting the spectrogram for the window")
    ax = axes[i]
    trace_stft = stream_stft_window[0]
    trace_stft.to_db()
    psd_mat = trace_stft.psd_mat
    freqax = trace_stft.freqs
    timeax = trace_stft.times

    ax.set_xlim(starttime, endtime)

    mappable = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_db, vmax = max_db)
    if i == num_rows - 1:
        format_datetime_xlabels(ax,
                                label = True,
                                major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, date_format = "%Y-%m-%d")
    else:
        format_datetime_xlabels(ax,
                                label = False,
                                major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, date_format = "%Y-%m-%d")

    if i < num_rows - 1:       
        add_horizontal_scalebar(ax, scalebar_coord, "1d", 1.0, color = color_ref, plot_label = False)
    else:
        add_horizontal_scalebar(ax, scalebar_coord, "1d", 1.0, color = color_ref, plot_label = True,
                                label = "1d", label_offsets = scalebar_label_offsets, fontsize = 10, fontweight = "bold")

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)


fig.suptitle(f"Stationary resonance {mode_name} on {station_to_plot}.{location_to_plot}", y = 0.92, fontsize = 14, fontweight = "bold")

# Add colorbar
ax = axes[-1]
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, position, colorbar_label, 
             mappable = mappable,
             orientation = "vertical")

# Save the figure
figname = f"stationary_resonance_hydro_spectrogram_multi_rows_{mode_name}_{station_to_plot}.{location_to_plot}_{suffix_spec}.png"
save_figure(fig, figname)

