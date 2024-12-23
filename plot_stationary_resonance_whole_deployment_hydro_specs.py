# Plot the 9-month spectrograms of one stationary resonance recorded by the hydrophone at one station

# Imports
from os.path import join
from argparse import ArgumentParser
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro
from utils_spec import get_spectrogram_file_suffix, read_hydro_power_spectrograms
from utils_plot import POWER_LABEL as colorbar_label
from utils_plot import add_colorbar, plot_hydro_stft_spectrograms, save_figure

# Inputs
# Command line arguments
parser = ArgumentParser(description = "Plot the 9-month spectrograms of one stationary resonance recorded by the hydrophone at one station")
parser.add_argument("--station", type = str, help = "Station to plot")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--window_length", type = float, help = "Length of the window in seconds")
parser.add_argument("--overlap", type = float, help = "Overlap fraction")
parser.add_argument("--min_freq", type = float, help = "Minimum frequency in Hz")
parser.add_argument("--max_freq", type = float, help = "Maximum frequency in Hz")
parser.add_argument("--min_db", type = float, help = "Minimum power in dB")
parser.add_argument("--max_db", type = float, help = "Maximum power in dB")
parser.add_argument("--major_time_spacing", type = str, help = "Major time spacing")
parser.add_argument("--num_minor_time_ticks", type = int, help = "Number of minor time ticks")
parser.add_argument("--major_freq_spacing", type = float, help = "Major frequency spacing")
parser.add_argument("--num_minor_freq_ticks", type = int, help = "Number of minor frequency ticks")

# Parse the command line arguments
args = parser.parse_args()
station_to_plot = args.station
mode_name = args.mode_name
window_length = args.window_length
overlap = args.overlap
min_freq = args.min_freq
max_freq = args.max_freq
min_db = args.min_db
max_db = args.max_db
major_time_spacing = args.major_time_spacing
num_minor_time_ticks = args.num_minor_time_ticks
major_freq_spacing = args.major_freq_spacing
num_minor_freq_ticks = args.num_minor_freq_ticks

# Constants
starttime = starttime_hydro
endtime = endtime_hydro

column_width = 10.0
row_height = 2.0

colorbar_width = 0.01
colorbar_gap = 0.02

location_label_x = 0.01
location_label_y = 0.94

# Read the spectrograms
print("Reading the spectrograms...")
# Signal 1
suffix = get_spectrogram_file_suffix(window_length, overlap)
filename = f"whole_deployment_daily_hydro_power_spectrograms_{station_to_plot}_{suffix}.h5"
inpath = join(indir, filename)

hydro_spec1 = read_hydro_power_spectrograms(inpath,
                                            starttime = starttime, endtime = endtime,
                                            min_freq = min_freq, max_freq = max_freq)

# Plotting
print("Plotting the spectrograms...")
num_loc = len(hydro_spec1)
fig, axes = subplots(num_loc, 1, figsize=(column_width, row_height * num_loc), sharex = True)

# Plot Signal 1
axes, mappable = plot_hydro_stft_spectrograms(hydro_spec1, 
                                       axes = axes,
                                       dbmin = min_db, dbmax = max_db,
                                       major_time_spacing = major_time_spacing, num_minor_time_ticks = num_minor_time_ticks,
                                       major_freq_spacing = major_freq_spacing, num_minor_freq_ticks = num_minor_freq_ticks,
                                       location_label_x = location_label_x, location_label_y = location_label_y,
                                       title = mode_name)

# Add colorbar
ax = axes[-1]
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, position, "Power (dB)",
             mappable = mappable,
             orientation = "vertical")

# Save the figure
figname = f"stationary_resonances_whole_deployment_hydro_specs_{mode_name}_{station_to_plot}_{suffix}.png"
save_figure(fig, figname)


