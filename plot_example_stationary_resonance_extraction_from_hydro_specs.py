"""
Plot example stationary resonance extraction from a hydrophone spectrogram
"""

###
# Import modules
#
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, read_hdf
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as dirpath_spec, STARTTIME_HYDRO as starttime, ENDTIME_HYDRO as endtime
from utils_basic import get_mode_order
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_hydro_stft
from utils_plot import HYDRO_PSD_LABEL as psd_label
from utils_plot import format_datetime_xlabels, format_freq_ylabels, save_figure


###
# Parse command-line arguments
###
parser = ArgumentParser(description = "Plot example stationary resonance extraction from a hydrophone spectrogram")
parser.add_argument("--mode_name", type = str, help = "Mode name", default = "PR02549")
parser.add_argument("--station", type = str, help = "Station name", default = "A00")
parser.add_argument("--location", type = str, help = "Location name", default = "03")
parser.add_argument("--min_db", type = float, help = "Minimum dB", default = -20.0)
parser.add_argument("--max_db", type = float, help = "Maximum dB", default = 15.0)
parser.add_argument("--cmap_name", type = str, help = "Colormap name", default = "inferno")
parser.add_argument("--marker_size", type = float, help = "Marker size", default = 2.0)

parser.add_argument("--window_length", type = float, help = "Window length in seconds", default = 300.0)
parser.add_argument("--overlap", type = float, help = "Overlap in seconds", default = 0.0)
parser.add_argument("--min_prom", type = float, help = "Minimum prominence in dB", default = 15.0)
parser.add_argument("--min_rbw", type = float, help = "Minimum reverse bandwidth in 1/Hz", default = 15.0)
parser.add_argument("--max_mean_db", type = float, help = "Maximum mean dB for excluding noisy windows", default = -15.0)

parser.add_argument("--figwidth", type = float, help = "Figure width in inches", default = 15.0)
parser.add_argument("--figheight", type = float, help = "Figure height in inches", default = 15.0)

parser.add_argument("--major_time_spacing", type = str, help = "Major time spacing", default = "30d")
parser.add_argument("--num_minor_time_ticks", type = int, help = "Number of minor time ticks", default = 6)

parser.add_argument("--major_freq_spacing", type = float, help = "Major frequency spacing", default = 0.5)
parser.add_argument("--num_minor_freq_ticks", type = int, help = "Number of minor frequency ticks", default = 5)

parser.add_argument("--fontsize_title", type = int, help = "Fontsize of the title", default = 14)
parser.add_argument("--fontsize_axis_label", type = int, help = "Fontsize of the axis labels", default = 12)
parser.add_argument("--fontsize_tick_label", type = int, help = "Fontsize of the tick labels", default = 10)

parser.add_argument("--cbar_x", type = float, help = "Colorbar x", default = 0.02)
parser.add_argument("--cbar_y", type = float, help = "Colorbar y", default = 0.15)
parser.add_argument("--cbar_width", type = float, help = "Colorbar width", default = 0.15)
parser.add_argument("--cbar_height", type = float, help = "Colorbar height", default = 0.03)

args = parser.parse_args()

mode_name = args.mode_name
station = args.station
location = args.location
min_db = args.min_db
max_db = args.max_db
cmap_name = args.cmap_name
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
figwidth = args.figwidth
figheight = args.figheight
major_time_spacing = args.major_time_spacing
num_minor_time_ticks = args.num_minor_time_ticks
major_freq_spacing = args.major_freq_spacing
num_minor_freq_ticks = args.num_minor_freq_ticks
marker_size = args.marker_size
fontsize_title = args.fontsize_title
fontsize_axis_label = args.fontsize_axis_label
fontsize_tick_label = args.fontsize_tick_label
cbar_x = args.cbar_x
cbar_y = args.cbar_y
cbar_width = args.cbar_width
cbar_height = args.cbar_height

###
# Load the data
###

# Read the plotting frequency range
filename = f"stationary_resonance_freq_ranges_hydro.csv"
inpath = join(dirpath_spec, filename)
freq_range_df = read_csv(inpath)

min_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_plot"].values[0]
max_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_plot"].values[0]

print(f"Plotting frequency range: {min_freq} - {max_freq} Hz")

# Read the spectrograms
print("Reading the spectrograms...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"whole_deployment_daily_hydro_stft_{station}_{suffix_spec}.h5"
inpath = join(dirpath_spec, filename)

stream_stft = read_hydro_stft(inpath,
                              locations = location,
                              min_freq = min_freq, max_freq = max_freq,
                              starttime = starttime, endtime = endtime)

# Read the spectral peaks
print("Reading the spectral peaks...")
filename = f"stationary_resonance_properties_hydro_smooth_filter_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(dirpath_spec, filename)

peaks_df = read_hdf(inpath, key = "properties")
peaks_df = peaks_df[(peaks_df["station"] == station) & (peaks_df["location"] == location)]

###
# Plot the spectrogram and the spectral peaks
###

# Generate the figure
print("Generating the figure...")
fig, axes = subplots(2, 1, figsize=(figwidth, figheight))
fig.subplots_adjust(hspace = 0.04)

# Plot the spectrogram
print("Plotting the spectrogram...")
cmap = colormaps[cmap_name]
norm = Normalize(vmin = min_db, vmax = max_db)
cmap.set_bad(color = "darkgray")

trace_stft = stream_stft[0]
trace_stft.to_db()
psd_mat = trace_stft.psd_mat
freqs = trace_stft.freqs
times = trace_stft.times

ax = axes[0]
mappable = ax.pcolormesh(times, freqs, psd_mat, cmap = cmap, norm = norm)

# Set the time axis
ax.set_xlim(starttime, endtime)
ax.set_ylim(min_freq, max_freq)

# Set the time labels
format_datetime_xlabels(ax,
                        plot_axis_label = False,
                        plot_tick_label = False,
                        major_tick_spacing = major_time_spacing,
                        num_minor_ticks = num_minor_time_ticks,
                        date_format = "%Y-%m-%d")

# Set the frequency labels
format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_spacing,
                    num_minor_ticks = num_minor_freq_ticks)

# Plot the spectral peaks
print("Plotting the spectral peaks...")
ax = axes[1]
ax.set_facecolor("lightgray")
times = peaks_df["time"]
freqs = peaks_df["frequency"]
powers = peaks_df["power"]

ax.scatter(times, freqs,
           c = powers, cmap = cmap, norm = norm, s = marker_size)

# Set the axis limits
ax.set_xlim(starttime, endtime)
ax.set_ylim(min_freq, max_freq)

# Set the time labels
format_datetime_xlabels(ax,
                        major_tick_spacing = major_time_spacing,
                        num_minor_ticks = num_minor_time_ticks,
                        date_format = "%Y-%m-%d")

# Set the frequency labels
format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_spacing,
                    num_minor_ticks = num_minor_freq_ticks)

# Add a colorbar
cax = ax.inset_axes([cbar_x, cbar_y, cbar_width, cbar_height])
cbar = fig.colorbar(mappable, cax = cax, orientation = "horizontal")
cbar.ax.tick_params(labelsize = fontsize_tick_label)
cbar.ax.set_xlabel(psd_label, fontsize = fontsize_tick_label)

# Set the suptitle
mode_order = get_mode_order(mode_name)
fig.suptitle(f"Mode {mode_order} on {station}.{location}", fontsize = fontsize_title, fontweight = "bold", y = 0.92)

###
# Save the figure
###

figname = f"example_stationary_resonance_extraction_from_hydro_specs_{mode_name}_{station}.{location}.png"
save_figure(fig, figname)


















