"""
Plot the 9-month hydro-spectrograms and the temperature time series
"""
###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import interp, nan, sin, cos, pi, deg2rad, linspace, histogram, deg2rad, isnan
from pandas import read_csv, DataFrame
from matplotlib.pyplot import subplots, figure
from matplotlib.patches import Rectangle
from matplotlib import colormaps

from utils_basic import SPECTROGRAM_DIR as dir_spec
from utils_basic import STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo
from utils_basic import get_baro_temp_data
from utils_basic import str2timestamp
from utils_spec import get_spectrogram_file_suffix, read_geo_stft, read_hydro_stft
from utils_plot import HYDRO_PSD_LABEL as hydro_psd_label, GEO_PSD_LABEL as geo_psd_label
from utils_plot import format_datetime_xlabels, format_freq_ylabels, save_figure

###
# Inputs
###

parser = ArgumentParser(description = "Plot the 9-month hydro-spectrograms and the temperature time series")

parser.add_argument("--mode_name", type=str, help="The name of the mode")
parser.add_argument("--window_length", type = float, default = 60.0, help = "Window length for the spectrograms in seconds")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap percentage")

parser.add_argument("--station", type = str, default = "A00", help = "Station name of the hydrophone")
parser.add_argument("--location", type = str, default = "03", help = "Location of the hydrophone")

parser.add_argument("--min_db", type = float, default = -10.0, help = "Minimum hydrophone dB")
parser.add_argument("--max_db", type = float, default = 10.0, help = "Maximum hydrophone dB")
parser.add_argument("--min_temp", type = float, default = 10.0, help = "Minimum temperature")
parser.add_argument("--max_temp", type = float, default = 40.0, help = "Maximum temperature")

parser.add_argument("--colormap_name", type = str, default = "inferno", help = "Name of the colormap")
parser.add_argument("--color_temp", type = str, default = "black", help = "Color of the temperature line")

parser.add_argument("--figwidth", type = float, default = 15.0, help = "Width of the figure")
parser.add_argument("--figheight", type = float, default = 10.0, help = "Height of the figure")

parser.add_argument("--label_x", type = float, default = 0.010, help = "X-coordinate of the station label")
parser.add_argument("--label_y", type = float, default = 0.96, help = "Y-coordinate of the station label")

parser.add_argument("--major_time_spacing", type = str, default = "30d", help = "Major time spacing for the time labels")
parser.add_argument("--num_minor_time_ticks", type = int, default = 6, help = "Number of minor time ticks for the time labels")

parser.add_argument("--major_freq_spacing", type = float, default = 0.5, help = "Major frequency spacing for the frequency labels")
parser.add_argument("--num_minor_freq_ticks", type = int, default = 5, help = "Number of minor frequency ticks for the frequency labels")
 
parser.add_argument("--fontsize_station_label", type = float, default = 14, help = "Fontsize of the station label")
parser.add_argument("--fontsize_tick_label", type = float, default = 12, help = "Fontsize of the tick labels")

parser.add_argument("--cax_x", type = float, default = 0.01, help = "X-coordinate of the colorbar")
parser.add_argument("--cax_y", type = float, default = 0.02, help = "Y-coordinate of the colorbar")
parser.add_argument("--cax_width", type = float, default = 0.1, help = "Width of the colorbar")
parser.add_argument("--cax_height", type = float, default = 0.01, help = "Height of the colorbar")
parser.add_argument("--cax_pad_x", type = float, default = 0.01, help = "Padding of the colorbar")
parser.add_argument("--cax_pad_y", type = float, default = 0.01, help = "Padding of the colorbar")

parser.add_argument("--cax_label_offset", type = float, default = 0.01, help = "Offset of the colorbar label")

parser.add_argument("--linewidth_temp", type = float, default = 0.1, help = "Linewidth of the temperature line")


# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name
window_length = args.window_length
overlap = args.overlap
station = args.station
location = args.location
min_db = args.min_db
max_db = args.max_db
min_temp = args.min_temp
max_temp = args.max_temp
colormap_name = args.colormap_name
color_temp = args.color_temp
figwidth = args.figwidth
figheight = args.figheight
label_x = args.label_x
label_y = args.label_y
major_time_spacing = args.major_time_spacing
num_minor_time_ticks = args.num_minor_time_ticks
major_freq_spacing = args.major_freq_spacing
num_minor_freq_ticks = args.num_minor_freq_ticks
fontsize_station_label = args.fontsize_station_label
fontsize_tick_label = args.fontsize_tick_label
cax_x = args.cax_x
cax_y = args.cax_y
cax_width = args.cax_width
cax_height = args.cax_height
cax_label_offset = args.cax_label_offset
cax_pad_x = args.cax_pad_x
cax_pad_y = args.cax_pad_y
linewidth_temp = args.linewidth_temp

###
# Read the input files
###

### Read the plotting frequency ranges ###
filename = f"stationary_resonance_freq_ranges_hydro.csv"
inpath = join(dir_spec, filename)
freq_range_df = read_csv(inpath)

min_freq_plot = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_plot"].values[0]
max_freq_plot = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_plot"].values[0]

### Read the hydrophone spectrograms ###
print(f"Reading the spectrograms of {station}.{location}...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

filename = f"whole_deployment_daily_hydro_stft_{station}_{suffix_spec}.h5"
inpath = join(dir_spec, filename)

stream = read_hydro_stft(inpath,
                        locations = location,
                        starttime = starttime_hydro, endtime = endtime_hydro,
                        min_freq = min_freq_plot, max_freq = max_freq_plot,
                        psd_only = True)

### Read the barometric pressure and temperature data ###
print("Reading the barometric pressure and temperature data...")
baro_temp_df = get_baro_temp_data()

###
# Plotting
###

### Generate the subplots ###
fig, axs = subplots(2, 1, figsize = (figwidth, figheight))

### Plot the spectrograms ###
print("Plotting the spectrograms...")
ax_spec = axs[0]

trace_stft = stream[0]
trace_stft.to_db()
psd_mat = trace_stft.psd_mat
freqax = trace_stft.freqs
timeax = trace_stft.times

ax_spec.set_xlim(starttime_hydro, endtime_hydro)
ax_spec.set_ylim(min_freq_plot, max_freq_plot)

cmap = colormaps[colormap_name]
mappable_hydro = ax_spec.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_db, vmax = max_db,
                                    zorder = 1)

ax_spec.text(label_x, label_y, f"{station}.{location}",
        fontsize = fontsize_station_label, fontweight = "bold", transform = ax_spec.transAxes,
        va = "top", ha = "left",
        bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"),
        zorder = 2)
        
format_datetime_xlabels(ax_spec,
                        plot_axis_label = False, plot_tick_label = True,
                        major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, date_format = "%Y-%m-%d")

format_freq_ylabels(ax_spec,
                    plot_axis_label = True, plot_tick_label = True,
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

# Add the colorbar with a frame
print("Adding the colorbar...")
ax_spec.add_patch(Rectangle((0.0, 0.0), cax_x + cax_width + cax_pad_x, cax_y + cax_height + 2 * cax_pad_y, transform=ax_spec.transAxes, 
                      facecolor='white', edgecolor='black', zorder = 3))

cax = ax_spec.inset_axes([cax_x, cax_y, cax_width, cax_height])
cbar = fig.colorbar(mappable_hydro, cax=cax, orientation="horizontal")
cbar.ax.tick_params(labelsize = fontsize_tick_label)
cbar.ax.set_xlabel(hydro_psd_label, fontsize = fontsize_tick_label, ha = "left", va = "top")
cbar.ax.xaxis.set_label_coords(0.0, cax_label_offset)

### Plot the temperature time series ###
print("Plotting the temperature time series...")
ax_temp = axs[1]

ax_temp.plot(baro_temp_df.index, baro_temp_df["temperature"], color = color_temp, linewidth = linewidth_temp)

ax_temp.set_xlim(starttime_hydro, endtime_hydro)
ax_temp.set_ylim(min_temp, max_temp)

format_datetime_xlabels(ax_temp,
                        plot_axis_label = True, plot_tick_label = True,
                        major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, date_format = "%Y-%m-%d")

ax_temp.set_ylabel("Temperature (°C)", fontsize = fontsize_tick_label)


###
# Save the figure
###
filename = f"hydro_spectrogram_and_temp_time_series.png"
save_figure(fig, filename)
