# Plot the spectrograms of one mode in Liu et al. (2025a) for presentation purposes

# Imports
from os.path import join
from argparse import ArgumentParser
from json import loads
from pandas import Timedelta, Timestamp
from pandas import date_range, read_csv
from matplotlib.pyplot import colormaps, figure, Line2D
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.dates import date2num
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import BboxConnector, BboxPatch

from utils_basic import SPECTROGRAM_DIR as dir_spec, PLOTTING_DIR as dir_plot
from utils_basic import STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo
from utils_basic import str2timestamp, get_mode_order
from utils_spec import get_spectrogram_file_suffix, read_geo_stft, read_hydro_stft
from utils_plot import HYDRO_PSD_LABEL as hydro_psd_label, GEO_PSD_LABEL as geo_psd_label
from utils_plot import add_zoom_effect, format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
# Parse the arguments
parser = ArgumentParser(description = "Plot the spectrograms of one mode in Liu et al. (2025a) for presentation purposes")
parser.add_argument("--mode_name", type = str, help = "Name of the mode to plot")
parser.add_argument("--location1", type = str, help = "Location of the hydrophone for plotting PR02549")
parser.add_argument("--location2", type = str, help = "Location of the hydrophone for plotting PR03822")
parser.add_argument("--station1", type = str, help = "Geophone station 1 for plotting")
parser.add_argument("--station2", type = str, help = "Geophone station 2 for plotting")

parser.add_argument("--window_length", type = float, default = 60.0, help = "Window length for the spectrograms in seconds")
parser.add_argument("--min_hydro_db", type = float, default = -10.0, help = "Minimum hydrophone dB")
parser.add_argument("--max_hydro_db", type = float, default = 10.0, help = "Maximum hydrophone dB")
parser.add_argument("--min_geo_db", type = float, default = -10.0, help = "Minimum geophone dB")
parser.add_argument("--max_geo_db", type = float, default = 10.0, help = "Maximum geophone dB")

parser.add_argument("--colormap_name", type = str, default = "plasma", help = "Name of the colormap")
parser.add_argument("--color_ref", type = str, default = "limegreen", help = "Color of the reference line")

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name
location1 = args.location1
location2 = args.location2
station1 = args.station1
station2 = args.station2

window_length = args.window_length
min_hydro_db = args.min_hydro_db
max_hydro_db = args.max_hydro_db
min_geo_db = args.min_geo_db
max_geo_db = args.max_geo_db

colormap_name = args.colormap_name
color_ref = args.color_ref

# Constants
station_hydro = "A00"

overlap = 0.0

hspace = 0.1
wspace = 0.04

figwidth = 16.0
figheight = 8.0

station_label_fontsize = 14
long_label_x = 0.010
long_label_y = 0.96

short_label_x = 0.035
short_label_y = 0.96

cax_x = 0.04
cax_y = 0.15
cax_width = 0.3
cax_height = 0.03

cax_pad1_x = 0.07
cax_pad1_y = 0.01

cax_pad2_x = 0.12
cax_pad2_y = 0.01

cax_label_offset = -3.0

major_time_spacing_long = "30d"
num_minor_time_ticks_long = 6

major_time_spacing_short = "10d"
num_minor_time_ticks_short = 10

major_freq_spacing = 0.5
num_minor_freq_ticks = 5

linewidth_time = 3.0
linewidth_box = 2.0
linewidth_arrow = 1.5

annoatation_size = 10

noise_annotation_gap = Timedelta("4.5d")
arrow_length_noise = Timedelta("4d")

time_jump = str2timestamp("2019-12-10 00:00:00")
arrow_length_jumping = 0.2

time_break = str2timestamp("2019-07-20 00:00:00")
arrow_length_break = 0.2

# scalebar_coord_long = (0.03, 0.1)
# scalebar_label_offsets_long = (0.02, 0.0)

# scalebar_coord_short = (0.06, 0.1)
# scalebar_label_offsets_short = (0.03, 0.0)

scalebar_length = 1.0 # in days

marker_size = 0.2

axis_label_size = 12
tick_label_size = 10
title_fontsize = 14

subplot_label_fontsize = 14
subplot_label_offset = (-0.01, 0.01)

# Print the parameters
print("### Plotting the spectrograms in Liu et al. (2025a) ###")
print("Location 1: ", location1)
print("Location 2: ", location2)
print("Station 1: ", station1)
print("Station 2: ", station2)
print(f"Hydrophone dB range: {min_hydro_db} - {max_hydro_db} dB")
print(f"Geophone dB range: {min_geo_db} - {max_geo_db} dB")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")

### Generate the figure ###
print("Generating the figure...")
fig = figure(figsize = (figwidth, figheight))

cmap = colormaps[colormap_name].copy()
cmap.set_bad(color='darkgray')


### Generate the subplots ###
# Top subplots for Mode 2
# 1st row: 9-month hydrophone spectrogram, 2nd row: 21-day hydrophone spectrogram and geophone spectrograms of the two stations
gs = GridSpec(2, 3, figure=fig, hspace = hspace, wspace = wspace)

### Read the plotting frequency ranges ###
filename = f"stationary_resonance_freq_ranges_hydro.csv"
inpath = join(dir_spec, filename)
freq_range_df = read_csv(inpath)

min_mode_hydro_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_plot"].values[0]
max_mode_hydro_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_plot"].values[0]

filename = f"stationary_resonance_freq_ranges_geo.csv"
inpath = join(dir_spec, filename)
freq_range_df = read_csv(inpath)

min_mode_geo_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_plot"].values[0]
max_mode_geo_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_plot"].values[0]

### Read the hydrophone and geophone spectrograms ###
print("Reading the spectrograms of PR02549 and PR03822...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

filename = f"whole_deployment_daily_hydro_stft_{station_hydro}_{suffix_spec}.h5"
inpath = join(dir_spec, filename)

stream_hydro = read_hydro_stft(inpath,
                                locations = location1,
                                starttime = starttime_hydro, endtime = endtime_hydro,
                                min_freq = min_mode_hydro_freq, max_freq = max_mode_hydro_freq,
                                psd_only = True)

filename = f"whole_deployment_daily_geo_stft_{station1}_{suffix_spec}.h5"
inpath = join(dir_spec, filename)

stream_geo1 = read_geo_stft(inpath,
                                starttime = starttime_geo, endtime = endtime_geo,
                                min_freq = min_mode_geo_freq, max_freq = max_mode_geo_freq,
                                psd_only = True)

filename = f"whole_deployment_daily_geo_stft_{station2}_{suffix_spec}.h5"
inpath = join(dir_spec, filename)

stream_geo2 = read_geo_stft(inpath,
                            starttime = starttime_geo, endtime = endtime_geo,
                            min_freq = min_mode_geo_freq, max_freq = max_mode_geo_freq,
                            psd_only = True)

# # Read the frequencies of the instrument noise
# filename = f"liu_2025a_spectrograms_instrument_noise_freqs_{mode1_name}.csv"
# inpath = join(dir_plot, filename)
# noise_freq1_df = read_csv(inpath, index_col = 0)

# filename = f"liu_2025a_spectrograms_instrument_noise_freqs_{mode2_name}.csv"
# inpath = join(dir_plot, filename)
# noise_freq2_df = read_csv(inpath, index_col = 0)

### Plot PR02549 (Mode 2) ###
# Plot the 9-month hydrophone spectrogram
print("Plotting the 9-months hydrophone spectrograms of Mode 2 and 3...")

ax = fig.add_subplot(gs[0, :])
trace_stft = stream_hydro[0]
trace_stft.to_db()
psd_mat = trace_stft.psd_mat
freqax = trace_stft.freqs
timeax = trace_stft.times

ax.set_xlim(starttime_hydro, endtime_hydro)
ax.set_ylim(min_mode_hydro_freq, max_mode_hydro_freq)

mappable_hydro = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_hydro_db, vmax = max_hydro_db)

ax.text(long_label_x, long_label_y, f"{station_hydro}.{location1}",
        fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
        va = "top", ha = "left",
        bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))
        
format_datetime_xlabels(ax,
                        plot_axis_label = False, plot_tick_label = True,
                        major_tick_spacing = major_time_spacing_long, num_minor_ticks = num_minor_time_ticks_long, date_format = "%Y-%m-%d")


# add_horizontal_scalebar(ax, scalebar_coord_long, "1d", 1.0, color = color_ref, plot_label = True,
#                         label = "1 day", label_offsets = scalebar_label_offsets_long, fontsize = 10, fontweight = "bold")

format_freq_ylabels(ax,
                    plot_axis_label = True, plot_tick_label = True,
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

mode_order = get_mode_order(mode_name)
ax.set_title(f"Mode {mode_order}", fontsize = title_fontsize, fontweight = "bold")

ax_long = ax

# Plot the arrow pointing to the break
freq = (max_mode_hydro_freq + min_mode_hydro_freq) / 2
ax.annotate("Break", xy = (time_break, freq), xytext = (time_break, freq - arrow_length_break),
            arrowprops = dict(color = color_ref, arrowstyle = "->", linewidth = linewidth_arrow),
            fontsize = annoatation_size, fontweight = "bold", color = color_ref,
            ha = "center", va = "top")      

# ax.annotate("Freq. increase", xy = (time_jump, freq), xytext = (time_jump, freq - arrow_length_jumping),
#             arrowprops = dict(color = color_ref, arrowstyle = "->", linewidth = linewidth_arrow),
#             fontsize = annoatation_size, fontweight = "bold", color = color_ref,
#             ha = "center", va = "top")

# # Plot the arrows at the frequencies of the instrument noise
# noise_freqs = noise_freq1_df["frequency"].values
# for noise_freq in noise_freqs:
#     ax.annotate("", xy = (endtime, noise_freq), xytext = (endtime + arrow_length_noise, noise_freq),
#                 arrowprops = dict(color = "gray", arrowstyle = "->", linewidth = linewidth_arrow))

# Plot the 21-day hydrophone spectrogram
ax = fig.add_subplot(gs[1, 2])
stream_hydro_short = stream_hydro.slice_time(starttime = starttime_geo, endtime = endtime_geo)
trace_stft = stream_hydro_short[0]
trace_stft.to_db()
psd_mat = trace_stft.psd_mat
freqax = trace_stft.freqs
timeax = trace_stft.times

ax.set_xlim(starttime_geo, endtime_geo)
ax.set_ylim(min_mode_hydro_freq, max_mode_hydro_freq)

ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_hydro_db, vmax = max_hydro_db)

format_datetime_xlabels(ax,
                        plot_axis_label = True, plot_tick_label = True,
                        major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")

format_freq_ylabels(ax,
                    plot_axis_label = False, plot_tick_label = False,
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

ax.text(short_label_x, short_label_y, f"{station_hydro}.{location1}",
        fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
        va = "top", ha = "left",
        bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

# Add the colorbar with a frame
ax.add_patch(Rectangle((0.0, 0.0), cax_x + cax_width + cax_pad1_x, cax_y + cax_height + 2 * cax_pad1_y, transform=ax.transAxes, 
                      facecolor='white', edgecolor='black'))

cax = ax.inset_axes([cax_x, cax_y, cax_width, cax_height])
cbar = fig.colorbar(mappable_hydro, cax=cax, orientation="horizontal")
cbar.ax.tick_params(labelsize = tick_label_size)
cbar.ax.set_xlabel(hydro_psd_label, fontsize = tick_label_size, ha = "left", va = "top")
cbar.ax.xaxis.set_label_coords(0.0, cax_label_offset)

# Add the scale bar
scalebar = AnchoredSizeBar(ax.transData, scalebar_length, 
                            f"{scalebar_length:.0f} day", loc = "lower right", bbox_transform = ax.transAxes, frameon = True, size_vertical = 0.01, pad = 0.5, sep = 5.0, fontproperties = FontProperties(size = tick_label_size))
ax.add_artist(scalebar)

ax_short = ax

# Connect the zoom-in effect between the long and short spectrograms
prop_lines = {"color": color_ref, "linewidth": linewidth_time}
prop_patches = {"edgecolor": color_ref, "linewidth": linewidth_box, "facecolor": "none", "zorder": 10}

_, _, _, _ = add_zoom_effect(ax_long, ax_short, date2num(starttime_geo), date2num(endtime_geo), prop_lines, prop_patches)

# Plot the 21-day geophone spectrograms
# Station 1
ax = fig.add_subplot(gs[1, 0])
trace_stft = stream_geo1[0]
trace_stft.to_db()
psd_mat = trace_stft.psd_mat
freqax = trace_stft.freqs
timeax = trace_stft.times

ax.set_xlim(starttime_geo, endtime_geo)
ax.set_ylim(min_mode_geo_freq, max_mode_geo_freq)

mappable_geo = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_geo_db, vmax = max_geo_db)

format_datetime_xlabels(ax,
                        plot_axis_label = False, plot_tick_label = True,
                        major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")

format_freq_ylabels(ax,
                    plot_axis_label = False, plot_tick_label = True,
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

ax.text(short_label_x, short_label_y, f"{station1}",
        fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
        va = "top", ha = "left",
        bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

# Add the colorbar and a frame
ax.add_patch(Rectangle((0.0, 0.0), cax_x + cax_width + cax_pad2_x, cax_y + cax_height + 2 * cax_pad2_y, transform=ax.transAxes, 
                      facecolor='white', edgecolor='black'))

cax = ax.inset_axes([cax_x, cax_y, cax_width, cax_height])
cbar = fig.colorbar(mappable_geo, cax=cax, orientation="horizontal")
cbar.ax.tick_params(labelsize = tick_label_size)
cbar.ax.set_xlabel(geo_psd_label, fontsize = tick_label_size, ha = "left", va = "top")
cbar.ax.xaxis.set_label_coords(0.0, cax_label_offset)

# Station 2
ax = fig.add_subplot(gs[1, 1])
trace_stft = stream_geo2[0]
trace_stft.to_db()
psd_mat = trace_stft.psd_mat
freqax = trace_stft.freqs
timeax = trace_stft.times

ax.set_xlim(starttime_geo, endtime_geo)
ax.set_ylim(min_mode_geo_freq, max_mode_geo_freq)

ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_geo_db, vmax = max_geo_db)

format_datetime_xlabels(ax,
                        plot_axis_label = False, plot_tick_label = True,
                        major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")

format_freq_ylabels(ax,
                    plot_axis_label = False, plot_tick_label = False,
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

ax.text(short_label_x, short_label_y, f"{station2}",
        fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
        va = "top", ha = "left",
        bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))


### Save the figure ###
print("Saving the figure...")
figname = f"liu_2025a_spectrograms_{mode_name}.png"
save_figure(fig, figname, dpi = 600)