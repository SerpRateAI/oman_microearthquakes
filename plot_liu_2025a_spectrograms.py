# Plot the spectrograms in Liu et al. (2025a)
# The figure consist of 8 rows and 2 columns, with the top 4 rows showing the hydrophone spectrograms of PR02549 (Column 1) and PR03822 (Column 2) during the 9-month deoployment period plotted in 4 rows.
# The bottom 4 rows show the zoom-in view of the hydrophone spectrograms of PR02549 and PR03822 during the 21-day period of the geophone deployment and the corresponding geophone spectrograms of 3 stations in the same period

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
from mpl_toolkits.axes_grid1.inset_locator import BboxConnector, BboxPatch

from utils_basic import SPECTROGRAM_DIR as dir_spec, PLOTTING_DIR as dir_plot
from utils_basic import STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo
from utils_basic import str2timestamp
from utils_spec import get_spectrogram_file_suffix, read_geo_stft, read_hydro_stft
from utils_plot import HYDRO_PSD_LABEL as hydro_psd_label, GEO_PSD_LABEL as geo_psd_label
from utils_plot import add_colorbar, add_horizontal_scalebar, format_datetime_xlabels, format_freq_ylabels, save_figure

# Functions for connecting the boxes in the overview and the corners of the zoom-in views
def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches):

    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2

def add_zoom_effect(ax1, ax2, xmin, xmax, prop_lines, prop_patches):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both Axes will
    be marked.

    Parameters
    ----------
    ax1
        The main Axes.
    ax2
        The zoomed Axes.
    xmin, xmax
        The limits of the colored area in both plot Axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    bbox = Bbox.from_extents(xmin, 0, xmax, 1)

    mybbox1 = TransformedBbox(bbox, ax1.get_xaxis_transform())
    mybbox2 = TransformedBbox(bbox, ax2.get_xaxis_transform())

    c1, c2, bbox_patch1, bbox_patch2 = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=prop_lines, prop_patches=prop_patches)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)

    return c1, c2, bbox_patch1, bbox_patch2

# Inputs
# Parse the arguments
parser = ArgumentParser(description = "Plot the spectrograms in Liu et al. (2025a)")
parser.add_argument("--location1", type = str, help = "Location of the hydrophone for plotting PR02549")
parser.add_argument("--location2", type = str, help = "Location of the hydrophone for plotting PR03822")
parser.add_argument("--stations_geo", type = str, help = "Geophone stations to plot")

parser.add_argument("--window_length", type = float, default = 60.0, help = "Window length for the spectrograms in seconds")
parser.add_argument("--min_hydro_db", type = float, default = -10.0, help = "Minimum hydrophone dB")
parser.add_argument("--max_hydro_db", type = float, default = 10.0, help = "Maximum hydrophone dB")
parser.add_argument("--min_geo_db", type = float, default = -15.0, help = "Minimum geophone dB")
parser.add_argument("--max_geo_db", type = float, default = 15.0, help = "Maximum geophone dB")

# Parse the arguments
args = parser.parse_args()
location1 = args.location1
location2 = args.location2
stations_geo = loads(args.stations_geo)

window_length = args.window_length
min_hydro_db = args.min_hydro_db
max_hydro_db = args.max_hydro_db
min_geo_db = args.min_geo_db
max_geo_db = args.max_geo_db

# Constants
station_hydro = "A00"
mode1_name = "PR02549"
mode2_name = "PR03822"
mode1_num = 2
mode2_num = 3
overlap = 0.0

num_row_long = 4

hspace_long = 0.2
hspace_short = 0.1

vspace = 0.05

figwidth = 16.0
figheight = 16.0

station_label_fontsize = 12
station_label_x = 0.015
station_label_y = 0.92

colorbar_width = 0.005
colorbar_gap = 0.01

major_time_spacing_long = "15d"
num_minor_time_ticks_long = 3

major_time_spacing_short = "5d"
num_minor_time_ticks_short = 5

major_freq_spacing = 0.5
num_minor_freq_ticks = 5

color_ref = "deepskyblue"
linewidth_time = 3.0
linewidth_box = 2.0
linewidth_arrow = 1.5

annoatation_size = 10

noise_annotation_gap = Timedelta("4.5d")
arrow_length_noise = Timedelta("4d")

time_hammer = str2timestamp("2020-01-25 08:00:00")
arrow_length_hammer = 0.2

time_break = str2timestamp("2019-07-20 00:00:00")
arrow_length_break = 0.2

scalebar_coord_long = (0.03, 0.1)
scalebar_label_offsets_long = (0.02, 0.0)

scalebar_coord_short = (0.06, 0.1)
scalebar_label_offsets_short = (0.03, 0.0)

marker_size = 0.2

title_fontsize = 14

subplot_label_fontsize = 14
subplot_label_offset = (-0.01, 0.01)

# Print the parameters
print("### Plotting the spectrograms in Liu et al. (2025a) ###")
print("Location 1: ", location1)
print("Location 2: ", location2)
print("Geophone stations: ", stations_geo)
print(f"Hydrophone dB range: {min_hydro_db} - {max_hydro_db} dB")
print(f"Geophone dB range: {min_geo_db} - {max_geo_db} dB")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")

### Generate the subplots ###
print("Generating the figure...")
fig = figure(figsize = (figwidth, figheight))

# Create the top four rows for the 9-month hydrophone spectrograms and the bottom four rows for the 21-day geophone spectrograms
top_gs = GridSpec(num_row_long, 2, figure=fig, hspace = hspace_long, top = 0.95, bottom = 0.53)

# Create the bottom four rows with larger spacing
num_geo = len(stations_geo)
bottom_gs = GridSpec(num_geo + 1, 2, figure=fig, hspace = hspace_short, top = 0.47, bottom = 0.05)

### Read the plotting frequency ranges ###
filename = f"stationary_resonance_freq_ranges_hydro.csv"
inpath = join(dir_spec, filename)
freq_range_df = read_csv(inpath)

min_mode1_hydro_freq = freq_range_df[freq_range_df["mode_name"] == mode1_name]["min_freq_plot"].values[0]
max_mode1_hydro_freq = freq_range_df[freq_range_df["mode_name"] == mode1_name]["max_freq_plot"].values[0]

min_mode2_hydro_freq = freq_range_df[freq_range_df["mode_name"] == mode2_name]["min_freq_plot"].values[0]
max_mode2_hydro_freq = freq_range_df[freq_range_df["mode_name"] == mode2_name]["max_freq_plot"].values[0]

filename = f"stationary_resonance_freq_ranges_geo.csv"
inpath = join(dir_spec, filename)
freq_range_df = read_csv(inpath)

min_mode1_geo_freq = freq_range_df[freq_range_df["mode_name"] == mode1_name]["min_freq_plot"].values[0]
max_mode1_geo_freq = freq_range_df[freq_range_df["mode_name"] == mode1_name]["max_freq_plot"].values[0]

min_mode2_geo_freq = freq_range_df[freq_range_df["mode_name"] == mode2_name]["min_freq_plot"].values[0]
max_mode2_geo_freq = freq_range_df[freq_range_df["mode_name"] == mode2_name]["max_freq_plot"].values[0]

### Plot the 9-months hydrophone spectrograms of PR02549 and PR03822 in 4 rows
# Read the spectrograms
print("Reading the spectrograms of PR02549 and PR03822...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

filename = f"whole_deployment_daily_hydro_stft_{station_hydro}_{suffix_spec}.h5"
inpath = join(dir_spec, filename)

stream_mode1_stft = read_hydro_stft(inpath,
                                    locations = location1,
                                    starttime = starttime_hydro, endtime = endtime_hydro,
                                    min_freq = min_mode1_hydro_freq, max_freq = max_mode1_hydro_freq,
                                    psd_only = True)

stream_mode2_stft = read_hydro_stft(inpath,
                                    locations = location2,
                                    starttime = starttime_hydro, endtime = endtime_hydro,
                                    min_freq = min_mode2_hydro_freq, max_freq = max_mode2_hydro_freq,
                                    psd_only = True)

# # Read the frequencies of the instrument noise
# filename = f"liu_2025a_spectrograms_instrument_noise_freqs_{mode1_name}.csv"
# inpath = join(dir_plot, filename)
# noise_freq1_df = read_csv(inpath, index_col = 0)

# filename = f"liu_2025a_spectrograms_instrument_noise_freqs_{mode2_name}.csv"
# inpath = join(dir_plot, filename)
# noise_freq2_df = read_csv(inpath, index_col = 0)

# Plot each time window
print("Plotting the 9-months hydrophone spectrograms of PR02549 and PR03822...")
windows = date_range(starttime_hydro, endtime_hydro, periods = num_row_long + 1)
cmap = colormaps["inferno"].copy()
cmap.set_bad(color='darkgray')

for i in range(num_row_long):
    starttime = windows[i]
    endtime = windows[i + 1]

    print("Plotting the time window: ", starttime, " - ", endtime)
    stream_mode1_stft_window = stream_mode1_stft.slice_time(starttime = starttime, endtime = endtime)
    stream_mode2_stft_window = stream_mode2_stft.slice_time(starttime = starttime, endtime = endtime)
    
    print(f"Plotting the spectrogram of {mode1_name} for the window")
    ax = fig.add_subplot(top_gs[i, 0])
    trace_stft = stream_mode1_stft_window[0]
    trace_stft.to_db()
    psd_mat = trace_stft.psd_mat
    freqax = trace_stft.freqs
    timeax = trace_stft.times

    ax.set_xlim(starttime, endtime)

    # Plot the spectrogram
    mappable = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_hydro_db, vmax = max_hydro_db)

    # Plot the location label
    if i == 0:
        ax.text(station_label_x, station_label_y, f"{station_hydro}.{location1}",
                fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
                va = "top", ha = "left",
                bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))
        

    format_datetime_xlabels(ax,
                            label = False,
                            major_tick_spacing = major_time_spacing_long, num_minor_ticks = num_minor_time_ticks_long, date_format = "%Y-%m-%d")

    if i < num_row_long - 1:    
        add_horizontal_scalebar(ax, scalebar_coord_long, "1d", 1.0, color = color_ref, plot_label = False)
    else:
        add_horizontal_scalebar(ax, scalebar_coord_long, "1d", 1.0, color = color_ref, plot_label = True,
                                label = "1d", label_offsets = scalebar_label_offsets_long, fontsize = 10, fontweight = "bold")

    if i < num_row_long - 1:
        format_freq_ylabels(ax,
                            label = False,
                            major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)
    else:
        format_freq_ylabels(ax,
                            label = True,
                            major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

    if i == 0:
        ax.set_title(f"Mode {mode1_num}", fontsize = title_fontsize, fontweight = "bold")

    if i == num_row_long - 1:
        ax_main1 = ax

    # Plot the arrow pointing to the break
    if time_break >= starttime and time_break <= endtime:
        freq = (max_mode1_hydro_freq + min_mode1_hydro_freq) / 2
        ax.annotate("Break", xy = (time_break, freq), xytext = (time_break, freq - arrow_length_break),
                    arrowprops = dict(color = color_ref, arrowstyle = "->", linewidth = linewidth_arrow),
                    fontsize = annoatation_size, fontweight = "bold", color = color_ref, 
                    ha = "center", va = "top")

    # # Plot the arrows at the frequencies of the instrument noise
    # noise_freqs = noise_freq1_df["frequency"].values
    # for noise_freq in noise_freqs:
    #     ax.annotate("", xy = (endtime, noise_freq), xytext = (endtime + arrow_length_noise, noise_freq),
    #                 arrowprops = dict(color = "gray", arrowstyle = "->", linewidth = linewidth_arrow))

    # Plot the subplot label
    if i == 0:
        bbox = ax.get_position()
        top_left_x = bbox.x0
        top_left_y = bbox.y1
        fig.text(top_left_x + subplot_label_offset[0], top_left_y + subplot_label_offset[1], "(a)", 
                 fontsize = subplot_label_fontsize, fontweight = "bold",
                 va = "bottom", ha = "right")

    print(f"Plotting the spectrogram of {mode2_name} for the window")
    ax = fig.add_subplot(top_gs[i, 1])
    trace_stft = stream_mode2_stft_window[0]
    trace_stft.to_db()
    psd_mat = trace_stft.psd_mat
    freqax = trace_stft.freqs
    timeax = trace_stft.times

    ax.set_xlim(starttime, endtime)
    ax.set_ylim(min_mode2_hydro_freq, max_mode2_hydro_freq)

    if i == 0:
        ax.text(station_label_x, station_label_y, f"{station_hydro}.{location2}",
                fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
                va = "top", ha = "left",
                bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

    # Plot the spectrogram
    mappable = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_hydro_db, vmax = max_hydro_db)

    format_datetime_xlabels(ax,
                            label = False,
                            major_tick_spacing = major_time_spacing_long, num_minor_ticks = num_minor_time_ticks_long, date_format = "%Y-%m-%d")
        
    if i < num_row_long - 1:
        add_horizontal_scalebar(ax, scalebar_coord_long, "1d", 1.0, color = color_ref, plot_label = False)
    else:
        add_horizontal_scalebar(ax, scalebar_coord_long, "1d", 1.0, color = color_ref, plot_label = True,
                                label = "1d", label_offsets = scalebar_label_offsets_long, fontsize = 10, fontweight = "bold")
        
    format_freq_ylabels(ax, 
                        label = False,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

    if i == 0:
        ax.set_title(f"Mode {mode2_num}", fontsize = title_fontsize, fontweight = "bold")

    if i == num_row_long - 1:
        ax_main2 = ax

    # Plot the arrow pointing to the break
    if time_break >= starttime and time_break <= endtime:
        freq = (max_mode2_hydro_freq + min_mode2_hydro_freq) / 2
        ax.annotate("", xy = (time_break, freq), xytext = (time_break, freq - arrow_length_break),
                    arrowprops = dict(color = color_ref, arrowstyle = "->", linewidth = linewidth_arrow),
                    fontsize = annoatation_size, fontweight = "bold", color = color_ref, 
                    ha = "center", va = "top")

    # # Plot the arrows at the frequencies of the instrument noise
    # noise_freqs = noise_freq2_df["frequency"].values
    # for j, noise_freq in enumerate(noise_freqs):
    #     ax.annotate("", xy = (endtime, noise_freq), xytext = (endtime + arrow_length_noise, noise_freq),
    #                 arrowprops = dict(color = "gray", arrowstyle = "->", linewidth = linewidth_arrow))

    # # Plot the annoations of the instrument noise
    # if i == 2:
    #     ax.text(endtime + noise_annotation_gap, noise_freqs[-1], "Instrument noise",
    #             fontsize = annoatation_size, fontweight = "bold", color = "gray", ha = "left", va = "center", rotation = 90)

    # Plot the subplot label
    if i == 0:
        bbox = ax.get_position()
        top_left_x = bbox.x0
        top_left_y = bbox.y1
        fig.text(top_left_x + subplot_label_offset[0], top_left_y + subplot_label_offset[1], "(b)", 
                 fontsize = subplot_label_fontsize, fontweight = "bold",
                 va = "bottom", ha = "right")
        
        
### Plot the 21-days hydrophone spectrograms ###
print("Plotting the 21-days hydrophone spectrograms...")

# Read the plotting frequency range
filename = f"stationary_resonance_freq_ranges_geo.csv"
inpath = join(dir_spec, filename)
freq_range_df = read_csv(inpath)

min_mode1_freq = freq_range_df[freq_range_df["mode_name"] == mode1_name]["min_freq_plot"].values[0]
max_mode1_freq = freq_range_df[freq_range_df["mode_name"] == mode1_name]["max_freq_plot"].values[0]

min_mode2_freq = freq_range_df[freq_range_df["mode_name"] == mode2_name]["min_freq_plot"].values[0]
max_mode2_freq = freq_range_df[freq_range_df["mode_name"] == mode2_name]["max_freq_plot"].values[0]

# Slice the hydrophone spectrograms in time and frequency
stream_mode1_stft_window = stream_mode1_stft.slice_time(starttime = starttime_geo, endtime = endtime_geo)
stream_mode1_stft_window = stream_mode1_stft_window.slice_freq(min_freq = min_mode1_freq, max_freq = max_mode1_freq)

stream_mode2_stft_window = stream_mode2_stft.slice_time(starttime = starttime_geo, endtime = endtime_geo)
stream_mode2_stft_window = stream_mode2_stft_window.slice_freq(min_freq = min_mode2_freq, max_freq = max_mode2_freq)

# Plot the hydrophone spectrograms
print(f"Plotting the 21-days hydrophone spectrograms of {mode1_name}...")
ax = fig.add_subplot(bottom_gs[0, 0])

trace_stft = stream_mode1_stft_window[0]
trace_stft.to_db()

psd_mat = trace_stft.psd_mat
freqax = trace_stft.freqs
timeax = trace_stft.times

ax.set_xlim(starttime_geo, endtime_geo)
ax.set_ylim(min_mode1_freq, max_mode1_freq)

# Connect the boxes in the overview and the corners of the the zoom-in views
prop_lines = {"color": color_ref, "linewidth": linewidth_time}
prop_patches = {"edgecolor": color_ref, "linewidth": linewidth_box, "facecolor": "none", "zorder": 10}

_, _, _, _ = add_zoom_effect(ax_main1, ax, date2num(starttime_geo), date2num(endtime_geo), prop_lines, prop_patches)

# Plot the spectrogram
mappable = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_hydro_db, vmax = max_hydro_db)

ax.text(station_label_x, station_label_y, f"{station_hydro}.{location1}",
        fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
        va = "top", ha = "left",
        bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

format_datetime_xlabels(ax,
                        label = False,
                        major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")

ax.set_xticklabels([])

add_horizontal_scalebar(ax, scalebar_coord_short, "1d", 1.0, color = color_ref, plot_label = False)

if i < num_row_long - 1:
    format_freq_ylabels(ax,
                        label = False,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)
else:
    format_freq_ylabels(ax, 
                        label = False,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

# Plot the annotation of the hammer
ax.annotate("Active-source experiment", xy = (time_hammer, max_mode1_freq), xytext = (time_hammer, max_mode1_freq + arrow_length_hammer),
            arrowprops = dict(color = "gray", arrowstyle = "->", linewidth = linewidth_arrow),
            fontsize = annoatation_size, fontweight = "bold", color = "gray", 
            ha = "center", va = "bottom")

# Add the subplot label
bbox = ax.get_position()
top_left_x = bbox.x0
top_left_y = bbox.y1
fig.text(top_left_x + subplot_label_offset[0], top_left_y + subplot_label_offset[1], "(c)",
            fontsize = subplot_label_fontsize, fontweight = "bold",
            va = "bottom", ha = "right")

print(f"Plotting the 21-days hydrophone spectrograms of {mode2_name}...")

ax = fig.add_subplot(bottom_gs[0, 1])
trace_stft = stream_mode2_stft_window[0]
trace_stft.to_db()

psd_mat = trace_stft.psd_mat
freqax = trace_stft.freqs
timeax = trace_stft.times

ax.set_xlim(starttime_geo, endtime_geo)
ax.set_ylim(min_mode2_freq, max_mode2_freq)

# Connect the boxes in the overview and the corners of the the zoom-in views
_, _, _, _ = add_zoom_effect(ax_main2, ax, date2num(starttime_geo), date2num(endtime_geo), prop_lines, prop_patches)

mappable = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_hydro_db, vmax = max_hydro_db)

ax.text(station_label_x, station_label_y, f"{station_hydro}.{location2}",
        fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
        va = "top", ha = "left",
        bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

format_datetime_xlabels(ax,
                        label = False,
                        major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")

ax.set_xticklabels([])

add_horizontal_scalebar(ax, scalebar_coord_short, "1d", 1.0, color = color_ref, plot_label = False)

format_freq_ylabels(ax, 
                    label = False,
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

# Plot the annotation of the hammer
ax.annotate("", xy = (time_hammer, max_mode2_freq), xytext = (time_hammer, max_mode2_freq + arrow_length_hammer),
            arrowprops = dict(color = "gray", arrowstyle = "->", linewidth = linewidth_arrow), fontsize = annoatation_size, color = "gray", ha = "center", va = "bottom")

# Add the colorbar for the hydrophone spectrograms
print("Adding the colorbar for the hydrophone spectrograms...")
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, position, hydro_psd_label, 
             mappable = mappable,
             orientation = "vertical")

# Add the subplot label
bbox = ax.get_position()
top_left_x = bbox.x0
top_left_y = bbox.y1
fig.text(top_left_x + subplot_label_offset[0], top_left_y + subplot_label_offset[1], "(d)",
         fontsize = subplot_label_fontsize, fontweight = "bold",
         va = "bottom", ha = "right")

### Plot the 21-days geophone spectrograms ###
print("Plotting the geophone spectrograms...")
# Read the geophone spectrograms
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

for i, station in enumerate(stations_geo):
    print(f"Reading the spectrograms of {station}...")
    filename = f"whole_deployment_daily_geo_stft_{station}_{suffix_spec}.h5"
    inpath = join(dir_spec, filename)

    stream_mode1_stft = read_geo_stft(inpath,
                                    starttime = starttime_geo, endtime = endtime_geo,
                                    min_freq = min_mode1_geo_freq, max_freq = max_mode1_geo_freq,
                                    psd_only = True)
    
    stream_mode2_stft = read_geo_stft(inpath,
                                    starttime = starttime_geo, endtime = endtime_geo,
                                    min_freq = min_mode2_geo_freq, max_freq = max_mode2_geo_freq,
                                    psd_only = True)
    

    # Compute the total PSD
    print("Computing the total PSD of the geophone spectrograms...")
    trace_mode1_stft = stream_mode1_stft.get_total_psd()
    trace_mode2_stft = stream_mode2_stft.get_total_psd()

    trace_mode1_stft.to_db()
    trace_mode2_stft.to_db()

    # Plot the geophone spectrograms
    print(f"Plotting the geophone spectrograms of {mode1_name} at {station}...")
    ax = fig.add_subplot(bottom_gs[i + 1, 0])
    psd_mat = trace_mode1_stft.psd_mat
    freqax = trace_mode1_stft.freqs
    timeax = trace_mode1_stft.times

    ax.set_xlim(starttime_geo, endtime_geo)

    # Plot the spectrogram
    mappable = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_geo_db, vmax = max_geo_db)

    # Plot the location label
    ax.text(station_label_x, station_label_y, f"{station}",
            fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
            va = "top", ha = "left",
            bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

    if i < num_geo - 1:
        format_datetime_xlabels(ax,
                                label = False,
                                major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")
    else:
        format_datetime_xlabels(ax,
                                label = True,
                                major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")
    
    if i < len(stations_geo) - 1:
        ax.set_xticklabels([])

    if i == len(stations_geo) - 1:
        add_horizontal_scalebar(ax, scalebar_coord_short, "1d", 1.0, color = color_ref, plot_label = True,
                                label = "1d", label_offsets = scalebar_label_offsets_short, fontsize = 10, fontweight = "bold")
    else:
        add_horizontal_scalebar(ax, scalebar_coord_short, "1d", 1.0, color = color_ref, plot_label = False)

    if i < num_geo - 1:
        format_freq_ylabels(ax,
                            label = False,
                            major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)
    else:
        format_freq_ylabels(ax,
                            label = True,
                            major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)


    print(f"Plotting the geophone spectrograms of {mode2_name} at {station}...")
    ax = fig.add_subplot(bottom_gs[i + 1, 1])
    psd_mat = trace_mode2_stft.psd_mat
    freqax = trace_mode2_stft.freqs
    timeax = trace_mode2_stft.times

    ax.set_xlim(starttime_geo, endtime_geo)

    # Plot the spectrogram
    mappable = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_geo_db, vmax = max_geo_db)

    ax.text(station_label_x, station_label_y, f"{station}",
            fontsize = station_label_fontsize, fontweight = "bold", transform = ax.transAxes,
            va = "top", ha = "left",
            bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

    if i < num_geo - 1:
        format_datetime_xlabels(ax,
                                label = False,
                                major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")
    else:
        format_datetime_xlabels(ax,
                                label = True,
                                major_tick_spacing = major_time_spacing_short, num_minor_ticks = num_minor_time_ticks_short, date_format = "%Y-%m-%d")

    if i < len(stations_geo) - 1:
        ax.set_xticklabels([])
    
    if i == len(stations_geo) - 1:
        add_horizontal_scalebar(ax, scalebar_coord_short, "1d", 1.0, color = color_ref, plot_label = True,
                                label = "1d", label_offsets = scalebar_label_offsets_short, fontsize = 10, fontweight = "bold")
    else:
        add_horizontal_scalebar(ax, scalebar_coord_short, "1d", 1.0, color = color_ref, plot_label = False)

    format_freq_ylabels(ax, 
                        label = False,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

# Add the colorbar for the geophone spectrograms
print("Adding the colorbar for the geophone spectrograms...")
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, position, geo_psd_label, 
             mappable = mappable,
             orientation = "vertical")

### Save the figure ###
print("Saving the figure...")
figname = "liu_2025a_spectrograms.png"
save_figure(fig, figname, dpi = 600)