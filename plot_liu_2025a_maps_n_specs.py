# Plot station maps, hydrophone depth profiles, and 3C spectra of a geophone station for Liu et al. (2025a)
from os.path import join
from numpy import cos, pi, linspace
from argparse import ArgumentParser
from json import loads
from pandas import Timedelta
from pandas import concat, read_csv, read_hdf
from matplotlib.pyplot import subplots
from rasterio import open
from rasterio.plot import reshape_as_image
# from cartopy.io.img_tiles import Stamen
# from cartopy.io.shapereader import Reader
# from cartopy.io.raster import RasterSource
# from cartopy.mpl.geoaxes import GeoAxes
from cartopy.crs import Orthographic, Geodetic
import cartopy.feature as cfeature

from utils_basic import EASTMIN_WHOLE as min_east_array, EASTMAX_WHOLE as max_east_array, NORTHMIN_WHOLE as min_north_array, NORTHMAX_WHOLE as max_north_array
from utils_basic import HYDRO_DEPTHS as depth_dict, GEO_COMPONENTS as components
from utils_basic import SPECTROGRAM_DIR as dir_spec, PLOTTING_DIR as dir_plot
from utils_basic import CENTER_LONGITUDE as lon, CENTER_LATITUDE as lat
from utils_basic import IMAGE_DIR as dir_img
from utils_basic import get_geophone_coords, get_borehole_coords, str2timestamp
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_time_slice_from_geo_stft
from utils_plot import component2label, format_east_xlabels, format_db_ylabels, format_freq_xlabels, format_north_ylabels, format_depth_ylabels, get_geo_component_color, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for plotting the station maps, hydrophone depth profiles, and 3C spectra of a geophone station for Liu et al. (2025a).")
parser.add_argument("--stations_highlight", type=str, help="List of highlighted geophone stations.")
parser.add_argument("--station_spec", type=str, help="Station whose 3C spectra will be plotted.")
parser.add_argument("--time_window", type=str, help="Time window for the 3C spectra.")
parser.add_argument("--window_length", type=float, default=300.0, help="Window length in seconds for computing the STFT.")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between consecutive windows for computing the STFT.")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type=float, default=10.0, help="Maximum mean dB value for excluding noise windows.")

parser.add_argument("--color_geo", type=str, default="gold", help="Color of the geophone markers.")
parser.add_argument("--color_borehole", type=str, default="violet", help="Color of the borehole markers.")
parser.add_argument("--color_hydro", type=str, default="violet", help="Color of the hydrophone markers.")
parser.add_argument("--color_highlight", type=str, default="crimson", help="Color of the highlighted geophone markers.")

# Parse the command line arguments
args = parser.parse_args()

stations_highlight = loads(args.stations_highlight)
station_spec = args.station_spec
time_window = args.time_window
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

color_geo = args.color_geo
color_borehole = args.color_borehole
color_hydro = args.color_hydro
color_highlight = args.color_highlight

# Constants
fig_width = 15.0
fig_height = 12.0
spec_gap = 0.05

axis_offset = 0.1

min_depth = 0.0
max_depth = 450.0

hydro_min = -0.5
hydro_max = 1.5

freq_min = 0.0
freq_max = 200.0

min_db = -40.0
max_db = 55.0
min_arrow_db = 0.0

water_level = 15.0
water_amp = 2.5
water_period = 0.2

linewidth_marker = 1.0
linewidth_coast = 0.2
linewidth_water = 2.0
linewidth_spec = 1.0
linewidth_arrow = 1.0

station_size = 100.0
borehole_size = 100.0
hydro_size = 100.0

station_font_size = 12.0
station_label_x = 7.0
station_label_y = 7.0

borehole_font_size = 12.0
borehole_label_x = 40.0
borehole_label_y = -40.0

location_font_size = 12.0

water_font_size = 12.0

major_dist_spacing = 25.0
major_depth_spacing = 50.0
major_freq_spacing = 50.0
major_db_spacing = 20.0

axis_label_size = 12.0
tick_label_size = 12.0
title_size = 14.0
component_label_size = 14.0
freq_label_size = 12.0

legend_size = 12.0

major_tick_length = 5.0
minor_tick_length = 2.0
tick_width = 1.0

frame_width = 1.0

arrow_gap = 5.0
arrow_length = 10.0
arrow_width = 1.0
arrow_headwidth = 5.0
arrow_headlength = 5.0

subplot_label_size = 18.0
subplot_offset_x = -0.04
subplot_offset_y = 0.02

filename_image = "composite_local.tif"

# Load the geophone and borehole coordinates
geo_df = get_geophone_coords()
boho_df = get_borehole_coords()

# Load the satellite image
inpath = join(dir_img, filename_image)
with open(inpath) as src:
    # Read the image in RGB format
    rgb_band = src.read([1, 2, 3])

    # Reshape the image
    rgb_image = reshape_as_image(rgb_band)

    # Extract the extent of the image
    extent_img = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# Generate the figure and axes
# Compute the aspect ratio
east_range = max_east_array - min_east_array
north_range = max_north_array - min_north_array
aspect_ratio = north_range / east_range

fig, ax_sta = subplots(1, 1, figsize = (fig_width, fig_height))

### Plot the station map ###
# Plot the satellite image as the background
ax_sta.imshow(rgb_image, extent = extent_img, zorder = 0)

# Plot the geophone locations
for station, coords in geo_df.iterrows():
    east = coords["east"]
    north = coords["north"]

    if station in stations_highlight:
        ax_sta.scatter(east, north, marker = "^", s = station_size, color = color_geo, edgecolors = color_highlight, linewidths = linewidth_marker)
        ax_sta.annotate(station, (east, north), 
                        textcoords = "offset points", xytext = (station_label_x, station_label_y), ha = "left", va = "bottom", fontsize = station_font_size, 
                        color = color_highlight, fontweight = "bold", bbox = dict(facecolor = "white", edgecolor = "none", alpha = 0.5))
    else:
        ax_sta.scatter(east, north, marker = "^", s = station_size, color = color_geo, edgecolors = "black", linewidths = linewidth_marker, label = "Geophone")

# Plot the borehole locations
for borehole, coords in boho_df.iterrows():
    east = coords["east"]
    north = coords["north"]

    ax_sta.scatter(east, north, marker = "o", s = borehole_size, color = color_hydro, edgecolors = "black", linewidths = linewidth_marker, label = "Borehole/Hydrophones")
    ax_sta.annotate(borehole, (east, north), 
                    textcoords = "offset points", xytext = (borehole_label_x, borehole_label_y), ha = "left", va = "top", fontsize = borehole_font_size, fontweight = "bold", 
                    color = color_hydro, arrowprops=dict(arrowstyle = "-", color = "black"), bbox = dict(facecolor = "white", edgecolor = "none", alpha = 0.5))

# Plot the legend
handles, labels = ax_sta.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
legend = ax_sta.legend(unique_labels.values(), unique_labels.keys(), loc = "upper left", frameon = True, fancybox = False, fontsize = legend_size, bbox_to_anchor = (0.0, 1.0), bbox_transform = ax_sta.transAxes)
legend.get_frame().set_facecolor("white")


# Set the axis limits
ax_sta.set_xlim(min_east_array, max_east_array)
ax_sta.set_ylim(min_north_array, max_north_array)

ax_sta.set_aspect("equal")

# Set the axis ticks
format_east_xlabels(ax_sta, label = True, major_tick_spacing = major_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
format_north_ylabels(ax_sta, label = True, major_tick_spacing = major_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

# Set the axis labels
ax_sta.set_xlabel("East (m)")
ax_sta.set_ylabel("North (m)")

# # Adjust the frame width
# for spine in ax_sta.spines.values():
#     spine.set_linewidth(frame_width)

# Plot the large-scale map with coastlines
# Define the orthographic projection centered at the given longitude and latitude
ax_coast = fig.add_axes([0.25, 0.15, 0.2, 0.2], projection = Orthographic(central_longitude=lon, central_latitude=lat))
ax_coast.set_global()

# Add features
ax_coast.add_feature(cfeature.LAND, color='lightgray')
ax_coast.add_feature(cfeature.OCEAN, color='skyblue')

# Add coastlines
ax_coast.coastlines(linewidth = linewidth_coast)

# Plot a star at the given longitude and latitude
ax_coast.scatter(lon, lat, marker = '*', s = 100, color=color_hydro, edgecolor = "black", linewidths = linewidth_marker, transform = Geodetic(), zorder = 10)

# Add the subplot label
bbox_sta = ax_sta.get_position()
top_left_x = bbox_sta.x0
top_left_y = bbox_sta.y1
# print(f"Top left: ({top_left_x}, {top_left_y})")
fig.text(top_left_x + subplot_offset_x, top_left_y + subplot_offset_y, "(a)", fontsize = subplot_label_size, fontweight = "bold", va = "bottom", ha = "right")

### Plot the hydrophone depth profiles ###
# Add the axis 
bbox = ax_sta.get_position()
map_height = bbox.height
map_width = bbox.width
profile_height = map_height
profile_width = map_width / 3
ax_hydro = fig.add_axes([bbox.x1 + axis_offset, bbox.y0, profile_width, profile_height])

# Plot the hydrophones
for offset in [0, 1]:
    for location in depth_dict.keys():
        depth = depth_dict[location]

        if offset == 0 and location in ["01", "02"]:
            ax_hydro.scatter(offset, depth, marker = "o", color = "lightgray", edgecolors = "black", s = hydro_size, linewidths = linewidth_marker, label = "Broken")
        else:
            ax_hydro.scatter(offset, depth, marker = "o", color = color_hydro, edgecolors = "black", s = hydro_size, linewidths = linewidth_marker, label = "Functional")

        ax_hydro.text(0.5, depth, location, color = "black", fontsize = location_font_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")

ax_hydro.text(0.0, -15.0, "BA1A\n(A00)", color = "black", fontsize = location_font_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")
ax_hydro.text(1.0, -15.0, "BA1B\n(B00)", color = "black", fontsize = location_font_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "center")

max_hydro_depth = max(depth_dict.values())
ax_hydro.plot([0.0, 0.0], [min_depth, max_hydro_depth], color = "black", linewidth = linewidth_marker, zorder = 0)
ax_hydro.plot([1.0, 1.0], [min_depth, max_hydro_depth], color = "black", linewidth = linewidth_marker, zorder = 0)

# Plot the legend
handles, labels = ax_hydro.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax_hydro.legend(unique_labels.values(), unique_labels.keys(), loc = "lower left", frameon = False, fontsize = legend_size)

# Plot the water level
water_line_x = linspace(hydro_min, hydro_max, 100)
water_line_y = water_level + water_amp * cos(2 * pi * water_line_x / water_period)

ax_hydro.plot(water_line_x, water_line_y, color = "dodgerblue", linewidth = linewidth_water)
ax_hydro.text(-0.6, water_level, "Water table", color = "dodgerblue", fontsize = water_font_size, verticalalignment = "center", horizontalalignment = "right")

# Set the axis limits
ax_hydro.set_xlim(hydro_min, hydro_max)
ax_hydro.set_ylim(min_depth, max_depth)

ax_hydro.invert_yaxis()

# Set the axis ticks
ax_hydro.set_xticks([])
format_depth_ylabels(ax_hydro, label = True, major_tick_spacing = major_depth_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

# # Adjust the frame width
# for spine in ax_hydro.spines.values():
#     spine.set_linewidth(frame_width)

# Add the subplot label
bbox = ax_hydro.get_position()
top_left_x = bbox.x0
top_left_y = bbox.y1
fig.text(top_left_x + subplot_offset_x, top_left_y + subplot_offset_y, "(b)", fontsize = subplot_label_size, fontweight = "bold", va = "bottom", ha = "right")

### Plot the spectra of the 3C geophone station ###
# Read the spectra
print(f"Reading the 3C spectra of station {station_spec} for time window {time_window}...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

filename = f"whole_deployment_daily_geo_stft_{station_spec}_{suffix_spec}.h5"
filepath = join(dir_spec, filename)

psd_dict = read_time_slice_from_geo_stft(filepath, time_window)

# Read the resonance frequencies
print(f"Getting the resonance frequencies of station {station_spec}...")
filename = f"stationary_harmonic_series_PR02549_base2.csv"
filepath = join(dir_spec, filename)

suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

harmonic_df = read_csv(filepath)
mode_names = []
freqs_resonance = []
time_window = str2timestamp(time_window)
mode_marker_dfs = []
for mode_name in harmonic_df["mode_name"]:
    if mode_name.startswith("MH"):
        continue

    filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
    filepath = join(dir_spec, filename)

    resonance_df = read_hdf(filepath)
    mode_marker_df = resonance_df.loc[(resonance_df["station"] == station_spec) & (resonance_df["time"] == time_window)]
    mode_marker_df = mode_marker_df[["frequency", "power_1", "power_2", "power_z"]]
    mode_marker_df["harmonic_number"] = harmonic_df.loc[harmonic_df["mode_name"] == mode_name, "harmonic_number"].values[0]

    if mode_marker_df.empty:
        continue

    mode_marker_dfs.append(mode_marker_df)

mode_marker_df = concat(mode_marker_dfs, axis = 0)

# Compute the height of each spectrum panel
# print(f"Profile height: {profile_height}")
# print(f"Spec gap: {spec_gap}")
spec_width = map_width
spec_height = (profile_height - 2 * spec_gap) / 3
# print(f"Spec height: {spec_height}")
# print(f"Spec width: {spec_width}")

# Plot the spectra and the resonance frequencies
print(f"Plotting the 3C spectra of station {station_spec}...")
spec_axes = []
bbox = ax_hydro.get_position()
for i, component in enumerate(components):
    ax = fig.add_axes([bbox.x1 + axis_offset, 
                        bbox.y0 + i * (spec_height + spec_gap),
                        spec_width, spec_height])
    spec_axes.append(ax)

    spec = psd_dict[component]
    freqax = psd_dict["freqs"]

    color = get_geo_component_color(component)
    ax.plot(freqax, spec, color = color, linewidth = linewidth_spec, zorder = 2)

    # Plot the resonance frequencies
    for j, harmonic_num in enumerate(mode_marker_df["harmonic_number"].unique()):
        freq_resonance = mode_marker_df.loc[mode_marker_df["harmonic_number"] == harmonic_num, "frequency"].values[0]
        power = mode_marker_df.loc[mode_marker_df["harmonic_number"] == harmonic_num, f"power_{component.lower()}"].values[0]

        if power < min_arrow_db:
            continue

        if i == 1:
            if j == 0:
                ax.annotate(f"Mode {harmonic_num}", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                            color = color_highlight, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                            arrowprops=dict(facecolor=color_highlight, edgecolor=color_highlight, shrink=0.05, width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength))
            else:
                ax.annotate(f"{harmonic_num}", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                            color = color_highlight, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                            arrowprops=dict(facecolor=color_highlight, edgecolor=color_highlight, shrink=0.05, width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength))
        else:
            ax.annotate("", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                        arrowprops=dict(facecolor=color_highlight, edgecolor=color_highlight, shrink=0.05, width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength))
        
        # # Plot the resonance frequency labels
        # if i == 2:
        #     offset_x = freq_label_df.loc[freq_label_df["mode_name"] == mode_name, "offset_x"].values[0]
        #     offset_y = freq_label_df.loc[freq_label_df["mode_name"] == mode_name, "offset_y"].values[0]
        #     va = freq_label_df.loc[freq_label_df["mode_name"] == mode_name, "va"].values[0]
        #     ha = freq_label_df.loc[freq_label_df["mode_name"] == mode_name, "ha"].values[0]
        #     base = freq_label_df.loc[freq_label_df["mode_name"] == mode_name, "base"].values[0]
            
        #     if base == "min":
        #         db_base = min_db
        #     else:
        #         db_base = (max_db + min_db) / 2

        #     ax.annotate(mode_name, 
        #                 xy = (freq_resonance, db_base), xytext = (freq_resonance + offset_x, db_base + offset_y),
        #                 fontsize = freq_label_size, fontweight = "bold", va = va, ha = ha, color = color_highlight,
        #                 arrowprops = dict(arrowstyle = "-", color = color_highlight, width = 1))

                        
    # Set the axis limits
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(min_db, max_db)

    # Plot the component label
    label = component2label(component)
    ax.text(0.01, 0.97, label, fontsize = component_label_size, fontweight = "bold", verticalalignment = "top", horizontalalignment = "left", transform = ax.transAxes)

    # Set the x-axis labels
    if i == 0:
        format_freq_xlabels(ax, 
                            label = True, major_tick_spacing = major_freq_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
    else:
        format_freq_xlabels(ax,
                            label = False, major_tick_spacing = major_freq_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
    
    # Set the y-axis labels
    format_db_ylabels(ax, 
                      label = True, major_tick_spacing = major_db_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
    
    # Set the title
    if i == 2:
        starttime = time_window - Timedelta(window_length / 2, unit = "s")
        endtime = time_window + Timedelta(window_length / 2, unit = "s")
        ax.set_title(f"{station_spec}, {starttime:%Y-%m-%d %H:%M:%S} - {endtime:%Y-%m-%d %H:%M:%S}", fontsize = title_size, fontweight = "bold")

    # Add the subplot label
    if i == 2:
        bbox_top = ax.get_position()
        top_left_x = bbox_top.x0
        top_left_y = bbox_top.y1
        fig.text(top_left_x + subplot_offset_x, top_left_y + subplot_offset_y, "(c)", fontsize = subplot_label_size, fontweight = "bold", va = "bottom", ha = "right")

spec_axes[-1].set_xticklabels([])

# Save the figure
save_figure(fig, "liu_2025a_maps_n_specs.png")