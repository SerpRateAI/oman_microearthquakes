# Plot the figure in Liu et al., 2025a showing the source properties, including the localization results and the inferred source dimensions

### Import the necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import interp, nan, sin, cos, pi, deg2rad, linspace, histogram
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure
from rasterio import open
from rasterio.plot import reshape_as_image
from matplotlib.colors import Normalize
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils_basic import SPECTROGRAM_DIR as dirname_spec, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, PHYS_DIR as dirname_phys, IMAGE_DIR as dirname_img
from utils_basic import GEO_COMPONENTS as components, MT_DIR as dirname_mt, INNER_STATIONS as inner_stations, MIDDLE_STATIONS as middle_stations, OUTER_STATIONS as outer_stations
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_borehole_coords
from utils_plot import add_colorbar, save_figure, format_east_xlabels, format_north_ylabels
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the source properties, including the localization results and the inferred source dimensions.")

parser.add_argument("--max_back_azi_std", type=float, help="Maximum standard deviation of the back azimuth", default=15.0)

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--window_length_mt", type=float, help="Window length of the multitaper analysis", default=900.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence of the multitaper analysis", default=0.85)
parser.add_argument("--min_num_obs", type=int, help="Minimum number of observations for a station triad", default=100)

# Parse the arguments
args = parser.parse_args()
max_back_azi_std = args.max_back_azi_std

mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
min_num_obs = args.min_num_obs

# Constants
frac_dim = 0.2 # Height fraction of the source-dimension plot
figwidth = 12.0

hspace = 0.05
margin_x = 0.03
margin_y = 0.03

base_mode_name = "PR02549"
base_mode_order = 2

linewidth_triad = 1.0

scale_factor = 10.0

min_vel_app = 0.0
max_vel_app = 3000.0

quiver_width = 0.003
quiver_head_width = 6.0
quiver_head_length = 7.0
quiver_linewidth = 0.5

cbar_x = 0.05
cbar_y = 0.7
cbar_width = 0.02
cbar_height = 0.25

rose_x = 0.7
rose_y = 0.4
rose_width = 0.25
rose_height = 0.25

dim_x = 0.08
dim_y = 0.07
dim_width = 0.5
dim_height = 0.5

count_ticks = [4, 8]

depths_to_plot = [0.0, 20.0, 200.0]
freq_base = 12.0

filename_image = "maxar_2019-09-17_local.tif"

# Starting indices for the colors of the velocities and dimensions in the colormap tab20c
linewidth_dim = 1.5

bottom_harmonic = 0.67
top_corr = 0.66
bottom_corr = 0.43
top_phys = 0.42

db_range = 30.0

scalebar_x = 0.02
scalebar_y = 0.95
scalebar_length = 0.2

label_offset_x = 0.01
label_offset_y = 0.00

axis_label_size = 12
tick_label_size = 10

panel_label_size = 14

panel_label1_offset_x = -0.04
panel_label1_offset_y = 0.00

panel_label2_offset_x = -0.04
panel_label2_offset_y = 0.00

dim_label_offset_x = 0.0
dim_label_offset_y = 5.0

title_size = 14

linewidth_phys = 1.5
label_size_phys = 12

legend_size = 12

### Compute the figure size and generate the figure ###
print("Computing the figure size...")
aspect_ratio_map = (max_north - min_north) / (max_east - min_east)
aspect_ratio_fig = aspect_ratio_map * (1 - 2 * margin_x) / (1 - 2 * margin_y)
figheight = figwidth * aspect_ratio_fig

print("Generating the figure...")
fig = figure(figsize=(figwidth, figheight))

### Plot the average apparent velocities of all station triads ###
# Load the harmonic data
filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order:d}.csv"
filepath = join(dirname_spec, filename)
harmonic_df = read_csv(filepath)

mode_order = harmonic_df.loc[harmonic_df["mode_name"] == mode_name, "mode_order"].values[0]

# Load the geophone and borehole coordinates
geo_df = get_geophone_coords()
boho_df = get_borehole_coords()

# Station triads
filename = f"delaunay_station_triads.csv"
filepath = join(dirname_mt, filename)
triad_df = read_csv(filepath)

# Apparent velocities
filename = f"station_triad_avg_app_vels_{mode_name}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}_min_num_obs{min_num_obs:d}.csv"
filepath = join(dirname_mt, filename)
vel_df = read_csv(filepath)

# Assemble the apparent-velocity vectors
stations_to_plot = inner_stations + middle_stations + outer_stations
vec_dicts = []
for _, row in vel_df.iterrows():
    station1, station2, station3 = row["station1"], row["station2"], row["station3"]

    # Check if the stations are in the list of stations to plot
    if (station1 not in stations_to_plot) or (station2 not in stations_to_plot) or (station3 not in stations_to_plot):
        continue

    # Find the center coordinate of the triad
    triad_rows = triad_df.loc[(triad_df["station1"] == station1) & (triad_df["station2"] == station2) & (triad_df["station3"] == station3)]
    east_triad = triad_rows["east"].values[0]
    north_triad = triad_rows["north"].values[0]

    # Find the coordinates of the stations
    east1 = geo_df.loc[station1, "east"]
    north1 = geo_df.loc[station1, "north"]
    east2 = geo_df.loc[station2, "east"]
    north2 = geo_df.loc[station2, "north"]
    east3 = geo_df.loc[station3, "east"]
    north3 = geo_df.loc[station3, "north"]

            
    # Check if the back azimuth standard deviation is within the maximum and keep only the one with the smaller standard deviation
    back_azi_std = max_back_azi_std
    for component in components:
        back_azi_std_comp = row[f"std_back_azi_{component.lower()}"]

        if back_azi_std_comp < back_azi_std:
            back_azi_std = back_azi_std_comp
            avg_vel_app = row[f"avg_vel_app_{component.lower()}"]
            avg_back_azi = row[f"avg_back_azi_{component.lower()}"]

            vec_east = sin(avg_back_azi / 180 * pi)
            vec_north = cos(avg_back_azi / 180 * pi)
            avg_vel_app = avg_vel_app
        
    if back_azi_std < max_back_azi_std:
        vec_dict = {"station1": station1, "station2": station2, "station3": station3, 
                    "east_center": east_triad, "north_center": north_triad,
                    "vec_east": vec_east, "vec_north": vec_north,
                    "vel_app": avg_vel_app, "back_azi": avg_back_azi}

        vec_dicts.append(vec_dict)

vec_df = DataFrame(vec_dicts)

# Load the satellite image
inpath = join(dirname_img, filename_image)
with open(inpath) as src:
    # Read the image in RGB format
    rgb_band = src.read([1, 2, 3])

    # Reshape the image
    rgb_image = reshape_as_image(rgb_band)

    # Extract the extent of the image
    extent_img = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# Create the subplot
ax_loc = fig.add_axes([margin_x, margin_y + frac_dim + hspace, 1 - 2 * margin_x, 1 - 2 * margin_y - frac_dim - hspace])

# Define the colormap for the velocities
norm = Normalize(vmin = min_vel_app, vmax = max_vel_app)
cmap = colormaps["plasma"]

# Plot the satellite image
ax_loc.imshow(rgb_image, extent = extent_img, alpha = 0.5)

# Plot the station triads and vectors
edges_plotted = set()
stations_to_plot = set()
for _, row in vec_df.iterrows():
    # Plot the station triad
    edges = set([tuple(sorted([row["station1"], row["station2"]])), tuple(sorted([row["station2"], row["station3"]])), tuple(sorted([row["station3"], row["station1"]]))])
    stations = set([row["station1"], row["station2"], row["station3"]])
    for edge in edges:
        if edge not in edges_plotted:
            station1, station2 = edge
            east1, north1 = geo_df.loc[station1, "east"], geo_df.loc[station1, "north"]
            east2, north2 = geo_df.loc[station2, "east"], geo_df.loc[station2, "north"]
            ax_loc.plot([east1, east2], [north1, north2], color="gray", linewidth=linewidth_triad, zorder=1)

            edges_plotted.add(edge)
            stations_to_plot.update(stations)

    # Plot the vector
    east_center = row["east_center"]
    north_center = row["north_center"]
    vec_east = row["vec_east"]
    vec_north = row["vec_north"]
    vel_app = row["vel_app"]


    quiver = ax_loc.quiver(east_center, north_center, vec_east, vec_north, vel_app, 
                            cmap=cmap, norm=norm,
                            scale=scale_factor, width=quiver_width,
                            headwidth=quiver_head_width, headlength=quiver_head_length,
                            zorder=3)

# Add the axes for the rose diagram 
ax_rose = ax_loc.inset_axes([rose_x, rose_y, rose_width, rose_height], projection="polar")

# Plot the rose diagram
# Convert to radians
angles = vec_df["back_azi"].values
angles = deg2rad(angles)

# Define bins (e.g., 10-degree intervals)
num_bins = 24  # 10-degree bins
bins = linspace(-pi, pi, num_bins + 1)
centers = (bins[:-1] + bins[1:]) / 2

# Compute histogram
counts, _ = histogram(angles, bins=bins)

# Plot the rose diagram in the inset
ax_rose.bar(centers, counts, width=(2 * pi / num_bins), align='center', color="orange", edgecolor='k')
ax_rose.grid(True, linestyle=":")

ax_rose.set_theta_zero_location("N")  # North at the top
ax_rose.set_theta_direction(-1)  # Clockwise

ax_rose.set_xticklabels([])

ax_rose.set_yticks(count_ticks)
ax_rose.set_yticklabels(count_ticks, ha = "right", va = "center")
ax_rose.set_rlabel_position(90)

# Add a colorbar
cax = ax_loc.inset_axes([cbar_x, cbar_y, cbar_width, cbar_height])
cbar = add_colorbar(fig, cax, "Velocity (m s$^{-1}$)",
                    cmap=cmap, norm=norm)

# Set the x and y limits
ax_loc.set_xlim(min_east, max_east)
ax_loc.set_ylim(min_north, max_north)

# Set the x and y labels
format_east_xlabels(ax_loc)
format_north_ylabels(ax_loc)

# Set the aspect ratio
ax_loc.set_aspect("equal")

# Add the title
ax_loc.set_title(f"Mode {mode_order:d}, horizontal apparent velocities and back azimuths", fontsize = title_size, fontweight = "bold")

# Add the panel label
ax_loc.text(panel_label1_offset_x, 1.0 + panel_label1_offset_y, "(a)", ha = "right", va = "bottom", transform = ax_loc.transAxes, fontsize = panel_label_size, fontweight = "bold")
### Plot the inferred source dimensions as an inset ###
ax_dim = ax_loc.inset_axes([dim_x, dim_y, dim_width, dim_height])

num_depth = len(depths_to_plot)
for i, depth in enumerate(depths_to_plot):
    for gas in ["hydrogen", "methane"]:
        if gas == "hydrogen":
            color = "tab:orange"
        else:
            color = "gray"

        # Load the data
        filename = f"source_dimensions_{gas}_{depth:.0f}m.csv"
        filepath = join(dirname_phys, filename)

        data_df = read_csv(filepath)

        # Plot the dimensions
        ax_dim.plot(data_df["gas_frac"], data_df["dimension"], 
                    linewidth=linewidth_phys, color=color,
                    label = gas)


        # Plot the depth label for the dimension
        dim_label = interp(0.5, data_df["gas_frac"], data_df["dimension"])
        if gas == "hydrogen":
            if i == num_depth - 1:
                ax_dim.annotate(f"Burial depth\n= {depth:.0f} m", (0.5, dim_label), ha = "center", va = "bottom",
                                xytext = (dim_label_offset_x, dim_label_offset_y), textcoords = "offset points",
                                fontsize = label_size_phys, fontweight = "bold", color = color)
            else:
                ax_dim.annotate(f"{depth:.0f} m", (0.5, dim_label), ha = "center", va = "bottom",
                                xytext = (dim_label_offset_x, dim_label_offset_y), textcoords = "offset points",
                                fontsize = label_size_phys, fontweight = "bold", color = color)

# Set the axis limits
ax_dim.set_xlim(0, 1)

# Set the y-axis scale to logarithmic
ax_dim.set_yscale('log')

# Turn on the grid
ax_dim.grid(True, linestyle='--', alpha=0.5)
ax_dim.grid(True, which='minor', linestyle=':', alpha=0.5)

# Set the axis labels
ax_dim.set_ylabel("Dimension (m)", fontsize = axis_label_size)

# Set the x-axis label
ax_dim.set_xlabel("Gas volume fraction", fontsize = axis_label_size)

# Set the title
ax_dim.set_title(f"Inferred fracture dimension", fontsize = title_size, fontweight = "bold")

# Plot the legend
handles, labels = ax_dim.get_legend_handles_labels()
labels = [label.capitalize() for label in labels]

# Keep only unique legend entries
unique_labels = dict(zip(labels, handles))

ax_dim.legend(unique_labels.values(), unique_labels.keys(), fontsize = legend_size, loc = "upper left")

# Add the panel labels
ax_dim.text(panel_label2_offset_x, 1.0 + panel_label2_offset_y, "(b)", ha = "right", va = "bottom", transform = ax_dim.transAxes, fontsize = panel_label_size, fontweight = "bold")

# # Add the legends
# legend = ax_dim.legend(fontsize=legend_size, loc="upper center", ncol = len(depths_to_plot), title="Burial depth")

### Save the figure ###
print("Saving the figure...")
filename = f"liu_2025a_source_props.png"
save_figure(fig, filename)
