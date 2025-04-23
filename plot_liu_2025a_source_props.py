# Plot the figure in Liu et al., 2025a showing the source properties, including the localization results and the inferred source dimensions

### Import the necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import amax, array, interp, nan, isnan, pi, deg2rad, linspace, histogram, concatenate, log10
from json import loads
from pandas import read_csv, DataFrame, read_hdf
from matplotlib.pyplot import figure
from rasterio import open
from rasterio.plot import reshape_as_image
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils_basic import SPECTROGRAM_DIR as dirname_spec, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, PHYS_DIR as dirname_phys, IMAGE_DIR as dirname_img
from utils_basic import GEO_COMPONENTS as components, MT_DIR as dirname_mt, LOC_DIR as dirname_loc, INNER_STATIONS as inner_stations, MIDDLE_STATIONS as middle_stations, OUTER_STATIONS as outer_stations, GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import GEO_COMPONENTS as components
from utils_basic import get_geophone_coords, get_borehole_coords, get_geophone_triads
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import add_colorbar, save_figure, format_east_xlabels, format_north_ylabels, plot_station_triads

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the source properties, including the localization results and the inferred source dimensions.")

parser.add_argument("--image_alpha", type=float, help="Opacity of the image", default=0.3)
parser.add_argument("--colormap_name_vel", type=str, help="Name of the colormap", default="plasma")
parser.add_argument("--colormap_name_depth", type=str, help="Name of the colormap", default="accent")
parser.add_argument("--color_observed", type=str, help="Color for the seismic observations", default="deepskyblue")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--base_name", type=str, help="Base name of the stationary harmonic series", default="PR02549")
parser.add_argument("--base_order", type=int, help="Order of the base harmonic", default=2)
parser.add_argument("--window_length_stft", type=float, help="Window length for computing the STFT in seconds", default=300.0)
parser.add_argument("--window_length_mt", type=float, help="Window length of the multitaper analysis", default=900.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence of the multitaper analysis", default=0.85)
parser.add_argument("--min_num_obs", type=int, help="Minimum number of observations for a station triad", default=100)

parser.add_argument("--scale_factor", type=float, help="Scale factor of the velocity vectors", default=20.0)
parser.add_argument("--min_vel_app", type=float, help="Minimum velocity of the velocity vectors", default=0.0)
parser.add_argument("--max_vel_app", type=float, help="Maximum velocity of the velocity vectors", default=4000.0)

parser.add_argument("--figwidth", type=float, help="Width of the figure", default=12.0)
parser.add_argument("--rose_a_x", type=float, help="X-coordinate of the rose diagram for Subarray A", default=0.6)
parser.add_argument("--rose_a_y", type=float, help="Y-coordinate of the rose diagram for Subarray A", default=0.5)
parser.add_argument("--rose_b_x", type=float, help="X-coordinate of the rose diagram for Subarray B", default=0.6)
parser.add_argument("--rose_b_y", type=float, help="Y-coordinate of the rose diagram for Subarray B", default=0.5)

parser.add_argument("--rose_width", type=float, help="Width of the rose diagram", default=0.25)
parser.add_argument("--rose_height", type=float, help="Height of the rose diagram", default=0.25)

parser.add_argument("--source_prop_x", type=float, help="X-coordinate of the source-property plots", default=0.07)
parser.add_argument("--source_prop_y", type=float, help="Y-coordinate of the source-property plots", default=0.03)
parser.add_argument("--source_prop_width", type=float, help="Width of each source-property plots", default=0.25)
parser.add_argument("--source_prop_height", type=float, help="Height of each source-property plots", default=0.25)
parser.add_argument("--source_prop_hspace", type=float, help="Horizontal space between the source-property plots", default=0.02)
parser.add_argument("--source_depths", type=str, help="Depths of the source-property plots", default="[15.0,150.0]")
parser.add_argument("--qf_label_offsets", type=str, help="X-coordinate offset of the quality factor labels", default="[5.0,-5.0,5.0]")
parser.add_argument("--qf_label_rotation", type=float, help="Rotation of the quality factor labels", default=0.0)

parser.add_argument("--fontsize_label_rose", type=float, help="Fontsize of the labels of the rose diagram", default=12)
parser.add_argument("--fontsize_label_phys", type=float, help="Fontsize of the labels of the physical properties", default=12)
parser.add_argument("--fontsize_title", type=float, help="Fontsize of the titles", default=15)

parser.add_argument("--linewidth_source_prop", type=float, help="Linewidth of the source properties", default=3.0)
parser.add_argument("--label_size_source_prop", type=float, help="Label size of the source properties", default=12)
parser.add_argument("--legend_size", type=float, help="Legend size", default=12)

parser.add_argument("--linewidth_triad", type=float, help="Linewidth of the station triads", default=1.0)
parser.add_argument("--quiver_width", type=float, help="Width of the velocity vectors", default=0.003)
parser.add_argument("--quiver_head_width", type=float, help="Width of the head of the velocity vectors", default=6.0)
parser.add_argument("--quiver_head_length", type=float, help="Length of the head of the velocity vectors", default=7.0)
parser.add_argument("--quiver_linewidth", type=float, help="Linewidth of the velocity vectors", default=0.5)

parser.add_argument("--min_log_qf", type=float, help="Minimum quality factor to plot", default=1.0)
parser.add_argument("--max_log_qf", type=float, help="Maximum quality factor to plot", default=5.0)

parser.add_argument("--alpha_shade", type=float, help="Opacity of the shaded area", default=0.2)

parser.add_argument("--panel_label_size", type=float, help="Size of the panel labels", default=14)
parser.add_argument("--panel_label1_offset_x", type=float, help="X-coordinate offset of the panel label 1", default=-0.04)
parser.add_argument("--panel_label1_offset_y", type=float, help="Y-coordinate offset of the panel label 1", default=0.00)
parser.add_argument("--panel_label2_offset_x", type=float, help="X-coordinate offset of the panel label 2", default=-0.04)
parser.add_argument("--panel_label2_offset_y", type=float, help="Y-coordinate offset of the panel label 2", default=0.1)

parser.add_argument("--qf_obs_anno_x", type=float, help="X-coordinate of the quality factor observation annotation", default=0.4)
parser.add_argument("--qf_obs_anno_y", type=float, help="Y-coordinate of the quality factor observation annotation", default=3.5)
parser.add_argument("--qf_obs_anno_text_x", type=float, help="X-coordinate offset of the quality factor observation annotation text", default=0.5)
parser.add_argument("--qf_obs_anno_text_y", type=float, help="Y-coordinate offset of the quality factor observation annotation text", default=2.0)

parser.add_argument("--qf_sat_label_x", type=float, help="X-coordinate of the quality factor saturation label", default=0.5)
parser.add_argument("--qf_sat_label_y", type=float, help="Y-coordinate of the quality factor saturation label", default=4.0)

# Parse the arguments
args = parser.parse_args()

colormap_name_vel = args.colormap_name_vel
colormap_name_depth = args.colormap_name_depth
image_alpha = args.image_alpha
color_observed = args.color_observed

mode_name = args.mode_name
base_name = args.base_name
base_order = args.base_order
window_length_stft = args.window_length_stft
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe
min_num_obs = args.min_num_obs

scale_factor = args.scale_factor

min_vel_app = args.min_vel_app
max_vel_app = args.max_vel_app

figwidth = args.figwidth

rose_a_x = args.rose_a_x
rose_a_y = args.rose_a_y
rose_b_x = args.rose_b_x
rose_b_y = args.rose_b_y
rose_width = args.rose_width
rose_height = args.rose_height

source_prop_x = args.source_prop_x
source_prop_y = args.source_prop_y
source_prop_width = args.source_prop_width
source_prop_height = args.source_prop_height
source_prop_hspace = args.source_prop_hspace
depths_to_plot = loads(args.source_depths)

qf_label_offsets = loads(args.qf_label_offsets)
qf_label_rotation = args.qf_label_rotation

fontsize_label_rose = args.fontsize_label_rose
fontsize_label_phys = args.fontsize_label_phys
fontsize_title = args.fontsize_title

linewidth_triad = args.linewidth_triad
linewidth_source_prop = args.linewidth_source_prop
label_size_source_prop = args.label_size_source_prop
legend_size = args.legend_size

quiver_width = args.quiver_width
quiver_head_width = args.quiver_head_width
quiver_head_length = args.quiver_head_length
quiver_linewidth = args.quiver_linewidth

alpha_shade = args.alpha_shade

min_log_qf = args.min_log_qf
max_log_qf = args.max_log_qf

panel_label_size = args.panel_label_size
panel_label1_offset_x = args.panel_label1_offset_x
panel_label1_offset_y = args.panel_label1_offset_y
panel_label2_offset_x = args.panel_label2_offset_x
panel_label2_offset_y = args.panel_label2_offset_y

qf_obs_anno_x = args.qf_obs_anno_x
qf_obs_anno_y = args.qf_obs_anno_y
qf_obs_anno_text_x = args.qf_obs_anno_text_x
qf_obs_anno_text_y = args.qf_obs_anno_text_y

qf_sat_label_x = args.qf_sat_label_x
qf_sat_label_y = args.qf_sat_label_y



# Constants
hspace = 0.05
margin_x = 0.03
margin_y = 0.03

base_mode_name = "PR02549"
base_mode_order = 2

cbar_x = 0.05
cbar_y = 0.7
cbar_width = 0.02
cbar_height = 0.25

count_ticks = [4, 8]

freq_base = 12.0

filename_image = "maxar_2019-09-17_local.tif"

label_offset_x = 0.01
label_offset_y = 0.00

axis_label_size = 12
tick_label_size = 10

dim_label_offset_x = 0.0
dim_label_offset_y = 5.0

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

# Load the geophone triads
triad_df = get_geophone_triads()

# Apparent velocities
filename = f"stationary_resonance_station_triad_avg_app_vels_{mode_name}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}_min_num_obs{min_num_obs:d}.csv"
filepath = join(dirname_loc, filename)
vel_df = read_csv(filepath)

for component in components:
    vel_df[f"vel_app_cov_mat_{component.lower()}"] = vel_df[f"vel_app_cov_mat_{component.lower()}"].apply(lambda x: array(loads(x)))

# Filter the station triads
stations_to_plot = inner_stations + middle_stations + outer_stations
triad_df = triad_df[triad_df["station1"].isin(stations_to_plot) & triad_df["station2"].isin(stations_to_plot) & triad_df["station3"].isin(stations_to_plot)]
vel_df = vel_df[vel_df["station1"].isin(stations_to_plot) & vel_df["station2"].isin(stations_to_plot) & vel_df["station3"].isin(stations_to_plot)]

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
ax_loc = fig.add_axes([margin_x, margin_y, 1 - 2 * margin_x, 1 - 2 * margin_y])

# Define the colormap for the velocities
norm = Normalize(vmin = min_vel_app, vmax = max_vel_app)
cmap = colormaps[colormap_name_vel]

# Plot the satellite image
ax_loc.imshow(rgb_image, extent = extent_img, alpha = image_alpha)

# Plot the station triads and vectors
back_azis_to_plot_dicts = []
for _, row in vel_df.iterrows():
    # Find the center of the station triad
    station1, station2, station3 = row["station1"], row["station2"], row["station3"]
    east_center = triad_df[(triad_df["station1"] == station1) & (triad_df["station2"] == station2) & (triad_df["station3"] == station3)]["east"].values[0]
    north_center = triad_df[(triad_df["station1"] == station1) & (triad_df["station2"] == station2) & (triad_df["station3"] == station3)]["north"].values[0]

    print(f"Plotting the vectors for the station triad {station1}-{station2}-{station3}.")

    # Find the component with the smallest apparent velocity variance
    flag = False
    for i, component in enumerate(components):
        vel_app_east = row[f"avg_vel_app_east_{component.lower()}"]
        vel_app_north = row[f"avg_vel_app_north_{component.lower()}"]
        vel_app = row[f"avg_vel_app_{component.lower()}"]
        back_azi = row[f"avg_back_azi_{component.lower()}"]
        cov_mat = row[f"vel_app_cov_mat_{component.lower()}"]
        
        if isnan(vel_app_east):
            continue
        
        # print(f"The component is {component}.")
        # print(f"The east component of the average apparent velocity is {vel_app_east:.0f} m/s.")
        # print(f"The north component of the average apparent velocity is {vel_app_north:.0f} m/s.")
        # print(f"The average apparent velocity is {vel_app:.0f} m/s.")
        # print(f"The back azimuth is {back_azi:.0f} degrees.")
        
        # Calculate apparent velocity variance from covariance matrix
        app_vel_var = cov_mat[0,0] + cov_mat[1,1]

        if not flag:
            min_app_vel_var = app_vel_var
            vec_east = vel_app_east / vel_app
            vec_north = vel_app_north / vel_app
            vec_amp = vel_app
            back_azi_plot = back_azi
            component_plot = component
            flag = True
        else:
            if app_vel_var < min_app_vel_var:
                min_app_vel_var = app_vel_var
                vec_east = vel_app_east / vel_app
                vec_north = vel_app_north / vel_app
                vec_amp = vel_app
                back_azi_plot = back_azi
                component_plot = component

    if not flag:
        print(f"No valid vector found for the station triad {station1}-{station2}-{station3}. The station triad is skipped.")
        continue
    
    back_azis_to_plot_dicts.append({"station1": station1, "station2": station2, "station3": station3, "back_azi": back_azi_plot})

    quiver = ax_loc.quiver(east_center, north_center, vec_east, vec_north, vec_amp, 
                    cmap=cmap, norm=norm,
                    scale=scale_factor, width=quiver_width,
                    headwidth=quiver_head_width, headlength=quiver_head_length,
                    zorder=3)
    
# Plot the station triads
back_azis_to_plot_df = DataFrame(back_azis_to_plot_dicts)
plot_station_triads(ax_loc, linewidth = linewidth_triad, triads_to_plot = back_azis_to_plot_df)

# Add the axes for the rose diagram 
ax_rose_a = ax_loc.inset_axes([rose_a_x, rose_a_y, rose_width, rose_height], projection="polar")
ax_rose_b = ax_loc.inset_axes([rose_b_x, rose_b_y, rose_width, rose_height], projection="polar")

# Plot the rose diagram
# Divide the back azimuths by subarray
back_azis_a = back_azis_to_plot_df[back_azis_to_plot_df["station1"].isin(stations_a) & back_azis_to_plot_df["station2"].isin(stations_a) & back_azis_to_plot_df["station3"].isin(stations_a)]["back_azi"].values
back_azis_b = back_azis_to_plot_df[back_azis_to_plot_df["station1"].isin(stations_b) & back_azis_to_plot_df["station2"].isin(stations_b) & back_azis_to_plot_df["station3"].isin(stations_b)]["back_azi"].values

# Convert to radians
angles_a = array(back_azis_a)
angles_b = array(back_azis_b)

angles_a = deg2rad(angles_a)
angles_b = deg2rad(angles_b)

# Define bins (e.g., 10-degree intervals)
num_bins = 24  # 10-degree bins
bins = linspace(-pi, pi, num_bins + 1)
centers = (bins[:-1] + bins[1:]) / 2

# Compute histogram
counts_a, _ = histogram(angles_a, bins=bins)
counts_b, _ = histogram(angles_b, bins=bins)

# Plot the rose diagram in the inset
ax_rose_a.bar(centers, counts_a, width=(2 * pi / num_bins), align='center', color=color_observed, edgecolor='k')
ax_rose_b.bar(centers, counts_b, width=(2 * pi / num_bins), align='center', color=color_observed, edgecolor='k')
ax_rose_a.grid(True, linestyle=":")
ax_rose_b.grid(True, linestyle=":")

ax_rose_a.set_theta_zero_location("N")  # North at the top
ax_rose_a.set_theta_direction(-1)  # Clockwise

ax_rose_b.set_theta_zero_location("N")  # North at the top
ax_rose_b.set_theta_direction(-1)  # Clockwise

ax_rose_a.set_thetagrids([0, 90, 180, 270], labels=["N", "E", "S", "W"], fontsize = fontsize_label_rose, fontweight = "bold")
ax_rose_b.set_thetagrids([0, 90, 180, 270], labels=["N", "E", "S", "W"], fontsize = fontsize_label_rose, fontweight = "bold")

ax_rose_a.set_yticks(count_ticks)
ax_rose_a.set_yticklabels(count_ticks, ha = "right", va = "center")
ax_rose_a.set_rlabel_position(90)

ax_rose_b.set_yticks(count_ticks)
ax_rose_b.set_yticklabels(count_ticks, ha = "right", va = "center")
ax_rose_b.set_rlabel_position(90)

ax_rose_a.set_title("Subarray A", fontsize = fontsize_title, fontweight = "bold")
ax_rose_b.set_title("Subarray B", fontsize = fontsize_title, fontweight = "bold")

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
ax_loc.set_title(f"Mode {mode_order:d}, horizontal apparent velocities and propagation directions", fontsize = fontsize_title, fontweight = "bold")

# Add the panel label
ax_loc.text(panel_label1_offset_x, 1.0 + panel_label1_offset_y, "(a)", ha = "right", va = "bottom", transform = ax_loc.transAxes, fontsize = panel_label_size, fontweight = "bold")

### Plot the inferred source dimensions and quality factors ###

# Read the quality factor histogram
filename = f"stationary_resonance_qf_log_histogram_{mode_name}.csv"
filepath = join(dirname_spec, filename)
qf_df = read_csv(filepath)

# Extract the the quality factors
bin_centers = qf_df["log_quality_factor"].values
counts = qf_df["normalized_count"].values

# Plot the inferred source dimensions and quality factors
ax_dim = ax_loc.inset_axes([source_prop_x, source_prop_y, source_prop_width, source_prop_height])
ax_qf_curve = ax_loc.inset_axes([source_prop_x, source_prop_y + source_prop_height + source_prop_hspace, source_prop_width, source_prop_height])

num_depth = len(depths_to_plot)
colors = colormaps[colormap_name_depth]
for i, depth in enumerate(depths_to_plot):
    color = colors(i)

    # Load the data
    filename = f"source_properties_hydrogen_depth{depth:.0f}m.csv"
    filepath = join(dirname_phys, filename)

    data_df = read_csv(filepath)

    # Plot the dimensions
    ax_dim.plot(data_df["gas_fraction"], data_df["fracture_dimension"], 
                linewidth=linewidth_source_prop, color=color,
                label=f"{depth:.0f} m") 
    
    # Plot the quality factors
    ax_qf_curve.plot(data_df["gas_fraction"], data_df["quality_factor"], 
                linewidth=linewidth_source_prop, color=color)
            
# Adjust the y-axis limits
min_qf_plot = 10 ** min_log_qf
max_qf_plot = 10 ** max_log_qf
ax_qf_curve.set_ylim(min_qf_plot, max_qf_plot)

# Load the maximum resolvable quality factor
filename = f"stationary_harmonic_series_max_qf_{base_name}_base{base_order}_window{window_length_stft:.0f}s.csv"
filepath = join(dirname_spec, filename)
max_qf_df = read_csv(filepath)

max_qf_resolve = max_qf_df.loc[max_qf_df["mode_name"] == mode_name, "max_qf"].values[0]

# Shade the quality-factor-saturation zone
patch = Rectangle((0, max_qf_resolve), 1, max_qf_plot - max_qf_resolve, fill = True, color ="gray", alpha = alpha_shade)
ax_qf_curve.add_patch(patch)

# Add the axis for plotting the histogram of the quality factors
ax_qf_hist = ax_qf_curve.inset_axes([0, 0, 1, 1])
ax_qf_hist.fill_betweenx(bin_centers, 0, counts, color = color_observed, alpha = 0.2)

# Add the annotation for the observed quality factor
ax_qf_hist.annotate("Observations", xy = (qf_obs_anno_x, qf_obs_anno_y), xytext = (qf_obs_anno_text_x, qf_obs_anno_text_y),
                    arrowprops = dict(arrowstyle = "-"),
                     fontsize = fontsize_label_phys, fontweight = "bold", transform = ax_qf_hist.transData)

# Add the annotation for the quality-factor-saturation zone
ax_qf_hist.text(qf_sat_label_x, qf_sat_label_y, "Observation saturated", ha = "center", va = "bottom", fontsize = fontsize_label_phys, fontweight = "bold")

# Turn off all spines and ticks for the histogram axis
ax_qf_hist.spines['top'].set_visible(False)
ax_qf_hist.spines['right'].set_visible(False) 
ax_qf_hist.spines['bottom'].set_visible(False)
ax_qf_hist.spines['left'].set_visible(False)
ax_qf_hist.tick_params(axis='both', which='both', length=0)
ax_qf_hist.set_xticks([])
ax_qf_hist.set_yticks([])

# Set background color to none
ax_qf_hist.set_facecolor('none')

# Set the axis limits
ax_qf_hist.set_xlim(0, 1)
ax_qf_hist.set_ylim(min_log_qf, max_log_qf)

# ax_qf.add_patch(Rectangle((0, min_qf), 1, max_qf - min_qf, fill = True, color = color_observed, alpha = 0.2))
# ax_qf.text(0.5, max_qf * 1.5, "Observed range", ha = "center", va = "bottom", fontsize = fontsize_label_phys, fontweight = "bold", color = color_observed)

# Set the axis limits
ax_dim.set_xlim(0, 1)
ax_qf_curve.set_xlim(0, 1)
ax_qf_hist.set_xlim(0, 1)

# Set the y-axis scale to logarithmic
ax_dim.set_yscale('log')
ax_qf_curve.set_yscale('log')

# Turn on the grid
ax_dim.grid(True, linestyle='--', alpha=0.5)
ax_dim.grid(True, which='minor', linestyle=':', alpha=0.5)
ax_qf_curve.grid(True, linestyle='--', alpha=0.5)
ax_qf_curve.grid(True, which='minor', linestyle=':', alpha=0.5)

# Set the axis labels
ax_dim.set_ylabel("Dimension (m)", fontsize = axis_label_size)
ax_qf_curve.set_ylabel("Quality factor", fontsize = axis_label_size)

# Set the x-axis label
ax_dim.set_xlabel("Gas volume fraction", fontsize = axis_label_size)

# Set the title
ax_dim.set_title(f"Inferred fracture dimension", fontsize = fontsize_title, fontweight = "bold")
ax_qf_curve.set_title(f"Observed & modeled quality factors", fontsize = fontsize_title, fontweight = "bold")

# Plot the legend
ax_dim.legend(fontsize = legend_size, loc = "upper left", title="Burial depth", title_fontsize = legend_size)

# Add the panel labels
ax_qf_curve.text(panel_label2_offset_x, 1.0 + panel_label2_offset_y, "(b)", ha = "right", va = "bottom", transform = ax_qf_curve.transAxes, fontsize = panel_label_size, fontweight = "bold")

# # Add the legends
# legend = ax_dim.legend(fontsize=legend_size, loc="upper center", ncol = len(depths_to_plot), title="Burial depth")

### Save the figure ###
print("Saving the figure...")
filename = f"liu_2025a_source_props.png"
save_figure(fig, filename)
