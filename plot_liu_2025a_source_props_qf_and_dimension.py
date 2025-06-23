# Plot the quality-factor and fracture dimension subplots in the source properties figure in Liu et al., 2025a

### Import the necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import amax, array, interp, nan, isnan, pi, deg2rad, linspace, histogram, concatenate, log10
from json import loads
from pandas import read_csv, DataFrame, read_hdf
from matplotlib.pyplot import figure, subplots
from rasterio import open
from rasterio.plot import reshape_as_image
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils_basic import SPECTROGRAM_DIR as dirname_spec, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, PHYS_DIR as dirname_phys, IMAGE_DIR as dirname_img
from utils_basic import GEO_COMPONENTS as components, MT_DIR as dirname_mt, LOC_DIR as dirname_loc, INNER_STATIONS as inner_stations, MIDDLE_STATIONS as middle_stations, OUTER_STATIONS as outer_stations, GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import GEO_COMPONENTS as components, CORE_STATIONS as stations_to_plot
from utils_basic import get_geophone_coords, get_borehole_coords, get_geophone_triads
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_satellite import load_maxar_image        
from utils_plot import add_colorbar, save_figure, format_east_xlabels, format_north_ylabels, plot_station_triads

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Plot the quality-factor and fracture dimension subplots in the source properties figure in Liu et al., 2025a")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--base_name", type=str, help="Base name of the stationary harmonic series", default="PR02549")
parser.add_argument("--base_order", type=int, help="Order of the base harmonic", default=2)
parser.add_argument("--window_length_stft", type=float, help="Window length for computing the STFT in seconds", default=300.0)

parser.add_argument("--colormap_name_depth", type=str, help="Name of the colormap", default="Accent")
parser.add_argument("--color_observed", type=str, help="Color for the seismic observations", default="orange")

parser.add_argument("--figwidth", type=float, help="Width of the figure", default=9.0)
parser.add_argument("--figheight", type=float, help="Height of the figure", default=6.0)

parser.add_argument("--source_depths", type=str, help="Depths of the source-property plots", default="[15.0,150.0]")

parser.add_argument("--fontsize_label_phys", type=float, help="Fontsize of the labels of the physical properties", default=12)
parser.add_argument("--fontsize_title", type=float, help="Fontsize of the titles", default=15)

parser.add_argument("--linewidth_source_prop", type=float, help="Linewidth of the source properties", default=3.0)
parser.add_argument("--label_size_source_prop", type=float, help="Label size of the source properties", default=12)
parser.add_argument("--legend_size", type=float, help="Legend size", default=12)

parser.add_argument("--min_log_qf", type=float, help="Minimum quality factor to plot", default=1.0)
parser.add_argument("--max_log_qf", type=float, help="Maximum quality factor to plot", default=5.0)

parser.add_argument("--alpha_shade", type=float, help="Opacity of the shaded area", default=0.2)

parser.add_argument("--panel_label_size", type=float, help="Size of the panel labels", default=14)
parser.add_argument("--panel_label_offset_x", type=float, help="X-coordinate offset of the panel label", default=-0.04)
parser.add_argument("--panel_label_offset_y", type=float, help="Y-coordinate offset of the panel label", default=0.00)

parser.add_argument("--qf_obs_anno_x", type=float, help="X-coordinate of the quality factor observation annotation", default=0.4)
parser.add_argument("--qf_obs_anno_y", type=float, help="Y-coordinate of the quality factor observation annotation", default=3.5)
parser.add_argument("--qf_obs_anno_text_x", type=float, help="X-coordinate offset of the quality factor observation annotation text", default=0.5)
parser.add_argument("--qf_obs_anno_text_y", type=float, help="Y-coordinate offset of the quality factor observation annotation text", default=2.0)

parser.add_argument("--qf_sat_label_x", type=float, help="X-coordinate of the quality factor saturation label", default=0.5)
parser.add_argument("--qf_sat_label_y", type=float, help="Y-coordinate of the quality factor saturation label", default=4.0)

# Parse the arguments
args = parser.parse_args()

mode_name = args.mode_name
base_name = args.base_name
base_order = args.base_order
window_length_stft = args.window_length_stft

colormap_name_depth = args.colormap_name_depth
color_observed = args.color_observed

fontsize_label_phys = args.fontsize_label_phys
fontsize_title = args.fontsize_title

linewidth_source_prop = args.linewidth_source_prop

figwidth = args.figwidth
figheight = args.figheight

# Source properties
depths_to_plot = loads(args.source_depths)

legend_size = args.legend_size

alpha_shade = args.alpha_shade

min_log_qf = args.min_log_qf
max_log_qf = args.max_log_qf

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

axis_label_size = 12
tick_label_size = 10

dim_label_offset_x = 0.0
dim_label_offset_y = 5.0

### Plot the inferred source dimensions and quality factors ###
fig_dim, ax_dim = subplots(figsize=(figwidth, figheight))
fig_qf, ax_qf_curve = subplots(figsize=(figwidth, figheight))

# Read the quality factor histogram
filename = f"stationary_resonance_qf_log_histogram_{mode_name}.csv"
filepath = join(dirname_spec, filename)
qf_df = read_csv(filepath)

# Extract the the quality factors
bin_centers = qf_df["log_quality_factor"].values
counts = qf_df["normalized_count"].values

# Plot the inferred source dimensions and quality factors

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
                    linewidth=linewidth_source_prop, color=color,
                    label=f"{depth:.0f} m")
            
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
ax_qf_curve.set_xlabel("Gas volume fraction", fontsize = axis_label_size)
# ax_dim.set_xlabel("Gas volume fraction", fontsize = axis_label_size)

# Set the title
ax_dim.set_title(f"Inferred fracture dimension", fontsize = fontsize_title, fontweight = "bold")
ax_qf_curve.set_title(f"Observed & modeled quality factors", fontsize = fontsize_title, fontweight = "bold")

# Plot the legends
ax_qf_curve.legend(fontsize = legend_size, loc = "upper left", title="Burial depth", title_fontsize = legend_size)

### Save the figures ###
print("Saving the figures...")
filename_qf = f"liu_2025a_source_props_qf.png"
save_figure(fig_qf, filename_qf)

filename_dim = f"liu_2025a_source_props_dimension.png"
save_figure(fig_dim, filename_dim)
