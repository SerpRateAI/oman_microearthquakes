"""
Plot the back-azimuth standard deviations of the apparent velocities of the approaching vehicle and the stationary-resonance over the two subarrays against the distance between the vehicle and the boreholes
"""

###
# Import the necessary libraries
###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from numpy import deg2rad, rad2deg, sqrt, sum, nan
from matplotlib.pyplot import subplots
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from utils_basic import LOC_DIR as dirpath_loc
from utils_basic import get_mode_order
from utils_basic import INNER_STATIONS_A as inner_stations_a, MIDDLE_STATIONS_A as middle_stations_a, OUTER_STATIONS_A as outer_stations_a
from utils_basic import INNER_STATIONS_B as inner_stations_b, MIDDLE_STATIONS_B as middle_stations_b, OUTER_STATIONS_B as outer_stations_b
from utils_basic import GEO_COMPONENTS as components
from utils_plot import save_figure

###
# Input parameters
###
parser = ArgumentParser()
parser.add_argument('--mode_name', type=str, default='PR02549', help='The name of the stationary-resonance mode to plot')
parser.add_argument('--min_num_triad', type=int, default=5, help='The minimum number of station-triads to compute the standard deviation for the apparent velocities of the approaching vehicle')
parser.add_argument('--max_back_azi_std', type=float, default=15.0, help='The maximum back-azimuth standard deviation to plot')
parser.add_argument('--occurrence', type=str, default='approaching', help='The occurrence of the vehicle')
parser.add_argument('--window_length', type=int, default=900, help='The length of the window to compute the average apparent velocity of the stationary resonance for a station-triad')
parser.add_argument('--min_cohe_reson', type=float, default=0.85, help='The minimum coherence to use for the apparent velocities of the stationary resonance')
parser.add_argument('--min_num_obs_reson', type=int, default=100, help='The minimum number of observations to compute the standard deviation for the apparent velocities of the stationary resonance')

parser.add_argument('--figwidth', type=float, default=10.0, help='The width of the figure')
parser.add_argument('--figheight', type=float, default=10.0, help='The height of the figure')

parser.add_argument('--markersize', type=float, default=120.0, help='The size of the markers')
parser.add_argument('--linewidth_vehicle', type=float, default=1.0, help='The edge width of the markers for the apparent velocities of the approaching vehicle')
parser.add_argument('--linewidth_resonance', type=float, default=3.0, help='The line width of the apparent velocities of the stationary resonance')

parser.add_argument('--cmap_name', type=str, default="Set2", help='The name of the colormap to use')
parser.add_argument('--i_color_a', type=int, default=0, help='The index of the color for Subarray A')
parser.add_argument('--i_color_b', type=int, default=1, help='The index of the color for Subarray B')

parser.add_argument('--min_std', type=float, default=0.0, help='The minimum standard deviation to plot')
parser.add_argument('--max_std', type=float, default=110.0, help='The maximum standard deviation to plot')

parser.add_argument('--fontsize_axis_label', type=float, default=12.0, help='The font size of the axis labels')
parser.add_argument('--fontsize_legend', type=float, default=12.0, help='The font size of the legend')
parser.add_argument('--fontsize_title', type=float, default=14.0, help='The font size of the title')

args = parser.parse_args()  
mode_name = args.mode_name
min_num_triad = args.min_num_triad
max_back_azi_std = args.max_back_azi_std
occurrence = args.occurrence
window_length = args.window_length
min_cohe_reson = args.min_cohe_reson
min_num_obs_reson = args.min_num_obs_reson

figwidth = args.figwidth
figheight = args.figheight

cmap_name = args.cmap_name
i_color_a = args.i_color_a
i_color_b = args.i_color_b

linewidth_vehicle = args.linewidth_vehicle
linewidth_resonance = args.linewidth_resonance
markersize = args.markersize

min_std = args.min_std
max_std = args.max_std

fontsize_axis_label = args.fontsize_axis_label
fontsize_legend = args.fontsize_legend
fontsize_title = args.fontsize_title

###
# Read the Input files
###

print("Reading the vehicle back-azimuth standard deviations...")
filename = f'vehicle_subarray_back_azi_stds_vs_dist_{occurrence}_min_num_triad{min_num_triad:d}_max_back_azi_std{max_back_azi_std:.0f}.csv'
inpath = join(dirpath_loc, filename)
vehicle_std_df = read_csv(inpath)

print("Reading the stationary-resonance apparent velocities...")
filename = f'stationary_resonance_subarray_back_azi_stds_{mode_name}_mt_win{window_length:.0f}s_min_cohe{min_cohe_reson:.2f}_min_num_obs{min_num_obs_reson:d}.csv'
inpath = join(dirpath_loc, filename)
reson_std_df = read_csv(inpath)

###
# Plot the standard deviations of the back-azimuths of the hammer-shot against the distance and the standard deviations of the back-azimuths of the stationary resonance as horizontal lines
###

fig, ax = subplots(1, 1, figsize=(figwidth, figheight))

cmap = colormaps[cmap_name]
color_a = cmap(i_color_a)
color_b = cmap(i_color_b)

# Plot the standard deviations of the back-azimuths of the vehicle
ax.scatter(vehicle_std_df['distance_a'], vehicle_std_df['back_azi_std_a'], 
           color=color_a, marker='o', s=markersize, edgecolor="black", linewidth=linewidth_vehicle,
           label='A')

ax.scatter(vehicle_std_df['distance_b'], vehicle_std_df['back_azi_std_b'], 
           color=color_b, marker='o', s=markersize, edgecolor="black", linewidth=linewidth_vehicle,
           label='B')

# Plot the standard deviations of the back-azimuths of the stationary resonance
back_azi_std_a = reson_std_df.loc[reson_std_df['subarray'] == 'A', 'back_azi_std'].values[0]
back_azi_std_b = reson_std_df.loc[reson_std_df['subarray'] == 'B', 'back_azi_std'].values[0]
ax.axhline(back_azi_std_a, color=color_a, linewidth=linewidth_resonance)
ax.axhline(back_azi_std_b, color=color_b, linewidth=linewidth_resonance)

# Set the labels and title
ax.set_xlabel('Vehicle distance to BA1A/BA1B (m)', fontsize=fontsize_axis_label)
ax.set_ylabel('Propagation-direction standard deviation (deg)', fontsize=fontsize_axis_label)

# Set the limits of the y-axis
ax.set_ylim(min_std, max_std)

# Add the legend
legend = ax.legend(fontsize=fontsize_legend, loc='lower left', framealpha=1.0)
legend.set_title('Subarray', prop={'size': fontsize_legend, 'weight': 'bold'})

# Add the title
mode_order = get_mode_order(mode_name)
ax.set_title(f'{occurrence.capitalize()} vehicle vs Mode {mode_order}', fontsize=fontsize_title, fontweight='bold')

###
# Save the figure
###
save_figure(fig, f'vehicle_and_resonance_subarray_baz_stds_vs_dist_{occurrence}.png')







