"""
Plot the standard deviations of the back-azimuths of the hammer-shot and stationary-resonance apparent velocities in Subarray A against the distance between the hammer location and Hole A
"""

###
# Import the necessary libraries
###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from numpy import deg2rad, rad2deg, sqrt, sum, nan
from matplotlib.pyplot import subplots
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from utils_basic import LOC_DIR as dirpath_loc
from utils_basic import INNER_STATIONS_A as inner_stations_a, MIDDLE_STATIONS_A as middle_stations_a, OUTER_STATIONS_A as outer_stations_a
from utils_basic import INNER_STATIONS_B as inner_stations_b, MIDDLE_STATIONS_B as middle_stations_b, OUTER_STATIONS_B as outer_stations_b
from utils_basic import GEO_COMPONENTS as components
from utils_basic import get_borehole_coords, get_angles_std, get_mode_order
from utils_plot import save_figure, get_geo_component_color, component2label

###
# Input parameters
###
parser = ArgumentParser()
parser.add_argument('--mode_name', type=str, default='PR02549', help='The name of the stationary-resonance mode to plot')
parser.add_argument('--min_num_obs_hammer', type=int, default=10, help='The minimum number of observations to compute the standard deviation for the hammer-shot apparent velocities')
parser.add_argument('--min_num_obs_reson', type=int, default=100, help='The minimum number of observations to compute the average apparent velocity of the stationary resonance for a station-triad')
parser.add_argument('--freq_target', type=float, default=25.0, help='The target frequency of the apparent velocities to plot')
parser.add_argument('--window_length', type=int, default=900, help='The length of the window to compute the average apparent velocity of the stationary resonance for a station-triad')
parser.add_argument('--min_cohe_hammer', type=float, default=0.50, help='The minimum coherence to use for the hammer-shot apparent velocities')
parser.add_argument('--min_cohe_reson', type=float, default=0.85, help='The minimum coherence to use for the stationary-resonance apparent velocities')

parser.add_argument('--figwidth', type=float, default=15.0, help='The width of the figure')
parser.add_argument('--figheight', type=float, default=7.5, help='The height of the figure')
parser.add_argument('--wspace', type=float, default=0.1, help='The width of the space between the subplots')

parser.add_argument('--markersize_hammer', type=float, default=120.0, help='The size of the markers for the hammer shots')
parser.add_argument('--linewidth_reson', type=float, default=3.0, help='The width of the lines for the stationary resonance')
parser.add_argument('--linewidth_hammer', type=float, default=2.0, help='The width of the lines for the hammer shots')

parser.add_argument('--fontsize_axis_label', type=float, default=12.0, help='The font size of the axis labels')
parser.add_argument('--fontsize_legend', type=float, default=12.0, help='The font size of the legend')
parser.add_argument('--fontsize_title', type=float, default=14.0, help='The font size of the title')



args = parser.parse_args()
mode_name = args.mode_name
min_num_obs_hammer = args.min_num_obs_hammer
min_num_obs_reson = args.min_num_obs_reson
freq_target = args.freq_target
min_cohe_hammer = args.min_cohe_hammer
min_cohe_reson = args.min_cohe_reson
window_length = args.window_length

figwidth = args.figwidth
figheight = args.figheight
wspace = args.wspace

linewidth_reson = args.linewidth_reson
linewidth_hammer = args.linewidth_hammer
markersize_hammer = args.markersize_hammer
fontsize_axis_label = args.fontsize_axis_label
fontsize_legend = args.fontsize_legend
fontsize_title = args.fontsize_title

###
# Read the Input files
###
print("Reading the borehole location...")
borehole_df = get_borehole_coords()
east_hole_a = borehole_df.loc["BA1A", 'east']
north_hole_a = borehole_df.loc["BA1A", 'north']

print("Reading the hammer location...")
inpath = join(dirpath_loc, 'hammer_locations.csv')
hammer_loc_df = read_csv(inpath, dtype={'hammer_id': str})

print("Reading the stationary-resonance apparent velocities...")
filename = f'stationary_resonance_station_triad_avg_app_vels_{mode_name}_mt_win{window_length:.0f}s_min_cohe{min_cohe_reson:.2f}_min_num_obs{min_num_obs_reson:d}.csv'
inpath = join(dirpath_loc, filename)
reson_vel_df = read_csv(inpath)

###
# Compute the distance and standard deviation for each hammer shot
###
stations_a = inner_stations_a + middle_stations_a + outer_stations_a
stations_b = inner_stations_b + middle_stations_b + outer_stations_b

print("Computing the distance and standard deviation for each hammer shot...")
hammer_std_dicts = []
for _, row in hammer_loc_df.iterrows():
    hammer_id = row['hammer_id']
    east_hammer = row['east']
    north_hammer = row['north']

    # Compute the distance to Hole A
    dist = sqrt((east_hammer - east_hole_a) ** 2 + (north_hammer - north_hole_a) ** 2)
    
    # Compute the standard deviation of the apparent velocities
    # Read the apparent velocities
    filename = f"hammer_station_triad_app_vels_mt_{hammer_id}_{freq_target:.0f}hz_min_cohe{min_cohe_hammer:.2f}.csv"
    try:
        inpath = join(dirpath_loc, filename)
        vel_df = read_csv(inpath)
    except FileNotFoundError:
        print(f"File not found. Skipping hammer shot {hammer_id}...")
        continue

    # Keep only the rows consisting of the stations in Subarray A or Subarray B
    vel_a_df = vel_df[(vel_df['station1'].isin(stations_a)) & (vel_df['station2'].isin(stations_a)) & (vel_df['station3'].isin(stations_a))]
    vel_b_df = vel_df[(vel_df['station1'].isin(stations_b)) & (vel_df['station2'].isin(stations_b)) & (vel_df['station3'].isin(stations_b))]
    
    # Compute the standard deviation of the apparent velocities
    hammer_std_dict = {'hammer_id': hammer_id, 'distance': dist}
    for component in components:
        back_azis_a = vel_a_df[f'back_azi_{component.lower()}'].values[~vel_a_df[f'back_azi_{component.lower()}'].isna()]
        back_azis_b = vel_b_df[f'back_azi_{component.lower()}'].values[~vel_b_df[f'back_azi_{component.lower()}'].isna()]

        if len(back_azis_a) >= min_num_obs_hammer:
            back_azis_a = deg2rad(back_azis_a)
            std_back_azi_a = get_angles_std(back_azis_a)
            hammer_std_dict[f'std_a_{component.lower()}'] = rad2deg(std_back_azi_a)
        else:
            hammer_std_dict[f'std_a_{component.lower()}'] = nan

        if len(back_azis_b) >= min_num_obs_hammer:
            back_azis_b = deg2rad(back_azis_b)
            std_back_azi_b = get_angles_std(back_azis_b)
            hammer_std_dict[f'std_b_{component.lower()}'] = rad2deg(std_back_azi_b)
        else:
            hammer_std_dict[f'std_b_{component.lower()}'] = nan

    hammer_std_dicts.append(hammer_std_dict)

hammer_std_df = DataFrame(hammer_std_dicts)

###
# Compute the standard deviation of the apparent velocities for the stationary resonance
###
reson_std_dict = {}
for component in components:
    # Keep only the rows consisting of the stations in Subarray A
    reson_vel_a_df = reson_vel_df[(reson_vel_df['station1'].isin(stations_a)) & (reson_vel_df['station2'].isin(stations_a)) & (reson_vel_df['station3'].isin(stations_a))]
    reson_vel_b_df = reson_vel_df[(reson_vel_df['station1'].isin(stations_b)) & (reson_vel_df['station2'].isin(stations_b)) & (reson_vel_df['station3'].isin(stations_b))]

    # Compute the standard deviation of the apparent velocities
    back_azis_a = reson_vel_a_df[f'avg_back_azi_{component.lower()}'].values[~reson_vel_a_df[f'avg_back_azi_{component.lower()}'].isna()]
    back_azis_b = reson_vel_b_df[f'avg_back_azi_{component.lower()}'].values[~reson_vel_b_df[f'avg_back_azi_{component.lower()}'].isna()]

    back_azis_a = deg2rad(back_azis_a)
    back_azis_b = deg2rad(back_azis_b)

    std_back_azi_a = get_angles_std(back_azis_a)
    std_back_azi_b = get_angles_std(back_azis_b)
    reson_std_dict[f'std_a_{component.lower()}'] = rad2deg(std_back_azi_a)
    reson_std_dict[f'std_b_{component.lower()}'] = rad2deg(std_back_azi_b)

###
# Plot the standard deviations of the back-azimuths of the hammer-shot against the distance and the standard deviations of the back-azimuths of the stationary resonance as horizontal lines
###
fig, axs = subplots(1, 2, figsize=(figwidth, figheight), sharey=True)
fig.subplots_adjust(wspace=wspace)

for component in components:
    color = get_geo_component_color(component)
    component_label = component2label(component)

    # Plot the standard deviations of the back-azimuths of the hammer-shot
    axs[0].scatter(hammer_std_df['distance'], hammer_std_df[f'std_a_{component.lower()}'], 
               marker='o', s=markersize_hammer, facecolor=color, edgecolor=color, linewidth=linewidth_hammer, label=f'{component_label}')
    axs[0].axhline(reson_std_dict[f'std_a_{component.lower()}'], color=color, linewidth=linewidth_reson)

    # Plot the standard deviations of the back-azimuths of Subarray B
    axs[1].scatter(hammer_std_df['distance'], hammer_std_df[f'std_b_{component.lower()}'], 
               marker='o', s=markersize_hammer, facecolor=color, edgecolor=color, linewidth=linewidth_hammer)
    # Plot the standard deviations of the back-azimuths of the stationary resonance
    axs[1].axhline(reson_std_dict[f'std_b_{component.lower()}'], color=color, linewidth=linewidth_reson)

axs[0].set_xlabel('Shot distance to BA1A (m)', fontsize=fontsize_axis_label)
axs[0].set_ylabel('Propagation-direction standard deviation (deg)', fontsize=fontsize_axis_label)

axs[1].set_xlabel('Shot distance to BA1A (m)', fontsize=fontsize_axis_label)

axs[0].set_title(f"Subarray A", fontsize=fontsize_title, fontweight='bold')
axs[1].set_title(f"Subarray B", fontsize=fontsize_title, fontweight='bold')

axs[0].legend()

save_figure(fig, 'hammer_app_vels_baz_std_vs_dist.png')





