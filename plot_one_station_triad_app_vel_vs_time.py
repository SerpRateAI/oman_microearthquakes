"""
Plot the apparent velocity vs time for a station triad
"""

from os.path import join
from argparse import ArgumentParser
from pandas import read_csv
from matplotlib.pyplot import subplots
from matplotlib.gridspec import GridSpec

from utils_basic import MT_DIR as dirname_mt, GEO_COMPONENTS as components, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_station_triad_indices
from utils_plot import component2label, format_datetime_xlabels, format_east_xlabels, format_north_ylabels, format_app_vel_ylabels, format_back_azi_ylabels, get_geo_component_color, save_figure

###
# Inputs
###

# Command-line arguments
parser = ArgumentParser(description="Plot the apparent velocity vs time for a station triad")

parser.add_argument("--station1", type=str, help="Station 1")
parser.add_argument("--station2", type=str, help="Station 2")
parser.add_argument("--station3", type=str, help="Station 3")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--window_length_mt", type=float, help="Window length for multitaper analysis", default=900.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence", default=0.85)

# Parse the command line inputs
args = parser.parse_args()

station1_plot = args.station1
station2_plot = args.station2
station3_plot = args.station3

mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe

# Constants
figwidth = 15.0
figheight = 6.0

vel_marker_size = 5.0
sta_marker_size = 15.0

min_vel = 0.0
max_vel = 4000.0

ax_gap = 0.07

linewidth_triad_thin = 0.5
linewidth_triad_thick = 1.0
linewidth_sta = 0.5

# Print the input
print(f"Station 1: {station1_plot}")
print(f"Station 2: {station2_plot}")
print(f"Station 3: {station3_plot}")
print(f"Mode name: {mode_name}")
print(f"Window length: {window_length_mt:.0f} s")
print(f"Minimum coherence: {min_cohe:.2f}")

###
# Load the data
###

# Load the station coordinates
coord_df = get_geophone_coords()

# Load the station triad information
filename = "delaunay_station_triads.csv"
filepath = join(dirname_mt, filename)
triad_df = read_csv(filepath)

# Load the apparent velocities
filename = f"station_triad_app_vels_{mode_name}_{station1_plot}_{station2_plot}_{station3_plot}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirname_mt, filename)
vel_df = read_csv(filepath, parse_dates=["time"])

###
# Plot the data
###

# Plot the stations and the triads while highlighting the one whose apparent velocities are shown on the right
fig, ax_map = subplots(1, 1, figsize=(figwidth, figheight))

# Plot the stations
ax_map.scatter(coord_df["east"], coord_df["north"], sta_marker_size, marker="^", facecolor="lightgray", edgecolor="black", linewidth=linewidth_sta, zorder=2)

# Plot the triads
# Plot the highlighted triad first

east1 = coord_df.loc[station1_plot, "east"]
north1 = coord_df.loc[station1_plot, "north"]

east2 = coord_df.loc[station2_plot, "east"] 
north2 = coord_df.loc[station2_plot, "north"]

east3 = coord_df.loc[station3_plot, "east"]
north3 = coord_df.loc[station3_plot, "north"]

ax_map.plot([east1, east2], [north1, north2], color="crimson", linewidth=linewidth_triad_thick, zorder=1)
ax_map.plot([east2, east3], [north2, north3], color="crimson", linewidth=linewidth_triad_thick, zorder=1)
ax_map.plot([east3, east1], [north3, north1], color="crimson", linewidth=linewidth_triad_thick, zorder=1)

plotted_edges = set([tuple(sorted([station1_plot, station2_plot])), tuple(sorted([station2_plot, station3_plot])), tuple(sorted([station3_plot, station1_plot]))])

# Plot the other triads while avoiding duplicates
for _, row in triad_df.iterrows():
    station1_triad = row["station1"] 
    station2_triad = row["station2"]
    station3_triad = row["station3"]
    
    east1 = coord_df.loc[station1_triad, "east"]
    north1 = coord_df.loc[station1_triad, "north"]
    
    east2 = coord_df.loc[station2_triad, "east"] 
    north2 = coord_df.loc[station2_triad, "north"]
    
    east3 = coord_df.loc[station3_triad, "east"]
    north3 = coord_df.loc[station3_triad, "north"]
        
    # Plot each edge only if not already plotted
    # Sort station pairs to ensure consistent edge identification
    edges = set([tuple(sorted([station1_triad, station2_triad])), tuple(sorted([station2_triad, station3_triad])), tuple(sorted([station3_triad, station1_triad]))])
    
    for edge in edges:
        if edge not in plotted_edges:
            station1, station2 = edge
            east1 = coord_df.loc[station1, "east"]
            north1 = coord_df.loc[station1, "north"]
            east2 = coord_df.loc[station2, "east"]
            north2 = coord_df.loc[station2, "north"]
            ax_map.plot([east1, east2], [north1, north2], color="lightgray", linewidth=linewidth_triad_thin, zorder=1)
            plotted_edges.add(edge)

# Set the map limits
ax_map.set_xlim(min_east, max_east)
ax_map.set_ylim(min_north, max_north)

# Set the map aspect ratio
ax_map.set_aspect("equal")

# Set the axis labels
format_east_xlabels(ax_map)
format_north_ylabels(ax_map)

# Add two subplots for the apparent velocities and the back azimuths
bbox = ax_map.get_position()  # Get the current axis's position

# Create two vertically stacked axes to the right of the map
pos_new = [bbox.x1 + ax_gap, bbox.y0, 1-bbox.width-ax_gap, bbox.height] 
gs = GridSpec(2, 1, height_ratios=[1, 1])
gs.update(left=pos_new[0], bottom=pos_new[1], right=pos_new[0]+pos_new[2], top=pos_new[1]+pos_new[3])

ax_vel_app = fig.add_subplot(gs[0])  # Top subplot
ax_back_azi = fig.add_subplot(gs[1])  # Bottom subplot

# Plot the apparent velocities
for component in components:
    color = get_geo_component_color(component)
    label = component2label(component)
    ax_vel_app.scatter(vel_df["time"], vel_df[f"vel_app_{component.lower()}"], label=label, color=color, s=vel_marker_size)

ax_vel_app.legend()
ax_vel_app.set_xlim(starttime, endtime)
ax_vel_app.set_ylim(min_vel, max_vel)

format_datetime_xlabels(ax_vel_app,
                        plot_axis_label = False,
                        plot_tick_label = True,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing="5d", 
                        num_minor_ticks=5)

format_app_vel_ylabels(ax_vel_app,
                        plot_axis_label = True,
                        plot_tick_label = True,
                        major_tick_spacing=1000.0,
                        num_minor_ticks=5)

# Plot the back azimuths
for component in components:
    color = get_geo_component_color(component)
    ax_back_azi.scatter(vel_df["time"], vel_df[f"back_azi_{component.lower()}"], label=label, color=color, s=vel_marker_size)

ax_back_azi.set_xlabel("Time")
ax_back_azi.set_ylabel("Back azimuth (deg)")
ax_back_azi.set_xlim(starttime, endtime)
ax_back_azi.set_ylim(-180.0, 180.0)

format_datetime_xlabels(ax_back_azi,
                        plot_axis_label = True,
                        plot_tick_label = True,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing="5d", 
                        num_minor_ticks=5)

format_back_azi_ylabels(ax_back_azi,
                        plot_axis_label = True,
                        plot_tick_label = True,
                        major_tick_spacing=45.0,
                        num_minor_ticks=3)

fig.suptitle(f"{mode_name}, {station1_plot}-{station2_plot}-{station3_plot}", fontsize=14, fontweight="bold", y=0.93)

###
# Save the figure
###
figname = f"station_triad_app_vel_vs_time_{mode_name}_{station1_plot}_{station2_plot}_{station3_plot}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.png"
save_figure(fig, figname)

