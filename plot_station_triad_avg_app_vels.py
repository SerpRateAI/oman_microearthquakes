"""
Plot the average apparent velocities of station triads satisfying the criteria
Only the triangles formed by the inner and intermdiate layers are considered
"""

### Import packages ###
from os.path import join
from argparse import ArgumentParser
from numpy import isnan, pi, sin, cos, nan
from matplotlib.pyplot import subplots
from matplotlib import colormaps
from matplotlib.colors import Normalize
from pandas import read_csv
from pandas import DataFrame

from utils_basic import get_geophone_coords
from utils_basic import GEO_COMPONENTS as components, MT_DIR as dirname_mt, INNER_STATIONS as inner_stations, MIDDLE_STATIONS as middle_stations, OUTER_STATIONS as outer_stations
from utils_basic import EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north

from utils_plot import add_colorbar, format_east_xlabels, format_north_ylabels, get_geo_component_color, save_figure

### Input arguments ###
# Command line arguments
parser = ArgumentParser()

parser.add_argument("--max_back_azi_std", type=float, help="Maximum standard deviation of the back azimuth", default=15.0)

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--window_length_mt", type=float, help="Window length of the multitaper analysis", default=900.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence of the multitaper analysis", default=0.85)

args = parser.parse_args()

max_back_azi_std = args.max_back_azi_std

mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe

# Constants
figwidth = 7.0

marker_size_station = 30.0
linewidth_station = 0.5

linewidth_triad = 1.0

scale_factor = 50.0

quiver_width = 0.003
quiver_head_width = 3.0
quiver_head_length = 5.0
quiver_linewidth = 0.5

quiver_key_length = 2.0

min_vel_app = 0.0
max_vel_app = 3000.0

colorbar_offset_x = 0.05
colorbar_offset_y = 0.1
colorbar_width = 0.02
colorbar_height = 0.3

### Load data ###
# Station coordinates
coord_df = get_geophone_coords()

# Station triads
filename = f"delaunay_station_triads.csv"
filepath = join(dirname_mt, filename)
triad_df = read_csv(filepath)

# Apparent velocities
filename = f"station_triad_avg_app_vels_{mode_name}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirname_mt, filename)
vel_df = read_csv(filepath)

### Filter data ###
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
    east1 = coord_df.loc[station1, "east"]
    north1 = coord_df.loc[station1, "north"]
    east2 = coord_df.loc[station2, "east"]
    north2 = coord_df.loc[station2, "north"]
    east3 = coord_df.loc[station3, "east"]
    north3 = coord_df.loc[station3, "north"]

    vec_dict = {"station1": station1, "station2": station2, "station3": station3, 
                "east_center": east_triad, "north_center": north_triad}
            
    # Check if the back azimuth standard deviation is within the maximum
    num_valid_comp = 0
    for component in components:
        back_azi_std = row[f"std_back_azi_{component.lower()}"]
        if back_azi_std > max_back_azi_std:
            vec_dict[f"avg_vel_app_{component.lower()}_east"] = nan
            vec_dict[f"avg_vel_app_{component.lower()}_north"] = nan
            continue
        
        num_valid_comp += 1
        vel_app = row[f"avg_vel_app_{component.lower()}"]
        back_azi = row[f"avg_back_azi_{component.lower()}"]

        # Length of the vector is inversely proportional to the standard deviation of the back azimuth
        vec_length = 1.0 / back_azi_std * max_back_azi_std

        vec_dict[f"vel_app_{component.lower()}"] = vel_app
        vec_dict[f"vec_{component.lower()}_east"] = vec_length * sin(back_azi / 180 * pi)
        vec_dict[f"vec_{component.lower()}_north"] = vec_length * cos(back_azi / 180 * pi)

    if num_valid_comp == 0:
        continue

    vec_dicts.append(vec_dict)

vec_df = DataFrame(vec_dicts)

### Plot ###
# Compute the figure size
figheight = figwidth / (max_east - min_east) * (max_north - min_north)

# Create the figure
fig, ax = subplots(figsize=(figwidth, figheight))

# Define the colormap for the vectors
norm = Normalize(vmin=min_vel_app, vmax=max_vel_app)
cmap = colormaps["gray"]


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
            east1, north1 = coord_df.loc[station1, "east"], coord_df.loc[station1, "north"]
            east2, north2 = coord_df.loc[station2, "east"], coord_df.loc[station2, "north"]
            ax.plot([east1, east2], [north1, north2], color="lightgray", linewidth=linewidth_triad, zorder=1)

            edges_plotted.add(edge)
            stations_to_plot.update(stations)

    # Plot the vectors
    for component in components:
        color_comp = get_geo_component_color(component)
        vec_east = row[f"vec_{component.lower()}_east"]
        vec_north = row[f"vec_{component.lower()}_north"]
        vel_app = row[f"vel_app_{component.lower()}"]

        if not isnan(vec_east) and not isnan(vec_north):
            quiver = ax.quiver(row["east_center"], row["north_center"], vec_east, vec_north, vel_app, 
                               cmap=cmap, norm=norm,
                               scale=scale_factor, width=quiver_width,
                               headwidth=quiver_head_width, headlength=quiver_head_length,
                               edgecolor=color_comp, linewidth=quiver_linewidth,
                               zorder=3)

# Plot the stations 
stations_to_plot = list(stations_to_plot)
coord_plot_df = coord_df.loc[stations_to_plot]
# ax.scatter(coord_plot_df["east"], coord_plot_df["north"], s=marker_size_station, marker="^", facecolor="lightgray", edgecolor="black", linewidth=linewidth_station, zorder=2)

# Add a quiver key
quiver_key = ax.quiverkey(quiver, 0.1, 0.05, quiver_key_length, f"{max_back_azi_std / quiver_key_length:.0f} deg", labelpos="E", coordinates="axes", color="black")

# Add a colorbar
bbox = ax.get_position()
position = [bbox.x0 + colorbar_offset_x, bbox.y0 + colorbar_offset_y, colorbar_width, colorbar_height]
cbar = add_colorbar(fig, position, "Apparent velocity (m s$^{-1}$)",
                    cmap=cmap, norm=norm)

# Set the x and y limits
ax.set_xlim(min_east, max_east)
ax.set_ylim(min_north, max_north)

# Set the x and y labels
format_east_xlabels(ax)
format_north_ylabels(ax)

# Set the aspect ratio
ax.set_aspect("equal")

# Set the title
ax.set_title(f"Time-averaged apparent velocities of {mode_name}", fontsize=12, fontweight="bold")

# Save the figure
figname = f"station_triad_avg_app_vels_{mode_name}.png"
save_figure(fig, figname)

# Save the data
filename = f"station_triad_avg_app_vels_to_plot_{mode_name}_window_length_mt{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}_max_back_azi_std{max_back_azi_std:.0f}deg.csv"
filepath = join(dirname_mt, filename)
vec_df.to_csv(filepath, index=True, na_rep="nan")
print(f"Saved the data to {filepath}")

