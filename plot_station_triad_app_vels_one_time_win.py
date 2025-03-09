"""
Plot the 3C apparent velocities of all available station triads for a given time window
"""

### Import necessary modules ###

from os.path import join
from argparse import ArgumentParser
from numpy import amax, cos, sin, ones, pi, array
from pandas import read_csv, DataFrame
from matplotlib.pyplot import subplots
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from utils_basic import SPECTROGRAM_DIR as dirname_spec, MT_DIR as dirname_mt, GEO_COMPONENTS as components, EASTMIN_WHOLE as min_east, EASTMAX_WHOLE as max_east, NORTHMIN_WHOLE as min_north, NORTHMAX_WHOLE as max_north
from utils_basic import get_geophone_coords, get_station_triad_indices, str2timestamp, time2suffix
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_mt import mt_cspec, get_avg_phase_diff, get_indep_phase_diffs
from utils_plot import component2label, format_east_xlabels, format_north_ylabels, get_geo_component_color, save_figure

### Inputs ###

# Command line inputs
parser = ArgumentParser(description="Plot the station-triad 3C apparent velocities for a given time window.")
parser.add_argument("--time_window", type=str, help="Center of the time window (YYYY-MM-DDTHH:MM:SS)")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--window_length_mt", type=float, help="Window length for multitaper analysis", default=900.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence", default=0.85)

# Parse the command line inputs
args = parser.parse_args()

time_window = str2timestamp(args.time_window)
mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe

# Constants
figwidth = 15

scale_factor = 50000.0

station_size = 20

linewidth_triad = 0.5
linewidth_sta = 0.5

arrow_width = 0.002
arrow_head_width = 5
arrow_head_length = 10

east_key = -100.0
north_key = -90.0
vel_key = 2000.0

### Read in the data ###
# Read the station-triad list
print("Reading the station-triad list...")
filename = "delaunay_station_triads.csv"
filepath = join(dirname_mt, filename)
triad_df = read_csv(filepath)

# Read the station coordinates
print("Reading the station coordinates...")
coords_df = get_geophone_coords()

# Get the triangle indices
print("Getting the triangle indices...")
sta_ind_mat = get_station_triad_indices(coords_df, triad_df)

# Read the apparent velocities
print("Reading the apparent velocities...")
vec_dicts = []
for _, row in triad_df.iterrows():
    station1, station2, station3 = row["station1"], row["station2"], row["station3"]
    print(f"Reading the apparent velocities for station triad {station1}-{station2}-{station3}...")

    filename = f"station_triad_app_vels_{mode_name}_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
    filepath = join(dirname_mt, filename)
    vel_df = read_csv(filepath, parse_dates=["time"], na_values=["nan"])

    # Filter the data for the time window
    vel_df = vel_df[vel_df["time"] == time_window]

    if len(vel_df) == 0:
        print(f"No data found for station triad {station1}-{station2}-{station3} at time window {time_window}. Skipping...")
        continue
    
    # Extract the 3C apparent velocities
    east = row["east"]
    north = row["north"]
    vec_dict = {"east": east, "north": north}

    for component in components:
        vel_app = vel_df[f"vel_app_{component.lower()}"].values[0]
        back_azi = vel_df[f"back_azi_{component.lower()}"].values[0]

        vec_dict[f"vel_app_{component.lower()}"] = vel_app
        vec_dict[f"back_azi_{component.lower()}"] = back_azi

    vec_dicts.append(vec_dict)

vec_df = DataFrame(vec_dicts)

### Plot the data ###
print("Plotting the data...")

# Compute the correct figure size
figheight = figwidth * (max_north - min_north) / (max_east - min_east) / 3
figsize = (figwidth, figheight)

# Plot the data
fig, axes = subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

for i, component in enumerate(components):
    print(f"Plotting the {component} component...")
    ax = axes[i]

    # Plot the stations
    print("Plotting the stations...")
    ax.scatter(coords_df["east"], coords_df["north"], s=station_size, marker="^", c="lightgray", edgecolors="k", linewidths=linewidth_sta, zorder = 2)

    # Plot the triads
    print("Plotting the triads...")
    ax.triplot(coords_df["east"], coords_df["north"], sta_ind_mat, color="lightgray", linewidth=linewidth_triad, zorder = 1)

    # Plot the vectors
    print("Plotting the vectors...")
    vec_comp_df = vec_df[vec_df[f"vel_app_{component.lower()}"].notna()]

    easts = vec_comp_df["east"].values
    norths = vec_comp_df["north"].values
    vels_app = vec_comp_df[f"vel_app_{component.lower()}"].values
    back_azis = vec_comp_df[f"back_azi_{component.lower()}"].values

    u = vels_app * sin(back_azis * pi / 180)
    v = vels_app * cos(back_azis * pi / 180)

    color_comp = get_geo_component_color(component)
    quivers = ax.quiver(easts, norths, u, v, color = color_comp, scale = scale_factor, width = arrow_width, headwidth = arrow_head_width, headlength = arrow_head_length, zorder = 3)

    # Set the title
    label_comp = component2label(component)
    ax.set_title(f"{label_comp}", fontsize=14, fontweight="bold")

    # Set the x and y limits
    ax.set_xlim(min_east, max_east)
    ax.set_ylim(min_north, max_north)

    # Set the aspect ratio
    ax.set_aspect("equal")

    # Set the x and y labels
    format_east_xlabels(ax)

    if i == 0:
        format_north_ylabels(ax)
       
        ax.quiverkey(quivers, east_key, north_key, vel_key, label=fr"{vel_key:.0f} m s$^{{-1}}$", labelpos='E', coordinates='data')

# Set the super title
fig.suptitle(f"{time_window.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=16, fontweight="bold", y=0.95)    

# Save the figure
print("Saving the figure...")
figname = f"station_triad_app_vels_time_win_{time2suffix(time_window)}.png"
save_figure(fig, figname)