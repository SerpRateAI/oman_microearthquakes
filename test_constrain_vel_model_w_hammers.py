"""
Test constraining the velocity model with the hammer arrival times.
"""

#-----------
# Imports
#-----------

from pathlib import Path
from utils_basic import VEL_MODEL_DIR as dirpath_vel, LOC_DIR as dirpath_loc, get_geophone_coords
from numpy import asarray, sqrt, zeros, unravel_index
from tqdm import tqdm
from h5py import File
from matplotlib.pyplot import subplots

from utils_basic import EASTMIN_A_LOC as min_east, EASTMAX_A_LOC as max_east, NORTHMIN_A_LOC as min_north, NORTHMAX_A_LOC as max_north
from utils_basic import PICK_DIR as dirpath_pick
from utils_snuffler import read_time_windows
from utils_loc import process_arrival_info
from utils_plot import save_figure
from pandas import read_csv, DataFrame, Timestamp

#-----------
# Helper functions
#-----------

# Read the hammer travel time parameters
def read_travel_time_parameters(h5file):
    surface_vel_grid = h5file["surface_vel"][:]
    vel_gradient_grid = h5file["vel_gradient"][:]
    east_grid = h5file["east_grid"][:]
    north_grid = h5file["north_grid"][:]

    return surface_vel_grid, vel_gradient_grid, east_grid, north_grid


# Compute the misfit matrix for all grid points at once
def get_misfit_matrix_vectorized(stations, arrival_times, weights, travel_time_dict):
    # shape: (n_stations, n_north, n_east)
    travel_times = asarray([travel_time_dict[station] for station in stations])

    # shape: (n_stations, n_north, n_east)
    origin_estimates = arrival_times[:, None, None] - travel_times

    # Weighted mean origin time at each grid point
    w2 = 1.0 / weights**2
    origin_time = (origin_estimates * w2[:, None, None]).sum(axis=0) / w2.sum()

    # Residuals after removing the origin time
    residuals = arrival_times[:, None, None] - travel_times - origin_time[None, :, :]

    # Weighted RMS misfit
    misfit_mat = sqrt(((residuals / weights[:, None, None]) ** 2).sum(axis=0) / w2.sum())

    return misfit_mat, origin_time


# Plot the source and station locations
def plot_source_and_station_locations(source_df):
    station_df = get_geophone_coords()

    fig, ax = subplots(1, 1, figsize=(10, 10))
    ax.scatter(
        station_df["east"],
        station_df["north"],
        s=100,
        marker="^",
        color="lightgray",
        edgecolor="black",
        linewidth=0.8,
        label="Stations",
    )
    ax.scatter(
        source_df["east"],
        source_df["north"],
        s=140,
        marker="*",
        color="salmon",
        edgecolor="black",
        linewidth=0.8,
        label="Hammer sources",
        alpha=0.9,
    )
    ax.set_xlim(min_east, max_east)
    ax.set_ylim(min_north, max_north)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Hammer source and station locations", fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    save_figure(fig, "hammer_source_and_station_locations.png")


# Plot the average misfit matrix in velocity parameter space
def plot_avg_misfit_matrix(vel_gradient_grid, surface_vel_grid, avg_misfit_mat, vel_gradient_min, surface_vel_min):
    fig, ax = subplots(1, 1, figsize=(10, 10))
    im = ax.pcolormesh(vel_gradient_grid, surface_vel_grid, avg_misfit_mat, cmap="binary", vmax=3e-3)
    ax.set_xlabel("Velocity gradient (m s$^{-1}$ m$^{-1}$)")
    ax.set_ylabel("Surface velocity (m s$^{-1}$)")
    ax.set_title("Average misfit", fontsize=14, fontweight="bold")
    ax.scatter(vel_gradient_min, surface_vel_min, color="crimson", marker="*", s=200, edgecolor="black", linewidth=1.0)

    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Misfit (s)")

    save_figure(fig, "misfit_vs_vel_model_params_all_hammers.png")


#-----------
# Main function
#-----------


print(f"Testing constraining the velocity model with the hammer arrival times using a series of hammers...")

# Read the hammer IDs
print("Loading the hammer IDs...")
filepath = Path(dirpath_pick) / "hammer_ids_for_vel_model_estimate.csv"
hammer_df = read_csv(filepath, dtype={"hammer_id": str})
hammer_ids = hammer_df["hammer_id"].to_list()
num_hammers = len(hammer_ids)
print(f"Number of hammers: {num_hammers}")

# Read the the arrival time picks for all hammers
print("Loading the arrival time picks for all hammers...")
arrival_info_dict = {}
for hammer_id in hammer_ids:
    filename = f"hammer_{hammer_id}.mkr"
    filepath = Path(dirpath_pick) / filename
    arrival_df = read_time_windows(filepath, phase_marker=True)
    arrival_df = process_arrival_info(arrival_df, "manual_stack")
    arrival_df = arrival_df.reset_index(drop=True)
    arrival_info_dict[hammer_id] = {
        "stations": arrival_df["station"].to_list(),
        "arrival_times": arrival_df["arrival_time"].to_numpy(),
        "weights": arrival_df["uncertainty"].to_numpy(),
    }

# Load the hammer travel time maps
print("Loading the hammer travel time maps...")
filename = "hammer_travel_time_maps.h5"
filepath = Path(dirpath_vel) / filename



# Iterate over the parameter space
with File(filepath, "r") as h5file:
    # Read the travel time parameters
    surface_vel_grid, vel_gradient_grid, east_grid, north_grid = read_travel_time_parameters(h5file)

    # Initialize the misfit sum matrix
    min_misfit_sum_mat = zeros((len(surface_vel_grid), len(vel_gradient_grid)))

    # Loop through the parameter space
    print("Computing the misfit matrix for each parameter pair...")


    for i_surface_vel, surface_vel in tqdm(
        enumerate(surface_vel_grid),
        total=len(surface_vel_grid),
        desc="Surface velocity"
    ):
        for i_vel_gradient, vel_gradient in enumerate(vel_gradient_grid):
            group = h5file[f"surface_vel_{i_surface_vel}_vel_gradient_{i_vel_gradient}"]

            # Read the hammer arrival times
            for hammer_id in hammer_ids:
                arrival_info = arrival_info_dict[hammer_id]
                stations = arrival_info["stations"]
                arrival_times = arrival_info["arrival_times"]
                weights = arrival_info["weights"]

                # Compute the misfit matrix
                misfit_mat, _ = get_misfit_matrix_vectorized(
                    stations,
                    arrival_times,
                    weights,
                    group
                )
                
                # Get the minimum misfit
                min_misfit = misfit_mat.min()
                min_misfit_sum_mat[i_surface_vel, i_vel_gradient] += min_misfit

# Compute the average misfit
avg_misfit_mat = min_misfit_sum_mat / num_hammers

# Get the minimum misfit
min_misfit = avg_misfit_mat.min()
i_surface_vel_min, i_vel_gradient_min = unravel_index(avg_misfit_mat.argmin(), avg_misfit_mat.shape)
surface_vel_min = surface_vel_grid[i_surface_vel_min]
vel_gradient_min = vel_gradient_grid[i_vel_gradient_min]

print(f"Average minimum misfit: {min_misfit:.4f} s")
print(f"Surface velocity: {surface_vel_min:.1f} m/s")
print(f"Velocity gradient: {vel_gradient_min:.1f} 1/s")

# Save the optimum location of each hammer for the best velocity model
print("Computing and saving optimum location of each hammer shot...")
with File(filepath, "r") as h5file:
    group = h5file[f"surface_vel_{i_surface_vel_min}_vel_gradient_{i_vel_gradient_min}"]
    hammer_location_dicts = []

    for hammer_id in hammer_ids:
        arrival_info = arrival_info_dict[hammer_id]
        stations = arrival_info["stations"]
        arrival_times = arrival_info["arrival_times"]
        weights = arrival_info["weights"]

        misfit_mat, origin_time_mat = get_misfit_matrix_vectorized(
            stations,
            arrival_times,
            weights,
            group
        )
        i_north_min, i_east_min = unravel_index(misfit_mat.argmin(), misfit_mat.shape)

        hammer_location_dicts.append(
            {
                "hammer_id": hammer_id,
                "east": east_grid[i_east_min],
                "north": north_grid[i_north_min],
                "origin_time": Timestamp(origin_time_mat[i_north_min, i_east_min], unit="s"),
                "min_misfit": misfit_mat[i_north_min, i_east_min],
                "surface_velocity": surface_vel_min,
                "velocity_gradient": vel_gradient_min,
            }
        )

location_df = DataFrame(hammer_location_dicts)
filepath = Path(dirpath_loc) / "hammer_locations_for_vel_model_estimate.csv"
location_df.to_csv(filepath, index=False)
print(f"Saved optimum hammer locations to {filepath}.")

# Plot the source and station locations
plot_source_and_station_locations(location_df)
print("Saved source and station location plot.")
    
# Plot the misfit matrix
plot_avg_misfit_matrix(
    vel_gradient_grid,
    surface_vel_grid,
    avg_misfit_mat,
    vel_gradient_min,
    surface_vel_min,
)

# Save the velocity model parameters 
filepath = Path(dirpath_vel) / "vel_model_params_from_hammers.csv"
vel_model_params_df = DataFrame([{
    "surface_vel": surface_vel_min,
    "vel_gradient": vel_gradient_min,
}])
vel_model_params_df.to_csv(filepath, index=False)
print(f"Saved velocity model parameters to {filepath}.")
