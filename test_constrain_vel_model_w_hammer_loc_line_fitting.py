"""
Constrain the velocity model using multiple hammer shots.

For each velocity-model parameter pair:
    1. Compute the source-location misfit grid for each hammer shot
    2. Pick the best-fitting source location for each shot independently
    3. Fit a line to the inferred source locations
    4. Use the line-fit RMS perpendicular distance as the velocity-model misfit
"""

# -----------
# Imports
# -----------

from argparse import ArgumentParser
from pathlib import Path

from numpy import array, zeros, sqrt, mean, linalg, linspace, argmin, unravel_index, column_stack, inf
from numpy.linalg import svd
from h5py import File
from matplotlib.pyplot import subplots
from pandas import read_csv
from tqdm import tqdm

from utils_basic import VEL_MODEL_DIR as dirpath_vel
from utils_basic import PICK_DIR as dirpath_pick
from utils_loc import process_arrival_info
from utils_plot import save_figure
from utils_snuffler import read_time_windows


# -----------
# I/O helpers
# -----------

def read_travel_time_parameters(h5file):
    surface_vel_grid = h5file["surface_vel"][:]
    vel_gradient_grid = h5file["vel_gradient"][:]
    east_grid = h5file["east_grid"][:]
    north_grid = h5file["north_grid"][:]
    return surface_vel_grid, vel_gradient_grid, east_grid, north_grid


def read_hammer_ids(filepath):
    df = read_csv(filepath, dtype={"hammer_id": str})
    
    return df["hammer_id"].tolist()


def load_arrival_info_for_hammer(hammer_id):
    filepath = Path(dirpath_pick) / f"hammer_{hammer_id}.mkr"
    arrival_df = read_time_windows(filepath, phase_marker=True)
    arrival_df = process_arrival_info(arrival_df, "manual_stack")
    arrival_df = arrival_df.reset_index(drop=True)
    return arrival_df


# -----------
# Misfit calculation
# -----------

def get_misfit_matrix_vectorized(stations, arrival_times, weights, travel_time_dict):
    """
    Compute the source-location misfit over the full spatial grid.

    Parameters
    ----------
    stations : list[str]
    arrival_times : ndarray, shape (n_stations,)
    weights : ndarray, shape (n_stations,)
    travel_time_dict : dict[station] -> ndarray, shape (n_north, n_east)

    Returns
    -------
    misfit_mat : ndarray, shape (n_north, n_east)
    """
    # shape: (n_stations, n_north, n_east)
    travel_times = array([travel_time_dict[station] for station in stations])

    # shape: (n_stations, n_north, n_east)
    origin_estimates = arrival_times[:, None, None] - travel_times

    # weighted mean origin time at each grid point
    w2 = 1.0 / weights**2
    origin_time = (origin_estimates * w2[:, None, None]).sum(axis=0) / w2.sum()

    # residuals after removing origin time
    residuals = arrival_times[:, None, None] - travel_times - origin_time[None, :, :]

    # weighted RMS arrival-time misfit
    misfit_mat = sqrt(((residuals / weights[:, None, None]) ** 2).sum(axis=0) / w2.sum())

    return misfit_mat


def get_best_source_location(misfit_mat, east_grid, north_grid):
    """
    Return the best-fitting source location for one hammer shot.
    """
    i_north, i_east = unravel_index(argmin(misfit_mat), misfit_mat.shape)

    return {
        "i_east": int(i_east),
        "i_north": int(i_north),
        "east": float(east_grid[i_east]),
        "north": float(north_grid[i_north]),
        "misfit": float(misfit_mat[i_north, i_east]),
    }


# -----------
# Line fitting
# -----------

def fit_line_total_least_squares(east, north):
    """
    Fit a 2D line to points using total least squares / PCA.

    Returns
    -------
    result : dict with:
        centroid : ndarray, shape (2,)
        direction : ndarray, shape (2,)
        normal : ndarray, shape (2,)
        distances : ndarray, shape (n_points,)
        rms : float
    """
    points = column_stack([east, north])
    centroid = points.mean(axis=0)

    centered = points - centroid[None, :]

    # SVD of centered coordinates
    # first right-singular vector = line direction
    _, _, vh = svd(centered, full_matrices=False)

    direction = vh[0]
    normal = vh[1]

    # signed perpendicular distances to line
    distances = centered @ normal
    rms = float(sqrt(mean(distances**2)))

    return {
        "centroid": centroid,
        "direction": direction,
        "normal": normal,
        "distances": distances,
        "rms": rms,
    }


# -----------
# Main
# -----------

parser = ArgumentParser()
parser.add_argument(
    "--hammer_ids_file",
    type=str,
    default="hammer_ids_line_1.csv",
    help="CSV file containing hammer IDs",
)
parser.add_argument(
    "--combine_with_arrival_misfit",
    action="store_true",
    help="Also compute and save the summed per-shot arrival misfit for reference",
)

args = parser.parse_args()

hammer_ids_file = args.hammer_ids_file
combine_with_arrival_misfit = args.combine_with_arrival_misfit

print("Loading hammer IDs...")
filepath = Path(dirpath_pick) / hammer_ids_file
hammer_ids = read_hammer_ids(filepath)
print(f"Found {len(hammer_ids)} hammer shots.")

print("Loading hammer arrival picks...")
arrival_info_list = []
all_stations = set()

for hammer_id in hammer_ids:
    arrival_df = load_arrival_info_for_hammer(hammer_id)

    stations = arrival_df["station"].tolist()
    arrival_times = arrival_df["arrival_time"].to_numpy()
    weights = arrival_df["uncertainty"].to_numpy()

    arrival_info_list.append(
        {
            "hammer_id": hammer_id,
            "stations": stations,
            "arrival_times": arrival_times,
            "weights": weights,
        }
    )
    all_stations.update(stations)

print("Loading travel-time parameter grids...")
filepath_h5 = Path(dirpath_vel) / "hammer_travel_time_maps.h5"

with File(filepath_h5, "r") as h5file:
    surface_vel_grid, vel_gradient_grid, east_grid, north_grid = read_travel_time_parameters(h5file)

    # objective used to choose the velocity model:
    # line-fit RMS of independently inferred hammer locations
    line_misfit_mat = zeros((len(surface_vel_grid), len(vel_gradient_grid)), dtype=float)

    # optional diagnostic: sum of minimum per-shot arrival-time misfits
    arrival_misfit_avg_mat = zeros((len(surface_vel_grid), len(vel_gradient_grid)), dtype=float)

    best_global = {
        "line_misfit": inf,
        "i_surface_vel": None,
        "i_vel_gradient": None,
        "surface_vel": None,
        "vel_gradient": None,
        "shot_locations": None,
        "line_fit": None,
        "per_shot_misfit_mats": None,
    }

    print("Searching velocity-model parameter space...")
    num_misfit = 0
    for i_surface_vel, surface_vel in tqdm(
        enumerate(surface_vel_grid),
        total=len(surface_vel_grid),
        desc="Surface velocity",
    ):
        for i_vel_gradient, vel_gradient in enumerate(vel_gradient_grid):
            group = h5file[f"surface_vel_{i_surface_vel}_vel_gradient_{i_vel_gradient}"]

            # Only read stations that actually appear in the data
            travel_time_dict = {station: group[station][:] for station in all_stations}

            shot_locations = []
            per_shot_misfit_mats = []
            arrival_misfit_sum = 0.0

            # Locate each hammer shot independently
            num_misfit = 0
            for info in arrival_info_list:
                misfit_mat = get_misfit_matrix_vectorized(
                    info["stations"],
                    info["arrival_times"],
                    info["weights"],
                    travel_time_dict,
                )
                per_shot_misfit_mats.append(misfit_mat)

                best_loc = get_best_source_location(misfit_mat, east_grid, north_grid)
                best_loc["hammer_id"] = info["hammer_id"]
                shot_locations.append(best_loc)

                arrival_misfit_sum += best_loc["misfit"]
                num_misfit += 1

            east = array([item["east"] for item in shot_locations])
            north = array([item["north"] for item in shot_locations])

            line_fit = fit_line_total_least_squares(east, north)
            line_misfit = line_fit["rms"]

            line_misfit_mat[i_surface_vel, i_vel_gradient] = line_misfit
            arrival_misfit_avg = arrival_misfit_sum / num_misfit
            arrival_misfit_avg_mat[i_surface_vel, i_vel_gradient] = arrival_misfit_avg

            if line_misfit < best_global["line_misfit"]:
                best_global.update(
                    {
                        "line_misfit": line_misfit,
                        "i_surface_vel": i_surface_vel,
                        "i_vel_gradient": i_vel_gradient,
                        "surface_vel": float(surface_vel),
                        "vel_gradient": float(vel_gradient),
                        "shot_locations": shot_locations,
                        "line_fit": line_fit,
                        "per_shot_misfit_mats": array(per_shot_misfit_mats),
                        "arrival_misfit_avg": arrival_misfit_avg,
                    }
                )

print()
print("Best velocity model from line-fit misfit:")
print(f"  surface_vel = {best_global['surface_vel']}")
print(f"  vel_gradient = {best_global['vel_gradient']}")
print(f"  line RMS misfit = {best_global['line_misfit']:.6f} m")
print(f"  average arrival-time misfit = {best_global['arrival_misfit_avg']:.4f} s")

print()
print("Best hammer locations:")
for item, dist in zip(best_global["shot_locations"], best_global["line_fit"]["distances"]):
    print(
        f"  hammer {item['hammer_id']}: "
        f"east = {item['east']:.2f}, north = {item['north']:.2f}, "
        f"arrival misfit = {item['misfit']:.4f} s, "
        f"line residual = {dist:.4f} m"
    )

# -----------
# Plot 1: line-fit misfit in velocity-model space
# -----------

fig, ax = subplots(1, 1, figsize=(10, 10))
im = ax.pcolormesh(
    vel_gradient_grid,
    surface_vel_grid,
    line_misfit_mat,
    cmap="binary",
    shading="auto",
    vmax = 10.0,
)
ax.set_box_aspect(1)
ax.set_xlabel("Velocity gradient (m s$^{-1}$ m$^{-1}$)")
ax.set_ylabel("Surface velocity (m s$^{-1}$)")
ax.set_title("Velocity-model score from line fit to inferred hammer locations", fontsize=14, fontweight="bold")

pos = ax.get_position()
cax = fig.add_axes([pos.x0 + pos.width + 0.01, pos.y0, 0.02, pos.height])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Line-fit RMS misfit (m)")

save_figure(fig, "misfit_vs_vel_model_params_linefit.png")


# -----------
# Plot 2: optional diagnostic arrival-misfit sum
# -----------

if combine_with_arrival_misfit:
    fig, ax = subplots(1, 1, figsize=(10, 10))
    im = ax.pcolormesh(
        vel_gradient_grid,
        surface_vel_grid,
        arrival_misfit_avg_mat,
        cmap="binary",
        shading="auto",
    )
    ax.set_box_aspect(1)
    ax.set_xlabel("Velocity gradient (m s$^{-1}$ m$^{-1}$)")
    ax.set_ylabel("Surface velocity (m s$^{-1}$)")
    ax.set_title("Average minimum per-shot arrival-time misfit", fontsize=14, fontweight="bold")

    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width + 0.01, pos.y0, 0.02, pos.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Average arrival-time misfit (s)")

    save_figure(fig, "misfit_vs_vel_model_params_arrival_avg.png")


# -----------
# Plot 3: best-fit hammer locations and best-fit line
# -----------

shot_locations = best_global["shot_locations"]
east = array([item["east"] for item in shot_locations])
north = array([item["north"] for item in shot_locations])

centroid = best_global["line_fit"]["centroid"]
direction = best_global["line_fit"]["direction"]

fig, ax = subplots(1, 1, figsize=(10, 10))

# show points
ax.plot(east, north, "ro", label="Best-fit hammer locations")

for item in shot_locations:
    ax.text(item["east"], item["north"], f" {item['hammer_id']}", va="bottom", ha="left")

# show best-fit line
line_half_length = max(
    east_grid.max() - east_grid.min(),
    north_grid.max() - north_grid.min(),
)
t = linspace(-line_half_length, line_half_length, 200)
line_pts = centroid[None, :] + t[:, None] * direction[None, :]
ax.plot(line_pts[:, 0], line_pts[:, 1], "k-", label="Best-fit line")

ax.set_xlabel("East (m)")
ax.set_ylabel("North (m)")
ax.set_title("Recovered hammer locations for optimal velocity model")
ax.legend()

save_figure(fig, "best_hammer_locations_and_linefit.png")