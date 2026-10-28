"""
Compute event travel-time volumes for Subarray A or B using a linear velocity gradient model.

Reads unscaled ``surface_vel`` and ``vel_gradient`` from ``linear_vel_model_params_a.csv``
(default under ``VEL_MODEL_DIR``). For each scale factor ``S``, both parameters are scaled as
``v0 = v0_unscaled * S`` and ``alpha = alpha_unscaled * S``.

Output HDF5 (one file per subarray): ``event_travel_time_volumes_subarray_{a|b}.h5`` with datasets
``scale_factors``, coordinate axes, and groups ``i_scale_factor_{index}/<station>``.
"""

from argparse import ArgumentParser
from pathlib import Path

from h5py import File
from matplotlib.pyplot import subplots
from numpy import arange, arccosh, asarray, isclose, meshgrid, ndarray, sqrt
from pandas import read_csv

from utils_basic import (
    CORE_STATIONS_A as stations_a,
    CORE_STATIONS_B as stations_b,
    EASTMAX_A_LOC as max_east_a,
    EASTMAX_B_LOC as max_east_b,
    EASTMIN_A_LOC as min_east_a,
    EASTMIN_B_LOC as min_east_b,
    NORTHMAX_A_LOC as max_north_a,
    NORTHMAX_B_LOC as max_north_b,
    NORTHMIN_A_LOC as min_north_a,
    NORTHMIN_B_LOC as min_north_b,
    VEL_MODEL_DIR as dirpath_vel,
    get_geophone_coords,
)
from utils_plot import format_east_xlabels, format_north_ylabels, save_figure

SUBARRAY_CONFIG = {
    "A": {
        "stations": stations_a,
        "min_east": min_east_a,
        "max_east": max_east_a,
        "min_north": min_north_a,
        "max_north": max_north_a,
    },
    "B": {
        "stations": stations_b,
        "min_east": min_east_b,
        "max_east": max_east_b,
        "min_north": min_north_b,
        "max_north": max_north_b,
    },
}


def get_travel_time_3d(x, z_s, v0, alpha):
    """
    Travel time from a source at depth z_s to a surface receiver.

    v(z) = v0 + alpha*z  (if alpha = 0, v is constant v0)
    """
    x = asarray(x, dtype=float)
    z_s = asarray(z_s, dtype=float)
    alpha_arr = asarray(alpha, dtype=float)

    if alpha_arr.size == 1 and isclose(float(alpha_arr), 0.0):
        return sqrt(x**2 + z_s**2) / v0

    alpha = alpha_arr
    v_s = v0 + alpha * z_s
    arg = 1.0 + alpha**2 * (x**2 + z_s**2) / (2.0 * v0 * v_s)
    return (1.0 / alpha) * arccosh(arg)


def initialize_output_file(
    filepath: Path,
    scale_factors: list,
    easts_grid: ndarray,
    norths_grid: ndarray,
    depths_grid: ndarray,
):
    with File(filepath, "w") as f:
        f.create_dataset("scale_factors", data=scale_factors, dtype=float)
        f.create_dataset("easts_grid", data=easts_grid, dtype=float)
        f.create_dataset("norths_grid", data=norths_grid, dtype=float)
        f.create_dataset("depths_grid", data=depths_grid, dtype=float)


def save_travel_time_volumes(filepath: Path, i_scale_factor: int, travel_time_dict: dict):
    with File(filepath, "a") as f:
        group = f.require_group(f"i_scale_factor_{i_scale_factor:d}")
        for station, travel_time_vol in travel_time_dict.items():
            group.create_dataset(station, data=travel_time_vol, dtype=float)


def plot_travel_time_slice(
    travel_time_vol: ndarray,
    easts_grid: ndarray,
    norths_grid: ndarray,
    depths_grid: ndarray,
    i_depth_to_plot: int,
    east_station: float,
    north_station: float,
    station: str,
    subarray: str,
):
    fig, ax = subplots(1, 1, figsize=(10, 10))
    travel_time_vol_slice = travel_time_vol[:, :, i_depth_to_plot]
    im = ax.pcolormesh(easts_grid, norths_grid, travel_time_vol_slice, cmap="viridis")
    ax.scatter(east_station, north_station, marker="^", s=100, color="salmon", edgecolors="black")
    format_east_xlabels(ax)
    format_north_ylabels(ax)

    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + pos.width + 0.01, pos.y0, 0.02, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Travel time (s)")

    depth_to_plot = depths_grid[i_depth_to_plot]
    save_figure(
        fig,
        f"event_travel_time_slice_subarray_{subarray.lower()}_{station}_depth{depth_to_plot:.0f}m.png",
    )


def event_travel_time_volumes_filepath(subarray: str, dirpath_vel_model: Path) -> Path:
    letter = str(subarray).upper()
    if letter not in SUBARRAY_CONFIG:
        raise ValueError(f"Unsupported subarray {subarray!r}; expected 'A' or 'B'.")
    return dirpath_vel_model / f"event_travel_time_volumes_subarray_{letter.lower()}.h5"


parser = ArgumentParser(
    description=(
        "Compute event travel-time volumes for Subarray A or B from a scaled linear "
        "velocity-gradient model (linear_vel_model_params_a.csv)."
    ),
)
parser.add_argument(
    "--subarray",
    type=str,
    required=True,
    choices=["a", "b", "A", "B"],
    help="Subarray name (A or B)",
)
parser.add_argument(
    "--scale_factors",
    type=float,
    nargs="+",
    default=[0.1 * i for i in range(1, 11)],
    help="Velocity-model scale factors applied to v0 and alpha",
)
parser.add_argument(
    "--vel_params_file",
    type=Path,
    default=None,
    help="CSV with surface_vel and vel_gradient (default: VEL_MODEL_DIR/linear_vel_model_params_a.csv)",
)
parser.add_argument("--east_int", type=float, default=1.0, help="East grid spacing (m)")
parser.add_argument("--north_int", type=float, default=1.0, help="North grid spacing (m)")
parser.add_argument("--depth_int", type=float, default=1.0, help="Depth grid spacing (m)")
parser.add_argument("--max_depth", type=float, default=30.0, help="Maximum depth (m)")
parser.add_argument("--test", action="store_true", help="Run one station and one scale factor only")

args = parser.parse_args()
subarray = args.subarray.upper()
scale_factors = args.scale_factors
east_int = args.east_int
north_int = args.north_int
depth_int = args.depth_int
max_depth = args.max_depth
test = args.test

vel_params_path = args.vel_params_file
if vel_params_path is None:
    vel_params_path = Path(dirpath_vel) / "linear_vel_model_params_a.csv"
else:
    vel_params_path = Path(vel_params_path)

print(f"Computing event travel-time volumes for Subarray {subarray}...")
print(f"Velocity parameters: {vel_params_path}")
print(f"Scale factors: {scale_factors}")
if test:
    print("Running in test mode.")

vel_params_df = read_csv(vel_params_path)
surface_vel = float(vel_params_df.loc[0, "surface_vel"])
vel_gradient = float(vel_params_df.loc[0, "vel_gradient"])

config = SUBARRAY_CONFIG[subarray]
stations = config["stations"]
coords_df = get_geophone_coords()

easts_grid = arange(config["min_east"], config["max_east"], east_int)
norths_grid = arange(config["min_north"], config["max_north"], north_int)
depths_grid = arange(0.0, max_depth, depth_int)

filepath = event_travel_time_volumes_filepath(subarray, Path(dirpath_vel))
initialize_output_file(filepath, scale_factors, easts_grid, norths_grid, depths_grid)

for i_scale_factor, scale_factor in enumerate(scale_factors):
    print(f"Computing travel-time volumes for scale factor {scale_factor:.3f}...")
    surface_vel_scaled = surface_vel * scale_factor
    vel_gradient_scaled = vel_gradient * scale_factor

    travel_time_dict = {}
    for i_station, station in enumerate(stations):
        print(f"  Station {station}...")
        station_east = coords_df.loc[station, "east"]
        station_north = coords_df.loc[station, "north"]

        easts_mesh, norths_mesh, depths_mesh = meshgrid(easts_grid, norths_grid, depths_grid)
        distance_mesh = sqrt((easts_mesh - station_east) ** 2 + (norths_mesh - station_north) ** 2)
        travel_time_vol = get_travel_time_3d(
            distance_mesh, depths_mesh, surface_vel_scaled, vel_gradient_scaled,
        )
        travel_time_dict[station] = travel_time_vol

        if i_station == 0 and test:
            plot_travel_time_slice(
                travel_time_vol,
                easts_grid,
                norths_grid,
                depths_grid,
                0,
                station_east,
                station_north,
                station,
                subarray,
            )

    save_travel_time_volumes(filepath, i_scale_factor, travel_time_dict)

    if test:
        print("Finished in test mode.")
        break

print(f"Saved travel-time volumes to {filepath}.")
