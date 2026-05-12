"""
Joint hammer localization and effective constant-velocity fit in one step.

Uses arrival times from every station that has a pick and known coordinates to estimate,
in a single weighted nonlinear least-squares inversion:

    - source east
    - source north
    - source origin time
    - one effective constant velocity (homogeneous medium, path-averaged)

Pick uncertainties are used as weights. Source position is bounded to the inner-pad
extent from utils_basic (same as the previous inner-only localization).
"""

# -----------
# Imports
# -----------

from argparse import ArgumentParser
from pathlib import Path

from numpy import sqrt, ones, clip, diag
from numpy.linalg import pinv
from scipy.optimize import least_squares
from pandas import Timestamp
from matplotlib.pyplot import subplots

from utils_basic import PICK_DIR as dirpath_pick
from utils_basic import (
    EASTMIN_A_INNER as min_east_inner,
    EASTMAX_A_INNER as max_east_inner,
    NORTHMIN_A_INNER as min_north_inner,
    NORTHMAX_A_INNER as max_north_inner,
)
from utils_snuffler import read_time_windows
from utils_loc import process_arrival_info
from utils_basic import get_geophone_coords
from utils_plot import save_figure


# -----------
# Helper functions
# -----------

def weighted_residual_function(theta, station_coords, arrival_times, arrival_uncerts):
    """
    Residual function for weighted least squares.

    Parameters
    ----------
    theta : array-like
        [x, y, tau, c]
    station_coords : ndarray, shape (N, 2)
        Station east/north coordinates in meters.
    arrival_times : ndarray, shape (N,)
        Observed arrival times in seconds.
    arrival_uncerts : ndarray, shape (N,)
        Pick standard deviations in seconds.

    Returns
    -------
    residuals : ndarray, shape (N,)
        Weighted residuals.
    """
    x, y, tau, c = theta

    if c <= 0.0:
        return 1e12 * ones(len(arrival_times))

    dx = station_coords[:, 0] - x
    dy = station_coords[:, 1] - y
    dist = sqrt(dx**2 + dy**2)

    residuals = (arrival_times - tau - dist / c) / arrival_uncerts
    return residuals


def localize_hammer_shot(arrival_df, c_init=500.0, arrival_uncert_col="uncertainty"):
    arrival_times = arrival_df["arrival_time"].to_numpy(dtype=float)
    station_coords = arrival_df[["east", "north"]].to_numpy(dtype=float)

    # Shift times to avoid optimizing on epoch seconds
    t_ref = arrival_times.min()
    arrival_times_rel = arrival_times - t_ref

    if arrival_uncert_col in arrival_df.columns:
        print(f"Using uncertainties from column '{arrival_uncert_col}'...")
        arrival_uncerts = arrival_df[arrival_uncert_col].to_numpy(dtype=float)
    else:
        print("No uncertainty column found. Using unit weights.")
        arrival_uncerts = ones(len(arrival_df), dtype=float)

    arrival_uncerts = clip(arrival_uncerts, 1e-6, None)

    x_init = station_coords[:, 0].mean()
    y_init = station_coords[:, 1].mean()
    tau_init = arrival_times_rel.mean() - sqrt(
        (x_init - station_coords[0, 0])**2 + (y_init - station_coords[0, 1])**2
    ) / c_init

    source_params_init = [x_init, y_init, tau_init, c_init]

    print("--------------------------------")
    print("Initial guess:")
    print(f"  x = {x_init:.2f} m E")
    print(f"  y = {y_init:.2f} m N")
    print(f"  tau_rel = {tau_init:.6f} s")
    print(f"  velocity = {c_init:.2f} m/s")
    print("--------------------------------")

    lower_bounds = [min_east_inner, min_north_inner, -1.0, 50.0]
    upper_bounds = [max_east_inner, max_north_inner, arrival_times_rel.max() + 1.0, 5000.0]

    result = least_squares(
        weighted_residual_function,
        source_params_init,
        args=(station_coords, arrival_times_rel, arrival_uncerts),
        bounds=(lower_bounds, upper_bounds),
    )

    x_opt, y_opt, tau_rel_opt, c_opt = result.x

    dx = station_coords[:, 0] - x_opt
    dy = station_coords[:, 1] - y_opt
    dist = sqrt(dx**2 + dy**2)

    residuals_sec = arrival_times_rel - tau_rel_opt - dist / c_opt
    weights = 1.0 / arrival_uncerts**2

    wrms_seconds = sqrt((weights * residuals_sec**2).sum() / weights.sum())

    chi2 = ((residuals_sec / arrival_uncerts) ** 2).sum()
    dof = max(len(arrival_times_rel) - 4, 1)
    reduced_chi2 = chi2 / dof

    jt_j = result.jac.T @ result.jac
    source_cov = reduced_chi2 * pinv(jt_j)
    source_std = sqrt(clip(diag(source_cov), 0.0, None))

    # Convert tau back to absolute seconds
    result.x[2] = t_ref + tau_rel_opt

    return result, wrms_seconds, reduced_chi2, source_cov, source_std


def plot_station_and_source_locations(arrival_df, source_params_opt):
    """
    Plot station positions and inferred source location.
    """
    fig, ax = subplots(1, 1, figsize=(10, 10))
    ax.scatter(
        arrival_df["east"],
        arrival_df["north"],
        marker="^",
        label="Station",
        color="lightgray",
        edgecolors="black",
        s=100,
    )
    ax.scatter(
        source_params_opt[0],
        source_params_opt[1],
        marker="*",
        label="Source",
        color="salmon",
        edgecolors="black",
        s=140,
    )
    ax.legend()
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")
    ax.set_xlim(min_east_inner, max_east_inner)
    ax.set_ylim(min_north_inner, max_north_inner)

    return fig, ax


# -----------
# Main
# -----------

parser = ArgumentParser()
parser.add_argument("--hammer_id", type=str, required=True, help="Hammer ID")
parser.add_argument(
    "--arrival_uncert_col",
    type=str,
    default="uncertainty",
    help="Column name containing pick uncertainty in seconds",
)

args = parser.parse_args()
hammer_id = args.hammer_id
arrival_uncert_col = args.arrival_uncert_col

print(f"Joint hammer localization for hammer {hammer_id}...")

print("Loading station coordinates...")
station_df = get_geophone_coords().copy()
station_df["station"] = station_df.index

print("Loading hammer arrival times...")
filepath = Path(dirpath_pick) / f"hammer_{hammer_id}.mkr"
arrival_df = read_time_windows(filepath, phase_marker=True)
arrival_df = process_arrival_info(arrival_df, "manual_stack")
arrival_df = arrival_df.reset_index(drop=True).copy()

arrival_loc_df = arrival_df.merge(
    station_df[["station", "east", "north"]],
    on="station",
    how="inner",
).reset_index(drop=True)

if len(arrival_loc_df) < 4:
    raise ValueError(
        "Need at least 4 stations with picks and known coordinates to estimate x, y, tau, and c."
    )

print(f"Using {len(arrival_loc_df)} station arrivals for joint inversion.")

print("Finding the best-fitting source, origin time, and effective velocity...")
source_params_opt, wrms_loc, redchi2_loc, _, source_std = localize_hammer_shot(
    arrival_loc_df,
    arrival_uncert_col=arrival_uncert_col,
)

east_opt = source_params_opt.x[0]
north_opt = source_params_opt.x[1]
origin_time_opt = source_params_opt.x[2]
velocity_opt = source_params_opt.x[3]

print("--------------------------------")
print("Best-fitting source and homogeneous velocity:")
print(f"  x = {east_opt:.2f} +/- {source_std[0]:.2f} m E")
print(f"  y = {north_opt:.2f} +/- {source_std[1]:.2f} m N")
print(f"  tau = {Timestamp(origin_time_opt, unit='s')} +/- {source_std[2]:.4f} s")
print(f"  velocity = {velocity_opt:.2f} +/- {source_std[3]:.2f} m/s")
print(f"  Weighted RMS misfit = {wrms_loc:.4f} s")
print(f"  Reduced chi^2 = {redchi2_loc:.3f}")
print("--------------------------------")

fig, ax = plot_station_and_source_locations(arrival_loc_df, source_params_opt.x)
ax.set_title(f"Hammer location, {hammer_id}", fontsize=14, fontweight="bold")
save_figure(fig, f"hammer_location_{hammer_id}.png")