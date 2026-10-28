"""
Plot the 2D east–north arrival-time misfit map for one hammer shot (AGU Advances).

Reads ``hammer_locations_for_vel_model_estimate.h5`` written by
``test_constrain_vel_model_w_hammers.py``. The best-fit location is a salmon star;
stations used for that shot are light-gray triangles.
"""

from argparse import ArgumentParser
from pathlib import Path

from h5py import File
from matplotlib.pyplot import subplots

from utils_basic import (
    EASTMAX_A_LOC,
    EASTMIN_A_LOC,
    LOC_DIR,
    NORTHMAX_A_LOC,
    NORTHMIN_A_LOC,
    get_geophone_coords,
)
from utils_plot import format_east_xlabels, format_north_ylabels, save_figure

DEFAULT_LOCATION_H5 = Path(LOC_DIR) / "hammer_locations_for_vel_model_estimate.h5"
DEFAULT_HAMMER_ID = "100129"


def load_hammer_misfit_map(location_h5, hammer_id):
    """Load grids, misfit map, best-fit location, and stations for one hammer."""
    if not location_h5.is_file():
        raise FileNotFoundError(
            f"Hammer location file not found: {location_h5}. "
            "Run test_constrain_vel_model_w_hammers.py to produce it."
        )

    with File(location_h5, "r") as h5file:
        if hammer_id not in h5file:
            available = sorted(h5file.keys())
            raise ValueError(
                f"Hammer {hammer_id} not found in {location_h5}. "
                f"Available groups: {available}"
            )

        east_grid = h5file["east_grid"][:]
        north_grid = h5file["north_grid"][:]
        group = h5file[hammer_id]
        misfit_mat = group["misfit"][:]
        east = float(group.attrs["east"])
        north = float(group.attrs["north"])
        min_misfit = float(group.attrs["min_misfit"])
        stations = list(group["arrival_time"].attrs.keys())

    return east_grid, north_grid, misfit_mat, east, north, min_misfit, stations


def main():
    parser = ArgumentParser(
        description=(
            "Plot the 2D arrival-time misfit map for one hammer shot used in "
            "the 1D P velocity-model estimate (AGU Advances)."
        )
    )
    parser.add_argument(
        "--location_h5",
        type=Path,
        default=DEFAULT_LOCATION_H5,
        help="HDF5 with per-hammer 2D misfit maps and locations",
    )
    parser.add_argument(
        "--hammer_id",
        type=str,
        default=DEFAULT_HAMMER_ID,
        help="Hammer shot ID to plot",
    )
    parser.add_argument(
        "--misfit_vmax",
        type=float,
        default=0.01,
        help="Color scale upper limit for misfit (s)",
    )
    parser.add_argument("--figwidth", type=float, default=10.0, help="Figure width (in)")
    parser.add_argument("--figheight", type=float, default=10.0, help="Figure height (in)")
    parser.add_argument("--markersize_hammer", type=float, default=140.0, help="Hammer marker size")
    parser.add_argument("--markersize_station", type=float, default=100.0, help="Station marker size")
    parser.add_argument("--linewidth", type=float, default=0.8, help="Marker edge width")
    parser.add_argument("--color_hammer", type=str, default="salmon", help="Hammer marker color")
    parser.add_argument("--color_station", type=str, default="lightgray", help="Station marker color")
    parser.add_argument("--major_tick_spacing", type=float, default=20.0, help="East/north tick spacing (m)")
    parser.add_argument("--axis_label_size", type=float, default=12.0, help="Axis label font size")
    parser.add_argument("--tick_label_size", type=float, default=10.0, help="Tick label font size")
    parser.add_argument("--legend_size", type=float, default=12.0, help="Legend font size")
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help="Output figure filename (default: aguadv_hammer_misfit_map_{hammer_id}.png)",
    )
    args = parser.parse_args()

    location_h5 = args.location_h5
    hammer_id = args.hammer_id
    misfit_vmax = args.misfit_vmax
    figwidth = args.figwidth
    figheight = args.figheight
    markersize_hammer = args.markersize_hammer
    markersize_station = args.markersize_station
    linewidth = args.linewidth
    color_hammer = args.color_hammer
    color_station = args.color_station
    major_tick_spacing = args.major_tick_spacing
    axis_label_size = args.axis_label_size
    tick_label_size = args.tick_label_size
    legend_size = args.legend_size
    figname = args.figname
    if figname is None:
        figname = f"aguadv_hammer_misfit_map_{hammer_id}.png"

    print(f"Loading misfit map for hammer {hammer_id} from {location_h5}...")
    east_grid, north_grid, misfit_mat, east, north, min_misfit, stations = load_hammer_misfit_map(
        location_h5,
        hammer_id,
    )
    print(
        f"Best location: {east:.1f} m E, {north:.1f} m N; "
        f"min misfit = {min_misfit:.4f} s; {len(stations)} stations"
    )

    station_df = get_geophone_coords()
    station_df = station_df.loc[station_df.index.isin(stations)]

    fig, ax = subplots(1, 1, figsize=(figwidth, figheight))
    im = ax.pcolormesh(
        east_grid,
        north_grid,
        misfit_mat,
        vmin=0.0,
        vmax=misfit_vmax,
        shading="auto",
        zorder=1,
    )
    ax.scatter(
        station_df["east"],
        station_df["north"],
        s=markersize_station,
        marker="^",
        color=color_station,
        edgecolors="black",
        linewidth=linewidth,
        label="Geophone",
        zorder=2,
    )
    ax.scatter(
        east,
        north,
        s=markersize_hammer,
        marker="*",
        color=color_hammer,
        edgecolors="black",
        linewidth=linewidth,
        label="Hammer",
        zorder=3,
    )

    ax.set_xlim(EASTMIN_A_LOC, EASTMAX_A_LOC)
    ax.set_ylim(NORTHMIN_A_LOC, NORTHMAX_A_LOC)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Arrival-time misfit, Hammer {hammer_id}",
        fontsize=axis_label_size + 2,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=legend_size, framealpha=1.0)

    format_east_xlabels(
        ax,
        major_tick_spacing=major_tick_spacing,
        axis_label_size=axis_label_size,
        tick_label_size=tick_label_size,
    )
    format_north_ylabels(
        ax,
        major_tick_spacing=major_tick_spacing,
        axis_label_size=axis_label_size,
        tick_label_size=tick_label_size,
    )

    cax = ax.inset_axes([1.04, 0.0, 0.025, 1.0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Misfit (s)")

    print(f"Saving figure to {figname}...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
