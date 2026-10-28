"""
Plot 3D arrival-time misfit distributions for the Group 18 hub event (AGU Advances).

Panel (a) uses the unscaled velocity model (scale factor 1). Panel (b) uses the
input velocity scale factor (default 0.5). Reads
``hub_event_location_info_group18.h5`` written by
``localize_hub_event_3d_per_scale_factor.py``. Map view is the best-fit depth
slice; the profile is the east–depth slice at the best-fit north.
"""

from argparse import ArgumentParser
from pathlib import Path

from h5py import File
from numpy import abs, argmin

from utils_basic import (
    EASTMAX_A_MIDDLE,
    EASTMIN_A_MIDDLE,
    LOC_DIR,
    NORTHMAX_A_MIDDLE,
    NORTHMIN_A_MIDDLE,
    get_geophone_coords,
)
from utils_loc import _get_hub_event_scale_factor_group, plot_misfit_distribution
from utils_plot import save_figure


GROUP_LABEL = 18
SCALE_FACTOR_UNSCALED = 1.0
N_COLUMNS = 2
DEFAULT_LOCATION_H5 = Path(LOC_DIR) / f"hub_event_location_info_group{GROUP_LABEL:d}.h5"
DEFAULT_FIGNAME = "aguadv_hub_event_misfit_distribution.png"


def load_misfit_distribution(location_h5, scale_factor):
    """Load misfit volume, grids, best-fit indices, and stations at ``scale_factor``."""
    if not location_h5.is_file():
        raise FileNotFoundError(
            f"Hub-event location file not found: {location_h5}. "
            "Run localize_hub_event_3d_per_scale_factor.py to produce it."
        )

    with File(location_h5, "r") as h5file:
        group, scale_factor_used = _get_hub_event_scale_factor_group(h5file, scale_factor)
        east = float(group.attrs["east"])
        north = float(group.attrs["north"])
        depth = float(group.attrs["depth"])
        min_misfit = float(group.attrs["min_misfit"])

        vol = group["misfit_volume"]
        easts_grid = vol["east_grid"][:]
        norths_grid = vol["north_grid"][:]
        depths_grid = vol["depth_grid"][:]
        misfit_vol = vol["misfit"][:]
        stations = list(group["arrival_time"].attrs.keys())

    i_east = int(argmin(abs(easts_grid - east)))
    i_north = int(argmin(abs(norths_grid - north)))
    i_depth = int(argmin(abs(depths_grid - depth)))

    station_df = get_geophone_coords()
    station_df = station_df[station_df.index.isin(stations)]
    station_df = station_df.reset_index(drop=False)
    if "name" not in station_df.columns:
        first = station_df.columns[0]
        station_df = station_df.rename(columns={first: "name"})

    return (
        misfit_vol,
        easts_grid,
        norths_grid,
        depths_grid,
        i_east,
        i_north,
        i_depth,
        station_df,
        east,
        north,
        depth,
        min_misfit,
        scale_factor_used,
    )


def add_panel_label(ax, label):
    ax.text(
        -0.1,
        1.08,
        label,
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )


def plot_misfit_column(data, misfitmax, title, panel_label, column_index, fig=None, show_cbar=True, show_legend=False):
    (
        misfit_vol,
        easts_grid,
        norths_grid,
        depths_grid,
        i_east,
        i_north,
        i_depth,
        station_df,
        east,
        north,
        depth,
        min_misfit,
        scale_factor_used,
    ) = data
    print(
        f"Panel {panel_label}: scale factor {scale_factor_used:.4f}; "
        f"best location: {east:.1f} m E, {north:.1f} m N, {depth:.1f} m D; "
        f"min misfit = {min_misfit:.6f} s; {len(station_df):d} stations"
    )
    fig, ax_map, _, _ = plot_misfit_distribution(
        misfit_vol,
        easts_grid,
        norths_grid,
        depths_grid,
        i_east,
        i_north,
        i_depth,
        station_df,
        misfitmax=misfitmax,
        title=title,
        east_lim=(EASTMIN_A_MIDDLE, EASTMAX_A_MIDDLE),
        north_lim=(NORTHMIN_A_MIDDLE, NORTHMAX_A_MIDDLE),
        fig=fig,
        column_index=column_index,
        n_columns=N_COLUMNS,
        annotate_stations=False,
        show_cbar=show_cbar,
        show_legend=show_legend,
    )
    add_panel_label(ax_map, panel_label)
    return fig


def main():
    parser = ArgumentParser(
        description=(
            "Plot 3D arrival-time misfit distributions for the Group 18 hub event "
            "at the unscaled model and at an input velocity scale factor (AGU Advances)."
        )
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.5,
        help="Velocity scale factor for panel (b) (default: 0.5)",
    )
    parser.add_argument(
        "--location_h5",
        type=Path,
        default=DEFAULT_LOCATION_H5,
        help="HDF5 with hub-event localizations (default: LOC_DIR/hub_event_location_info_group18.h5)",
    )
    parser.add_argument(
        "--misfitmax",
        type=float,
        default=0.01,
        help="Upper color scale for misfit (seconds)",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=DEFAULT_FIGNAME,
        help=f"Output figure filename (default: {DEFAULT_FIGNAME})",
    )
    args = parser.parse_args()
    scale_factor = args.scale_factor
    location_h5 = args.location_h5
    misfitmax = args.misfitmax
    figname = args.figname

    print("--------------------------------")
    print(f"File: {location_h5}")
    print(f"Panel (a): unscaled, scale factor {SCALE_FACTOR_UNSCALED:.2f}")
    print(f"Panel (b): scale factor {scale_factor:.2f}")
    print("--------------------------------")

    print(
        f"Loading Group {GROUP_LABEL:d} misfit distribution at unscaled "
        f"scale factor {SCALE_FACTOR_UNSCALED:.2f}..."
    )
    data_unscaled = load_misfit_distribution(location_h5, SCALE_FACTOR_UNSCALED)

    print(
        f"Loading Group {GROUP_LABEL:d} misfit distribution at input scale factor "
        f"{scale_factor:.2f}..."
    )
    data_scaled = load_misfit_distribution(location_h5, scale_factor)

    fig = plot_misfit_column(
        data_unscaled,
        misfitmax,
        title=f"Scale factor = {SCALE_FACTOR_UNSCALED:.2f}",
        panel_label="(a)",
        column_index=0,
        fig=None,
        show_legend=True,
    )
    plot_misfit_column(
        data_scaled,
        misfitmax,
        title=f"Scale factor = {scale_factor:.2f}",
        panel_label="(b)",
        column_index=1,
        fig=fig,
        show_cbar=False,
    )
    fig.suptitle("Event Group 18", fontsize=14, fontweight="bold", y=1.05)
    print(
        f"Map axis limits (Subarray A middle): "
        f"east [{EASTMIN_A_MIDDLE:g}, {EASTMAX_A_MIDDLE:g}] m, "
        f"north [{NORTHMIN_A_MIDDLE:g}, {NORTHMAX_A_MIDDLE:g}] m"
    )

    print(f"Saving figure to {figname}...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
