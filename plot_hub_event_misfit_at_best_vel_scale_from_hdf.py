"""
Read the HDF5 output of ``localize_hub_event_3d.py``, select the velocity scale factor with the
lowest stored grid-search minimum misfit, and plot the misfit distribution with best-fit event
location and stations using ``plot_misfit_distribution`` from ``utils_loc.py``.

The localization script writes ``hub_event_location_info_group{label}.h5`` under ``LOC_DIR`` with
either a multi-scale layout (root attribute ``scale_factors``, groups ``scale_factor_0``, …) or
legacy groups named ``scale_factor_{value:.2f}``. This script discovers all ``scale_factor_*``
groups and picks the one with smallest ``min_misfit`` attribute.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from h5py import File

from utils_basic import LOC_DIR as dirpath_loc, get_geophone_coords
from utils_loc import plot_misfit_distribution
from utils_plot import save_figure


def _list_hub_scale_groups(filepath):
    """Return list of dicts with keys ``group``, ``scale_factor``, ``min_misfit``."""
    rows = []
    with File(filepath, "r") as f:
        for name in f.keys():
            if not isinstance(name, str) or not name.startswith("scale_factor_"):
                continue
            g = f[name]
            rows.append(
                {
                    "group": name,
                    "scale_factor": float(g.attrs["scale_factor"]),
                    "min_misfit": float(g.attrs["min_misfit"]),
                }
            )
    if not rows:
        raise ValueError(
            f"No groups named scale_factor_* found in {filepath}. "
            "Expected output from localize_hub_event_3d.py."
        )
    return rows


def _load_misfit_plot_inputs_from_group(filepath, group_name):
    """
    Load grids, misfit volume, best-fit map coordinates, and grid indices for
    ``plot_misfit_distribution``.
    """
    with File(filepath, "r") as f:
        g = f[group_name]
        east = float(g.attrs["east"])
        north = float(g.attrs["north"])
        depth = float(g.attrs["depth"])
        min_misfit = float(g.attrs["min_misfit"])
        scale_factor = float(g.attrs["scale_factor"])

        vol = g["misfit_volume"]
        easts_grid = vol["east_grid"][:]
        norths_grid = vol["north_grid"][:]
        depths_grid = vol["depth_grid"][:]
        misfit_vol = vol["misfit"][:]

        arrival = g["arrival_time"]
        stations = list(arrival.attrs.keys())

    i_east = int(np.argmin(np.abs(easts_grid - east)))
    i_north = int(np.argmin(np.abs(norths_grid - north)))
    i_depth = int(np.argmin(np.abs(depths_grid - depth)))

    coord_df = get_geophone_coords()
    coord_df = coord_df[coord_df.index.isin(stations)]
    coord_df = coord_df.reset_index(drop=False)
    if "name" not in coord_df.columns:
        first = coord_df.columns[0]
        coord_df = coord_df.rename(columns={first: "name"})

    return {
        "misfit_vol": misfit_vol,
        "easts_grid": easts_grid,
        "norths_grid": norths_grid,
        "depths_grid": depths_grid,
        "i_east": i_east,
        "i_north": i_north,
        "i_depth": i_depth,
        "station_df": coord_df,
        "east": east,
        "north": north,
        "depth": depth,
        "min_misfit": min_misfit,
        "scale_factor": scale_factor,
        "group_name": group_name,
    }


def main():
    parser = ArgumentParser(
        description=(
            "Plot misfit distribution at the velocity scale factor that minimizes "
            "min_misfit in localize_hub_event_3d.py HDF5 output."
        )
    )
    parser.add_argument("--group_label", type=int, required=True, help="Hub group label")
    parser.add_argument(
        "--filepath",
        type=str,
        default=None,
        help="Override path to hub_event_location_info_group*.h5 (default: LOC_DIR)",
    )
    parser.add_argument(
        "--misfitmax",
        type=float,
        default=0.01,
        help="Upper color scale for misfit (seconds), passed to plot_misfit_distribution",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help="Output figure filename (default: derived from group_label and best scale)",
    )
    args = parser.parse_args()

    if args.filepath is not None:
        filepath = Path(args.filepath)
    else:
        filepath = Path(dirpath_loc) / f"hub_event_location_info_group{args.group_label:d}.h5"

    if not filepath.is_file():
        raise FileNotFoundError(filepath)

    rows = _list_hub_scale_groups(filepath)
    best_row = min(rows, key=lambda r: r["min_misfit"])
    data = _load_misfit_plot_inputs_from_group(filepath, best_row["group"])

    print("--------------------------------")
    print(f"File: {filepath}")
    print(f"HDF5 group: {data['group_name']}")
    print(f"Best velocity scale factor: {data['scale_factor']:.4f}")
    print(f"Minimum misfit (stored): {data['min_misfit']:.6f} s")
    print(f"Best location: {data['east']:.1f} m E, {data['north']:.1f} m N, {data['depth']:.1f} m depth")
    print("All scales (group, scale_factor, min_misfit):")
    for r in sorted(rows, key=lambda x: x["scale_factor"]):
        mark = " <-- best" if r["group"] == best_row["group"] else ""
        print(f"  {r['group']}: scale={r['scale_factor']:.4f}, min_misfit={r['min_misfit']:.6f} s{mark}")
    print("--------------------------------")

    title = (
        f"Group {args.group_label:d}, best scale factor {data['scale_factor']:.4f} "
        f"(min misfit {data['min_misfit']:.4f} s)"
    )
    fig, ax_map, ax_profile, cbar = plot_misfit_distribution(
        data["misfit_vol"],
        data["easts_grid"],
        data["norths_grid"],
        data["depths_grid"],
        data["i_east"],
        data["i_north"],
        data["i_depth"],
        data["station_df"],
        misfitmax=args.misfitmax,
        title=title,
    )

    if args.figname is not None:
        figname = args.figname
    else:
        figname = (
            f"misfit_distribution_group{args.group_label:d}_best_scale"
            f"{data['scale_factor']:.2f}.png"
        )
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
