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

from numpy import argmin, abs
from h5py import File

from utils_basic import (
    EASTMAX_A_MIDDLE,
    EASTMAX_B_MIDDLE,
    EASTMIN_A_MIDDLE,
    EASTMIN_B_MIDDLE,
    LOC_DIR as dirpath_loc,
    NORTHMAX_A_MIDDLE,
    NORTHMAX_B_MIDDLE,
    NORTHMIN_A_MIDDLE,
    NORTHMIN_B_MIDDLE,
    get_geophone_coords,
)
from utils_loc import plot_misfit_distribution
from utils_plot import save_figure


def _infer_subarray_letter(stations):
    """Return ``'A'`` or ``'B'`` from geophone names (first character), matching hub localization."""
    letters = {str(s)[0].upper() for s in stations if len(str(s)) >= 1}
    if not letters:
        raise ValueError("No stations available to infer subarray.")
    if letters - {"A", "B"}:
        raise ValueError(
            f"Expected only A- and B-line stations; got first-character set {letters!r}."
        )
    if len(letters) != 1:
        raise ValueError(
            f"Stations span multiple subarray letters {letters!r}; cannot pick one map extent."
        )
    return letters.pop()


def _middle_axis_limits_for_subarray(subarray_letter):
    """East/north limits (m) for map view: ``*_MIDDLE`` constants from ``utils_basic``."""
    if subarray_letter == "A":
        return (
            EASTMIN_A_MIDDLE,
            EASTMAX_A_MIDDLE,
            NORTHMIN_A_MIDDLE,
            NORTHMAX_A_MIDDLE,
        )
    if subarray_letter == "B":
        return (
            EASTMIN_B_MIDDLE,
            EASTMAX_B_MIDDLE,
            NORTHMIN_B_MIDDLE,
            NORTHMAX_B_MIDDLE,
        )
    raise ValueError(f"Unsupported subarray letter: {subarray_letter!r}")


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
    ``plot_misfit_distribution``. After plotting, east/north axis limits on the map and the east
axis on the depth profile are set from ``EAST*_*_MIDDLE`` / ``NORTH*_*_MIDDLE`` in
``utils_basic.py`` according to the subarray inferred from station names (``A`` vs ``B`` line).
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

    i_east = int(argmin(abs(easts_grid - east)))
    i_north = int(argmin(abs(norths_grid - north)))
    i_depth = int(argmin(abs(depths_grid - depth)))

    coord_df = get_geophone_coords()
    coord_df = coord_df[coord_df.index.isin(stations)]
    coord_df = coord_df.reset_index(drop=False)
    if "name" not in coord_df.columns:
        first = coord_df.columns[0]
        coord_df = coord_df.rename(columns={first: "name"})

    subarray_letter = _infer_subarray_letter(stations)

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
        "stations": stations,
        "subarray_letter": subarray_letter,
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
    print(f"Inferred subarray: {data['subarray_letter']} (from station names)")
    print("All scales (group, scale_factor, min_misfit):")
    for r in sorted(rows, key=lambda x: x["scale_factor"]):
        mark = " <-- best" if r["group"] == best_row["group"] else ""
        print(f"  {r['group']}: scale={r['scale_factor']:.4f}, min_misfit={r['min_misfit']:.6f} s{mark}")
    print("--------------------------------")

    title = (
        f"Group {args.group_label:d}, best scale factor {data['scale_factor']:.4f} "
        f"(min misfit {data['min_misfit']:.4f} s)"
    )
    east_lo, east_hi, north_lo, north_hi = _middle_axis_limits_for_subarray(data["subarray_letter"])
    fig, _, _, _ = plot_misfit_distribution(
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
        east_lim=(east_lo, east_hi),
        north_lim=(north_lo, north_hi),
    )
    print(
        f"Map axis limits (subarray {data['subarray_letter']} middle): "
        f"east [{east_lo:g}, {east_hi:g}] m, north [{north_lo:g}, {north_hi:g}] m"
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
