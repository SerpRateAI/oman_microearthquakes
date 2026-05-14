"""
Plot minimum grid-search misfit and best-fit depth versus velocity scale factor from a
multi-scale hub location HDF5 file (``hub_event_location_info_group*.h5``).

This script calls ``plot_hub_event_min_misfit_and_depth_vs_scale_factor`` in ``utils_loc.py``,
which reads root attribute ``scale_factors`` and groups ``scale_factor_{i}``.
"""

from argparse import ArgumentParser
from pathlib import Path

from utils_basic import LOC_DIR as dirpath_loc
from utils_basic import DETECTION_DIR as dirpath_event
from utils_loc import plot_hub_event_min_misfit_and_depth_vs_scale_factor
from pandas import read_csv
from utils_plot import save_figure
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_basic import get_freq_limits_string


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Plot min misfit and depth vs velocity scale factor from a hub multi-scale location HDF5 file",
    )
    parser.add_argument("--group_label", type=int, required=True, help="Group label (used for default HDF5 path and output name)")
    parser.add_argument(
        "--filepath",
        type=Path,
        default=None,
        help="Path to hub_event_location_info_group*.h5 (default: LOC_DIR/hub_event_location_info_group{label}.h5)",
    )
    parser.add_argument("--figwidth", type=float, default=8.0)
    parser.add_argument("--figheight", type=float, default=5.5)
    parser.add_argument("--marker_size", type=float, default=7.0)
    parser.add_argument("--min_freq_filter", type=float, default=20.0)
    parser.add_argument("--max_freq_filter", type=float, default=None)
    parser.add_argument("--min_cc", type=float, default=0.85)
    parser.add_argument("--min_num_similar_snippet", type=int, default=10)
    parser.add_argument("--min_num_similar_station", type=int, default=3)
    parser.add_argument("--sta_window_sec", type=float, default=0.005)
    parser.add_argument("--lta_window_sec", type=float, default=0.05)
    parser.add_argument("--thr_on", type=float, default=4.0)
    parser.add_argument("--thr_off", type=float, default=1.0)

    args = parser.parse_args()
    group_label = args.group_label
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off

    # Get the hub event information
    print("Getting the hub event information...")

    # Build the suffix
    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    # Load the group information
    filename = f"event_group_info_{suffix_group}.csv"
    filepath = Path(dirpath_event) / filename
    group_df = read_csv(filepath)
    id_hub = group_df.loc[group_df["label"] == group_label, "id_hub"].values[0]
    print(f"Hub event ID: {id_hub}")

    # Plot the minimum misfit and depth vs scale factor
    filepath_h5 = Path(dirpath_loc) / f"hub_event_location_info_group{group_label:d}.h5"
    fig, _, _ = plot_hub_event_min_misfit_and_depth_vs_scale_factor(
        filepath_h5,
        title=f"Group {group_label:d}, Hub Event {id_hub}",
    )

    out_name = f"hub_event_min_misfit_depth_vs_scale_factor_group{group_label:d}.png"
    save_figure(fig, out_name)
