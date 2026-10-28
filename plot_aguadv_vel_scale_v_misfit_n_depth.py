"""
Plot minimum grid-search misfit and best-fit depth versus velocity scale factor for hub
event groups 18 and 19 (Subarray A) and 20 (Subarray B) in a 2×2 figure (AGU Advances).

Reads ``hub_event_location_info_group{N}.h5`` from ``localize_hub_event_3d_per_scale_factor.py``.
Each panel uses the scale factors stored in its localization HDF5 file; Subarray A panels (18, 19)
share the x-axis, while Subarray B (20) has an independent x-axis. Panel (d) shows preferred 1D
models from ``linear_vel_model_params_a.csv`` with separate scale factors for subarrays A and B.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import subplots
from numpy import linspace
from pandas import read_csv

from utils_basic import DETECTION_DIR as dirpath_event
from utils_basic import LOC_DIR as dirpath_loc
from utils_basic import VEL_MODEL_DIR as dirpath_vel
from utils_basic import get_freq_limits_string
from utils_cc import get_repeating_snippet_suffix
from utils_loc import (
    load_hub_event_min_misfit_depth_vs_scale_factor_from_hdf,
    plot_hub_event_min_misfit_and_depth_vs_scale_factor_arrays,
)
from utils_plot import WAVE_VELOCITY_UNIT, save_figure
from utils_sta_lta import get_sta_lta_suffix


GROUP_AXES = {18: (0, 0), 19: (1, 0), 20: (0, 1)}
PANEL_LABELS = {18: "a", 19: "b", 20: "c"}
GROUP_SUBARRAY = {18: "A", 19: "A", 20: "B"}


def hub_event_hdf_path(group_label: int) -> Path:
    return Path(dirpath_loc) / f"hub_event_location_info_group{group_label:d}.h5"


def load_linear_vel_model_params():
    """Unscaled v0 and alpha from linear_vel_model_params_a.csv."""
    vel_path = Path(dirpath_vel) / "linear_vel_model_params_a.csv"
    vel_params_df = read_csv(vel_path)
    surface_vel = float(vel_params_df.loc[0, "surface_vel"])
    vel_gradient = float(vel_params_df.loc[0, "vel_gradient"])
    return surface_vel, vel_gradient


def velocity_profile(surface_vel, vel_gradient, depths):
    """v(z) = v0 + alpha * z with depth positive downward."""
    return surface_vel + vel_gradient * depths


def plot_preferred_velocity_models(
    ax,
    scale_factor_a_prefer,
    scale_factor_b_prefer,
    max_depth,
):
    """Plot v(z) for the initial model and preferred scaled models for subarrays A and B."""
    surface_vel, vel_gradient = load_linear_vel_model_params()
    depths = linspace(0.0, max_depth, 200)

    v_initial = velocity_profile(surface_vel, vel_gradient, depths)
    v0_a = surface_vel * scale_factor_a_prefer
    alpha_a = vel_gradient * scale_factor_a_prefer
    v_subarray_a = velocity_profile(v0_a, alpha_a, depths)

    v0_b = surface_vel * scale_factor_b_prefer
    alpha_b = vel_gradient * scale_factor_b_prefer
    v_subarray_b = velocity_profile(v0_b, alpha_b, depths)

    ax.plot(
        v_initial,
        depths,
        color="gray",
        linestyle="--",
        linewidth=2.0,
        label="Initial",
        zorder=1,
    )
    ax.plot(
        v_subarray_a,
        depths,
        color="#6baed6",
        linewidth=2.0,
        label=f"Subarray A (scale = {scale_factor_a_prefer:.1f})",
        zorder=2,
    )
    ax.plot(
        v_subarray_b,
        depths,
        color="#08519c",
        linewidth=2.0,
        label=f"Subarray B (scale = {scale_factor_b_prefer:.1f})",
        zorder=2,
    )

    ax.set_xlabel(f"Velocity ({WAVE_VELOCITY_UNIT})", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_ylim(max_depth, 0.0)
    ax.set_title("Preferred 1D velocity models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=1.0)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Plot min misfit and depth vs velocity scale factor for groups 18, 19 "
            "(Subarray A) and 20 (Subarray B) (AGU Advances)"
        ),
    )
    parser.add_argument("--figwidth", type=float, default=12.0)
    parser.add_argument("--figheight", type=float, default=10.0)
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
    parser.add_argument("--min_misfit_ax", type=float, default=0.0008, help="Left y-axis lower limit (min. misfit, s)")
    parser.add_argument("--max_misfit_ax", type=float, default=0.0022, help="Left y-axis upper limit (min. misfit, s)")
    parser.add_argument("--min_depth_ax", type=float, default=0.0, help="Right y-axis lower limit (event depth, m)")
    parser.add_argument("--max_depth_ax", type=float, default=30.0, help="Right y-axis upper limit (event depth, m)")
    parser.add_argument(
        "--scale_factor_a_prefer",
        type=float,
        default=0.5,
        help="Preferred scale factor for Subarray A (groups 18–19; crimson line and panel d)",
    )
    parser.add_argument(
        "--scale_factor_b_prefer",
        type=float,
        default=0.5,
        help="Preferred scale factor for Subarray B (group 20; crimson line and panel d)",
    )

    args = parser.parse_args()

    figwidth = args.figwidth
    figheight = args.figheight
    marker_size = args.marker_size
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    min_cc = args.min_cc
    min_num_similar_snippet = args.min_num_similar_snippet
    min_num_similar_station = args.min_num_similar_station
    sta_window_sec = args.sta_window_sec
    lta_window_sec = args.lta_window_sec
    thr_on = args.thr_on
    thr_off = args.thr_off
    min_misfit_ax = args.min_misfit_ax
    max_misfit_ax = args.max_misfit_ax
    min_depth_ax = args.min_depth_ax
    max_depth_ax = args.max_depth_ax
    scale_factor_a_prefer = args.scale_factor_a_prefer
    scale_factor_b_prefer = args.scale_factor_b_prefer

    suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
    suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    suffix_repeating_snippet = get_repeating_snippet_suffix(
        min_cc=min_cc, min_num_similar=min_num_similar_snippet,
    )
    suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
    suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

    print("Getting the hub event information...")
    filename = f"event_group_info_{suffix_group}.csv"
    filepath = Path(dirpath_event) / filename
    group_df = read_csv(filepath)

    fig, axes = subplots(
        2, 2,
        figsize=(figwidth, figheight),
        sharex=False,
        gridspec_kw={"hspace": 0.25, "wspace": 0.35},
    )
    # Share x-axis only between Subarray A panels (groups 18 and 19), not with Subarray B.
    axes[1, 0].sharex(axes[0, 0])

    for group_label, (row, col) in GROUP_AXES.items():
        ax = axes[row, col]
        subarray = GROUP_SUBARRAY[group_label]
        id_hub = group_df.loc[group_df["label"] == group_label, "id_hub"].values[0]
        print(f"Group {group_label:d}, Hub event ID: {id_hub}, Subarray {subarray}")

        filepath_h5 = hub_event_hdf_path(group_label)
        scale_factors, min_misfits, depths = load_hub_event_min_misfit_depth_vs_scale_factor_from_hdf(
            filepath_h5,
        )

        title = f"Group {group_label:d}, Subarray {subarray}"
        show_xlabel = (row, col) == (1, 0) or (row, col) == (0, 1)

        _, ax_misfit, ax_depth = plot_hub_event_min_misfit_and_depth_vs_scale_factor_arrays(
            scale_factors,
            min_misfits,
            depths,
            title=title,
            marker_size=marker_size,
            ax=ax,
            show_xlabel=show_xlabel,
            misfit_ylim=(min_misfit_ax, max_misfit_ax),
            depth_ylim=(min_depth_ax, max_depth_ax),
        )

        if group_label == 18:
            ax_depth.set_ylabel("")
        elif group_label == 20:
            ax_misfit.set_ylabel("")

        ax.text(
            -0.15, 1.08, f"({PANEL_LABELS[group_label]})",
            fontsize=14, fontweight="bold", transform=ax.transAxes, ha="right", va="top",
        )

        scale_factor_prefer = (
            scale_factor_a_prefer if subarray == "A" else scale_factor_b_prefer
        )
        ax.axvline(
            scale_factor_prefer,
            color="crimson",
            linestyle="--",
            linewidth=2.0,
            zorder=1,
        )

    ax_vel = axes[1, 1]
    plot_preferred_velocity_models(
        ax_vel,
        scale_factor_a_prefer,
        scale_factor_b_prefer,
        max_depth_ax,
    )
    ax_vel.text(
        -0.15, 1.08, "(d)",
        fontsize=14, fontweight="bold", transform=ax_vel.transAxes, ha="right", va="top",
    )

    save_figure(fig, "aguadv_vel_scale_v_misfit_n_depth.png")
