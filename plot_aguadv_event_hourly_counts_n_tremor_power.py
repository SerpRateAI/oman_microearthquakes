"""
Plot repeated STA/LTA detections, subarray hourly event counts, and harmonic
tremor mode power (AGU Advances).

Default four-panel figure: A04/B04 repeated STA/LTA hourly counts on top,
Subarray A and B hourly event counts next, hourly precipitation and relative
water table, then harmonic mode power on bottom.
Use --no-env-panel to omit the precipitation/water-table panel (three-panel figure).
"""

from argparse import ArgumentParser
from os.path import join
from pathlib import Path

from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure
from pandas import Timestamp, date_range, read_csv, read_hdf, to_datetime

from utils_basic import (
    DETECTION_DIR,
    ENDTIME_GEO,
    HAMMER_DAY,
    SPECTROGRAM_DIR,
    STARTTIME_GEO,
    get_precip_data,
    get_sunrise_sunset_times,
    get_water_level_data,
)
from utils_plot import (
    add_day_night_shading,
    format_datetime_xlabels,
    format_db_ylabels,
    save_figure,
)
from utils_spec import get_spec_peak_file_suffix, get_spectrogram_file_suffix

DEFAULT_SUFFIX = (
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10_num_sim_sta3"
)
DEFAULT_REPEATING_SUFFIX = (
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10"
)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot repeated STA/LTA detections, subarray hourly event counts, "
            "and harmonic tremor mode power."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_SUFFIX,
        help="Suffix for subarray hourly-count CSV files",
    )
    parser.add_argument(
        "--repeating-suffix",
        type=str,
        default=DEFAULT_REPEATING_SUFFIX,
        help="Suffix for repeating STA/LTA detection CSV files",
    )
    parser.add_argument(
        "--detection-dir",
        type=Path,
        default=Path(DETECTION_DIR),
        help="Directory containing hourly_counts_subarray_*.csv files",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default=None,
        help=(
            "Output figure filename "
            "(default: aguadv_event_hourly_counts_n_tremor_power_<station>.png)"
        ),
    )
    parser.add_argument(
        "--color_mode_face",
        type=str,
        required=True,
        help="The color for the face of the harmonic mode",
    )
    parser.add_argument(
        "--color_mode_edge",
        type=str,
        required=True,
        help="The color for the edge of the harmonic mode",
    )
    parser.add_argument(
        "--base_mode_name",
        type=str,
        default="PR02549",
        help="The base mode name",
    )
    parser.add_argument(
        "--base_mode_order",
        type=int,
        default=2,
        help="The base mode order",
    )
    parser.add_argument(
        "--mode_order",
        type=int,
        default=3,
        help="The mode order",
    )
    parser.add_argument(
        "--window_length",
        type=float,
        default=300.0,
        help="The window length in seconds",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="The overlap fraction",
    )
    parser.add_argument(
        "--min_prom",
        type=float,
        default=15.0,
        help="The minimum prominence",
    )
    parser.add_argument(
        "--min_rbw",
        type=float,
        default=15.0,
        help="The minimum reverse bandwidth",
    )
    parser.add_argument(
        "--max_mean_db",
        type=float,
        default=15.0,
        help="The maximum mean dB value",
    )
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Station whose harmonic tremor mode power is plotted",
    )
    parser.add_argument(
        "--no-env-panel",
        action="store_true",
        help=(
            "Omit the precipitation and water-table panel "
            "(three-panel figure of detections, event counts, and mode power)"
        ),
    )
    return parser.parse_args()


def load_hourly_precip(starttime, endtime):
    precip_df = get_precip_data()
    precip_df = precip_df.loc[starttime:endtime].sort_index()
    if precip_df.empty:
        raise ValueError("No precipitation data in the requested time range.")

    return precip_df["precipitation_mm"]


def load_water_table(starttime, endtime):
    water_level_df = get_water_level_data()
    water_level_df = water_level_df.loc[starttime:endtime].sort_index()
    if water_level_df.empty:
        raise ValueError("No water-table data in the requested time range.")

    return water_level_df["water_table"]


def load_repeating_detection_hourly_counts(detection_dir, repeating_suffix, station):
    """Bin repeating STA/LTA detections into hourly counts over the geo window."""
    filename = f"sta_lta_detections_repeating_{repeating_suffix}_{station}.csv"
    filepath = detection_dir / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing repeating detections file: {filepath}")

    detection_df = read_csv(filepath, parse_dates=["starttime", "endtime"])
    detection_df["starttime"] = to_datetime(detection_df["starttime"], utc=False)

    start = Timestamp(STARTTIME_GEO)
    end = Timestamp(ENDTIME_GEO)
    geo_tz = start.tz

    if geo_tz is not None and detection_df["starttime"].dt.tz is None:
        detection_df["starttime"] = detection_df["starttime"].dt.tz_localize(geo_tz)

    # Remove Jan 25 local daytime detections (sunrise–sunset; possible hammer shots)
    sun_df = get_sunrise_sunset_times()
    hammer_day = Timestamp(HAMMER_DAY).normalize()
    sun_row = sun_df.loc[sun_df["day"].dt.normalize() == hammer_day]
    if sun_row.empty:
        raise ValueError(f"No sunrise/sunset times for hammer day {hammer_day.date()}")
    daytime_start = Timestamp(sun_row["sunrise"].iloc[0])
    daytime_end = Timestamp(sun_row["sunset"].iloc[0])
    detection_df = detection_df.loc[
        (detection_df["starttime"] < daytime_start)
        | (detection_df["starttime"] > daytime_end)
    ]

    df_win = detection_df.loc[
        (detection_df["starttime"] >= start) & (detection_df["starttime"] < end)
    ].copy()

    hour_index = date_range(
        start=start.floor("h"),
        end=end.floor("h"),
        freq="h",
        tz=geo_tz,
    )

    hourly_counts = (
        df_win.assign(hour=df_win["starttime"].dt.floor("h"))
        .groupby("hour")
        .size()
        .reindex(hour_index, fill_value=0)
        .rename_axis("hour")
        .reset_index(name="count")
    )
    return hourly_counts


def plot_repeating_detection_hourly_counts(ax_det, detection_dir, repeating_suffix):
    """Plot hourly repeated STA/LTA detection counts for A04 and B04."""
    print("Plotting the hourly repeated STA/LTA detection counts...")
    for station, color in (("A04", "dodgerblue"), ("B04", "darkorange")):
        count_df = load_repeating_detection_hourly_counts(
            detection_dir, repeating_suffix, station
        )
        ax_det.plot(
            count_df["hour"],
            count_df["count"],
            color=color,
            linestyle="--",
            label=station,
        )

    print("Adding the day and night shading...")
    add_day_night_shading(ax_det)

    ax_det.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    ax_det.set_ylim(1, 5000)
    format_datetime_xlabels(
        ax_det,
        plot_axis_label=False,
        plot_tick_label=False,
        major_tick_spacing="5d",
        num_minor_ticks=4,
        date_format="%Y-%m-%d",
        axis_label_size=12,
        tick_label_size=10,
    )
    ax_det.set_ylabel("Number of detections per hour", fontsize=12)
    ax_det.set_yscale("log")
    ax_det.set_title("Hourly repeated STA/LTA detections", fontweight="bold")
    ax_det.legend(loc="upper right", fontsize=10, framealpha=1.0)


def plot_subarray_hourly_counts(ax_event, detection_dir, suffix):
    """Plot hourly event counts for subarrays A and B."""
    print("Plotting the hourly event counts...")
    for subarray, color in (("a", "dodgerblue"), ("b", "darkorange")):
        filename = f"hourly_counts_subarray_{subarray}_{suffix}.csv"
        filepath = detection_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"Missing hourly counts file: {filepath}")

        count_df = read_csv(filepath, parse_dates=["hour"])
        label = f"Subarray {subarray.upper()}"
        ax_event.plot(
            count_df["hour"],
            count_df["count"],
            color=color,
            label=label,
        )

    print("Adding the day and night shading...")
    add_day_night_shading(ax_event)

    ax_event.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    ax_event.set_ylim(1, 500)
    format_datetime_xlabels(
        ax_event,
        plot_axis_label=False,
        plot_tick_label=False,
        major_tick_spacing="5d",
        num_minor_ticks=4,
        date_format="%Y-%m-%d",
        axis_label_size=12,
        tick_label_size=10,
    )
    ax_event.set_ylabel("Number of events per hour", fontsize=12)
    ax_event.set_yscale("log")
    ax_event.set_title("Hourly event counts", fontweight="bold")
    ax_event.legend(loc="upper right", fontsize=10, framealpha=1.0)


def plot_precipitation_and_water_table(ax_env):
    """Plot hourly precipitation and relative water table on twin axes."""
    print("Plotting the hourly precipitation and relative water table...")
    hourly_precip = load_hourly_precip(STARTTIME_GEO, ENDTIME_GEO)
    water_table = load_water_table(STARTTIME_GEO, ENDTIME_GEO)
    ax_water = ax_env.twinx()

    ax_env.plot(
        hourly_precip.index,
        hourly_precip.values,
        color="deepskyblue",
        linewidth=2.5,
        label="_nolegend_",
        zorder=2,
    )
    ax_water.plot(
        water_table.index,
        water_table.values,
        color="royalblue",
        linewidth=2.5,
        label="_nolegend_",
        zorder=1,
    )

    print("Adding the day and night shading...")
    add_day_night_shading(ax_env)

    ax_env.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    ax_water.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    format_datetime_xlabels(
        ax_env,
        plot_axis_label=False,
        plot_tick_label=False,
        major_tick_spacing="5d",
        num_minor_ticks=4,
        date_format="%Y-%m-%d",
        axis_label_size=12,
        tick_label_size=10,
    )
    ax_env.set_ylabel("Hourly precipitation (mm)", color="black", fontsize=12)
    ax_water.set_ylabel("Relative water table (m)", color="black", fontsize=12)
    ax_env.tick_params(axis="y", colors="black", labelsize=10)
    ax_water.tick_params(axis="y", colors="black", labelsize=10)
    ax_env.set_title(
        "Precipitation and water table",
        fontweight="bold",
    )

    precip_legend = Line2D(
        [0],
        [0],
        color="deepskyblue",
        linewidth=2.5,
        label="Precipitation",
    )
    water_table_legend = Line2D(
        [0],
        [0],
        color="royalblue",
        linewidth=2.5,
        label="Water table",
    )
    ax_env.legend(
        [precip_legend, water_table_legend],
        ["Precipitation", "Water table"],
        loc="lower right",
        fontsize=10,
        framealpha=1.0,
    )


def plot_harmonic_mode_power(ax_spec, args):
    """Plot harmonic tremor mode power (same as reference script)."""
    print("Plotting the harmonic mode power...")

    filename = (
        f"stationary_harmonic_series_{args.base_mode_name}_base"
        f"{args.base_mode_order}.csv"
    )
    filepath = join(SPECTROGRAM_DIR, filename)
    harmonic_df = read_csv(filepath)
    mode_name = harmonic_df.loc[
        harmonic_df["mode_order"] == args.mode_order, "mode_name"
    ].values[0]

    suffix_spec = get_spectrogram_file_suffix(args.window_length, args.overlap)
    suffix_peak = get_spec_peak_file_suffix(
        args.min_prom, args.min_rbw, args.max_mean_db
    )
    filename_in = (
        f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
    )
    inpath = join(SPECTROGRAM_DIR, filename_in)
    resonance_df = read_hdf(inpath, key="properties")
    resonance_df = resonance_df[resonance_df["station"] == args.station]

    ax_spec.scatter(
        resonance_df["time"],
        resonance_df["total_power"],
        marker="o",
        s=15,
        color=args.color_mode_face,
        edgecolors=args.color_mode_edge,
        label=mode_name,
        zorder=2,
    )

    print("Adding the day and night shading...")
    add_day_night_shading(ax_spec)

    ax_spec.set_xlim(STARTTIME_GEO, ENDTIME_GEO)
    format_datetime_xlabels(
        ax_spec,
        plot_axis_label=True,
        plot_tick_label=True,
        major_tick_spacing="5d",
        num_minor_ticks=4,
        date_format="%Y-%m-%d",
        axis_label_size=12,
        tick_label_size=10,
    )
    format_db_ylabels(ax_spec, axis_label_size=12, tick_label_size=10)
    ax_spec.set_title(f"Mode {args.mode_order} power on {args.station}", fontweight="bold")


def main():
    args = parse_args()
    plot_env_panel = not args.no_env_panel
    detection_dir = args.detection_dir
    suffix = args.suffix
    repeating_suffix = args.repeating_suffix

    if plot_env_panel:
        fig = figure(figsize=(10, 16))
        gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.15)
        ax_det = fig.add_subplot(gs[0])
        ax_event = fig.add_subplot(gs[1])
        ax_env = fig.add_subplot(gs[2])
        ax_spec = fig.add_subplot(gs[3])
        panel_axes = (ax_det, ax_event, ax_env, ax_spec)
        panel_labels = ("(a)", "(b)", "(c)", "(d)")
    else:
        fig = figure(figsize=(10, 12))
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.15)
        ax_det = fig.add_subplot(gs[0])
        ax_event = fig.add_subplot(gs[1])
        ax_spec = fig.add_subplot(gs[2])
        panel_axes = (ax_det, ax_event, ax_spec)
        panel_labels = ("(a)", "(b)", "(c)")

    plot_repeating_detection_hourly_counts(ax_det, detection_dir, repeating_suffix)
    plot_subarray_hourly_counts(ax_event, detection_dir, suffix)
    if plot_env_panel:
        plot_precipitation_and_water_table(ax_env)
    plot_harmonic_mode_power(ax_spec, args)

    for ax, label in zip(panel_axes, panel_labels):
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )

    figname = (
        args.figname
        or f"aguadv_event_hourly_counts_n_tremor_power.png"
    )
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
