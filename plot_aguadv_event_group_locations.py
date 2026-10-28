"""
Plot horizontal locations of selected hub-event groups on Subarrays A and B (AGU Advances).

Four map panels in a 2×2 layout: the top row shows inner and middle geophones and boreholes in
color; the bottom row adds event groups as diamonds (log-scaled ``hot`` colormap) with stations and
boreholes drawn as empty markers. Panels (c) and (d) include square event-depth histogram insets
in the upper-left corner.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib import colormaps
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
from numpy import arange, log10
from pandas import DataFrame, read_csv

from plot_aguadv_event_depth_histograms import (
    BIN_WIDTH,
    MAX_DEPTH,
    MIN_DEPTH,
    SUBARRAY_COLORS,
    plot_depth_histogram,
)

from utils_basic import (
    LOC_DIR,
    PLOTTING_DIR,
    EASTMIN_A_MIDDLE,
    EASTMAX_A_MIDDLE,
    NORTHMIN_A_MIDDLE,
    NORTHMAX_A_MIDDLE,
    EASTMIN_B_MIDDLE,
    EASTMAX_B_MIDDLE,
    NORTHMIN_B_MIDDLE,
    NORTHMAX_B_MIDDLE,
    INNER_STATIONS_A,
    INNER_STATIONS_B,
    MIDDLE_STATIONS_A,
    MIDDLE_STATIONS_B,
    get_borehole_coords,
    get_geophone_coords,
)
from utils_plot import (
    format_east_xlabels,
    format_north_ylabels,
    get_label_alignments,
    save_figure,
)

DEFAULT_STATION_LABEL_OFFSETS_CSV = Path(PLOTTING_DIR) / "station_label_offsets_event_group_location.csv"
DEFAULT_EVENT_GROUP_LABEL_COORDS_CSV = Path(PLOTTING_DIR) / "event_group_label_coords.csv"

DEFAULT_LOCATIONS_CSV = Path(
    "/proj/mazu/tianze.liu/oman/locations/"
    "hub_event_locations_preferred_scale_event_group_info_selected_"
    "freq20hz_sta5_lta50_on4_off1_cc0.85_num_sim_det10_num_sim_sta3.csv"
)

STATIONS_A = INNER_STATIONS_A + MIDDLE_STATIONS_A
STATIONS_B = INNER_STATIONS_B + MIDDLE_STATIONS_B

AXIS_LIMITS_A = (EASTMIN_A_MIDDLE, EASTMAX_A_MIDDLE, NORTHMIN_A_MIDDLE, NORTHMAX_A_MIDDLE)
AXIS_LIMITS_B = (EASTMIN_B_MIDDLE, EASTMAX_B_MIDDLE, NORTHMIN_B_MIDDLE, NORTHMAX_B_MIDDLE)

HAMMER_ID = "100129"
DEFAULT_HAMMER_LOCATIONS_CSV = Path(LOC_DIR) / "hammer_locations_3d.csv"
HAMMER_LABEL_EAST = 35.0
HAMMER_LABEL_NORTH = -75.0
HAMMER_LABEL_HA = "left"
HAMMER_LABEL_VA = "top"


def station_label_alignments(offset_x, offset_y):
    """Text alignment for station labels; extends get_label_alignments for zero offsets."""
    if offset_x == 0 and offset_y > 0:
        return "center", "bottom"
    if offset_x == 0 and offset_y < 0:
        return "center", "top"
    if offset_x > 0 and offset_y == 0:
        return "left", "center"
    if offset_x < 0 and offset_y == 0:
        return "right", "center"
    return get_label_alignments(offset_x, offset_y)


def load_event_group_label_coords(filepath):
    """Return label coordinates indexed by group label (int)."""
    df = read_csv(filepath)
    required = {"label", "east", "north", "va", "ha"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {filepath}; got {list(df.columns)}"
        )
    df = df.copy()
    df["label"] = df["label"].astype(int)
    return df.set_index("label")


def load_hammer_location(hammer_id, filepath):
    """Return (east, north) for a hammer shot from hammer_locations_3d.csv."""
    df = read_csv(filepath, dtype={"hammer_id": str})
    required = {"hammer_id", "east", "north"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {filepath}; got {list(df.columns)}"
        )
    matches = df.loc[df["hammer_id"] == hammer_id]
    if matches.empty:
        raise ValueError(f"Hammer {hammer_id} not found in {filepath}")
    row = matches.iloc[0]
    return float(row["east"]), float(row["north"])


def plot_hammer_location(
    ax,
    east,
    north,
    *,
    label_east,
    label_north,
    label_ha,
    label_va,
    label_size,
    size,
    linewidth,
    linewidth_label,
):
    """Plot a hammer-shot location with label and leader line."""
    ax.scatter(
        east,
        north,
        marker="*",
        s=size,
        color="deepskyblue",
        edgecolors="black",
        linewidths=linewidth,
        zorder=4,
    )
    ax.plot(
        [label_east, east],
        [label_north, north],
        color="black",
        linewidth=linewidth_label,
        zorder=3.5,
        clip_on=False,
    )
    ax.text(
        label_east,
        label_north,
        "Hammer 100129",
        fontsize=label_size,
        fontweight="bold",
        color="black",
        ha=label_ha,
        va=label_va,
        zorder=5,
        clip_on=False,
    )


def load_station_label_offsets(filepath):
    """Return {station: (offset_x, offset_y)} in map coordinates (m)."""
    df = read_csv(filepath)
    required = {"station", "offset_x", "offset_y"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {filepath}; got {list(df.columns)}"
        )
    return {
        str(row["station"]): (float(row["offset_x"]), float(row["offset_y"]))
        for _, row in df.iterrows()
    }


def log_scaled_marker_sizes(num_events, size_min, size_max, log_vmin, log_vmax):
    """Marker areas linear in log10(event count), using a shared log range."""
    log_vals = log10(num_events.astype(float))
    if log_vmax <= log_vmin:
        return size_min + 0.5 * (size_max - size_min)
    frac = (log_vals - log_vmin) / (log_vmax - log_vmin)
    return size_min + frac * (size_max - size_min)


def plot_stations_and_boreholes(
    ax,
    geo_df,
    boho_df,
    stations,
    axis_limits,
    station_label_offsets,
    *,
    hollow_markers,
    show_station_labels,
    station_size,
    borehole_size,
    color_geo,
    color_borehole,
    linewidth_marker,
    station_label_size,
):
    """Plot geophones and boreholes, filled or hollow."""
    min_east, max_east, min_north, max_north = axis_limits

    for station, coords in geo_df.iterrows():
        if station not in stations:
            continue
        if hollow_markers:
            ax.scatter(
                coords["east"],
                coords["north"],
                marker="^",
                s=station_size,
                facecolors="none",
                edgecolors="black",
                linewidths=linewidth_marker,
                zorder=2,
            )
        else:
            ax.scatter(
                coords["east"],
                coords["north"],
                marker="^",
                s=station_size,
                color=color_geo,
                edgecolors="black",
                linewidths=linewidth_marker,
                zorder=2,
            )
        if show_station_labels and station in station_label_offsets:
            offset_x, offset_y = station_label_offsets[station]
            ha, va = station_label_alignments(offset_x, offset_y)
            ax.text(
                coords["east"] + offset_x,
                coords["north"] + offset_y,
                station,
                fontsize=station_label_size,
                fontweight="bold",
                color="black",
                ha=ha,
                va=va,
                zorder=4,
            )

    for borehole, coords in boho_df.iterrows():
        east = coords["east"]
        north = coords["north"]
        if not (min_east <= east <= max_east and min_north <= north <= max_north):
            continue
        if hollow_markers:
            ax.scatter(
                east,
                north,
                marker="o",
                s=borehole_size,
                facecolors="none",
                edgecolors="black",
                linewidths=linewidth_marker,
                zorder=2,
            )
        else:
            ax.scatter(
                east,
                north,
                marker="o",
                s=borehole_size,
                color=color_borehole,
                edgecolors="black",
                linewidths=linewidth_marker,
                zorder=2,
            )


def plot_subarray_panel(
    ax,
    geo_df,
    boho_df,
    stations,
    groups_df,
    axis_limits,
    station_label_offsets,
    event_label_coords,
    *,
    background_color,
    hollow_markers,
    show_events,
    show_station_labels,
    station_size,
    borehole_size,
    color_geo,
    color_borehole,
    linewidth_marker,
    linewidth_event,
    event_size_min,
    event_size_max,
    log_vmin,
    log_vmax,
    norm,
    cmap,
    scale_bar_length,
    scale_bar_size_vertical,
    show_scale_bar,
    plot_east_axis_label,
    plot_north_axis_label,
    station_label_size,
    event_label_size,
    linewidth_event_label,
    axis_label_size,
    major_dist_spacing,
    tick_label_size,
    major_tick_length,
    minor_tick_length,
    tick_width,
    title,
):
    """Map panel with stations, boreholes, and optionally event groups."""
    min_east, max_east, min_north, max_north = axis_limits

    ax.set_facecolor(background_color)

    plot_stations_and_boreholes(
        ax,
        geo_df,
        boho_df,
        stations,
        axis_limits,
        station_label_offsets,
        hollow_markers=hollow_markers,
        show_station_labels=show_station_labels,
        station_size=station_size,
        borehole_size=borehole_size,
        color_geo=color_geo,
        color_borehole=color_borehole,
        linewidth_marker=linewidth_marker,
        station_label_size=station_label_size,
    )

    event_scatter = None
    if show_events and not groups_df.empty:
        sizes = log_scaled_marker_sizes(
            groups_df["num_events"],
            event_size_min,
            event_size_max,
            log_vmin,
            log_vmax,
        )
        event_scatter = ax.scatter(
            groups_df["east"],
            groups_df["north"],
            c=groups_df["num_events"],
            s=sizes,
            marker="D",
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=linewidth_event,
            zorder=3,
        )

        for _, row in groups_df.iterrows():
            label = int(row["label"])
            if label not in event_label_coords.index:
                continue
            label_row = event_label_coords.loc[label]
            label_east = float(label_row["east"])
            label_north = float(label_row["north"])
            event_east = float(row["east"])
            event_north = float(row["north"])
            ax.plot(
                [label_east, event_east],
                [label_north, event_north],
                color="black",
                linewidth=linewidth_event_label,
                zorder=3.5,
                clip_on=False,
            )
            ax.text(
                label_east,
                label_north,
                f"Group {label}",
                fontsize=event_label_size,
                fontweight="bold",
                color="black",
                ha=str(label_row["ha"]),
                va=str(label_row["va"]),
                zorder=5,
                clip_on=False,
            )

    if show_scale_bar:
        scale_bar = AnchoredSizeBar(
            ax.transData,
            scale_bar_length,
            f"{scale_bar_length:.0f} m",
            loc="lower right",
            bbox_transform=ax.transAxes,
            frameon=True,
            size_vertical=scale_bar_size_vertical,
            pad=0.5,
            fontproperties=FontProperties(size=axis_label_size),
        )
        ax.add_artist(scale_bar)

    ax.set_xlim(min_east, max_east)
    ax.set_ylim(min_north, max_north)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=axis_label_size + 2, fontweight="bold")

    format_east_xlabels(
        ax,
        plot_axis_label=plot_east_axis_label,
        major_tick_spacing=major_dist_spacing,
        axis_label_size=tick_label_size + 2,
        tick_label_size=tick_label_size,
        major_tick_length=major_tick_length,
        minor_tick_length=minor_tick_length,
        tick_width=tick_width,
    )
    format_north_ylabels(
        ax,
        plot_axis_label=plot_north_axis_label,
        major_tick_spacing=major_dist_spacing,
        axis_label_size=tick_label_size + 2,
        tick_label_size=tick_label_size,
        major_tick_length=major_tick_length,
        minor_tick_length=minor_tick_length,
        tick_width=tick_width,
    )

    return event_scatter


def add_depth_histogram_inset(
    parent_ax,
    groups_df,
    subarray,
    size_in,
    label_size,
    tick_label_size,
    bin_edges,
    left,
    plot_axis_labels,
):
    """Add a square event-depth histogram in the upper-left of a map panel.

    ``left`` is the inset's anchor as a fraction of the parent axes width, chosen
    so the depth-axis label lies on the map rather than in the margin.
    """
    ax = inset_axes(
        parent_ax,
        width=size_in,
        height=size_in,
        loc="upper left",
        bbox_to_anchor=(left, 0.0, 1.0, 1.0),
        bbox_transform=parent_ax.transAxes,
        borderpad=0.4,
    )
    ax.set_facecolor("white")
    ax.set_zorder(parent_ax.get_zorder() + 1)
    max_count = plot_depth_histogram(
        ax,
        groups_df["depth"].to_numpy(),
        groups_df["num_events"].to_numpy(),
        SUBARRAY_COLORS[subarray],
        subarray,
        bin_edges,
        plot_axis_labels,
        show_title=False,
        show_panel_label=False,
        plot_xlabel=plot_axis_labels,
        xlabel="Event count",
        label_size=label_size,
        tick_label_size=tick_label_size,
    )
    ax.set_yticks([0, 5, 10, 15])
    return ax, max_count


def main():
    parser = ArgumentParser(
        description="Plot hub-event group locations on Subarrays A and B (AGU Advances)."
    )
    parser.add_argument(
        "--locations_csv",
        type=Path,
        default=DEFAULT_LOCATIONS_CSV,
        help="CSV with preferred-scale hub-event group locations",
    )
    parser.add_argument(
        "--station_label_offsets_csv",
        type=Path,
        default=DEFAULT_STATION_LABEL_OFFSETS_CSV,
        help="CSV with per-station label offsets (station, offset_x, offset_y) in m",
    )
    parser.add_argument(
        "--event_group_label_coords_csv",
        type=Path,
        default=DEFAULT_EVENT_GROUP_LABEL_COORDS_CSV,
        help="CSV with event-group label positions (label, east, north, va, ha)",
    )
    parser.add_argument(
        "--hammer_locations_csv",
        type=Path,
        default=DEFAULT_HAMMER_LOCATIONS_CSV,
        help="CSV with 3D hammer locations (hammer_id, east, north, ...)",
    )
    parser.add_argument(
        "--hammer_id",
        type=str,
        default=HAMMER_ID,
        help="Hammer shot ID to mark on the Subarray A event-location panel",
    )
    parser.add_argument(
        "--hammer_size",
        type=float,
        default=400.0,
        help="Hammer marker size on the Subarray A event-location panel",
    )
    parser.add_argument(
        "--hammer_label_east",
        type=float,
        default=HAMMER_LABEL_EAST,
        help="East coordinate of the hammer label (m)",
    )
    parser.add_argument(
        "--hammer_label_north",
        type=float,
        default=HAMMER_LABEL_NORTH,
        help="North coordinate of the hammer label (m)",
    )
    parser.add_argument(
        "--hammer_label_ha",
        type=str,
        default=HAMMER_LABEL_HA,
        help="Horizontal alignment of the hammer label",
    )
    parser.add_argument(
        "--hammer_label_va",
        type=str,
        default=HAMMER_LABEL_VA,
        help="Vertical alignment of the hammer label",
    )
    parser.add_argument("--fig_width", type=float, default=15.0, help="Figure width (in)")
    parser.add_argument("--margin_x", type=float, default=0.06, help="Horizontal margin")
    parser.add_argument("--margin_y", type=float, default=0.08, help="Vertical margin")
    parser.add_argument("--hspace", type=float, default=0.05, help="Height between panel rows")
    parser.add_argument(
        "--cbar_width",
        type=float,
        default=0.02,
        help="Thickness of the horizontal colorbar",
    )
    parser.add_argument(
        "--cbar_pad",
        type=float,
        default=0.02,
        help="Gap between the bottom map panels and the colorbar",
    )
    parser.add_argument(
        "--background_color",
        type=str,
        default="0.85",
        help="Axes background color (solid gray)",
    )
    parser.add_argument("--station_size", type=float, default=150.0, help="Geophone marker size")
    parser.add_argument("--borehole_size", type=float, default=150.0, help="Borehole marker size")
    parser.add_argument("--event_size_min", type=float, default=30.0, help="Min event marker size")
    parser.add_argument("--event_size_max", type=float, default=400.0, help="Max event marker size")
    parser.add_argument("--linewidth_marker", type=float, default=1.0, help="Station edge width")
    parser.add_argument(
        "--linewidth_event",
        type=float,
        default=1.0,
        help="Event diamond edge line width",
    )
    parser.add_argument("--color_geo", type=str, default="gold", help="Geophone fill color")
    parser.add_argument("--color_borehole", type=str, default="violet", help="Borehole fill color")
    parser.add_argument(
        "--station_label_size",
        type=float,
        default=10.0,
        help="Font size of geophone station labels",
    )
    parser.add_argument(
        "--event_label_size",
        type=float,
        default=12.0,
        help="Font size of event-group labels",
    )
    parser.add_argument(
        "--linewidth_event_label",
        type=float,
        default=1.0,
        help="Line width for event-group label leader lines",
    )
    parser.add_argument("--scale_bar_length", type=float, default=10.0, help="Scale bar length (m)")
    parser.add_argument(
        "--scale_bar_size_vertical",
        type=float,
        default=0.5,
        help="Scale bar line width (half of site-map default)",
    )
    parser.add_argument("--axis_label_size", type=float, default=12.0)
    parser.add_argument("--tick_label_size", type=float, default=12.0)
    parser.add_argument("--major_dist_spacing", type=float, default=10.0)
    parser.add_argument("--major_tick_length", type=float, default=5.0)
    parser.add_argument("--minor_tick_length", type=float, default=2.0)
    parser.add_argument("--tick_width", type=float, default=1.0)
    parser.add_argument("--subplot_label_size", type=float, default=18.0)
    parser.add_argument("--subplot_offset_x", type=float, default=-0.02)
    parser.add_argument("--subplot_offset_y", type=float, default=0.01)
    parser.add_argument(
        "--depth_hist_size",
        type=float,
        default=1.7,
        help="Side length of the square depth-histogram insets (inches)",
    )
    parser.add_argument(
        "--depth_hist_tick_label_size",
        type=float,
        default=10.0,
        help="Tick label size of the depth-histogram insets",
    )
    parser.add_argument(
        "--depth_hist_left",
        type=float,
        default=0.22,
        help="Left edge of each depth-histogram inset as a fraction of the map width",
    )
    parser.add_argument(
        "--figname",
        type=str,
        default="aguadv_event_group_locations.png",
        help="Output figure filename",
    )
    args = parser.parse_args()

    groups_df = read_csv(args.locations_csv)
    required = {"subarray", "east", "north", "num_events", "label", "depth"}
    if not required.issubset(groups_df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {args.locations_csv}; "
            f"got {list(groups_df.columns)}"
        )

    geo_df = get_geophone_coords()
    boho_df = get_borehole_coords()
    station_label_offsets = load_station_label_offsets(args.station_label_offsets_csv)
    event_label_coords = load_event_group_label_coords(args.event_group_label_coords_csv)
    hammer_east, hammer_north = load_hammer_location(args.hammer_id, args.hammer_locations_csv)

    num_events = groups_df["num_events"].astype(float)
    if (num_events <= 0).any():
        raise ValueError("num_events must be positive for log scaling.")
    norm = LogNorm(vmin=num_events.min(), vmax=num_events.max())
    log_vmin = log10(num_events.min())
    log_vmax = log10(num_events.max())
    cmap = colormaps["hot"]

    groups_a = groups_df[groups_df["subarray"] == "A"]
    groups_b = groups_df[groups_df["subarray"] == "B"]
    empty_groups = DataFrame(columns=groups_df.columns)

    aspect_a = (NORTHMAX_A_MIDDLE - NORTHMIN_A_MIDDLE) / (EASTMAX_A_MIDDLE - EASTMIN_A_MIDDLE)
    aspect_b = (NORTHMAX_B_MIDDLE - NORTHMIN_B_MIDDLE) / (EASTMAX_B_MIDDLE - EASTMIN_B_MIDDLE)
    aspect_panel = max(aspect_a, aspect_b)

    cbar_label_room = 0.04
    cbar_gap = (args.margin_y + args.cbar_pad) / 2.0
    bottom_y = cbar_label_room + args.cbar_width + cbar_gap
    panel_height_frac = (1.0 - args.margin_y - bottom_y - args.hspace) / 2.0
    # Match the column gap to the row gap in inches, and size each panel box to the
    # map aspect so equal-aspect adjustment does not widen the column gap.
    span_x = 1.0 - 2.0 * args.margin_x
    gap_to_panel = args.hspace * aspect_panel / panel_height_frac
    wspace = gap_to_panel * span_x / (2.0 + gap_to_panel)
    panel_width_frac = (span_x - wspace) / 2.0
    fig_height = args.fig_width * aspect_panel * panel_width_frac / panel_height_frac

    fig = figure(figsize=(args.fig_width, fig_height))

    top_y = bottom_y + panel_height_frac + args.hspace
    left_x = args.margin_x
    right_x = args.margin_x + panel_width_frac + wspace

    ax_top_a = fig.add_axes([left_x, top_y, panel_width_frac, panel_height_frac])
    ax_top_b = fig.add_axes([right_x, top_y, panel_width_frac, panel_height_frac])
    ax_bot_a = fig.add_axes([left_x, bottom_y, panel_width_frac, panel_height_frac])
    ax_bot_b = fig.add_axes([right_x, bottom_y, panel_width_frac, panel_height_frac])

    panel_kwargs = dict(
        background_color=args.background_color,
        station_size=args.station_size,
        borehole_size=args.borehole_size,
        color_geo=args.color_geo,
        color_borehole=args.color_borehole,
        linewidth_marker=args.linewidth_marker,
        linewidth_event=args.linewidth_event,
        event_size_min=args.event_size_min,
        event_size_max=args.event_size_max,
        log_vmin=log_vmin,
        log_vmax=log_vmax,
        norm=norm,
        cmap=cmap,
        scale_bar_length=args.scale_bar_length,
        scale_bar_size_vertical=args.scale_bar_size_vertical,
        station_label_size=args.station_label_size,
        station_label_offsets=station_label_offsets,
        event_label_coords=event_label_coords,
        event_label_size=args.event_label_size,
        linewidth_event_label=args.linewidth_event_label,
        axis_label_size=args.axis_label_size,
        major_dist_spacing=args.major_dist_spacing,
        tick_label_size=args.tick_label_size,
        major_tick_length=args.major_tick_length,
        minor_tick_length=args.minor_tick_length,
        tick_width=args.tick_width,
    )

    scatter_top_a = plot_subarray_panel(
        ax_top_a,
        geo_df,
        boho_df,
        STATIONS_A,
        empty_groups,
        AXIS_LIMITS_A,
        hollow_markers=False,
        show_events=False,
        show_station_labels=True,
        show_scale_bar=False,
        plot_east_axis_label=False,
        plot_north_axis_label=True,
        title="Subarray A",
        **panel_kwargs,
    )
    plot_subarray_panel(
        ax_top_b,
        geo_df,
        boho_df,
        STATIONS_B,
        empty_groups,
        AXIS_LIMITS_B,
        hollow_markers=False,
        show_events=False,
        show_station_labels=True,
        show_scale_bar=False,
        plot_east_axis_label=False,
        plot_north_axis_label=False,
        title="Subarray B",
        **panel_kwargs,
    )
    scatter_bot_a = plot_subarray_panel(
        ax_bot_a,
        geo_df,
        boho_df,
        STATIONS_A,
        groups_a,
        AXIS_LIMITS_A,
        hollow_markers=True,
        show_events=True,
        show_station_labels=False,
        show_scale_bar=False,
        plot_east_axis_label=True,
        plot_north_axis_label=True,
        title=None,
        **panel_kwargs,
    )
    plot_hammer_location(
        ax_bot_a,
        hammer_east,
        hammer_north,
        label_east=args.hammer_label_east,
        label_north=args.hammer_label_north,
        label_ha=args.hammer_label_ha,
        label_va=args.hammer_label_va,
        label_size=args.event_label_size,
        size=args.hammer_size,
        linewidth=args.linewidth_marker,
        linewidth_label=args.linewidth_event_label,
    )
    scatter_bot_b = plot_subarray_panel(
        ax_bot_b,
        geo_df,
        boho_df,
        STATIONS_B,
        groups_b,
        AXIS_LIMITS_B,
        hollow_markers=True,
        show_events=True,
        show_station_labels=False,
        show_scale_bar=True,
        plot_east_axis_label=True,
        plot_north_axis_label=False,
        title=None,
        **panel_kwargs,
    )

    bin_edges = arange(MIN_DEPTH, MAX_DEPTH + BIN_WIDTH, BIN_WIDTH)
    hist_ax_a, max_count_a = add_depth_histogram_inset(
        ax_bot_a,
        groups_a,
        "A",
        args.depth_hist_size,
        args.depth_hist_tick_label_size + 2,
        args.depth_hist_tick_label_size,
        bin_edges,
        args.depth_hist_left,
        True,
    )
    hist_ax_b, max_count_b = add_depth_histogram_inset(
        ax_bot_b,
        groups_b,
        "B",
        args.depth_hist_size,
        args.depth_hist_tick_label_size + 2,
        args.depth_hist_tick_label_size,
        bin_edges,
        args.depth_hist_left / 2.0,
        False,
    )
    max_count = max(max_count_a, max_count_b)
    if max_count > 0:
        hist_ax_a.set_xlim(1.0, max_count * 2.0)
        hist_ax_b.set_xlim(1.0, max_count * 2.0)

    event_mappable = scatter_bot_a if scatter_bot_a is not None else scatter_bot_b
    if event_mappable is None:
        event_mappable = scatter_top_a
    if event_mappable is not None:
        cbar_length = 1.0 / 3.0
        cax = fig.add_axes(
            [
                (1.0 - cbar_length) / 2.0,
                cbar_label_room,
                cbar_length,
                args.cbar_width,
            ]
        )
        cbar = fig.colorbar(event_mappable, cax=cax, orientation="horizontal")
        cbar.set_label("Event count", fontsize=args.tick_label_size + 2)
        cbar.ax.tick_params(labelsize=args.tick_label_size)

    for label, ax in zip(
        ("(a)", "(b)", "(c)", "(d)"),
        (ax_top_a, ax_top_b, ax_bot_a, ax_bot_b),
    ):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + args.subplot_offset_x,
            bbox.y1 + args.subplot_offset_y,
            label,
            fontsize=args.subplot_label_size,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    save_figure(fig, args.figname)


if __name__ == "__main__":
    main()
