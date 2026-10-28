"""
Plot hammer locations used to estimate the 1D P velocity model (AGU Advances).

Stations are gold triangles and boreholes are violet circles on a solid gray background;
hammers are salmon stars. Axis limits match the Subarray A localization window used in
test_constrain_vel_model_w_hammers.py.
"""

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import subplots
from pandas import read_csv

from utils_basic import (
    EASTMAX_A_LOC,
    EASTMIN_A_LOC,
    LOC_DIR,
    NORTHMAX_A_LOC,
    NORTHMIN_A_LOC,
    get_borehole_coords,
    get_geophone_coords,
)
from utils_plot import format_east_xlabels, format_north_ylabels, save_figure

DEFAULT_HAMMERS_CSV = Path(LOC_DIR) / "hammer_locations_for_vel_model_estimate.csv"
HAMMER_ID = "100129"
HAMMER_LABEL_EAST = 35.0
HAMMER_LABEL_NORTH = -75.0
HAMMER_LABEL_HA = "left"
HAMMER_LABEL_VA = "top"


def annotate_hammer(
    ax,
    east,
    north,
    *,
    text,
    label_east,
    label_north,
    label_ha,
    label_va,
    label_size,
    linewidth_label,
):
    """Annotate a hammer-shot location with a label and leader line."""
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
        text,
        fontsize=label_size,
        fontweight="bold",
        color="black",
        ha=label_ha,
        va=label_va,
        zorder=5,
        clip_on=False,
    )


def main():
    parser = ArgumentParser(
        description=(
            "Plot hammer locations from the 1D P velocity-model estimate on a "
            "station and borehole map (AGU Advances)."
        )
    )
    parser.add_argument(
        "--hammers_csv",
        type=Path,
        default=DEFAULT_HAMMERS_CSV,
        help="CSV with hammer locations (hammer_id, east, north, ...)",
    )
    parser.add_argument("--figwidth", type=float, default=10.0, help="Figure width (in)")
    parser.add_argument("--figheight", type=float, default=10.0, help="Figure height (in)")
    parser.add_argument(
        "--background_color",
        type=str,
        default="0.85",
        help="Axes background color (solid gray)",
    )
    parser.add_argument("--markersize_hammer", type=float, default=140.0, help="Hammer marker size")
    parser.add_argument("--markersize_station", type=float, default=100.0, help="Station marker size")
    parser.add_argument("--markersize_borehole", type=float, default=100.0, help="Borehole marker size")
    parser.add_argument("--linewidth", type=float, default=0.8, help="Marker edge width")
    parser.add_argument("--color_hammer", type=str, default="salmon", help="Hammer marker color")
    parser.add_argument("--color_station", type=str, default="gold", help="Station marker color")
    parser.add_argument("--color_borehole", type=str, default="violet", help="Borehole marker color")
    parser.add_argument(
        "--hammer_id",
        type=str,
        default=HAMMER_ID,
        help="Hammer shot ID to annotate",
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
    parser.add_argument("--hammer_label_size", type=float, default=10.0, help="Hammer label font size")
    parser.add_argument(
        "--linewidth_hammer_label",
        type=float,
        default=1.0,
        help="Line width for the hammer label leader line",
    )
    parser.add_argument("--major_tick_spacing", type=float, default=20.0, help="East/north tick spacing (m)")
    parser.add_argument("--axis_label_size", type=float, default=12.0, help="Axis label font size")
    parser.add_argument("--tick_label_size", type=float, default=10.0, help="Tick label font size")
    parser.add_argument("--legend_size", type=float, default=12.0, help="Legend font size")
    parser.add_argument(
        "--figname",
        type=str,
        default="aguadv_hammer_locations_for_vel_model_estimate.png",
        help="Output figure filename",
    )
    args = parser.parse_args()

    hammers_csv = args.hammers_csv
    figwidth = args.figwidth
    figheight = args.figheight
    background_color = args.background_color
    markersize_hammer = args.markersize_hammer
    markersize_station = args.markersize_station
    markersize_borehole = args.markersize_borehole
    linewidth = args.linewidth
    color_hammer = args.color_hammer
    color_station = args.color_station
    color_borehole = args.color_borehole
    hammer_id = args.hammer_id
    hammer_label_east = args.hammer_label_east
    hammer_label_north = args.hammer_label_north
    hammer_label_ha = args.hammer_label_ha
    hammer_label_va = args.hammer_label_va
    hammer_label_size = args.hammer_label_size
    linewidth_hammer_label = args.linewidth_hammer_label
    major_tick_spacing = args.major_tick_spacing
    axis_label_size = args.axis_label_size
    tick_label_size = args.tick_label_size
    legend_size = args.legend_size
    figname = args.figname

    print(f"Loading hammer locations from {hammers_csv}...")
    hammers_df = read_csv(hammers_csv, dtype={"hammer_id": str})
    required = {"hammer_id", "east", "north"}
    if not required.issubset(hammers_df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)} in {hammers_csv}; got {list(hammers_df.columns)}"
        )

    hammer_matches = hammers_df.loc[hammers_df["hammer_id"] == hammer_id]
    if hammer_matches.empty:
        raise ValueError(f"Hammer {hammer_id} not found in {hammers_csv}")
    hammer_east = float(hammer_matches.iloc[0]["east"])
    hammer_north = float(hammer_matches.iloc[0]["north"])

    print("Loading station and borehole coordinates...")
    station_df = get_geophone_coords()
    borehole_df = get_borehole_coords()

    fig, ax = subplots(1, 1, figsize=(figwidth, figheight))
    ax.set_facecolor(background_color)

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
        borehole_df["east"],
        borehole_df["north"],
        s=markersize_borehole,
        marker="o",
        color=color_borehole,
        edgecolors="black",
        linewidth=linewidth,
        label="Borehole",
        zorder=2,
    )
    ax.scatter(
        hammers_df["east"],
        hammers_df["north"],
        s=markersize_hammer,
        marker="*",
        color=color_hammer,
        edgecolors="black",
        linewidth=linewidth,
        label="Hammer",
        alpha=0.9,
        zorder=3,
    )
    annotate_hammer(
        ax,
        hammer_east,
        hammer_north,
        text=f"Hammer {hammer_id}",
        label_east=hammer_label_east,
        label_north=hammer_label_north,
        label_ha=hammer_label_ha,
        label_va=hammer_label_va,
        label_size=hammer_label_size,
        linewidth_label=linewidth_hammer_label,
    )

    ax.set_xlim(EASTMIN_A_LOC, EASTMAX_A_LOC)
    ax.set_ylim(NORTHMIN_A_LOC, NORTHMAX_A_LOC)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Locations of hammers used for velocity-model estimate", fontsize=axis_label_size + 2, fontweight="bold")
    ax.legend(loc="best", fontsize=legend_size)

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

    print(f"Saving figure to {figname}...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
