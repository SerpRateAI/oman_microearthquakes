"""
Plot the regional geology map and whole-array site map for AGU Advances.

Panel (a) reproduces the left panel of plot_jgr_regional_n_satellite_map.py:
Samail Ophiolite lithology, coast/land, study-site marker, and globe inset.
Panel (b) reproduces the left panel of plot_jgr_site_map.py: satellite background,
uniform geophone markers (no highlighting or station labels), borehole markers with
labels, subarray labels, scale bar, and legend. Overlays crimson rectangles for the
Subarray A and B middle boundaries defined by the *_MIDDLE constants in utils_basic.py.
"""

from argparse import ArgumentParser

from cartopy.crs import Orthographic, PlateCarree
from cartopy.feature import LAND
from cartopy.io.shapereader import natural_earth
from geopandas import read_file
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle
from matplotlib.pyplot import figure
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pyproj import Transformer
from utm import to_latlon

from utils_basic import (
    CENTER_LATITUDE,
    CENTER_LONGITUDE,
    EASTMAX_A_MIDDLE,
    EASTMAX_B_MIDDLE,
    EASTMAX_WHOLE,
    EASTMIN_A_MIDDLE,
    EASTMIN_B_MIDDLE,
    EASTMIN_WHOLE,
    NORTHMAX_A_MIDDLE,
    NORTHMAX_B_MIDDLE,
    NORTHMAX_WHOLE,
    NORTHMIN_A_MIDDLE,
    NORTHMIN_B_MIDDLE,
    NORTHMIN_WHOLE,
    get_borehole_coords,
    get_geophone_coords,
)
from utils_plot import format_east_xlabels, format_north_ylabels, save_figure
from utils_satellite import load_maxar_image


def add_middle_boundary_rectangle(ax, eastmin, eastmax, northmin, northmax, *, linewidth):
    """Draw an outline rectangle for a subarray middle boundary."""
    rect = Rectangle(
        (eastmin, northmin),
        eastmax - eastmin,
        northmax - northmin,
        edgecolor="crimson",
        facecolor="none",
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(rect)


def plot_regional_geology_map(
    ax,
    path_gpkg,
    color_mafic,
    color_ultramafic,
    min_east_geology,
    max_east_geology,
    min_north_geology,
    max_north_geology,
    legend_size,
):
    """Plot the JGR regional geology map (Samail Ophiolite lithology and globe inset)."""
    globe_x = 0.5
    globe_y = 0.5
    globe_width = 0.3
    globe_height = 0.3

    try:
        gdf = read_file(path_gpkg, layer="Lithology")

        mafic_mask = gdf["lithology"].astype(str).str.contains(
            r"mafic|gabbro|basalt|diabase", case=False, na=False
        )
        ultramafic_mask = gdf["lithology"].astype(str).str.contains(
            r"ultramafic|harzburgite|dunite|wehrlite|lherzolite", case=False, na=False
        )

        gdf_mafic = gdf[mafic_mask]
        gdf_ultra = gdf[ultramafic_mask]

        coast_path = natural_earth(resolution="10m", category="physical", name="coastline")
        coast = read_file(coast_path)
        coast = coast.to_crs("EPSG:32640")

        land_path = natural_earth(resolution="10m", category="physical", name="land")
        land = read_file(land_path)
        land = land.to_crs("EPSG:32640")

        if not coast.empty:
            coast.plot(ax=ax, color="black", linewidth=0.4, zorder=1)
        if not land.empty:
            land.plot(ax=ax, color="lightgray", linewidth=0.4, zorder=2)

        if not gdf_ultra.empty:
            print("Plotting the ultramafic polygons...")
            gdf_ultra.plot(ax=ax, facecolor=color_ultramafic, edgecolor="black", linewidth=0.2, zorder=3)
        if not gdf_mafic.empty:
            print("Plotting the mafic polygons...")
            gdf_mafic.plot(ax=ax, facecolor=color_mafic, edgecolor="black", linewidth=0.2, zorder=4)

        legend_handles = []
        if not gdf_mafic.empty:
            legend_handles.append(Patch(facecolor=color_mafic, label="Mafic"))
        if not gdf_ultra.empty:
            legend_handles.append(Patch(facecolor=color_ultramafic, label="Ultramafic"))
        if legend_handles:
            leg_inset = ax.legend(
                handles=legend_handles, loc="upper right",
                fontsize=legend_size,
                frameon=True,
            )
            leg_inset.get_frame().set_edgecolor("black")
            leg_inset.get_frame().set_alpha(1.0)

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32640", always_xy=True)
        site_coords = transformer.transform(CENTER_LONGITUDE, CENTER_LATITUDE)
        ax.scatter(
            site_coords[0], site_coords[1],
            marker="*", color="crimson", edgecolors="black", s=300, linewidths=1.0, zorder=10,
        )
        ax.annotate(
            "Study site", (site_coords[0], site_coords[1]),
            textcoords="offset points", xytext=(0, 20.0),
            ha="center", va="bottom", fontsize=14, fontweight="bold", color="crimson",
        )

        ax.set_xlim(min_east_geology, max_east_geology)
        ax.set_ylim(min_north_geology, max_north_geology)

        ax.set_aspect("equal")
        ax.set_xlabel("UTM East (m)", fontsize=14)
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_major_locator(MultipleLocator(1e5))
        ax.set_ylabel("UTM North (m)", fontsize=14)
        ax.yaxis.set_major_formatter("{x:.0f}")
        ax.yaxis.set_major_locator(MultipleLocator(1e5))
        ax.tick_params(axis="both", labelsize=12)

        ax_globe = ax.inset_axes(
            [globe_x, globe_y, globe_width, globe_height],
            projection=Orthographic(central_longitude=CENTER_LONGITUDE, central_latitude=CENTER_LATITUDE),
        )
        ax_globe.set_global()
        ax_globe.add_feature(LAND, color="lightgray")
        ax_globe.coastlines(linewidth=0.4)

        min_lat, min_lon = to_latlon(min_east_geology, min_north_geology, 40, "N")
        max_lat, max_lon = to_latlon(max_east_geology, max_north_geology, 40, "N")

        print(f"Min longitude: {min_lon}, Max longitude: {max_lon}")
        print(f"Min latitude: {min_lat}, Max latitude: {max_lat}")

        rect_lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
        rect_lats = [min_lat, min_lat, max_lat, max_lat, min_lat]

        ax_globe.plot(
            rect_lons, rect_lats,
            transform=PlateCarree(),
            linewidth=1.2, color="crimson",
        )

    except Exception as e:
        print(f"[WARN] Could not render lithology inset from {path_gpkg}: {e}")


def main():
    parser = ArgumentParser(
        description="Plot the regional geology map and whole-array site map (AGU Advances)."
    )
    parser.add_argument("--image_alpha", type=float, default=0.5, help="Opacity of the satellite image.")
    parser.add_argument("--station_size", type=float, default=150.0, help="Size of the geophone markers.")
    parser.add_argument("--borehole_size", type=float, default=150.0, help="Size of the borehole markers.")
    parser.add_argument("--linewidth_marker", type=float, default=1.0, help="Line width of the markers.")
    parser.add_argument(
        "--linewidth_rectangle",
        type=float,
        default=2.0,
        help="Line width of the middle-boundary rectangles.",
    )
    parser.add_argument("--color_geo", type=str, default="gold", help="Color of the geophone markers.")
    parser.add_argument("--color_borehole", type=str, default="violet", help="Color of the borehole markers.")
    parser.add_argument("--color_hydro", type=str, default="violet", help="Color of the hydrophone markers.")
    parser.add_argument("--subarray_label_size", type=float, default=15.0, help="Font size of the subarray labels.")
    parser.add_argument("--axis_label_size", type=float, default=12.0, help="Font size of the axis labels.")
    parser.add_argument("--tick_label_size", type=float, default=12.0, help="Font size of the tick labels.")
    parser.add_argument("--legend_size", type=float, default=12.0, help="Font size of the legend.")
    parser.add_argument("--subarray_a_label_x", type=float, default=-5.0, help="X-coordinate of the subarray A label.")
    parser.add_argument("--subarray_a_label_y", type=float, default=-40.0, help="Y-coordinate of the subarray A label.")
    parser.add_argument("--subarray_b_label_x", type=float, default=-60.0, help="X-coordinate of the subarray B label.")
    parser.add_argument("--subarray_b_label_y", type=float, default=70.0, help="Y-coordinate of the subarray B label.")
    parser.add_argument("--figheight", type=float, default=8.0, help="Height of the figure.")
    parser.add_argument(
        "--wspace",
        type=float,
        default=0.05,
        help="Fractional width of the space between the geology map and the site map.",
    )
    parser.add_argument("--min_east_geology", type=float, default=320000, help="Minimum east of the geology map.")
    parser.add_argument("--max_east_geology", type=float, default=750000, help="Maximum east of the geology map.")
    parser.add_argument("--min_north_geology", type=float, default=2450000, help="Minimum north of the geology map.")
    parser.add_argument("--max_north_geology", type=float, default=2900000, help="Maximum north of the geology map.")
    parser.add_argument(
        "--path_gpkg",
        type=str,
        default="/proj/mazu/tianze.liu/oman/geology/Nicolas_ea_5_lith_4.gpkg",
        help="Path to the simplified Samail Ophiolite GeoPackage (.gpkg).",
    )
    parser.add_argument("--color_mafic", type=str, default="lightskyblue", help="Fill color for mafic units.")
    parser.add_argument("--color_ultramafic", type=str, default="palegreen", help="Fill color for ultramafic units.")
    parser.add_argument("--figname", type=str, default="aguadv_site_map.png", help="Output figure filename.")
    args = parser.parse_args()

    image_alpha = args.image_alpha
    station_size = args.station_size
    borehole_size = args.borehole_size
    linewidth_marker = args.linewidth_marker
    linewidth_rectangle = args.linewidth_rectangle
    color_geo = args.color_geo
    color_hydro = args.color_hydro
    subarray_label_size = args.subarray_label_size
    axis_label_size = args.axis_label_size
    tick_label_size = args.tick_label_size
    legend_size = args.legend_size
    subarray_a_label_x = args.subarray_a_label_x
    subarray_a_label_y = args.subarray_a_label_y
    subarray_b_label_x = args.subarray_b_label_x
    subarray_b_label_y = args.subarray_b_label_y
    fig_height = args.figheight
    wspace = args.wspace
    min_east_geology = args.min_east_geology
    max_east_geology = args.max_east_geology
    min_north_geology = args.min_north_geology
    max_north_geology = args.max_north_geology
    path_gpkg = args.path_gpkg
    color_mafic = args.color_mafic
    color_ultramafic = args.color_ultramafic
    figname = args.figname

    margin_x = 0.05
    margin_y = 0.05
    scale_bar_length = 25.0
    borehole_font_size = 14.0
    borehole_label_x = 50.0
    borehole_label_y = -50.0
    major_dist_spacing = 25.0
    major_tick_length = 5.0
    minor_tick_length = 2.0
    tick_width = 1.0

    geo_df = get_geophone_coords()
    boho_df = get_borehole_coords()
    rgb_image, extent_img = load_maxar_image()

    aspect_ratio_geology = (max_north_geology - min_north_geology) / (max_east_geology - min_east_geology)
    aspect_ratio_site = (NORTHMAX_WHOLE - NORTHMIN_WHOLE) / (EASTMAX_WHOLE - EASTMIN_WHOLE)
    ax_height = fig_height * (1 - 2 * margin_y)
    ax_width_abs_geology = ax_height / aspect_ratio_geology
    ax_width_abs_site = ax_height / aspect_ratio_site
    fig_width = (ax_width_abs_geology + ax_width_abs_site) / (1 - 2 * margin_x - wspace)
    ax_width_rel_geology = ax_width_abs_geology / fig_width
    ax_width_rel_site = ax_width_abs_site / fig_width

    fig = figure(figsize=(fig_width, fig_height))
    ax_geology = fig.add_axes([margin_x, margin_y, ax_width_rel_geology, 1 - 2 * margin_y])
    ax_map = fig.add_axes(
        [margin_x + ax_width_rel_geology + wspace, margin_y, ax_width_rel_site, 1 - 2 * margin_y]
    )

    ax_geology.text(
        -0.05, 1.05, "(a)",
        fontsize=16, fontweight="bold", transform=ax_geology.transAxes, ha="right", va="top",
    )
    ax_map.text(
        -0.05, 1.05, "(b)",
        fontsize=16, fontweight="bold", transform=ax_map.transAxes, ha="right", va="top",
    )

    print("Plotting the regional geology map...")
    plot_regional_geology_map(
        ax_geology,
        path_gpkg,
        color_mafic,
        color_ultramafic,
        min_east_geology,
        max_east_geology,
        min_north_geology,
        max_north_geology,
        legend_size,
    )

    print("Plotting the station map...")
    ax_map.imshow(rgb_image, extent=extent_img, zorder=0, alpha=image_alpha)

    add_middle_boundary_rectangle(
        ax_map,
        EASTMIN_A_MIDDLE,
        EASTMAX_A_MIDDLE,
        NORTHMIN_A_MIDDLE,
        NORTHMAX_A_MIDDLE,
        linewidth=linewidth_rectangle,
    )
    add_middle_boundary_rectangle(
        ax_map,
        EASTMIN_B_MIDDLE,
        EASTMAX_B_MIDDLE,
        NORTHMIN_B_MIDDLE,
        NORTHMAX_B_MIDDLE,
        linewidth=linewidth_rectangle,
    )

    for _, coords in geo_df.iterrows():
        ax_map.scatter(
            coords["east"],
            coords["north"],
            marker="^",
            s=station_size,
            color=color_geo,
            edgecolors="black",
            linewidths=linewidth_marker,
            label="Geophone",
        )

    for borehole, coords in boho_df.iterrows():
        east = coords["east"]
        north = coords["north"]
        ax_map.scatter(
            east,
            north,
            marker="o",
            s=borehole_size,
            color=color_hydro,
            edgecolors="black",
            linewidths=linewidth_marker,
            label="Borehole/Hydrophones",
        )
        ax_map.annotate(
            borehole,
            (east, north),
            textcoords="offset points",
            xytext=(borehole_label_x, borehole_label_y),
            ha="left",
            va="top",
            fontsize=borehole_font_size,
            fontweight="bold",
            color="black",
            arrowprops=dict(arrowstyle="-", color="black"),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
        )

    ax_map.text(
        subarray_a_label_x,
        subarray_a_label_y,
        "Subarray A",
        color="crimson",
        fontsize=subarray_label_size,
        fontweight="bold",
        verticalalignment="center",
        horizontalalignment="center",
    )
    ax_map.text(
        subarray_b_label_x,
        subarray_b_label_y,
        "Subarray B",
        color="crimson",
        fontsize=subarray_label_size,
        fontweight="bold",
        verticalalignment="center",
        horizontalalignment="center",
    )

    scale_bar = AnchoredSizeBar(
        ax_map.transData,
        scale_bar_length,
        f"{scale_bar_length:.0f} m",
        loc="lower right",
        bbox_transform=ax_map.transAxes,
        frameon=True,
        size_vertical=1.0,
        pad=0.5,
        fontproperties=FontProperties(size=axis_label_size),
    )
    ax_map.add_artist(scale_bar)

    handles, labels = ax_map.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    legend = ax_map.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper left",
        frameon=True,
        fancybox=False,
        fontsize=legend_size,
        bbox_transform=ax_map.transAxes,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_alpha(1.0)

    ax_map.set_xlim(EASTMIN_WHOLE, EASTMAX_WHOLE)
    ax_map.set_ylim(NORTHMIN_WHOLE, NORTHMAX_WHOLE)
    ax_map.set_aspect("equal")

    format_east_xlabels(
        ax_map,
        plot_axis_label=True,
        major_tick_spacing=major_dist_spacing,
        axis_label_size=axis_label_size,
        tick_label_size=tick_label_size,
        major_tick_length=major_tick_length,
        minor_tick_length=minor_tick_length,
        tick_width=tick_width,
    )
    format_north_ylabels(
        ax_map,
        plot_axis_label=True,
        major_tick_spacing=major_dist_spacing,
        axis_label_size=axis_label_size,
        tick_label_size=tick_label_size,
        major_tick_length=major_tick_length,
        minor_tick_length=minor_tick_length,
        tick_width=tick_width,
    )
    ax_map.set_xlabel("East (m)")
    ax_map.set_ylabel("North (m)")

    print("Saving the figure...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
