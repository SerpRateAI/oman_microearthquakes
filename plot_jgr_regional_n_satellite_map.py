
# Plot only the map and hydrophone depth profiles in Fig. 1 in Liu et al. (2025a) for presentation purposes
from os.path import join
from numpy import isnan, float32, issubdtype, floating, repeat, zeros, empty_like, isfinite, percentile, clip
from numpy import amax, ma


from argparse import ArgumentParser
from json import loads
from scipy.interpolate import interp1d
from pandas import DataFrame, Timedelta
from pandas import concat, read_csv, read_hdf
from matplotlib.pyplot import figure
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
from utm import to_latlon
from matplotlib.ticker import MultipleLocator

from cartopy.crs import Orthographic, Geodetic
import cartopy.feature as cfeature
from cartopy.io import shapereader 
from pyproj import Transformer
import cartopy.crs as ccrs
from geopandas import read_file, datasets as gpd_datasets
from fiona import listlayers

from utils_basic import EASTMIN_WHOLE as min_east_array, EASTMAX_WHOLE as max_east_array, NORTHMIN_WHOLE as min_north_array, NORTHMAX_WHOLE as max_north_array
from utils_basic import GEO_COMPONENTS as components
from utils_basic import SPECTROGRAM_DIR as dir_spec, MT_DIR as dir_mt
from utils_basic import CENTER_LONGITUDE as lon, CENTER_LATITUDE as lat
from utils_basic import IMAGE_DIR as dir_img
from utils_basic import get_geophone_coords, get_borehole_coords, str2timestamp
from utils_basic import power2db
from utils_satellite import load_psscene_image
from utils_plot import format_east_xlabels, format_db_ylabels, format_freq_xlabels, format_north_ylabels, format_depth_ylabels, get_geo_component_color, save_figure


### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for plotting the large-scale map of the surrounding area of the study site.")

parser.add_argument("--image_alpha", type=float, default=0.5, help="Opacity of the satellite image.")
parser.add_argument("--figheight", type=float, default=8.0, help="Height of the figure.")
parser.add_argument("--margin_x", type=float, default=0.05, help="Margin of the figure on the x-axis.")
parser.add_argument("--margin_y", type=float, default=0.05, help="Margin of the figure on the y-axis.")
parser.add_argument("--min_east_satellite", type=float, default=-4000.0, help="Minimum east of the satellite image")
parser.add_argument("--max_east_satellite", type=float, default=4000.0, help="Maximum east of the satellite image")
parser.add_argument("--min_north_satellite", type=float, default=-4000.0, help="Minimum north of the satellite image")
parser.add_argument("--max_north_satellite", type=float, default=3100.0, help="Maximum north of the satellite image")
parser.add_argument("--min_east_geology", type=float, default=320000, help="Minimum east of the geology")
parser.add_argument("--max_east_geology", type=float, default=750000, help="Maximum east of the geology")
parser.add_argument("--min_north_geology", type=float, default=2450000, help="Minimum north of the geology")
parser.add_argument("--max_north_geology", type=float, default=2900000, help="Maximum north of the geology")
parser.add_argument("--wspace", type=float, default=0.05, help="Width of the space between the satellite image and the geology")

parser.add_argument("--path_gpkg", type=str, default="/proj/mazu/tianze.liu/oman/geology/Nicolas_ea_5_lith_4.gpkg",
                    help="Path to the simplified Samail Ophiolite GeoPackage (.gpkg).")
parser.add_argument("--color_mafic", type=str, default="lightskyblue",
                    help="Fill color for mafic units in the inset.")
parser.add_argument("--color_ultramafic", type=str, default="palegreen",
                    help="Fill color for ultramafic units in the inset.")

# Parse the command line arguments
args = parser.parse_args()

image_alpha = args.image_alpha
fig_height = args.figheight
margin_x = args.margin_x
margin_y = args.margin_y
min_east_satellite = args.min_east_satellite
max_east_satellite = args.max_east_satellite
min_north_satellite = args.min_north_satellite
max_north_satellite = args.max_north_satellite
min_east_geology = args.min_east_geology
max_east_geology = args.max_east_geology
min_north_geology = args.min_north_geology
max_north_geology = args.max_north_geology
wspace = args.wspace

path_gpkg = args.path_gpkg
color_mafic = args.color_mafic
color_ultramafic = args.color_ultramafic

# Constants
globe_x = 0.5
globe_y = 0.5
globe_width = 0.3
globe_height = 0.3

legend_size = 12.0

# # --- Generate the figure and axes ---
# Aspect ratios of the figure
aspect_ratio_geology = (max_north_geology - min_north_geology) / (max_east_geology - min_east_geology)
aspect_ratio_satellite = (max_north_satellite - min_north_satellite) / (max_east_satellite - min_east_satellite)

# Compute the dimensions of the figure
ax_width_abs_geology = fig_height * (1 - 2 * margin_y) / aspect_ratio_geology
ax_width_abs_satellite = fig_height * (1 - 2 * margin_y) / aspect_ratio_satellite
fig_width = (ax_width_abs_geology + ax_width_abs_satellite) / (1 - 2 * margin_x - wspace)

ax_width_rel_geology = ax_width_abs_geology / fig_width
ax_width_rel_satellite = ax_width_abs_satellite / fig_width

# Generate the figure and axes
fig = figure(figsize = (fig_width, fig_height))
ax_geology = fig.add_axes([margin_x, margin_y, ax_width_rel_geology, 1 - 2 * margin_y])
ax_satellite = fig.add_axes([margin_x + ax_width_rel_geology + wspace, margin_y, ax_width_rel_satellite, 1 - 2 * margin_y])

ax_geology.text(-0.05, 1.05, "(a)", fontsize = 14, fontweight = "bold", transform = ax_geology.transAxes, ha = "right", va = "top")
ax_satellite.text(-0.05, 1.05, "(b)", fontsize = 14, fontweight = "bold", transform = ax_satellite.transAxes, ha = "right", va = "top")

# Plot the geological map
try:
    # Load GPKG
    gdf = read_file(path_gpkg, layer = "Lithology")

    mafic_mask = gdf["lithology"].astype(str).str.contains(
        r"mafic|gabbro|basalt|diabase", case=False, na=False
    )
    ultramafic_mask = gdf["lithology"].astype(str).str.contains(
        r"ultramafic|harzburgite|dunite|wehrlite|lherzolite", case=False, na=False
    )

    gdf_mafic = gdf[mafic_mask]
    gdf_ultra = gdf[ultramafic_mask]

    # Read the coastline shapefiles
    coast_path = shapereader.natural_earth(resolution="10m", category="physical", name="coastline")
    coast = read_file(coast_path)
    coast = coast.to_crs("EPSG:32640")

    land_path = shapereader.natural_earth(resolution="10m", category="physical", name="land")
    land = read_file(land_path)
    land = land.to_crs("EPSG:32640")

    if not coast.empty:
        coast.plot(ax=ax_geology, color="black", linewidth=0.4, zorder=1)
    if not land.empty:
        land.plot(ax=ax_geology, color="lightgray", linewidth=0.4, zorder=2)

    # Plot the lithology polygons
    if not gdf_ultra.empty:
        print("Plotting the ultramafic polygons...")
        gdf_ultra.plot(ax=ax_geology, facecolor=color_ultramafic, edgecolor="black", linewidth=0.2, zorder=3)
    if not gdf_mafic.empty:
        print("Plotting the mafic polygons...")
        gdf_mafic.plot(ax=ax_geology, facecolor=color_mafic, edgecolor="black", linewidth=0.2, zorder=4)

    # Add a small legend for the lithology
    legend_handles = []
    legend_labels = []
    if not gdf_mafic.empty:
        legend_handles.append(Patch(facecolor=color_mafic, label="Mafic"))
        legend_labels.append("Mafic")
    if not gdf_ultra.empty:
        legend_handles.append(Patch(facecolor=color_ultramafic, label="Ultramafic"))
        legend_labels.append("Ultramafic")
    if legend_handles:
        leg_inset = ax_geology.legend(
            handles=legend_handles, loc="upper right",
            fontsize=legend_size,
            frameon=True
        )
        leg_inset.get_frame().set_edgecolor("black")
        leg_inset.get_frame().set_alpha(1.0)

    # Transform the site coordinates and plot the site locations
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32640", always_xy=True)
    site_coords = transformer.transform(lon, lat)
    ax_geology.scatter(site_coords[0], site_coords[1], marker = "*", color = "violet", edgecolors = "black", s = 300, linewidths = 1.0, zorder = 10)
    ax_geology.annotate("Study site", (site_coords[0], site_coords[1]), textcoords = "offset points", xytext = (0, 20.0), ha = "center", va = "bottom", fontsize = 14, fontweight = "bold", color = "violet")
    

    # Set the axis limits
    ax_geology.set_xlim(min_east_geology, max_east_geology)
    ax_geology.set_ylim(min_north_geology, max_north_geology)

    # Clean up inset look
    ax_geology.set_aspect("equal")
    ax_geology.set_xlabel("UTM East (m)", fontsize = 14)
    ax_geology.xaxis.set_major_formatter('{x:.0f}')
    ax_geology.xaxis.set_major_locator(MultipleLocator(1e5))
    ax_geology.set_ylabel("UTM North (m)", fontsize = 14)
    ax_geology.yaxis.set_major_formatter('{x:.0f}')
    ax_geology.yaxis.set_major_locator(MultipleLocator(1e5))
    ax_geology.tick_params(axis='both', labelsize=12)

    # Add a global map with coastlines
    ax_globe = ax_geology.inset_axes([globe_x, globe_y, globe_width, globe_height], projection = Orthographic(central_longitude=lon, central_latitude=lat))
    ax_globe.set_global()
    ax_globe.add_feature(cfeature.LAND, color="lightgray")
    ax_globe.coastlines(linewidth = 0.4)

    # Add the boundary of the geological map
    min_lat, min_lon = to_latlon(min_east_geology, min_north_geology, 40, "N")
    max_lat, max_lon = to_latlon(max_east_geology, max_north_geology, 40, "N")

    print(f"Min longitude: {min_lon}, Max longitude: {max_lon}")
    print(f"Min latitude: {min_lat}, Max latitude: {max_lat}")

    # 2) Build the rectangle corners (close the loop)
    rect_lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
    rect_lats = [min_lat, min_lat, max_lat, max_lat, min_lat]

    #3 Plot on the orthographic inset; the data are lon/lat, so use PlateCarree
    ax_globe.plot(
        rect_lons, rect_lats,
        transform=ccrs.PlateCarree(),
        linewidth=1.2, color="crimson"
)
    
except Exception as e:
    print(f"[WARN] Could not render lithology inset from {path_gpkg}: {e}")

print("Loading the satellite image...")
rgb_image, extent_img = load_psscene_image()

# Plot the satellite image
ax_satellite.imshow(rgb_image, extent=extent_img, zorder=0, alpha=image_alpha)

# Plot the site location
ax_satellite.scatter(0.0, 0.0, marker = "*", color = "white", edgecolors = "black", s = 300, linewidths = 1.0, zorder = 10)

# Plot the farm location
ax_satellite.scatter(0, 0, marker = "*", color = "white", edgecolors = "black", s = 300, linewidths = 1.0, zorder = 10)
ax_satellite.annotate("Study site", (0, 0), textcoords = "offset points", xytext = (0, 10.0), ha = "center", va = "bottom", fontsize = 14, fontweight = "bold", color = "white")
ax_satellite.annotate("Farm", (1000.0, -3100.0), 
                textcoords = "offset points", xytext = (10, -10.0), ha = "left", va = "top", fontsize = 14, fontweight = "bold", 
                color = "greenyellow", arrowprops=dict(arrowstyle = "->", color = "greenyellow", lw = 1.0))

# Set axes limits
ax_satellite.set_xlim(min_east_satellite, max_east_satellite)
ax_satellite.set_ylim(min_north_satellite, max_north_satellite)

ax_satellite.set_xlabel("East (m)", fontsize = 14)
ax_satellite.set_ylabel("North (m)", fontsize = 14)
ax_satellite.tick_params(axis='both', labelsize=12)

ax_satellite.set_aspect("equal")

### Save the figure ###
print("Saving the figure...")
figname = "jgr_regional_n_satellite_map.png"
save_figure(fig, figname)