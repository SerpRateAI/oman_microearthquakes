# Plot the large-scale map containing the locations of the GFZ stations

# Imports
from os.path import join
from matplotlib.pyplot import get_cmap, subplots
from rasterio import open
from rasterio.plot import plotting_extent

from utils_basic import ROOTDIR_MAP as indir, CENTER_LATITUDE as mbo_lat, CENTER_LONGITUDE as mbo_lon
from utils_basic import get_broadband_metadata
from utils_plot import add_colorbar, save_figure

# Inputs
dem_filename = "merged_dem.tif"
gfz_stations = ["COO03", "COO32"]

min_lon = 58.45
max_lon = 58.80
min_lat = 22.75
max_lat = 23.15

min_elev = 300.0
max_elev = 15
00.0

figwidth = 10.0

colormap_topo = "gray"
colormap_sta = "tab20"
marker_size = 200
i_color_mbo = 4
i_color_gfz = 5

linewidth_station = 2.0
label_fontsize = 14

# Load the GFZ station locations
gfz_meta = get_broadband_metadata()

# Read the DEM file
print("Reading the DEM file...")
inpath = join(indir, dem_filename)
with open(inpath) as src:
    dem = src.read(1)
    meta = src.meta

# Plotting
print("Plotting the map...")
figheight = figwidth * meta["height"] / meta["width"]
fig, ax = subplots(figsize = (figwidth, figheight))
extent = plotting_extent(src)

mappable = ax.imshow(dem, extent = extent, cmap = colormap_topo, vmin = min_elev, vmax = max_elev)
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")

color_mbo = get_cmap(colormap_sta)(i_color_mbo)
ax.scatter(mbo_lon, mbo_lat, s = marker_size, c = color_mbo, marker = "^", facecolor = color_mbo, linewidth = None)
ax.annotate("MBO", (mbo_lon, mbo_lat), 
            xytext = (0.0, 8.0), textcoords = "offset points", color = color_mbo, 
            fontsize = label_fontsize, fontweight = "bold",
            ha = "center", va = "bottom")

for i, station in enumerate(gfz_stations):
    id = f"5H.{station}..HHZ"
    lat = gfz_meta.get_coordinates(id)["latitude"]
    lon = gfz_meta.get_coordinates(id)["longitude"]

    color_gfz = get_cmap(colormap_sta)(i_color_gfz)
    ax.scatter(lon, lat, s = marker_size, c = color_gfz, marker = "^", facecolor = color_gfz, linewidth = None)
    ax.annotate(station, (lon, lat), xytext = (0.0, 8.0), textcoords = "offset points", color = color_gfz, 
                fontsize = label_fontsize, fontweight = "bold",
                ha = "center", va = "bottom")
    
bbox = ax.get_position()
cbar_pos = [bbox.x1 + 0.02, bbox.y0, 0.01, bbox.height]
add_colorbar(fig, mappable, "Elevation (m)", cbar_pos)
           
# Save the figure
print("Saving the figure...")
figname = "mbo_n_gfz_station_map.png"
save_figure(fig, figname)