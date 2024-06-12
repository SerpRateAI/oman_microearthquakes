# Plot the global map in the MIT-WHOI retreat poster

# Imports
from matplotlib.pyplot import figure
from cartopy.crs import Orthographic, Geodetic
import cartopy.feature as cfeature

from utils_basic import CENTER_LONGITUDE as lon, CENTER_LATITUDE as lat
from utils_plot import save_figure

# Inputs
fig_width = 5.0
fig_height = 5.0

linewidth_coast = 1.0
linewidth_marker = 1.0
linewidth_outline = 2.0

# Create a figure with a specific size
fig = figure(figsize=(fig_width, fig_height))

# Define the orthographic projection centered at the given longitude and latitude
ax = fig.add_subplot(1, 1, 1, projection = Orthographic(central_longitude=lon, central_latitude=lat))

# Add features
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.OCEAN, color='skyblue')

# Optionally add coastlines for better clarity
ax.coastlines(linewidth = linewidth_coast)

# Plot a star at the given longitude and latitude
ax.scatter(lon, lat, marker = '*', s = 100, color='darkviolet', edgecolor = "black", linewidths = linewidth_marker, transform = Geodetic(), zorder = 10)

# Ensure the star is within the map limits
ax.set_global()

# Set the outline width
for spine in ax.spines.values():
    spine.set_linewidth(linewidth_outline)

# Save the figure
filename = "mit_whoi_poster_globe.png"
save_figure(fig, filename)

filename = "mit_whoi_poster_globe.pdf"
save_figure(fig, filename)