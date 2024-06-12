# Plot the maps and hydrophone depth profiles for the MIT-WHOI retreat poster
from numpy import cos, pi,linspace
from matplotlib.pyplot import subplots

from utils_basic import EASTMIN_WHOLE as eastmin_whole, EASTMAX_WHOLE as eastmax_whole, NORTHMIN_WHOLE as northmin_whole, NORTHMAX_WHOLE as northmax_whole
from utils_basic import HYDRO_DEPTHS as depth_dict
from utils_basic import get_geophone_coords, get_borehole_coords
from utils_plot import save_figure, format_east_xlabels, format_north_ylabels, format_depth_ylabels

# Inputs
fig_width = 13.0
fig_height = 13.0

axis_offset = 0.15

min_depth = 0.0
max_depth = 450.0

hydro_min = -0.5
hydro_max = 1.5

water_level = 15.0
water_amp = 2.5
water_period = 0.2

stations_highlight = ["A01", "A16", "B01", "B19"]

linewidth_marker = 2.0
linewidth_water = 5.0

station_size = 200.0
borehole_size = 200.0
hydro_size = 200.0

station_font_size = 25.0
station_label_x = 5.0
station_label_y = 5.0

borehole_font_size = 25.0
borehole_label_x = 60.0
borehole_label_y = -60.0

location_font_size = 25.0

water_font_size = 15.0

major_dist_spacing = 25.0
minor_dist_spacing = 5.0

major_depth_spacing = 50.0
minor_depth_spacing = 10.0  

axis_label_size = 25.0
tick_label_size = 20.0

legend_size = 20.0

major_tick_length = 10.0
minor_tick_length = 5.0
tick_width = 2.0

frame_width = 2.0

# Load the geophone and borehole coordinates
geo_df = get_geophone_coords()
boho_df = get_borehole_coords()

# Generate the figure and axes
# Compute the aspect ratio
east_range = eastmax_whole - eastmin_whole
north_range = northmax_whole - northmin_whole
aspect_ratio = north_range / east_range
fig_height = fig_width * aspect_ratio

fig, ax_sta = subplots(1, 1, figsize = (fig_width, fig_height))


# Plot the geophone locations
for idx, row in geo_df.iterrows():
    east = row["east"]
    north = row["north"]
    station = row["name"]

    if station in stations_highlight:
        ax_sta.scatter(east, north, marker = "^", s = station_size, color = "lightgray", edgecolors = "crimson", linewidths = linewidth_marker)
        ax_sta.annotate(station, (east, north), textcoords = "offset points", xytext = (station_label_x, station_label_y), ha = "left", va = "bottom", fontsize = station_font_size, color = "crimson")
    else:
        ax_sta.scatter(east, north, marker = "^", s = station_size, color = "lightgray", edgecolors = "black", linewidths = linewidth_marker, label = "Geophone")

# Plot the borehole locations
for idx, row in boho_df.iterrows():
    east = row["east"]
    north = row["north"]
    borehole = row["name"]

    ax_sta.scatter(east, north, marker = "o", s = borehole_size, color = "darkviolet", edgecolors = "black", linewidths = linewidth_marker, label = "Borehole/Hydrophones")
    ax_sta.annotate(borehole, (east, north), textcoords = "offset points", xytext = (borehole_label_x, borehole_label_y), ha = "left", va = "top", fontsize = borehole_font_size, color = "darkviolet", arrowprops=dict(arrowstyle = "-", color = "black"))

# Plot the legend
handles, labels = ax_sta.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax_sta.legend(unique_labels.values(), unique_labels.keys(), loc = "upper left", frameon = False, fontsize = legend_size)

# Set the axis limits
ax_sta.set_xlim(eastmin_whole, eastmax_whole)
ax_sta.set_ylim(northmin_whole, northmax_whole)

ax_sta.set_aspect("equal")

# Set the axis ticks
format_east_xlabels(ax_sta, label = True, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
format_north_ylabels(ax_sta, label = True, major_tick_spacing = major_dist_spacing, minor_tick_spacing = minor_dist_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

# Set the axis labels
ax_sta.set_xlabel("East (m)")
ax_sta.set_ylabel("North (m)")

# Adjust the frame width
for spine in ax_sta.spines.values():
    spine.set_linewidth(frame_width)

# Add the axis for plotting the hydrophone depth profiles
bbox = ax_sta.get_position()
map_height = bbox.height
map_width = bbox.width
profile_height = map_height
profile_width = map_width / 3
ax_hydro = fig.add_axes([bbox.x1 + axis_offset, bbox.y0, profile_width, profile_height])

# Plot the hydrophone depth profiles
for offset in [0, 1]:
    for location in depth_dict.keys():
        depth = depth_dict[location]

        if offset == 0 and location in ["01", "02"]:
            ax_hydro.scatter(offset, depth, marker = "o", color = "lightgray", edgecolors = "black", s = hydro_size, linewidths = linewidth_marker, label = "Broken")
        else:
            ax_hydro.scatter(offset, depth, marker = "o", color = "darkviolet", edgecolors = "black", s = hydro_size, linewidths = linewidth_marker, label = "Functional")

        ax_hydro.text(0.5, depth, location, color = "black", fontsize = location_font_size, verticalalignment = "center", horizontalalignment = "center")

ax_hydro.text(0.0, -15.0, "BA1A\n(A00)", color = "black", fontsize = location_font_size, verticalalignment = "center", horizontalalignment = "center")
ax_hydro.text(1.0, -15.0, "BA1B\n(B00)", color = "black", fontsize = location_font_size, verticalalignment = "center", horizontalalignment = "center")

max_hydro_depth = max(depth_dict.values())
ax_hydro.plot([0.0, 0.0], [min_depth, max_hydro_depth], color = "black", linewidth = linewidth_marker, zorder = 0)
ax_hydro.plot([1.0, 1.0], [min_depth, max_hydro_depth], color = "black", linewidth = linewidth_marker, zorder = 0)

# Plot the legend
handles, labels = ax_hydro.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax_hydro.legend(unique_labels.values(), unique_labels.keys(), loc = "lower left", frameon = False, fontsize = legend_size)

# Plot the water level
water_line_x = linspace(hydro_min, hydro_max, 100)
water_line_y = water_level + water_amp * cos(2 * pi * water_line_x / water_period)

ax_hydro.plot(water_line_x, water_line_y, color = "skyblue", linewidth = linewidth_water)
ax_hydro.text(-0.6, water_level, "Water table", color = "skyblue", fontsize = water_font_size, verticalalignment = "center", horizontalalignment = "right")

# Set the axis limits
ax_hydro.set_xlim(hydro_min, hydro_max)
ax_hydro.set_ylim(min_depth, max_depth)

ax_hydro.invert_yaxis()

# Set the axis ticks
ax_hydro.set_xticks([])
format_depth_ylabels(ax_hydro, label = True, major_tick_spacing = major_depth_spacing, minor_tick_spacing = minor_depth_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)

# Adjust the frame width
for spine in ax_hydro.spines.values():
    spine.set_linewidth(frame_width)

# Save the figure
save_figure(fig, "mit_whoi_poster_station_map.png")
save_figure(fig, "mit_whoi_poster_station_map.pdf")
