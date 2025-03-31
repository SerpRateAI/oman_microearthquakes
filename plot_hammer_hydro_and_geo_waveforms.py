"""
Plot the hydrophone and geophone waveforms of a hammer shot
"""

###
# Import the required modules
###
from os.path import join
from argparse import ArgumentParser
from numpy import abs, amax
from pandas import read_csv
from pandas import Timestamp
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec

from utils_basic import LOC_DIR as dirpath_loc, HYDRO_LOCATIONS as loc_dict, HYDRO_DEPTHS as depth_dict
from utils_preproc import read_and_process_windowed_hydro_waveforms, read_and_process_windowed_geo_waveforms
from utils_plot import get_geo_component_color, save_figure
from utils_plot import WAVE_VELOCITY_UNIT as vel_unit

###
# Define the input arguments
###

# Define the input arguments
parser = ArgumentParser()
parser.add_argument("--hammer_id", type = str, required = True, help = "The ID of the hammer shot")

parser.add_argument("--window_length", type = float, default = 1.0, help = "The length of the window (s)")
parser.add_argument("--component_to_plot", type = str, default = "Z", help = "The geophone component to plot")
parser.add_argument("--station_geo", type = str, default = "A01", help = "The geophone station to plot")
parser.add_argument("--station_hydro", type = str, default = "A00", help = "The hydrophone station to plot")

parser.add_argument("--scale_factor", type = float, default = 1.0, help = "The scale factor for the hydrophone waveforms")
parser.add_argument("--figwidth", type = float, default = 10.0, help = "The width of the figure (in)")
parser.add_argument("--row_height", type = float, default = 2.5, help = "The height of the figure (in)")
parser.add_argument("--linewidth", type = float, default = 1.0, help = "The linewidth for the waveforms")
parser.add_argument("--min_depth", type = float, default = 130.0, help = "The minimum depth to plot for the hydrophone waveforms")
parser.add_argument("--max_depth", type = float, default = 420.0, help = "The maximum depth to plot for the hydrophone waveforms")
parser.add_argument("--time0", type = float, default = 0.08, help = "The begin time of the tube-wave travel time curve")
parser.add_argument("--vel_tube", type = float, default = 1500.0, help = "The velocity of the tube-wave")
parser.add_argument("--label_geo_x", type = float, default = 0.01, help = "The x coordinate of the geophone label")
parser.add_argument("--label_geo_y", type = float, default = 0.03, help = "The y coordinate of the geophone label")
parser.add_argument("--label_hydro_x", type = float, default = 0.01, help = "The x coordinate of the hydrophone label")
parser.add_argument("--label_offset_hydro_y", type = float, default = 10.0, help = "The y coordinate offset of the hydrophone label")
parser.add_argument("--label_tube_x", type = float, default = 0.1, help = "The x coordinate of the tube-wave travel time curve label")
parser.add_argument("--label_tube_y", type = float, default = 140.0, help = "The y coordinate of the tube-wave travel time curve label")
parser.add_argument("--fontsize_trace_label", type = float, default = 12, help = "The fontsize of the trace labels")
parser.add_argument("--fontsize_suptitle", type = float, default = 14, help = "The fontsize of the suptitle")

# Parse the input arguments
args = parser.parse_args()
hammer_id = args.hammer_id
window_length = args.window_length
figwidth = args.figwidth
row_height = args.row_height
scale_factor = args.scale_factor
component_to_plot = args.component_to_plot
linewidth = args.linewidth
min_depth = args.min_depth
max_depth = args.max_depth
label_geo_x = args.label_geo_x
label_geo_y = args.label_geo_y
label_hydro_x = args.label_hydro_x
label_offset_hydro_y = args.label_offset_hydro_y
label_tube_x = args.label_tube_x
label_tube_y = args.label_tube_y
fontsize_trace_label = args.fontsize_trace_label
fontsize_suptitle = args.fontsize_suptitle
time0 = args.time0
vel_tube = args.vel_tube

station_geo = args.station_geo
station_hydro = args.station_hydro

###
# Read the data
###

# Read the hammer origin time
print(f"Reading the hammer origin time for {hammer_id}...")
filename = f"hammer_locations.csv"
filepath = join(dirpath_loc, filename)
hammer_df = read_csv(filepath, dtype = {"hammer_id": str}, parse_dates = ["origin_time"])
origin_time = hammer_df.loc[hammer_df["hammer_id"] == hammer_id, "origin_time"].values[0]
print(type(origin_time))

# Read the hydrophone waveforms
print(f"Reading the hydrophone waveforms for {hammer_id}...")
start_time = Timestamp(origin_time)
stream_hydro = read_and_process_windowed_hydro_waveforms(start_time, dur = window_length, stations = station_hydro)

# Read the geophone waveforms
print(f"Reading the geophone waveforms for {hammer_id}...")
stream_geo = read_and_process_windowed_geo_waveforms(start_time, dur = window_length, stations = station_geo, components = component_to_plot)

###
# Plot the waveforms
###

# Create a figure
locations = loc_dict[station_hydro]
num_rows = len(locations) + 1
figheight = row_height * num_rows
fig = figure(figsize = (figwidth, figheight))
gs = GridSpec(num_rows, 1, figure = fig)

# Plot the geophone waveform
ax_geo = fig.add_subplot(gs[0, 0])
trace_geo = stream_geo[0]
timeax = trace_geo.times()
data_geo = trace_geo.data
data_geo = data_geo / amax(abs(data_geo))

ax_geo.plot(timeax, data_geo, linewidth = linewidth, color = get_geo_component_color(component_to_plot))
ax_geo.set_xlim(timeax[0], timeax[-1])
ax_geo.set_ylim(-1.0, 1.0)
ax_geo.set_ylabel(f"Normalized amplitude")
ax_geo.text(label_geo_x, label_geo_y, f"{station_geo}.{component_to_plot}", transform = ax_geo.transAxes, fontsize = fontsize_trace_label, ha = "left", va = "bottom", fontweight = "bold")

# Plot the hydrophone waveforms arranged by depth
ax_hydro = fig.add_subplot(gs[1:, 0])

for location in locations:
    trace_hydro = stream_hydro.select(location = location)[0]

    timeax = trace_hydro.times()
    data_hydro = trace_hydro.data
    data_hydro = data_hydro / amax(abs(data_hydro))

    depth = depth_dict[location]
    data_to_plot = data_hydro * scale_factor + depth
    ax_hydro.plot(timeax, data_to_plot, linewidth = linewidth, color = "tab:purple")

    ax_hydro.text(label_hydro_x, label_offset_hydro_y + depth, f"{station_hydro}.{location}", transform = ax_hydro.transData, fontsize = fontsize_trace_label, ha = "left", va = "bottom", fontweight = "bold")

# Plot the tube-wave travel time curve
time1 = time0 + (max_depth - min_depth) / vel_tube
ax_hydro.plot([time0, time1], [min_depth, max_depth], linewidth = linewidth, color = "crimson", linestyle = "--")

ax_hydro.text(label_tube_x, label_tube_y, f"{vel_tube:.0f} {vel_unit}", transform = ax_hydro.transData, fontsize = 12, ha = "left", va = "bottom", color = "crimson")

ax_hydro.set_xlim(timeax[0], timeax[-1])
ax_hydro.set_ylim(min_depth, max_depth)
ax_hydro.set_xlabel("Time (s)")
ax_hydro.set_ylabel("Depth (m)")

ax_hydro.invert_yaxis()

fig.suptitle(f"Hammer shot {hammer_id}", fontsize = fontsize_suptitle, fontweight = "bold", y = 0.91)

###
# Save the figure
###

fig_name = f"hammer_hydro_and_geo_waveforms_{hammer_id}_{station_hydro}_{station_geo}.png"
save_figure(fig, fig_name)






