"""
Plot the frequency of a stationary resonance and the air temperature as a function of time
"""

######
# Imports
######

from os.path import join
from argparse import ArgumentParser
from pandas import Timestamp, Timedelta
from pandas import read_hdf
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.dates import date2num

from utils_basic import SPECTROGRAM_DIR as dirpath_spec, STARTTIME_HYDRO as starttime, ENDTIME_HYDRO as endtime
from utils_basic import get_baro_temp_data, get_mode_order
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import format_datetime_xlabels, save_figure, add_zoom_effect, add_day_night_shading

######
# Inputs
######

parser = ArgumentParser()
parser.add_argument("--mode_name", type=str, default="PR02549", help="The mode name")
parser.add_argument("--starttime_zoom1", type=str, default="2019-06-30", help="The start time of the first zoom-in window")
parser.add_argument("--starttime_zoom2", type=str, default="2020-01-21", help="The start time of the second zoom-in window")
parser.add_argument("--window_length_zoom", type=str, default="5d", help="The window length of the zoom-in windows")

parser.add_argument("--window_length_spec", type=float, default=300.0, help="The window length of the spectrogram")
parser.add_argument("--overlap", type=float, default=0.0, help="The overlap")
parser.add_argument("--min_prom", type=float, default=15.0, help="The minimum prominence")
parser.add_argument("--min_rbw", type=float, default=15.0, help="The minimum rbw")
parser.add_argument("--max_mean_db", type=float, default=-10.0, help="The maximum mean db")

parser.add_argument("--figwidth", type=float, default=12, help="The width of the figure")
parser.add_argument("--figheight", type=float, default=10, help="The height of the figure")

parser.add_argument("--color_temp", type=str, default="tab:blue", help="The color of the temperature")
parser.add_argument("--color_freq", type=str, default="tab:orange", help="The color of the frequency")
parser.add_argument("--color_zoom", type=str, default="crimson", help="The color of the zoom frame")

parser.add_argument("--linewidth_thick", type=float, default=0.5, help="The linewidth of the thick line")
parser.add_argument("--linewidth_thin", type=float, default=0.1, help="The linewidth of the thin line")
parser.add_argument("--linewidth_zoom", type=float, default=2.0, help="The linewidth of the zoom frame")

parser.add_argument("--markersize_data", type=float, default=0.5, help="The markersize of the data")

parser.add_argument("--max_temp", type=float, default=44.0, help="The maximum temperature")
parser.add_argument("--min_temp", type=float, default=19.0, help="The minimum temperature")
parser.add_argument("--max_freq", type=float, default=25.72, help="The maximum frequency")
parser.add_argument("--min_freq", type=float, default=25.34, help="The minimum frequency")

parser.add_argument("--major_time_spacing_full", type=str, default="30d", help="The major time spacing")
parser.add_argument("--num_minor_ticks_full", type=int, default=6, help="The number of minor ticks")
parser.add_argument("--major_time_spacing_zoom", type=str, default="5d", help="The major time spacing")
parser.add_argument("--num_minor_ticks_zoom", type=int, default=5, help="The number of minor ticks")
parser.add_argument("--rotation_zoom", type=float, default=15.0, help="The rotation of the zoom labels")
parser.add_argument("--fontsize_axis_label", type=int, default=12, help="The fontsize of the axis label")
parser.add_argument("--fontsize_tick_label", type=int, default=10, help="The fontsize of the tick label")

parser.add_argument("--fontsize_title", type=int, default=14, help="The fontsize of the title")

args = parser.parse_args()
mode_name = args.mode_name
starttime_zoom1 = Timestamp(args.starttime_zoom1, tz = "UTC")
starttime_zoom2 = Timestamp(args.starttime_zoom2, tz = "UTC")
window_length_zoom = Timedelta(args.window_length_zoom)
window_length_spec = args.window_length_spec
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
figwidth = args.figwidth
figheight = args.figheight
color_temp = args.color_temp
color_freq = args.color_freq
color_zoom = args.color_zoom
linewidth_thick = args.linewidth_thick
linewidth_thin = args.linewidth_thin
linewidth_zoom = args.linewidth_zoom
markersize_data = args.markersize_data
max_temp = args.max_temp
min_temp = args.min_temp
max_freq = args.max_freq
min_freq = args.min_freq
major_time_spacing_full = args.major_time_spacing_full
num_minor_ticks_full = args.num_minor_ticks_full
major_time_spacing_zoom = args.major_time_spacing_zoom
num_minor_ticks_zoom = args.num_minor_ticks_zoom
fontsize_axis_label = args.fontsize_axis_label
fontsize_tick_label = args.fontsize_tick_label
rotation_zoom = args.rotation_zoom
fontsize_title = args.fontsize_title

######
# Load the data
######

# Read the the temperature data
print("Reading the temperature data")
baro_temp_df = get_baro_temp_data()
times_temp = baro_temp_df.index
temps = baro_temp_df["temperature"]

# Read the frequency data
print("Reading the resonance frequency data")
suffix_spec = get_spectrogram_file_suffix(window_length_spec, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_resonance_profile_hydro_smooth_filter_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(dirpath_spec, filename)
reson_df = read_hdf(filepath, key = "profile")

times_reson = reson_df["time"]
freqs = reson_df["frequency"]

######
# Plotting
######
print("Plotting...")

# Initialize the figure
print("Initializing the figure")
fig = figure(figsize = (figwidth, figheight))
gs = GridSpec(2, 2, figure = fig)

# Plot the temperature data
print("Plotting the temperature data")
ax_temp = fig.add_subplot(gs[0, :])
ax_temp.plot(times_temp, temps, color = color_temp, label = "Temperature", linewidth = linewidth_thin)

# Add a twin axis for the frequency data
print("Adding a twin axis for the frequency data")
ax_freq = ax_temp.twinx()

# Plot the resonance frequency data
print("Plotting the resonance frequency data")
ax_freq.plot(times_reson, freqs, color = color_freq, label = "Frequency", linewidth = linewidth_thin)

# Color the left y-axis
print("Coloring the left y-axis")
ax_temp.set_ylabel("Temperature (°C)", color = color_temp, fontsize = fontsize_axis_label)
ax_temp.tick_params(axis = "y", colors = color_temp, labelsize = fontsize_tick_label)
ax_temp.set_ylim(min_temp, max_temp)

# Color the right y-axis
print("Coloring the right y-axis")
ax_freq.set_ylabel("Frequency (Hz)", color = color_freq, fontsize = fontsize_axis_label)
ax_freq.tick_params(axis = "y", colors = color_freq, labelsize = fontsize_tick_label)
ax_freq.set_ylim(min_freq, max_freq)

# Set the x-axis limits
print("Setting the x-axis limits")
ax_temp.set_xlim(starttime, endtime)

# Format the x-axis
print("Formatting the x-axis")
format_datetime_xlabels(ax_temp,
                        major_tick_spacing = major_time_spacing_full,
                        num_minor_ticks = num_minor_ticks_full,
                        date_format = "%Y-%m-%d",
                        axis_label_size = fontsize_axis_label,
                        tick_label_size = fontsize_tick_label)

# Add the title
mode_order = get_mode_order(mode_name)
title = f"Mode {mode_order} frequency and air temperature"
ax_temp.set_title(title, fontsize = fontsize_title, fontweight = "bold")

# Add the first zoom-in plot
print("Adding the first zoom-in plot")
ax_temp_zoom1 = fig.add_subplot(gs[1, 0])

# Plot the temperature data in the first zoom-in plot
print("Plotting the temperature data in the first zoom-in plot")
endtime_zoom = starttime_zoom1 + window_length_zoom
times_temp_zoom = times_temp[(times_temp >= starttime_zoom1) & (times_temp <= endtime_zoom)]
temps_zoom = temps[(times_temp >= starttime_zoom1) & (times_temp <= endtime_zoom)]

ax_temp_zoom1.plot(times_temp_zoom, temps_zoom, color = color_temp, label = "Temperature", linewidth = linewidth_thick)

# Plot the resonance frequency data in the first zoom-in plot
print("Plotting the resonance frequency data in the first zoom-in plot")
ax_freq_zoom1 = ax_temp_zoom1.twinx()
times_reson_zoom = times_reson[(times_reson >= starttime_zoom1) & (times_reson <= endtime_zoom)]
freqs_zoom = freqs[(times_reson >= starttime_zoom1) & (times_reson <= endtime_zoom)]

ax_freq_zoom1.plot(times_reson_zoom, freqs_zoom, color = color_freq, label = "Frequency", linewidth = linewidth_thick)

# Format the first zoom-in plot
print("Formatting the first zoom-in plot")
ax_temp_zoom1.set_xlim(starttime_zoom1, endtime_zoom)

# Color the left y-axis
print("Coloring the left y-axis")
ax_temp_zoom1.tick_params(axis = "y", colors = color_temp, labelsize = fontsize_tick_label)

# Remove the right y tick labels
print("Removing the right y tick labels")
ax_freq_zoom1.set_yticklabels([])

# Set the x-axis limits
print("Setting the x-axis limits")
ax_temp_zoom1.set_xlim(starttime_zoom1, endtime_zoom)

# Set the y-axis limits
print("Setting the y-axis limits")
ax_temp_zoom1.set_ylim(min_temp, max_temp)
ax_freq_zoom1.set_ylim(min_freq, max_freq)

# Set the x-axis labels
format_datetime_xlabels(ax_temp_zoom1,
                        major_tick_spacing = major_time_spacing_zoom,
                        num_minor_ticks = num_minor_ticks_zoom,
                        date_format = "%Y-%m-%d",
                        axis_label_size = fontsize_axis_label,
                        tick_label_size = fontsize_tick_label,
                        va = "top", ha = "right",
                        rotation = rotation_zoom)

# Add the day-night shading
print("Adding the day-night shading")
add_day_night_shading(ax_temp_zoom1)

# Add the zoom-in effect
print("Adding the zoom-in effect")
prop_lines = {"color": color_zoom, "linewidth": linewidth_zoom}
prop_patches = {"edgecolor": color_zoom, "linewidth": linewidth_zoom, "facecolor": "none", "zorder": 10}
add_zoom_effect(ax_temp, ax_temp_zoom1, date2num(starttime_zoom1), date2num(endtime_zoom), prop_lines, prop_patches)

# Add the title
print("Adding the title")
ax_temp_zoom1.set_title("Summer", fontsize = fontsize_title, fontweight = "bold")

# Add the second zoom-in plot
print("Adding the second zoom-in plot")
ax_temp_zoom2 = fig.add_subplot(gs[1, 1])

# Plot the temperature data in the second zoom-in plot
print("Plotting the temperature data in the second zoom-in plot")
endtime_zoom = starttime_zoom2 + window_length_zoom
times_temp_zoom = times_temp[(times_temp >= starttime_zoom2) & (times_temp <= endtime_zoom)]
temps_zoom = temps[(times_temp >= starttime_zoom2) & (times_temp <= endtime_zoom)]

ax_temp_zoom2.plot(times_temp_zoom, temps_zoom, color = color_temp, label = "Temperature", linewidth = linewidth_thick)

# Plot the resonance frequency data in the second zoom-in plot
print("Plotting the resonance frequency data in the second zoom-in plot")
ax_freq_zoom2 = ax_temp_zoom2.twinx()
times_reson_zoom = times_reson[(times_reson >= starttime_zoom2) & (times_reson <= endtime_zoom)]
freqs_zoom = freqs[(times_reson >= starttime_zoom2) & (times_reson <= endtime_zoom)]

ax_freq_zoom2.plot(times_reson_zoom, freqs_zoom, color = color_freq, label = "Frequency", linewidth = linewidth_thick)

# Format the x-axis of the second zoom-in plot
print("Formatting the x-axis")
ax_temp_zoom2.set_xlim(starttime_zoom2, endtime_zoom)

# Remove the left y tick labels
print("Removing the left y tick labels")
ax_temp_zoom2.set_yticklabels([])

# Color the right y-axis
print("Coloring the right y-axis")
ax_freq_zoom2.tick_params(axis = "y", colors = color_freq, labelsize = fontsize_tick_label)

# Set the x-axis limits
print("Setting the x-axis limits")
ax_temp_zoom2.set_xlim(starttime_zoom2, endtime_zoom)

# Set the y-axis limits
print("Setting the y-axis limits")
ax_temp_zoom2.set_ylim(min_temp, max_temp)
ax_freq_zoom2.set_ylim(min_freq, max_freq)

# Set the x-axis labels
format_datetime_xlabels(ax_temp_zoom2,
                        major_tick_spacing = major_time_spacing_zoom,
                        num_minor_ticks = num_minor_ticks_zoom,
                        date_format = "%Y-%m-%d",
                        axis_label_size = fontsize_axis_label,
                        tick_label_size = fontsize_tick_label,
                        va = "top", ha = "right",
                        rotation = rotation_zoom)

# Add the day-night shading
print("Adding the day-night shading")
add_day_night_shading(ax_temp_zoom2)

# Add the zoom-in effect
print("Adding the zoom-in effect")
prop_lines = {"color": color_zoom, "linewidth": linewidth_zoom}
prop_patches = {"edgecolor": color_zoom, "linewidth": linewidth_zoom, "facecolor": "none", "zorder": 10}
add_zoom_effect(ax_temp, ax_temp_zoom2, date2num(starttime_zoom2), date2num(endtime_zoom), prop_lines, prop_patches)

# Add the title
print("Adding the title")
ax_temp_zoom2.set_title("Winter", fontsize = fontsize_title, fontweight = "bold")

# Save the figure
print("Saving the figure")
save_figure(fig, f"stationary_resonance_freq_n_temperature_vs_time_{mode_name}.png")







