"""
Plot the hourly count of the associated events and the mode power of Mode 3 of the harmonic tremor
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, read_hdf
from matplotlib.pyplot import subplots
from matplotlib import colormaps

# Import modules
from utils_basic import DETECTION_DIR as dirpath_event, SPECTROGRAM_DIR as dirpath_spec
from utils_basic import STARTTIME_GEO as starttime_bin, ENDTIME_GEO as endtime_bin
from utils_basic import get_geophone_coords, get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import save_figure, format_datetime_xlabels, format_db_ylabels, add_day_night_shading

# Read the information of the associated events
parser = ArgumentParser()
## Input arguments for the associated events
parser.add_argument("--group_labels", type=int, nargs="+", required=True, help="The group labels")
parser.add_argument("--colors_group", type=str, nargs="+", required=True, help="The colors for the group labels")
parser.add_argument("--color_mode_face", type=str, required=True, help="The color for the face of the harmonic mode")
parser.add_argument("--color_mode_edge", type=str, required=True, help="The color for the edge of the harmonic mode")

parser.add_argument("--min_cc", type=float, default=0.85)
parser.add_argument("--min_num_similar_snippet", type=int, default=10)
parser.add_argument("--min_num_similar_station", type=int, default=3)
parser.add_argument("--sta_window_sec", type=float, default=0.005)
parser.add_argument("--lta_window_sec", type=float, default=0.05)
parser.add_argument("--thr_on", type=float, default=4.0)
parser.add_argument("--thr_off", type=float, default=1.0)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)

## Input arguments for the harmonic mode power
parser.add_argument("--base_mode_name", type=str, default="PR02549", help="The base mode name")
parser.add_argument("--base_mode_order", type=int, default=2, help="The base mode order")
parser.add_argument("--mode_order", type=int, default=3, help="The mode order")
parser.add_argument("--window_length", type=float, default=300.0, help="The window length in seconds")
parser.add_argument("--overlap", type=float, default=0.0, help="The overlap fraction")
parser.add_argument("--min_prom", type=float, default=15.0, help="The minimum prominence")
parser.add_argument("--min_rbw", type=float, default=15.0, help="The minimum reverse bandwidth")
parser.add_argument("--max_mean_db", type=float, default=15.0, help="The maximum mean dB value")

parser.add_argument("--station", type=str, default="A13", help="The station name")


args = parser.parse_args()
group_labels = args.group_labels
colors_group = args.colors_group
color_mode_face = args.color_mode_face
color_mode_edge = args.color_mode_edge
min_cc = args.min_cc
min_num_similar_snippet = args.min_num_similar_snippet
min_num_similar_station = args.min_num_similar_station
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
thr_on = args.thr_on
thr_off = args.thr_off
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
min_stations = args.min_num_similar_station

base_mode_name = args.base_mode_name
base_mode_order = args.base_mode_order
mode_order = args.mode_order
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
station = args.station

# Determine if the number of colors is equal to the number of group labels
if len(colors_group) != len(group_labels):
    raise ValueError("The number of colors must be equal to the number of group labels")

# Generate the figure
fig, axes = subplots(2, 1, figsize=(10, 10))
ax_event = axes[0]
ax_spec = axes[1]
fig.subplots_adjust(hspace=0.1)

# Plot the hourly counts of the events
print("Plotting the hourly event counts...")

## Generate the suffix for the associated events
suffix_event = freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix_event += f"_{sta_lta_suffix}"
repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
suffix_event += f"_{repeating_snippet_suffix}"

## Plot the hourly counts of the events in the highlighted groups
print("Plotting the hourly counts of the events of the highlighted groups...")
for i_group, group_label in enumerate(group_labels):
    suffix_group = suffix_event + f"_num_sim_sta{min_num_similar_station:d}"
    filename = f"hourly_counts_grouped_events_group{group_label}_{suffix_group}.csv"
    filepath = join(dirpath_event, filename)
    count_group_df = read_csv(filepath, parse_dates=["hour"])
    ax_event.plot(count_group_df["hour"], count_group_df["count"], color=colors_group[i_group], label=f"Group {group_label:d}")

## Add the day and night shading
print("Adding the day and night shading...")
add_day_night_shading(ax_event)

## Format the axes
ax_event.set_xlim(starttime_bin, endtime_bin)
ax_event.set_ylim(1, 500)

format_datetime_xlabels(ax_event, plot_axis_label=False, plot_tick_label=False,
                        major_tick_spacing="5d", num_minor_ticks=4, date_format="%Y-%m-%d",
                        axis_label_size=12, tick_label_size=10)

ax_event.set_ylabel("Number of events per hour", fontsize=12)
ax_event.set_yscale("log")
ax_event.set_title(f"Hourly event counts", fontweight="bold")

ax_event.legend(loc="upper left", fontsize=10, framealpha=1.0)

# Plot the harmonic mode power
print("Plotting the harmonic mode power...")

## Get the mode name
filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order}.csv"
filepath = join(dirpath_spec, filename)

harmonic_df = read_csv(filepath)

mode_name = harmonic_df.loc[harmonic_df["mode_order"] == mode_order, "mode_name"].values[0]

## Get the suffix for the stationary resonance properties
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

## Read the stationary resonance properties
filename_in = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(dirpath_spec, filename_in)
resonance_df = read_hdf(inpath, key = "properties")
resonance_df = resonance_df[ resonance_df["station"] == station]

## Plot the harmonic mode power
ax_spec.scatter(resonance_df["time"], resonance_df["total_power"], marker = "o", s = 15, color = color_mode_face, edgecolors = color_mode_edge,
            label = mode_name,
            zorder = 2)

# Add the day and night shading
print("Adding the day and night shading...")
add_day_night_shading(ax_spec)

## Format the axes
ax_spec.set_xlim(starttime_bin, endtime_bin)

format_datetime_xlabels(ax_spec, plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing="5d", num_minor_ticks=4, date_format="%Y-%m-%d",
                        axis_label_size=12, tick_label_size=10)

format_db_ylabels(ax_spec, axis_label_size=12, tick_label_size=10)

## Set the title
ax_spec.set_title(f"Mode {mode_order} power", fontweight="bold")

# Save or show
save_figure(fig, f"hourly_repeating_event_count_n_harmonic_mode_power.png")
# plt.show()

