"""
Plot the hourly count of the associated events
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser
from pandas import read_json, to_datetime, date_range, cut
from numpy import histogram2d, linspace
from matplotlib.pyplot import figure

# Import modules
from utils_basic import DETECTION_DIR as dirpath
from utils_basic import STARTTIME_GEO as starttime_bin, ENDTIME_GEO as endtime_bin
from utils_basic import get_geophone_coords, get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix   
from utils_plot import save_figure, format_datetime_xlabels, add_day_night_shading

# Read the information of the associated events
parser = ArgumentParser()
parser.add_argument("--group_label", type=int, required=True, help="The group label")

parser.add_argument("--min_cc", type=float, default=0.85)
parser.add_argument("--min_num_similar_snippet", type=int, default=10)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--sta_window_sec", type=float, default=0.005)
parser.add_argument("--lta_window_sec", type=float, default=0.05)
parser.add_argument("--on_threshold", type=float, default=4.0)
parser.add_argument("--off_threshold", type=float, default=1.0)
parser.add_argument("--min_num_similar_station", type=int, default=3)


args = parser.parse_args()
group_label = args.group_label
min_cc = args.min_cc
min_num_similar_snippet = args.min_num_similar_snippet
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
on_threshold = args.on_threshold
off_threshold = args.off_threshold
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
min_num_similar_station = args.min_num_similar_station

# Get the suffices
suffix = freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, on_threshold, off_threshold)
suffix += f"_{sta_lta_suffix}"
repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
suffix += f"_{repeating_snippet_suffix}"
suffix += f"_num_sim_sta{min_num_similar_station:d}"

# Read the associated events
print("Reading the associated events...")
filename = f"grouped_events_group{group_label}_{suffix}.jsonl"
filepath = join(dirpath, filename)
event_df = read_json(filepath, lines = True)

event_df["first_onset"] = to_datetime(
    event_df["first_onset"],
    errors="coerce"
)
# Bin by hour
# Define the bin edges
bin_edges = date_range(starttime_bin, endtime_bin, freq="h")
bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

# Bin the events by hour
event_df["hour"] = cut(event_df["first_onset"], bins=bin_edges, labels=bin_centers)

# Count the number of events per hour
hourly_counts = event_df.groupby("hour").size().reindex(bin_centers, fill_value=0)

# Plot
fig = figure(figsize=(10, 4))
ax = fig.add_subplot(111)

ax.plot(bin_centers, hourly_counts.values)

ax.set_xlim(bin_centers.min(), bin_centers.max())
ax.set_ylim(1, 500)

add_day_night_shading(ax)

format_datetime_xlabels(ax, 
                        major_tick_spacing="1d", num_minor_ticks=4, date_format="%Y-%m-%d",
                        axis_label_size=12, tick_label_size=10,
                        va="top", ha="right", rotation=30)

ax.set_ylabel("Number of events per hour", fontsize=12)

ax.set_yscale("log")
ax.set_title(f"Hourly count of Event Group {group_label}", fontweight="bold")

# Save or show
save_figure(fig, f"hourly_event_count_group{group_label}_{suffix}.png")
# plt.show()

