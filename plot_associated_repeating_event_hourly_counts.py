"""
Plot the hourly count of the associated events
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv
from matplotlib.pyplot import figure
from matplotlib import colormaps

# Import modules
from utils_basic import DETECTION_DIR as dirpath
from utils_basic import STARTTIME_GEO as starttime_bin, ENDTIME_GEO as endtime_bin
from utils_basic import get_geophone_coords, get_freq_limits_string
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix   
from utils_plot import save_figure, format_datetime_xlabels, add_day_night_shading

# Read the information of the associated events
parser = ArgumentParser()
parser.add_argument("--group_labels", type=int, nargs="+", required=True, help="The group labels")

parser.add_argument("--min_cc", type=float, default=0.85)
parser.add_argument("--min_num_similar_snippet", type=int, default=10)
parser.add_argument("--min_num_similar_station", type=int, default=3)
parser.add_argument("--sta_window_sec", type=float, default=0.005)
parser.add_argument("--lta_window_sec", type=float, default=0.05)
parser.add_argument("--thr_on", type=float, default=4.0)
parser.add_argument("--thr_off", type=float, default=1.0)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)


args = parser.parse_args()
group_labels = args.group_labels
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

# Get the suffices
suffix = freq_limits_string = get_freq_limits_string(min_freq_filter, max_freq_filter)
sta_lta_suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix += f"_{sta_lta_suffix}"
repeating_snippet_suffix = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
suffix += f"_{repeating_snippet_suffix}"

# Read the associated events
print("Reading all associated events...")
filename = f"hourly_counts_associated_events_repeating_{suffix}.csv"
filepath = join(dirpath, filename)
count_all_df = read_csv(filepath, parse_dates=["hour"])


# Plot the hourly counts of all events
print("Plotting the hourly counts of all events...")
fig = figure(figsize=(10, 4))
ax = fig.add_subplot(111)

ax.plot(count_all_df["hour"], count_all_df["count"], color="black", label="All")

ax.set_xlim(count_all_df["hour"].min(), count_all_df["hour"].max())
ax.set_ylim(1, 500)

# Plot the hourly counts of the events of each group
print("Plotting the hourly counts of the events of the highlighted groups...")
cmap = colormaps["cet_glasbey_hv"]
for i_group, group_label in enumerate(group_labels):
    suffix_group = suffix + f"_num_sim_sta{min_num_similar_station:d}"
    filename = f"hourly_counts_grouped_events_group{group_label}_{suffix_group}.csv"
    filepath = join(dirpath, filename)
    count_group_df = read_csv(filepath, parse_dates=["hour"])
    ax.plot(count_group_df["hour"], count_group_df["count"], color=cmap(i_group), label=f"Group {group_label:d}")

print("Adding the day and night shading...")
add_day_night_shading(ax)

# format_datetime_xlabels(ax, 
#                         major_tick_spacing="1d", num_minor_ticks=4, date_format="%Y-%m-%d",
#                         axis_label_size=12, tick_label_size=10,
#                         va="top", ha="right", rotation=30)

ax.set_ylabel("Number of events per hour", fontsize=12)
ax.set_yscale("log")
ax.set_title("Hourly count of events", fontweight="bold")

ax.legend(loc="upper left", fontsize=10, framealpha=1.0)

# Save or show
save_figure(fig, f"hourly_repeating_event_count_{suffix}.png")
# plt.show()

