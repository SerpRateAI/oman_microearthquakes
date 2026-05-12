"""
Plot the MT spectra of a hub event.
"""
#------------------------------------------------------------------------------
# Import packages
#------------------------------------------------------------------------------

import argparse
from pathlib import Path
from pandas import read_json, to_datetime, read_csv, Timedelta
from os.path import join
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots

from utils_basic import DETECTION_DIR as dirpath_event, GEO_COMPONENTS as components, INNER_STATIONS_A as stations_inner_a, INNER_STATIONS_B as stations_inner_b, ROOTDIR_GEO as dirpath_waveform, SAMPLING_RATE as sampling_rate
from utils_basic import get_freq_limits_string, power2db
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_autospec
from utils_sta_lta import get_sta_lta_suffix
from utils_cc import get_repeating_snippet_suffix
from utils_cont_waveform import load_waveform_slice
from utils_plot import get_geo_component_color, save_figure, format_db_ylabels, format_freq_xlabels
#--------------------------------------------------------------------------------------------------
# Define the functions
#--------------------------------------------------------------------------------------------------
"""
Convert the per-station times to datetime objects
"""
def convert_station_times(station_dict):
    return {
        station: to_datetime(times, utc=True, format="ISO8601")
        for station, times in station_dict.items()
    }

#------------------------------------------------------------------------------
# Parse command-line arguments
#------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Plot the MT spectra of a hub event")
group_label = parser.add_argument("--group_label", type=int, help="Group label", required=True)

parser.add_argument("--min_cc", type=float, help="Minimum correlation coefficient", default=0.85)
parser.add_argument("--min_num_similar_snippet", type=int, help="Minimum number of similar snippets", default=10)
parser.add_argument("--min_num_similar_station", type=int, help="Minimum number of similar stations", default=3)
parser.add_argument("--sta_window_sec", type=float, help="Station window length in seconds", default=0.005)
parser.add_argument("--lta_window_sec", type=float, help="Long-term average window length in seconds", default=0.05)
parser.add_argument("--thr_on", type=float, help="On threshold", default=4.0)
parser.add_argument("--thr_off", type=float, help="Off threshold", default=1.0)
parser.add_argument("--min_freq_filter", type=float, help="Minimum frequency for filtering", default=20.0)
parser.add_argument("--max_freq_filter", type=float, help="Maximum frequency for filtering", default=None)
parser.add_argument("--buffer_before_sec", type=float, help="Buffer before the first onset time in seconds", default=0.05)
parser.add_argument("--buffer_after_sec", type=float, help="Buffer after the first onset time in seconds", default=0.2)
parser.add_argument("--nw", type=int, help="Time-bandwidth product", default=3)
parser.add_argument("--min_db", type=float, help="Minimum decibel to plot", default=-20.0)
parser.add_argument("--max_db", type=float, help="Maximum decibel to plot", default=40.0)

args = parser.parse_args()
group_label = args.group_label
nw = args.nw
min_cc = args.min_cc
min_num_similar_snippet = args.min_num_similar_snippet
min_num_similar_station = args.min_num_similar_station
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
thr_on = args.thr_on
thr_off = args.thr_off
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
buffer_before_sec = args.buffer_before_sec
buffer_after_sec = args.buffer_after_sec
min_db = args.min_db
max_db = args.max_db
# Constants
figwidth = 15.0
figheight = 15.0
linewidth = 1.5

#------------------------------------------------------------------------------
# Read the hub event waveforms
#------------------------------------------------------------------------------

# Build the suffixes
suffix_freq = get_freq_limits_string(min_freq_filter, max_freq_filter)
suffix_sta_lta = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
suffix_repeating_snippet = get_repeating_snippet_suffix(min_cc = min_cc, min_num_similar = min_num_similar_snippet)
suffix = f"{suffix_freq}_{suffix_sta_lta}_{suffix_repeating_snippet}"
suffix_group = f"{suffix}_num_sim_sta{min_num_similar_station:d}"

# Get the event alignment information
print("Getting the event alignment information...")
filename = f"event_alignments_group{group_label:d}_{suffix_group}.csv"
filepath = join(dirpath_event, filename)
alignment_df = read_csv(filepath)
alignment_df["aligned_first_onset"] = to_datetime(alignment_df["aligned_first_onset"], format="ISO8601")
alignment_df["hub"] = alignment_df["hub"].astype(bool)
num_events = len(alignment_df)

# Get the first-onset time and station of the hub event
first_onset_time = alignment_df.loc[alignment_df["hub"] == True, "aligned_first_onset"].values[0]
first_onset_station = alignment_df.loc[alignment_df["hub"] == True, "first_onset_station"].values[0]
event_id = alignment_df.loc[alignment_df["hub"] == True, "id"].values[0]

# Read the waveform data
if first_onset_station.startswith("A"):
    stations_to_plot = stations_inner_a
elif first_onset_station.startswith("B"):
    stations_to_plot = stations_inner_b

print(stations_to_plot)
filename = f"preprocessed_data.h5"
filepath = join(dirpath_waveform, filename)
starttime = first_onset_time - Timedelta(seconds=buffer_before_sec)
endtime = first_onset_time + Timedelta(seconds=buffer_after_sec)
print(starttime, endtime)
waveform_dict = {}
stream = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime, stations = stations_to_plot)

# Compute the MT spectra
mt_spectra_dict = {}
for station in stations_to_plot:
    mt_spectra_sta_dict = {}
    for component in components:
        trace = stream.select(station=station, component=component)[0]
        waveform = trace.data
        num_pts = len(waveform)
        taper_mat, ratio_vec = dpss(num_pts, nw, 2 * nw - 1, return_ratios=True)
        aspec_param = mt_autospec(waveform, taper_mat, ratio_vec, sampling_rate)
        aspec = aspec_param.aspec
        freqax = aspec_param.freqax
        aspec_db = power2db(aspec)
        
        mt_spectra_sta_dict[component] = aspec_db

    mt_spectra_dict[station] = mt_spectra_sta_dict

# Plot the MT spectra
fig, axs = subplots(3, 1, figsize=(figwidth, figheight))
for i, component in enumerate(components):
    ax = axs[i]
    for station in stations_to_plot:
        aspec_db = mt_spectra_dict[station][component]
        color = get_geo_component_color(component)
        ax.plot(freqax, aspec_db, color=color, linewidth=linewidth)

    format_db_ylabels(ax)

    if i == 2:
        format_freq_xlabels(ax, plot_axis_label=True, plot_tick_label=True,
                            major_tick_spacing=100.0, num_minor_ticks=5)
    else:
        format_freq_xlabels(ax, plot_axis_label=False, plot_tick_label=False,
                            major_tick_spacing=100.0, num_minor_ticks=5)

    ax.set_xlim(freqax[0], freqax[-1])
    ax.set_ylim(min_db, max_db)

# Set the suptitle
fig.suptitle(f"Group {group_label:d}, Hub Event {event_id}", fontsize=16, fontweight="bold", y=0.9)

# Save the figure
figname = f"hub_event_mt_spectra_group{group_label:d}.png"
save_figure(fig, figname)