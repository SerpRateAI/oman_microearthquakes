"""
Plot the records of a series of hammer shots on a station
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import sqrt, interp, amax
from pandas import read_csv, DataFrame, Timestamp, Timedelta
from matplotlib.pyplot import subplots

from utils_basic import LOC_DIR as dirpath_loc
from utils_basic import get_geophone_coords, GEO_COMPONENTS as components
from utils_plot import get_geo_component_color, save_figure, component2label
from utils_preproc import read_and_process_windowed_geo_waveforms

parser = ArgumentParser(description = "Plot the records of a series of hammer shots on a station")

parser.add_argument("--station", type = str, help = "The station to plot")
parser.add_argument("--hammer_ids", type = str, nargs = "+", help = "The hammer IDs to plot", default = None)
parser.add_argument("--min_freq_filter", type = float, help = "The minimum frequency to filter", default = 100.0)
parser.add_argument("--max_freq_filter", type = float, help = "The maximum frequency to filter", default = 200.0)
parser.add_argument("--begin_time", type = float, help = "The beginning time with respect to the origin time to plot", default = -0.2)
parser.add_argument("--end_time", type = float, help = "The end time with respect to the origin time to plot", default = 1.0)
parser.add_argument("--distance_scale_factor", type = float, help = "The scale factor for the distance", default = 0.5)

args = parser.parse_args()

station = args.station
hammer_ids = args.hammer_ids
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
begin_time = args.begin_time
end_time = args.end_time
distance_scale_factor = args.distance_scale_factor

figwidth = 15.0
figheight = 15.0
fontsize_title = 14

###
# Load the data
###

# Load the station locations
station_df = get_geophone_coords()
east_sta = station_df.loc[station, "east"]
north_sta = station_df.loc[station, "north"]

# Load the hammer locations
filename = f"hammer_locations.csv"
filepath = join(dirpath_loc, filename)
hammer_df = read_csv(filepath, dtype = {"hammer_id": str}, parse_dates = ["origin_time"])

if hammer_ids is not None:
    hammer_df = hammer_df[hammer_df["hammer_id"].isin(hammer_ids)]

# Load the records for each hammer
stream_dict = {}
stream_filtered_dict = {}
print(f"Loading the records for {len(hammer_df)} hammer shots...")
for _, row_hammer in hammer_df.iterrows():
    hammer_id = row_hammer["hammer_id"]
    origin_time = row_hammer["origin_time"]
    starttime = Timestamp(origin_time) + Timedelta(seconds = begin_time)
    endtime = Timestamp(origin_time) + Timedelta(seconds = end_time)

    stream_dict[hammer_id] = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime)
    stream_filtered_dict[hammer_id] = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime, filter = True, filter_type = "butter",
                                                                                min_freq = min_freq_filter, max_freq = max_freq_filter)


# Make the plots for each component
print(f"Making the plots for each component...")
fig, axs = subplots(1, 3, figsize = (figwidth, figheight))
for i, component in enumerate(components):
    print(f"Making the plots for {component} component...")
    ax = axs[i]
    
    for hammer_id in stream_dict.keys():
        north_hammer = hammer_df.loc[hammer_df["hammer_id"] == hammer_id, "north"].values[0]
        east_hammer = hammer_df.loc[hammer_df["hammer_id"] == hammer_id, "east"].values[0]
        distance = sqrt((north_hammer - north_sta) ** 2 + (east_hammer - east_sta) ** 2)

        stream = stream_dict[hammer_id]
        stream_filtered = stream_filtered_dict[hammer_id]
        trace = stream.select(component = component)[0]
        trace_filtered = stream_filtered.select(component = component)[0]

        timeax = trace.times()
        data = trace.data
        data_filtered = trace_filtered.data

        data = data / amax(abs(data)) * distance_scale_factor + distance
        data_filtered = data_filtered / amax(abs(data_filtered)) * distance_scale_factor + distance

        ax.plot(timeax, data, color = "gray")
        ax.plot(timeax, data_filtered, color = get_geo_component_color(component), alpha = 0.5)

        ax.set_xlim(timeax[0], timeax[-1])

    ax.set_title(f"{component2label(component)}", fontsize = fontsize_title, fontweight = "bold")
    ax.set_xlabel("Time (s)")

    if i == 0:
        ax.set_ylabel("Distance (m)")
    else:
        ax.set_ylabel("")

fig.suptitle(f"{station}, {min_freq_filter:.0f} - {max_freq_filter:.0f} Hz", fontsize = fontsize_title, fontweight = "bold", y = 0.92)
save_figure(fig, f"hammer_record_section_{station}_{min_freq_filter:.0f}-{max_freq_filter:.0f}hz.png")
