"""
Plot the comparison of the max amplitude of the hammer and tremor events.
"""

from os.path import join
from argparse import ArgumentParser
from numpy import sqrt, amax, mean
from pandas import read_csv, Timestamp, Timedelta, DataFrame
from matplotlib.pyplot import subplots

from utils_basic import LOC_DIR as dirpath_loc, GEO_COMPONENTS as components
from utils_basic import get_geophone_coords
from utils_plot import save_figure, get_geo_component_color, component2label
from utils_plot import GROUND_VELOCITY_UNIT as vel_unit
from utils_preproc import read_and_process_windowed_geo_waveforms

parser = ArgumentParser(description = "Plot the maximum amplitude of the hammer and tremor signals recorded at a station.")

parser.add_argument("--station", type = str, help = "The station to plot")
parser.add_argument("--min_freq", type = float, help = "The minimum frequency to plot", default = 140.0)
parser.add_argument("--max_freq", type = float, help = "The maximum frequency to plot", default = 160.0)
parser.add_argument("--window_length_hammer", type = float, help = "The window length in seconds", default = 0.5)
parser.add_argument("--starttime_tremor", type = str, help = "The starttime of the tremor", default = "2020-01-13T19:50:00")
parser.add_argument("--endtime_tremor", type = str, help = "The starttime of the tremor", default = "2020-01-13T19:55:00")

args = parser.parse_args()
station = args.station
min_freq = args.min_freq
max_freq = args.max_freq
window_length_hammer = args.window_length_hammer
starttime_tremor = args.starttime_tremor
endtime_tremor = args.endtime_tremor

###
# Load the data
###

# Load the station locations
station_df = get_geophone_coords()
east_sta = station_df.loc[station, "east"]
north_sta = station_df.loc[station, "north"]

# Load the hammer information
filename = "hammer_locations.csv"
filepath = join(dirpath_loc, filename)
hammer_df = read_csv(filepath, dtype = {"hammer_id": str})

# Compute the maximum amplitudes for each hammer shot
amplitude_dicts = []
print(f"Computing the amplitudes for {len(hammer_df)} hammer shots...")
for i_hammer, row_hammer in hammer_df.iterrows():
    hammer_id = row_hammer["hammer_id"]
    north_hammer = row_hammer["north"]
    east_hammer = row_hammer["east"]
    origin_time = row_hammer["origin_time"]
    starttime = Timestamp(origin_time) 
    endtime = Timestamp(origin_time) + Timedelta(seconds = window_length_hammer)

    stream = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime, filter = True, filter_type = "butter", stations = station,
                                                    min_freq = min_freq, max_freq = max_freq)
    if len(stream) == 0:
        continue

    amplitude_dict = {"distance": sqrt((north_hammer - north_sta) ** 2 + (east_hammer - east_sta) ** 2)}
    for component in components:
        trace = stream.select(component = component)[0]
        if trace is None:
            continue
        data = trace.data
        max_amplitude = amax(abs(data))
        amplitude_dict[component] = max_amplitude
    amplitude_dicts.append(amplitude_dict)

amplitude_df = DataFrame(amplitude_dicts)

# Compute the maximum amplitudes for the tremor signals
print(f"Computing the amplitudes for the tremor signals...")
stream_tremor = read_and_process_windowed_geo_waveforms(starttime_tremor, endtime = endtime_tremor, filter = True, filter_type = "butter", stations = station,
                                                    min_freq = min_freq, max_freq = max_freq)
if len(stream_tremor) == 0:
    raise ValueError(f"No tremor data found for {station} at {starttime_tremor} to {endtime_tremor}")
amplitude_tremor_dict = {}
for component in components:
    trace = stream_tremor.select(component = component)[0]
    if trace is None:
        continue
    data = trace.data
    max_amplitude = amax(abs(data))
    mean_amplitude = mean(abs(data))
    amplitude_tremor_dict[component] = max_amplitude
    #amplitude_tremor_dict[component] = mean_amplitude

###
# Compute the maximum amplitudes for each tremor event
###
print("Plotting...")
fig, ax = subplots(1, 1)
for component in components:
    color = get_geo_component_color(component)
    ax.scatter(amplitude_df["distance"], amplitude_df[component], label = component2label(component), color = color)
    ax.axhline(amplitude_tremor_dict[component], color = color, linestyle = "--")
ax.set_xlabel("Distance (m)")
ax.set_ylabel(f"Ground velocity ({vel_unit})")
ax.set_yscale("log")
ax.legend(loc = "upper right")
ax.set_title(f"{station}, {min_freq:.0f} - {max_freq:.0f} Hz", 
             fontsize = 14,
             fontweight = "bold")

save_figure(fig, f"hammer_n_tremor_max_amplitude_comparison_{station}_{min_freq:.0f}-{max_freq:.0f}hz.png")
