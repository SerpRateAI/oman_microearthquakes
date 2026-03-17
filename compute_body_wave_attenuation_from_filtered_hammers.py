"""
Compute the body wave attenuation from the filtered hammer shot waveforms
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import sqrt, interp, amax, abs
from pandas import read_csv, DataFrame, Timestamp, Timedelta
from matplotlib.pyplot import subplots

from utils_basic import LOC_DIR as dirpath_loc, PICK_DIR as dirpath_pick
from utils_basic import get_geophone_coords, GEO_COMPONENTS as components, timestamp_to_utcdatetime, power2db
from utils_plot import save_figure
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_snuffler import read_time_windows

parser = ArgumentParser(description = "Compute the body wave attenuation from the filtered hammer shot waveforms")

parser.add_argument("--station", type = str, help = "The station to compute the body wave attenuation from")
parser.add_argument("--begin", type = float, help = "Begin time in second relative to the P pick", default = -0.2)
parser.add_argument("--end", type = float, help = "End time in second relative to the P pick", default = 1.0)
parser.add_argument("--hammer_ids", type = str, nargs = "+", help = "The hammer IDs to compute the body wave attenuation from", default = None)
parser.add_argument("--min_freq_filter", type = float, help = "The minimum frequency to filter", default = 100.0)
parser.add_argument("--max_freq_filter", type = float, help = "The maximum frequency to filter", default = 200.0)
args = parser.parse_args()

station = args.station
hammer_ids = args.hammer_ids
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
begin = args.begin
end = args.end

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

# Load the hammer P windows
filename = f"hammer_p_windows_{station}.mkr"
path_in = join(dirpath_pick, filename)
hammer_p_windows_df = read_time_windows(path_in)


# Load the records for each hammer and measure the amplitude
power_distance_dicts = []
print(f"Loading computing the powers for {len(hammer_df)} hammer shots...")
for i_hammer, window_row in hammer_p_windows_df.iterrows():
    starttime_window = window_row["starttime"]
    endtime_window = window_row["endtime"]
    # Read the hammer waveform
    stream = read_and_process_windowed_geo_waveforms(starttime_window, endtime = endtime_window, filter = True, filter_type = "butter", stations = station, components = "Z",
                                                    min_freq = min_freq_filter, max_freq = max_freq_filter)
    trace = stream[0]

    if len(row_station) == 0:
        continue

    starttime_window = row_station["starttime"].values[0]
    endtime_window = row_station["endtime"].values[0]

    trace.trim(timestamp_to_utcdatetime(starttime_window), timestamp_to_utcdatetime(endtime_window))
    data = trace.data
    max_power = amax(abs(data)) ** 2
    max_power_db = power2db(max_power)

    # Compute the distance
    north_hammer = hammer_df.loc[hammer_df["hammer_id"] == hammer_id, "north"].values[0]
    east_hammer = hammer_df.loc[hammer_df["hammer_id"] == hammer_id, "east"].values[0]
    distance = sqrt((north_hammer - north_sta) ** 2 + (east_hammer - east_sta) ** 2)

    power_distance_dicts.append({"distance": distance, "power": max_power_db})

power_distance_df = DataFrame(power_distance_dicts)

# Plot the results
print("Plotting ")
fig, ax = subplots(1, 1)
ax.scatter(power_distance_df["distance"], power_distance_df["power"])
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Power (dB)")

save_figure(fig, f"body_wave_power_vs_distance_{station}.png")     



