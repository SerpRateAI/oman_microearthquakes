"""
This script computes the cross-correlation of low-frequency events between two hydrophone locations
"""


# Import necessary modules

from os.path import join
from argparse import ArgumentParser
from json import loads
from pandas import Timestamp
from obspy.signal.cross_correlation import correlate_template, xcorr_max

from utils_basic import HYDRO_LOCATIONS as loc_dict, HYDRO_DEPTHS as depth_dict, SAMPLING_RATE as sampling_rate
from utils_preproc import read_and_process_windowed_hydro_waveforms

# Parse command-line arguments

parser = ArgumentParser()
parser.add_argument("--station", type=str, required=True, help="Station to compute the cross-correlation for")
parser.add_argument("--starttime", type=Timestamp, required=True, help="Start time of the waveforms in UTC")
parser.add_argument("--duration", type=float, required=True, help="Duration of the waveforms in seconds")

parser.add_argument("--max_freq_filter", type=float, help="Maximum frequency of the filter", default=10.0)
parser.add_argument("--locations_to_plot", type=str, help="Locations to plot the waveforms", default='["01","02"]')

args = parser.parse_args()
station = args.station
starttime = args.starttime
duration = args.duration
max_freq_filter = args.max_freq_filter
locations_to_plot = loads(args.locations_to_plot)

# Read the waveforms
print(f"Reading the waveforms for {station}...")
stream = read_and_process_windowed_hydro_waveforms(starttime,
                                                    dur = duration,
                                                    stations = station,
                                                    filter = True, filter_type = "butter", min_freq = None, max_freq = max_freq_filter)

# Compute the cross-correlation
locations = loc_dict[station]
num_loc = len(locations)

print(f"Computing the cross-correlation between {num_loc} locations...")
for i, location_template in enumerate(locations):
    for j in range(i + 1, num_loc):
        location_target = locations[j]
        print(f"Computing the cross-correlation between {location_template} and {location_target}")

        trace_template = stream.select(location = location_template)[0]
        trace_target = stream.select(location = location_target)[0]

        xcorr = correlate_template(trace_target, trace_template)
        numpts_shift, xcorr_max_value = xcorr_max(xcorr)
        time_shift = numpts_shift / sampling_rate

        print(f"The maximum cross-correlation value is {xcorr_max_value} at a lag of {time_shift} seconds")
        print("")

        # depth_template = depth_dict[location_template]
        # depth_target = depth_dict[location_target]
        # app_vel = (depth_target - depth_template) / time_shift
        # print(f"The estimated apparent velocity is {app_vel} m/s")
        # print("")





