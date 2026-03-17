"""
Test the home-grown STA/LTA algorithm.
"""
#------------------------------------------------------------------------------
# Import libraries
#------------------------------------------------------------------------------
from os.path import join
from time import time
from pandas import to_timedelta

from utils_basic import ROOTDIR_GEO as dirpath_data, GEO_COMPONENTS as components, NETWORK as network, PICK_DIR as dirpath_pick
from utils_basic import get_freq_limits_string, geo_component2channel
from utils_cont_waveform import load_day_long_waveform_from_hdf
from utils_sta_lta import compute_sta_lta, pick_triggers
from utils_snuffler import write_time_windows

#------------------------------------------------------------------------------
# Input parameters
#------------------------------------------------------------------------------
day = "2020-01-15"
station = "A04"
component = "Z"
min_freq_filter = 20
max_freq_filter = None
window_length_sta = 5e-3
window_length_lta = 5e-2
on_threshold = 4.0
off_threshold = 1.0
cf_threshold_interval = 0.1

#------------------------------------------------------------------------------
# Load the data
#------------------------------------------------------------------------------
print(f"Loading the data for {day} {station} {component}")
clock1 = time()
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
filename = f"preprocessed_data_{freq_str}.h5"
filepath = join(dirpath_data, filename)
day_long_waveform= load_day_long_waveform_from_hdf(filepath, station, day)
starttime = day_long_waveform.starttime
clock2 = time()
print(f"Time taken: {clock2 - clock1} seconds")

# Compute the STA/LTA characteristic function
print(f"Computing the 3-component STA/LTA characteristic function")
for i, component in enumerate(components):
    clock3 = time()
    print(f"Computing the STA/LTA characteristic function for Component {component}")
    waveform = day_long_waveform.get_component(component)
    sampling_rate = day_long_waveform.sampling_rate
    cf = compute_sta_lta(waveform, sampling_rate, window_length_sta, window_length_lta)

    if i == 0:
        cf_stack = cf
    else:
        cf_stack = cf_stack + cf

    clock4 = time()
    print(f"Time taken: {clock4 - clock3} seconds")

cf_stack = cf_stack / len(components)

# Pick the triggers
print(f"Picking the triggers")
clock5 = time()
trigger_dict = pick_triggers(cf_stack, on_threshold, off_threshold, starttime = starttime, sampling_rate = sampling_rate)
trigger_windows = trigger_dict["event_windows"]
clock6 = time()
print(f"Time taken: {clock6 - clock5} seconds")
print(f"Identified {len(trigger_windows)} triggers")

# Assemble the output 
print(f"Saving the output...")
clock7 = time()
starttimes = [trigger_window[0] for trigger_window in trigger_windows]
endtimes = [trigger_window[1] for trigger_window in trigger_windows]
seed_ids = [f"{network}.{station}..{geo_component2channel("1")}"] * len(starttimes)
filename = f"sta_lta_home_grown_test_{day}.txt"
filepath = join(dirpath_pick, filename)
write_time_windows(filepath, starttimes, endtimes, seed_ids)
clock8 = time()
print(f"Results saved to {filepath}")
print(f"Time taken: {clock8 - clock7} seconds")