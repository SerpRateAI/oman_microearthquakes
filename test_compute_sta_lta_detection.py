"""
Test computing STA/LTA detection with the data of a geophone recorded on a specific day
"""

# Import libraries
from time import time
from os.path import join

from utils_basic import DETECTION_DIR as dirpath_detect, PICK_DIR as dirpath_pick
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_sta_lta import Events
from utils_sta_lta import run_sta_lta, triggers_to_snuffler_picks

# Define the input parameters
## Station and time
day = '2020-01-16'
station = 'A04'
component = 'Z'

## Filtering parameters

min_freq_filter = 10.0
max_freq_filter = None

## STA-LTA parameters
sta = 0.01
lta = 0.2
thr_on = 7
thr_off = 2

## Parameters for extracting the waveforms
buffer_start = 0.05
buffer_end = 0.05

## Output path
outpath_event = join(dirpath_detect, f'sta_lta_event_test_{day}.h5')
outpath_pick = join(dirpath_pick, f'sta_lta_event_test_{day}.txt')


# Read the data
print(f'Reading and preprocessing the data of {day}...')
clock1 = time()
stream = read_and_process_day_long_geo_waveforms(day, 
                                                 stations=station, 
                                                 filter = True,
                                                 filter_type = 'butter',
                                                 min_freq = min_freq_filter, max_freq = max_freq_filter)
clock2 = time()
elapsed = clock2 - clock1
print(f'Time taken: {elapsed}s')

# Run the detection
print(f'Running STA-LTA detection...')
clock1 = time()
triggers = run_sta_lta(stream, sta, lta, thr_on, thr_off)
clock2 = time()

elapsed = clock2 - clock1
print(f'Time taken: {elapsed}s')

# Create an Events object from the triggers
print(f'Creating an Events object from the triggers and the {component} component of the waveforms...')
trace = stream.select(component = component)[0]

events = Events.from_triggers(triggers, trace, buffer_start, buffer_end)

# Set the sequential IDs
events.set_sequential_ids()

# Save the Events object
print(f'Saving the events...')
events.to_hdf(outpath_event)

# Convert the triggers to a Snuffler marker file and save it
print(f'Converting the triggers to a Snuffler marker file...')
seed_id = trace.id
lines = triggers_to_snuffler_picks(triggers, seed_id)

with open(outpath_pick, "w") as file:
    for line in lines:
        file.write(line)

print(f'Picks are saved to {outpath_pick}.')


