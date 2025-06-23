"""
Compute the STA/LTA detection for a given station
"""

# Import libraries
from os.path import join
from argparse import ArgumentParser
from time import time

from utils_basic import DETECTION_DIR as dirpath_detect, PICK_DIR as dirpath_pick, NETWORK as network, HAMMER_DAY as hammer_day
from utils_basic import get_geophone_days
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_sta_lta import Snippets
from utils_sta_lta import run_sta_lta, snippets_to_snuffler_picks

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--station", type=str, required=True)
parser.add_argument("--min_freq_filter", type=float, default=10.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--filter_type", type=str, default='butter')

parser.add_argument("--sta", type=float, default=0.01)
parser.add_argument("--lta", type=float, default=0.2)

parser.add_argument("--thr_on", type=float, default=7.0)
parser.add_argument("--thr_off", type=float, default=2.0)

parser.add_argument("--buffer_start", type=float, default=0.05)
parser.add_argument("--buffer_end", type=float, default=0.05)

args = parser.parse_args()

station = args.station
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
filter_type = args.filter_type
sta = args.sta
lta = args.lta
thr_on = args.thr_on
thr_off = args.thr_off
buffer_start = args.buffer_start
buffer_end = args.buffer_end


# Define the output paths
outpath_event = join(dirpath_detect, f'raw_sta_lta_detections_{station}_on{thr_on:.1f}.h5')
outpath_pick = join(dirpath_pick, f'raw_sta_lta_detections_{station}_on{thr_on:.1f}.txt')
seed_id = f'{network}.{station}.GHZ'

# Compute the detection for each day
print(f'Computing the STA/LTA detection for {station}...')
detections_all = Snippets(station)
lines_all = []
for day in get_geophone_days():
    if day == hammer_day:
        print('--------------------------------')
        print(f'Skipping the day {day} when the active source experiment was conducted...')
        print('--------------------------------')
        continue

    print('--------------------------------')
    print(f'Computing the STA/LTA detection for {day}...')
    print('--------------------------------')

    # Read the waveforms
    print(f'Reading and preprocessing the data of {day}...')
    clock1 = time()
    stream = read_and_process_day_long_geo_waveforms(day, 
                                                        stations=station,
                                                        filter=True,
                                                        filter_type=filter_type,
                                                        min_freq=min_freq_filter,
                                                        max_freq=max_freq_filter)
    
    if stream is None:
        print(f'No data found for {day}...')
        continue

    clock2 = time()
    elapsed = clock2 - clock1
    print(f'Time taken to read and process the waveforms: {elapsed}s')

    # Compute the STA/LTA detection
    print(f'Running STA-LTA detection...')
    clock1 = time()
    triggers = run_sta_lta(stream, sta, lta, thr_on, thr_off)
    clock2 = time()
    elapsed = clock2 - clock1
    print(f'Time taken to compute the STA/LTA detection: {elapsed}s')

    # Convert the triggers to events
    print(f'Creating an Events object from the triggers and the waveforms...')
    detections = Snippets.from_triggers(triggers, stream, buffer_start, buffer_end)

    # Extend the events
    detections_all.extend(detections)

    # Convert the triggers to picks
    print('Converting the triggers to picks...')
    lines = snippets_to_snuffler_picks(triggers, seed_id, buffer_start, buffer_end)
    lines_all.extend(lines)

    print('--------------------------------')
    print(f'STA/LTA detection for {day} completed!')
    print('--------------------------------')

num_detections = len(detections_all)
print(f'Total number of detections: {num_detections}')

# Set the sequential IDs
detections_all.set_sequential_ids()

# Save the events
print(f'Saving the events to {outpath_event}...')
detections_all.to_hdf(outpath_event)

# Save the picks
print(f'Saving the picks to {outpath_pick}...')
with open(outpath_pick, 'w') as f:
    f.writelines(lines_all)

print(f'STA/LTA detection for {station} completed!')