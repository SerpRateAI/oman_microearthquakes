"""
Compute the multitaper inter-hydrophone location phase differences for a hammer shot
"""
###
# Imports
###
from os.path import join
from argparse import ArgumentParser
from scipy.signal.windows import dpss
from json import dumps
from numpy import array, ndarray
from pandas import read_csv, Timedelta, Timestamp
from pandas import DataFrame
from obspy.core import Stream

from utils_basic import LOC_DIR as dirpath_loc, MT_DIR as dirpath_mt, GEO_COMPONENTS as components, HYDRO_LOCATIONS as loc_dict 
from utils_preproc import read_and_process_windowed_hydro_waveforms
from utils_snuffler import read_normal_markers
from utils_mt import mt_cspec

###
# Inputs
###

# Command line arguments
parser = ArgumentParser(description = "Compute the multitaper inter-hydrophone location phase differences for the signals of a hammer shot")
parser.add_argument("--hammer_id", type = str, help = "Hammer ID")
parser.add_argument("--window_length", type = float, default = 1.0, help = "Window length in seconds")
parser.add_argument("--nw", type = float, default = 3.0, help = "Time-bandwidth parameter")

# Parse the arguments
args = parser.parse_args()
hammer_id = args.hammer_id
nw = args.nw
window_length = args.window_length

###
# Read the input files
###

# Read the hammer location file
print(f"Reading the hammer location file...")
filename = f"hammer_locations.csv"
inpath = join(dirpath_loc, filename)
hammer_df = read_csv(inpath, dtype={"hammer_id": str}, parse_dates=["origin_time"])

starttime = hammer_df[ hammer_df["hammer_id"] == hammer_id ]["origin_time"].values[0]
starttime = Timestamp(starttime)
endtime = starttime + Timedelta(seconds = window_length)

print(f"Computing the multitaper inter-station phase differences for the hammer shot {hammer_id} in the time window from {starttime} to {endtime}...")
# Read the geophone waveforms
print(f"Reading the geophone waveforms...")
stream = read_and_process_windowed_hydro_waveforms(starttime, endtime = endtime)

###
# Compute the multitaper inter-station phase differences
###

# Compute the phase differences for each station pair
for station in loc_dict.keys():
    print(f"Computing the phase differences for station {station}...")
    
    locations = loc_dict[station]
    num_loc = len(locations)

    # Process each location pair
    for i in range(num_loc - 1):
        output_dict = {}

        location1 = locations[i]
        location2 = locations[i + 1]

        # Get the waveforms for the two locations
        stream1 = stream.select(station = station, location = location1)
        stream2 = stream.select(station = station, location = location2)

        if len(stream1) == 0 or len(stream2) == 0:
            print(f"Warning: No waveforms found for locations {location1} and {location2}! Skipping...")
            continue

        trace1 = stream1[0]
        trace2 = stream2[0]

        signal1 = trace1.data
        signal2 = trace2.data

        sampling_rate = trace1.stats.sampling_rate

        # Determine if the signals are of the same length
        num_pts1 = len(signal1)
        num_pts2 = len(signal2)

        if num_pts1 != num_pts2:
            print(f"Warning: The signals for locations {location1} and {location2} have different lengths! Skipping...")
            continue

        num_pts = num_pts1

        # Generate the taper sequence matrix
        num_tapers = int(2 * nw - 1)
        taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = num_tapers, return_ratios = True)

        # Compute the multitaper inter-station phase differences
        cspec_param = mt_cspec(signal1, signal2, taper_mat, ratio_vec, 
                               sampling_rate, verbose = False, normalize = True, return_jk = True)

        # Save the phase differences
        freqax = cspec_param.freqax
        mt_aspec1 = cspec_param.aspec1
        mt_aspec2 = cspec_param.aspec2
        mt_cohe = cspec_param.cohe
        mt_phase_diff = cspec_param.phase_diff
        mt_phase_diff_uncer = cspec_param.phase_diff_uncer
        mt_phase_diff_jk = cspec_param.phase_diff_jk
        
        output_dict[f"aspec_loc1"] = mt_aspec1
        output_dict[f"aspec_loc2"] = mt_aspec2
        output_dict[f"cohe"] = mt_cohe
        output_dict[f"phase_diff"] = mt_phase_diff
        output_dict[f"phase_diff_uncer"] = mt_phase_diff_uncer

        output_dict[f"phase_diff_jk"] = list(mt_phase_diff_jk.T)

        print(f"Done for location pair {location1} and {location2}!")

        if len(output_dict) == 0:
            print(f"Warning: No valid phase differences found for location pair {location1} and {location2}! Skipping...")
            continue

        # Save the output
        output_dict["frequency"] = freqax
        output_df = DataFrame(output_dict)

        # Reorder columns to put frequency first
        cols = output_df.columns.tolist()
        cols.remove("frequency")
        cols = ["frequency"] + cols
        output_df = output_df[cols]

        # Convert the numpy arrays in the dataframe to JSON strings
        output_df = output_df.map(lambda x: x.tolist() if isinstance(x, ndarray) else x)

        print(f"Saving the results...")
        filename = f"hammer_mt_inter_hydro_loc_phase_diffs_{hammer_id}_{station}_{location1}_{location2}.csv"
        outpath = join(dirpath_mt, filename)
        output_df.to_csv(outpath, index = True)

        print(f"Done for location pair {location1} and {location2}!")
        print("")
