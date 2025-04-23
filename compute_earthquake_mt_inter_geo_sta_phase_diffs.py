"""
Compute the multitaper inter-station phase differences for the signals of an earthquake in a time window

"""
###
# Imports
###
from os.path import join
from argparse import ArgumentParser
from scipy.signal.windows import dpss
from json import dumps
from numpy import array, ndarray
from pandas import date_range, read_csv, read_hdf
from matplotlib.pyplot import subplots, colormaps
from pandas import DataFrame, Timedelta
from obspy.core import Stream

from utils_basic import MT_DIR as dirpath_mt, LOC_DIR as dirpath_loc, GEO_COMPONENTS as components
from utils_basic import str2timestamp
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_cspec

###
# Inputs
###

# Command line arguments
parser = ArgumentParser(description = "Compute the multitaper inter-station phase differences for the signals of an earthquake")
parser.add_argument("--earthquake_id", type = int, help = "The ID of the earthquake")
parser.add_argument("--window_id", type = int, help = "The ID of the time window")
parser.add_argument("--nw", type = float, default = 3.0, help = "Time-bandwidth parameter")

# Parse the arguments
args = parser.parse_args()
earthquake_id = args.earthquake_id
window_id = args.window_id
nw = args.nw

###
# Read the input files
###

# Read the time windows
inpath = join(dirpath_loc, f"earthquake_time_windows_eq{earthquake_id}.csv")
time_window_df = read_csv(inpath, parse_dates = ["start_time", "end_time"])

# Get the start and end times of the time window
start_time = time_window_df.loc[time_window_df["window_id"] == window_id, "start_time"].values[0]
end_time = time_window_df.loc[time_window_df["window_id"] == window_id, "end_time"].values[0]

# Read the geophone waveforms
print(f"Reading the geophone waveforms...")
stream = read_and_process_windowed_geo_waveforms(start_time, endtime = end_time)

###
# Compute the multitaper inter-station phase differences
###

# Read the list of station pairs
inpath = join(dirpath_mt, "delaunay_station_pairs.csv")
pair_df = read_csv(inpath)

# Compute the phase differences for each station pair
for _, row in pair_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]
    print(f"Computing the phase differences between stations {station1} and {station2}...")
    output_dict = {}

    # Process each component
    for component in components:
        print(f"Working on component {component}...")

        # Get the waveforms for the two stations
        stream1 = stream.select(station = station1, component = component)
        stream2 = stream.select(station = station2, component = component)

        if len(stream1) == 0 or len(stream2) == 0:
            print(f"Warning: No waveforms found for stations {station1} and {station2}! Skipping...")
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
            print(f"Warning: The signals for stations {station1} and {station2} have different lengths! Skipping...")
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
        
        output_dict[f"aspec_sta1_{component.lower()}"] = mt_aspec1
        output_dict[f"aspec_sta2_{component.lower()}"] = mt_aspec2
        output_dict[f"cohe_{component.lower()}"] = mt_cohe
        output_dict[f"phase_diff_{component.lower()}"] = mt_phase_diff
        output_dict[f"phase_diff_uncer_{component.lower()}"] = mt_phase_diff_uncer

        output_dict[f"phase_diff_jk_{component.lower()}"] = list(mt_phase_diff_jk.T)

        print(f"Done for component {component}!")

    if len(output_dict) == 0:
        print(f"Warning: No valid phase differences found for stations {station1} and {station2}! Skipping...")
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
    filename = f"earthquake_mt_inter_geo_sta_phase_diffs_eq{earthquake_id}_window{window_id}_{station1}_{station2}.csv"
    outpath = join(dirpath_mt, filename)
    output_df.to_csv(outpath, index = False)

    print(f"Done for station pair {station1} and {station2}!")
    print("")
