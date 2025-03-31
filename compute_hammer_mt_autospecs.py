"""
Compute the multitaper auto-spectra of the hammer signals
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame, Timestamp
from scipy.signal.windows import dpss
import numpy as np
from utils_basic import LOC_DIR as loc_dir, MT_DIR as mt_dir, GEO_STATIONS as stations, GEO_COMPONENTS as components
from utils_basic import power2db
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_autospec

###
# Input arguments
###

parser = ArgumentParser()
parser.add_argument("--hammer_id", type = str, help = "The ID of the hammer")
parser.add_argument("--nw", type = float, default = 3.0, help = "The time-bandwidth product")
parser.add_argument("--window_length", type = float, default = 1.0, help = "The length of the time window")
args = parser.parse_args()

hammer_id = args.hammer_id
nw = args.nw
window_length = args.window_length

###
# Load the data
###

print("########################################################")
print("Loading the data...")
print("########################################################")
print("")

# List of hammer locations
print("Loading the list of hammer locations...")
inpath = join(loc_dir, "hammer_locations.csv")
hammer_df = read_csv(inpath, parse_dates = ["origin_time"], dtype = {"hammer_id": str})
origin_time = hammer_df.loc[hammer_df["hammer_id"] == hammer_id, "origin_time"].values[0]

# Geophone waveforms
print("Loading the geophone waveforms...")
stream = read_and_process_windowed_geo_waveforms(origin_time, dur = window_length)

###
# Compute the multitaper auto-spectra for each station
###

print("########################################################")
print("Computing the multitaper auto-spectra...")
print("########################################################")
print("")
for station in stations:
    print(f"Computing the multitaper auto-spectra for {station}...")

    stream_sta = stream.select(station = station)

    for i, component in enumerate(components):
        print(f"Computing the multitaper auto-spectra for {component} component of {station}...")

        trace = stream_sta.select(component = component)[0]
        signal = trace.data
        sampling_rate = trace.stats.sampling_rate
        num_pts = len(signal)

        # Generate the taper sequence matrix and the concentration ratio vector
        taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)

        # Compute the multitaper auto-spectra
        mt_aspec_params = mt_autospec(signal, taper_mat, ratio_vec, sampling_rate)
        freqax = mt_aspec_params.freqax
        aspec = mt_aspec_params.aspec
        aspec_lo = mt_aspec_params.aspec_lo
        aspec_hi = mt_aspec_params.aspec_hi

        if i == 0:
            aspec_total = aspec
        else:
            aspec_total += aspec

        # Convert the auto-spectra to decibels
        aspec = power2db(aspec)
        aspec_lo = power2db(aspec_lo)
        aspec_hi = power2db(aspec_hi)

        # Save the multitaper auto-spectra
        if i == 0:
            output_dict = {
                "frequency": freqax,
                f"aspec_{component.lower()}": aspec,
                f"aspec_lo_{component.lower()}": aspec_lo,
                f"aspec_hi_{component.lower()}": aspec_hi
            }
        else:
            output_dict[f"aspec_{component.lower()}"] = aspec
            output_dict[f"aspec_lo_{component.lower()}"] = aspec_lo
            output_dict[f"aspec_hi_{component.lower()}"] = aspec_hi

    # Convert the total auto-spectrum to decibels
    aspec_total = power2db(aspec_total)
    output_dict["aspec_total"] = aspec_total

    # Save the multitaper auto-spectra
    print(f"Saving the multitaper auto-spectra for {station}...")

    # Reorder columns to put frequency and aspec_total first
    column_order = ["frequency", "aspec_total"]
    column_order.extend([col for col in output_dict.keys() if col not in column_order])
    output_dict = {col: output_dict[col] for col in column_order}

    output_df = DataFrame(output_dict)
    outpath = join(mt_dir, f"hammer_mt_aspecs_{hammer_id}_{station}.csv")
    output_df.to_csv(outpath, index = False)

    print(f"")
