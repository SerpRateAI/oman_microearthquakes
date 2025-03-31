"""
Compute the the phase differences between a pair of hydrophone locations using the multi-taper method
"""
###
# Import necessary modules
###

from os.path import join
from argparse import ArgumentParser
from numpy import nan, isnan, isrealobj, var
from pandas import DataFrame
from pandas import read_csv, to_datetime
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from time import time

from utils_basic import SPECTROGRAM_DIR as dirname_spec, MT_DIR as dirname_mt, GEO_COMPONENTS as components
from utils_basic import timestamp_to_utcdatetime
from utils_preproc import read_and_process_day_long_hydro_waveforms
from utils_mt import mt_cspec, get_avg_phase_diff


###
# Inputs
###

# Command line inputs
parser = ArgumentParser(description="Compute the the phase differences between a pair of hydrophone locations using the multi-taper method.")
parser.add_argument("--station", type=str, help="Station")
parser.add_argument("--location1", type=str, help="Location 1")
parser.add_argument("--location2", type=str, help="Location 2")
parser.add_argument("--window_length_mt", type=float, help="Window length for multitaper analysis")
parser.add_argument("--min_stft_win", type=int, help="Minimum number of STFT windows in a MT window")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--nw", type=float, help="Time-bandwidth product", default=3.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence", default=0.85)
parser.add_argument("--decimate_factor", type=int, help="Decimation factor", default=10)
parser.add_argument("--bandwidth", type=float, help="Resonance bandwidth for computing the average frequency", default=0.02) 



# Parse the command line inputs
args = parser.parse_args()

station = args.station
location1 = args.location1
location2 = args.location2
window_length_mt = args.window_length_mt
min_stft_win = args.min_stft_win

mode_name = args.mode_name
nw = args.nw
min_cohe = args.min_cohe
decimate_factor = args.decimate_factor

bandwidth = args.bandwidth

num_taper = int(2 * nw -1)

print("######")
print(f"Computing the phase differences between {station}.{location1} and {station}.{location2}...")
print("#######")
print("")

# Constants
window_length_stft = 300.0

# Print the input arguments
print("")
print(f"Station: {station}")
print(f"Location 1: {location1}")
print(f"Location 2: {location2}")
print(f"MT window length: {window_length_mt:.0f} s")
print(f"STFT window length: {window_length_stft:.0f} s")
print(f"Minimum number of STFT windows in a MT window: {min_stft_win}")
print(f"Minimum coherence: {min_cohe:.2f}")
print(f"Time-bandwidth product: {nw:.2f}")
print(f"Resonance bandwidth for computing the average frequency: {bandwidth:.2f}")
print("")


###
# Read and process the time window information
###

# Read the data
filename = f"multitaper_time_windows_{mode_name}_{station}_{location1}_{location2}_mt_win{window_length_mt:.0f}s_stft_win{window_length_stft:.0f}s.csv"
filepath = join(dirname_mt, filename)
time_win_df = read_csv(filepath, parse_dates = ["start", "end"])

# Keep only the MT windows with sufficient STFT windows
time_win_df = time_win_df.loc[ time_win_df["num_stft_windows"] >= min_stft_win]

# Sort the rows by time window
time_win_df.sort_values(by="start", inplace=True)

print(f"In total, {len(time_win_df)} MT time windows to compute.")

###
# Process each time window of each component
###

# Group the time windows by day 
time_win_grouped = time_win_df.groupby(time_win_df['start'].dt.date)

centertimes = []
freqs = []
phase_diffs = []
phase_diff_uncers = []

# Loop over each date
for date, group in time_win_grouped:
    print("######")
    print(f"Start processing the date {date}...")
    print("######")
    print("")

    print("Reading the waveform data...")
    clock1 = time()
    date_str = date.strftime("%Y-%m-%d")
    stream1 = read_and_process_day_long_hydro_waveforms(date_str, 
                                                        loc_dict = {station: [location1]},
                                                        decimate = True, decimate_factor = decimate_factor)
        
    stream2 = read_and_process_day_long_hydro_waveforms(date_str, 
                                                        loc_dict = {station: [location2]},
                                                        decimate = True, decimate_factor = decimate_factor)           
        
    clock2 = time()
    print(f"Elapsed time: {clock2 - clock1} s")
    print("")
        
    print("Processing each time window of the day...")
    for _, row in group.iterrows():
        clock1 = time()

        # Slice the data
        starttime = row["start"]
        endtime = row["end"]
        freq = row["mean_freq"]

        centertime = starttime + (endtime -  starttime) / 2

        print(f"Processing the time window {starttime}-{endtime}..")

        starttime = timestamp_to_utcdatetime(starttime)
        endtime = timestamp_to_utcdatetime(endtime)

        stream1_win = stream1.slice(starttime = starttime, endtime = endtime)
        stream2_win = stream2.slice(starttime = starttime, endtime = endtime)

        signal1 = stream1_win[0].data
        signal2 = stream2_win[0].data

        if any(isnan(signal1)) or any(isnan(signal2)):
            print("The input signals contain NaNs! Skipping...")
            continue

        num_pts1 = len(signal1)
        num_pts2 = len(signal2)

        if num_pts1 != num_pts2:
            print("The two signals have different lengths! Skipping...")
            continue

        num_pts = num_pts1
        sampling_rate = 1 / stream1[0].stats.delta

        # Get the DPSS windows
        print("Calculating the DPSS windows...")
        dpss_mat, ratio_vec = dpss(num_pts, nw, num_taper, return_ratios=True)

        # Perform multitaper cross-spectral analysis
        print("Performing multitaper cross-spectral analysis...")
        freqax, mt_aspec1, mt_aspec2, mt_trans, mt_cohe, mt_phase_diff, mt_aspec1_lo, mt_aspec1_hi, mt_aspec2_lo, mt_aspec2_hi, mt_trans_uncer, mt_cohe_uncer, mt_phase_diff_uncer = mt_cspec(signal1, signal2, dpss_mat, ratio_vec, sampling_rate, verbose = False)

        # Get the average phase difference
        print("Computing the average phase difference for the resonant frequency band...")
        freq_reson = row["mean_freq"]
        min_freq_reson = freq_reson - bandwidth / 2
        max_freq_reson = freq_reson + bandwidth / 2

        avg_phase_diff, avg_phase_diff_uncer = get_avg_phase_diff((min_freq_reson, max_freq_reson), freqax, mt_phase_diff, mt_phase_diff_uncer, mt_cohe, min_cohe = min_cohe, nw = nw, verbose = False)

        if avg_phase_diff is not None:
            print(f"Phase difference is {avg_phase_diff} +/- {avg_phase_diff_uncer}")

            centertimes.append(centertime)
            freqs.append(freq)
            phase_diffs.append(avg_phase_diff)                     
            phase_diff_uncers.append(avg_phase_diff_uncer)
        else:
            print("No reliable phase difference is estimated! Skipping...")
            
        clock2 = time()
        print(f"Elapsed time: {clock2 - clock1} s")
        print("")
    print("Finished processing the date.")
    print("")


print("Combining the results...")
phase_diff_df = DataFrame({"time": centertimes, "frequency": freqs, "phase_diff": phase_diffs, "phase_diff_uncer": phase_diff_uncers})
phase_diff_df["time"] = to_datetime(phase_diff_df["time"], utc = True)

###
# Save the results
###
filename = f"multitaper_inter_loc_phase_diffs_{mode_name}_{station}_{location1}_{location2}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirname_mt, filename)

phase_diff_df.to_csv(filepath, na_rep="nan")

print(f"Results are saved to {filepath}")






