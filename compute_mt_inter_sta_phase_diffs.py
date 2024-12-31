"""
Compute the 3C inter-station phase differences for a geophone station pair using the multi-taper method
"""
###
# Import necessary modules
###

from os.path import join
from argparse import ArgumentParser
from numpy import nan, isnan, isrealobj, var
from pandas import DataFrame
from pandas import read_csv
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from time import time

from utils_basic import SPECTROGRAM_DIR as dirname_spec, MT_DIR as dirname_mt, GEO_COMPONENTS as components
from utils_basic import timestamp_to_utcdatetime
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_mt import mt_cspec, get_avg_phase_diff, get_indep_phase_diffs
from utils_plot import save_figure


###
# Inputs
###

# Command line inputs
parser = ArgumentParser(description="Compute the 3C inter-station phase differences for a geophone station pair using the multi-taper method.")
parser.add_argument("--station1", type=str, help="Station 1")
parser.add_argument("--station2", type=str, help="Station 2")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--nw", type=float, help="Time-bandwidth product", default=3.0)
parser.add_argument("--min_cohe", type=float, help="Minimum coherence", default=0.95)
parser.add_argument("--decimate_factor", type=int, help="Decimation factor", default=10)
parser.add_argument("--bandwidth", type=float, help="Resonance bandwidth for computing the average frequency", default=0.02)

parser.add_argument("--min_stft_win", type=int, help="Minimum number of STFT windows in a MT window", default=5)


# Parse the command line inputs
args = parser.parse_args()

station1 = args.station1
station2 = args.station2

mode_name = args.mode_name
nw = args.nw
min_cohe = args.min_cohe
decimate_factor = args.decimate_factor

min_stft_win = args.min_stft_win
bandwidth = args.bandwidth

num_taper = int(2 * nw -1)

# Constants
win_len_mt = 1800.0
win_len_stft = 300.0

###
# Read and process the time window information
###

# Read the data
filename = f"multitaper_time_windows_{mode_name}_{station1}_{station2}_mt_win{win_len_mt:.0f}s_stft_win{win_len_stft:.0f}s.csv"
filepath = join(dirname_spec, filename)
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

# Loop over all components
result_dict = {}
for component in components:
    print("######")
    print(f"Start processing Component {component}...")
    print("######")
    print("")

    result_dict["time"] = []
    result_dict[f"phase_diff_{component.lower()}"] = []
    result_dict[f"phase_diff_uncer_{component.lower()}"] = []
    # Loop over each date
    for date, group in time_win_grouped:
            print("######")
            print(f"Start processing the date {date}...")
            print("######")
            print("")

            print("Reading the waveform data...")
            date_str = date.strftime("%Y-%m-%d")
            stream1 = read_and_process_day_long_geo_waveforms(date_str, 
                                                                stations = station1, components = component, 
                                                                decimate = True, decimate_factor = decimate_factor)
            
            stream2 = read_and_process_day_long_geo_waveforms(date_str, 
                                                                stations = station2, components = component, 
                                                                decimate = True, decimate_factor = decimate_factor)           
            
            print("Processing each time window of the day...")
            for _, row in group.iterrows():
                    clock1 = time()

                    # Slice the data
                    starttime = row["start"]
                    endtime = row["end"]

                    centertime = starttime + (starttime -  endtime) / 2
                    result_dict["time"].append(centertime)

                    print(f"Processing the time window {starttime}-{endtime}..")

                    starttime = timestamp_to_utcdatetime(starttime)
                    endtime = timestamp_to_utcdatetime(endtime)

                    stream1_win = stream1.slice(starttime = starttime, endtime = endtime)
                    stream2_win = stream2.slice(starttime = starttime, endtime = endtime)

                    signal1 = stream1_win[0].data
                    signal2 = stream2_win[0].data

                    if any(isnan(signal1)) or any(isnan(signal2)):
                        print("The input signals contain NaNs! Skipping...")
                        result_dict[f"phase_diff_{component.lower()}"].append(nan)
                        result_dict[f"phase_diff_uncer_{component.lower()}"].append(nan)
                        
                        continue

                    num_pts = len(signal1)
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

                    avg_phase_diff, avg_phase_diff_uncer = get_avg_phase_diff((min_freq_reson, max_freq_reson), freqax, mt_phase_diff, mt_phase_diff_uncer, mt_cohe, nw = nw, verbose = False)

                    if avg_phase_diff is not None:
                        print(f"Phase difference is {avg_phase_diff} +/- {avg_phase_diff_uncer}")

                        result_dict[f"phase_diff_{component.lower()}"].append(avg_phase_diff)                     
                        result_dict[f"phase_diff_uncer_{component.lower()}"].append(avg_phase_diff_uncer)
                    else:
                        print("The time wnidow produces no phase difference")

                        result_dict[f"phase_diff_{component.lower()}"].append(nan)
                        result_dict[f"phase_diff_uncer_{component.lower()}"].append(nan)


                    clock2 = time()
                    print(f"Elapsed time: {clock2 - clock1} s")

            print("Finished processing the date.")
            print("")
    
    print("Finished processing the component.")
    print("")

###
# Save the results
###

print("Saving the results...")
result_df = DataFrame(result_dict)

# Reorder the columns to put the column 'time' first
columns_order = ['time'] + [col for col in result_df.columns if col != 'time']
result_df = result_df[columns_order]

filename = f"multitaper_inter_sta_phase_diffs_{mode_name}_{station1}_{station2}.csv"
filepath = join(dirname_mt, filename)

result_df.to_csv(filepath, na_rep="nan")

print(f"Results are saved to {filepath}")






