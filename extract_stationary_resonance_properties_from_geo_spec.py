# Extract the properties of one stationary resonance for all geophone stations

# Import the required libraries
from os.path import exists, join
from argparse import ArgumentParser
from numpy import nan
from scipy.signal import find_peaks
from pandas import DataFrame
from pandas import concat, read_csv

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, GEO_COMPONENTS as components
from utils_spec import get_quality_factor,get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_geo_stft, read_geo_spec_peaks, read_spec_peak_array_counts

# Functions
def get_component_quality_factor(trace_stft, resonance_df, min_prom):
    trace_stft.to_db()
    freqax = trace_stft.freqs
    freq_interval = freqax[1] - freqax[0]

    quality_factors = []
    for _, row in resonance_df.iterrows():
        time_window = row["time"]
        freq_total = row["frequency"]
        # print(f"Frequency: {freq_total}")

        psd = trace_stft.get_psd(time_window)
        
        # Find the peaks in the PSD satisfying the prominence threshold
        idx_peaks, _ = find_peaks(psd, prominence = min_prom)
        if len(idx_peaks) == 0:
            quality_factors.append(nan)
            continue

        # Find the peak frequencies
        freq_peaks = freqax[idx_peaks]
        # print(f"Peak frequencies: {freq_peaks}")

        # Determine if the peak frequency from the total PSD is within one frequency bin of any of the peak frequencies
        idx = abs(freq_total - freq_peaks).argmin()
        if abs(freq_total - freq_peaks[idx]) < freq_interval:
            quality_factor, _ = get_quality_factor(freqax, psd, freq_total)
            quality_factors.append(quality_factor)
        else:
            quality_factors.append(nan)

    return quality_factors
        
# Inputs
# Command-line arguments
parser = ArgumentParser(description = "Extract the properties of one stationary resonance for all geophone stations.")
parser.add_argument("--mode_name", type = str, help = "Mode name.")
                    
parser.add_argument("--window_length", type = float, default = 60.0, help = "Window length in seconds for computing the spectrogram.")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap between consecutive windows for computing the spectrogram.")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type = float, default = 3.0, help = "Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type = float, default = 10.0, help = "Maximum mean dB value for excluding noise windows.")

parser.add_argument("--freq_buffer", type = float, default = 0.1, help = "Frequency buffer in Hz for reading the spectrogram.")

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
freq_buffer = args.freq_buffer

# Print the inputs
print(f"### Extracting the properties of a stationary resonance from the spectral peaks of the geophone spectrograms ###")
print("")

print(f"# Spectrogram computation #")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print("")

print(f"# Spectral-peak detection #")
print(f"Prominence threshold: {min_prom}")
print(f"Reverse bandwidth threshold: {min_rbw}")
print("")

print("Stationary-resonance property extraction:")
print(f"Mode name: {mode_name}")
print("")

# Read the resonance frequency range
print("Reading the resonance frequency range...")
filename = f"stationary_resonance_freq_ranges_geo.csv"
filepath = join(indir, filename)
freq_range_df = read_csv(filepath)

# Select the resonance frequency range
min_freq_reson = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_extract"].values[0]
max_freq_reson = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_extract"].values[0]

# Read the spectral peak counts
print("Reading the spectral peak array counts...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
filename_in = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}.h5"

inpath = join(indir, filename_in)
array_count_df = read_spec_peak_array_counts(inpath, min_freq = min_freq_reson, max_freq = max_freq_reson)

# Find the frequency with the largest number of stations at each time window
print("Finding the frequency with the largest number of stations at each time window...")
array_count_group = array_count_df.groupby("time")
resonance_dicts = []
for time, array_count_time_df in array_count_group:
    i_max = array_count_time_df["count"].idxmax()
    freq_max_count = array_count_time_df.loc[i_max, "frequency"]

    resonance_dicts.append({"time": time, "frequency": freq_max_count})

resonance_freq_time_df = DataFrame(resonance_dicts)

# Process each station
print(f"Extracting the properties of {mode_name}... in the frequency range of [{min_freq_reson}, {max_freq_reson}] Hz.")

all_station_dfs = []
# stations = ["A01"]
for station in stations:
    print(f"### Working on {station}... ###")

    # Read the spectrograms
    print("Reading the spectral peaks..")
    filename = f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
    inpath = join(indir, filename)
    peak_df = read_geo_spec_peaks(inpath, min_freq = min_freq_reson, max_freq = max_freq_reson)

    # Find the rows with frequency and time windows matching the resonance frequency and time windows
    print("Finding the rows with frequency and time windows matching the resonance frequency and time windows...")
    resonance_df = peak_df.merge(resonance_freq_time_df, on = ["time", "frequency"], how = "inner")
    print(f"Number of time windows with resonance detected: {len(resonance_df)}")

    # Compute the quality factors for each component
    print("Computing the quality factors...")
    min_freq = resonance_df["frequency"].min() - freq_buffer
    max_freq = resonance_df["frequency"].max() + freq_buffer
    for component in components:
        print(f"Computing the quality factors for component {component}...")

        # Read the spectrogram in the frequency range of the resonance
        print(f"Reading the spectrogram...")
        filename = f"whole_deployment_daily_geo_stft_{station}_{suffix_spec}.h5"
        inpath = join(indir, filename)
        stream_stft = read_geo_stft(inpath, components = component, min_freq = min_freq, max_freq = max_freq, psd_only = True)

        # Compute the quality factors for each time window
        trace_stft = stream_stft[0]
        trace_stft.to_db()
        quality_factors = get_component_quality_factor(trace_stft, resonance_df, min_prom)
        resonance_df[f"quality_factor_{component.lower()}"] = quality_factors

        # Print the number of non-NaN quality factors computed
        num_non_nan = len(resonance_df[f"quality_factor_{component.lower()}"].dropna())
        print(f"Number of non-NaN quality factors computed: {num_non_nan}")
    print("")

    all_station_dfs.append(resonance_df)

resonance_sta_df = concat(all_station_dfs)
resonance_sta_df.reset_index(drop = True, inplace = True)

# # Exclude the time windows in the hammering period
# print("Excluding the time windows in the hammering period...")
# resonance_df = resonance_df[(resonance_df.index < hammer_starttime) | (resonance_df.index > hammer_endtime)]

# Compute the average total power and station count in each time window
resonance_prof_df = resonance_sta_df.groupby("time").agg(mean_power=("total_power", "mean"), count = ("total_power", "size"))
resonance_prof_df = resonance_prof_df.merge(resonance_freq_time_df, on = ["time"], how = "inner")
resonance_prof_df = resonance_prof_df[["time", "frequency", "count", "mean_power"]]

# Save the results
print("Saving the resonance properties of individual stations..")
print("Saving the results to CSV...")
outpath = join(indir, f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.csv")
resonance_sta_df.to_csv(outpath, index = False, na_rep = "nan")
print(f"Saved the results to {outpath}")

print("Saving the results to HDF...")
outpath = join(indir, f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5")
resonance_sta_df.to_hdf(outpath, key = "properties", mode = "w", index = False)
print(f"Saved the results to {outpath}")

print("Saving the profile of the resonance...")
print("Saving the results to CSV...")
outpath = join(indir, f"stationary_resonance_profile_geo_{mode_name}_{suffix_spec}_{suffix_peak}.csv")
resonance_prof_df.to_csv(outpath, index = False, na_rep = "nan")
print(f"Saved the results to {outpath}")

print("Saving the results to HDF...")
outpath = join(indir, f"stationary_resonance_profile_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5")
resonance_prof_df.to_hdf(outpath, key = "profile", mode = "w", index = False)
print(f"Saved the results to {outpath}")
