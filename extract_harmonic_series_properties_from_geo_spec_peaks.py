# Extract the properties of a harmonic series of stationary resonances from the spectral peaks of the geophone spectrograms
# Results of each resonance are saved to a CSV and an HDF file

# Imports
from os.path import join
from pandas import concat, read_csv
from numpy import isnan

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, HAMMER_STARTTIME as hammer_starttime, HAMMER_ENDTIME as hammer_endtime
from utils_spec import extract_geo_station_stationary_resonance_properties, get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_geo_spec_axes, read_geo_spec_peaks, read_spec_peak_array_counts

# Inputs
# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral peaks
prom_threshold = 10.0
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200

# Array counts
array_count_threshold_peak = 9

# Stationary resonance extraction
array_count_threshold_resonance = 15

# Harmonic series
base_name = "PR02549"
base_mode = 2

# Print the inputs
print(f"Extract the properties of the harmonic series with {base_name} as the base model and model number of {base_mode}...")
print("Inputs:")
print("")
print("Spectrogram computation:")
print(f"Window length: {window_length:.1f} s")
print(f"Overlap: {overlap:.1f}")
print(f"Downsample: {downsample}")
print(f"Downsample factor: {downsample_factor}")
print("")

print("Spectral-peak finding:")
print(f"Prominence threshold: {prom_threshold:.1f} dB")
print(f"Reverse bandwidth threshold: {rbw_threshold:.1f} 1 / Hz")
print(f"Minimum frequency: {min_freq_peak}")
print(f"Maximum frequency: {max_freq_peak}")
print("")

print("Array detection:")
print(f"Count threshold: {array_count_threshold_peak}")
print("")

print("Stationary resonance extraction:")
print(f"Count threshold: {array_count_threshold_resonance}")
print("")

# Resonance extracting
# Input file name containing the information of the stationary resonances
filename_param = f"stationary_harmonic_series_{base_name}_base{base_mode:d}.csv"

# Frequency buffer zone width for constructing the boolean array
freq_buffer = 0.1

# Isolated peak removal
min_patch_size = 3

# Read the stationary-resonance information
inpath = join(indir, filename_param)
params_df = read_csv(inpath, index_col=0)

# Extract each harmonic
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)
for i, row in params_df.iterrows():
    name = row["name"]
    min_freq_res = row["extract_min_freq"]
    max_freq_res = row["extract_max_freq"]

    if isnan(min_freq_res) or isnan(max_freq_res):
        print(f"Skipping the missing harmonic {name}...")
        continue

    print(f"Extracting the properties of {name}... in the frequency range of [{min_freq_res}, {max_freq_res}] Hz.")
    # Read the spectral peak counts
    print("Reading the spectral peak array counts...")
    filename_in = f"geo_spectral_peak_array_counts_{suffix_spec}_{suffix_peak}_count{array_count_threshold_peak:d}.h5"
    inpath = join(indir, filename_in)
    array_count_df = read_spec_peak_array_counts(inpath, min_freq = min_freq_res, max_freq = max_freq_res)
    
    all_resonance_dfs = []
    for station in stations:
        print(f"### Working on {station}... ###")

        # Read the spectrograms
        print("Reading the spectrogram axes ...")

        inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5")
        timeax, freqax = read_geo_spec_axes(inpath, min_freq = min_freq_res, max_freq = max_freq_res)
        # Read the spectral peaks
        print("Reading the spectral peaks...")
        inpath = join(indir, f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5")

        peak_df = read_geo_spec_peaks(inpath, min_freq = min_freq_res, max_freq = max_freq_res)
        # print(peak_df.head())

        # Extract the resonance
        print("Extracting the resonance...")
        resonance_df = extract_geo_station_stationary_resonance_properties(station, min_freq_res, max_freq_res, peak_df, array_count_df, array_count_threshold = array_count_threshold_resonance)
        print(f"Number of active time windows: {resonance_df.shape[0]}")

        # # Fill the time gaps with NaN
        # print("Filling the time gaps with NaN...")
        # resonance_df = resonance_df.asfreq("min")
        # resonance_df["station"] = station

        # Compute the quality factors
        print("Computing the quality factors...")
        resonance_df["quality_factor"] = resonance_df["frequency"] * resonance_df["reverse_bandwidth"]
        print("")

        all_resonance_dfs.append(resonance_df)

    resonance_df = concat(all_resonance_dfs)

    # Exclude the time windows in the hammering period
    print("Excluding the time windows in the hammer time window...")
    resonance_df = resonance_df[(resonance_df["time"] < hammer_starttime) | (resonance_df["time"] > hammer_endtime)]

    # # Set the time as the Level 0 index and the station as the Level 1 index
    # resonance_df.set_index(["time", "station"], inplace = True)


    # Save the results
    # print("Saving the results to TXT...")
    # outpath = join(indir, f"stationary_resonance_properties_{name}_geo.txt")
    # string_out = resonance_df.to_string()
    # with open(outpath, "w") as f:
    #     f.write(string_out)

    print("Saving the results to CSV...")
    outpath = join(indir, f"stationary_resonance_properties_{name}_geo.csv")
    resonance_df.to_csv(outpath, index = True)

    print("Saving the results to HDF...")
    outpath = join(indir, f"stationary_resonance_properties_{name}_geo.h5")
    resonance_df.to_hdf(outpath, key = "properties", mode = "w")
