# Extract the properties of one stationary resonance for all geophone stations

# Imports
from os.path import join
from pandas import concat, read_csv

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_spec import extract_stationary_resonance, get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_geo_spec_axes, read_spectral_peaks, read_spectral_peak_counts

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

peak_file_ext = "h5"

# Array counts
count_threshold = 9
count_fille_ext = "h5"

# Resonance extracting
# Name
name = "SR140a"

# Frequency buffer zone width for constructing the boolean array
freq_buffer = 0.1

# Isolated peak removal
min_patch_size = 3

# Saving the results
to_csv = True
to_hdf = True

# Read the stationary-resonance extraction parameters
inpath = join(indir, "stationary_resonance_extraction_params.csv")
params_df = read_csv(inpath, index_col = "name")
min_freq_res = params_df.loc[name, "min_freq"]
max_freq_res = params_df.loc[name, "max_freq"]

# Process each station
print(f"Extracting the properties of {name}... in the frequency range of [{min_freq_res}, {max_freq_res}] Hz.")

all_resonance_dfs = []
# stations = ["A01"]
for station in stations:
    print(f"### Working on {station}... ###")

    # Read the spectrograms
    print("Reading the spectrogram axes ...")
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
    inpath = join(indir, f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5")

    timeax, freqax = read_geo_spec_axes(inpath, min_freq = min_freq_res - freq_buffer, max_freq = max_freq_res + freq_buffer)
    #print(timeax[0])

    # Read the spectral peaks
    print("Reading the spectral peaks...")
    suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)
    inpath = join(indir, f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.{peak_file_ext}")

    peak_df = read_spectral_peaks(inpath)
    # print(peak_df.head())
    peak_df = peak_df.loc[(peak_df["frequency"] >= min_freq_res) & (peak_df["frequency"] <= max_freq_res)]


    # Read the spectral peak counts
    print("Reading the spectral peak array counts...")
    filename_in = f"geo_spectral_peak_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.{count_fille_ext}"
    inpath = join(indir, filename_in)
    array_count_df = read_spectral_peak_counts(inpath)
    array_count_df = array_count_df.loc[(array_count_df["frequency"] >= min_freq_res) & (array_count_df["frequency"] <= max_freq_res)]

    # Extract the resonance
    print("Extracting the resonance...")
    resonance_df = extract_stationary_resonance(peak_df, array_count_df, timeax, freqax)
    print(f"Number of active time windows: {resonance_df.shape[0]}")

    # Fill the time gaps with NaN
    print("Filling the time gaps with NaN...")
    resonance_df = resonance_df.asfreq("min")
    resonance_df["station"] = station

    # Compute the quality factors
    print("Computing the quality factors...")
    resonance_df["quality_factor"] = resonance_df["frequency"] * resonance_df["reverse_bandwidth"]
    print("")

    all_resonance_dfs.append(resonance_df)

resonance_df = concat(all_resonance_dfs)

# # Compute the average frequency for each group while disregarding NaN values
# average_frequency = filtered_resonance_df.groupby("time")["frequency"].mean()
# print("Average frequency for each group:")
# print(average_frequency)

# Save the results
if to_csv:
    print("Saving the results to CSV...")
    outpath = join(indir, f"stationary_resonance_properties_{name}_geo.csv")
    resonance_df.to_csv(outpath, index = True, na_rep = "NaN")

if to_hdf:
    print("Saving the results to HDF...")
    outpath = join(indir, f"stationary_resonance_properties_{name}_geo.h5")
    resonance_df.to_hdf(outpath, key = "properties", mode = "w")
