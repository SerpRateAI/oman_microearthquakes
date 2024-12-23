# Extract one harmonic series from the automatically detected ones and format it for manual editing.
# Imports
from os.path import join
from numpy import abs, argmin, nan
from pandas import DataFrame
from pandas import concat, read_csv

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

# Inputs
# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Spectral-peak finding
prom_spec_threshold = 10.0
rbw_threshold = 3.0

min_freq_peak = None
max_freq_peak = 200

# Array grouping
count_threshold = 9

# Peak detection
frac_threshold = -3.0 # In log10 units
prom_frac_threshold = 0.5 # In log10 units

# Harmonic relations
base_mode = 1
freq_base_in = 6.0

# Print the inputs
print(f"Extracting harmonic seris with a based frequency of {freq_base_in:.2f} Hz and mode of {base_mode:d}...")
print("Inputs:")
print("")
print("Spectrogram computation:")
print(f"Window length: {window_length:.1f} s")
print(f"Overlap: {overlap:.1f}")
print(f"Downsample: {downsample}")
print(f"Downsample factor: {downsample_factor}")
print("")

print("Spectral-peak finding:")
print(f"Prominence threshold: {prom_spec_threshold:.1f} dB")
print(f"Reverse bandwidth threshold: {rbw_threshold:.1f} 1 / Hz")
print(f"Minimum frequency: {min_freq_peak}")
print(f"Maximum frequency: {max_freq_peak}")
print("")

print("Array grouping:")
print(f"Count threshold: {count_threshold}")
print("")

print("Resonance detection:")
print(f"Fractional threshold: {frac_threshold:.1f}")
print(f"Prominence fractional threshold: {prom_frac_threshold:.1f}")

# Read the harmonic relations
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_spec_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)
filename_in = f"stationary_resonance_harmonic_relations_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_frac{frac_threshold:.1f}_prom{prom_frac_threshold:.1f}_base{base_mode:d}.csv"

inpath = join(indir, filename_in)
all_harmo_df = read_csv(inpath, index_col = [0, 1])

# Extract the harmonic series with a base frequency of freq_base
# Extract unique base_frequency values
unique_base_freqs = all_harmo_df.index.get_level_values('base_frequency').unique()

# Compute the absolute differences
i_freq = argmin(abs(unique_base_freqs - freq_base_in))

# Find the base_frequency with the minimum difference
freq_base = unique_base_freqs[i_freq]

# Extract the harmonic series with the base frequency of freq_base
harmo_df_in = all_harmo_df.loc[freq_base]

# Add the name, detected, and predicted frequency columns
harmo_df_out = harmo_df_in.copy()
harmo_df_out["name"] = harmo_df_in["frequency"].apply(lambda x: f"{x:.2f}".replace(".", "").zfill(5))
harmo_df_out["name"] = harmo_df_out["name"].apply(lambda x: f"PR{x}")
harmo_df_out["predicted_freq"] = freq_base / base_mode * harmo_df_out["harmonic_number"]
harmo_df_out["detected"] = True

harmo_df_out.rename(columns={"frequency": "observed_freq"}, inplace=True)
harmo_df_out = harmo_df_out[["name", "detected", "observed_freq", "predicted_freq", "harmonic_number", "error", "lower_freq_bound", "upper_freq_bound"]]


# Fill the missing harmonic numbers
harmo_num_max = harmo_df_out["harmonic_number"].max()
for i in range(1, harmo_num_max + 1):
    if i not in harmo_df_out["harmonic_number"].values:
        freq_pred = freq_base / base_mode * i
        name = f"{freq_pred:.2f}".replace(".", "").zfill(5)
        name = f"MH{name}"
        new_harmo_dict = {"name": name,
                     "detected": False,
                     "observed_freq": nan,
                     "predicted_freq": freq_pred,
                     "harmonic_number": i,
                     "error": nan}
        new_harmo_df = DataFrame([new_harmo_dict])
        
        harmo_df_out = concat([harmo_df_out, new_harmo_df], ignore_index = True)
        
harmo_df_out.sort_values("predicted_freq", inplace = True, ignore_index = True)

# Save the harmonic series
freq_base_str = f"{freq_base:.2f}".replace(".", "").zfill(5)
filename_out = f"stationary_harmonic_series_PR{freq_base_str}_base{base_mode:d}.csv"
outpath = join(indir, filename_out)

harmo_df_out["detected"] = harmo_df_out["detected"].map({True: "true", False: "false"})
harmo_df_out.rename(columns = {"lower_freq_bound": "extract_min_freq", "upper_freq_bound": "extract_max_freq"}, inplace = True)
harmo_df_out.to_csv(outpath, na_rep = "nan")
print(f"Saved the harmonic series with a base frequency of {freq_base:.2f} Hz to {outpath}.")