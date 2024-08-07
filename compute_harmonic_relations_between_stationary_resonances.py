# Compute the harmonic relations between the stationary resonances
# Imports
from os.path import join
from pandas import DataFrame
from pandas import concat, read_csv

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import is_subset_df
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

file_ext_in = "h5"

# Array grouping
count_threshold = 9

# Peak detection
frac_threshold = -3.0 # In log10 units
prom_frac_threshold = 0.5 # In log10 units

# Harmonic relations
base_mode = 2
error_threshold = 0.003
gap_threshold = 2

# Read the stationary-resonance detections
print("Reading the stationary resonance detections...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_spec_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)
filename_in = f"stationary_resonances_detected_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_frac{frac_threshold:.1f}_prom{prom_frac_threshold:.1f}.csv"

inpath = join(indir, filename_in)
detections_df = read_csv(inpath, index_col = 0)
detections_df.sort_values('frequency', inplace = True)

# Compute the harmonic relations between the stationary resonances
# Compute the frequency ratio between each pair of stationary resonances
print("Computing the harmonic relations...")
harmo_dict = {}
for i, row1 in detections_df.iterrows():
    freq1 = row1['frequency']
    print(f"Searching for harmonic series with a base frequency of {freq1} Hz...")
    # print(f"i = {i}")

    harmo_list = [{'frequency': freq1, 'harmonic_number': base_mode, 'error': 0.0}]
    for j in range(i + 1, len(detections_df)):
        row2 = detections_df.iloc[j]
        freq2 = row2['frequency']
        # print(f"Comparing with {freq2} Hz...")
        # print(f"j = {j}")

        ratio = freq2 / freq1 * base_mode
        harmo = round(ratio)
        error = abs(ratio - harmo) / harmo

        if error < error_threshold:
            harmo_list.append({'frequency': freq2, 'harmonic_number': harmo, 'error': error})

    if len(harmo_list) > 1:
        harmo_df = DataFrame(harmo_list)
        # Find the repeated harmonic numbers
        harmo_counts = harmo_df['harmonic_number'].value_counts()
        harmo_repeated = harmo_counts[harmo_counts > 1].index

        for harmo in harmo_repeated:
            harmo_repeated_df = harmo_df.loc[harmo_df['harmonic_number'] == harmo]
            harmo_min_error_df = harmo_repeated_df[harmo_repeated_df['error'] == harmo_repeated_df['error'].min()]
            harmo_to_remove_df = harmo_repeated_df.drop(harmo_min_error_df.index)
            harmo_df.drop(harmo_to_remove_df.index, inplace = True)
            harmo_df.reset_index(drop = True, inplace = True)

        harmo_dict[freq1] = harmo_df

# Truncate the harmonic series when the gap exceeds the threshold
print("Truncating the harmonic series when the gap exceeds the threshold...")
keys_to_remove = []
for freq_base, harmo_df in harmo_dict.items():
    for i, row in harmo_df.iloc[:-1].iterrows():
        harmo = row['harmonic_number']
        freq = row['frequency']
        error = row['error']

        next_row = harmo_df.iloc[i + 1]
        next_harmo = next_row['harmonic_number']
        gap = next_harmo - harmo

        if gap > gap_threshold:
            harmo_df.drop(harmo_df.index[i + 1:], inplace = True)
            harmo_df.reset_index(drop = True, inplace = True)
            break

    if len(harmo_df) < 2:
        keys_to_remove.append(freq_base)

for key in keys_to_remove:
    harmo_dict.pop(key)

# Remove the harmonic series that are subsets of others
print("Removing the harmonic series that are subsets of others...")
keys_to_remove = []
for freq_base1, harmo_df1 in harmo_dict.items():
    harmo1_freq_df = harmo_df1.drop(columns = ['harmonic_number', 'error'])
    for freq_base2, harmo_df2 in harmo_dict.items():
        harmo2_freq_df = harmo_df2.drop(columns = ['harmonic_number', 'error'])

        if freq_base1 == freq_base2:
            continue

        if freq_base1 not in keys_to_remove and freq_base2 not in keys_to_remove:
            if is_subset_df(harmo1_freq_df, harmo2_freq_df):
                keys_to_remove.append(freq_base1)
            elif is_subset_df(harmo1_freq_df, harmo2_freq_df):
                keys_to_remove.append(freq_base2)
                
print(f"Removing {len(keys_to_remove)} harmonic series that are subsets of others...")
for key in keys_to_remove:
    harmo_dict.pop(key)

print(f"In total, {len(harmo_dict)} harmonic series were found.")

# Print the harmonic relations
print("Harmonic series found:")
for freq1, harmo_df in harmo_dict.items():
    print(f"Base frequency: {freq1:.1f} Hz")
    print(harmo_df)
    print()

# Convert the harmonic relations to a nested DataFrame and save to a CSV file
print("Saving the harmonic relations...")
all_harmo_df = concat(harmo_dict, axis = 0)

filename_out = f"stationary_resonance_harmonic_relations_{suffix_spec}_{suffix_peak}_count{count_threshold:d}_frac{frac_threshold:.1f}_prom{prom_frac_threshold:.1f}_base{base_mode}.csv"
outpath = join(indir, filename_out)
all_harmo_df.to_csv(outpath, index_label = ["base_frequency", "mode_index"])
print(f"Saved the harmonic relations to {outpath}.")
        