# Compute the cross-component phase differences and amplitude ratios of a stationary resonance on all geophone stations

# Import necessary libraries
from os.path import join
from argparse import ArgumentParser
from pandas import DataFrame
from pandas import concat, read_hdf

from utils_basic import GEO_STATIONS as stations, PHASE_DIFF_COMPONENT_PAIRS as component_pairs, SPECTROGRAM_DIR as indir
from utils_basic import powers2amplitude_ratio

# Command line arguments
parser = ArgumentParser(description = "Compute the cross-component phase differences and amplitude ratios of a stationary resonance on all geophone stations.")
parser.add_argument("--mode_name", type=str, help="Mode name")

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name

# Print the inputs
print(f"### Computing the cross-component phase differences and amplitude ratios of {mode_name} on all geophone stations ###")

# Read the resonance properties
filename = f"stationary_resonance_properties_{mode_name}_geo.h5"
filepath = join(indir, filename)
input_df = read_hdf(filepath, key = "properties")

# Compute the cross-component phase differences and amplitude ratios
print("Computing the cross-component phase differences and amplitude ratios...")
result_dicts = []
for i, row in input_df.iterrows():
    result_dict = {}
    for component_pair in component_pairs:
        component1, component2 = component_pair
        
        phase1 = row[f"phase_{component1.lower()}"]
        phase2 = row[f"phase_{component2.lower()}"]
        phase_diff = phase2 - phase1

        power1 = row[f"power_{component1.lower()}"]
        power2 = row[f"power_{component2.lower()}"]
        amp_ratio = powers2amplitude_ratio(power1, power2)

        result_dict[f"phase_diff_{component1.lower()}_{component2.lower()}"] = phase_diff
        result_dict[f"amp_ratio_{component1.lower()}_{component2.lower()}"] = amp_ratio

    result_dicts.append(result_dict)

# Convert the list of dictionaries to a DataFrame
result_df = DataFrame(result_dicts)
# result_df.index = input_df.index

# Concatenate the output DataFrame with the input DataFrame
output_df = concat([input_df, result_df], axis = 1)

# Keep only the necessary columns
output_df = output_df.drop(columns = ["reverse_bandwidth", "phase_1", "phase_2", "phase_z", "quality_factor"])

# Save the output DataFrame
print("Saving the output DataFrame to a CSV file...")
filename = f"stationary_resonance_cross_comp_pha_diff_amp_rat_{mode_name}.csv"
filepath = join(indir, filename)
output_df.to_csv(filepath)
print(f"Output DataFrame saved to {filepath}")

print("Saving the output DataFrame to an HDF5 file...")
filename = f"stationary_resonance_cross_comp_pha_diff_amp_rat_{mode_name}.h5"
filepath = join(indir, filename)
output_df.to_hdf(filepath, key = "properties", mode = "w")
print(f"Output DataFrame saved to {filepath}")





