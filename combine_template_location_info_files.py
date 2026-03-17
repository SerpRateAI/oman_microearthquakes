"""
Combine the location information HDF5 files for the template events
"""

#-----------
# Imports
#-----------
from os import remove
from argparse import ArgumentParser
from pathlib import Path
from re import match
from pandas import read_csv, concat

from utils_basic import LOC_DIR as dirpath
from utils_basic import get_freq_limits_string
from utils_loc import load_location_from_hdf_individual, save_location_to_hdf_combined

#-----------
# Argument parser
#-----------
parser = ArgumentParser()
parser.add_argument("--template_id", type=str, required=True)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=200.0)

args = parser.parse_args()
template_id = args.template_id
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter

#-----------
# Main
#-----------
print(f"Combining HDF5 files for template {template_id}...")

# Combine all HDF5 files for the template event
filepaths = []
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
for f in Path(dirpath).iterdir():
    if match(rf"location_info_template_{template_id}_{freq_str}.+\.h5", f.name):
        filepaths.append(f)

print(f"Found {len(filepaths)} HDF5 files.")

# Load the location information from the individual HDF5 files and combine them into a single HDF5 file
filename = f"location_info_template_{template_id}_{freq_str}.h5"
filepath_combined = Path(dirpath) / filename
print(f"Saving the combined location information to {filepath_combined}...")

for filepath in filepaths:
    param_dict, location_dict, arrival_time_dict, station_misfit_dict, grid_dict = load_location_from_hdf_individual(filepath)
    save_location_to_hdf_combined(filepath_combined, param_dict, location_dict, arrival_time_dict, station_misfit_dict, grid_dict)
    print(f"Saved the location information of {param_dict['arrival_type']} {param_dict['phase']} with scale factor {param_dict['scale_factor']}.")
    
    remove(filepath)
    print(f"Removed {filepath}.")

# Combine all CSV files for the template event
filepaths = []
for f in Path(dirpath).iterdir():
    if match(rf"location_info_template_{template_id}_{freq_str}.+\.csv", f.name):
        filepaths.append(f)

print(f"Found {len(filepaths)} CSV files.")

# Load the location information from the individual CSV files and combine them into a single CSV file
## Combine the individual CSV files
filename = f"location_info_template_{template_id}_{freq_str}.csv"
filepath_combined = Path(dirpath) / filename
print(f"Saving the combined location information to {filepath_combined}...")

for filepath in filepaths:
    location_df = read_csv(filepath, dtype={"template_id": str})
    # print(location_df)
    if filepath == filepaths[0]:
        location_df_append = location_df
    else:
        location_df_append = concat([location_df_append, location_df])

    remove(filepath)
    print(f"Removed {filepath}.")

location_df_append = location_df_append.sort_values(by=["arrival_type", "phase", "scale_factor", "subarray", "weight"])

# Combine with the existing CSV file
if filepath_combined.exists():
    location_df_existing = read_csv(filepath_combined, dtype={"template_id": str})
    location_df_all = concat([location_df_existing, location_df_append])
else:
    location_df_all = location_df_append

location_df_all = location_df_all.sort_values(by=["arrival_type", "phase", "scale_factor", "subarray", "weight"])
location_df_all = location_df_all.drop_duplicates(subset=["arrival_type", "phase", "scale_factor", "subarray", "weight"], keep="last")
print(location_df_all)
location_df_all.to_csv(filepath_combined, index=False)

print(f"Saved the location information to {filepath_combined}.")