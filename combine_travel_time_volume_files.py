"""
Combine the travel time volume files of different scale factors into a single file.
"""

#-----------
# Imports
#-----------
from os import remove
from argparse import ArgumentParser
from pathlib import Path
from re import match

from utils_basic import VEL_MODEL_DIR as dirpath_vel
from utils_loc import load_travel_time_volumes_individual, save_travel_time_volumes_combined

#-----------
# Argument parser
#-----------
parser = ArgumentParser()
parser.add_argument("--phase", type=str, required=True)
parser.add_argument("--subarray", type=str, required=True)

args = parser.parse_args()
phase = args.phase
subarray = args.subarray

#-----------
# Main
#-----------

print(f"Combining HDF5 files for {phase} {subarray}...")

# Combine all HDF5 files for the template event
filepaths = []
for f in Path(dirpath_vel).iterdir():
    if match(rf"travel_time_volumes_{phase.lower()}_{subarray.lower()}.+\.h5", f.name):
        filepaths.append(f)

print(f"Found {len(filepaths)} HDF5 files.")

# Load the travel time volumes from the individual HDF5 files and combine them into a single HDF5 file
filename = f"travel_time_volumes_{phase.lower()}_{subarray.lower()}.h5"
filepath_combined = Path(dirpath_vel) / filename
print(f"Saving the combined travel time volumes to {filepath_combined}...")

for filepath in filepaths:
    phase, subarray, scale_factor, easts_grid, norths_grid, depths_grid, travel_time_dict = load_travel_time_volumes_individual(filepath)
    save_travel_time_volumes_combined(filepath_combined, scale_factor, easts_grid, norths_grid, depths_grid, travel_time_dict)
    print(f"Saved the travel time volumes of scale factor {scale_factor}.")

    remove(filepath)
    print(f"Removed {filepath}.")