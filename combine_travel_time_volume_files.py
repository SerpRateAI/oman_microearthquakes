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
from pandas import read_csv

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

print(f"Combining HDF5 files for {phase} phase and Subarray {subarray}...")

# Load the scale factors
filename = f"scale_factors_{phase.lower()}.csv"
filepath = Path(dirpath_vel) / filename
scale_factors = read_csv(filepath)["scale_factor"].values
print(f"Loaded the scale factors: {scale_factors}")


# Load the travel time volumes from the individual HDF5 files and combine them into a single HDF5 file
filename = f"travel_time_volumes_{phase.lower()}_{subarray.lower()}.h5"
filepath_combined = Path(dirpath_vel) / filename
print(f"Saving the combined travel time volumes to {filepath_combined}...")

for scale_factor in scale_factors:
    print(f"Loading the travel time volumes for scale factor {scale_factor}...")
    filename = f"travel_time_volumes_{phase.lower()}_{subarray.lower()}_scale{scale_factor:.1f}.h5"
    filepath = Path(dirpath_vel) / filename
    easts_grid, norths_grid, depths_grid, travel_time_dict = load_travel_time_volumes_individual(filepath)
    print(f"Saving the travel time volumes for scale factor {scale_factor}...")
    save_travel_time_volumes_combined(filepath_combined, scale_factor, easts_grid, norths_grid, depths_grid, travel_time_dict)
    print(f"Saved the travel time volumes for scale factor {scale_factor}.")

print(f"Combined the travel time volumes.")