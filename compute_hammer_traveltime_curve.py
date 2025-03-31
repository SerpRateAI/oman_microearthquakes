"""
Compute the traveltime as a function of distance for a hammer shot using a 1D velocity model.
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import linspace
from pandas import DataFrame
from pyrocko.cake import load_model, m2d, d2m
from pyrocko.cake import PhaseDef

from utils_basic import get_geophone_coords, VEL_MODEL_DIR as dirpath

###
# Input parameters
###

# Command line arguments
parser = ArgumentParser()
parser.add_argument("--model_name", type = str, required = True, help = "Name of the velocity model")
parser.add_argument("--min_dist", type = float, default = 1.0, help = "Minimum distance (m)")
parser.add_argument("--max_dist", type = float, default = 100.0, help = "Maximum distance (m)")
parser.add_argument("--num_dist", type = int, default = 100, help = "Number of distances")

# Parse the command line arguments
args = parser.parse_args()
model_name = args.model_name
min_dist = args.min_dist
max_dist = args.max_dist
num_dist = args.num_dist

###
# Load the velocity model
###

# Load the velocity model
print(f"Loading the velocity model {model_name}...")
filename = f"{model_name}.nd"
model_path = join(dirpath, filename)
model = load_model(model_path)

###
# Compute the traveltime curve
###

# Define the phases
phases = [PhaseDef("P"), PhaseDef("p")]

# Define the distance array
distances_m = linspace(min_dist, max_dist, num_dist)

# Compute the traveltime curve
distances_deg = distances_m * m2d
output_dicts = []
for i, distance_deg in enumerate(distances_deg):
    distance_m = distances_m[i]
    arrivals = model.arrivals([distance_deg], phases = phases, zstart = 0.0)
    if len(arrivals) == 0:
        print(f"No arrivals found for distance {distance_m:.2f} m")
        continue

    arrival = arrivals[0]
    
    atime = arrival.t

    output_dicts.append({"distance": distance_m, "traveltime": atime})

# Convert the output dictionaries to a pandas DataFrame
output_df = DataFrame(output_dicts)

# Save the output DataFrame to a CSV file
filename = f"hammer_traveltime_curve_{model_name}.csv"
output_path = join(dirpath, filename)
output_df.to_csv(output_path, index = False)
print(f"Traveltime curve saved to {output_path}")
