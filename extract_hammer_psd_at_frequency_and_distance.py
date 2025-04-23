"""
Extract the PSD of hammer shots recorded at a station at a certain frequency and shot distance
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import sqrt, interp
from pandas import read_csv, DataFrame  
from matplotlib.pyplot import subplots

from utils_basic import MT_DIR as dirpath_mt, LOC_DIR as dirpath_loc
from utils_basic import get_geophone_coords
from utils_plot import save_figure

parser = ArgumentParser(description = "Plot the PSD of hammer shots at a certian frequency recorded at a station vs distance to the shots.")

parser.add_argument("--station", type = str, help = "The station name")
parser.add_argument("--freq_target", type = float, help = "The frequency of the PSD to plot", default = 150.0)

args = parser.parse_args()

station = args.station
freq_target = args.freq_target

###
# Load the data
###

# Load the station locations
station_df = get_geophone_coords()
east_sta = station_df.loc[station, "east"]
north_sta = station_df.loc[station, "north"]

# Load the hammer locations
filename = "hammer_locations.csv"
filepath = join(dirpath_loc, filename)
hammer_df = read_csv(filepath, dtype = {"hammer_id": str})

###
# Extract the PSD at the target frequency for each hammer

psds_target = []
distances = []
for _, row in hammer_df.iterrows():
    hammer_id = row["hammer_id"]
    east_hammer = row["east"]
    north_hammer = row["north"]

    # Load the MT auto-spectra
    filename = f"hammer_mt_aspecs_{hammer_id}_{station}.csv"
    filepath = join(dirpath_mt, filename)
    psd_df = read_csv(filepath)

    # Extract the PSD at the target frequency
    freqs = psd_df["frequency"]
    psds = psd_df["aspec_total"]

    # Interpolate the PSD at the target frequency
    psd = interp(freq_target, freqs, psds)

    distance = sqrt((east_hammer - east_sta)**2 + (north_hammer - north_sta)**2)
    psds_target.append(psd)
    distances.append(distance)

###
# Save the PSDs and distances
###

output_df = DataFrame({"hammer_id": hammer_df["hammer_id"],
                       "distance": distances,
                       "psd": psds_target})

# Sort the dataframe by distance
output_df = output_df.sort_values(by = "distance")

# Save the dataframe
output_filepath = join(dirpath_mt, f"hammer_mt_psd_vs_distance_{freq_target:.0f}hz_{station}.csv")
output_df.to_csv(output_filepath, index = False)
print(f"Saved the PSDs and distances to {output_filepath}")