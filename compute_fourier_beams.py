# Compute the Fourier frequency-dependent beams of one mode for all time windows

### Import ###
from os.path import join
from argparse import ArgumentParser
from multiprocessing import Pool
from pandas import Timestamp
from pandas import read_hdf
from time import time

from utils_basic import SPECTROGRAM_DIR as indir, BEAM_DIR as outdir
from utils_basic import GEO_STATIONS as stations_all, GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_basic import INNER_STATIONS_A as inner_stations_a, INNER_STATIONS_B as inner_stations_b, MIDDLE_STATIONS_A as middle_stations_a, MIDDLE_STATIONS_B as middle_stations_b
from utils_basic import get_geophone_coords
from utils_array import FourierBeamCollection
from utils_array import get_fourier_beam_window, get_slowness_axes, select_stations

# Wrapper function for multiprocessing
def get_fourier_beam_window_wrapper(freq_reson, peak_coeff_df, coords_df, xslowax, yslowax, window_length, kwargs):
    return get_fourier_beam_window(freq_reson, peak_coeff_df, coords_df, xslowax, yslowax, window_length, **kwargs)


### Inputs ###
# Data
parser = ArgumentParser(description="Input parameters for computing the Fourier beams of one mode for time windows in a specific subarray.")
parser.add_argument("--mode_name", type=str, help="Name of the mode.")
parser.add_argument("--array_selection", type=str, help="Selection of the subarray: A or B.")


parser.add_argument("--window_length", type=float, default=60.0, help="Length of the time window in seconds.")
parser.add_argument("--array_aperture", type=str, default="small", help="Aperture of the array: small, medium, or large.")
parser.add_argument("--min_vel_app", type=float, default=500.0, help="Minimum apparent velocity in m/s.")
parser.add_argument("--num_vel_app", type=int, default=51,help="Number of apparent velocities along each axis.")
parser.add_argument("--num_process", type=int, default=4, help="Number of processes for multiprocessing.")

args = parser.parse_args()
mode_name = args.mode_name
array_selection = args.array_selection

window_length = args.window_length
array_aperture = args.array_aperture
min_vel_app = args.min_vel_app
num_vel_app = args.num_vel_app
num_process = args.num_process

# Print the inputs
print("###")
print(f"Computing the Fourier beams of {mode_name} for all time windows in a specific subarray.")
print(f"Selection of the subarray: {array_selection}.")

print(f"Length of the time window: {window_length:.1f} s.")
print(f"Aperture of the array: {array_aperture}.")
print(f"Minimum apparent velocity: {min_vel_app:.1f} m/s.")
print(f"Number of apparent velocities along each axis: {num_vel_app:d}.")
print("###")
print("")

### Read the data ###
# Load the station coordinates
print("Loading the station coordinates...")
coords_df = get_geophone_coords()

# Read the stationary resonance information
print("Loading the phase information of the stationary resonances...")
filename = f"stationary_resonance_properties_{mode_name}_geo.h5"
inpath = join(indir, filename)
properties_df = read_hdf(inpath, key = "properties")
# print(properties_df.columns)

# Get the stations and the minimum number of stations
stations, min_num_stations = select_stations(array_selection, array_aperture)
print(f"Stations: {stations}")
print(f"Minimum number of stations: {min_num_stations}")
properties_df = properties_df[properties_df["station"].isin(stations)]


# Define the x and y slow axes
xslowax, yslowax = get_slowness_axes(min_vel_app, num_vel_app)

# Compute the Fourier frequency-dependent beams for all time windows in serial
if num_process == 1:
    print("Computing the Fourier beams for all time windows in serial...")
    beam_windows = []
    for time_window in properties_df["time"].unique():
        clock1 = time()
        print(f"Time window: {time_window}")

        # Extract the stationary-resonance information of the desired time window
        properties_window_df = properties_df.loc[properties_df["time"] == time_window]
        freq_reson = properties_window_df["frequency"].values[0]    
        print(f"Resonance frequency: {freq_reson:.3f} Hz.")

        # Compute the Fourier beams for the frequency of the stationary resonance
        print(f"Computing the Fourier beams...")
        beam_window = get_fourier_beam_window(freq_reson, properties_window_df, coords_df, xslowax, yslowax, window_length,
                                              min_num_stations = min_num_stations)
        
        if beam_window is not None:
            beam_windows.append(beam_window)

        clock2 = time()
        print("Done.")
        print(f"Elapsed time: {clock2 - clock1:.2f} seconds.")
        print("")

    print("All time windows processed.")
    print("")
else:
    print(f"Computing the Fourier beams for all time windows in parallel using {num_process} processes...")
    # Assemble the arguments
    print("Assembling the arguments for multiprocessing...")
    args = []
    for time_window in properties_df["time"].unique():
        # Extract the stationary-resonance information of the desired time window
        properties_window_df = properties_df.loc[properties_df["time"] == time_window]
        freq_reson = properties_window_df["frequency"].values[0]  

        arg = [freq_reson, properties_window_df, coords_df, xslowax, yslowax, window_length, {"min_num_stations": min_num_stations}]
        args.append(arg)

    # Assign the arguments to the processes
    print(f"Processing the time windows...")
    with Pool(num_process) as pool:
        beam_windows = pool.starmap(get_fourier_beam_window_wrapper, args)

    # Remove the None values
    print("Removing the None values...")
    beam_windows = [beam_window for beam_window in beam_windows if beam_window is not None] 

    print("All time windows processed.")
    print("")

# Build the beam window collection
print("Building the beam window collection...")
beam_collection = FourierBeamCollection(beam_windows)
print(f"Number of beam windows: {len(beam_collection)}")

### Save the beam window collection to HDF5
print("Saving the beam window collection to an HDF5 file...")
array_suffix = f"array_{array_selection.lower()}_{array_aperture}"

filename = f"fourier_beam_image_collection_{mode_name}_{array_suffix}.h5"
outpath = join(outdir, filename)

beam_collection.to_hdf(outpath)


    