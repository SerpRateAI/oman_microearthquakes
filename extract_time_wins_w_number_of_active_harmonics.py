# Extract the time windows when more than a certain number of harmonics of a series are active 

# Import the necessary libraries
from os.path import join
from numpy import abs, argmin, nan

from pandas import concat, read_csv, read_hdf, DataFrame

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations

# Inputs
base_name = "PR02549"
base_mode = 2

min_num_modes = 5 # Minimum number of active modes

# Read the inputs
# Read the harmonic series
print(f"Reading the harmonic series with the base name of {base_name} and mode of {base_mode}...")
filename_in = f"stationary_harmonic_series_{base_name}_base{base_mode:d}.csv"
inpath = join(indir, filename_in)
harmonic_series_df = read_csv(inpath, index_col=0, dtype={"detected": bool})
harmonic_series_df = harmonic_series_df[harmonic_series_df["detected"]]

# Read the properties of the stationary resonances
print("Reading the properties of the stationary resonances...")
property_dict = {}
for mode_name in harmonic_series_df["name"]:
    print(f"Reading the properties of {mode_name}...")
    # Read the properties of the stationary resonance
    filename_in = f"stationary_resonance_properties_{mode_name}_geo.h5"
    inpath = join(indir, filename_in)
    property_df = read_hdf(inpath, key = "properties")
    
    property_dict[mode_name] = property_df
print("")

# Extract the time windows for each station
print(f"Extracting the time windows with more than {min_num_modes:d} active modes for each station...")
time_window_dfs = []
for station in stations:
    print(f"Working on {station}... ")
    station_dfs = []
    for mode_name, property_df in property_dict.items():
        print(f"Reading the properties of {mode_name}... ")
        # Read the properties of the stationary resonance
        property_df = property_df[property_df["station"] == station]
        station_dfs.append(property_df)

    # Concatenate the dataframes
    station_df = concat(station_dfs, axis = 0)

    # Extract the time windows with more than a certain number of active modes
    print(f"Extracting the time windows with more than {min_num_modes:d} active modes... ")
    count_sr = station_df.groupby("time").size()
    count_sr.name = "num_modes"
    count_sr = count_sr[count_sr >= min_num_modes]
    count_df = count_sr.to_frame()
    count_df["station"] = station
    count_df.reset_index(inplace = True)
    count_df = count_df[["station", "time", "num_modes"]]

    
    print(f"Number of time windows with more than {min_num_modes:d} active modes: {count_df.shape[0]}")
    print("")
    time_window_dfs.append(count_df)

# Concatenate the dataframes
time_window_df = concat(time_window_dfs, axis = 0)

# Save the results
print("Saving the results... ")
print("Saving the results to CSV... ")
filename_out = f"time_windows_w_number_of_active_harmonics_{base_name}_base{base_mode:d}_num{min_num_modes:d}.csv"
outpath = join(indir, filename_out)
time_window_df.to_csv(outpath, index = True)
print(f"Results are saved to {outpath}.")

print("Saving the results to HDF... ")
filename_out = f"time_windows_w_number_of_active_harmonics_{base_name}_base{base_mode:d}_num{min_num_modes:d}.h5"
outpath = join(indir, filename_out)
time_window_df.to_hdf(outpath, key = "time_windows", mode = "w")
print(f"Results are saved to {outpath}.")



