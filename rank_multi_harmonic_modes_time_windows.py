# Rank the time windows based on the power and number of stations of multiple harmonic modes
#

# Imports
from os.path import join
from pandas import MultiIndex
from pandas import concat, read_hdf, merge

from utils_basic import SPECTROGRAM_DIR as indir

# Inputs
mode_names = ["PR02549", "PR03822"]

# Read the properties of the harmonic modes
print("Reading the properties of the harmonic modes...")
resonance_dfs = []
for mode_name in mode_names:
    print(f"Reading the properties of {mode_name}...")
    filename = f"stationary_resonance_geo_summary_{mode_name}.h5"
    inpath = join(indir, filename)
    resonance_df = read_hdf(inpath, key = "properties")
    resonance_df["time"] = resonance_df.index
    resonance_df.reset_index(drop = True, inplace = True)
    resonance_df["mode_name"] = mode_name
    resonance_df.rename(columns={"mean_power": "mean_station_power"}, inplace=True)
    resonance_dfs.append(resonance_df)

# Merge the properties of the harmonic modes
print("Merging the properties of the harmonic modes...")
for i, resonance_df in enumerate(resonance_dfs):
    if i == 0:
        merged_resonance_df = resonance_df
    else:
        merged_resonance_df = merge(merged_resonance_df, resonance_df, on = "time", how = "inner", suffixes = (f"_{i}", f"_{i+1}"))

print(f"{len(merged_resonance_df)} time windows with all harmonic modes estimated.")

# Compute the mean power and number of stations for each time window
print("Computing the mean power and number of stations for each time window...")
merged_resonance_df["mean_mode_power"] = merged_resonance_df[[f"mean_station_power_{i + 1}" for i in range(len(mode_names))]].mean(axis = 1)
merged_resonance_df["mean_num_stations"] = merged_resonance_df[[f"num_stations_{i + 1}" for i in range(len(mode_names))]].mean(axis = 1)

# Reorganize the merged data frame into one with hierchical columns
print("Reorganizing the merged data frame...")
melted_dfs = []
for i, mode_name in enumerate(mode_names):
    melted_df = merged_resonance_df[["time", f"frequency_{i + 1}", f"mean_station_power_{i + 1}", f"num_stations_{i + 1}"]]
    melted_df.columns = ["time", "frequency", "mean_station_power", "num_stations"]
    melted_df["mode_name"] = mode_name
    melted_dfs.append(melted_df)

combined_resonance_df = concat(melted_dfs, axis = 0)
combined_resonance_df = combined_resonance_df.merge(merged_resonance_df[["time", "mean_mode_power", "mean_num_stations"]], on = "time", how = "left")
combined_resonance_df.set_index(['time', 'mode_name'], inplace = True)
combined_resonance_df = combined_resonance_df.reindex(columns = ["mean_num_stations", "mean_mode_power", "frequency", "num_stations", "mean_station_power"])

print("Ranking the time windows based first on the mean number of stations and then on the mean power of the harmonic modes...")
combined_resonance_df.sort_values(by = ["mean_num_stations", "mean_mode_power", "time"], ascending = [False, False, True], inplace = True)



# Save the results
print("Saving the results to text format...")
filename = f"multi_harmonic_modes_summary.txt"
outpath = join(indir, filename)
string_out = combined_resonance_df.to_string()
with open(outpath, "w") as f:
    f.write(string_out)
print(f"Results saved to {outpath}.")

print("Saving the results to HDF format...")



