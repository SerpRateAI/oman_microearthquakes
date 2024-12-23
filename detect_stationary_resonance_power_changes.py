# Detect abrupt changes in stainary resonance power

# Import
from os.path import join
from pandas import DataFrame, Timedelta
from pandas import read_hdf

from utils_basic import SPECTROGRAM_DIR as indir

# Inputs
mode_name = "PR15295"

window_length = 60.0

power_diff_threshold = 5.0
min_num_sta = 15

print(f"### Detecting abrupt changes in the stationary resonance power of mode {mode_name} ###")

# Read the stationary resonance properties
print("Reading the stationary resonance properties...")
filename = f"stationary_resonance_properties_{mode_name}_geo.h5"
inpath = join(indir, filename)
resonance_df = read_hdf(inpath, key="properties")

# Find the time windows with more stations than the minimum number
resonance_df = resonance_df.groupby("time").filter(lambda x: len(x) > min_num_sta)

# Sort the dataframe by time
resonance_df.set_index(["time", "station"], inplace = True)


# Find the abrupt changes in the power
print("Finding the abrupt changes in the power...")
power_diff_dicts = []
for time1 in resonance_df.index.get_level_values("time").unique():
    time2 = time1 + Timedelta(seconds = window_length)

    if time2 not in resonance_df.index.get_level_values("time"):
        continue

    # Compute the average power for the two time windows
    power1 = resonance_df.loc[time1, "power"].mean()
    power2 = resonance_df.loc[time2, "power"].mean()

    # Compute the power difference
    power_diff = abs(power2 - power1)

    if power_diff > power_diff_threshold:
        power_diff_dicts.append({"time1": time1, "time2": time2, "power_diff": power_diff})

# Convert the list of dictionaries to a dataframe
power_diff_df = DataFrame(power_diff_dicts)
power_diff_df.sort_values("power_diff", ascending = False, inplace = True)
power_diff_df.reset_index(drop = True, inplace = True)
print(f"Detected {len(power_diff_df)} abrupt changes in the power.")

# Save the results
print("Saving the results to a CSV file...")
outpath = join(indir, f"stationary_resonance_power_diff_{mode_name}_geo.csv")
power_diff_df.to_csv(outpath)
print(f"Saved the results to {outpath}.")

print("Saving the results to an HDF5 file...")
outpath = join(indir, f"stationary_resonance_power_diff_{mode_name}_geo.h5")
power_diff_df.to_hdf(outpath, key = "power_diff")
print(f"Saved the results to {outpath}.")


