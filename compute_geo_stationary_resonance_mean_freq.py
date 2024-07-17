# Compute the mean frequency of the stationary resonance for each time window on the geophone stations

# Imports
from os.path import join
from pandas import read_hdf

from utils_basic import GEO_STATIONS as stations, SPECTROGRAM_DIR as indir

# Inputs
name = "SR191a"
time_window = 'min'

min_num_sta = 9

# Read the data
print(f"Reading the properties of {name}...")
filename = f"stationary_resonance_properties_{name}_geo.h5"
inpath = join(indir, filename)

resonance_df = read_hdf(inpath, key="properties")
print(f"{len(resonance_df)} rows read from {inpath}.")

# Find the time windows with the required number of stations recording non-na values
print(f"Finding the time windows with at least {min_num_sta} stations recording non-NaN values...")
filtered_resonance_df = resonance_df.groupby("time").filter(lambda x: x["frequency"].count() >= min_num_sta)

# Group filtered_resonance_df by time and compute the mean frequency of each group while ignoring NaNs
print("Computing the mean frequency of the stationary resonance for each time window...")
mean_freq_by_time = filtered_resonance_df.groupby("time")["frequency"].mean()

print(f"{len(mean_freq_by_time)} time windows with at least {min_num_sta} stations recording non-NaN values.")

# Fill the missing time windows with NaN
print("Filling the missing time windows with NaN...")
mean_freq_by_time = mean_freq_by_time.asfreq(time_window)

# Save the results
print("Saving the results to CSV...")
outpath = join(indir, f"stationary_resonance_mean_freq_{name}_geo_num{min_num_sta}.csv")
mean_freq_by_time.to_csv(outpath, index = True, na_rep = "NaN")
print(f"Results saved to {outpath}.")