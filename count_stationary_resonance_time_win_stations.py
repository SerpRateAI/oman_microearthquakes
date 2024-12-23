# Count the number of active stations at each time window for a stationary resonance
from os.path import join
from pandas import read_hdf, DataFrame

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo

# Inputs
name = "PR00397"

# Load the stationary resonance properties
print(f"Loading the stationary resonance properties of {name}...")
filename = f"stationary_resonance_properties_{name}_geo.h5"
inpath = join(indir, filename)
reson_df = read_hdf(inpath, key="properties")

# Count the occurance of each time window
print(f"Counting the number of active stations at each time window for {name}...")
count_df = reson_df.groupby("time").size().to_frame()
count_df.columns = ["count"]
count_df.sort_values(by=["count", "time"], ascending=[False, True], inplace=True)

# Save the results
print(f"Saving the results...")
outpath = join(indir, f"stationary_resonance_time_window_station_counts_{name}.csv")
count_df.to_csv(outpath, index=True, na_rep="NaN")
print(f"Results saved to {outpath}.")
