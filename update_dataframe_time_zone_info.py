# Update the time zone information in the data frame and save the updated data frame to the same file

# Imports
from os.path import join, splitext
from pandas import read_csv, read_hdf

from utils_basic import SPECTROGRAM_DIR as indir

# Inputs
filename = "stationary_resonance_mean_freq_SR38a_geo_num9.csv"
h5key = "properties"

# Read the data
inpath = join(indir, filename)

# Get the file extension of inpath
_, file_ext = splitext(inpath)

if file_ext == ".csv":
    print(f"Reading the data from {inpath}...")
    df = read_csv(inpath, index_col=0, parse_dates=True)
elif file_ext == ".h5":
    print(f"Reading the data from {inpath}...")
    df = read_hdf(inpath, key=h5key)

# Update the time zone information
df.index = df.index.tz_localize("UTC")

# Save the updated data frame to the same file
df.to_csv(inpath, index=True, na_rep="NaN")
print(f"Updated data frame saved to {inpath}.")