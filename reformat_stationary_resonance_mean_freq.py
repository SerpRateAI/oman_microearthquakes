# Reformat the stationary resonance mean frequency data for MATLAB
# Import
from os.path import join
from pandas import read_csv

from utils_basic import SPECTROGRAM_DIR as indir

# Inputs
name = "SR25a"

filename_in = f"stationary_resonance_mean_freq_{name}_geo_num9.csv"

# Read the data
inpath = join(indir, filename_in)
data = read_csv(inpath, parse_dates=True, index_col=0, na_values="nan")

# Reformat the data
filename_out = filename_in.replace(".csv", "_reformat.csv")
outpath = join(indir, filename_out)

data.to_csv(outpath, date_format="%Y-%m-%d %H:%M:%S", na_rep="nan")