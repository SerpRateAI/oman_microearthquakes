"""
Compute the maximum resolvable quality factor of a stationary harmonic series
"""

###
# Import necessary modules
###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame

from utils_basic import SPECTROGRAM_DIR as dirpath

###
# Input arguments
###

parser = ArgumentParser() 
parser.add_argument("--base_name", type=str, help="Base name of the stationary harmonic series", default="PR02549")
parser.add_argument("--base_order", type=int, help="Order of the base harmonic", default=2)
parser.add_argument("--window_length", type=float, help="Window length for computing the STFT in seconds", default=300.0)

args = parser.parse_args()

base_name = args.base_name
base_order = args.base_order
window_length = args.window_length

###
# Load the data
###

filename = f"stationary_harmonic_series_{base_name}_base{base_order}.csv"
filepath = join(dirpath, filename)
harmonic_df = read_csv(filepath)

###
# Compute the maximum resolvable quality factor
###
mode_names = harmonic_df["mode_name"].values
mode_orders = harmonic_df["mode_order"].values
freqs_peak = harmonic_df["observed_freq"].values
min_bandwidth = 1 / window_length * 2

max_qfs = freqs_peak / min_bandwidth

###
# Save the results
###
output_df = DataFrame({
    "mode_name": mode_names,
    "mode_order": mode_orders,
    "observed_freq": freqs_peak,
    "max_qf": max_qfs
})

filename = f"stationary_harmonic_series_max_qf_{base_name}_base{base_order}_window{window_length:.0f}s.csv"
filepath = join(dirpath, filename)
output_df.to_csv(filepath, index=False, na_rep="nan")
print(f"Saved the results to {filepath}")

















