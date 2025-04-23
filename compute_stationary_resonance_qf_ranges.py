"""
Compute the ranges of quality factor for each mode of a harmonic series.
"""

###
# Import modules
###

from argparse import ArgumentParser
from os.path import join
from pandas import read_csv, DataFrame

from utils_basic import SPECTROGRAM_DIR as dirname_spec
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

###
# Input arguments
###

parser = ArgumentParser()
parser.add_argument("--base_name", type=str, help="Base mode name", default="PR02549")
parser.add_argument("--base_order", type=int, help="Base mode order", default=2)

parser.add_argument("--window_length", type=float, help="Window length for computing the STFT", default=300.0)
parser.add_argument("--overlap", type=float, help="Overlap for computing the STFT", default=0.0)
parser.add_argument("--min_prom", type=float, help="Minimum prominence", default=15.0)
parser.add_argument("--min_rbw", type=float, help="Minimum reversed bandwidth", default=15.0)
parser.add_argument("--max_mean_db", type=float, help="Maximum mean power for window exclusion", default=10.0)

args = parser.parse_args()

base_name = args.base_name
base_order = args.base_order

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

###
# Read the list of harmonic series
###

# Read the harmonic series
filename = f"stationary_harmonic_series_{base_name}_base{base_order}.csv"
harmonic_df = read_csv(join(dirname_spec, filename))


###
# Compute the ranges of quality factor for each mode of a harmonic series
###

suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

output_dicts = []
for _, row in harmonic_df.iterrows():
    mode_name = row["mode_name"]
    detected = row["detected"]

    if not detected:
        continue
    
    # Read the properties of the mode
    filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.csv"
    properties_df = read_csv(join(dirname_spec, filename))

    # Compute the ranges of quality factor
    min_qf = properties_df["quality_factor"].min()
    max_qf = properties_df["quality_factor"].max()

    output_dicts.append({"mode_name": mode_name, "min_qf": min_qf, "max_qf": max_qf})

output_df = DataFrame(output_dicts)
outpath = join(dirname_spec, f"stationary_resonance_qf_ranges_{base_name}_base{base_order}.csv")
output_df.to_csv(outpath, index = False)
print(f"Saved the results to {outpath}.")