"""
Compute the log histogram of the quality factors of a stationary resonance
"""

### Import the necessary libraries ###

from os.path import join
from argparse import ArgumentParser
from numpy import log10, histogram, concatenate, amax, percentile, isnan
from pandas import read_hdf, DataFrame  

from utils_basic import SPECTROGRAM_DIR as dirname_spec, GEO_COMPONENTS as components
from utils_spec import get_spec_peak_file_suffix, get_spectrogram_file_suffix

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Compute the log histogram of the quality factors of a stationary resonance.")

parser.add_argument("--num_bins_log_qf", type=int, help="Number of bins for the quality factor histogram", default=25)
parser.add_argument("--min_log_qf", type=float, help="Minimum quality factor for the histogram", default=1.0)
parser.add_argument("--max_log_qf", type=float, help="Maximum quality factor for the histogram", default=5.0)

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")
parser.add_argument("--window_length", type=float, help="Window length of the STFT", default=300.0)
parser.add_argument("--overlap", type=float, help="Overlap of the STFT", default=0.0)
parser.add_argument("--min_prom", type=float, help="Minimum prominence for detecting peaks in the power spectrum", default=15.0)
parser.add_argument("--min_rbw", type=float, help="Minimum reversed bandwidth for detecting peaks in the power spectrum", default=15.0)
parser.add_argument("--max_mean_db", type=float, help="Maximum mean power for excluding a time window in peak detection", default=15.0)

parser.add_argument("--quantile", type=float, help="Quantile of the quality factors to compute", default=95.0)

# Parse the arguments
args = parser.parse_args()

mode_name = args.mode_name
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
quantile = args.quantile

num_bins_log_qf = args.num_bins_log_qf
min_log_qf = args.min_log_qf
max_log_qf = args.max_log_qf

### Read the data ###
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_spec_peaks = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_spec_peaks}.h5"
filepath = join(dirname_spec, filename)
resonance_df = read_hdf(filepath, key = "properties")

### Compute the log histogram of the quality factors ###
qfs = []
for component in components:
    # Get quality factors for this component and remove NaN values
    qf_vals = resonance_df[f"quality_factor_{component.lower()}"].values
    qf_vals = qf_vals[~isnan(qf_vals)]
    qfs.append(qf_vals)

qfs = concatenate(qfs)
log_qfs = log10(qfs)

counts, bin_edges = histogram(log_qfs, bins = num_bins_log_qf, range = (min_log_qf, max_log_qf))
counts = counts / amax(counts)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

### Compute the confidence interval of the quality factors ###
qf_lower, qf_upper = percentile(qfs, [(100.0 - quantile) / 2.0, 100.0 - (100.0 - quantile) / 2.0])
print(f"{quantile}% confidence interval of the quality factors: {qf_lower:.2f} - {qf_upper:.2f}")

### Save the results ###
filename = f"stationary_resonance_qf_log_histogram_{mode_name}.csv"
filepath = join(dirname_spec, filename)

data_df = DataFrame({"log_quality_factor": bin_centers, "normalized_count": counts})
data_df.to_csv(filepath, index = False)
print(f"Saved the results to {filepath}")

filename = f"stationary_resonance_qf_quantile_interval_{quantile}.csv"
filepath = join(dirname_spec, filename)
data_df = DataFrame({"lower_qf": [qf_lower], "upper_qf": [qf_upper]})
data_df.to_csv(filepath, index = False)
print(f"Saved the confidence interval to {filepath}")









