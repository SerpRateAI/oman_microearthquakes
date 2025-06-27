# Plot the average power correlation matrix of a subset of stationary harmonic modes.


### Import necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import zeros, corrcoef, nan, isnan, fill_diagonal
from json import loads
from pandas import Timedelta
from pandas import DataFrame
from pandas import read_csv, read_hdf, concat
from matplotlib import colormaps
from matplotlib.pyplot import subplots

from utils_basic import GEO_STATIONS as stations
from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import add_colorbar, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description='Plot the average power correlation matrix of a subset of stationary harmonic modes')
parser.add_argument("--base_mode", type=str, default="PR02549", help="Base name of the harmonic series.")
parser.add_argument("--base_order", type=int, default=2, help="Harmonic number of the base frequency.")

parser.add_argument("--window_length", type=float, default=60.0, help="Window length in seconds for computing the spectrogram.")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between consecutive windows for computing the spectrogram.")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type=float, default=15.0, help="Maximum mean dB value for excluding noise windows.")

parser.add_argument("--figwidth", type=float, default=10.0, help="Figure width in inches.")
parser.add_argument("--figheight", type=float, default=10.0, help="Figure height in inches.")

parser.add_argument("--min_corr", type=float, default=0.0, help="Minimum correlation value.")
parser.add_argument("--max_corr", type=float, default=1.0, help="Maximum correlation value.")

# Parse the command line arguments
args = parser.parse_args()
base_mode = args.base_mode
base_order = args.base_order

figwidth = args.figwidth
figheight = args.figheight

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

min_corr = args.min_corr
max_corr = args.max_corr

# Print the inputs
print("###")
print("Plotting the average power correlation matrix of different stationary harmonic modes.")
print(f"Base name of the harmonic series: {base_mode}.")
print(f"Harmonic number of the base frequency: {base_order:d}.")
print(f"Minimum correlation value: {min_corr}.")
print(f"Maximum correlation value: {max_corr}.")

# Read the cross-correlation results
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_harmonic_avg_power_corr_{base_mode}_base{base_order}_{suffix_spec}_{suffix_peak}.csv"
filepath = join(indir, filename)
corr_df = read_csv(filepath)

# Assemble the correlation matrix
num_harmonics = corr_df["mode_i_order"].unique().shape[0] + 1
corr_mat = zeros((num_harmonics, num_harmonics))

mode_order_dict = {}
for _, row in corr_df.iterrows():
    mode_i_index = int(row["mode_i_index"])
    mode_j_index = int(row["mode_j_index"])

    corr = row["correlation"]

    corr_mat[mode_i_index, mode_j_index] = corr
    corr_mat[mode_j_index, mode_i_index] = corr

    mode_order_dict[mode_i_index] = int(row["mode_i_order"])
    mode_order_dict[mode_j_index] = int(row["mode_j_order"])

# Fill the diagonal with 1.0
fill_diagonal(corr_mat, 1.0)

# Plot the correlation matrix
fig, ax = subplots(1, 1, figsize=(figwidth, figheight))
cmap = colormaps["viridis"]

cax = ax.matshow(corr_mat, cmap=cmap, vmin=min_corr, vmax=max_corr)
ax.set_yticklabels([])

tick_labels = [f"Mode {mode_order_dict[i]:d}" for i in range(num_harmonics)]
print(tick_labels)
ax.set_xticks(range(num_harmonics))
ax.set_xticklabels(tick_labels, rotation=30, ha="left", va="bottom", fontweight="bold", fontsize=12)

for i, label in enumerate(ax.get_xticklabels()):
    if i == 0:
        label.set_color("crimson")
    else:
        label.set_color("black")

# Add a colorbar
bbox = ax.get_position()
position = [bbox.x1 + 0.04, bbox.y0, 0.02, bbox.height]
add_colorbar(fig, position, "Average power correlation", 
             mappable = cax)

# Save the figure
print("Saving the figure...")
filename = f"stationary_harmonic_avg_geo_corr_mat_{base_mode}_base{base_order}_{suffix_spec}_{suffix_peak}.png"
save_figure(fig, filename)











