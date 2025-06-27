# Compute the power correlation matrix between all modes for each geophone station.

### Import necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import zeros, corrcoef, nan, isnan, fill_diagonal
from json import loads
from pandas import Timedelta
from pandas import DataFrame
from pandas import read_csv, read_hdf, concat
from matplotlib.pyplot import subplots, get_cmap

from utils_basic import GEO_STATIONS as stations
from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import add_colorbar, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description='Plot the average power correlation matrix of different stationary harmonic modes')
parser.add_argument("--base_mode", type=str, default="PR02549", help="Base name of the harmonic series.")
parser.add_argument("--base_order", type=int, default=2, help="Harmonic number of the base frequency.")
parser.add_argument("--mode_orders_to_avg", type=int, default=[2, 3, 4, 6, 9, 10, 12, 13, 14, 15], nargs="+", help="Mode orders to average across all available stations.")

parser.add_argument("--min_num_window", type=int, default=10, help="Minimum number of time windows for computing the correlation values.")
parser.add_argument("--min_num_station", type=int, default=5, help="Minimum number of stations for computing the correlation values.")

parser.add_argument("--window_length", type=float, default=60.0, help="Window length in seconds for computing the spectrogram.")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between consecutive windows for computing the spectrogram.")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type=float, default=3.0, help="Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type=float, default=10.0, help="Maximum mean dB value for excluding noise windows.")

# Constants
figwidth_sta = 20.0

figwidth_avg = 10.0

min_corr = 0.0
max_corr = 1.0

num_row = 6
num_col = 6

order_label_offset = -0.3

# Parse the command line arguments
args = parser.parse_args()
base_mode = args.base_mode
base_order = args.base_order
mode_orders_to_avg = args.mode_orders_to_avg

min_num_window = args.min_num_window
min_num_station = args.min_num_station

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Print the inputs
print("###")
print("Plotting the average power correlation matrix of different stationary harmonic modes.")
print(f"Base name of the harmonic series: {base_mode}.")
print(f"Harmonic number of the base frequency: {base_order:d}.")
print(f"Mode orders to average across all available stations: {mode_orders_to_avg}.")
print(f"Minimum number of time windows for computing the correlation values: {min_num_window:d}.")

### Read the harmonic series ###
# Read the harmonic series
filename = f"stationary_harmonic_series_{base_mode}_base{base_order}.csv"
filepath = join(indir, filename)
harmonic_df = read_csv(filepath)

# Remove the missing harmonics
harmonic_df = harmonic_df.drop(harmonic_df[harmonic_df["mode_name"].str.startswith("MH")].index)
harmonic_df = harmonic_df.reset_index(drop=True)

### Read the properties of all harmonics ###
print("Loading the properties of the stationary resonances for each harmonic mode...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

# Initialize the list to store the data frames
resonance_dfs = []
for mode_name in harmonic_df["mode_name"]:
    if mode_name.startswith("MH"):
        print(f"Skipping the non-existent mode {mode_name}...")
        continue

    filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
    filepath = join(indir, filename)

    print(f"Loading the properties of the stationary resonances for {mode_name}...")
    resonance_df = read_hdf(filepath, key="properties")
    resonance_df["mode_name"] = mode_name

    resonance_dfs.append(resonance_df)

resonance_df = concat(resonance_dfs, ignore_index=True)

### Compute the correlation matrix ###
print("Computing the correlation matrices for each station...")
# Initialize the correlation matrix
num_harmonics = harmonic_df.shape[0]
corr_mat = zeros((num_harmonics, num_harmonics))
sta_corr_dict = {} # Dictionary to store the correlation matrices for each station

# Define the color map
cmap = get_cmap("viridis")
cmap.set_bad("gray")

# Compute the correlation matrix for each station
sta_corr_dicts = []
for station in stations:
    print(f"Working on {station}...")

    for i in range(num_harmonics):
        for j in range(i + 1, num_harmonics):
            # Read the properties of the stationary resonances
            mode_i_name = harmonic_df["mode_name"][i]
            mode_j_name = harmonic_df["mode_name"][j]

            mode_i_order = harmonic_df["mode_order"][i]
            mode_j_order = harmonic_df["mode_order"][j]

            # Extract the power values
            resonance_i_sta_df = resonance_df[(resonance_df["mode_name"] == mode_i_name ) & (resonance_df["station"] == station)]
            resonance_j_sta_df = resonance_df[(resonance_df["mode_name"] == mode_j_name ) & (resonance_df["station"] == station)]

            # Merge the data frames on the time column
            merged_df = resonance_i_sta_df.merge(resonance_j_sta_df, on="time", suffixes=("_i", "_j"), how="inner")

            if merged_df.shape[0] < min_num_window:
                print(f"Skipping {station} due to insufficient number of time windows...")
                corr = nan
            else:
                # Compute the correlation coefficient
                powers_i = merged_df["total_power_i"].values
                powers_j = merged_df["total_power_j"].values
                corr = corrcoef(powers_i, powers_j)[0, 1]

            # Store the correlation value
            sta_corr_dicts.append({"station": station, "mode_i_name": mode_i_name, "mode_j_name": mode_j_name, "mode_i_order": mode_i_order, "mode_j_order": mode_j_order, "correlation": corr})

# Convert the list of dictionaries to a data frame
sta_corr_df = DataFrame(sta_corr_dicts)

# Save the correlation data frame
print("Saving the correlation data frame...")
filename = f"stationary_harmonic_station_power_corr_{base_mode}_base{base_order}_{suffix_spec}_{suffix_peak}.csv"
filepath = join(indir, filename)
sta_corr_df.to_csv(filepath, na_rep="nan")
print(f"Correlation data frame saved to {filepath}.")

### Plot the correlation matrix for each station ###
num_station = len(stations)
if num_station != num_row * num_col:
    raise ValueError("The number of stations does not match the number of subplots.")

fig, ax = subplots(num_row, num_col, figsize=(figwidth_sta, figwidth_sta), sharex=True, sharey=True)
for i in range(num_row):
    for j in range(num_col):
        station = stations[num_col * i + j]
        sta_corr_mat = zeros((num_harmonics, num_harmonics))

        for k in range(num_harmonics):
            for l in range(k + 1, num_harmonics):
                mode_i_name = harmonic_df["mode_name"][k]
                mode_j_name = harmonic_df["mode_name"][l]

                # Extract the correlation value
                corr = sta_corr_df[(sta_corr_df["station"] == station) & (sta_corr_df["mode_i_name"] == mode_i_name) & (sta_corr_df["mode_j_name"] == mode_j_name)]["correlation"].values[0]
                sta_corr_mat[k, l] = corr
                sta_corr_mat[l, k] = corr

        # Fill the diagonal with 1.0
        fill_diagonal(sta_corr_mat, 1.0)

        cax = ax[i, j].matshow(sta_corr_mat, cmap=cmap, vmin=min_corr, vmax=max_corr)
        ax[i, j].set_title(station, fontweight="bold")

# Add a colorbar
bbox = ax[-1, -1].get_position()
position = [bbox.x1 + 0.04, bbox.y0, 0.02, bbox.height]
add_colorbar(fig, position, "Correlation", 
             mappable = cax)

# Save the figure
print("Saving the figure...")
filename = f"stationary_harmonic_station_corr_mat_{base_mode}_base{base_order}_{suffix_spec}_{suffix_peak}.png"
save_figure(fig, filename)

### Compute the average correlation matrix ###
print("Computing the average correlation matrix...")

# Read the list of mode numbers whose power correlatios are to be averaged across all available stations
num_harmonics_to_avg = len(mode_orders_to_avg)

avg_corr_mat = zeros((num_harmonics_to_avg, num_harmonics_to_avg))

avg_corr_dicts = []
for i, mode_i_order in enumerate(mode_orders_to_avg):
    for j in range(i + 1, len(mode_orders_to_avg)):
        mode_j_order = mode_orders_to_avg[j]

        # Extract the correlation values
        corr_values = sta_corr_df[(sta_corr_df["mode_i_order"] == mode_i_order) & (sta_corr_df["mode_j_order"] == mode_j_order)]["correlation"].values

        # Exclude the NaN values
        corr_values = corr_values[~isnan(corr_values)]

        if len(corr_values) < min_num_station:
            print(f"Skipping the pair (Mode {mode_i_order}-{mode_j_order}) due to insufficient number of stations...")
            avg_corr = nan
        else:
            avg_corr = corr_values.mean()

        avg_corr_mat[i, j] = avg_corr
        avg_corr_dicts.append({"mode_i_index": i, "mode_j_index": j, "mode_i_order": mode_i_order, "mode_j_order": mode_j_order, "correlation": avg_corr})

# Convert the list of dictionaries to a data frame
avg_corr_df = DataFrame(avg_corr_dicts)

# Save the average correlation data frame
print("Saving the average correlation data frame...")
filename = f"stationary_harmonic_avg_power_corr_{base_mode}_base{base_order}_{suffix_spec}_{suffix_peak}.csv"
filepath = join(indir, filename)
avg_corr_df.to_csv(filepath, na_rep="nan", index=False)
print(f"Average correlation data frame saved to {filepath}.")

# Fill the lower triangular part
avg_corr_mat = avg_corr_mat + avg_corr_mat.T

# Fill the diagonal with 1.0
fill_diagonal(avg_corr_mat, 1.0)

# Plot the correlation matrix
fig, ax = subplots(1, 1, figsize=(figwidth_avg, figwidth_avg))

cax = ax.matshow(avg_corr_mat, cmap=cmap, vmin=min_corr, vmax=max_corr)
ax.set_title("Average correlation matrix", fontweight="bold")
ax.set_xticklabels([])
ax.set_yticklabels([])

# Add the mode labels
for i in range(num_harmonics_to_avg):
    mode_order = mode_orders_to_avg[i]
    ax.text(order_label_offset, i, f"Mode {mode_order:d}", ha="right", va="center", fontweight="bold", fontsize=10)
    
# Add a colorbar
bbox = ax.get_position()
position = [bbox.x1 + 0.04, bbox.y0, 0.02, bbox.height]
add_colorbar(fig, position, "Correlation", 
             mappable = cax)

# Save the figure
print("Saving the figure...")
filename = f"stationary_harmonic_avg_geo_corr_mat_{base_mode}_base{base_order}_{suffix_spec}_{suffix_peak}.png"
save_figure(fig, filename)











