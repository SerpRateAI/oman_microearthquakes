# Count the number of harmonics recorded by each station at each time window and rank them in descending order

### Import the necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from pandas import concat, read_csv, read_hdf 

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for counting the number of harmonics recorded by each station at each time window.")
parser.add_argument("--base_mode", type=str, default="PR02549", help="Base name of the harmonic series.")
parser.add_argument("--base_num", type=int, default=2, help="Harmonic number of the base frequency.")

parser.add_argument("--window_length", type=float, default=60.0, help="Window length in seconds for computing the spectrogram.")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between consecutive windows for computing the spectrogram.")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type=float, default=10.0, help="Maximum mean dB value for excluding noise windows.")


# Parse the command line arguments
args = parser.parse_args()
base_mode = args.base_mode
base_num = args.base_num

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Print the inputs
print("###")
print("Counting the number of harmonics recorded by each station at each time window and ranking them in descending order.")
print(f"Base name of the harmonic series: {base_mode}.")
print(f"Harmonic number of the base frequency: {base_num:d}.")
print("###")

### Read the data ###
# Read the harmonic series
print("Loading the harmonic series...")
filename = f"stationary_harmonic_series_{base_mode}_base{base_num}.csv"
filepath = join(indir, filename)
harmonic_df = read_csv(filepath)

# Generate the spectrogram and spectral-peak file suffices
spec_suffix = get_spectrogram_file_suffix(window_length, overlap)
peak_suffix = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

# Read the properties of all harmonics
resonance_dfs = []
for mode_name in harmonic_df["name"]:
    if mode_name.startswith("MH"):
        print(f"Skipping the non-existent mode {mode_name}...")
        continue

    filename = f"stationary_resonance_properties_geo_{mode_name}_{spec_suffix}_{peak_suffix}.h5"
    filepath = join(indir, filename)

    print(f"Loading the properties of the stationary resonances for {mode_name}...")
    resonance_df = read_hdf(filepath, key="properties")
    resonance_df["mode"] = mode_name

    resonance_dfs.append(resonance_df)

resonance_df = concat(resonance_dfs, ignore_index=True)
# print(resonance_df.columns)

# Count the number of harmonics recorded by each station at each time window
print("Counting the number of harmonics recorded by each station at each time window...")
harmonic_count_df = resonance_df.groupby(["station", "time"]).size().reset_index(name="num_harmonics")
harmonic_count_df = harmonic_count_df.sort_values(by=["num_harmonics", "station", "time"], ascending=[False, True, True])
harmonic_count_df = harmonic_count_df.reset_index(drop=True)

# Save the harmonic count dataframe
print("Saving the harmonic count dataframe...")
filename = f"stationary_harmonic_count_{base_mode}_base{base_num}_{spec_suffix}_{peak_suffix}.csv"
filepath = join(indir, filename)
harmonic_count_df.to_csv(filepath)
print(f"Harmonic count dataframe saved to {filepath}.")
