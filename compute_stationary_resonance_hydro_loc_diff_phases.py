# Compute the differential phases of a stationary resonance between two hydrophone loctions
# The phase difference is defined as location 2 - location 1

### Import the required libraries ###
from os.path import join
from argparse import ArgumentParser
from pandas import read_hdf
from numpy import linspace, arange
from matplotlib.pyplot import subplots

from utils_spec import SPECTROGRAM_DIR as indir
from utils_spec import get_stream_fft
from utils_plot import save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description = "Compute the differential phases of a stationary resonance between two hydrophone locations")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--station", type = str, help = "Hydrophone station to process")
parser.add_argument("--location1", type = str, help = "First hydrophone location")
parser.add_argument("--location2", type = str, help = "Second hydrophone location")

parser.add_argument("--window_length", type = float, help = "Length of the window in seconds")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap fraction between adjacent windows")
parser.add_argument("--min_prom", type=float, default=15.0, help="Prominence threshold for peak detection")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Reverse bandwidth threshold for peak detection")
parser.add_argument("--max_mean_db", type=float, default=-10.0, help="Maximum mean dB value for peak detection")

parser.add_argument("--min_freq_peak", type=float, default=0.0, help="Minimum frequency in Hz for peak detection")
parser.add_argument("--max_freq_peak", type=float, default=200.0, help="Maximum frequency in Hz for peak detection")
parser.add_argument("--phase_bin_width", type=float, default=10.0, help="Width of the phase bins in degrees")

# Parse the command line arguments
args = parser.parse_args()
station = args.station
location1 = args.location1
location2 = args.location2
mode_name = args.mode_name

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
min_freq_peak = args.min_freq_peak
max_freq_peak = args.max_freq_peak
phase_bin_width = args.phase_bin_width

# Print the inputs
print(f"Station: {station}")
print(f"Location 1: {location1}")
print(f"Location 2: {location2}")
print(f"Mode name: {mode_name}")
print(f"Window length: {window_length}")
print(f"Overlap: {overlap}")
print(f"Min prominence: {min_prom}")
print(f"Min reverse bandwidth: {min_rbw}")
print(f"Min frequency peak: {min_freq_peak}")
print(f"Max frequency peak: {max_freq_peak}")

# Read the peak properties
filename_in = f"stationary_resonance_properties_{mode_name}_hydro_window{window_length:.0f}s_overlap{overlap:.1f}_prom{min_prom:.0f}db_rbw{min_rbw:.1f}_max_mean{max_mean_db:.0f}db_freq{min_freq_peak:.0f}to{max_freq_peak:.0f}hz.h5"
filepath_in = join(indir, filename_in)

print(f"Reading the peak properties from {filepath_in}")
properties_df = read_hdf(filepath_in, key = "properties")

# Get the properties for the two locations
properties_sta_df = properties_df[properties_df["station"] == station]
properties_loc1_df = properties_sta_df[properties_sta_df["location"] == location1]
properties_loc2_df = properties_sta_df[properties_sta_df["location"] == location2]

# Compute the phase 
print("Merging the properties for the two locations")
properties_merged_df = properties_loc1_df.merge(properties_loc2_df, on = "time", suffixes = ("_loc1", "_loc2"), how = "inner")

# Compute the phase difference
print("Computing the phase differences")
properties_merged_df["phase_diff"] = properties_merged_df["phase_loc2"] - properties_merged_df["phase_loc1"]

# Convert the phase difference to the range 0 to 360 degrees
print("Converting the phase differences to the range 0 to 360 degrees")
properties_merged_df["phase_diff"] = properties_merged_df["phase_diff"].apply(lambda x: x + 360 if x < 0 else x)
properties_merged_df["phase_diff"] = properties_merged_df["phase_diff"].apply(lambda x: x - 360 if x >= 360 else x)

# Define the figure and axes
print("Plotting the histogram of the phase differences")
fig, ax = subplots(1, 1, figsize = (8, 6))

# Plot the histogram of the phase differences
bin_edges = arange(0, 360 + phase_bin_width, phase_bin_width)
properties_merged_df["phase_diff"].hist(bins = bin_edges, ax = ax, color = "tab:blue", edgecolor = "black")
ax.set_title(f"{mode_name}, {station}, {location2} - {location1}", fontsize = 12, fontweight = "bold")
ax.set_xlabel("Phase difference (degrees)")
ax.set_ylabel("Count")

# Save the histogram figure
figname = f"stationary_resonance_hydro_loc_diff_phases_histogram_{mode_name}_{station}_{location1}_{location2}.png"
save_figure(fig, figname)






