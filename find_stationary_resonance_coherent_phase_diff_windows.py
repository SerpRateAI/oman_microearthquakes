# Find the time windows in which a stationary resonance has coherent phase differences across a specific set of stations

### Import the required libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import arctan2, cos, sin, deg2rad, rad2deg, sqrt
from pandas import DataFrame
from pandas import read_hdf
from json import loads
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, GEO_COMPONENTS, INNER_STATIONS_B
from utils_plot import save_figure

# Get the coherence of a list of phase differences
def get_phase_coherence(phase_diffs):
    # Get the number of phase differences
    n = len(phase_diffs)
    
    # Get the sum of the cosines and sines of the phase differences
    sum_cos = sum([cos(phase_diff) for phase_diff in phase_diffs])
    sum_sin = sum([sin(phase_diff) for phase_diff in phase_diffs])
    
    # Get the average cosine and sine of the phase differences
    avg_cos = sum_cos / n
    avg_sin = sum_sin / n
    
    # Get the average phase difference
    avg_phase_diff = rad2deg(arctan2(avg_sin, avg_cos))

    # Get the coherence
    coherence = sqrt(avg_cos ** 2 + avg_sin ** 2)
    
    return coherence, avg_phase_diff


### Inputs ###
# Command line arguments
parser = ArgumentParser(description = "Find the time windows in which a stationary resonance has coherent phase differences across a specific set of stations")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--components", type = str, default = GEO_COMPONENTS, help = "Components to consider")
parser.add_argument("--stations", type = str, default = INNER_STATIONS_B, help = "Stations to consider")
parser.add_argument("--min_num_stations", type = int, default = 5, help = "Minimum number of stations")
parser.add_argument("--min_coherence", type = float, default = 0.8, help = "Minimum coherence")
parser.add_argument("--max_count", type = int, help = "Maximum number of time windows")

# Parse the command line arguments
args = parser.parse_args()
mode_name = args.mode_name

components = args.components
if not isinstance(components, list):
    components = loads(components)

stations = args.stations
if not isinstance(stations, list):
    stations = loads(stations)

min_num_stations = args.min_num_stations
min_coherence = args.min_coherence

max_count = args.max_count

# Constants
num_bin = 36 # Number of bins for the phase difference histogram
figwidth = 12 # Width of the figure
figheight = 5 # Height of the figure

# Print the inputs
print(f"Finding the time windows in which {mode_name} has coherent phase differences across {components} at {stations}...")
print(f"Minimum coherence: {min_coherence}")


### Read the phase differences ###
print(f"Reading the phase differences of {mode_name}...")
filename = f"stationary_resonance_cross_comp_pha_diff_amp_rat_{mode_name}.h5"
inpath = join(indir, filename)
properties_df = read_hdf(inpath, key = "properties")

# Keep only the specified stations
properties_df = properties_df.loc[properties_df["station"].isin(stations)]

### Process the data ###
# Group the data by time
time_groups = properties_df.groupby("time")

# Get the component pairs
if "Z" in components:
    component_pairs = [("1", "2"), ("1", "Z")]
else:
    component_pairs = [("1", "2")]

# Initialize the dictionary of coherent time windows
output_dict = {}
output_dict["time"] = []

# Initialize the list of coherences and average phase differences
for component1, component2 in component_pairs:
    output_dict[f"coherence_{component1.lower()}_{component2.lower()}"] = []
    output_dict[f"avg_phase_diff_{component1.lower()}_{component2.lower()}"] = []

# Loop over the time groups
for time, group in time_groups:
    # Get the number of stations
    num_stations = len(group["station"].unique())
    if num_stations < min_num_stations:
        continue

    # Get the phase differences for each component pair
    cohenrence_dict = {}
    for component1, component2 in component_pairs:
        pha_diff_name = f"phase_diff_{component1.lower()}_{component2.lower()}"
        phase_diffs = deg2rad(group[pha_diff_name].values)
        
        # Get the coherence of the phase differences
        coherence, avg_phase_diff = get_phase_coherence(phase_diffs)
        cohenrence_dict[(component1, component2)] = (coherence, avg_phase_diff)

    # Determine if all component pairs have a coherence greater than the minimum coherence
    if all([coherence >= min_coherence for coherence in [cohenrence_dict[component_pair][0] for component_pair in component_pairs]]):
        # Add the time to the output 
        output_dict["time"].append(time)

        # Add the coherences and average phase differences to the output
        for component1, component2 in component_pairs:
            coherence, avg_phase_diff = cohenrence_dict[(component1, component2)]
            output_dict[f"coherence_{component1.lower()}_{component2.lower()}"].append(coherence)
            output_dict[f"avg_phase_diff_{component1.lower()}_{component2.lower()}"].append(avg_phase_diff)

# Convert the output dictionary to a DataFrame
output_df = DataFrame(output_dict)
print(f"Found {len(output_df)} time windows in which {mode_name} has coherent phase differences across {components} at {stations}")

# Plot the rose diagram of the phase differences
num_plots = len(component_pairs)
fig, axes = subplots(1, num_plots, subplot_kw={'projection': 'polar'}, figsize = (figwidth, figheight))

if num_plots == 1:
    ax = axes

    component1, component2 = component_pairs[0]
    pha_diff_name = f"phase_diff_{component1.lower()}_{component2.lower()}"
    angles_rad = deg2rad(output_df[f"avg_phase_diff_{component1.lower()}_{component2.lower()}"].values)

    # Plot the rose diagram
    num_bins = 36
    ax.hist(angles_rad, bins = num_bins, color = "tab:blue", edgecolor = "black", linewidth = 1.0)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_rmax(max_count)
    ax.set_title(f"{component1}-{component2}", fontsize = 12, fontweight = "bold")
else:
    for i, component_pair in enumerate(component_pairs):
        component1, component2 = component_pair
        pha_diff_name = f"phase_diff_{component1.lower()}_{component2.lower()}"
        angles_rad = deg2rad(output_df[f"avg_phase_diff_{component1.lower()}_{component2.lower()}"].values)

        # Plot the rose diagram
        ax = axes[i]
        num_bins = 36
        ax.hist(angles_rad, bins = num_bins, color = "tab:blue", edgecolor = "black", linewidth = 1.0)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.set_rmax(max_count)
        ax.set_title(f"{component1}-{component2}", fontsize = 12, fontweight = "bold")

# Save the figure
filename = f"stationary_resonance_average_phase_diffs_{mode_name}.png"
save_figure(fig, filename)

### Save the output ###
print("Saving the output...")
filename = f"stationary_resonance_coherent_phase_diff_windows_{mode_name}.h5"
outpath = join(indir, filename)
output_df.to_hdf(outpath, key = "windows", mode = "w")
print(f"Saved the coherences to {outpath}")

filename = f"stationary_resonance_coherent_phase_diff_windows_{mode_name}.csv"
outpath = join(indir, filename)
output_df.to_csv(outpath)
print(f"Saved the coherences to {outpath}")





