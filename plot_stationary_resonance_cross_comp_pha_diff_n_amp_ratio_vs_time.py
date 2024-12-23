# Plot the cross component phase difference and amplitude ratio of a stationary resonance at all stations as a function of time

### Import the required libraries ###
from os.path import join
from argparse import ArgumentParser
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, PHASE_DIFF_COMPONENT_PAIRS as comp_pairs
from utils_plot import plot_stationary_resonance_properties_vs_time, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description = "Plot the cross component phase difference and amplitude ratio of a stationary resonance at all stations as a function of time")
parser.add_argument("--mode_name", type = str, help = "Mode name")

# Constants
min_amp_rat = 0.0
max_amp_rat = 3.0

# Parse the command line arguments
args = parser.parse_args()
mode_name = args.mode_name

print(f"Plotting the cross component phase difference and amplitude ratio of {mode_name} at all stations as a function of time...")

### Read the data ###
print(f"Reading the properties of {mode_name}...")
filename = f"stationary_resonance_cross_comp_pha_diff_amp_rat_{mode_name}.h5"
inpath = join(indir, filename)
properties_df = read_hdf(inpath, key = "properties")

### Plot the data ###
print("Plotting the data...")
for i, pair in enumerate(comp_pairs):
    component1, component2 = pair
    pha_diff_name = f"phase_diff_{component1.lower()}_{component2.lower()}"
    amp_ratio_name = f"amp_ratio_{component1.lower()}_{component2.lower()}"

    # Plot the phase difference
    print(f"Plotting the phase difference of {component1} and {component2}...")
    title = f"Phase difference, {component1}-{component2}"
    fig, ax, cbar = plot_stationary_resonance_properties_vs_time(pha_diff_name, properties_df,
                                                                title = title)

    print(f"Saving the figure...")
    figname = f"stationary_resonance_pha_diff_{mode_name}_{component1.lower()}{component2.lower()}.png"
    save_figure(fig, figname)

    # Plot the amplitude ratio
    print(f"Plotting the amplitude ratio of {component1} and {component2}...")
    title = f"Amplitude ratio, {component1}-{component2}"
    fig, ax, cbar = plot_stationary_resonance_properties_vs_time(amp_ratio_name, properties_df,
                                                                min_amp_rat = min_amp_rat, max_amp_rat = max_amp_rat,
                                                                title = title)

    print(f"Saving the figure...")
    figname = f"stationary_resonance_amp_ratio_{mode_name}_{component1.lower()}{component2.lower()}.png"
    save_figure(fig, figname)