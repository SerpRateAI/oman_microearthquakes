# Plot the stacked Fourier beams

# Import necessary modules
from os.path import join
from argparse import ArgumentParser

from utils_basic import BEAM_DIR as indir
from utils_array import read_fourier_beam_collection
from utils_plot import plot_beam_images, save_figure

# Inputs
parser = ArgumentParser(description="Input parameters for plotting the stacked Fourier beams.")
parser.add_argument("--mode_name", type=str, help="Name of the mode.")
parser.add_argument("--array_selection", type=str, help="Selection of the subarray: A or B.")
parser.add_argument("--array_aperture", type=str, help="Aperture of the array: small, medium, or large.")

parser.add_argument("--min_coherence", type=float, default=0.8, help="Minimum coherence for a beam image to be considered.")

args = parser.parse_args()
mode_name = args.mode_name
array_selection = args.array_selection
array_aperture = args.array_aperture

min_coherence = args.min_coherence

print(f"Plotting the stacked Fourier beams of mode {mode_name}")
print(f"Selection of the subarray: {array_selection}.")
print(f"Aperture of the array: {array_aperture}.")

print(f"Minimum coherence: {min_coherence:.1f}.")


# Read the Fourier beam collection
print("Reading the Fourier beam collection...")
array_suffix = f"array_{array_selection.lower()}_{array_aperture}"

filename = f"fourier_beam_image_collection_{mode_name}_{array_suffix}.h5"
inpath = join(indir, filename)
beam_collection = read_fourier_beam_collection(inpath)
num_windows = beam_collection.num_windows
print(f"Number of input time windows: {num_windows:d}")

# Compute the stacked Fourier beams
print("Computing the stacked Fourier beams...")
xslowax, yslowax, bimage_stack_dict, num_stack = beam_collection.get_stacked_beam_images(min_coherence = min_coherence)

# Plot the stacked Fourier beams
print("Plotting the stacked Fourier beams...")
fig,axes = plot_beam_images(xslowax, yslowax, bimage_stack_dict,
                            plot_global_max = True, plot_local_max = False)

# Set the title
title = f"Stacked Fourier beams, {mode_name}, Array {array_selection}, {array_aperture} aperture, coherence > {min_coherence:.1f}, {num_stack:d} windows"
fig.suptitle(title, fontsize = 14, fontweight = "bold", y = 0.94)

# Save the figure
print("Saving the figure...")
figname = f"stacked_fourier_beams_{mode_name}_{array_suffix}_min_coherence_{min_coherence:.1f}.png"
save_figure(fig, figname)




