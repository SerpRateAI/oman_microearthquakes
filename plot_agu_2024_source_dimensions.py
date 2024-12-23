'''
Plot source dimensions as functions of gas fraction and source depths as predicted by the vibrating-crack model.
'''

### Import the required libraries ###
from os.path import join
from json import loads
from argparse import ArgumentParser
from pandas import read_csv
from matplotlib.pyplot import subplots
from matplotlib import colormaps

from utils_basic import PHYS_DIR as dirname
from utils_plot import save_figure

### Input parameters ###
# Command line arguments
parser = ArgumentParser(description="Plot source dimensions as functions of gas fraction and source depths as predicted by the vibrating-crack model.")
parser.add_argument("--depths", type=str, help="Source depths")
parser.add_argument("--freq", type=float, default = 12.0, help="Fundamental frequency of the source")

# Parse the command line arguments
args = parser.parse_args()
depths = loads(args.depths)
freq = args.freq

# Constants
figwidth = 7.0
figheight = 5.0

legend_fontsize = 10
title_fontsize = 12

linewidth = 1.5

# Starting indices for the colors of the velocities and dimensions in the colormap tab20c
i_color_vel = 0 
i_color_dim = 8 

### Plotting ###
# Create the subplots
fig, axes = subplots(2, 1, figsize=(figwidth, figheight), sharex=False, sharey=False, gridspec_kw={"hspace": 0.4})

# Plot the velocity as a function of gas fraction and source depth
ax_vel = axes[0]
ax_dim = axes[1]

cmap = colormaps["tab20c"]

for i, depth in enumerate(depths):
    # Load the data
    filename = f"source_dimensions_{depth:.0f}m.csv"
    filepath = join(dirname, filename)

    data_df = read_csv(filepath)

    # Plot the velocities
    color = cmap(i_color_vel + i)
    ax_vel.plot(data_df["gas_frac"], data_df["velocity"], label=f"{depth:.0f} m", linewidth=linewidth, color=color)

    # Plot the dimensions
    color = cmap(i_color_dim + i)
    ax_dim.plot(data_df["gas_frac"], data_df["dimension"], label=f"{depth:.0f} m", linewidth=linewidth, color=color)

# Set the axis limits
ax_vel.set_yscale('log')
ax_dim.set_yscale('log')

# Set the axis labels
ax_vel.set_ylabel("Velocity (m s$^{-1}$)")
ax_dim.set_ylabel("Length (m)")

# Set the x-axis label
ax_dim.set_xlabel("Gas volume fraction")

# Add the legends
legend = ax_vel.legend(fontsize=legend_fontsize, loc="upper center", ncol = len(depths), title="Burial depth")
legend.get_title().set_fontweight("bold")

legend = ax_dim.legend(fontsize=legend_fontsize, loc="upper center", ncol = len(depths), title="Burial depth")
legend.get_title().set_fontweight("bold")

# Add the titles
ax_vel.set_title("Sound speed of hydrogen-water mixture", fontsize=title_fontsize, fontweight="bold")
ax_dim.set_title(f"Source dimension with fundamental frequency = {freq:.0f} Hz", fontsize=title_fontsize, fontweight="bold")

# Save the figure
save_figure(fig, "agu_2024_source_dimensions.png")