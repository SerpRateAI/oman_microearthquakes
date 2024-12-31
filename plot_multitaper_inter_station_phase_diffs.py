"""
Plot the phase differences between the three components of a pair of stations measured using the multitaper method
"""

### Imports ###
from os.path import join
from json import loads
from argparse import ArgumentParser
from numpy import pi
from pandas import read_csv
from matplotlib.pyplot import subplots

from utils_basic import GEO_COMPONENTS as components, MT_DIR as dirname_mt, SPECTROGRAM_DIR as dirname_spec, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_plot import component2label, format_datetime_xlabels, format_phase_diff_ylabels, get_geo_component_color, save_figure

### Input parameters ###
# Command line arguments
parser = ArgumentParser(description="Plot the inter-station 3C phase differences between a pair of geophone stations")
parser.add_argument("--station1", type=str, help="Station 1")
parser.add_argument("--station2", type=str, help="Station 2")

parser.add_argument("--mode_name", type=str, help="Mode name", default="PR02549")

# Parse the command line arguments
args = parser.parse_args()
station1 = args.station1
station2 = args.station2

mode_name = args.mode_name

# Constants
linewidth = 0.5
markersize = 2.0
fontsize_legend = 10.0
fontsize_title = 12.0

major_date_tick_spacing = "5d"
num_minor_date_ticks = 5

### Read the phase differences ###
filename = f"multitaper_inter_sta_phase_diffs_{mode_name}_{station1}_{station2}.csv"
filepath = join(dirname_mt, filename)
phase_diff_df = read_csv(filepath, parse_dates=["time"])

fig, ax = subplots(1, 1, figsize=(6, 3))
for component in components:
    # Get the color of the component
    color = get_geo_component_color(component)

    # Plot the data
    ax.errorbar(phase_diff_df["time"], phase_diff_df[f"phase_diff_{component.lower()}"], yerr = phase_diff_df[f"phase_diff_uncer_{component.lower()}"], 
                               fmt = "o", markerfacecolor="none", markeredgecolor=color, label = component2label(component), markersize = markersize,
                               markeredgewidth = linewidth, elinewidth = linewidth, capsize=2, zorder=2)

# Set the axis limits
ax.set_xlim(starttime, endtime)
ax.set_ylim(-pi, pi)

# Set the axis labels
format_datetime_xlabels(ax,
                        plot_axis_label=True, plot_tick_label=True,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing = major_date_tick_spacing, num_minor_ticks=num_minor_date_ticks)
    
# Set the y-axis labels
format_phase_diff_ylabels(ax,
                          plot_axis_label=True, plot_tick_label=True)

# Add the legend
ax.legend(loc="lower right", fontsize=fontsize_legend, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)

# Add the station names
ax.set_title(f"{station1}-{station2}", fontsize = fontsize_title, fontweight = "bold")

### Save the figure ###
figname = f"multitaper_inter_sta_phase_diffs_{mode_name}_{station1}_{station2}.png"
save_figure(fig, figname)
