"""
Plot the histograms of the hammer power decay rates for the two subarrays
"""

###
# Imports
###

from os.path import join
from pandas import read_csv
from numpy import linspace
from matplotlib.pyplot import subplots
from argparse import ArgumentParser

from utils_basic import MT_DIR as dirpath_mt, GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_plot import save_figure

###
# Input arguments
###

parser = ArgumentParser()
parser.add_argument("--freq_target", type=float, help="The frequency to plot the histograms for", default=25.0)
parser.add_argument("--figwidth", type=float, help="The width of the figure", default=10)
parser.add_argument("--figheight", type=float, help="The height of the figure", default=5)
parser.add_argument("--color_subarray_a", type=str, help="The color for the subarray A", default="tab:blue")
parser.add_argument("--color_subarray_b", type=str, help="The color for the subarray B", default="tab:orange")
parser.add_argument("--num_bins", type=int, help="The number of bins for the histograms", default=20)
parser.add_argument("--min_rate", type=float, help="The minimum power decay rate to plot", default=0.0)
parser.add_argument("--max_rate", type=float, help="The maximum power decay rate to plot", default=1.2)
parser.add_argument("--num_bins_hist", type=int, help="The number of bins for the histograms", default=12)
args = parser.parse_args()

freq_target = args.freq_target
figwidth = args.figwidth
figheight = args.figheight
color_subarray_a = args.color_subarray_a
color_subarray_b = args.color_subarray_b
num_bins_hist = args.num_bins_hist
min_rate = args.min_rate
max_rate = args.max_rate
###
# Read the data
###

filename = f"hammer_power_decay_rate_{freq_target:.0f}hz.csv"
filepath = join(dirpath_mt, filename)
data_df = read_csv(filepath)
data_a_df = data_df[data_df["station"].isin(stations_a)]
data_b_df = data_df[data_df["station"].isin(stations_b)]

###
# Plot the histograms
###

fig, ax = subplots(1, 1, figsize = (figwidth, figheight))

bin_edges = linspace(min_rate, max_rate, num_bins_hist + 1)
ax.hist(data_a_df["power_decay_rate"].abs(), bins = bin_edges, color = color_subarray_a, label = "Subarray A", linewidth = 1.5, edgecolor = "black", alpha = 0.5)
ax.hist(data_b_df["power_decay_rate"].abs(), bins = bin_edges, color = color_subarray_b, label = "Subarray B", linewidth = 1.5, edgecolor = "black", alpha = 0.5)
ax.set_xlim(min_rate, max_rate)
ax.set_xlabel("Power decay rate (dB m$^{-1}$)")
ax.set_ylabel("Count")
ax.legend(fontsize = 12)
ax.set_title(f"Hammer power decay rate at {freq_target:.0f} Hz", fontsize = 14, fontweight = "bold")

filename_out = f"hammer_power_decay_rate_histograms_{freq_target:.0f}hz.png"
save_figure(fig, filename_out)