"""
Plot the stationary resonance powers recorded on hydrophones as a function of time
"""

###
# Import modules
###

import argparse
from os.path import join
from numpy import ones
from pandas import read_hdf
from matplotlib.pyplot import subplots
from matplotlib import colormaps
from matplotlib.colors import Normalize

from utils_basic import HYDRO_LOCATIONS as loc_dict, HYDRO_DEPTHS as depth_dict, SPECTROGRAM_DIR as dirpath, STARTTIME_HYDRO as starttime, ENDTIME_HYDRO as endtime
from utils_basic import get_mode_order
from utils_spec import get_spec_peak_file_suffix, get_spectrogram_file_suffix
from utils_plot import HYDRO_PSD_LABEL as ylabel, add_colorbar, format_datetime_xlabels, format_hydro_psd_ylabels, save_figure

###
# Input arguments
###

# Command line arguments
parser = argparse.ArgumentParser(description = "Plot the stationary resonance powers recorded on hydrophones as a function of time")

parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--window_length", type = float, default = 300.0, help = "Window length in seconds")
parser.add_argument("--overlap", type = float, default = 0.0, help = "Overlap in seconds")
parser.add_argument("--min_prom", type = float, default = 15.0, help = "Minimum prominence in dB")
parser.add_argument("--min_rbw", type = float, default = 15.0, help = "Minimum relative bandwidth")
parser.add_argument("--max_mean_db", type = float, default = -15.0, help = "Maximum mean dB for excluding noisy windows")

args = parser.parse_args()

mode_name = args.mode_name
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Constants
figwidth1 = 15.0
figheight1 = 10.0

figwidth2 = 15.0
figheight2 = 7.0

cmap_name = "viridis_r"

markersize = 5.0
linewidth = 1

min_depth = 30.0
max_depth = 380.0

min_power = -20.0
max_power = 15.0

major_time_tick_spacing = "30d"
num_minor_time_ticks = 7

colrbar_offset = 0.02
colorbar_width = 0.01

loc_label_offsets = (0.05, 0.98)

fontsize_title = 14

###
# Load data
###

print(f"Loading data for {mode_name}")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
filename = f"stationary_resonance_properties_hydro_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
inpath = join(dirpath, filename)

power_df = read_hdf(inpath, key = "properties")

###
# Plot the two stations in two subplots
###

mode_order = get_mode_order(mode_name)

cmap = colormaps[cmap_name]
norm = Normalize(vmin=min_depth, vmax=max_depth)


print(f"Plotting the two stations in two subplots...")
fig, axs = subplots(1, 2, figsize = (figwidth1, figheight1))
fig.subplots_adjust(wspace = 0.04)


for i, station in enumerate(loc_dict.keys()):
    ax = axs[i]
    ax.set_facecolor("lightgray")

    for location in loc_dict[station]:
        depth = depth_dict[location]
        
        print(f"Plotting {station}.{location}")
        power_loc_df = power_df[(power_df["station"] == station) & (power_df["location"] == location)]

        num_powers = len(power_loc_df)
        depths = depth * ones(num_powers)
        edge_colors = cmap(norm(depths))
        ax.scatter(power_loc_df["time"], power_loc_df["power"], 
                   s = markersize, marker = "o",
                   facecolors = 'none', edgecolors = edge_colors, 
                   linewidths = linewidth)

        if i == 0:
            format_hydro_psd_ylabels(ax)
        else:
            format_hydro_psd_ylabels(ax,
                                     plot_axis_label=False,
                                     plot_tick_label=False)

        format_datetime_xlabels(ax,
                                date_format = "%Y-%m-%d",
                                plot_tick_label = True,
                                major_tick_spacing = major_time_tick_spacing,
                                num_minor_ticks = num_minor_time_ticks,
                                va = "top", ha = "right",
                                rotation = 30)

        ax.set_xlim(starttime, endtime)
        ax.set_ylim(min_power, max_power)

    ax.set_title(f"{station}", fontsize = 14, fontweight = "bold")

bbox = ax.get_position()
position = [bbox.x1 + colrbar_offset, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, position, "Hydrophone depth (m)",
             cmap = cmap, norm = norm,
             orientation = "vertical")

figname = f"stationary_resonance_hydro_power_vs_time_{mode_name}_station_plots.png"
save_figure(fig, figname)

###
# Plot locations 03 to 06 in four subplots
###

print(f"Plotting locations 03 to 06 in four subplots...")
fig, axs = subplots(1, 4, figsize = (figwidth2, figheight2))

fig.subplots_adjust(hspace = 0.04)

for i, location in enumerate(["03", "04", "05", "06"]):
    ax = axs[i]

    print(f"Plotting {location}...")
    for station in loc_dict.keys():
        powers_df = power_df[(power_df["station"] == station) & (power_df["location"] == location)]
        
        if station == "A00":
            color = "darkviolet"
        elif station == "B00":
            color = "violet"

        if i == 0:
            ax.scatter(powers_df["time"], powers_df["power"], 
                       s = markersize, marker = "o",
                       facecolors = 'none', edgecolors = color, 
                       linewidths = linewidth, label = station)
            
            format_datetime_xlabels(ax,
                                    date_format = "%Y-%m-%d",
                                    major_tick_spacing = major_time_tick_spacing,
                                    num_minor_ticks = num_minor_time_ticks,
                                    va = "top", ha = "right",
                                    rotation = 30)
            
            format_hydro_psd_ylabels(ax)
            
            ax.legend(loc = "lower right", fontsize = 10, framealpha = 1.0)
        else:
            ax.scatter(powers_df["time"], powers_df["power"], 
                       s = markersize, marker = "o",
                       facecolors = 'none', edgecolors = color, 
                       linewidths = linewidth)
            
            format_datetime_xlabels(ax,
                                    date_format = "%Y-%m-%d",
                                    plot_tick_label = False,
                                    plot_axis_label = False,
                                    major_tick_spacing = major_time_tick_spacing,
                                    num_minor_ticks = num_minor_time_ticks,
                                    va = "top", ha = "right",
                                    rotation = 30)
            
            format_hydro_psd_ylabels(ax,
                                     plot_axis_label=False,
                                     plot_tick_label=False)

        ax.set_xlim(starttime, endtime)
        ax.set_ylim(min_power, max_power)
        ax.text(loc_label_offsets[0], loc_label_offsets[1], f"{location}", 
                fontsize = 12, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0),
                transform = ax.transAxes, va = "top", ha = "left")

fig.suptitle(f"Mode {mode_order:d}, hydrophones, power vs time", fontsize = fontsize_title, fontweight = "bold", y = 0.95)

figname = f"stationary_resonance_hydro_power_vs_time_{mode_name}_location_plots.png"
save_figure(fig, figname)






