"""
Plot the whole-day spectrograms for a few example geophone stations
"""

from os.path import join
from argparse import ArgumentParser
from json import loads
from pandas import Timedelta
from pandas import read_csv
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle
from utils_basic import SPECTROGRAM_DIR as dirname_spec
from utils_basic import time2suffix, str2timestamp
from utils_spec import StreamSTFT
from utils_spec import get_spectrogram_file_suffix, read_geo_stft, string_to_time_label
from utils_plot import GEO_PSD_LABEL
from utils_plot import format_datetime_xlabels, format_freq_ylabels, save_figure

###
# Inputs
###

# Command-line arguments
parser = ArgumentParser(description="Plot the whole-day spectrograms for a few example geophone stations")

parser.add_argument("--stations", type=str, nargs="+", help="Station list")
parser.add_argument("--day", type=str, help="Day of the data to plot")
parser.add_argument("--time_to_mark", type=str, help="Time window to mark")
parser.add_argument("--min_db", type=float, help="Minimum decibel value", default=-10.0)
parser.add_argument("--max_db", type=float, help="Maximum decibel value", default=10.0)

parser.add_argument("--window_length", type=float, help="Window length in seconds", default=300.0)
parser.add_argument("--overlap", type=float, help="Overlap percentage", default=0.0)

# Parse the command line inputs
args = parser.parse_args()

stations = args.stations
day = args.day
time_to_mark = str2timestamp(args.time_to_mark) 
min_db = args.min_db
max_db = args.max_db
window_length = args.window_length
overlap = args.overlap

# Constants
min_freq = 0.0
max_freq = 200.0

figwidth = 10.0
panelheight = 5.0

hspace = 0.05

major_freq_tick_spacing = 50.0
num_minor_freq_ticks = 5

major_datetime_tick_spacing = "6h"
num_minor_datetime_ticks = 6

ax_label_size = 12
tick_label_size = 10
sta_label_x = 0.01
sta_label_y = 0.97
sta_label_size = 12

time_mark_label = "Fig. 1c"

harmo_arrow_length = Timedelta(hours = 1)
time_arrow_length = 15.0

mark_label_size = 12

cax_width = 0.01
cax_height = 0.3

cax_offset_x = 0.02
cax_offset_y = 0.1
cax_pad1_x = 0.05
cax_pad1_y = 0.1

###
# Load the data
###

# Spectrograms
stream_all = StreamSTFT()

for station in stations:
    print(f"Reading the PSD of {station}...")
    suffix = get_spectrogram_file_suffix(window_length, overlap)
    filename = f"whole_deployment_daily_geo_stft_{station}_{suffix}.h5"

    inpath = join(dirname_spec, filename)
    time_label = string_to_time_label(day)
    stream_psd = read_geo_stft(inpath, 
                               time_labels = time_label, min_freq = min_freq, max_freq = max_freq, psd_only = True)

    trace_psd = stream_psd.get_total_psd()
    stream_all.append(trace_psd)

print("Done!")

# Harmonic series
filename = f"stationary_harmonic_series_PR02549_base2.csv"
inpath = join(dirname_spec, filename)

harmonic_df = read_csv(inpath)

###
# Plot the data
###

num_plots = len(stations)

fig, axs = subplots(num_plots, 1, figsize = (figwidth, num_plots * panelheight), sharex = True)
fig.subplots_adjust(hspace = hspace)

for i_sta, station in enumerate(stations):
    ax = axs[i_sta]
    trace_psd = stream_all.traces[i_sta]
    trace_psd.to_db()

    psd_mat = trace_psd.psd_mat
    time_ax = trace_psd.times
    freq_ax = trace_psd.freqs

    mappable = ax.pcolormesh(time_ax, freq_ax, psd_mat, cmap = "inferno", vmin = min_db, vmax = max_db)

    ax.set_xlim(time_ax[0], time_ax[-1])
    ax.set_ylim(freq_ax[0], freq_ax[-1])

    ax.text(sta_label_x, sta_label_y, station, 
            transform = ax.transAxes, 
            fontsize = sta_label_size, fontweight = "bold",
            ha = "left", va = "top",
            bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))

    format_freq_ylabels(ax,
                        major_tick_spacing = major_freq_tick_spacing,
                        num_minor_ticks = num_minor_freq_ticks)

    # Plot the arrow marking the time window to mark
    if i_sta == 0:
        ax.annotate(f"{time_mark_label}", xy = (time_to_mark, freq_ax[-1]), xytext = (time_to_mark, freq_ax[-1] + time_arrow_length),
                    arrowprops = dict(arrowstyle = "->", color = "black"),
                    fontsize = mark_label_size, fontweight = "bold",
                    ha = "center", va = "bottom")

    # Plot the arrow marking the frequency of the harmonic series
    for i_mode, row in harmonic_df.iterrows():
        mode_name = row["mode_name"]
        mode_order = row["mode_order"]
        freq = row["observed_freq"]

        if mode_name.startswith("PR"):
            if i_sta == 0:
                if i_mode == 1:
                    ax.annotate(f"Mode {mode_order}", xy = (time_ax[-1], freq), xytext = (time_ax[-1] + harmo_arrow_length, freq),
                            arrowprops = dict(arrowstyle = "->", color = "black"),
                            fontsize = mark_label_size, fontweight = "bold",
                            ha = "left", va = "center")
                else:
                    ax.annotate(f"{mode_order}", xy = (time_ax[-1], freq), xytext = (time_ax[-1] + harmo_arrow_length, freq),
                                arrowprops = dict(arrowstyle = "->", color = "black"),
                                fontsize = mark_label_size, fontweight = "bold",
                                ha = "left", va = "center")
            else:
                ax.annotate("", xy = (time_ax[-1], freq), xytext = (time_ax[-1] + harmo_arrow_length, freq),
                            arrowprops = dict(arrowstyle = "->", color = "black"),
                            fontsize = mark_label_size, fontweight = "bold",
                            ha = "left", va = "center")            

    if i_sta == num_plots - 1:
        format_datetime_xlabels(ax,
                               major_tick_spacing = major_datetime_tick_spacing,
                               num_minor_ticks = num_minor_datetime_ticks)

        # Add the colorbar with a frame
        bbox = ax.get_position()


        ax.add_patch(Rectangle((0.0, 0.0), cax_width + 2 * cax_pad1_x, cax_height + 2 * cax_pad1_y, transform=ax.transAxes, 
                            facecolor='white', edgecolor='black'))

        cax = ax.inset_axes([cax_offset_x, cax_offset_y, cax_width, cax_height])
        cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize = tick_label_size)
        cbar.ax.set_ylabel(GEO_PSD_LABEL, fontsize = tick_label_size, ha = "center", va = "top")
        # cbar.ax.xaxis.set_label_coords(0.0, cax_label_offset)

###
# Save the figure
###
figname = f"whole_day_spec_examples_{day}.png"
save_figure(fig, figname, dpi = 600)










