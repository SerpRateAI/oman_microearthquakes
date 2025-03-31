"""
Plot the multitaper cross-spectra analysis results of a hammer shot between all pairs of adjacent hydrophone locations in a borehole
"""

###
# Imports
###
from os.path import join
from argparse import ArgumentParser
from numpy import abs, amax
from pandas import read_csv, Timedelta, Timestamp
from matplotlib.pyplot import figure, subplots
from matplotlib.gridspec import GridSpec

from utils_basic import MT_DIR as dirpath_mt, LOC_DIR as dirpath_loc, HYDRO_LOCATIONS as loc_dict
from utils_basic import power2db
from utils_preproc import read_and_process_windowed_hydro_waveforms
from utils_plot import component2label, format_phase_diff_ylabels, format_norm_amp_ylabels, format_coherence_ylabels, format_norm_psd_ylabels, format_freq_xlabels, get_geo_component_color, save_figure

###
# Inputs
###

# Command line arguments
parser = ArgumentParser(description = "Plot the multitaper cross-spectra analysis results of a hammer shot between all pairs of adjacent hydrophone locations in a borehole")
parser.add_argument("--hammer_id", type = str, help = "Hammer ID")
parser.add_argument("--window_length", type = float, default = 1.0, help = "Window length in seconds")
parser.add_argument("--freq_target", type = float, default = 25.0, help = "Frequency target in Hz")

# Parse the command line arguments
args = parser.parse_args()
hammer_id = args.hammer_id
window_length = args.window_length
freq_target = args.freq_target

# Constants
min_freq = 0.0
max_freq = 100.0

min_db = -40.0
max_db = -10.0

figwidth = 15
figheight = 12

margin_x = 0.05
margin_y = 0.05

wspace = 0.02

hspace_major = 0.05
hspace_minor = 0.025

major_freq_tick_spacing = 25.0
num_minor_freq_ticks = 5
linewidth = 2.0

axis_label_size = 12
title_size = 14

###
# Read the input files
###

# Read the waveforms
print(f"Reading the waveforms...")
inpath = join(dirpath_loc, "hammer_locations.csv")
hammer_df = read_csv(inpath, dtype={"hammer_id": str}, parse_dates=["origin_time"])
otime = hammer_df[ hammer_df["hammer_id"] == hammer_id ]["origin_time"].values[0]
otime = Timestamp(otime)

starttime = otime
endtime = otime + Timedelta(seconds = window_length)

stream = read_and_process_windowed_hydro_waveforms(starttime, endtime = endtime) 

###
# Plot the cross-spectra
###

### Compute the subplot dimensions ###
subplot_height = (1.0 - 2 * margin_y - hspace_major - 2 * hspace_minor) / 4
subplot_width = (1.0 - 2 * margin_x - 2 *wspace)

### Iterate over the location pairs ###
for station in loc_dict.keys():
    print(f"Plotting the results for station {station}...")
    locations = loc_dict[station]
    num_loc = len(locations)

    for i in range(num_loc - 1):
        location1 = locations[i]
        location2 = locations[i + 1]

        print(f"Plotting the results for {station} and {location1} and {location2}...")

        # Read the cross-spectral analysis results
        filename = f"hammer_mt_inter_hydro_loc_phase_diffs_{hammer_id}_{station}_{location1}_{location2}.csv"
        inpath = join(dirpath_mt, filename)
        cspec_df = read_csv(inpath)
        freqax = cspec_df["frequency"].values

        # Initialize the figure
        fig = figure(figsize = (figwidth, figheight))

        # Plot the waveforms
        left = margin_x
        bottom = margin_y + 3 * subplot_height + 2 * hspace_minor + hspace_major
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        trace1 = stream.select(station = station, location = location1)[0]
        trace2 = stream.select(station = station, location = location2)[0]

        signal1 = trace1.data   
        signal2 = trace2.data

        signal1 = signal1 / amax(abs(signal1))
        signal2 = signal2 / amax(abs(signal2))

        timeax = trace1.times()
        ax.plot(timeax, signal1, color = "mediumpurple", linewidth = linewidth, label = location1)
        ax.plot(timeax, signal2, color = "gold", linewidth = linewidth, label = location2)

        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_xlabel("Time (s)", fontsize = axis_label_size)

        format_norm_amp_ylabels(ax,
                                axis_label_size = axis_label_size)
        
        ax.legend(fontsize = axis_label_size, loc = "upper right")

        # Plot the autospectra
        left = margin_x
        bottom = margin_y + 2 * subplot_height + 2 * hspace_minor
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        aspec1 = cspec_df[f"aspec_loc1"].values
        aspec2 = cspec_df[f"aspec_loc2"].values

        aspec1_db = power2db(aspec1)
        aspec2_db = power2db(aspec2)

        ax.plot(freqax, aspec1_db, color = "mediumpurple", linewidth = linewidth, label = location1)
        ax.plot(freqax, aspec2_db, color = "gold", linewidth = linewidth, label = location2)

        ax.axvline(freq_target, color = "crimson", linewidth = linewidth)

        ax.set_xlim(min_freq, max_freq)
        ax.set_ylim(min_db, max_db)

        ax.set_xticks([])

        format_norm_psd_ylabels(ax,
                                axis_label_size = axis_label_size)
            
        format_freq_xlabels(ax,
                            major_tick_spacing = major_freq_tick_spacing,
                            num_minor_ticks = num_minor_freq_ticks,
                            plot_axis_label = False,
                            plot_tick_label = False,
                            axis_label_size = axis_label_size)

        # Plot the coherence
        left = margin_x
        bottom = margin_y + subplot_height + hspace_minor
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        coherences = cspec_df[f"cohe"].values
        ax.plot(freqax, coherences, color = "black", linewidth = linewidth)

        ax.axvline(freq_target, color = "crimson", linewidth = linewidth)

        ax.set_xlim(min_freq, max_freq)
        ax.set_ylim(0.0, 1.0)

        ax.set_xticks([])

        format_coherence_ylabels(ax,
                                 axis_label_size = axis_label_size)

        format_freq_xlabels(ax,
                            major_tick_spacing = major_freq_tick_spacing,
                            num_minor_ticks = num_minor_freq_ticks,
                            plot_axis_label = False,
                            plot_tick_label = False,
                            axis_label_size = axis_label_size)

        # Plot the phase differences and the uncertainties
        phase_diffs = cspec_df[f"phase_diff"].values
        phase_diff_uncers = cspec_df[f"phase_diff_uncer"].values

        left = margin_x
        bottom = margin_y
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        ax.plot(freqax, phase_diffs, color = "black", linewidth = linewidth)
        ax.fill_between(freqax, phase_diffs - phase_diff_uncers, phase_diffs + phase_diff_uncers, color = "black", alpha = 0.2)

        ax.axvline(freq_target, color = "crimson", linewidth = linewidth)

        ax.set_xlim(min_freq, max_freq)

        format_phase_diff_ylabels(ax,
                                 axis_label_size = axis_label_size)

        # Set the x-axis label
        format_freq_xlabels(ax,
                            major_tick_spacing = 25.0,
                            num_minor_ticks = 5,
                            plot_axis_label = True,
                            plot_tick_label = True,
                            axis_label_size = axis_label_size)

        # Set the suptitle
        fig.suptitle(f"Hammer {hammer_id}, {station}, {location1}-{location2}", fontsize = title_size, fontweight = "bold", y = 0.98)

        # Save the figure
        figname = f"hammer_mt_inter_hydro_loc_cspec_{hammer_id}_{station}_{location1}_{location2}.png"
        save_figure(fig, figname)

        print("Done!")
        print("")

    