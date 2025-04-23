"""
Plot the 3C multitaper inter-station cross-spectra analysis between all station pairs for an earthquake
"""

###
# Imports
###
from os.path import join
from argparse import ArgumentParser
from numpy import abs, amax
from pandas import read_csv, Timestamp
from matplotlib.pyplot import figure, subplots
from matplotlib.gridspec import GridSpec

from utils_basic import MT_DIR as dirpath_mt, GEO_COMPONENTS as components, LOC_DIR as dirpath_loc
from utils_basic import power2db, str2timestamp
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_plot import component2label, format_phase_diff_ylabels, format_norm_amp_ylabels, format_coherence_ylabels, format_norm_psd_ylabels, format_freq_xlabels, get_geo_component_color, save_figure

###
# Inputs
###

# Command line arguments
parser = ArgumentParser(description = "Plot the 3C multitaper inter-station cross-spectra analysis between all station pairs for an earthquake")
parser.add_argument("--earthquake_id", type = int, help = "Earthquake ID")
parser.add_argument("--freq_target", type = float, default = 25.5, help = "Frequency target in Hz")

# Parse the command line arguments
args = parser.parse_args()
earthquake_id = args.earthquake_id
freq_target = args.freq_target

# Constants
min_freq = 0.0
max_freq = 100.0

min_db = -50.0
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
title_size = 12
suptitle_size = 14

###
# Read the input files
###

# Read the window list
print(f"Reading the window list...")
inpath = join(dirpath_loc, f"earthquakes.csv")
earthquake_df = read_csv(inpath, parse_dates = ["start_time", "end_time"])

# Read the station pairs
print(f"Reading the station-pair list...")
inpath = join(dirpath_mt, "delaunay_station_pairs.csv")
pair_df = read_csv(inpath)

# Read the waveform
print(f"Reading the waveforms...")
start_time = Timestamp(earthquake_df.loc[earthquake_df["earthquake_id"] == earthquake_id, "start_time"].values[0])
end_time = Timestamp(earthquake_df.loc[earthquake_df["earthquake_id"] == earthquake_id, "end_time"].values[0])
stream = read_and_process_windowed_geo_waveforms(start_time, endtime = end_time)

###
# Plot the cross-spectra
###

### Compute the subplot dimensions ###
subplot_height = (1.0 - 2 * margin_y - hspace_major - 2 * hspace_minor) / 4
subplot_width = (1.0 - 2 * margin_x - 2 *wspace) / 3

### Iterate over the station pairs ###
for _, row in pair_df.iterrows():
    station1 = row["station1"]
    station2 = row["station2"]
    print(f"Plotting the results for {station1} and {station2}...")

    # Read the cross-spectral analysis results
    filename = f"earthquake_mt_inter_geo_sta_phase_diffs_eq{earthquake_id}_{station1}_{station2}.csv"
    try:
        inpath = join(dirpath_mt, filename)
        cspec_df = read_csv(inpath)
    except FileNotFoundError:
        print(f"The file {inpath} does not exist. Skipping the station pair {station1} and {station2}...")
        continue

    freqax = cspec_df["frequency"].values

    # Initialize the figure
    fig = figure(figsize = (figwidth, figheight))

    # Plot each component
    for i, component in enumerate(components):
        print(f"Plotting component {component}...")

        # Plot the waveforms
        left = margin_x + i * (subplot_width + wspace)
        bottom = margin_y + 3 * subplot_height + 2 * hspace_minor + hspace_major
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        trace1 = stream.select(station = station1, component = component)[0]
        trace2 = stream.select(station = station2, component = component)[0]

        signal1 = trace1.data   
        signal2 = trace2.data

        signal1 = signal1 / amax(abs(signal1))
        signal2 = signal2 / amax(abs(signal2))

        timeax = trace1.times()
        ax.plot(timeax, signal1, color = "mediumpurple", linewidth = linewidth, label = station1)
        ax.plot(timeax, signal2, color = "gold", linewidth = linewidth, label = station2)

        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_xlabel("Time (s)", fontsize = axis_label_size)

        if i == 0:
            format_norm_amp_ylabels(ax,
                                    plot_axis_label = True,
                                    plot_tick_label = True,
                                    axis_label_size = axis_label_size)
        else:
            format_norm_amp_ylabels(ax,
                                    plot_axis_label = False,
                                    plot_tick_label = False,
                                    axis_label_size = axis_label_size)
        if i == 0:
            ax.legend(fontsize = axis_label_size, loc = "upper right")

        ax.set_title(component2label(component), fontsize = title_size, fontweight = "bold")

        # Plot the autospectra
        left = margin_x + i * (subplot_width + wspace)
        bottom = margin_y + 2 * subplot_height + 2 * hspace_minor
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        aspec1 = cspec_df[f"aspec_sta1_{component.lower()}"].values
        aspec2 = cspec_df[f"aspec_sta2_{component.lower()}"].values

        aspec1_db = power2db(aspec1)
        aspec2_db = power2db(aspec2)

        ax.plot(freqax, aspec1_db, color = "mediumpurple", linewidth = linewidth, label = station1)
        ax.plot(freqax, aspec2_db, color = "gold", linewidth = linewidth, label = station2)

        ax.axvline(freq_target, color = "crimson", linewidth = linewidth)

        ax.set_xlim(min_freq, max_freq)
        ax.set_ylim(min_db, max_db)

        ax.set_xticks([])

        if i == 0:
            format_norm_psd_ylabels(ax,
                                    plot_axis_label = True,
                                    plot_tick_label = True,
                                    axis_label_size = axis_label_size)
        else:
            format_norm_psd_ylabels(ax,
                                    plot_axis_label = False,
                                    plot_tick_label = False,
                                    axis_label_size = axis_label_size)
            
        format_freq_xlabels(ax,
                            major_tick_spacing = major_freq_tick_spacing,
                            num_minor_ticks = num_minor_freq_ticks,
                            plot_axis_label = False,
                            plot_tick_label = False,
                            axis_label_size = axis_label_size)

        # Plot the coherence
        left = margin_x + i * (subplot_width + wspace)
        bottom = margin_y + subplot_height + hspace_minor
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        color_comp = get_geo_component_color(component)
        coherences = cspec_df[f"cohe_{component.lower()}"].values
        ax.plot(freqax, coherences, color = color_comp, linewidth = linewidth)

        ax.axvline(freq_target, color = "crimson", linewidth = linewidth)

        ax.set_xlim(min_freq, max_freq)
        ax.set_ylim(0.0, 1.0)

        ax.set_xticks([])

        if i == 0:
            format_coherence_ylabels(ax,
                                     plot_axis_label = True,
                                     plot_tick_label = True,
                                     axis_label_size = axis_label_size)
        else:
            format_coherence_ylabels(ax,
                                     plot_axis_label = False,
                                     plot_tick_label = False,
                                     axis_label_size = axis_label_size)

        format_freq_xlabels(ax,
                            major_tick_spacing = major_freq_tick_spacing,
                            num_minor_ticks = num_minor_freq_ticks,
                            plot_axis_label = False,
                            plot_tick_label = False,
                            axis_label_size = axis_label_size)

        # Plot the phase differences and the uncertainties
        phase_diffs = cspec_df[f"phase_diff_{component.lower()}"].values
        phase_diff_uncers = cspec_df[f"phase_diff_uncer_{component.lower()}"].values

        left = margin_x + i * (subplot_width + wspace)
        bottom = margin_y
        ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

        ax.plot(freqax, phase_diffs, color = color_comp, linewidth = linewidth)
        ax.fill_between(freqax, phase_diffs - phase_diff_uncers, phase_diffs + phase_diff_uncers, color = color_comp, alpha = 0.2)

        ax.axvline(freq_target, color = "crimson", linewidth = linewidth)

        ax.set_xlim(min_freq, max_freq)

        if i == 0:
            format_phase_diff_ylabels(ax,
                                     plot_axis_label = True,
                                     plot_tick_label = True,
                                     axis_label_size = axis_label_size)
        else:
            format_phase_diff_ylabels(ax,
                                     plot_axis_label = False,
                                     plot_tick_label = False,
                                     axis_label_size = axis_label_size)

        # Set the x-axis label
        format_freq_xlabels(ax,
                            major_tick_spacing = 25.0,
                            num_minor_ticks = 5,
                            plot_axis_label = True,
                            plot_tick_label = True,
                            axis_label_size = axis_label_size)

    # Set the suptitle

    fig.suptitle(f"Earthquake {earthquake_id}, {station1}-{station2}, {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%H:%M:%S')}", fontsize = suptitle_size, fontweight = "bold", y = 1.0)

    # Save the figure
    figname = f"earthquake_mt_inter_geo_sta_cspec_eq{earthquake_id}_{station1}_{station2}.png"
    save_figure(fig, figname)

    # Close the figure
    fig.clf()

    print("Done!")
    print("")

    