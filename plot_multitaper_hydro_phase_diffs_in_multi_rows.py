# Plot the inter-locationphase differences of a stationary resonance as a function of time in multiple rows

# Imports
from os.path import join
from argparse import ArgumentParser
from pandas import date_range, read_csv, read_hdf
from matplotlib.pyplot import subplots, colormaps
from matplotlib.colors import Normalize

from utils_basic import MT_DIR as indir, STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro
from utils_plot import format_datetime_xlabels, format_phase_diff_ylabels, save_figure

# Inputs
# Parse the arguments
parser = ArgumentParser(description = "Plot the powers of a stationary resonance as a function of time in multiple rows")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--station", type = str, help = "Station name")
parser.add_argument("--location1", type = str, help = "Location 1")
parser.add_argument("--location2", type = str, help = "Location 2")
parser.add_argument("--window_length_mt", type = float, help = "MT window length in second")
parser.add_argument("--min_cohe", type = float, help = "Minimum coherence")

starttime = starttime_hydro
endtime = endtime_hydro

# Parse the arguments
args = parser.parse_args()
mode_name = args.mode_name
station = args.station
location1 = args.location1
location2 = args.location2
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe

# Print the parameters
print(f"### Plotting the inter-location phase differences of {station}.{location1} and {station}.{location2} ###")
print(f"Mode name: {mode_name}")
print(f"Station: {station}")
print(f"Location 1: {location1}")
print(f"Location 2: {location2}")
print(f"MT window length: {window_length_mt} s")
print(f"Minimum coherence: {min_cohe}")

# Constants
num_rows = 4
column_width = 10.0
row_height = 2.0

major_time_spacing = "15d"
num_minor_time_ticks = 3

major_freq_spacing = 0.5
num_minor_freq_ticks = 5

markersize = 1.5
linewidth = 0.5

# Read the plotting frequency range
filename = f"multitaper_inter_loc_phase_diffs_{mode_name}_{station}_{location1}_{location2}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"

inpath = join(indir, filename)
phase_diff_df = read_csv(inpath, parse_dates=["time"])

print(f"Plotting the inter-location phase differences of {station}.{location1} and {station}.{location2}...")

# Generate the figure
print("Generating the figure...")
fig, axes = subplots(num_rows, 1, figsize=(column_width, row_height * num_rows), sharey = True)

# Plotting
print("Plotting the inter-location phase differences...")

# Plot each time window
windows = date_range(starttime, endtime, periods = num_rows + 1)

for i in range(num_rows):
    starttime = windows[i]
    endtime = windows[i + 1]

    print("Plotting the time window: ", starttime, " - ", endtime)
    phase_diff_df_window = phase_diff_df[(phase_diff_df["time"] >= starttime) & (phase_diff_df["time"] <= endtime)]

    print(f"Plotting the inter-location phase differences for the window")
    ax = axes[i]
    ax.errorbar(phase_diff_df_window["time"], phase_diff_df_window["phase_diff"], yerr = phase_diff_df_window["phase_diff_uncer"], 
                fmt = "o", markerfacecolor="none", markeredgecolor="tab:purple", markersize = markersize, 
                color = "tab:purple", markeredgewidth = linewidth, elinewidth = linewidth, capsize=1, zorder=2)
    
    ax.set_xlim(starttime, endtime)

    if i == num_rows - 1:
        format_datetime_xlabels(ax,
                                plot_axis_label = True,
                                major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, date_format = "%Y-%m-%d")
    else:
        format_datetime_xlabels(ax,
                                plot_axis_label = False,
                                major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, date_format = "%Y-%m-%d")

    # Set the y-axis labels
    if i == num_rows - 1:
        format_phase_diff_ylabels(ax,
                                  plot_axis_label=True, plot_tick_label=True)
    else:
        format_phase_diff_ylabels(ax,
                                  plot_axis_label=False, plot_tick_label=True)

fig.suptitle(f"{mode_name}, {station}.{location1}-{location2}", y = 0.92, fontsize = 14, fontweight = "bold")

# Save the figure
figname = f"multitaper_inter_loc_phase_diffs_{mode_name}_{station}_{location1}_{location2}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.png"
save_figure(fig, figname)
