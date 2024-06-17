# Plot the stationary resonance frequencies as a function of time for all geophone stations

# Imports
from os.path import join
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_plot import format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
# Data
name = "SR38a"

# Plotting
min_freq_plot = 38.0
max_freq_plot = 38.5

major_time_tick_spacing = "24h"
num_minor_time_ticks = 4
date_format = "%Y-%m-%d"

major_freq_tick_spacing = 0.1
num_minor_freq_ticks = 5

# Read the data
filename = f"stationary_resonance_properties_{name}.h5"
inpath = join(indir, filename)

all_stations_df = read_hdf(inpath)

# Plot the data
fig, ax = subplots(figsize = (15, 5))

for station in stations:
    station_df = all_stations_df[all_stations_df["station"] == station]

    times = station_df.index
    freqs = station_df["frequency"]
    ax.plot(times, freqs, linewidth = 0.5, color = "lightgray")

format_datetime_xlabels(ax,
                        major_tick_spacing = major_time_tick_spacing, num_minor_ticks = num_minor_time_ticks,
                        date_format = date_format, rotation = 45, va = "top", ha = "right")

ax.set_ylim(min_freq_plot, max_freq_plot)
format_freq_ylabels(ax,
                    major_tick_spacing = major_freq_tick_spacing, num_minor_ticks = num_minor_freq_ticks)

# Save the figure
figname = f"stationary_resonance_frequencies_one_panel_{name}.png"
save_figure(fig, figname)

