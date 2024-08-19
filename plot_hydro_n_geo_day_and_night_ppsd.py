# Plot the day and night hydrophone and geophone probablistic power spectral densities

# Imports
from os.path import join
from numpy import arange
from pandas import DataFrame, IntervalIndex, Timestamp, Timedelta
from pandas import concat, crosstab, cut
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo
from utils_basic import get_geo_sunrise_sunset_times, is_daytime
from utils_spec import get_prob_psd, get_organpipe_freqs, get_spectrogram_file_suffix, read_geo_spectrograms, read_hydro_spectrograms
from utils_plot import add_colorbar, save_figure, format_freq_xlabels, format_db_ylabels


# Inputs
# Data
# Stations and locations to plot
station_geo = "A01"
station_hydro = "A00"
location_hydro = "03"

day_to_plot = None

# Spectrogram
window_length = 300.0
overlap = 0.0
downsample = False
downsample_factor = 60

min_freq = 0.0
max_freq = 6.0

# Binning
freq_bin_width = 0.05 # Hz

min_db_hydro = -80.0 # dB
max_db_hydro = 0.0 # dB

min_db_geo = -15.0 # dB
max_db_geo = 30.0 # dB

db_bin_width_geo = 1.0 # dB
db_bin_width_hydro = 2.0 # dB

# Organpipe modes
orders = [1, 3, 5]

# Plotting
linewidth = 1.0

major_freq_spacing = 1.0
num_minor_freq_ticks = 5

major_geo_db_spacing = 10.0
num_minor_geo_db_ticks = 5

major_hydro_db_spacing = 20.0
num_minor_hydro_db_ticks = 5

max_density = 2.0

cbar_offset = 0.02
cbar_width = 0.01

cbar_tick_spacing = 0.5

fontsize_title = 12
fontsize_sup_title = 14

# Read the geophone spectrograms
if day_to_plot is None:
    starttime = starttime_geo
    endtime = endtime_geo
    time_range = "whole_deployment"
else:
    starttime = Timestamp(day_to_plot, tz = "UTC")
    endtime = starttime + Timedelta(days = 1)
    time_range = day_to_plot

print("Reading the geophone and hydrophone spectrograms...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
filename_in = f"whole_deployment_daily_geo_spectrograms_{station_geo}_{suffix_spec}.h5"
inpath = join(indir, filename_in)

stream_geo = read_geo_spectrograms(inpath, starttime = starttime, endtime = endtime, min_freq = min_freq, max_freq = max_freq)
trace_total_geo = stream_geo.get_total_power()
trace_total_geo.to_db()


# Read the hydrophone spectrograms
filename_in = f"whole_deployment_daily_hydro_spectrograms_{station_hydro}_{suffix_spec}.h5"
inpath = join(indir, filename_in)

stream_hydro = read_hydro_spectrograms(inpath, starttime = starttime, endtime = endtime, locations = location_hydro, min_freq = min_freq, max_freq = max_freq)
trace_hydro = stream_hydro[0]
trace_hydro.to_db()

freq_int_hydro = trace_hydro.freq_interval

# Read the sunrise and sunset times
sun_df = get_geo_sunrise_sunset_times()

# Generate the figure
fig, axes = subplots(2, 2, figsize = (10, 6), sharex = True)

# Extract the day and night hydrophone spectra
print("Extracting the day and night hydrophone spectra...")
timeax = trace_hydro.times
freqax = trace_hydro.freqs
data = trace_hydro.data

hydro_day_dfs = []
hydro_night_dfs = []
for i, time in enumerate(timeax):
    is_day = is_daytime(time, sun_df)

    psd = data[:, i]
    hydro_df = DataFrame({"frequency": freqax, "power": psd})

    if is_day:
        hydro_day_dfs.append(hydro_df)
    else:
        hydro_night_dfs.append(hydro_df)

hydro_day_df = concat(hydro_day_dfs, axis = 0)
hydro_night_df = concat(hydro_night_dfs, axis = 0)

print("Computing the hydophone PPSDs...")
bin_edges_freq, bin_edges_db_hydro, table_hydro_day_df = get_prob_psd(hydro_day_df, 
                                                                min_freq = min_freq, max_freq = max_freq, freq_bin_width = freq_bin_width, min_db = min_db_hydro, max_db = max_db_hydro, db_bin_width = db_bin_width_hydro)

_, _, table_hydro_night_df = get_prob_psd(hydro_night_df,
                                        min_freq = min_freq, max_freq = max_freq, freq_bin_width = freq_bin_width, min_db = min_db_hydro, max_db = max_db_hydro, db_bin_width = db_bin_width_hydro)

# Extract the day and night geophone spectra
timeax = trace_total_geo.times
freqax = trace_total_geo.freqs
data = trace_total_geo.data

geo_day_dfs = []
geo_night_dfs = []
for i, time in enumerate(timeax):
    is_day = is_daytime(time, sun_df)

    psd = data[:, i]
    geo_df = DataFrame({"frequency": freqax, "power": psd})

    if is_day:
        geo_day_dfs.append(geo_df)
    else:
        geo_night_dfs.append(geo_df)

geo_day_df = concat(geo_day_dfs, axis = 0)
geo_night_df = concat(geo_night_dfs, axis = 0)

print("Computing the geophone PPSDs...")
bin_edges_freq, bin_edges_db_geo, table_geo_day_df = get_prob_psd(geo_day_df,
                                min_freq = min_freq, max_freq = max_freq, freq_bin_width = freq_bin_width, min_db = min_db_geo, max_db = max_db_geo, db_bin_width = db_bin_width_geo)

_, _, table_geo_night_df = get_prob_psd(geo_night_df,
                                  min_freq = min_freq, max_freq = max_freq, freq_bin_width = freq_bin_width, min_db = min_db_geo, max_db = max_db_geo, db_bin_width = db_bin_width_geo)

# Compute the organpipe modes
print("Computing the organpipe modes...")
freqs_org = get_organpipe_freqs(orders)
print(f"Organpipe modes: {freqs_org}")

# Plotting the probablistic spectral densities and the organpipe modes
# Generate the figure and axes
fig, axes = subplots(2, 2, figsize = (10, 6), sharex = True)

# Plot the hydrophone probablistic power spectral densities and the organpipe modes
print("Plotting the hydrophone probablistic power spectral densities...")
ax = axes[0, 0]
mappable = ax.pcolormesh(bin_edges_freq, bin_edges_db_hydro, table_hydro_day_df, cmap = "plasma", vmax = max_density)

for freq_org in freqs_org:
    ax.axvline(freq_org, color = "lime", linestyle = "--", linewidth = linewidth)

ax = axes[0, 1]
ax.pcolormesh(bin_edges_freq, bin_edges_db_hydro, table_hydro_night_df, cmap = "plasma", vmax = max_density)

for freq_org in freqs_org:
    ax.axvline(freq_org, color = "lime", linestyle = "--", linewidth = linewidth)

# Plot the geophone probablistic power spectral densities
print("Plotting the geophone probablistic power spectral densities...")
ax = axes[1, 0]
ax.pcolormesh(bin_edges_freq, bin_edges_db_geo, table_geo_day_df, cmap = "plasma", vmax = max_density)

ax = axes[1, 1]
ax.pcolormesh(bin_edges_freq, bin_edges_db_geo, table_geo_night_df, cmap = "plasma", vmax = max_density)

# Format the axes
print("Formatting the axes...")

# Set the x-limits
axes[0, 0].set_xlim(min_freq, max_freq)

# Set the x-ticks
format_freq_xlabels(axes[1, 0], 
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

format_freq_xlabels(axes[1, 1],
                    major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

# Set the y-limits
axes[0, 0].set_ylim(min_db_hydro, max_db_hydro)
axes[0, 1].set_ylim(min_db_hydro, max_db_hydro)
axes[1, 0].set_ylim(min_db_geo, max_db_geo)
axes[1, 1].set_ylim(min_db_geo, max_db_geo)

# Set the y-ticks
format_db_ylabels(axes[0, 0], 
                  label = True,
                  major_tick_spacing = major_hydro_db_spacing, num_minor_ticks = num_minor_hydro_db_ticks)

format_db_ylabels(axes[0, 1],
                  label = False,
                  major_tick_spacing = major_hydro_db_spacing, num_minor_ticks = num_minor_hydro_db_ticks)

format_db_ylabels(axes[1, 0],
                  label = True,
                  major_tick_spacing = major_geo_db_spacing, num_minor_ticks = num_minor_geo_db_ticks)

format_db_ylabels(axes[1, 1],
                  label = False,
                  major_tick_spacing = major_geo_db_spacing, num_minor_ticks = num_minor_geo_db_ticks)

# Set the subplot titles
axes[0, 0].set_title(f"Hydrophone {station_hydro}.{location_hydro}, Day", fontweight = "bold", fontsize = fontsize_title)
axes[0, 1].set_title(f"Hydrophone {station_hydro}.{location_hydro}, Night", fontweight = "bold", fontsize = fontsize_title)
axes[1, 0].set_title(f"Geophone {station_geo}, Day", fontweight = "bold", fontsize = fontsize_title)
axes[1, 1].set_title(f"Geophone {station_geo}, Night", fontweight = "bold", fontsize = fontsize_title)

# Set the figure title
if day_to_plot is None:
    fig.suptitle(f"Hydrophone and Geophone Probablistic Power Spectral Density", fontweight = "bold", y = 0.97, fontsize = fontsize_sup_title)
else:
    fig.suptitle(f"Hydrophone and Geophone Probablistic Power Spectral Density, {day_to_plot}", fontweight = "bold", y = 0.97, fontsize = fontsize_sup_title)

# Add the colorbar
ax = axes[1, 1]
bbox = ax.get_position()
position = [bbox.x1 + cbar_offset, bbox.y0, cbar_width, bbox.height]

cbar = add_colorbar(fig, mappable, "Probability density", position, orientation = "vertical", major_tick_spacing = cbar_tick_spacing)

# Save the figure
print("Saving the figure...")
figname = f"hydro_n_geo_day_and_night_ppsd_{time_range}_{station_hydro}.{location_hydro}_{station_geo}_{suffix_spec}_freq{min_freq:.2f}to{max_freq:.2f}hz.png"
save_figure(fig, figname)

