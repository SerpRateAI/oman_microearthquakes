# Plot the CWT power of a stationary resonance as a function of time for all stations

# Imports
from os.path import join
from numpy import abs, linspace, unravel_index
from pandas import read_csv, read_hdf
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, SAMPLING_RATE as sampling_rate
from utils_basic import get_geophone_coords, str2timestamp, time2suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_wavelet import get_stream_cwt
from utils_plot import format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
# Stationary resonance properties
base_name = "PR02549"
base_mode = 2

mode_name = "PR15295"
starttime = str2timestamp("2020-01-13 22:07:30")
endtime = str2timestamp("2020-01-13 22:09:30")

# CWT computation
center_freq = 20.0
bandwidth = 2.0

freq_window = 10.0
num_scales = 100

# Plotting
# Frequency-time plot
num_rows = 3
num_cols = 2
stations_to_plot = ["A01", "A16", "A19", "B01", "B19", "B20"]

figwidth_ft = 15
figheight_ft = 15

min_db = 10.0
max_db = 20.0

# Time plot
scale = 0.05
base = 10.0

figwidth_time = 15
figheight_time = 15

# Read the harmonic series information
print(f"Reading the information of the harmonic series with {base_name} as the base mode {base_mode}...")
filename = f"stationary_harmonic_series_{base_name}_base{base_mode}.csv"
inpath = join(indir, filename)
harmonic_series_df = read_csv(inpath)
freq_mode = harmonic_series_df[harmonic_series_df["name"] == mode_name]["observed_freq"].values[0]
max_freq = freq_mode + freq_window
min_freq = freq_mode - freq_window

# Compute the scales
print("Computing the scales...")
max_scale = center_freq * sampling_rate / min_freq
min_scale = center_freq * sampling_rate / max_freq
scales = linspace(min_scale, max_scale, num_scales)

# Read the waveforms in the time window
print("Reading and processing the waveforms in the time window...")
stream = read_and_process_windowed_geo_waveforms(starttime, endtime = endtime)

# Compute the CWT
print("Computing the CWT of the waveforms...")
stream_cwt = get_stream_cwt(stream, scales = scales, center_freq = center_freq, bandwidth = bandwidth)

# Plot the CWT power of the example stations
print("Plotting the CWT power of the example stations...")
power_dict = stream_cwt.get_total_power(db = True)

if num_rows * num_cols != len(stations_to_plot):
    raise ValueError("The number of rows and columns do not match the number of stations to plot.")

fig, axes = subplots(num_rows, num_cols, figsize = (figwidth_ft, figheight_ft), sharex = True, sharey = True)

for i, station in enumerate(stations_to_plot):
    ax = axes.flatten()[i]

    timeax = power_dict[station]["times"]
    freqax = power_dict[station]["freqs"]
    power_mat = power_dict[station]["power"]

    ax.pcolormesh(timeax, freqax, power_mat, cmap = "inferno", vmin = min_db, vmax = max_db)
    ax.set_title(station, fontsize = 12, fontweight = "bold")

    i_row, i_col = unravel_index(i, (num_rows, num_cols))

    if i_row == num_rows - 1:
        format_datetime_xlabels(ax, major_tick_spacing = "1min", num_minor_ticks=4)
    else:
        format_datetime_xlabels(ax, label = False, major_tick_spacing = "1min", num_minor_ticks=4)

    if i_col == 0:
        format_freq_ylabels(ax, major_tick_spacing = 2.0, num_minor_ticks=4)
    else:
        format_freq_ylabels(ax, label = False, major_tick_spacing = 2.0, num_minor_ticks=4)

# Save the plot
print("Saving the plot...")
starttime = time2suffix(starttime)
endtime = time2suffix(endtime)
figname = f"stationary_resonance_cwt_power_{mode_name}_{starttime}_{endtime}.png"
save_figure(fig, figname)

# Extract the CWT power at the resonance frequency
print("Extracting the CWT power at the resonance frequency...")
power_vs_time_dict = {}
for station in stations:
    power = power_dict[station]["power"]
    freqax = power_dict[station]["freqs"]
    timeax = power_dict[station]["times"]

    i_freq = abs(freqax - freq_mode).argmin()
    power_vs_time = power_mat[i_freq, :]
    power_vs_time_dict[station] = {"times": timeax, "power": power_vs_time}

# Get the station locations
stacoord_df = get_geophone_coords()
stacoord_df.sort_values(by = "north", inplace = True)

# Plot the CWT power at the resonance frequency
print("Plotting the CWT power at the resonance frequency...")
fig, ax = subplots(1, 1, figsize = (figwidth_time, figheight_time))

for i, station in enumerate(stacoord_df.index):
    power = power_vs_time_dict[station]["power"]
    timeax = power_vs_time_dict[station]["times"]

    ax.plot(timeax, (power - base) * scale + i, color = "black", linewidth = 1.0)
    ax.text(timeax[0], i, station, fontsize = 12, fontweight = "bold", ha = "left", va = "center")

format_datetime_xlabels(ax, major_tick_spacing = "1min", num_minor_ticks=4)
ax.set_ylim(-1, len(stacoord_df))

# Save the plot
print("Saving the plot...")
figname = f"stationary_resonance_cwt_power_vs_time_{mode_name}_{starttime}_{endtime}.png"
save_figure(fig, figname)





