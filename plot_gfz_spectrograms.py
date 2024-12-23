# Plot the spectrograms of the GFZ broadband data
# Imports
from os.path import join
from matplotlib.pyplot import subplots

from utils_basic import time2suffix
from utils_preproc import read_and_process_windowed_broadband_waveforms
from utils_torch import get_stream_spectrograms
from utils_plot import add_colorbar, format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
stations = ["COO03", "COO32"]
starttime = "2015-11-23 01:27:00"
endtime = "2015-11-23 01:29:00"
window_length = 1.0

panel_width = 15
panel_height = 5

min_db = -5.0
max_db = 30.0

major_time_tick_spacing = "1min"
num_minor_time_ticks = 4

major_freq_tick_spacing = 10.0
num_minor_freq_ticks = 5

# Read and process the data
print("Reading and processing the data...")
stream = read_and_process_windowed_broadband_waveforms(stations, starttime, endtime = endtime)

# Get the spectrograms
print("Computing the spectral the spectrograms...")
stream_stft = get_stream_spectrograms(stream, window_length = window_length)

# Plotting
print("Plotting the spectrograms...")
num_stations = len(stations)
fig, axes = subplots(nrows = num_stations, ncols = 1, figsize = (panel_width, panel_height * num_stations), sharex = True, sharey = True)

for i, station in enumerate(stations):
    ax = axes[i]
    stream_stft_sta = stream_stft.select(stations = station)
    total_psd = stream_stft_sta.get_total_psd()
    total_psd.to_db()

    psd = total_psd.psd_mat
    timeax = total_psd.times
    freqax = total_psd.freqs


    mappable = ax.pcolormesh(timeax, freqax, psd, cmap = "inferno", vmin = min_db, vmax = max_db)
    ax.text(0.01, 0.97, station, transform = ax.transAxes, va = "top", ha = "left", fontsize = 12, fontweight = "bold", bbox = dict(facecolor = "white", alpha = 1.0))
    
    if i == num_stations - 1:
        format_datetime_xlabels(ax,
                                major_tick_spacing = major_time_tick_spacing,
                                num_minor_ticks = num_minor_time_ticks,
                                va = "top", ha = "right", rotation = 15)

    format_freq_ylabels(ax,
                        major_tick_spacing = major_freq_tick_spacing,
                        num_minor_ticks = num_minor_freq_ticks)
                        
ax = axes[-1]
bbox = ax.get_position()
position = [bbox.x1 + 0.02, bbox.y0, 0.01, bbox.height]
cbar = add_colorbar(fig, mappable, "PSD (dB)", position)


# Save the figure
print("Saving the figure...")

starttime_suffix = time2suffix(starttime)
endtime_suffix = time2suffix(endtime)
figname = f"gfz_spectrograms_example_windoow{window_length:0}s_{starttime_suffix}_to_{endtime_suffix}.png"

save_figure(fig, figname)

    

                            




