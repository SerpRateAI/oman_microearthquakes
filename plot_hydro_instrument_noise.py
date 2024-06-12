# Plot the instrument noise for the hydrophones

# Import
from os.path import join

from utils_preproc import read_and_process_windowed_hydro_waveforms
from utils_plot import plot_windowed_hydro_waveforms, save_figure

# Input
# Data
station_to_plot = "B00"
starttime = "2020-01-13T20:00:00"
dur = 2.0
min_freq = 5.0

# Plotting
title = "Periodic instrument noise"
figname = "periodic_instrument_noise.png"

major_time_spacing = "1s"
num_minor_time_ticks = 5

depth_lim = (0.0, 450.0)

scalebar_label_size = 10
scalebar_width = 1.0

time_scalebar_offset_x = "300ms"
time_scalebar_offset_y = 15.0
time_scalebar_length = "100ms"

pa_scalebar_offset_x = "50ms"
pa_scalebar_offset_y = 15.0
pa_scalebar_length = 1e-2

# Read and preprocess the waveforms
print("Reading and processing hydrophone waveforms...")
stream = read_and_process_windowed_hydro_waveforms(starttime, dur = dur, station = station_to_plot, min_freq = min_freq)

# Plotting
print("Plotting hydrophone waveforms...")
fig, ax = plot_windowed_hydro_waveforms(stream,
                                        scale = 1e3,
                                        depth_lim = depth_lim,
                                        date_format = "%Y-%m-%d %H:%M:%S", major_time_spacing = major_time_spacing, num_minor_time_ticks = num_minor_time_ticks,
                                        va = "top", ha = "right", rotation = 5,
                                        plot_pa_scalebar = True, plot_time_scalebar = True,
                                        scalebar_width = scalebar_width, scalebar_label_size = scalebar_label_size,
                                        pa_scalebar_offset_x = pa_scalebar_offset_x, pa_scalebar_offset_y = pa_scalebar_offset_y, pa_scalebar_length = pa_scalebar_length,
                                        time_scalebar_offset_x = time_scalebar_offset_x, time_scalebar_offset_y = time_scalebar_offset_y, time_scalebar_length = time_scalebar_length)

ax.set_title(title, fontsize = 15, fontweight = "bold")

# Save the figure
print("Saving the figure...")
save_figure(fig, figname)

