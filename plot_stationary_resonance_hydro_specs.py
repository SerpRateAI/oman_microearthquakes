# Plot the comparison of the hydrophone spectrograms of two stationary resonances 

# Imports
from os.path import join
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro
from utils_spec import get_spectrogram_file_suffix, read_hydro_spectrograms
from utils_plot import POWER_LABEL as colorbar_label
from utils_plot import add_colorbar, plot_hydro_stft_spectrograms, save_figure

# Inputs
# Data
station_to_plot = "A00"
name = "SR76a"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

starttime = starttime_hydro
endtime = endtime_hydro

min_freq = 76.0
max_freq = 77.0

pred_freq = 76.5

min_db = -80.0
max_db = -60.0

major_time_spacing = "15d"
num_minor_time_ticks = 3

major_freq_spacing = 0.5
num_minor_freq_ticks = 5

# Plotting
column_width = 10.0
row_height = 2.0

colorbar_width = 0.01
colorbar_gap = 0.02

location_label_x = 0.02
location_label_y = 0.94

# Read the spectrograms
print("Reading the spectrograms...")
# Signal 1
suffix = get_spectrogram_file_suffix(window_length, overlap, downsample)
filename = f"whole_deployment_daily_hydro_spectrograms_{station_to_plot}_{suffix}.h5"
inpath = join(indir, filename)

hydro_spec1 = read_hydro_spectrograms(inpath,
                                      starttime = starttime, endtime = endtime,
                                      min_freq = min_freq, max_freq = max_freq)

# Plotting
print("Plotting the spectrograms...")
num_loc = len(hydro_spec1)
fig, axes = subplots(num_loc, 1, figsize=(column_width, row_height * num_loc), sharex = True)

# Plot Signal 1
axes, quadmesh = plot_hydro_stft_spectrograms(hydro_spec1, 
                                       axes = axes,
                                       dbmin = min_db, dbmax = max_db,
                                       major_time_spacing = major_time_spacing, num_minor_time_ticks = num_minor_time_ticks,
                                       major_freq_spacing = major_freq_spacing, num_minor_freq_ticks = num_minor_freq_ticks,
                                       location_label_x = location_label_x, location_label_y = location_label_y,
                                       title = name)

for ax in axes:
    ax.axhline(y = pred_freq, color = "aqua", linestyle = ":", linewidth = 1.0)


# Add colorbar
ax = axes[-1]
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, quadmesh, colorbar_label, position, orientation = "vertical")

# Save the figure
figname = f"stationary_resonances_hydro_specs_{name}_{station_to_plot}_{suffix}.png"
save_figure(fig, figname)


