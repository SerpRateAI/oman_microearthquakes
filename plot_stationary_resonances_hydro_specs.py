# Plot the comparison of the hydrophone spectrograms of two stationary resonances 

# Imports
from os.path import join
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_spec import get_spectrogram_file_suffix, read_hydro_spectrograms
from utils_plot import POWER_LABEL as colorbar_label
from utils_plot import add_colorbar, plot_hydro_stft_spectrograms, save_figure

# Inputs
# Data
station_to_plot = "B00"
name1 = "SR38a"
name2 = "SR25a"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

min_freq1 = 38.0
max_freq1 = 39.0

min_freq2 = 25.0
max_freq2 = 26.0

min_db = -80.0
max_db = -50.0

major_time_spacing = "24h"
num_minor_time_ticks = 4

major_freq_spacing = 0.5
num_minor_freq_ticks = 5

# Plotting
column_width = 8.0
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
                                      min_freq = min_freq1, max_freq = max_freq1)


# Signal 2
suffix = get_spectrogram_file_suffix(window_length, overlap, downsample)
filename = f"whole_deployment_daily_hydro_spectrograms_{station_to_plot}_{suffix}.h5"
inpath = join(indir, filename)

hydro_spec2 = read_hydro_spectrograms(inpath,
                                      starttime = starttime, endtime = endtime,
                                      min_freq = min_freq2, max_freq = max_freq2)

# Plotting
print("Plotting the spectrograms...")
num_loc = len(hydro_spec1)
fig, axes = subplots(num_loc, 2, figsize=(column_width * 2, row_height * num_loc), sharex = True)
# Plot Signal 1
axes1 = axes[:, 0]
axes1, _ = plot_hydro_stft_spectrograms(hydro_spec1, 
                                       axes = axes1,
                                       dbmin = min_db, dbmax = max_db,
                                       major_time_spacing = major_time_spacing, num_minor_time_ticks = num_minor_time_ticks,
                                       major_freq_spacing = major_freq_spacing, num_minor_freq_ticks = num_minor_freq_ticks,
                                       location_label_x = location_label_x, location_label_y = location_label_y,
                                       title = name1)


# Plot Signal 2
axes2 = axes[:, 1]
axes2, quadmesh = plot_hydro_stft_spectrograms(hydro_spec2, 
                                       axes = axes2,
                                       dbmin = min_db, dbmax = max_db,
                                       major_time_spacing = major_time_spacing, num_minor_time_ticks = num_minor_time_ticks,
                                       major_freq_spacing = major_freq_spacing, num_minor_freq_ticks = num_minor_freq_ticks,
                                       location_label_x = location_label_x, location_label_y = location_label_y,
                                       title = name2)

# Turn off the y labels of axes2
for ax in axes2:
    ax.set_ylabel("")

# Add colorbar
ax = axes2[-1]
bbox = ax.get_position()
position = [bbox.x1 + colorbar_gap, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, quadmesh, colorbar_label, position, orientation = "vertical")

# Save the figure
figname = f"stationary_resonances_hydro_specs_{name1}_{name2}_{station_to_plot}.png"
save_figure(fig, figname)


