# Plot the daily spectrograms of a geophone station

from os.path import join
from pandas import Timestamp

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, read_hydro_spectrograms, string_to_time_label
from utils_plot import plot_hydro_stft_spectrograms, save_figure

# Inputs

# Data
station = "A00"
day = "2020-01-13"
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Plotting
dbmin = -80.0
dbmax = -50.0

min_freq = 0.0
max_freq = 50.0

major_time_spacing = "6h"
minor_time_spacing = "1h"

major_freq_spacing = 10.0
minor_freq_spacing = 2.0

dpi = 1200

# marker = True

# starttime_marker = Timestamp("2020-01-13T20:00:00")
# endtime_marker = Timestamp("2020-01-13T21:00:00")

# Read the spectrograms
print(f"Reading the spectrogram of {station} on {day}...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
filename = f"whole_deployment_daily_hydro_spectrograms_{station}_{suffix_spec}.h5"
inpath = join(indir, filename)

time_label = string_to_time_label(day)
# print(time_label)
stream_spec = read_hydro_spectrograms(inpath, time_labels = time_label, min_freq = min_freq, max_freq = max_freq)

# Plot the spectrograms
print("Plotting the spectrograms...")
fig, axes, cbar = plot_hydro_stft_spectrograms(stream_spec, 
                                               dbmin = dbmin, dbmax = dbmax,
                                               min_freq = min_freq, max_freq = max_freq,
                                               date_format = "%Y-%m-%d %H:%M:%S",
                                               major_time_spacing = major_time_spacing, minor_time_spacing = minor_time_spacing, 
                                               major_freq_spacing = major_freq_spacing, minor_freq_spacing = minor_freq_spacing,
                                               time_tick_rotation = 5)


# Save the figure
figname = f"daily_hydro_spectrograms_{day}_{station}_{suffix_spec}_{min_freq:.0f}to{max_freq:.0f}hz.png"
save_figure(fig, figname, dpi = dpi)

