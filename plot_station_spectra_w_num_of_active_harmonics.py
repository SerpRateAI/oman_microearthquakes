# Plot the spectra of a geophone station with more than a certain number of active harmonics
# The total PSD with be colored according to the time

# Imports
from os.path import join
from pandas import read_csv, read_hdf
from matplotlib.pyplot import subplots
from numpy import amax, argmax

from utils_basic import SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime
from utils_basic import power2db
from utils_spec import get_spectrogram_file_suffix, read_time_slice_from_geo_spectrograms
from utils_plot import format_db_ylabels, format_freq_xlabels, save_figure

# Inputs
# Spectrogram computation
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Stationary harmonic series
base_name = "PR01250"
base_mode = 1
min_num_modes = 5

# Station to plot
station_to_plot = "A19"

fig_width = 10.0
fig_height = 6.0

linewidth_spec = 0.5
linedwidth_resonance = 0.5

min_freq = 0.0
max_freq = 200.0

major_freq_tick_spacing = 50.0
num_minor_freq_ticks = 5

min_db = -30.0
max_db = 40.0

db_threshold = 40.0

major_db_tick_spacing = 10.0
num_minor_db_ticks = 5


# Print the inputs
print("### Plotting the spectra of a geophone station with more than a certain number of active harmonics ###")
print("Spectrogram computation:")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print(f"Downsample: {downsample}")
print(f"Downsample factor: {downsample_factor}")

print("Stationary harmonic series:")
print(f"Base name: {base_name}")
print(f"Base mode: {base_mode:d}")
print(f"Minimum number of active modes: {min_num_modes:d}")

print("Station to plot:")
print(f"Station: {station_to_plot}")

# Read the resonance frequencies
print("Reading the resonance frequencies...")
filename_in = f"stationary_harmonic_series_{base_name}_base{base_mode:d}.csv"
inpath = join(indir, filename_in)
resonance_df = read_csv(inpath)


# Read the time window information
print("Reading the time window information...")
filename_in = f"time_windows_w_number_of_active_harmonics_{base_name}_base{base_mode:d}_num{min_num_modes:d}.h5"
inpath = join(indir, filename_in)
time_window_df = read_hdf(inpath, key = "time_windows")
time_window_df = time_window_df[time_window_df["station"] == station_to_plot]

# # Find the row with the maximum "num_modes" value
# max_num_modes_row = time_window_df.loc[time_window_df['num_modes'].idxmax()]

# Read the spectrogra for each time window
print("Reading the spectra...")
suffix = get_spectrogram_file_suffix(window_length, overlap, downsample)
inpath = f"whole_deployment_daily_geo_spectrograms_{station_to_plot}_{suffix}.h5"
inpath = join(indir, inpath)
print(f"Reading the spectrograms from {inpath}...")
spec_dict = {}
for i_time, time in enumerate(time_window_df["time"]):
    print(f"Reading the spectrum at time {time}...")
    power_dict = read_time_slice_from_geo_spectrograms(inpath, time)
    if i_time == 0:
        freqs = power_dict["frequencies"]

    for i_component, component in enumerate(components):
        power = power_dict[component]
        if i_component == 0:
            total_power = power
        else:
            total_power += power

    total_power = power2db(total_power)

    print("Index of maximum power: ", argmax(total_power))

    if amax(total_power) > db_threshold:
        print(f"Time {time} has a abnormally large maximum power of {amax(total_power):.2f} dB. Skipping this time window...")
        continue

    spec_dict[time] = total_power

# Plot the spectra
print("Plotting the spectra...")
fig, ax = subplots(figsize = (fig_width, fig_height))
# i = 0
for time, spec in spec_dict.items():
    ax.plot(freqs, spec, c = "black", linewidth = linewidth_spec, label = f"Time: {time:.2f}")

    # if i < 5:
    #     ax.plot(freqs, spec, c = "black", linewidth_spec = linewidth_spec, label = f"Time: {time:.2f}")
    #     i += 1

# Add the resonance frequencies
for i, row in resonance_df.iterrows():
    freq = row["observed_freq"]
    ax.axvline(x = freq, color = "crimson", linestyle = ":", linewidth = linedwidth_resonance, label = f"Resonance: {freq:.2f}")

ax.set_xlim([min_freq, max_freq])
format_freq_xlabels(ax, major_tick_spacing = major_freq_tick_spacing, num_minor_ticks = num_minor_freq_ticks)

ax.set_ylim([min_db, max_db])
format_db_ylabels(ax, major_tick_spacing = major_db_tick_spacing, num_minor_ticks = num_minor_db_ticks)

# Save the figure
print("Saving the figure...")
filename_out = f"spectra_w_num_of_active_harmonics_{base_name}_base{base_mode:d}_num{min_num_modes:d}_{station_to_plot}.png"
save_figure(fig, filename_out)
