# Plot the day and night hydrophone and geophone probablistic power spectral densities

# Imports
from os.path import join
from numpy import arange
from pandas import DataFrame, IntervalIndex, Timestamp, Timedelta
from pandas import concat, crosstab, cut
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import get_geo_sunrise_sunset_times, is_daytime
from utils_spec import get_spectrogram_file_suffix, read_geo_spectrograms, read_hydro_spectrograms
from utils_plot import save_figure, format_freq_xlabels, format_db_ylabels


# Inputs
# Data
# Stations and locations to plot
station_geo = "B02"
station_hydro = "B00"
location_hydro = "04"

day_to_plot = "2020-01-20"

# Spectrogram
window_length = 300.0
overlap = 0.0
downsample = False
downsample_factor = 60

min_freq = 0.0
max_freq = 5.0

# Binning
freq_bin_width = 0.1 # Hz

min_db_hydro = -80.0 # dB
max_db_hydro = 0.0 # dB

min_db_geo = -15.0 # dB
max_db_geo = 30.0 # dB

db_bin_width = 1.0 # dB

# Plotting
linewidth = 0.1

major_freq_spacing = 1.0
num_minor_freq_ticks = 5

major_geo_db_spacing = 10.0
num_minor_geo_db_ticks = 5

major_hydro_db_spacing = 20.0
num_minor_hydro_db_ticks = 5

max_percent = 0.1

# Read the geophone spectrograms
starttime = Timestamp(day_to_plot, tz = "UTC")
endtime = starttime + Timedelta(days = 1)

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
# print(sun_df.loc[sun_df.index[0]])

# Generate the figure
fig, axes = subplots(2, 2, figsize = (10, 6), sharex = True)

# Extract the day and night hydrophone spectra
print("Plotting the day and night hydrophone spectra...")
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

print("Binning the hydrophone spectra...")
bin_edges_db = arange(min_db_hydro, max_db_hydro + db_bin_width, db_bin_width)
bin_edges_freq = arange(min_freq, max_freq + freq_bin_width, freq_bin_width)

print(f"Number of bins for power: {len(bin_edges_db) - 1}")
print(f"Number of bins for frequency: {len(bin_edges_freq) - 1}")

# Ensure consistent use of `cut` parameters
hydro_day_df["power_bin"] = cut(hydro_day_df["power"], bin_edges_db, right=False, include_lowest=True)
hydro_day_df["frequency_bin"] = cut(hydro_day_df["frequency"], bin_edges_freq, right=False, include_lowest=True)

hydro_night_df["power_bin"] = cut(hydro_night_df["power"], bin_edges_db, right=False, include_lowest=True)
hydro_night_df["frequency_bin"] = cut(hydro_night_df["frequency"], bin_edges_freq, right=False, include_lowest=True)

# Create contingency tables
contingency_day = crosstab(hydro_day_df["frequency_bin"], hydro_day_df["power_bin"])
contingency_night = crosstab(hydro_night_df["frequency_bin"], hydro_night_df["power_bin"])

print(contingency_day)

# Ensure the IntervalIndex matches the cut intervals
# interval_index_db = IntervalIndex.from_breaks(bin_edges_db, closed="left")
# interval_index_freq = IntervalIndex.from_breaks(bin_edges_freq, closed="left")

interval_index_db = IntervalIndex(hydro_day_df["power_bin"].cat.categories)
interval_index_freq = IntervalIndex(hydro_day_df["frequency_bin"].cat.categories)

# Reindex the contingency tables
contingency_day = contingency_day.reindex(index=interval_index_freq, columns=interval_index_db, fill_value=0)
contingency_night = contingency_night.reindex(index=interval_index_freq, columns=interval_index_db, fill_value=0)

# Convert to percentage
contingency_percent_day = contingency_day / contingency_day.sum().sum() * 100
contingency_percent_night = contingency_night / contingency_night.sum().sum() * 100

print(contingency_percent_day)

# Transpose if necessary
contingency_percent_day = contingency_percent_day.T
contingency_percent_night = contingency_percent_night.T

# Loop over the geophone times and plot the spectra in the day or night panel
print("Plotting the day and night geophone spectra...")
timeax = trace_total_geo.times
freqax = trace_total_geo.freqs
data = trace_total_geo.data

for i, time in enumerate(timeax):
    is_day = is_daytime(time, sun_df)

    if is_day:
        ax = axes[1, 0]
    else:
        ax = axes[1, 1]

    spec = data[:, i]
    ax.plot(freqax, spec, color = "black", alpha = 0.5, linewidth = linewidth)

# Generate the figure and axes
fig, axes = subplots(2, 2, figsize = (10, 6), sharex = True)

# Plot the hydrophone probablistic power spectral densities
print("Plotting the hydrophone probablistic power spectral densities...")
ax = axes[0, 0]
# bin_centers_db = bin_edges_db[:-1] + db_bin_width / 2
# bin_centers_freq = bin_edges_freq[:-1] + freq_int_hydro / 2
ax.pcolormesh(bin_edges_freq, bin_edges_db, contingency_percent_day, cmap = "plasma", vmax = max_percent)

# ax = axes[0, 1]
# ax.pcolormesh(bin_edges_freq, bin_edges_db, contingency_percent_night, cmap = "plasma")


# Format the axes
print("Formatting the axes...")

# Set the x-limits
# axes[0, 0].set_xlim(min_freq, max_freq)

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
axes[0, 0].set_title(f"Hydrophone {station_hydro}.{location_hydro}, Day", fontweight = "bold")
axes[0, 1].set_title(f"Hydrophone {station_hydro}.{location_hydro}, Night", fontweight = "bold")
axes[1, 0].set_title(f"Geophone {station_geo}, Day", fontweight = "bold")
axes[1, 1].set_title(f"Geophone {station_geo}, Night", fontweight = "bold")

# Set the figure title
fig.suptitle(f"Day and Night Hydrophone and Geophone Spectra on {day_to_plot}", fontweight = "bold", y = 0.95)

# Save the figure
print("Saving the figure...")
figname = f"hydro_n_geo_day_and_night_ppsd_{day_to_plot}_{station_hydro}.{location_hydro}_{station_geo}_{suffix_spec}_freq{min_freq:.2f}to{max_freq:.2f}hz.png"
save_figure(fig, figname)

