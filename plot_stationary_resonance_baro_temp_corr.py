# Plot the correlation between the stationary-resonance frequency and the baromatric and temperature data

# Imports
from os.path import join
from pandas import read_csv, read_excel
from matplotlib.pyplot import subplots

from utils_basic import STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, SPECTROGRAM_DIR as indir
from utils_basic import get_baro_temp_data
from utils_plot import add_day_night_shading, format_datetime_xlabels, save_figure

# Inputs
name = "SR25a"
min_num_sta = 9

# Read the data
print(f"Reading the mean frequency of the stationary resonance for {name}...")
filename = f"stationary_resonance_mean_freq_{name}_geo_num{min_num_sta}.csv"
inpath = join(indir, filename)

mean_freq_by_time = read_csv(inpath, index_col=0, parse_dates=True)
mean_freq_by_time.index = mean_freq_by_time.index.tz_localize('UTC')

baro_temp_df = get_baro_temp_data()

# Plot the data
print("Plotting...")
fig, axes = subplots(3, 1, figsize=(12, 6), sharex=True)

# Slice the data frame to the time of interest
mean_freq_by_time = mean_freq_by_time.loc[starttime:endtime]
baro_temp_df = baro_temp_df.loc[starttime:endtime]

# Plot the reasonance frequency data
ax = axes[0]
ax.scatter(mean_freq_by_time.index, mean_freq_by_time, s = 1, c = "black", label="Mean Frequency")

add_day_night_shading(ax)

ax.set_ylabel("Frequency (Hz)")

# Plot the barometric data
timeax = baro_temp_df.index

ax = axes[1]
baro = baro_temp_df['pressure']
ax.plot(timeax, baro, color="dodgerblue", label="Barometric Pressure")

add_day_night_shading(ax)

ax.set_ylabel("Pressure (Bar)")

# Plot the temperature data
ax = axes[2]
temp = baro_temp_df['temperature']
ax.plot(timeax, temp, color="darkorange", label="Temperature")

add_day_night_shading(ax)

ax.set_ylabel("Temperature ($\degree$C)")

# Set the x-axis limits and labels
ax.set_xlim(starttime, endtime)

format_datetime_xlabels(ax,
                        date_format = "%Y-%m-%d",
                        major_tick_spacing = "1d", num_minor_ticks = 4,
                        va = "top", ha = "right", rotation = 30,
                        axis_label_size = 12, tick_label_size = 10)

# Set the super title
fig.suptitle(f"{name}",
             fontsize = 15, fontweight = "bold", y = 0.93)


# Save the figure
filename = f"stationary_resonance_{name}_geo_baro_temp_corr.png"
save_figure(fig, filename)