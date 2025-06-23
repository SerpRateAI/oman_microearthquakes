"""
This script investigates the polarity issue of the hydrophone data
"""

# Import the necessary libraries
from matplotlib.pyplot import subplots
from pandas import Timestamp
from numpy import abs, amax

from utils_preproc import read_and_process_windowed_hydro_waveforms
from utils_plot import save_figure

# Define the station and the day
station = "B00"
starttime = Timestamp("2020-01-16T07:33:25.7")
location = "03"
duration = 0.5

# Read the waveforms
stream = read_and_process_windowed_hydro_waveforms(starttime,
                                                   dur = duration,
                                                   stations = station,
                                                   locations = location)

# Plot the waveforms
## Create the figure and the axis
fig, ax = subplots(nrows = 1, ncols = 1, figsize = (10, 10))

## Plot the waveform
trace = stream.select(location = location)[0]
waveform = trace.data
timeax = trace.times()

print(abs(waveform).max())

ax.plot(timeax, waveform)

## Format the x-axis
ax.set_xlim(timeax[0], timeax[-1])
ax.set_xlabel("Time (s)")

## Format the y-axis
ax.set_ylim(waveform.min(), waveform.max())
ax.set_ylabel("Amplitude")

## Format the title
ax.set_title(f"Waveform of {station} at {location} on {starttime}")

## Show the plot
save_figure(fig, "test_hydrophone_polarity.png")


