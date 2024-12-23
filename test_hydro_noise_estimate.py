# Test estimating the electrical noise signal in the hydrophone data through stacking large number of time windows

# Import the necessary libraries
from os.path import join
from random import sample
from argparse import ArgumentParser
from numpy import abs, amax, isnan, zeros
from scipy.fft import fft, fftfreq
from matplotlib.pyplot import subplots

from utils_preproc import read_and_process_day_long_hydro_waveforms
from utils_plot import HYDRO_COLOR as color_hydro, PRESSURE_LABEL as label_pressure
from utils_plot import format_freq_xlabels, save_figure


# Inputs
# Command line arguments
parser = ArgumentParser(description = "Estimate the electrical noise signal in the hydrophone data through stacking large number of time windows")
parser.add_argument("--station", type = str, help = "Hydrophone station to process")
parser.add_argument("--window_length", type = float, help = "Length of the window in seconds")
parser.add_argument("--day", type = str, help = "Day to process")
parser.add_argument("--fraction_of_day", type = float, help = "Fraction of the day to process")
parser.add_argument("--amp_threshold", type = float, help = "Amplitude threshold for excluding a time window")

# Constants
num_to_plot = 20
amp_min = -75.0
amp_max = 75.0

# Parse the command line arguments
args = parser.parse_args()
station = args.station
window_length = args.window_length
day = args.day
frac = args.fraction_of_day
amp_threshold = args.amp_threshold

# Print the command line arguments
print(f"Station: {station}")
print(f"Window length: {window_length:.0f} s")
print(f"Day: {day}")
print(f"Amplitude threshold: {amp_threshold}")

# Read the hydrophone data
print("Reading the hydrophone data...")
stream_day = read_and_process_day_long_hydro_waveforms(day, stations = station)

# Slice the time series into time windows
print("Slicing the time series into time windows...")
signals_window = []
for trace_day in stream_day:
    location = trace_day.stats.location
    print(f"Processing location: {location}")
    sampling_interval = trace_day.stats.delta
    signal = trace_day.data
    numpts_window = int(window_length / sampling_interval)
    signals_window_loc = [signal[i:i + numpts_window] for i in range(0, int(len(signal) * frac) - numpts_window, numpts_window)]
    signals_window += signals_window_loc

# Stack the time windows
print("Stacking the time windows...")
signal_stacked = zeros(numpts_window)

num_stack = 0
signals_to_stack = []
for signal_window in signals_window:
    if amax(abs(signal_window)) < amp_threshold and not isnan(signal_window).any():
        signal_stacked += signal_window
        signals_to_stack.append(signal_window)
        num_stack += 1

signal_stacked /= num_stack
print(f"Number of time windows stacked: {num_stack}")

# Randomly draw 10 elements from signals_to_stack to plot
if len(signals_to_stack) >= num_to_plot:
    signals_to_plot = sample(signals_to_stack, num_to_plot)
else:
    signals_to_plot = signals_to_stack

# Compute the amplitude spectrum of the stacked signal
print("Computing the amplitude spectrum of the stacked signal...")
signal_stacked_fft = fft(signal_stacked)
freqax = fftfreq(numpts_window, sampling_interval)
freqax = freqax[:numpts_window // 2]

amp_spec_stacked = abs(signal_stacked_fft)
amp_spec_stacked = amp_spec_stacked[:numpts_window // 2]
amp_spec_stacked /= amax(amp_spec_stacked)

# Plotting
fig, axes = subplots(2, 1, figsize = (15, 10))

# Plot the stacked signal and the randomly drawn signals
ax = axes[0]
timeax = [i * sampling_interval for i in range(numpts_window)]

for signal_to_plot in signals_to_plot:
    ax.plot(timeax, signal_to_plot, color = color_hydro, alpha = 0.1, linewidth = 1.0)

ax.plot(timeax, signal_stacked, color = color_hydro, linewidth = 1.0)

ax.set_xlim(timeax[0], timeax[-1])
ax.set_ylim(amp_min, amp_max)

ax.set_xlabel("Time (s)")
ax.set_ylabel(label_pressure)

# Plot the amplitude spectrum of the stacked signal
ax = axes[1]

ax.plot(freqax, amp_spec_stacked, color = color_hydro, linewidth = 1.0)

ax.set_xlim(0, 0.5 / sampling_interval)
ax.set_yscale("log")

format_freq_xlabels(ax,
                    major_tick_spacing = 50.0,
                    num_minor_ticks = 5)

ax.set_ylabel("Amplitude")


# Save the figure
figname = f"test_hydro_noise_estimate_{station}_{day}_window{window_length:.0f}s_amp_thresh{amp_threshold:.0f}_frac{frac:.2f}.png"
save_figure(fig, figname)



