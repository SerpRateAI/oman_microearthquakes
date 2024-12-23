# Test denoising the hydrophone data using the electrical noise signal estimated from the data

# Import the necessary libraries
from os.path import join
from random import sample
from argparse import ArgumentParser
from numpy import abs, amax, isnan, zeros
from scipy.fft import fft, fftfreq
from matplotlib.pyplot import subplots
from obspy import Stream

from utils_basic import NUM_SEONCDS_IN_DAY as num_seconds_in_day
from utils_preproc import read_and_process_day_long_hydro_waveforms
from utils_torch import get_stream_stft
from utils_spec import StreamSTFT
from utils_plot import HYDRO_COLOR as color_hydro, PRESSURE_LABEL as label_pressure
from utils_plot import format_freq_xlabels, save_figure

# Function to estimate the noise signal from a stream of hydrophone data
def estimate_noise_signal(stream, window_length, amp_threshold):
    sampling_interval = stream[0].stats.delta

    # Slice the time series into time windows
    print("Slicing the time series into time windows...")
    signals_window = []
    for trace in stream:
        signal = trace.data
        numpts_window = int(window_length / sampling_interval)
        signals_window_loc = [signal[i:i + numpts_window] for i in range(0, int(len(signal) * frac) - numpts_window, numpts_window)]
        signals_window += signals_window_loc

    # Stack the time windows
    print("Stacking the time windows...")
    signal_stacked = zeros(numpts_window)

    num_stack = 0
    for signal_window in signals_window:
        if amax(abs(signal_window)) < amp_threshold and not isnan(signal_window).any():
            signal_stacked += signal_window
            num_stack += 1

    signal_stacked /= num_stack
    print(f"Number of time windows stacked: {num_stack}")

    return signal_stacked

# Subtract the estimated noise signal from the hydrophone data
def denoise_hydrophone_data(stream_in, noise):
    stream_out = Stream()
    for trace_in in stream_in:
        signal_in = trace_in.data
        num_pts_signal = len(signal_in)
        num_pts_noise = len(noise)

        if len(signal_in) < num_pts_noise:
            raise ValueError("The length of the noise signal is longer than the hydrophone data!")
        else:
            num_win = int(num_pts_signal / num_pts_noise)
            for i in range(num_win):
                signal_in[i * num_pts_noise:(i + 1) * num_pts_noise] -= noise

        trace_out = trace_in.copy()
        trace_out.data = signal_in
        stream_out.append(trace_out)

    return stream_out
# Inputs
# Command line arguments
parser = ArgumentParser(description = "Estimate the electrical noise signal in the hydrophone data through stacking large number of time windows")
parser.add_argument("--station", type = str, help = "Hydrophone station to process")
parser.add_argument("--window_length", type = float, help = "Length of the window in seconds for noise estimation")
parser.add_argument("--day", type = str, help = "Day to process")
parser.add_argument("--fraction_of_day", type = float, help = "Fraction of the day for noise estimation")
parser.add_argument("--amp_threshold", type = float, help = "Amplitude threshold for excluding a time window")

# Constants
window_length_stft = 60.0 # Length of the window in seconds for computing the STFT
panel_width = 10
panel_height = 5.0

min_db = -30.0
max_db = 15.0

# Parse the command line arguments
args = parser.parse_args()
station = args.station
window_length_noise = args.window_length
day = args.day
frac = args.fraction_of_day
amp_threshold = args.amp_threshold

# Print the command line arguments
print(f"Station: {station}")
print(f"Window length: {window_length_noise:.0f} s")
print(f"Day: {day}")
print(f"Amplitude threshold: {amp_threshold}")
print(f"Fraction of the day: {frac}")
print(f"Amplitude threshold: {amp_threshold:.0f}")

# Read the hydrophone data
print("Reading the hydrophone data...")
stream_in = read_and_process_day_long_hydro_waveforms(day, stations = station)
starttime_day = stream_in[0].stats.starttime

# Compute the STFT of the original hydrophone data
stream_stft_in = get_stream_stft(stream_in, window_length = window_length_stft)

# Denoise each fraction of the day-long hydrophone data
num_frac = int(1 / frac)

print(f"Number of fractions: {num_frac}")
if num_frac == 1:
    print("The entire day is processed as a single fraction")

    # Estimate the noise signal
    print("Estimating the noise signal...")
    noise = estimate_noise_signal(stream_in, window_length_noise, amp_threshold)

    # Subtract the noise signal from the hydrophone data
    print("Denoising the hydrophone data...")
    stream_out = denoise_hydrophone_data(stream_in, noise)

    # Compute the STFT of the denoised hydrophone data
    print("Computing the STFT of the denoised hydrophone data...")
    stream_stft_out = get_stream_stft(stream_out, window_length = window_length_stft)
else:
    stream_stft_out = StreamSTFT()
    for i in range(num_frac):
        print(f"Processing fraction: {i + 1}")

        # Slice the stream into a fraction of the day
        print("Slicing the stream into a fraction of the day...")
        frac_start = i * frac
        frac_end = (i + 1) * frac
        stream_frac = stream_in.slice(starttime_day + frac_start * num_seconds_in_day, starttime_day + frac_end * num_seconds_in_day)

        # Estimate the noise signal
        print("Estimating the noise signal...")
        noise = estimate_noise_signal(stream_frac, window_length_noise, amp_threshold)

        # Subtract the noise signal from the hydrophone data
        print("Denoising the hydrophone data...")
        stream_out = denoise_hydrophone_data(stream_frac, noise)

        # Compute the STFT of the denoised hydrophone data
        print("Computing the STFT of the denoised hydrophone data...")
        stream_stft_frac_out = get_stream_stft(stream_out, window_length = window_length_stft)
        stream_stft_out.extend(stream_stft_frac_out)

    # Stitch the STFTs of the denoised hydrophone data
    print("Stitching the STFTs of the denoised hydrophone data...")
    stream_stft_out.stitch()
    
# Plotting
num_loc = len(stream_stft_out)
stream_stft_in.to_db()
stream_stft_out.to_db()

print("Plotting the spectrograms before and after denoising...")
fig, axes = subplots(num_loc, 2, figsize = (panel_width, panel_height * num_loc), sharex = True, sharey = True)
for i in range(num_loc):
    ax_in = axes[i, 0]

    trace_stft_in = stream_stft_in[i]
    psd_mat_in = trace_stft_in.psd_mat
    freqax_in = trace_stft_in.freqs
    timeax_in = trace_stft_in.times

    ax_in.pcolormesh(timeax_in, freqax_in, psd_mat_in, shading = "auto", vmin = min_db, vmax = max_db,
                    cmap = "inferno")

    ax_out = axes[i, 1]

    trace_stft_out = stream_stft_out[i]
    psd_mat_out = trace_stft_out.psd_mat
    freqax_out = trace_stft_out.freqs
    timeax_out = trace_stft_out.times

    ax_out.pcolormesh(timeax_out, freqax_out, psd_mat_out, shading = "auto", vmin = min_db, vmax = max_db, 
                      cmap = "inferno")


# Save the figure
print("Saving the figure...")
figname = f"test_hydro_denoise_{station}_{day}_amp_thresh{amp_threshold:.0f}_frac{frac:.2f}.png"
save_figure(fig, figname)


