###### Test cepstrum ######

### Import modules ###
from numpy import abs, arange, zeros, cos, pi
from scipy.fft import fft, fftfreq, fftshift
from matplotlib.pyplot import subplots

from utils_spec import get_cepstrum
from utils_plot import save_figure

### Input parameters ###
# Synthetic data parameters
time_interval = 1e-2
dur = 100.0
max_spike_time = 8.0
time_int_comb = 2.0 # Time interval for the comb signal

freq0 = 0.25
nums_harmo = [2, 3, 4, 5, 6]

# Plotting
linewidth = 0.5

### Generate synthetic data ###
print("Generating synthetic data...")
timeax = arange(0.0, dur, time_interval)
data = zeros(len(timeax))

for num in nums_harmo:
    data += cos(2 * pi * freq0 * num * timeax)

# # Generate a comb signal
# time_spike = time_int_comb
# while time_spike < max_spike_time:
#     data[int(time_spike / time_interval)] = 1.0
#     time_spike += time_int_comb

### Compute the Fourier transform of the data ###
print("Computing the Fourier transform...")
data_fft = fft(data)
mag_data_fft = fftshift(abs(data_fft))
freqax = fftshift(fftfreq(len(data), time_interval))

### Compute the cepstrum ###
print("Computing the cepstrum...")
cepstrum = get_cepstrum(data)

### Plot the cepstrum alongside the data ###
print("Plotting the cepstrum...")
fig, axes = subplots(3, 1, figsize = (13, 6))

axes[0].plot(timeax, data, color = "black", linewidth = linewidth)
axes[0].set_ylabel("Amplitude")


axes[1].plot(freqax, mag_data_fft, color = "black", linewidth = linewidth)
axes[1].set_ylabel("Magnitude")
#axes[1].set_yscale("log")
axes[1].set_xlabel("Frequency (Hz)")

axes[2].plot(timeax, cepstrum, color = "black", linewidth = linewidth)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Cepstrum")
axes[2].set_ylim(0, 0.5)
axes[2].axvline(x = 1 / freq0, color = "crimson", linestyle = "--", linewidth = linewidth)

save_figure(fig, "cepstrum_example.png")