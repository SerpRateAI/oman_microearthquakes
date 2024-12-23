# Plot the frequency response of a peak filter

# Imports
from numpy import angle, log10, maximum
from scipy.signal import butter, freqz, iirpeak
from matplotlib.pyplot import subplots

from utils_plot import save_figure

# Inputs
center_freq = 38.22  # Center frequency of the bandpass filter (Hz)
qf = 800.0  # Quality factor of the bandpass filter

# Sampling frequency (Hz)
sampling_rate = 1000  # Example sampling frequency, you can adjust this value as needed


# Design the digital Butterworth bandpass filter
b, a = iirpeak(center_freq, qf, sampling_rate)

# Calculate the frequency response
w, h = freqz(b, a, worN=80000, fs=sampling_rate)

# Plot the frequency response
fig, axes = subplots(2, 1, figsize=(10, 10), sharex=True)


# Plot the magnitude response
ax = axes[0]
ax.plot(w, 20 * log10(maximum(abs(h), 1e-5)))  # Convert to dB scale

ax.set_title('Amplitude response', fontsize = 14, fontweight = 'bold')
ax.set_ylabel('Amplitude [dB]')

# Plot the phase response
ax = axes[1]
ax.plot(w, angle(h))

ax.set_ylabel('Phase [radians]')

ax.set_title('Phase response', fontsize = 14, fontweight = 'bold')
ax.set_xlabel('Frequency [Hz]')
ax.set_xlim(center_freq - 10 / qf * center_freq, center_freq + 10 / qf * center_freq)

# Save the figure
save_figure(fig, 'peak_filter_freq_response.png')