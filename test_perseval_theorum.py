from numpy import arange, sin, pi, sum, linspace, abs, square, sqrt, log10, log2
from scipy.signal.windows import hann
from numpy.fft import fft

# Parameters
fs = 1000.0                 # Sampling frequency (Hz)
dt = 1 / fs                # Sampling interval (s)
num_samples = 1024                   # Number of samples
t = arange(num_samples) * dt
duration = num_samples * dt                 # Total duration

# Example signal: sine wave at 10 Hz
freq = 10.0
amp = 1e7
signal = amp * sin(2 * pi * freq * t)

# Energy in the time domain (correct calculation)
e_time = sum(signal ** 2) * dt

# Apply window to signal
window = hann(num_samples)
signal_win = signal * window
window_power = sum(window ** 2)

# FFT computation (one-sided)
signal_fft = fft(signal_win)[:num_samples//2 + 1]

# Frequency vector
freqs = linspace(0, fs / 2, num_samples // 2 + 1)

# Correctly normalized PSD calculation (one-sided)
psd = (abs(signal_fft) ** 2) / (fs * window_power)
psd[1:-1] *= 2  # Double the PSD except DC and Nyquist frequencies

# Frequency resolution (Hz)
df = fs / num_samples

# PSD integrated over frequency gives average power
avg_power_freq = sum(psd) * df

# Get the peak PSD
peak_psd = max(psd)

# Convert average power to total energy by multiplying by total duration
e_freq = avg_power_freq * duration

# Results
print(f"Time-domain energy:      {e_time:.6f}")
print(f"Frequency-domain energy: {e_freq:.6f}")
print(f"Energy ratio (time/freq): {e_time / e_freq:.3f}")
print(f"Peak PSD in dB: {10 * log10(peak_psd):.6f}")