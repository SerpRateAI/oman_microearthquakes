import numpy as np
from scipy.signal.windows import hann
from numpy.fft import fft

# Parameters
fs = 100.0                 # Sampling frequency (Hz)
dt = 1 / fs                # Sampling interval (s)
N = 1024                   # Number of samples
t = np.arange(N) * dt
T = N * dt                 # Total duration

# Example signal: sine wave at 10 Hz
freq = 10
signal = np.sin(2 * np.pi * freq * t)

# Energy in the time domain (correct calculation)
E_time = np.sum(signal ** 2) * dt

# Apply window to signal
window = hann(N)
signal_win = signal * window
window_power = np.sum(window ** 2)

# FFT computation (one-sided)
signal_fft = fft(signal_win)[:N//2 + 1]

# Frequency vector
freqs = np.linspace(0, fs / 2, N // 2 + 1)

# Correctly normalized PSD calculation (one-sided)
PSD = (np.abs(signal_fft) ** 2) / (fs * window_power)
PSD[1:-1] *= 2  # Double the PSD except DC and Nyquist frequencies

# Frequency resolution
df = fs / N

# PSD integrated over frequency gives average power
avg_power_freq = np.sum(PSD) * df

# Convert average power to total energy by multiplying by total duration
E_freq = avg_power_freq * T

# Results
print(f"Time-domain energy:      {E_time:.6f}")
print(f"Frequency-domain energy: {E_freq:.6f}")
print(f"Energy ratio (time/freq): {E_time / E_freq:.3f}")