# Test getting the polarization from the 3-C Fourier coefficients

# Imports
from numpy import angle, abs, linspace, imag, cos, deg2rad, pi, real
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann
from matplotlib.pyplot import subplots

from utils_pol import get_pol_from_fourier_coeffs
from utils_plot import plot_pm_projections, save_figure

# Inputs
numpts = 2000
sampling_interval = 1e-3
freq = 20.0

amp_z = 2.0
amp_1 = 1.0
amp_2 = 1.0

phi_z = 90.0
phi_1 = 0.0
phi_2 = 45.0

linewidth = 1.0

# Generate the 3-C signals
timeax = linspace(0, (numpts-1)*sampling_interval, numpts)
signal_z = amp_z * cos(2 * pi * freq * timeax + deg2rad(phi_z))
signal_1 = amp_1 * cos(2 * pi * freq * timeax + deg2rad(phi_1))
signal_2 = amp_2 * cos(2 * pi * freq * timeax + deg2rad(phi_2))

# Get the Fourier coefficients at the frequency of interest
signal_z = signal_z * hann(numpts)
signal_1 = signal_1 * hann(numpts)
signal_2 = signal_2 * hann(numpts)

spec_z = fft(signal_z)
spec_1 = fft(signal_1)
spec_2 = fft(signal_2)

freqax = fftfreq(numpts, d = sampling_interval)
i_freq = abs(freqax - freq).argmin()

coeff_z = spec_z[i_freq]
coeff_1 = spec_1[i_freq]
coeff_2 = spec_2[i_freq]

print(angle(coeff_z, deg = True))
print(angle(coeff_1, deg = True))
print(angle(coeff_2, deg = True))

# Get the polarization
pol_vec, strike_major, dip_major, strike_minor, dip_minor, ellip = get_pol_from_fourier_coeffs(coeff_z, coeff_1, coeff_2)
print(pol_vec)

print(f"Major axis strike: {strike_major} deg")
print(f"Major axis dip: {dip_major} deg")
print(f"Minor axis strike: {strike_minor} deg")
print(f"Minor axis dip: {dip_minor} deg")
print(f"Ellipticity: {ellip}")

# Plot the the projections of the particle motion on the three planes
fig, axes = plot_pm_projections(signal_1, signal_2, signal_z)

# Add the projections of the major and minor axes on the three planes
pol_vec_real = real(pol_vec)
pol_vec_imag = imag(pol_vec)

axes[0].plot([-pol_vec_real[0], pol_vec_real[0]], [-pol_vec_real[1], pol_vec_real[1]], color = "crimson", linewidth = linewidth, linestyle = "--")
axes[1].plot([-pol_vec_real[0], pol_vec_real[0]], [-pol_vec_real[2], pol_vec_real[2]], color = "crimson", linewidth = linewidth, linestyle = "--")
axes[2].plot([-pol_vec_real[1], pol_vec_real[1]], [-pol_vec_real[2], pol_vec_real[2]], color = "crimson", linewidth = linewidth, linestyle = "--")

axes[0].plot([-pol_vec_imag[0], pol_vec_imag[0]], [-pol_vec_imag[1], pol_vec_imag[1]], color = "crimson", linewidth = linewidth, linestyle = "--")
axes[1].plot([-pol_vec_imag[0], pol_vec_imag[0]], [-pol_vec_imag[2], pol_vec_imag[2]], color = "crimson", linewidth = linewidth, linestyle = "--")
axes[2].plot([-pol_vec_imag[1], pol_vec_imag[1]], [-pol_vec_imag[2], pol_vec_imag[2]], color = "crimson", linewidth = linewidth, linestyle = "--")

# Save the plot
figname = "pol_from_fourier_coeffs.png"
save_figure(fig, figname)