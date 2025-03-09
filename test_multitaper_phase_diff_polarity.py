"""
Test the polarity of the phase differences measured using multitaper cross-spectral analysis
"""

### Imports ###
from os.path import join
from numpy import array, cos, linspace, pi
from numpy.random import randn
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots

from utils_mt import mt_cspec
from utils_plot import save_figure

### Inputs ###
freq0 = 25.0
num_pts = 10001
sampling_int = 1e-3

std_noise = 0.5

nw = 3
num_taper =  2 * nw - 1

bandwidth = 5.0

### Compute the synthetic waveforms ###
starttime = 0.0
endtime = starttime + (num_pts - 1) * sampling_int

timeax = linspace(starttime, endtime, num_pts)

signal1 = cos(2 * pi * freq0 * timeax) + randn(num_pts) * std_noise
signal2 = cos(2 * pi * freq0 * timeax - pi / 2) + randn(num_pts) * std_noise

### Perform multitaper cross-spectral ###
dpss_mat, ratio_vec = dpss(num_pts, nw, num_taper, return_ratios=True)

sampling_rate = 1 / sampling_int
freqax, mt_aspec1, mt_aspec2, mt_trans, mt_cohe, mt_phase_diffs, mt_aspec1_lo, mt_aspec1_hi, mt_aspec2_lo, mt_aspec2_hi, mt_trans_uncers, mt_cohe_uncers, mt_phase_diff_uncers = mt_cspec(signal1, signal2, dpss_mat, ratio_vec, sampling_rate, verbose = False)

### Plot the results ###
fig, axes = subplots(3, 1, sharex = True)

axes[0].plot(freqax, mt_aspec1)
axes[0].plot(freqax, mt_aspec2)

axes[0].set_xlim(freq0 - bandwidth / 2, freq0 + bandwidth / 2)
axes[0].set_yscale("log")

axes[1].plot(freqax, mt_cohe)

axes[2].errorbar(freqax, mt_phase_diffs, yerr=mt_phase_diff_uncers, fmt="o", markersize=5, fillstyle="none", capsize=2, linewidth=0.5)
axes[2].set_ylim(-pi, pi)

axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Phase diff. (rad)")


### Save the figure ###
save_figure(fig, "test_multitaper_phase_diff_polarity.png")


