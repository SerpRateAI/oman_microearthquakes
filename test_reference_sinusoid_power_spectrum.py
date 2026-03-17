from numpy import sin, pi, linspace, amax
from scipy.signal.windows import dpss
from utils_mt import mt_autospec
from utils_basic import power2db

### Inputs ###
freq = 150.0
amplitude = 1e7
sampling_rate = 1000.0
duration = 300.0
nw = 3

### Generate the reference sinusoid ###
timeax = linspace(0, duration, int(duration * sampling_rate))
signal = amplitude * sin(2 * pi * freq * timeax)

### Compute the power spectrum ###
dpss_mat, ratio_vec = dpss(len(signal), nw, 2 * nw - 1, return_ratios=True)
aspec_params = mt_autospec(signal, dpss_mat, ratio_vec, sampling_rate)
freqax = aspec_params.freqax
aspec = aspec_params.aspec
aspec_db = power2db(aspec)

### Plot the power spectrum ###
print(f"The maximum amplitude of the power spectrum is {amax(aspec_db)} dB.")
