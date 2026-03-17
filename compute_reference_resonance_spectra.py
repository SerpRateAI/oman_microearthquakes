"""
Compute the spectrum of a reference resonance 
"""

###
# Imports
###
from os.path import join
from argparse import ArgumentParser
from pandas import read_csv, DataFrame, Timestamp
from scipy.signal.windows import dpss
from numpy import arange, sin, pi, amax
from utils_mt import mt_autospec
from utils_basic import power2db

###
# Inputs
###
parser = ArgumentParser()
parser.add_argument("--freq", type=float, help="The frequency of the reference resonance", default=150.0)
parser.add_argument("--amplitude", type=float, help="The amplitude of the reference resonance", default=1e7)
parser.add_argument("--sampling_rate", type=float, help="The sampling rate in Hz", default=1000.0)
parser.add_argument("--window_length", type=float, help="The window length in seconds", default=300.0)
parser.add_argument("--nw", type=float, help="The time-bandwidth product", default=3.0)
args = parser.parse_args()

freq = args.freq
sampling_rate = args.sampling_rate
window_length = args.window_length
nw = args.nw
amplitude = args.amplitude

###
# Generate the reference resonance signal
###

timeax = arange(0, window_length, 1.0 / sampling_rate)
num_pts = len(timeax)
signal = amplitude * sin(2 * pi * freq * timeax)

###
# Compute the spectrum
###

# Generate the taper sequence matrix and the concentration ratio vector
taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)

# Compute the multitaper auto-spectra
mt_aspec_params = mt_autospec(signal, taper_mat, ratio_vec, sampling_rate)
freqax = mt_aspec_params.freqax
aspec = mt_aspec_params.aspec

# Convert the auto-spectra to decibels
aspec_db = power2db(aspec)

print(f"The maximum amplitude of the auto-spectra is {amax(aspec_db)} dB.")
