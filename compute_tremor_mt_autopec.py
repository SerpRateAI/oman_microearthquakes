"""
Compute the multitaper auto-spectra of a time window containing signals
"""

###
# Load the necessary modules
###

from os.path import join
from argparse import ArgumentParser
from numpy import mean, var, amax, sum, sqrt, ones, conjugate
from pandas import read_csv, DataFrame, Timestamp
from scipy.signal.windows import dpss
from scipy.fft import fft
from scipy.signal import detrend
from scipy.signal.windows import hann


from utils_basic import MT_DIR as dirpath_mt, GEO_COMPONENTS as components
from utils_basic import power2db
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_mt import mt_autospec


###
# Define the input parameters
###
parser = ArgumentParser()
parser.add_argument("--station", type=str, help="Station name.")
parser.add_argument("--start_time", type=str, help="Start time of the time window.")
parser.add_argument("--duration", type=float, help="Duration of the time window in seconds.")

parser.add_argument("--nw", type=float, help="Time-bandwidth parameter.", default = 3.0)

args = parser.parse_args()

station = args.station
start_time = args.start_time
duration = args.duration
nw = args.nw


###
# Load the data
###

stream = read_and_process_windowed_geo_waveforms(stations = station, starttime = start_time, dur = duration)

###
# Compute the multitaper auto-spectra
###

for i, component in enumerate(components):
    trace = stream.select(component = component)[0]
    data = trace.data
    # print(data.max())
    sampling_rate = trace.stats.sampling_rate
    num_pts = len(data)

    if i == 0:
        taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)

        # for j in range(taper_mat.shape[0]):
        #     print(sum(taper_mat[j, :] ** 2))

    mt_aspec_params = mt_autospec(data, taper_mat, ratio_vec, sampling_rate)

    freqax = mt_aspec_params.freqax
    aspec = mt_aspec_params.aspec

    if i == 0:
        aspec_total_mt = aspec
    else:
        aspec_total_mt += aspec
        
    # Compute the FFT PSD
    data = detrend(data)
    window = hann(num_pts)
    data_win = data * window
    data_fft = fft(data_win)

    aspec = abs(data_fft) ** 2
    aspec = aspec[:num_pts//2 + 1]
    aspec[1:-1] *= 2
    aspec /= sum(window ** 2)

    print(f"Checking the validity of the Perseval Theorem for the FFT auto-spectrum...")
    check = sum(aspec) / sum(data ** 2)
    print(f"Power ratio: {check}")

    aspec /= sampling_rate

    if i == 0:
        aspec_total_fft = aspec
    else:
        aspec_total_fft += aspec

aspec_total_mt = power2db(aspec_total_mt)
aspec_total_fft = power2db(aspec_total_fft)
print(amax(aspec_total_mt))
print(amax(aspec_total_fft))

###
# Save the results
###
output_df = DataFrame({"frequency": freqax, "aspec_total_mt": aspec_total_mt, "aspec_total_fft": aspec_total_fft})

start_time = Timestamp(start_time).strftime("%Y%m%d%H%M%S")
filename = f"tremor_mt_aspec_{station}_{start_time}_{duration:.0f}s.csv"
outpath = join(dirpath_mt, filename)
output_df.to_csv(outpath, index = False)
print(f"The multitaper auto-spectra have been saved to {outpath}.")











