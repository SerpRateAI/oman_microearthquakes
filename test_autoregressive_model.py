"""
This script is used to test using autoregressive model to fit the stationary-resonance time series.
"""

###
# Import necessary libraries
###
from argparse import ArgumentParser
from numpy import angle, abs, log, amax, pi
from os.path import join
from pandas import Timedelta
from pandas import read_hdf
from statsmodels.tsa.ar_model import AutoReg
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as dirpath_spec, GEO_COMPONENTS as components, SAMPLING_RATE as sampling_rate
from utils_plot import get_geo_component_color, save_figure
from utils_spec import get_fft_psd, get_ar_psd, get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_preproc import read_and_process_windowed_geo_waveforms

###
# Input parameters
###

parser = ArgumentParser()
parser.add_argument("--station", type=str, default="A01", help="Station name")
parser.add_argument("--mode_name", type=str, default="PR03822", help="Model name")
parser.add_argument("--bandwidth", type=float, default=1.0, help="Bandwidth in Hz")
parser.add_argument("--order", type=int, default=2, help="Order of the autoregressive model")

parser.add_argument("--window_length", type=float, default=300.0, help="Window length in seconds")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum RBW")
parser.add_argument("--max_mean_db", type=float, default=10.0, help="Maximum mean DB")

parser.add_argument("--color_model", type=str, default="crimson", help="Color of the model")

args = parser.parse_args()

station = args.station
mode_name = args.mode_name
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db
bandwidth = args.bandwidth
order = args.order

color_model = args.color_model

###
# Load the data
###

# Load the spectral peaks
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"

filepath = join(dirpath_spec, filename)
peak_df = read_hdf(filepath, key="properties")

# Find the time window with the highest power for the station
peak_df = peak_df.loc[peak_df["station"] == station]
peak_df.reset_index(drop=True, inplace=True)

center_time = peak_df.loc[peak_df["total_power"].argmax(), "time"]
freq = peak_df.loc[peak_df["total_power"].argmax(), "frequency"]

# Load the time series while filtering around the frequency
start_time = center_time - Timedelta(seconds=window_length / 2)
end_time = center_time + Timedelta(seconds=window_length / 2)
print(f"Loading the data of {station} between {start_time} and {end_time}")
print(f"Filtering around {freq} Hz with a bandwidth of {bandwidth} Hz")

min_freq = freq - bandwidth / 2.0
max_freq = freq + bandwidth / 2.0
stream = read_and_process_windowed_geo_waveforms(stations = station, starttime = start_time, endtime = end_time,
                                                  filter = True, filter_type = "butter",
                                                  min_freq = min_freq, max_freq = max_freq, corners = 2)

###
# Fit the autoregressive model
###

fig, axs = subplots(3, 1, figsize=(10, 10), sharex=True)
# Work on each component
for i, component in enumerate(components):
    print(f"Working on {component} component")

    # Get the time series
    trace = stream.select(component=component)[0]
    data = trace.data
    std_data = data.std()

    print(f"The standard deviation of the data is {std_data}")

    # Fit the autoregressive model
    model = AutoReg(data, lags=order)
    results = model.fit()

    print(results.summary())
    print(results.params)

    # Get the frequency of the roots of the characteristic polynomial
    root = results.roots[0]
    freq_root = angle(root) / (2.0 * pi) * sampling_rate
    print(f"The frequency of the root of the characteristic polynomial is {freq_root}")

    # Compute the quality factor from the roots of the characteristic polynomial
    root = results.roots[0]
    qf = abs(angle(root) / (2.0 * log(abs(root))))
    print(f"The quality factor of {component} component is {qf}")

    # Compute the standard deviation of the residuals
    residuals = results.resid
    std_residuals = residuals.std()
    print(f"The standard deviation of the residuals of {component} component is {std_residuals}")

    # Compute the power spectral densities of the data and the model
    freqax_data, psd_data = get_fft_psd(data, sampling_rate, normalize = True)
    psd_data = psd_data[(freqax_data >= min_freq) & (freqax_data <= max_freq)]
    freqax_data = freqax_data[(freqax_data >= min_freq) & (freqax_data <= max_freq)]
    psd_data = psd_data / amax(psd_data)

    # Compute the model PSD
    num_pts = len(data)
    freqax_model, psd_model = get_ar_psd(results, num_pts, sampling_rate)
    print(f"The minimum of the model frequency axis is {freqax_model.min()}")
    print(f"The maximum of the model frequency axis is {freqax_model.max()}")
    print(f"The frequency of the maximum of the model PSD is {freqax_model[psd_model.argmax()]}")

    psd_model = psd_model / amax(psd_model)
    psd_model = psd_model[(freqax_model >= min_freq) & (freqax_model <= max_freq)]
    freqax_model = freqax_model[(freqax_model >= min_freq) & (freqax_model <= max_freq)]

    # Plot the PSDs
    color = get_geo_component_color(component)
    axs[i].plot(freqax_data, psd_data, color=color, label="Data")
    axs[i].plot(freqax_model, psd_model, color=color_model, label="Model")

    axs[i].set_xlabel("Frequency (Hz)")
    axs[i].set_ylabel("PSD")

    if i == 0:
        axs[i].legend()

    axs[i].set_xlim(min_freq, max_freq)

# Save the figure
filename = f"ar_model_example_{station}_{mode_name}.png"
save_figure(fig, filename)





