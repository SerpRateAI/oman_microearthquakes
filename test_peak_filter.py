# Test peak filtering

# Import
from os.path import join
from argparse import ArgumentParser
from pandas import Timestamp
from pandas import read_hdf, to_datetime
from matplotlib.pyplot import subplots

from utils_basic import GEO_COMPONENTS as components
from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import str2timestamp, time2suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_spec import get_stream_fft
from utils_plot import get_geo_component_color, save_figure

# Inputs
parser = ArgumentParser(description="Input parameters for testing peak filtering.")
parser.add_argument("--mode_name", type=str, help="Name of the mode.")
parser.add_argument("--starttime", type=str, help="Start time of the time window.")
parser.add_argument("--endtime", type=str, help="End time of the time window.")
parser.add_argument("--station", type=str, help="Station name.")

args = parser.parse_args()

# Parse the inputs
mode_name = args.mode_name
starttime = str2timestamp(args.starttime)
endtime = str2timestamp(args.endtime)
station = args.station

# Other inputs
freq_window = 1.0
linewidth = 0.5

# Print the inputs
print("###")
print(f"Testing peak filtering for the time window from {starttime} to {endtime}.")
print("###")

# Read the data
# Read the stationary resonance information
print("Reading the stationary resonance information...")
filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
resonance_df = read_hdf(join(indir, filename), key = "properties")

centertime = starttime + (endtime - starttime) / 2

# Get the resonance frequency and quality factor of the time window
print("Getting the resonance frequency and quality factor of the time window...")
freq_reson = resonance_df.loc[centertime, "frequency"]
qf_filter = resonance_df.loc[centertime, "mean_quality_factor"]

# Without filtering
print("Reading the data without filtering...")
stream = read_and_process_windowed_geo_waveforms(starttime,
                                                 endtime = endtime,
                                                 stations = station)

# With peak filtering
print("Reading the data with peak filtering...")
stream_filt = read_and_process_windowed_geo_waveforms(starttime,
                                                      endtime = endtime,
                                                      stations = station,
                                                      filter = True, filter_type = "peak",
                                                      freq = freq_reson, qf = qf_filter)

# Compute the Fourier transform
print("Computing the Fourier transform...")
stream_fft = get_stream_fft(stream)
stream_fft.to_db()

stream_filt_fft = get_stream_fft(stream_filt)
stream_filt_fft.to_db()

# Plot the results
print("Plotting the PSD spectra of the waveforms with and without filtering...")
fig, axes = subplots(3, 2, figsize = (12, 8))

# Full spectra
for i, component in enumerate(components):
    ax = axes[i, 0]
    color = get_geo_component_color(component)

    trace_fft = stream_fft.select(components = component)[0]
    psd_spec = trace_fft.psd
    freqs = trace_fft.freqs

    trace_filt_fft = stream_filt_fft.select(components = component)[0]
    psd_spec_filt = trace_filt_fft.psd
    freqs_filt = trace_filt_fft.freqs

    ax.plot(freqs, psd_spec, color = color, linewidth = linewidth, linestyle = ":")
    ax.plot(freqs_filt, psd_spec_filt, color = color, linewidth = linewidth)

# Spectra in the vicinity of the resonance
for i, component in enumerate(components):
    ax = axes[i, 1]
    color = get_geo_component_color(component)

    trace_fft = stream_fft.select(components = component)[0]
    psd_spec = trace_fft.psd
    freqs = trace_fft.freqs

    trace_filt_fft = stream_filt_fft.select(components = component)[0]
    psd_spec_filt = trace_filt_fft.psd
    freqs_filt = trace_filt_fft.freqs

    ax.plot(freqs, psd_spec, color = color, linewidth = linewidth, linestyle = ":")
    ax.plot(freqs_filt, psd_spec_filt, color = color, linewidth = linewidth)

    ax.set_xlim(freq_reson - freq_window / 2, freq_reson + freq_window / 2)

# Save the figure
print("Saving the figure...")
time_suffix = time2suffix(centertime)
figname = f"peak_filtering_test_{mode_name}_{time_suffix}_{station}.png"
save_figure(fig, figname)



    


