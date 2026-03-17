"""
Plot the multitaper autospectrum of a segment of power generator signal and a segment of the tremor signal
"""

from os.path import join
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots

from argparse import ArgumentParser
from pandas import read_csv, DataFrame, concat, Timedelta
from numpy import nan, exp, log10, pi
from scipy.interpolate import interp1d
from pandas import read_hdf
from utils_basic import SPECTROGRAM_DIR as dirpath_spec
from utils_basic import power2db, str2timestamp
from utils_preproc import read_and_process_windowed_karst_waveforms, read_and_process_windowed_geo_waveforms
from utils_mt import mt_autospec
from utils_spec import get_spec_peak_file_suffix, get_spectrogram_file_suffix
from utils_plot import format_freq_xlabels, format_db_ylabels, save_figure

# Parse the command line arguments
parser = ArgumentParser()
parser.add_argument("--quality_factor", type=float, help="Quality factor of the propagation medium", default=30.0)
parser.add_argument("--distance", type=float, help="Distance between the generator and the tremor station (m)", default=3000.0)
parser.add_argument("--velocity", type=float, help="Velocity of the wave in the propagation medium (m/s)", default=6000.0)
parser.add_argument("--station1_generator", type=str, help="Generator station to plot")
parser.add_argument("--station2_generator", type=str, help="Generator station to plot")
parser.add_argument("--station_tremor", type=str, help="Tremor station to plot")
parser.add_argument("--starttime_generator", type=str, help="Start time of the generator")
parser.add_argument("--starttime_tremor", type=str, help="Start time of the tremor")
parser.add_argument("--overlap", type=float, help="Overlap percentage", default=0.0)
parser.add_argument("--min_prom", type=float, help="Minimum prominence for peak detection", default=15.0)
parser.add_argument("--min_rbw", type=float, help="Minimum reverse bandwidth for peak detection", default=15.0)
parser.add_argument("--max_mean_db", type=float, help="Maximum mean db for peak detection", default=15.0)
parser.add_argument("--duration", type=float, help="Duration in seconds", default=300.0)
parser.add_argument("--nw", type=float, help="Time-bandwidth product", default=3.0)

parser.add_argument("--figwidth", type=float, help="Figure width", default=10.0)
parser.add_argument("--figheight", type=float, help="Figure height", default=5.0)
parser.add_argument("--min_db", type=float, help="Minimum power in dB", default=-20.0)
parser.add_argument("--max_db", type=float, help="Minimum power in dB", default=130.0)
parser.add_argument("--max_freq", type=float, help="Minimum power in dB", default=200.0)

args = parser.parse_args()

quality_factor = args.quality_factor
distance = args.distance
velocity = args.velocity

station1_generator = args.station1_generator
station2_generator = args.station2_generator
station_tremor = args.station_tremor
starttime_generator = args.starttime_generator
starttime_tremor = args.starttime_tremor
duration = args.duration
nw = args.nw

figwidth = args.figwidth
figheight = args.figheight
min_db = args.min_db
max_db = args.max_db
max_freq = args.max_freq

min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Read the generator data
print("Reading the generator waveforms...")
stream1 = read_and_process_windowed_karst_waveforms(stations=[station1_generator], starttime=starttime_generator, dur=duration)
stream2 = read_and_process_windowed_karst_waveforms(stations=[station2_generator], starttime=starttime_generator, dur=duration)

# Read the tremor mode frequencies
print(f"Getting the resonance frequencies of station {station_tremor}...")
filename = f"stationary_harmonic_series_PR02549_base2.csv"
filepath = join(dirpath_spec, filename)

harmonic_df = read_csv(filepath)
time_window = str2timestamp(starttime_tremor) + Timedelta(seconds = duration / 2)
suffix_spec = get_spectrogram_file_suffix(duration, 0.0)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
mode_freq_dfs = []
for mode_name in harmonic_df["mode_name"]:
    
    if mode_name.startswith("MH"):
        mode_freq_df = DataFrame({"mode_name": [mode_name], "frequency": [nan]})
    else:
        filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
        filepath = join(dirpath_spec, filename)

        resonance_df = read_hdf(filepath, key = "properties")

        freq = resonance_df.loc[(resonance_df["station"] == station_tremor) & (resonance_df["time"] == time_window), "frequency"].values[0]
        mode_freq_df = DataFrame({"mode_name": [mode_name], "frequency": [freq]})

    mode_freq_dfs.append(mode_freq_df)

mode_freq_df = concat(mode_freq_dfs, axis = 0)
mode_freq_df.reset_index(drop = True, inplace = True)

# Compute the autospectrum for generator Station 1
print("Computing the spectrum for Station 1...")
for i, trace in enumerate(stream1):
    waveform = trace.data
    sampling_rate = trace.stats.sampling_rate
    num_pts = len(waveform)

    taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)
    mt_aspec_params = mt_autospec(waveform, taper_mat, ratio_vec, sampling_rate)
    aspec = mt_aspec_params.aspec

    if i == 0:
        aspec1_generator_total = aspec
    else:
        aspec1_generator_total += aspec

aspec1_generator_total = power2db(aspec1_generator_total)

print("Computing the spectrum for Station 2...")
for i, trace in enumerate(stream2):
    waveform = trace.data
    sampling_rate = trace.stats.sampling_rate
    num_pts = len(waveform)

    taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)
    mt_aspec_params = mt_autospec(waveform, taper_mat, ratio_vec, sampling_rate)
    aspec = mt_aspec_params.aspec

    if i == 0:
        aspec2_generator_total = aspec
    else:
        aspec2_generator_total += aspec

aspec2_generator_total = power2db(aspec2_generator_total)

freqax_generator = mt_aspec_params.freqax

# Read the tremor data
print("Reading the tremor waveforms...")

stream_tremor = read_and_process_windowed_geo_waveforms(stations=station_tremor, starttime=starttime_tremor, dur=duration)

# Compute the autospectrum for tremor station
for i, trace in enumerate(stream_tremor):
    waveform = trace.data
    sampling_rate = trace.stats.sampling_rate
    num_pts = len(waveform)

    taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)
    mt_aspec_params = mt_autospec(waveform, taper_mat, ratio_vec, sampling_rate)
    aspec = mt_aspec_params.aspec

    if i == 0:
        aspec_tremor_total = aspec
    else:
        aspec_tremor_total += aspec

aspec_tremor_total = power2db(aspec_tremor_total)
freqax_tremor = mt_aspec_params.freqax

# Get the tremor mode powers at the mode frequencies
for i, row in mode_freq_df.iterrows():
    mode_name = row["mode_name"]
    frequency = row["frequency"]
    power = interp1d(freqax_tremor, aspec_tremor_total)(frequency)
    power_extrapolated = power + 20.0 * log10(exp(2 * pi * frequency * distance / quality_factor / velocity / 2.0))
    mode_freq_df.loc[i, "power"] = power
    mode_freq_df.loc[i, "power_extrapolated"] = power_extrapolated

# Plot the spectra
print("Plotting the spectra...")
fig, ax = subplots(1, 1, figsize = (figwidth, figheight))

# Plot the generator spectra
ax.plot(freqax_generator, aspec1_generator_total, color = "deepskyblue", label = f"Generator, {station1_generator}")
ax.plot(freqax_generator, aspec2_generator_total, color = "lightskyblue", label = f"Generator, {station2_generator}")

# Format the axes
ax.set_xlim(0.0, max_freq)
ax.set_ylim(min_db, max_db)
format_freq_xlabels(ax,
                    plot_axis_label=True, plot_tick_label=True,
                    major_tick_spacing=50.0, num_minor_ticks=5)

format_db_ylabels(ax,
                  plot_axis_label=True, plot_tick_label=True,
                  major_tick_spacing=20, num_minor_ticks=5)

# Plot the tremor spectrum
ax.plot(freqax_tremor, aspec_tremor_total, color = "darkorange", label = f"MBO Tremor, {station_tremor}")

# Plot the extrapolated tremor mode powers
ax.scatter(mode_freq_df["frequency"], mode_freq_df["power_extrapolated"], color = "darkorange", label = "Extrapolated mode powers", marker = "D", edgecolors = "black", zorder = 10)
for _, row in mode_freq_df.iterrows():
    power_extrapolated = row["power_extrapolated"]
    power = row["power"]
    frequency = row["frequency"]
    ax.vlines(frequency, power, power_extrapolated, color = "lightgray", linewidth = 1.0, linestyles = "--", zorder = 5)

ax.set_title(f"Generator and MBO tremor spectra", fontsize = 14, fontweight = "bold")

ax.legend(framealpha = 1.0, 
          facecolor = "white",
          edgecolor = "black",
          fontsize = 10)

# Save the figure
save_figure(fig, f"jgr_generator_spectra_comparison.png")