"""
Compute the correlation between the stationary resonance frequency and the environment observables using the multi-taper method
"""

### Import the necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import pi
from pandas import Timedelta
from pandas import read_hdf, date_range, read_csv
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from utils_basic import NUM_SEONCDS_IN_DAY
from utils_basic import STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, SPECTROGRAM_DIR as dirname_spec
from utils_basic import get_baro_temp_data, get_tidal_strain_data, power2db
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_mt import mt_cspec
from utils_plot import save_figure

### Input arguments ###
# Command line inputs
parser = ArgumentParser()
parser.add_argument("--mode_name", type=str, default="PR02549", help="The name of the mode to compute the correlation for")
parser.add_argument("--time_interval", type=str, default="5min", help="The time interval to interpolate the time series to")
parser.add_argument("--nw", type=float, default=3.0, help="The time-half bandwidth product for the multitaper method")

parser.add_argument("--window_length", type=float, default=300.0, help="The window length for the spectrogram")
parser.add_argument("--overlap", type=float, default=0.0, help="The overlap for the spectrogram")
parser.add_argument("--min_prom", type=float, default=15.0, help="The minimum prominence for the spectral peaks in dB")
parser.add_argument("--min_rbw", type=float, default=15.0, help="The minimum reverse bandwidth for the spectral peaks in dB")
parser.add_argument("--max_mean_db", type=float, default=10.0, help="The maximum mean power for the spectral peaks in dB")

args = parser.parse_args()
mode_name = args.mode_name
time_interval = args.time_interval
nw = args.nw

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

# Constants
figwidth = 10.0
figheight = 10.0

gap = 0.15

base_mode_name = "PR02549"
base_mode_order = 2

min_freq = 0.5
max_freq = 4

freq_ax_ticks = [0.5, 1.0, 2.0, 3.0]
freq_ax_ticklabels = [f"{tick:.1f}" for tick in freq_ax_ticks]

min_db = -40.0
max_db = -5.0

linewidth = 1.5

fill_alpha = 0.3 # The alpha value for the fill between the upper and lower bounds of the coherence

title_fontsize = 12

panel_label_x = -0.1
panel_label_y = 1.02

panel_label_fontsize = 12

### Read the data ###

# Read the harmonic series data
filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order}.csv"
filepath = join(dirname_spec, filename)
harm_df = read_csv(filepath)

mode_order = harm_df[ harm_df["mode_name"] == mode_name ]["mode_order"].values[0]

# Read the barometric and temperature data
baro_temp_df = get_baro_temp_data()

# Read the tidal strain data
tidal_strain_df = get_tidal_strain_data()

# Read the frequency data
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_resonance_profile_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(dirname_spec, filename)
reson_df = read_hdf(filepath, key = "profile")

reson_df = reson_df[["time", "frequency"]]
reson_df.set_index("time", inplace = True)

### Define the common time range and resample interval ###
start_time = max(reson_df.index.min(), baro_temp_df.index.min())  # Latest start time
end_time = min(reson_df.index.max(), baro_temp_df.index.max())  # Earliest end time

# Create a new common time index at 5-minute intervals
common_time_index = date_range(start=start_time, end=end_time, freq=time_interval)

### Resample both DataFrames to the common time index ###
reson_df = reson_df.reindex(common_time_index).interpolate(method="linear")

baro_temp_df = baro_temp_df.asfreq(time_interval).interpolate(method="linear")
baro_temp_df = baro_temp_df.loc[common_time_index[0]:common_time_index[-1]]
offset = common_time_index[0] - baro_temp_df.index[0]
baro_temp_df.index = baro_temp_df.index + offset

baro_temp_df = baro_temp_df.reindex(common_time_index).interpolate(method="linear").ffill().bfill()

tidal_strain_df = tidal_strain_df.asfreq(time_interval).interpolate(method="linear")
tidal_strain_df = tidal_strain_df.loc[common_time_index[0]:common_time_index[-1]]
offset = common_time_index[0] - tidal_strain_df.index[0]
tidal_strain_df.index = tidal_strain_df.index + offset

tidal_strain_df = tidal_strain_df.reindex(common_time_index).interpolate(method="linear").ffill().bfill()

### Compute the cross-spectra ###
# Extract the frequency, temperature, and pressure data
freqs = reson_df["frequency"].values
temps = baro_temp_df["temperature"].values
pressures = baro_temp_df["pressure"].values
strains = tidal_strain_df["strain"].values
# Generate the tapers
num_pts = len(freqs)
num_tapers = int(2 * nw - 1)
taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = num_tapers, return_ratios = True)

# Conver the frequency to cpd
time_interval = Timedelta(time_interval)
time_int_in_days = time_interval.total_seconds() / NUM_SEONCDS_IN_DAY
sampling_rate = 1.0 / time_int_in_days

# Compute the cross-spectra
freqax, aspec_freq, aspec_temp, _, cohe_temp, phase_diff_temp, _, _, _, _, _, mt_cohe_uncer, mt_phase_diff_uncer = mt_cspec(freqs, temps, taper_mat, ratio_vec, sampling_rate, normalize=True)
print(freqax)

_, _, aspec_press, _, _, _ = mt_cspec(freqs, pressures, taper_mat, ratio_vec, sampling_rate, get_uncer = False, normalize = True)

_, _, aspec_strain, _, _, _ = mt_cspec(freqs, strains, taper_mat, ratio_vec, sampling_rate, get_uncer = False, normalize = True)

aspec_freq = power2db(aspec_freq)
aspec_temp = power2db(aspec_temp)
aspec_press = power2db(aspec_press)
aspec_strain = power2db(aspec_strain)

### Plot the auto- and cross-spectra ###
fig, axes = subplots(3, 1, figsize = (figwidth, figheight), sharex = True)

fig.subplots_adjust(hspace=gap)

# Plot the auto-spectra
ax_auto = axes[0]
ax_auto.plot(freqax, aspec_freq, color = "tab:orange", label = "Frequency", linewidth = linewidth)
ax_auto.plot(freqax, aspec_temp, color = "tab:blue", label = "Temperature", linewidth = linewidth)
ax_auto.plot(freqax, aspec_press, color = "gray", linestyle = "--", label = "Pressure", linewidth = linewidth)
ax_auto.plot(freqax, aspec_strain, color = "gray", linestyle = ":", label = "Tidal strain", linewidth = linewidth)

ax_auto.legend(loc = "upper left")
ax_auto.set_xscale("log")
ax_auto.set_xlim(min_freq, max_freq)
ax_auto.set_ylim(min_db, max_db)

ax_auto.xaxis.set_major_locator(FixedLocator(freq_ax_ticks))
ax_auto.xaxis.set_major_formatter(FixedFormatter(freq_ax_ticklabels))

ax_auto.xaxis.set_minor_locator(NullLocator())

ax_auto.set_ylabel("Normalized power (dB)")

# Add the panel label
ax_auto.text(panel_label_x, panel_label_y, "(a)", transform = ax_auto.transAxes, fontsize = panel_label_fontsize, fontweight = "bold")

# Add the title
ax_auto.set_title(f"Mode {mode_order:d} frequency and environmental variables", fontsize = title_fontsize, fontweight = "bold")

# Plot the cross-spectral coherence
ax_cohe = axes[1]
ax_cohe.plot(freqax, cohe_temp, color = "tab:purple", label = "Coherence", linewidth = linewidth)

# Fill the area between the upper and lower bounds of the coherence
ax_cohe.fill_between(freqax, cohe_temp + mt_cohe_uncer, cohe_temp - mt_cohe_uncer, color = "tab:purple", alpha = fill_alpha)

ax_cohe.set_xscale("log")
ax_cohe.set_xlim(min_freq, max_freq)
ax_cohe.set_ylim(0, 1)


ax_cohe.set_ylabel("Coherence")

# Add the title
ax_cohe.set_title(f"Mode {mode_order:d} frequency vs temperature, coherence", fontsize = title_fontsize, fontweight = "bold")

# Add the panel label
ax_cohe.text(panel_label_x, panel_label_y, "(b)", transform = ax_cohe.transAxes, fontsize = panel_label_fontsize, fontweight = "bold")

# Add the phase difference plot
ax_phase = axes[2]
ax_phase.plot(freqax, phase_diff_temp, color = "tab:green", label = "Phase Difference", linewidth = linewidth)

# Fill the area between the upper and lower bounds of the phase difference
ax_phase.fill_between(freqax, phase_diff_temp + mt_phase_diff_uncer, phase_diff_temp - mt_phase_diff_uncer, color = "tab:green", alpha = fill_alpha)

ax_phase.set_xscale("log")
ax_phase.set_xlim(min_freq, max_freq)
ax_phase.set_ylim(-pi, pi)

# Set the x-tick labels 
ax_phase.xaxis.set_major_locator(FixedLocator(freq_ax_ticks))
ax_phase.xaxis.set_major_formatter(FixedFormatter(freq_ax_ticklabels))

ax_phase.xaxis.set_minor_locator(NullLocator())

# Change the y axis ticks to -pi, -pi/2, 0, pi/2, pi
ax_phase.set_yticks([-pi, -pi/2, 0, pi/2, pi])
ax_phase.set_yticklabels(["-$\pi$", "-$\pi$/2", "0", "$\pi$/2", "$\pi$"])

# Add the axis labels
ax_phase.set_xlabel("Frequency (cpd)")
ax_phase.set_ylabel("Phase difference (rad)")

# Add the title
ax_phase.set_title(f"Mode {mode_order:d} frequency vs temperature, phase difference", fontsize = title_fontsize, fontweight = "bold")

# Add the panel label
ax_phase.text(panel_label_x, panel_label_y, "(c)", transform = ax_phase.transAxes, fontsize = panel_label_fontsize, fontweight = "bold")

# Save the figure
save_figure(fig, "stationary_resonance_freq_corr_w_env_obs.png")



















