"""
Compute the correlation between the stationary resonance frequency and the environment observables using the multi-taper method
"""

### Import the necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import pi
from pandas import Timedelta
from pandas import read_hdf, date_range, to_datetime
from scipy.signal.windows import dpss
from matplotlib import colormaps
from matplotlib.pyplot import figure
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

from utils_basic import NUM_SEONCDS_IN_DAY
from utils_basic import STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, SPECTROGRAM_DIR as dirname_spec
from utils_basic import get_baro_temp_data, get_mode_order, power2db
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_mt import mt_cspec, mt_autospec
from utils_plot import format_phase_diff_ylabels, save_figure, add_zoom_effect

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
parser.add_argument("--max_mean_db", type=float, default=-10.0, help="The maximum mean power for the spectral peaks in dB")

parser.add_argument("--figwidth", type=float, default=12.0, help="The width of the figure")
parser.add_argument("--figheight", type=float, default=12.0, help="The height of the figure")
parser.add_argument("--margin_x", type=float, default=0.02, help="The margin for the x-axis")
parser.add_argument("--margin_y", type=float, default=0.02, help="The margin for the y-axis")
parser.add_argument("--hspace_major", type=float, default=0.08, help="The major hspace for the subplots")
parser.add_argument("--hspace_minor", type=float, default=0.05, help="The minor hspace for the subplots")

parser.add_argument("--min_freq_wide", type=float, default=0.02, help="The minimum frequency to plot in CPD for the wide-band plot")
parser.add_argument("--max_freq_wide", type=float, default=4.0, help="The maximum frequency to plot in CPD for the wide-band plot")
parser.add_argument("--min_freq_narrow", type=float, default=0.5, help="The minimum frequency to plot in CPD for the narrow-band plot")
parser.add_argument("--max_freq_narrow", type=float, default=2.0, help="The maximum frequency to plot in CPD for the narrow-band plot")
parser.add_argument("--freq_highlight", type=float, default=1.0, help="The frequency to highlight in the plots")
parser.add_argument("--min_db", type=float, default=-10.0, help="The minimum power to plot in dB for the wide-band plot")
parser.add_argument("--max_db", type=float, default=40.0, help="The maximum power to plot in dB for the wide-band plot")
parser.add_argument("--freq_ticks_wide", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0], help="The frequency ticks for the wide-band plot")
parser.add_argument("--freq_ticks_narrow", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="The frequency ticks for the narrow-band plot")

parser.add_argument("--freq_color", type=str, default="tab:orange", help="The color for the frequency")
parser.add_argument("--temp_color", type=str, default="tab:blue", help="The color for the temperature")
parser.add_argument("--cohe_color", type=str, default="tab:green", help="The color for the coherence")
parser.add_argument("--phase_color", type=str, default="tab:purple", help="The color for the phase difference")
parser.add_argument("--highlight_color", type=str, default="crimson", help="The color for the highlighted lines")

parser.add_argument("--linewidth_var", type=float, default=1.0, help="The linewidth for plotting the variables")
parser.add_argument("--linewidth_highlight", type=float, default=2.0, help="The linewidth for plotting the highlighted lines")
parser.add_argument("--fill_alpha", type=float, default=0.1, help="The alpha value for the fill between the upper and lower bounds of the coherence")
parser.add_argument("--title_fontsize", type=float, default=14, help="The fontsize for the title")
parser.add_argument("--panel_label_x", type=float, default=-0.07, help="The x-coordinate for the panel label")
parser.add_argument("--panel_label_y", type=float, default=1.02, help="The y-coordinate for the panel label")
parser.add_argument("--panel_label_fontsize", type=float, default=14, help="The fontsize for the panel label")
parser.add_argument("--legend_fontsize", type=float, default=12, help="The fontsize for the legend")
parser.add_argument("--axis_label_fontsize", type=float, default=12, help="The fontsize for the axis labels")
parser.add_argument("--axis_tick_fontsize", type=float, default=10, help="The fontsize for the axis ticks")

args = parser.parse_args()
mode_name = args.mode_name
time_interval = args.time_interval
nw = args.nw

window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

figwidth = args.figwidth
figheight = args.figheight
margin_x = args.margin_x
margin_y = args.margin_y
hspace_major = args.hspace_major
hspace_minor = args.hspace_minor

min_freq_wide = args.min_freq_wide
max_freq_wide = args.max_freq_wide
min_freq_narrow = args.min_freq_narrow
max_freq_narrow = args.max_freq_narrow
min_db = args.min_db
max_db = args.max_db
freq_highlight = args.freq_highlight
linewidth_var = args.linewidth_var
linewidth_highlight = args.linewidth_highlight

fill_alpha = args.fill_alpha

title_fontsize = args.title_fontsize
panel_label_x = args.panel_label_x
panel_label_y = args.panel_label_y
panel_label_fontsize = args.panel_label_fontsize
legend_fontsize = args.legend_fontsize
axis_label_fontsize = args.axis_label_fontsize
axis_tick_fontsize = args.axis_tick_fontsize

freq_color = args.freq_color
temp_color = args.temp_color
cohe_color = args.cohe_color
phase_color = args.phase_color
highlight_color = args.highlight_color
freq_ticks_wide = args.freq_ticks_wide
freq_ticks_narrow = args.freq_ticks_narrow

freq_ticklabels_wide = [f"{tick:.2f}" for tick in freq_ticks_wide]
freq_ticklabels_narrow = [f"{tick:.2f}" for tick in freq_ticks_narrow]

### Read the data ###

# Read the barometric and temperature data
baro_temp_df = get_baro_temp_data()

# Read the frequency data
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

filename = f"stationary_resonance_profile_hydro_smooth_filter_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
filepath = join(dirname_spec, filename)
reson_df = read_hdf(filepath, key = "profile")

reson_df = reson_df[["time", "frequency"]]
reson_df["time"] = to_datetime(reson_df["time"])
reson_df.set_index("time", inplace = True)

### Define the common time range and resample interval ###
start_time = max(reson_df.index.min(), baro_temp_df.index.min())  # Latest start time
end_time = min(reson_df.index.max(), baro_temp_df.index.max())  # Earliest end time

# Create a new common time index at 5-minute intervals
common_time_index = date_range(start=start_time, end=end_time, freq=time_interval)

print(common_time_index)

### Resample both DataFrames to the common time index ###
# print(reson_df.head(50))
reson_df = reson_df.loc[start_time:end_time]
offset = start_time - reson_df.index[0]
reson_df.index = reson_df.index + offset
reson_df = reson_df.resample(time_interval, origin=start_time).interpolate("linear")
# print(reson_df.head(50))

baro_temp_df = baro_temp_df.asfreq(time_interval).interpolate(method="linear")
baro_temp_df = baro_temp_df.loc[start_time:end_time]
offset = start_time - baro_temp_df.index[0]
baro_temp_df.index = baro_temp_df.index + offset

### Compute the cross-spectra ###
# Extract the frequency, temperature, and pressure data
freqs = reson_df["frequency"].values
temps = baro_temp_df["temperature"].values

# Generate the tapers
num_pts = len(freqs)
num_tapers = int(2 * nw - 1)
taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = num_tapers, return_ratios = True)

# Conver the frequency to cpd
time_interval = Timedelta(time_interval)
time_int_in_days = time_interval.total_seconds() / NUM_SEONCDS_IN_DAY
sampling_rate = 1.0 / time_int_in_days

# Compute the cross-spectra between the frequency and temperature
mt_cspec_params = mt_cspec(freqs, temps, taper_mat, ratio_vec, sampling_rate, normalize=True)
freqax = mt_cspec_params.freqax
cohe_temp = mt_cspec_params.cohe
phase_diff_temp = mt_cspec_params.phase_diff
mt_cohe_uncer_temp = mt_cspec_params.cohe_uncer
mt_phase_diff_uncer_temp = mt_cspec_params.phase_diff_uncer

mt_phase_diff_uncer_press = mt_cspec_params.phase_diff_uncer

# Compute the auto-spectra
mt_aspec_params = mt_autospec(freqs, taper_mat, ratio_vec, sampling_rate, normalize=True)
aspec_freq = mt_aspec_params.aspec

mt_aspec_params = mt_autospec(temps, taper_mat, ratio_vec, sampling_rate, normalize=True)
aspec_temp = mt_aspec_params.aspec

# Convert the auto-spectra to dB
aspec_freq = power2db(aspec_freq)
aspec_temp = power2db(aspec_temp)

###
# Plotting
###

# Generate the figure
fig = figure(figsize = (figwidth, figheight))
mode_order = get_mode_order(mode_name)

# Compute the dimensions of the subplots
width_frac = 1 - 2 * margin_x
height_frac = (1 - 2 * margin_y - hspace_major - 2 * hspace_minor) / 4

###
# Plot the auto-spectra
###

# Add the subplots
ax_x = margin_x
ax_y = margin_y + hspace_major + 2 * hspace_minor + 3 * height_frac

ax_auto = fig.add_axes([ax_x, ax_y, width_frac, height_frac])
ax_auto.plot(freqax, aspec_freq, color = freq_color, label = "Frequency", linewidth = linewidth_var)
ax_auto.plot(freqax, aspec_temp, color = temp_color, label = "Temperature", linewidth = linewidth_var)

legend = ax_auto.legend(loc = "lower left", frameon = True, framealpha = 1.0, fontsize = legend_fontsize)
ax_auto.set_xscale("log")
ax_auto.set_xlim(min_freq_wide, max_freq_wide)
ax_auto.set_ylim(min_db, max_db)

ax_auto.xaxis.set_major_locator(FixedLocator(freq_ticks_wide))
ax_auto.xaxis.set_major_formatter(FixedFormatter(freq_ticklabels_wide))

ax_auto.xaxis.set_minor_locator(NullLocator())

ax_auto.set_ylabel("Normalized power (dB)", fontsize = axis_label_fontsize)

# Add the panel label
ax_auto.text(panel_label_x, panel_label_y, "(a)", transform = ax_auto.transAxes, fontsize = panel_label_fontsize, fontweight = "bold")

# Add the title
ax_auto.set_title(f"Mode {mode_order:d} frequency and air temperature", fontsize = title_fontsize, fontweight = "bold")

###
# Plot the cross-spectral coherence in the wide-band plot
###

# Add the subplots
ax_x = margin_x
ax_y = margin_y + hspace_major + hspace_minor + 2 * height_frac

ax_cohe_wide = fig.add_axes([ax_x, ax_y, width_frac, height_frac])

ax_cohe_wide.plot(freqax, cohe_temp, color = cohe_color, label = "Temperature", linewidth = linewidth_var)

# Fill the area between the upper and lower bounds of the coherence
ax_cohe_wide.fill_between(freqax, cohe_temp + mt_cohe_uncer_temp, cohe_temp - mt_cohe_uncer_temp, color = cohe_color, alpha = fill_alpha)

# Set the x-scale to log
ax_cohe_wide.set_xscale("log")

# Set the x-limits
ax_cohe_wide.set_xlim(min_freq_wide, max_freq_wide)

# Set the x-tick labels 
ax_cohe_wide.xaxis.set_major_locator(FixedLocator(freq_ticks_wide))
ax_cohe_wide.xaxis.set_major_formatter(FixedFormatter(freq_ticklabels_wide))

ax_cohe_wide.xaxis.set_minor_locator(NullLocator())

# Set the y-limits
ax_cohe_wide.set_ylim(0, 1)

ax_cohe_wide.set_ylabel("Coherence", fontsize = axis_label_fontsize)

# Add the axis labels
ax_cohe_wide.set_xlabel("Frequency (CPD)", fontsize = axis_label_fontsize)

# Add the title
ax_cohe_wide.set_title(f"Frequency vs temperature, coherence", fontsize = title_fontsize, fontweight = "bold")

# Add the panel label
ax_cohe_wide.text(panel_label_x, panel_label_y, "(b)", transform = ax_cohe_wide.transAxes, fontsize = panel_label_fontsize, fontweight = "bold")

###
# Plot the cross-spectral coherence in the narrow-band plot
###

# Add the subplots
ax_x = margin_x
ax_y = margin_y + hspace_minor + height_frac

ax_cohe_narrow = fig.add_axes([ax_x, ax_y, width_frac, height_frac])

# Plot the coherence
ax_cohe_narrow.plot(freqax, cohe_temp, color = cohe_color, label = "Temperature", linewidth = linewidth_var)

# Plot the highlighted frequency
ax_cohe_narrow.axvline(freq_highlight, color = highlight_color, linewidth = linewidth_highlight, linestyle = "--")

# Fill the area between the upper and lower bounds of the phase difference
ax_cohe_narrow.fill_between(freqax, cohe_temp + mt_cohe_uncer_temp, cohe_temp - mt_cohe_uncer_temp, color = cohe_color, alpha = fill_alpha)

# Set the x-scale to log
ax_cohe_narrow.set_xscale("log")

# Set the x-limits
ax_cohe_narrow.set_xlim(min_freq_narrow, max_freq_narrow)

# Set the y-limits
ax_cohe_narrow.set_ylim(0, 1)

# Set the x-tick labels 
ax_cohe_narrow.xaxis.set_major_locator(FixedLocator(freq_ticks_narrow))
ax_cohe_narrow.xaxis.set_major_formatter(FixedFormatter(freq_ticklabels_narrow))

ax_cohe_narrow.xaxis.set_minor_locator(NullLocator())

ax_cohe_narrow.set_ylabel("Squared coherence", fontsize = axis_label_fontsize)


###
# Add the zoom-in effect between the wide-band and narrow-band plots
###

# Add the zoom-in effect
prop_lines = {"color": highlight_color, "linewidth": linewidth_highlight}
prop_patches = {"edgecolor": highlight_color, "linewidth": linewidth_highlight, "facecolor": "none", "zorder": 10}

_, _, _, _ = add_zoom_effect(ax_cohe_wide, ax_cohe_narrow, min_freq_narrow, max_freq_narrow, prop_lines, prop_patches)

###
# Plot the phase difference in the narrow-band plot
###

# Add the subplots
ax_x = margin_x
ax_y = margin_y

ax_phase = fig.add_axes([ax_x, ax_y, width_frac, height_frac])

# Plot the phase difference and the uncertainty
ax_phase.plot(freqax, phase_diff_temp, color = phase_color, label = "Temperature", linewidth = linewidth_var)
ax_phase.fill_between(freqax, phase_diff_temp + mt_phase_diff_uncer_temp, phase_diff_temp - mt_phase_diff_uncer_temp, color = phase_color, alpha = fill_alpha)

# Plot the highlighted frequency
ax_phase.axvline(freq_highlight, color = highlight_color, linewidth = linewidth_highlight, linestyle = "--")

# Set the x-scale to log
ax_phase.set_xscale("log")

# Set the x-limits
ax_phase.set_xlim(min_freq_narrow, max_freq_narrow)

# Change the y axis ticks to -pi, -pi/2, 0, pi/2, pi
format_phase_diff_ylabels(ax_phase)

# Set the x-tick labels 
ax_phase.xaxis.set_major_locator(FixedLocator(freq_ticks_narrow))
ax_phase.xaxis.set_major_formatter(FixedFormatter(freq_ticklabels_narrow))

ax_phase.xaxis.set_minor_locator(NullLocator())

# Add the axis labels
ax_phase.set_xlabel("Frequency (CPD)", fontsize = axis_label_fontsize)

# Add the title
ax_phase.set_title(f"Frequency vs temperature, phase difference", fontsize = title_fontsize, fontweight = "bold")

# Add the panel label
ax_phase.text(panel_label_x, panel_label_y, "(c)", transform = ax_phase.transAxes, fontsize = panel_label_fontsize, fontweight = "bold")

# Save the figure
save_figure(fig, "stationary_resonance_freq_corr_w_env_obs.png")



















