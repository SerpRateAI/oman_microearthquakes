"""
Plot the hyrophone and geophone spectra in the frequency range of the borehole organ pipe modes.
"""
from pathlib import Path
from pandas import Timestamp, Timedelta
from numpy import array
from scipy.signal.windows import dpss
from matplotlib.pyplot import subplots
from argparse import ArgumentParser

from utils_basic import SPECTROGRAM_DIR as indir, BOREHOLE_DEPTH as borehole_depth, SOUND_SPEED_WATER as sound_speed_water, WATER_TABLE as water_table
from utils_basic import power2db
from utils_preproc import read_and_process_windowed_geo_waveforms, read_and_process_windowed_hydro_waveforms
from utils_mt import get_mt_cspec, mt_autospec
from utils_plot import format_freq_xlabels, format_db_ylabels, save_figure

###
# Helper functions
###
def get_organpipe_freqs(water_table_depth):
    orders = array([1, 3, 5])
    wavelengths = (borehole_depth - water_table_depth) / orders * 4
    freqs = sound_speed_water / wavelengths

    return freqs

###
# Inputs
###

parser = ArgumentParser(description = "Plot the hyrophone and geophone spectra in the frequency range of the borehole organ pipe modes.")

parser.add_argument("--starttime", type = Timestamp, help = "Start time of the time window")

parser.add_argument("--water_table_depth", type = float, help = "Water table depth in meters", default = water_table)
parser.add_argument("--duration", type = float, help = "Duration of the time window in seconds", default = 300.0)
parser.add_argument("--station_hydro", type = str, default = "B00", help = "Station name of the hydrophone")
parser.add_argument("--location_hydro", type = str, default = "01", help = "Location of the hydrophone")
parser.add_argument("--station_geo", type = str, default = "B01", help = "Station name of the geophone")
parser.add_argument("--min_freq", type = float, help = "Minimum frequency in Hz", default = 0.0)
parser.add_argument("--max_freq", type = float, help = "Maximum frequency in Hz", default = 10.0)
parser.add_argument("--nw", type = float, help = "Time-bandwidth product", default = 3.0)
parser.add_argument("--figwidth", type = float, help = "Figure width in inches", default = 10.0)
parser.add_argument("--figheight", type = float, help = "Figure height in inches", default = 10.0)
parser.add_argument("--min_db_hydro", type = float, help = "Minimum decibel for the hydrophone spectrum", default = -80.0)
parser.add_argument("--max_db_hydro", type = float, help = "Maximum decibel for the hydrophone spectrum", default = 0.0)
parser.add_argument("--min_db_geo", type = float, help = "Minimum decibel for the geophone spectrum", default = -15.0)
parser.add_argument("--max_db_geo", type = float, help = "Maximum decibel for the geophone spectrum", default = 30.0)
parser.add_argument("--linewidth_curve", type = float, help = "Line width for the curve", default = 2.0)
parser.add_argument("--linewidth_marker", type = float, help = "Line width for the markers", default = 2.0)

args = parser.parse_args()
water_table_depth = args.water_table_depth
starttime = args.starttime
duration = args.duration
station_hydro = args.station_hydro
location_hydro = args.location_hydro
station_geo = args.station_geo
min_freq = args.min_freq
max_freq = args.max_freq
nw = args.nw
figwidth = args.figwidth
figheight = args.figheight
min_db_hydro = args.min_db_hydro
max_db_hydro = args.max_db_hydro
min_db_geo = args.min_db_geo
max_db_geo = args.max_db_geo
linewidth_curve = args.linewidth_curve
linewidth_marker = args.linewidth_marker

# Print the inputs
print(f"Start time: {starttime}")
print(f"Duration: {duration}")
print(f"Station hydro: {station_hydro}")
print(f"Location hydro: {location_hydro}")
print(f"Station geo: {station_geo}")
print(f"Minimum frequency: {min_freq}")
print(f"Maximum frequency: {max_freq}")
print(f"Time-bandwidth product: {nw}")
print(f"Minimum decibel for the hydrophone spectrum: {min_db_hydro}")
print(f"Maximum decibel for the hydrophone spectrum: {max_db_hydro}")
print(f"Minimum decibel for the geophone spectrum: {min_db_geo}")
print(f"Maximum decibel for the geophone spectrum: {max_db_geo}")
###
# Read the hydrophone and geophone waveforms
###

print("Reading the hydrophone waveforms...")
stream_hydro = read_and_process_windowed_hydro_waveforms(stations = station_hydro, locations = location_hydro, starttime = starttime, dur = duration)

if stream_hydro is None:
    raise ValueError(f"No hydrophone waveforms found for {station_hydro} at {location_hydro}.")

print("Reading the geophone waveforms...")
stream_geo = read_and_process_windowed_geo_waveforms(stations = station_geo, starttime = starttime, dur = duration)
if stream_geo is None:
    raise ValueError(f"No geophone waveforms found for {station_geo}.")

###
# Compute the hydrophone and geophone spectra
###

print("Computing the hydrophone spectrum...")
trace_hydro = stream_hydro[0]
waveform_hydro = trace_hydro.data
sampling_rate_hydro = trace_hydro.stats.sampling_rate
num_pts_hydro = len(waveform_hydro)
taper_mat, ratio_vec = dpss(num_pts_hydro, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)
mt_aspec_params = mt_autospec(waveform_hydro, taper_mat, ratio_vec, sampling_rate_hydro)
aspec_hydro = mt_aspec_params.aspec
freqax_hydro = mt_aspec_params.freqax

print("Computing the geophone spectrum...")
for i, trace in enumerate(stream_geo):
    waveform_geo = trace.data
    sampling_rate_geo = trace.stats.sampling_rate
    num_pts_geo = len(waveform_geo)
    taper_mat, ratio_vec = dpss(num_pts_geo, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)
    mt_aspec_params = mt_autospec(waveform_geo, taper_mat, ratio_vec, sampling_rate_geo)
    aspec_geo = mt_aspec_params.aspec
    if i == 0:
        aspec_geo_total = aspec_geo
        freqax_geo = mt_aspec_params.freqax
    else:
        aspec_geo_total = aspec_geo_total + aspec_geo

###
# Plot the hydrophone and geophone spectra
###

print("Plotting the hydrophone spectrum...")
fig, axes = subplots(2, 1, figsize = (figwidth, figheight))
fig.subplots_adjust(hspace = 0.12)

# Plot the hydrophone spectrum
ax = axes[0]
freqax = freqax_hydro
aspec = power2db(aspec_hydro)
ax.plot(freqax, aspec, label = f"{station_hydro} at {location_hydro}", color = "mediumpurple", linewidth = linewidth_curve)
ax.set_xlim(min_freq, max_freq)
ax.set_ylim(min_db_hydro, max_db_hydro)

format_freq_xlabels(ax,
                    plot_axis_label=False, plot_tick_label=False,
                    major_tick_spacing=1.0, num_minor_ticks=5)

format_db_ylabels(ax,
                  sensor = "hydro",
                  plot_axis_label=True, plot_tick_label=True,
                  major_tick_spacing=10.0, num_minor_ticks=5)

ax.set_title(f"Hydrophone {station_hydro}.{location_hydro}", fontsize = 14.0, fontweight = "bold")

# Plot the organ pipe modes
organpipe_freqs = get_organpipe_freqs(water_table_depth)
freq0 = organpipe_freqs[0]
ax.axvline(freq0, color = "crimson", linestyle = "--", linewidth = linewidth_marker)

# Plot the geophone spectrum
ax = axes[1]
freqax = freqax_geo
aspec = power2db(aspec_geo_total)
ax.plot(freqax, aspec, label = f"{station_geo}", color = "teal", linewidth = linewidth_curve)
ax.set_xlim(min_freq, max_freq)
ax.set_ylim(min_db_geo, max_db_geo)

format_freq_xlabels(ax,
                    plot_axis_label=True, plot_tick_label=True,
                    major_tick_spacing=1.0, num_minor_ticks=5)

format_db_ylabels(ax,
                  sensor = "geo",
                  plot_axis_label=True, plot_tick_label=True,
                  major_tick_spacing=10.0, num_minor_ticks=5)

ax.set_title(f"Geophone {station_geo}", fontsize = 14.0, fontweight = "bold")

endtime = starttime + Timedelta(seconds = duration)
title = f"{starttime:%Y-%m-%d %H:%M:%S} - {endtime:%H:%M:%S}"
fig.suptitle(title, fontsize = 14.0, fontweight = "bold", y = 0.93)

# Save the figure
figname = f"jgr_borehole_organ_pipe_modes.png"
save_figure(fig, figname)