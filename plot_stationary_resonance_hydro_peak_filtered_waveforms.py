# Plot the hydrophone waveforms peak-filtered around the stationary resonance peak frequency

### Imports ###
from os.path import join
from argparse import ArgumentParser
from numpy import isnan
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, HYDRO_LOCATIONS as loc_dict
from utils_basic import get_datetime_axis_from_trace, str2timestamp, time2suffix
from utils_spec import get_start_n_end_from_center, get_stream_fft
from utils_preproc import get_envelope, read_and_process_windowed_hydro_waveforms
from utils_plot import HYDRO_COLOR as color_hydro, PRESSURE_LABEL as label_pressure
from utils_plot import format_datetime_xlabels, format_db_ylabels, format_freq_xlabels, format_pressure_ylabels, save_figure


### Inputs ###
# Command line arguments
parser = ArgumentParser(description = "Plot the hydrophone waveforms peak-filtered around the stationary resonance peak frequency")
parser.add_argument("--station", type = str, help = "Hydrophone station to process")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--window_length", type = float, help = "Length of the window in seconds")
parser.add_argument("--center_time", type = str, help = "Center time of the window")
parser.add_argument("--freq_window_width", type = float, help = "Width of the frequency window in Hz around the geophone peak frequency")
parser.add_argument("--min_mpa", type = float, help = "Minimum pressure in millipascals")
parser.add_argument("--max_mpa", type = float, help = "Maximum pressure in millipascals")
parser.add_argument("--min_db", type = float, help = "Minimum power in dB")
parser.add_argument("--max_db", type = float, help = "Maximum power in dB")

# Parse the command line arguments
args = parser.parse_args()
station = args.station
mode_name = args.mode_name
window_length = args.window_length
center_time = str2timestamp(args.center_time)
freq_window_width = args.freq_window_width
min_mpa = args.min_mpa
max_mpa = args.max_mpa
min_db = args.min_db
max_db = args.max_db

# Print the command line arguments
print(f"Mode name: {mode_name}")
print(f"Station: {station}")
print(f"Window length: {window_length:.0f} s")
print(f"Center time: {center_time}")

# Constants
panel_width = 7.5
panel_height = 3.0

x_loc_label = 0.02
y_loc_label = 0.97

linewidth = 1.0

### Read the input data ###
# Read the stationary resonance properties
print("Reading the stationary resonance properties...")
filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
inpath = join(indir, filename)
geo_properties_df = read_hdf(inpath, key = "properties")

# Get the stationary resonance peak frequency
try:
    freq_peak = geo_properties_df.loc[center_time, "frequency"]
    qf_peak = geo_properties_df.loc[center_time, "mean_quality_factor"]
except KeyError:
    print(f"No stationary resonance properties found for the center time {center_time}! Quitting...")
    exit()

# Read the hydrophone waveforms with and without peak filtering
print("Reading the hydrophone waveforms...")
start_time, end_time = get_start_n_end_from_center(center_time, window_length)

qf_filter = qf_peak / 2

stream_wf = read_and_process_windowed_hydro_waveforms(start_time,
                                                      dur = window_length, stations = station,
                                                      filter = False)

stream_wf_filt = read_and_process_windowed_hydro_waveforms(start_time,
                                                      dur = window_length, stations = station,
                                                      filter = True,
                                                      filter_type = "peak",
                                                      freq = freq_peak, qf = qf_filter)

### Compute the FFT PSD of the waveforms ###
print("Computing the FFT PSD of the waveforms...")
stream_fft = get_stream_fft(stream_wf)
stream_filt_fft = get_stream_fft(stream_wf_filt)

stream_fft.to_db()
stream_filt_fft.to_db()

### Plot the waveforms and the spectra ###
print("Plotting the waveforms and the spectra...")
locations  = loc_dict[station]
num_loc = len(locations)

fig, axes = subplots(num_loc, 2, figsize = (2 * panel_width, num_loc * panel_height))
for i, location in enumerate(locations):
    ax_wf = axes[i, 0]
    ax_spec = axes[i, 1]

    # Plot the waveforms
    trace = stream_wf_filt.select(location = location)[0]
    data = trace.data
    env = get_envelope(data)
    timeax = get_datetime_axis_from_trace(trace)

    ax_wf.plot(timeax, env, color = color_hydro, label = "Peak-filtered", linewidth = linewidth)

    ax_wf.set_xlim(timeax[0], timeax[-1])
    ax_wf.set_ylim(min_mpa, max_mpa)

    if i < num_loc - 1:  
        format_datetime_xlabels(ax_wf,
                                label = False,
                                major_tick_spacing = "15s", num_minor_ticks = 3,
                                date_format = "%Y-%m-%d %H:%M:%S",
                                va = "top", ha = "right", rotation = 15)
        ax_wf.set_xticklabels([])

    else:
        format_datetime_xlabels(ax_wf,
                                label = True,
                                major_tick_spacing = "15s", num_minor_ticks = 3,
                                date_format = "%Y-%m-%d %H:%M:%S",
                                va = "top", ha = "right", rotation = 15)
    
    format_pressure_ylabels(ax_wf)

    ax_wf.text(x_loc_label, y_loc_label, f"{station}.{location}", transform = ax_wf.transAxes, fontsize = 12, fontweight = "bold", va = "top", ha = "left")

    # Plot the spectra of the raw and peak-filtered waveforms
    trace_fft = stream_fft.select(locations = location)[0]
    data = trace_fft.psd
    trace_filt_fft = stream_filt_fft.select(locations = location)[0]
    data_filt = trace_filt_fft.psd
    freqax = trace_fft.freqs

    ax_spec.plot(freqax, data, color = color_hydro, label = "Raw", linewidth = linewidth, alpha = 0.5)
    ax_spec.plot(freqax, data_filt, color = color_hydro, label = "Peak-filtered", linewidth = linewidth)

    min_freq = freq_peak - freq_window_width / 2
    max_freq = freq_peak + freq_window_width / 2
    ax_spec.set_xlim(min_freq, max_freq)
    ax_spec.set_ylim(min_db, max_db)

    if i < num_loc - 1:
        format_freq_xlabels(ax_spec,
                            label = False,
                            major_tick_spacing = 5.0, num_minor_ticks = 5)
        ax_spec.set_xticklabels([])
    else:
        format_freq_xlabels(ax_spec,
                            label = True,
                            major_tick_spacing = 5.0, num_minor_ticks = 5)
        
    format_db_ylabels(ax_spec,
                        major_tick_spacing = 25.0, num_minor_ticks = 5)

title = f"{mode_name}, {center_time.strftime('%Y-%m-%d %H:%M:%S')}, {window_length:.0f} s"
fig.suptitle(title, fontsize = 14, fontweight = "bold", y = 0.92)

### Save the figure ###
print(f"Saving the figure...")
filename = f"stationary_resonance_hydro_peak_filtered_waveforms_{mode_name}_{station}_{time2suffix(center_time)}_dur{window_length:.0f}s.png"
save_figure(fig, filename)





