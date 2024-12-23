###### Verify the Fourier beam forming algorithm by testing it on a synthetic data set ######

### Import necessary libraries ###
from numpy import amax, cos, pi, arange, sin, zeros
from numpy.random import normal
from pandas import Timedelta
from matplotlib.pyplot import figure
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

from utils_basic import STARTTIME_GEO as starttime
from utils_basic import get_geophone_coords
from utils_spec import StreamFFT, TraceFFT
from utils_spec import get_fft
from utils_array import get_fourier_beam_image
from utils_plot import add_colorbar, save_figure

### Input parameters ###
# Synthetic data parameters
time_interval = 1e-3
dur = 300.0
signal_dict = {1.0: (2000, 45), 3.0: (1500, 135), 5.0: (1000, 225)}
noise_std = 100.0

sampling_rate = 1 / time_interval

# Beam forming parameters
min_vel_app = 500
num_slow = 51

# Plotting
station_to_plot = "A01"
fig_width = 10
fig_height = 8

linewidth_spec = 1.0
linewidth_ref = 1.0
linewdith_marker = 0.5

markersize = 30

vels_app_ref = [500.0, 1000.0, 3000.0]

min_freq = 0.0
max_freq = 6.0

min_db = -20.0
max_db = 20.0

width_cbar = 0.01

### Load the station coordinates ###
print("Loading the station coordinates...")
coord_df = get_geophone_coords()

### Generate synthetic data ###
print("Generating synthetic data...")
data_dict = {}
for station, row in coord_df.iterrows():
    timeax = arange(0, dur, time_interval)
    data = zeros(len(timeax))
    xsta = row["east"]
    ysta = row["north"]

    for freq, (vel_app, azimuth) in signal_dict.items():
        slow = 1 / vel_app
        xslow = slow * sin(azimuth * pi / 180)
        yslow = slow * cos(azimuth * pi / 180)
        time_delay = xslow * xsta + yslow * ysta
        
        signal = cos(2 * pi * (timeax - time_delay) * freq)
        data += signal

    noise = normal(0, noise_std, len(timeax))
    data_dict[station] = data + noise

fig = figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(timeax, data_dict[station_to_plot], color='black')
ax.set_title(f"Synthetic data for station {station_to_plot}")
save_figure(fig, "verify_fourier_beam_forming_synthetic_data.png")

### Compute the Fourier spectra ###
print("Computing the Fourier spectra...")
stream_fft = StreamFFT()
endtime = starttime + Timedelta(seconds=dur)
for station, data in data_dict.items():
    freqs, data_fft, window = get_fft(data, sampling_rate)
    trace_fft = TraceFFT(station, "", "Z", starttime, endtime, freqs, data_fft, sampling_rate, window)
    stream_fft.append(trace_fft)

### Compute the beam image ###
print("Computing the beam images...")
beam_dict = {}
for freq in signal_dict.keys(): 
    xslow, yslow, bimage, xslow_max_power, yslow_max_power = get_fourier_beam_image(stream_fft, coord_df, freq, min_vel_app = min_vel_app, num_slow = num_slow)
    xslow = xslow * 1000
    yslow = yslow * 1000
    xslow_max_power = xslow_max_power * 1000
    yslow_max_power = yslow_max_power * 1000
    beam_dict[freq] = (xslow, yslow, bimage, xslow_max_power, yslow_max_power)

### Plot the beam images ###
print("Plotting the beam images...")
fig = figure(figsize=(fig_width, fig_height))

# Define a GridSpec with 2 rows and 3 columns
gs = GridSpec(2, 3, figure=fig)

# Plot the spectra of one of the stations
ax_spec = fig.add_subplot(gs[0, :])  # spans the first row across all 3 columns
trace_fft = stream_fft.select(stations = station_to_plot)[0]
trace_fft.to_db()
psd = trace_fft.psd
freqax = trace_fft.freqs

ax_spec.plot(freqax, psd, color='black', linewidth = linewidth_spec)
ax_spec.set_xlabel("Frequency (Hz)")
ax_spec.set_ylabel("PSD (dB)")

ax_spec.set_title(f"Spectrum of Station {station_to_plot}, Noise Std = {noise_std:.0f}", fontsize=12, fontweight='bold')

for freq, _ in signal_dict.items():
    ax_spec.axvline(x=freq, color='crimson', linestyle='--', linewidth = linewidth_spec)

ax_spec.set_xlim(min_freq, max_freq)
#ax_spec.set_ylim(min_db, max_db)

# Second subplot: first column in the second row
for i, freq_signal in enumerate(signal_dict.keys()):
    ax_beam = fig.add_subplot(gs[1, i])

    xslow, yslow, bimage, xslow_max_power, yslow_max_power = beam_dict[freq_signal]
    mappable = ax_beam.pcolor(xslow, yslow, bimage, cmap="inferno")
    ax_beam.set_title(f"{freq_signal:.1f} Hz", fontsize=12, fontweight='bold')

    if i == 0:
        ax_beam.set_xlabel("East slowness (s/km)")
        ax_beam.set_ylabel("North slowness (s/km)")

    ax_beam.set_aspect('equal')

    # Plot the true and estimated slownesses
    vel_app_signal = signal_dict[freq_signal][0]
    azimuth_signal = signal_dict[freq_signal][1]
    slow_signal = 1 / vel_app_signal * 1000
    xslow_signal = slow_signal * sin(azimuth_signal * pi / 180)
    yslow_signal = slow_signal * cos(azimuth_signal * pi / 180)
    ax_beam.scatter(xslow_signal, yslow_signal, marker = "D", color = "aqua", s = markersize, edgecolors = 'black', linewidth = linewdith_marker)
    ax_beam.scatter(xslow_max_power, yslow_max_power, marker = "D", color = "white", s = markersize, edgecolors = 'black', linewidth = linewdith_marker)

    # Plot the reference velocities
    for vel_app_ref in vels_app_ref:
        slow = 1 / vel_app_ref * 1000
        circle = Circle((0, 0), slow, color='aqua', linestyle='--', linewidth=linewidth_ref, fill=False)
        ax_beam.add_patch(circle)

# Add the colorbar
bbox = ax_beam.get_position()
position = [bbox.x1 + 2 * width_cbar, bbox.y0, width_cbar, bbox.height]
add_colorbar(fig, mappable, "Normalized power", position, orientation = "vertical")

# Save the figure
figname = f"verify_fourier_beam_forming_noise{noise_std:.1f}.png"
save_figure(fig, figname)






