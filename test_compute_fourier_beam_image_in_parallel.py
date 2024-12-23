# Compare the results and run time of the serial and parallel versions of get_fourier_beam_image
### Import ###
from os.path import join
from pandas import read_csv, read_hdf
from time import time

from utils_basic import GEO_STATIONS as stations_all, SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_basic import get_geophone_coords, str2timestamp, time2suffix
from utils_preproc import read_and_process_windowed_geo_waveforms

from utils_spec import StreamFFT
from utils_spec import get_start_n_end_from_center, get_geo_3c_fft
from utils_array import get_fourier_beam_window, get_slowness_axes
from utils_plot import save_figure

### Inputs ###
# Data
# Time series data
centertime = "2020-01-14 21:46:00"
dur = 60.0 # In seconds

# Stationary resonance information
name = "PR03822"

# Station selection
stations_to_use = "B"

# Beamforming
min_vel_app = 500.0 # Minimum apparent velocity in m/s
num_vel_app = 51 # Number of apparent velocities along each axis

### Read the data
# Load the station coordinates
print("Loading the station coordinates...")
coords_df = get_geophone_coords()

# Load the waveforms
centertime = str2timestamp(centertime)
starttime, _ = get_start_n_end_from_center(centertime, dur)
stream = read_and_process_windowed_geo_waveforms(starttime, dur = dur, stations = stations_all)

# Read the stationary resonance mean frequency
print(f"Reading the properties of {name}...")
filename = f"stationary_resonance_geo_summary_{name}.h5"
inpath = join(indir, filename)

resonance_df = read_hdf(inpath, key = "properties")
# print(resonance_df.head())

### Extract the stationary-resonance frequency of the desired time window ###
# Find the time window with the desired start time
print(f"Extracting the stationary resonance frequency of the desired time window...")
try:
    freq_reson = resonance_df.loc[centertime, "frequency"]
    #freq_reson = resonance_df.loc[starttime, "frequency"]
    print(f"Mean frequency of the stationary resonance: {freq_reson:.3f} Hz.")
except KeyError:
    print(f"Mean frequency of the stationary resonance not found for {centertime}. Finding the closest time window...")
    
    # Find the closest time window
    time_diff = abs(resonance_df.index - str2timestamp(centertime))
    closest_time = time_diff.idxmin()
    freq_reson = resonance_df.loc[closest_time, "frequency"]

# Compute the 3C FFT PSD for each station
print("Computing the 3C FFT of all stations...")

stream_fft = StreamFFT()
for station in stations_all:
    print(f"Processing Station {station}...")
    stream_sta = stream.select(station = station)
    stream_fft_sta = get_geo_3c_fft(stream_sta)

    stream_fft.extend(stream_fft_sta)

### Compute the Fourier beams for the frequency of the stationary resonance ###
# Find the stations to use
print(f"Computing the Fourier beams for the frequency of the stationary resonance: {freq_reson:.2f} Hz...")
if stations_to_use == "A":
    stations_bf = stations_a
    print(f"Using only Array A stations.")
elif stations_to_use == "B":
    print(f"Using only Array B stations.")
    stations_bf = stations_b
else:
    print(f"Using all stations.")
    stations_bf = stations_all

stream_fft_bf = stream_fft.select(stations = stations_bf)

# Define the x and y slow axes
xslowax, yslowax = get_slowness_axes(min_vel_app, num_vel_app)

# Compute the Fourier beams
clock1 = time()
beam_window_ser = get_fourier_beam_window(freq_reson, stream_fft_bf, coords_df, xslowax, yslowax,
                                      num_processes = 1)
clock2 = time()
print(f"Serial time: {clock2 - clock1:.2f} s.")

clock1 = time()
beam_window_par = get_fourier_beam_window(freq_reson, stream_fft_bf, coords_df, xslowax, yslowax,
                                      num_processes = 32)
clock2 = time()
print(f"Parallel time: {clock2 - clock1:.2f} s.")

# Compare the results
fig_ser, axes = beam_window_ser.plot_beam_images()
save_figure(fig_ser, "beam_images_serial.png")

fig_par, axes = beam_window_par.plot_beam_images()
save_figure(fig_par, "beam_images_parallel.png")
