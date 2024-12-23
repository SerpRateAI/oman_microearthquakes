# Compute the Fourier frequency-dependent beams for one time window

### Import ###
from os.path import join
from pandas import read_csv, read_hdf, to_datetime

from utils_basic import GEO_STATIONS as stations_all, SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_basic import get_geophone_coords, str2timestamp, time2suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_spec import StreamFFT
from utils_spec import get_start_n_end_from_center, get_stream_fft
from utils_array import get_fourier_beam_window, get_slowness_axes
from utils_plot import plot_all_geo_fft_psd_n_maps, plot_beam_images, save_figure


### Inputs ###
# Data
mode_name = "PR03822"
centertime = "2020-01-14 21:45:00"
dur = 60.0 # In seconds

# Station selection
station_selection = "B" # "A", "B", "All"

# Beamforming
min_vel_app = 500.0 # Minimum apparent velocity in m/s
num_vel_app = 51 # Number of apparent velocities along each axis
normalize = True # Normalize the Fourier coefficiet of each station or not

# Plotting
factor = 0.075
psd_plot_width = 5.0
psd_plot_hw_ratio = 1.5
linewidth_psd = 0.5
linewdith_reson = 1.0

reference_vels = [500.0, 1500.0, 3500.0]

fontsize_title = 14

### Read the data
# Load the station coordinates
print("Loading the station coordinates...")
coords_df = get_geophone_coords()

# Load the waveforms
if station_selection == "A":
    stations = stations_a
    print(f"Using only Array A stations.")
elif station_selection == "B":
    stations = stations_b
    print(f"Using only Array B stations.")
else:
    stations = stations_all
    print(f"Using all stations.")

centertime = str2timestamp(centertime)
starttime, _ = get_start_n_end_from_center(centertime, dur)

stream = read_and_process_windowed_geo_waveforms(starttime, dur = dur, stations = stations)

# Read the stationary resonance mean frequency
print(f"Reading the properties of {mode_name}...")
filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
inpath = join(indir, filename)

resonance_df = read_hdf(inpath, key = "properties")

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

### Compute and plot the FFT PSD for all stations ###
# Compute the 3C FFT PSD for each station
print("Computing the 3C FFT...")
stream_fft = get_stream_fft(stream)


### Compute the Fourier beams for the frequency of the stationary resonance ###
# Find the stations to use
print(f"Computing the Fourier beams for the frequency of the stationary resonance: {freq_reson:.2f} Hz...")

# Define the x and y slow axes
xslowax, yslowax = get_slowness_axes(min_vel_app, num_vel_app)

# Compute the Fourier beams
beam_window = get_fourier_beam_window(freq_reson, coords_df, xslowax, yslowax,
                                      stream_fft = stream_fft,
                                      normalize = normalize)

### Plot the Fourier beams ###
print("Plotting the Fourier beams...")
fig, axes = beam_window.plot_beam_images()

if station_selection == "A":
    title = f"Fourier Beams at {freq_reson:.2f} Hz, Array A only, {centertime.strftime('%Y-%m-%d %H:%M:%S')}, {dur:.0f} s"
elif station_selection == "B":
    title = f"Fourier Beams at {freq_reson:.2f} Hz, Array B only, {centertime.strftime('%Y-%m-%d %H:%M:%S')}, {dur:.0f} s"
else:
    title = f"Fourier Beams at {freq_reson:.2f} Hz, All Stations, {centertime.strftime('%Y-%m-%d %H:%M:%S')}, {dur:.0f} s"

fig.suptitle(title, fontweight = "bold", fontsize = fontsize_title, y = 0.95)

# Save the figure
normalize_str = "normalized_amp" if normalize else "original_amp"

if station_selection == "A":
    figname = f"fourier_beams_{mode_name}_array_a_{time2suffix(centertime)}_dur{dur:.0f}s_{normalize_str}.png"
elif station_selection == "B":
    figname = f"fourier_beams_{mode_name}_array_b_{time2suffix(centertime)}_dur{dur:.0f}s_{normalize_str}.png"
else:
    figname = f"fourier_beams_{mode_name}_{time2suffix(centertime)}_dur{dur:.0f}s_{normalize_str}.png"

save_figure(fig, figname)