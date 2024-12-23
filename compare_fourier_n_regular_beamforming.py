# Compare the results of Fourier beamforming and regular beamforming

### Import necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from pandas import Timestamp
from pandas import read_hdf, to_datetime

from utils_basic import GEO_COMPONENTS as components
from utils_basic import SPECTROGRAM_DIR as indir, BEAM_DIR as outdir
from utils_basic import GEO_STATIONS as stations_all, GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_basic import INNER_STATIONS_A as inner_stations_a, INNER_STATIONS_B as inner_stations_b, MIDDLE_STATIONS_A as middle_stations_a, MIDDLE_STATIONS_B as middle_stations_b
from utils_basic import str2timestamp, time2suffix
from utils_basic import get_geophone_coords
from utils_spec import get_start_n_end_from_center
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_array import FourierBeamCollection
from utils_array import get_beam_window, get_fourier_beam_window, get_slowness_axes, select_stations, set_station_coords
from utils_plot import save_figure

### Inputs ###
# Data
parser = ArgumentParser(description="Input parameters for computing the Fourier beams of one mode for all time windows.")
parser.add_argument("--mode_name", type=str, help="Name of the mode.")
parser.add_argument("--centertime", type=str, help="Center time of the time window.")
parser.add_argument("--duration", type=float, help="Duration of the time window in seconds.")
parser.add_argument("--array_selection", type=str, help="Selection of the subarray: A or B.")
parser.add_argument("--array_aperture", type=str, help="Aperture of the array: small, medium, or large.")
parser.add_argument("--min_vel_app", type=float, help="Minimum apparent velocity in m/s.")
parser.add_argument("--num_vel_app", type=int, help="Number of apparent velocities along each axis.")
parser.add_argument("--normalize", action="store_true", help="Normalize the Fourier coefficients of each station or not.")
parser.add_argument("--prom_threshold", type=float, help="Prominence threshold for the peaks in the spectrogram.")

args = parser.parse_args()
mode_name = args.mode_name
centertime = str2timestamp(args.centertime)
duration = args.duration
array_selection = args.array_selection
array_aperture = args.array_aperture
min_vel_app = args.min_vel_app
num_vel_app = args.num_vel_app
normalize = args.normalize
prom_threshold = args.prom_threshold

# Print the inputs
print("###")
print(f"Comparing the results of Fourier beamforming and regular beamforming for {mode_name} at {centertime}.")
print(f"Selection of the subarray: {array_selection}.")
print(f"Aperture of the array: {array_aperture}.")
print(f"Minimum apparent velocity: {min_vel_app:.1f} m/s.")
print(f"Number of apparent velocities along each axis: {num_vel_app:d}.")
if normalize:
    print("The Fourier coefficients and time series of each station are normalized.")
else:
    print("The Fourier coefficients and time series of each station are not normalized.")
print("###")

### Read the data ###
# Load the station coordinates
print("Loading the station coordinates...")
coords_df = get_geophone_coords()

# Read the stationary resonance information
filename = f"stationary_resonance_phase_amplitude_diff_{mode_name}_geo.h5"
inpath = join(indir, filename)
resonance_df = read_hdf(inpath, key = "properties")

# Get the stations and the minimum number of stations
stations, min_num_stations = select_stations(array_selection, array_aperture)
resonance_df = resonance_df.loc[resonance_df["station"].isin(stations)]

# Define the x and y slowness axes
xslowax, yslowax = get_slowness_axes(min_vel_app, num_vel_app)

### Compute and plot the Fourier beams ###
resonance_window_df = resonance_df.loc[resonance_df["time"] == centertime]
freq_reson = resonance_df["frequency"].values[0]    
print(f"Mean frequency of the stationary resonance: {freq_reson:.3f} Hz.")
# print(resonance_window_df)

# Compute the Fourier beams for the frequency of the stationary resonance
print(f"Computing the Fourier beams...")
fourier_beam_window = get_fourier_beam_window(freq_reson, coords_df, xslowax, yslowax,
                                                peak_coeff_df = resonance_window_df,
                                                normalize = normalize,
                                                prom_threshold = prom_threshold,
                                                min_num_stations = min_num_stations)

if fourier_beam_window is None:
    raise ValueError(f"No enough stations for computing Fourier beams.")


# Plot the Fourier beams
print("Plotting the Fourier beams...")
fig, axes = fourier_beam_window.plot_beam_images()
fig.suptitle(f"Fourier beams, {mode_name}, Array {array_selection}, {array_aperture} aperture, {centertime.strftime('%Y-%m-%d %H:%M:%S')}, {duration:.0f} s", fontweight = "bold", fontsize = 14, y = 0.93)

if normalize:
    normalize_str = "normalized_amp"
else:
    normalize_str = "original_amp"

time_suffix = time2suffix(centertime)

figname = f"fourier_beams_{mode_name}_array_{array_selection}_{array_aperture}_{time_suffix}_dur{duration:.0f}s_{normalize_str}.png"
save_figure(fig, figname)

### Compute and plot the time-domain beams ###
# Read the mean properties of the stationary resonance
print("Reading the resonance information...")
filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
inpath = join(indir, filename)
mean_properties_df = read_hdf(inpath, key = "properties")

qf_reson = mean_properties_df.loc[centertime, "mean_quality_factor"]
qf_filter = qf_reson / 2

# Read the waveforms and set the station coordinates
print("Reading and processing the waveforms...")
starttime, endtime = get_start_n_end_from_center(centertime, duration)
stream = read_and_process_windowed_geo_waveforms(starttime, 
                                                 endtime = endtime, stations = stations,
                                                 filter = True, filter_type = "peak",
                                                 freq = freq_reson, qf = qf_filter)

stream = set_station_coords(stream, coords_df)

# Compute the regular beams

print("Computing the time-domain beams...")
station_dict = {}
for component in components:
    station_dict[component] = fourier_beam_window.bimage_dict[component].stations

beam_window = get_beam_window(stream, xslowax, yslowax,
                                normalize = True, station_dict = station_dict)

# Plot the regular beams
print("Plotting the time-domain beams...")
fig, axes = beam_window.plot_beam_images()
fig.suptitle(f"Time-domain beams, {mode_name}, Array {array_selection}, {array_aperture} aperture, {centertime.strftime('%Y-%m-%d %H:%M:%S')}, {duration:.0f} s", fontweight = "bold", fontsize = 14, y = 0.93)

figname = f"time_domain_beams_{mode_name}_array_{array_selection}_{array_aperture}_{time_suffix}_dur{duration:.0f}s.png"
save_figure(fig, figname)






