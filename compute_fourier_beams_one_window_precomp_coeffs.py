# Compute the Fourier frequency-dependent beams for one time window using precompute coefficients

### Import ###
from os.path import join
from pandas import read_csv, read_hdf, to_datetime

from utils_basic import GEO_STATIONS as stations_all, SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_basic import get_geophone_coords, str2timestamp, time2suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_spec import StreamFFT
from utils_spec import get_start_n_end_from_center, get_geo_3c_fft
from utils_array import get_fourier_beam_window, get_slowness_axes
from utils_plot import plot_all_geo_fft_psd_n_maps, plot_beam_images, save_figure


### Inputs ###
# Data
mode_name = "PR03822"
centertime = "2020-01-14 21:44:00"
dur = 60.0 # In seconds

# Station selection
station_selection = "A" # "A", "B", "All"

# Beamforming
min_vel_app = 500.0 # Minimum apparent velocity in m/s
num_vel_app = 51 # Number of apparent velocities along each axis
normalize = True # Normalize the Fourier coefficiet of each station or not

prom_threshold = 10.0 # Prominence threshold for the peaks in the spectrogram

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

# Read the stationary resonance information
filename = f"stationary_resonance_phase_amplitude_diff_{mode_name}_geo.h5"
inpath = join(indir, filename)

centertime = str2timestamp(centertime)
resonance_df = read_hdf(inpath, key = "properties")
resonance_df["time"] = to_datetime(resonance_df["time"])
# print(type(resonance_df["time"].values[0]))
resonance_window_df = resonance_df.loc[resonance_df["time"] == centertime]


if resonance_window_df.empty:
    raise ValueError(f"No stationary resonance information found for {mode_name} at {centertime}.")

if station_selection == "A":
    resonance_window_df = resonance_window_df[resonance_window_df["station"].isin(stations_a)]
elif station_selection == "B":
    resonance_window_df = resonance_window_df[resonance_window_df["station"].isin(stations_b)]

# print(resonance_window_df)


freq_reson = resonance_window_df["frequency"].values[0]

# Define the x and y slow axes
xslowax, yslowax = get_slowness_axes(min_vel_app, num_vel_app)

# Compute the Fourier beams
beam_window = get_fourier_beam_window(freq_reson, coords_df, xslowax, yslowax,
                                      peak_coeff_df = resonance_window_df,
                                      normalize = normalize,
                                      prom_threshold = prom_threshold)

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
    figname = f"fourier_beams_{mode_name}_array_a_{time2suffix(centertime)}_dur{dur:.0f}s_{normalize_str}_precomp.png"
elif station_selection == "B":
    figname = f"fourier_beams_{mode_name}_array_b_{time2suffix(centertime)}_dur{dur:.0f}s_{normalize_str}_precomp.png"
else:
    figname = f"fourier_beams_{mode_name}_{time2suffix(centertime)}_dur{dur:.0f}s_{normalize_str}_precomp.png"

save_figure(fig, figname)