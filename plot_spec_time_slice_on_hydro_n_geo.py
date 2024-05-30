# Extract and plot the hydrophone and geophone spectra at a specific time
# All hydrophones are plotted, whereas only one geophone is plotted for reference

# Imports
from os.path import join
from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, read_time_from_geo_spectrograms, read_time_from_hydro_spectrograms

# Inputs
# Data
time_to_plot = "2020-01-13T20:00:00"
geo_station_a = "A01"
geo_station_b = "B01"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Plotting
hydro_base = -80.0
geo_base = -100.0

hydro_scale = 0.03
geo_scale = 0.03

# Read the spectra at the specific time
suffix = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)

print(f"Reading the hydrophone spectra at {time_to_plot}...")

filename = f"whole_deployment_daily_hydro_spectrograms_A00_{suffix}.h5"
inpath = join(indir, filename)
hydro_dict_a = read_time_from_hydro_spectrograms(inpath, time_to_plot)

filename = f"whole_deployment_daily_hydro_spectrograms_B00_{suffix}.h5"
inpath = join(indir, filename)
hydro_dict_b = read_time_from_hydro_spectrograms(inpath, time_to_plot)

print(f"Reading the geophone spectra at {time_to_plot}...")

filename = f"whole_deployment_daily_geo_spectrograms_{geo_station_a}_{suffix}.h5"
inpath = join(indir, filename)
geo_dict_a = read_time_from_geo_spectrograms(inpath, time_to_plot)

filename = f"whole_deployment_daily_geo_spectrograms_{geo_station_b}_{suffix}.h5"
inpath = join(indir, filename)
geo_dict_b = read_time_from_geo_spectrograms(inpath, time_to_plot)

