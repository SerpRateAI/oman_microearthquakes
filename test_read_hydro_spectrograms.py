# Test reading hydrophone spectrograms through referencing the time label

# Imports
from os.path import join

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, read_hydro_spectrograms, read_hydro_spec_time_labels

# Inputs
station = "A00"
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Read the list of time labels
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
filename_in = f"whole_deployment_daily_hydro_spectrograms_{station}_{suffix_spec}.h5"

inpath = join(indir, filename_in)
time_labels = read_hydro_spec_time_labels(inpath)

# Read the spectrogram of the first time label
time_label = time_labels[0]
print(f"Reading the spectrograms of {time_label}...")
spectrograms = read_hydro_spectrograms(inpath, time_labels = time_label)
