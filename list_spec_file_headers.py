# List spectral file headers
# Imports
from os.path import join

from utils_spec import read_geo_spec_headers, read_hydro_spec_headers
from utils_basic import SPECTROGRAM_DIR as indir

# Inputs
filename = "whole_deployment_daily_hydro_spectrograms_A00_window60s_overlap0.0.h5"
inpath = join(indir, filename)

# Read the headers
header_dict = read_hydro_spec_headers(inpath)
print(header_dict)
