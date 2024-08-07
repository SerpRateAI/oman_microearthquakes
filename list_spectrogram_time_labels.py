# List the time labels of a spectrogram file

# Imports
from os.path import join

from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import read_spec_time_labels

filename = "whole_deployment_daily_geo_spectrograms_B18_window60s_overlap0.0.h5"
inpath = join(indir, filename)

time_labels = read_spec_time_labels(inpath)
print(time_labels)