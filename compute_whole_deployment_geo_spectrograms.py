#!/usr/bin/env python
# coding: utf-8

# # Compute the whole deployment geophone spectrograms for a list of stations
# The results for each station is stored in a single HDF5 file, with the spectrograms for each day saved in a separate block for easy reading and writing

# In[1]:


# Imports
from os import makedirs
from os.path import join
from time import time

from utils_basic import SPECTROGRAM_DIR as outdir, GEO_STATIONS as stations
from utils_basic import get_geophone_days, get_geo_metadata
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_spec import create_geo_spectrogram_file, write_geo_spectrogram_block, finish_geo_spectrogram_file
from utils_torch import get_daily_geo_spectrograms


# In[2]:


# Inputs
block_type = "daily"
window_length = 60.0 # IN SECONDS
overlap = 0.0
downsample = False # Downsample along the frequency axis
downsample_factor = 60 # Downsample factor for the frequency axis
resample_in_parallel = True # Resample along the time axis in parallel
num_process_resample = 16 # Number of processes while resampling along the time axis in parallel
save_ds_only = False # Whether to save only the downsampled spectrograms

if not downsample and save_ds_only:
    raise ValueError("Conflicting options for downsampling!")


# In[3]:


# Create the output directory
makedirs(outdir, exist_ok=True)


# In[4]:


# Load the station metadata
metadata = get_geo_metadata()


# In[5]:


# Get the geophone deployment days
days = get_geophone_days()
# days = ["2020-01-13"]


# In[6]:


# Compute the frequency intervals
freq_interval = 1.0 / window_length
freq_interval_ds = freq_interval * downsample_factor

# Loop over stations
for station in stations:
    print("######")
    print(f"Processing {station}...")
    print("######")
    print("")
    
    # Create the HDF5 files
    if not save_ds_only:
        file = create_geo_spectrogram_file(station, 
                                           window_length = window_length, overlap = overlap, 
                                           freq_interval = freq_interval, downsample = False, outdir = outdir)

    if downsample:
        file_ds = create_geo_spectrogram_file(station, 
                                              window_length = window_length, overlap = overlap,
                                              freq_interval = freq_interval_ds, downsample = True, downsample_factor = downsample_factor, outdir = outdir)

    # Loop over days
    num_days = len(days)
    time_labels = []
    for i, day in enumerate(days):
        # Start the clock
        clock1 = time()
        print(f"### Processing {day} for {station}... ###")

        # Read and preprocess the data
        stream_day = read_and_process_day_long_geo_waveforms(day, metadata, stations = station)
        if stream_day is None:
            print(f"{day} is skipped.")
            continue

        # Compute the spectrogram
        stream_spec, stream_spec_ds = get_daily_geo_spectrograms(stream_day, 
                                                                 window_length = window_length, overlap = overlap,
                                                                 resample_in_parallel = resample_in_parallel, num_process_resample = num_process_resample,
                                                                 downsample = downsample, downsample_factor = downsample_factor)
        time_labels.append(stream_spec[0].time_label)
        
        # Write the spectrogram block
        if not save_ds_only:
            write_geo_spectrogram_block(file, stream_spec, close_file = False)

        if downsample:
            write_geo_spectrogram_block(file_ds, stream_spec_ds, close_file = False)

        # Stop the clock
        clock2 = time()
        elapse = clock2 - clock1
        
        print(f"Elapsed time: {elapse} s")

    # Finish the spectrogram file
    if not save_ds_only:
        finish_geo_spectrogram_file(file, time_labels)
        
    if downsample:
        finish_geo_spectrogram_file(file_ds, time_labels)
        
    print("")


# In[ ]:




