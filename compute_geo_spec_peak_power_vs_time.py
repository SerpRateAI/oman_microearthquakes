# Plot the power of a specific spectral peak as a function of time

# Imports
from os.path import join
from numpy import isin, concatenate
from pandas import DataFrame
from pandas import read_csv
from time import time

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_spec import get_spectrogram_file_suffix,  get_spec_peak_file_suffix, power2db, read_freq_from_geo_spectrograms, read_spectral_peaks
from utils_plot import plot_cum_freq_counts, save_figure
from multiprocessing import Pool

# Inputs
# Cumulaive frequency count index
freq_count_ind = 0

# Spectrogram
window_length = 60.0
overlap = 0.0
downsample = True
downsample_factor = 60

# Spectral peaks
prom_threshold = 5.0
rbw_threshold = 0.2

min_freq_peak = None
max_freq_peak = None

# Grouping
count_threshold = 4

# Read the cumulative frequency counts
print("Reading the cumulative frequency counts...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

filename_in = f"geo_spec_peak_cum_freq_counts_{suffix_spec}_{suffix_peak}_count{count_threshold:d}.csv"

inpath = join(indir, filename_in)
cum_count_df = read_csv(inpath)

freq_out =  cum_count_df.iloc[freq_count_ind]['frequency']

print(f"Extracting the power of spectral peak at {freq_out:.1f} Hz...")

# Process each station
stations_all = []
timeax_all = []
power_total_list = []
is_peak_list = []
for station in stations:
    clock1 = time()
    print(f"Processing station {station}...")

    # Read the spectral peaks
    print("Reading the spectral peaks...")
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
    suffix_peak = get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = min_freq_peak, max_freq = max_freq_peak)

    filename_in = f"geo_spectral_peaks_{station}_{suffix_spec}_{suffix_peak}.h5"
    inpath = join(indir, filename_in)
    peak_df = read_spectral_peaks(inpath)
    print(f"Read {peak_df.shape[0]:d} spectral peaks.")

    # Find the power of the spectral peak at the desired frequency
    print("Extracting the spectral peaks at the desired frequency...")
    peak_df_sub = peak_df[peak_df['frequency'] == freq_out]
    print(f"Found {peak_df_sub.shape[0]:d} spectral peaks.")

    # Reading the power of the frequency from the spectrograms
    print("Reading the power of the frequency from the spectrograms...")
    suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)
    filename_in = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
    inpath = join(indir, filename_in)

    power_dict = read_freq_from_geo_spectrograms(inpath, freq_out)
    timeax = power_dict['times']
    power_z = power_dict['Z']
    power_1 = power_dict['1']
    power_2 = power_dict['2']
    power_total = power_z + power_1 + power_2
    power_total = power2db(power_total)
    power_total_list.append(power_total)
    print("Done.")

    # Generate the boolean array indicating the presence of the spectral peak
    print("Generating the boolean array...")
    time_peak = peak_df_sub['time'].values
    is_peak = isin(timeax, time_peak)
    is_peak_list.append(is_peak)
    num_peak = is_peak.sum()
    print(f"{num_peak:d} spectral peaks are present.")
    print("Done.")

    # Generate the station label for all the data points
    stations_all.extend([station] * len(timeax))
    timeax_all.append(timeax)

    clock2 = time()
    print(f"Processed station {station} in {clock2 - clock1:.1f} seconds.")

# Assemble the results into a hierarchical dataframe
print("Assembling the results...")
power_total = concatenate(power_total_list)
is_peak = concatenate(is_peak_list)
timeax_all = concatenate(timeax_all)

data = {'station': stations_all, 'time': timeax_all, 'power': power_total, 'is_peak': is_peak}
out_df = DataFrame(data)
out_df.set_index(['station', 'time'], inplace = True)
print("Done.")

# Save the results
print("Saving the results...")
filename_out = f"geo_spec_power_vs_time_{suffix_spec}_{suffix_peak}_freq{freq_out:.1f}hz.h5"
outdir = indir
outpath = join(outdir, filename_out)

out_df.to_hdf(outpath, key = 'power', mode = 'w')
print(f"Saved to {outpath}.")