# Compute the cepstrums of a geophone station and extract the peaks


###### Imports ######
from os.path import join
from scipy.signal import find_peaks
from scipy.fft import ifft
from numpy import arange, isnan, any, concatenate, linspace, log10
from random import uniform
from pandas import Timedelta, DataFrame
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, SAMPLING_RATE as sampling_rate
from utils_spec import get_spectrogram_file_suffix, read_spec_time_labels, read_geo_spectrograms
from utils_plot import save_figure



###### Inputs ######
# Spectrogram computation
window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

# Station
station = "A01"

# Cepstrum computation and plotting
num_to_plot = 20 # Randomly select this number of cepstra to plot

min_freq = 3.0
max_freq = 30.0

height_threshold = 5e-3 # Threshold for the cepstrum peak height
prom_threshold = 3e-3 # Threshold for the cepstrum peak prominence

# Print the inputs
print("### Computing the stacked cepstrum for a geophone station ###")
print("Spectrogram computation:")
print(f"Window length: {window_length} s")
print(f"Overlap: {overlap}")
print(f"Downsample: {downsample}")
print(f"Downsample factor: {downsample_factor}")

print("Station:")
print(f"Station: {station}")

print("Cepstrum computation and plotting:")
print(f"Minimum frequency: {min_freq} Hz")
print(f"Maximum frequency: {max_freq} Hz")

###### Compute the cepstrum for each time window ######
# Read the spectrogram time labels
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
filename_in = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix_spec}.h5"
inpath = join(indir, filename_in)

time_labels = read_spec_time_labels(inpath)

# Process each block of the spectrogram
num_quefr = int(sampling_rate * window_length) + 1
quefr_ax = linspace(0, window_length, num_quefr)

print("### Computing the cepstrum for each time window ###")
peak_dicts = []
for time_label in time_labels:
    print(f"Processing {time_label}...")
    # Read the spectrogram and compute the total power
    print(f"Reading the spectrogram for {time_label}...")
    stream_spec = read_geo_spectrograms(inpath, time_labels = time_label)
    trace_spec = stream_spec.get_total_power()
    trace_spec.to_db()

    # Compute the cepstrum of each time window
    print(f"Computing the cepstrum for {time_label}...")
    spectrogram = trace_spec.data
    times = trace_spec.times

    for i, time in enumerate(times):
        # Get the spectrogram
        spectrum = spectrogram[:, i]

        # Check if the spectrogram contains NaN
        if any(isnan(spectrum)):
            continue

        # # Check if the spectrogram is below the threshold
        # if spectrum.max() > db_threshold:
        #     continue

        # Reconstruct the negative frequencies
        spectrum_neg_freq = spectrum[1:][::-1]
        spectrum_full = concatenate((spectrum, spectrum_neg_freq))

        cepstrum = abs(ifft(spectrum_full))
        cepstrum /= cepstrum.max()
        # log_cepstrum = log10(cepstrum)

        # Find the peaks
        i_peaks, _ = find_peaks(cepstrum, height = height_threshold, prominence = prom_threshold)
        num_peaks = 0
        for i_peak in i_peaks:
            frequency = 1 / quefr_ax[i_peak]
            if frequency < min_freq or frequency > max_freq:
                continue

            peak_dicts.append({"time": time, "frequency": frequency})
            num_peaks += 1
        print(f"### In total, {num_peaks} cepstrum peaks are found for {time} ###")

# print(f"### In total, {len(peak_dicts)} cepstrum peaks are found ###")

# Convert the list of dictionaries to a DataFrame
peak_df = DataFrame(peak_dicts)

# Save the DataFrame to CSV and HDF5
filename_out = f"geo_station_cepstrum_peaks_{station}_{suffix_spec}.csv"
outpath = join(indir, filename_out)
peak_df.to_csv(outpath, index = False)
print(f"Saved the cepstrum peaks to {outpath}")

filename_out = f"geo_station_cepstrum_peaks_{station}_{suffix_spec}.h5"
outpath = join(indir, filename_out)
peak_df.to_hdf(outpath, key = "cepstrum_peaks", mode = "w")
print(f"Saved the cepstrum peaks to {outpath}")



