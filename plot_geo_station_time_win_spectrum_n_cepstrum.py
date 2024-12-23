# Plot the cepstrum and the spectrum of geophone data in a time window
#

# Import modules
from os.path import join
from numpy import abs, concatenate, linspace
from scipy.fft import ifft
from pandas import read_csv
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, STARTTIME_GEO as starttime, ENDTIME_GEO as endtime, SAMPLING_RATE as sampling_rate
from utils_basic import power2db, time2suffix
from utils_spec import get_spectrogram_file_suffix, read_time_slice_from_geo_spectrograms
from utils_plot import format_freq_xlabels, save_figure

# Input parameters
# Time window
time_window = "2020-01-13 22:21:00"

# Station
station = "A01"

# Spectrogram computation
window_length = 60.0
overlap = 0.0
downsample = False

# Base frequency and maximum mode
freq_base = 6.410256410256411
max_mode = 15

# Plotting
figwidth = 9
figheight = 7   

max_ceps = 0.02
min_freq_spec = 0.0
max_freq_spec = 200.0

min_freq_ceps = 1.0
max_freq_ceps = 60.0

max_db = 20.0
min_db = -20.0

linewidth_spec = 0.5
linewidth_ceps = 1.0

linewidth_freq = 0.5
linewidth_quef = 1.0

# # Read the resonance frequencies
# print("Reading the resonance frequencies...")
# filename_in = f"stationary_harmonic_series_{base_name}_base{base_mode:d}.csv"
# inpath = join(indir, filename_in)
# resonance_df = read_csv(inpath)

# Read the data
print("Reading the data...")
suffix = get_spectrogram_file_suffix(window_length, overlap, downsample)
filename = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix}.h5"
inpath = join(indir, filename)

power_dict = read_time_slice_from_geo_spectrograms(inpath, time_window)
i = 0
for i, component in enumerate(components):
    psd = power_dict[component]
    if i == 0:
        psd_total = psd
    else:
        psd_total += psd

    i += 1

psd = power2db(psd_total)
freqax = power_dict["frequencies"]

# Reconstruct the full spectrum
print("Reconstructing the full spectrum...")
psd_neg_freq = psd[1:][::-1]
psd_full = concatenate([psd, psd_neg_freq])

# Compute the cepstrum and normalize
print("Computing the cepstrum...")
cepstrum = abs(ifft(psd_full))
cepstrum /= cepstrum.max()

# Construct the quefrency axis
quefrency = linspace(0, (len(cepstrum) -1) / sampling_rate, len(cepstrum))

# Plot the spectrum and the cepstrum
fig, axes = subplots(2, 1, figsize = (figwidth, figheight))

axes[0].plot(freqax, psd, color = "black", linewidth = linewidth_spec)
axes[0].set_ylabel("Power spectral density (dB)")
axes[0].set_xlim(min_freq_spec, max_freq_spec)
axes[0].set_ylim(min_db, max_db)

format_freq_xlabels(axes[0], major_tick_spacing = 20, num_minor_ticks = 4, axis_label_size = 10)

# for freq in resonance_df["observed_freq"]:
#     axes[0].axvline(freq, color = "red", linestyle = ":", linewidth = linewidth)

for freq in freq_base * linspace(1, max_mode, max_mode):
    axes[0].axvline(freq, color = "red", linestyle = ":", linewidth = linewidth_freq)

axes[1].plot(quefrency, cepstrum, color = "black", linewidth = linewidth_ceps)
axes[1].set_xlim(1 / max_freq_ceps, 1 / min_freq_ceps)
axes[1].set_ylim(0, max_ceps)

axes[1].set_xscale("log")
axes[1].set_xlabel("Quefrency (s)", fontsize = 10)
axes[1].set_ylabel("Normalized cepstrum amplitude", fontsize = 10) 

freqs = freq_base * linspace(1, max_mode, max_mode)
for i, freq in enumerate(freqs):
    if freq < min_freq_ceps or freq > max_freq_ceps:
        continue
    
    axes[1].axvline(1 / freq, color = "red", linestyle = ":", linewidth = linewidth_quef)
    if i == 0:
        axes[1].text(1 / freq, 2 * max_ceps / 3, f"{freq:.2f} Hz", fontsize = 10, color = "red", ha = "left", va = "center")

fig.suptitle(f"{station}, {time_window}", fontsize = 12, fontweight = "bold", y = 0.92)

# Save the figure
suffix_time = time2suffix(time_window)
outpath = f"geo_station_spectrum_n_cepstrum_{station}_{suffix_time}_ref{freq_base:.2f}hz.png"
save_figure(fig, outpath)


