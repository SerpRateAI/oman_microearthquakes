# Plot the particle motion of the waveforms peak-filtered at the stationary resonance frequency

# Imports
from os.path import join
from numpy import abs, amax
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, SAMPLING_RATE as sampling_rate, PM_COMPONENT_PAIRS as comp_pairs
from utils_basic import str2timestamp, time2suffix
from utils_preproc import read_and_process_windowed_geo_waveforms, peak_filter
from utils_spec import get_geo_3c_fft, get_start_n_end_from_center
from utils_plot import save_figure

# Inputs
mode_name = "PR02549"
station = "B02"
center_time = "2020-01-13 20:00:00"
window_length = 60.0

qf_rel = 0.5 # Relative quality factor of the peak filter

panel_width = 4.0
panel_height = 4.0
linewidth = 0.1

# Read the stationary resonance frequency
print(f"Reading the properties of {mode_name}...")
filename = f"stationary_resonance_properties_{mode_name}_geo.h5"
inpath = join(indir, filename)
resoance_df = read_hdf(inpath, key = "properties")

resonance_df = resoance_df[resoance_df["station"] == station]
resonance_df.set_index("time", inplace = True)

center_time = str2timestamp(center_time)
position = resonance_df.index.get_indexer([center_time], method='nearest')[0]
row = resonance_df.iloc[position]

freq = row["frequency"]
qf_res = row["quality_factor"]

# Get the start time of the window
starttime, endtime = get_start_n_end_from_center(center_time, window_length)

# Read the waveform data
print("Reading the data...")
stream = read_and_process_windowed_geo_waveforms(starttime, stations = [station], dur = 60.0)

# Peak filter the data
print("Peak filtering the data with causal and acausal filters...")

comp_dict_causal = {}
comp_dict_acausal = {}
max_amp_causal = 0.0
max_amp_acausal = 0.0
for component in components:
    trace = stream.select(component = component)[0]
    waveform = trace.data

    qf_filter = qf_rel * qf_res
    waveform_filt_causal = peak_filter(waveform, freq, qf_res, sampling_rate, zero_phase = False)
    waveform_filt_acausal = peak_filter(waveform, freq, qf_res, sampling_rate, zero_phase = True)

    max_amp_comp_causal = amax(abs(waveform_filt_causal))
    max_amp_comp_acausal = amax(abs(waveform_filt_acausal))

    if max_amp_comp_causal > max_amp_causal:
        max_amp_causal = max_amp_comp_causal

    if max_amp_comp_acausal > max_amp_acausal:
        max_amp_acausal = max_amp_comp_acausal

    comp_dict_causal[component] = waveform_filt_causal
    comp_dict_acausal[component] = waveform_filt_acausal

for component in components:
    comp_dict_causal[component] /= max_amp_causal
    comp_dict_acausal[component] /= max_amp_acausal

# Plot the particle motion
print("Plotting the particle motions...")

fig, axes = subplots(2, 3, figsize = (3 * panel_width, 2 * panel_height), sharex = True, sharey = True)

for i, pair in enumerate(comp_pairs):
    component1, component2 = pair

    ax = axes[0, i]
    waveform1 = comp_dict_causal[component1]
    waveform2 = comp_dict_causal[component2]

    ax.plot(waveform1, waveform2, color = "black", linewidth = linewidth)
    ax.set_title(f"{component1} - {component2} (Causal)", fontsize = 12, fontweight = "bold")

    ax.set_aspect("equal")

    ax = axes[1, i]
    waveform1 = comp_dict_acausal[component1]
    waveform2 = comp_dict_acausal[component2]

    ax.plot(waveform1, waveform2, color = "black", linewidth = linewidth)
    ax.set_title(f"{component1} - {component2} (Acausal)", fontsize = 12, fontweight = "bold")

ax = axes[0, 0]
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-1.0, 1.0)

# Save the figure
print("Saving the figure...")
time_suffix = time2suffix(center_time)
filename = f"particle_motion_peak_filtered_{mode_name}_{station}_{time_suffix}_dur{window_length:.0f}s.png"
save_figure(fig, filename)





