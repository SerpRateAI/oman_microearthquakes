# Plot only the 3C spectra in Fig. 2 in Liu et al. (2025a) for presentation purposes
from os.path import join
from numpy import cos, pi, linspace, isnan, nan


from argparse import ArgumentParser
from json import loads
from scipy.signal.windows import dpss
from scipy.interpolate import interp1d
from pandas import DataFrame, Timedelta, Timestamp
from pandas import concat, read_csv, read_hdf
from matplotlib.pyplot import figure
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
from colorcet import cm


from utils_basic import PLOTTING_DIR as dir_plot, SPECTROGRAM_DIR as dirpath_spec
from utils_basic import str2timestamp
from utils_basic import power2db
from utils_mt import mt_autospec
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix
from utils_plot import format_freq_xlabels,  format_db_ylabels, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for plotting the 3C spectra in Fig. 2 in Liu et al. (2025a) for presentation purposes.")
parser.add_argument("--station", type=str, help="Station whose 3C spectra will be plotted.")
parser.add_argument("--starttime", type=str, help="Time window for the 3C spectra.")
parser.add_argument("--window_length_stft", type=float, default=300.0, help="Window length in seconds for computing the STFT.")
parser.add_argument("--duration", type=float, help="Duration in seconds", default=300.0)
parser.add_argument("--subarray_label_size", type=float, default=15.0, help="Font size of the subarray labels.")
parser.add_argument("--freq_label_size", type=float, default=15.0, help="Font size of the frequency labels.")
parser.add_argument("--axis_label_size", type=float, default=12.0, help="Font size of the axis labels.")
parser.add_argument("--tick_label_size", type=float, default=12.0, help="Font size of the tick labels.")
parser.add_argument("--title_size", type=float, default=15.0, help="Font size of the title.")
parser.add_argument("--legend_size", type=float, default=12.0, help="Font size of the legend.")
parser.add_argument("--arrow_gap", type=float, default=5.0, help="Gap between the arrow and the text.")
parser.add_argument("--arrow_length", type=float, default=10.0, help="Length of the arrow.")
parser.add_argument("--arrow_width", type=float, default=1.5, help="Width of the arrow.")
parser.add_argument("--arrow_head_width", type=float, default=10.0, help="Width of the arrow head.")
parser.add_argument("--arrow_head_length", type=float, default=10.0, help="Length of the arrow head.")
parser.add_argument("--delta_f", type=float, default=12.75, help="Mode spacing in Hz")


parser.add_argument("--log_scale", action="store_true", help="Use a log scale for the x-axis.")

# Parse the command line arguments
args = parser.parse_args()

station = args.station
starttime = args.starttime
duration = args.duration

subarray_label_size = args.subarray_label_size
freq_label_size = args.freq_label_size
axis_label_size = args.axis_label_size
tick_label_size = args.tick_label_size
title_size = args.title_size
legend_size = args.legend_size

arrow_gap = args.arrow_gap
arrow_length = args.arrow_length
arrow_width = args.arrow_width
arrow_head_width = args.arrow_head_width
arrow_head_length = args.arrow_head_length
delta_f = args.delta_f


log_scale = args.log_scale

# Constants
overlap = 0.0
nw = 3
min_prom = 15.0
min_rbw = 15.0
max_mean_db = 15.0

fig_width = 15.0
fig_height = 7.5

margin_x = 0.05
margin_y = 0.05

scale_bar_length = 25.0

min_depth = 0.0
max_depth = 400.0

hydro_min = -0.5
hydro_max = 0.5

freq_min = 0.0
freq_max = 200.0

water_level = 15.0
water_amp = 2.5
water_period = 0.2

linewidth_coast = 0.2
linewidth_water = 2.0
linewidth_spec = 1.0
linewidth_arrow = 1.0

min_vel_app = 0.0
max_vel_app = 2000.0

station_font_size = 14.0
station_label_x = 7.0
station_label_y = 7.0

borehole_font_size = 14.0
borehole_label_x = 40.0
borehole_label_y = -40.0


location_label_x = 0.25

water_font_size = 14.0

major_dist_spacing = 25.0
major_depth_spacing = 50.0
major_freq_spacing = 50.0
major_db_spacing = 20.0

major_tick_length = 5.0
minor_tick_length = 2.0
tick_width = 1.0

frame_width = 1.0

subplot_label_size = 18.0
subplot_offset_x = -0.02
subplot_offset_y = 0.02

min_db = -20.0
max_db = 60.0

### Compute the spectrum ###
print(f"Computing the MT spectrum of station {station}")

# Read the tremor data
print("Reading the tremor waveforms...")

stream = read_and_process_windowed_geo_waveforms(stations=station, starttime=starttime, dur=duration)

# Compute the autospectrum for tremor station
for i, trace in enumerate(stream):
    waveform = trace.data
    sampling_rate = trace.stats.sampling_rate
    num_pts = len(waveform)

    taper_mat, ratio_vec = dpss(num_pts, nw, Kmax = 2 * int(nw) - 1, return_ratios = True)
    mt_aspec_params = mt_autospec(waveform, taper_mat, ratio_vec, sampling_rate)
    aspec = mt_aspec_params.aspec

    if i == 0:
        aspec_total = aspec
    else:
        aspec_total += aspec

aspec_total = power2db(aspec_total)
freqax = mt_aspec_params.freqax


# Read the tremor mode frequencies
print(f"Getting the resonance frequencies of station {station}...")
filename = f"stationary_harmonic_series_PR02549_base2.csv"
filepath = join(dirpath_spec, filename)

harmonic_df = read_csv(filepath)
time_window = str2timestamp(starttime) + Timedelta(seconds =  duration / 2) 
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
suffix_spec = get_spectrogram_file_suffix(duration, overlap)
mode_marker_dfs = []
for mode_name in harmonic_df["mode_name"]:
    print(mode_name)
    
    if mode_name.startswith("MH"):
        mode_marker_df = DataFrame({"mode_name": [mode_name], "frequency": [nan]})
    else:
        filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
        filepath = join(dirpath_spec, filename)

        resonance_df = read_hdf(filepath)
        print(time_window)

        freq = resonance_df.loc[(resonance_df["station"] == station) & (resonance_df["time"] == time_window), "frequency"].values[0]
        mode_marker_df = DataFrame({"mode_name": [mode_name], "frequency": [freq]})
    
    mode_marker_df["mode_order"] = harmonic_df.loc[harmonic_df["mode_name"] == mode_name, "mode_order"].values[0]

    # Handle the exception 
    if mode_marker_df.empty:
        continue

    mode_marker_dfs.append(mode_marker_df)

mode_marker_df = concat(mode_marker_dfs, axis = 0)
mode_marker_df.reset_index(drop = True, inplace = True)

# Fill in the missing modes
for i, row in mode_marker_df.iterrows():
    if isnan(row["frequency"]):
        if row["mode_order"] == 1:
            freq_higher = mode_marker_df.loc[mode_marker_df["mode_order"] == row["mode_order"] + 1, "frequency"].values[0]
            freq = freq_higher / 2 # Half of the next mode
        else:
            freq_lower = mode_marker_df.loc[mode_marker_df["mode_order"] == row["mode_order"] - 1, "frequency"].values[0]
            freq_upper = mode_marker_df.loc[mode_marker_df["mode_order"] == row["mode_order"] + 1, "frequency"].values[0]
            freq = (freq_lower + freq_upper) / 2 # Average of the two adjacent modes

        print(f"Mode {row['mode_name']} has no resonance frequency. Filling in with {freq:.2f} Hz...")

        mode_marker_df.loc[i, "frequency"] = freq

# Plot the spectra and labels for the resonance frequencies
print(f"Plotting the 3C spectra of station {station}...")

# Generate the figure and axes
fig = figure(figsize = (fig_width, fig_height))
ax_spec = fig.add_axes([margin_x, margin_y, 1.0 - 2 * margin_x, 1.0 - 2 * margin_y])

# Plot the total spectrum
ax_spec.plot(freqax, aspec_total, color = "black", linewidth = linewidth_spec, zorder = 2)

# Plot the resonance frequencies
flag = False
for mode_order in mode_marker_df["mode_order"].unique():
    mode_name = mode_marker_df.loc[mode_marker_df["mode_order"] == mode_order, "mode_name"].values[0]
    print(f"Plotting mode {mode_name}...")
    freq_resonance = mode_marker_df.loc[mode_marker_df["mode_order"] == mode_order, "frequency"].values[0]

    # Use interpolation to get the power at the resonance frequency
    power = interp1d(freqax, aspec_total)(freq_resonance)


    if mode_name.startswith("MH"):
        ax_spec.annotate(f"?", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + 2 * arrow_length),
                    color = "gray", fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                    arrowprops=dict(facecolor="gray", edgecolor="gray", shrink=0.05, width=arrow_width, headwidth=arrow_head_width, headlength=arrow_head_length))
    else:
        if not flag:
            ax_spec.annotate(f"Mode {mode_order}", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                        color = "crimson", fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                        arrowprops=dict(facecolor="crimson", edgecolor="crimson", shrink=0.05, width=arrow_width, headwidth=arrow_head_width, headlength=arrow_head_length))
            flag = True
        else:
            ax_spec.annotate(f"{mode_order}", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                        color = "crimson", fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                        arrowprops=dict(facecolor="crimson", edgecolor="crimson", shrink=0.05, width=arrow_width, headwidth=arrow_head_width, headlength=arrow_head_length))
   
# Plot the mode spacing labels
if not log_scale:
    filename = "mode_spacing_labels.csv"
    filepath = join(dir_plot, filename)

    mode_spacing_df = read_csv(filepath)

    for i, row in mode_spacing_df.iterrows():
        mode1 = row["mode1"]
        mode2 = row["mode2"]
        power = row["power"]

        freq1 = mode_marker_df.loc[mode_marker_df["mode_order"] == mode1, "frequency"].values[0]
        freq2 = mode_marker_df.loc[mode_marker_df["mode_order"] == mode2, "frequency"].values[0]

        ax_spec.annotate("", xy=(freq1, power), xytext=(freq2, power),
                        arrowprops=dict(arrowstyle='<->', color="crimson", linewidth=arrow_width))

        ax_spec.text((freq1 + freq2) / 2, power, r"$\boldsymbol{\Delta f}$", color = "crimson", fontsize = freq_label_size, fontweight = "bold", va = "center", ha = "center", zorder = 10, bbox = dict(facecolor = "white", edgecolor = "none", alpha = 1.0))

    ax_spec.text(2.0, max_db - 2.0, rf"$\boldsymbol{{\Delta f}}$ = {delta_f:.2f} Hz", color = "crimson", fontsize = freq_label_size, fontweight = "bold", va = "top", ha = "left")
                    
# Set the axis limits
ax_spec.set_xlim(freq_min, freq_max)
ax_spec.set_ylim(min_db, max_db)

if log_scale:
    ax_spec.set_xscale("log")
    ax_spec.set_xlabel("Frequency (Hz)", fontsize = axis_label_size)
    ax_spec.set_xlim(10.0, freq_max)
else:
    format_freq_xlabels(ax_spec, 
                        plot_axis_label = True, 
                        major_tick_spacing = major_freq_spacing, 
                        axis_label_size = axis_label_size, 
                        tick_label_size = tick_label_size)




# Set the y-axis labels
format_db_ylabels(ax_spec, 
                    plot_axis_label = True, 
                    major_tick_spacing = 20.0, 
                    axis_label_size = axis_label_size, 
                    tick_label_size = tick_label_size)

# Set the title
starttime = time_window - Timedelta(duration / 2, unit = "s")
endtime = time_window + Timedelta(duration / 2, unit = "s")
ax_spec.set_title(f"{station}, total power spectrum, {starttime:%Y-%m-%d %H:%M:%S} - {endtime:%H:%M:%S}", fontsize = title_size, fontweight = "bold")

### Save the figure ###
print("Saving the figure...")
if log_scale:
    figname = "jgr_specs_log_scale.png"
else:
    figname = "jgr_specs.png"
save_figure(fig, figname)