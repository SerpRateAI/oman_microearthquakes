# Plot only the 3C spectra in Fig. 2 in Liu et al. (2025a) for presentation purposes
from os.path import join
from numpy import cos, pi, linspace, isnan, nan


from argparse import ArgumentParser
from json import loads
from scipy.interpolate import interp1d
from pandas import DataFrame, Timedelta
from pandas import concat, read_csv, read_hdf
from matplotlib.pyplot import figure
from matplotlib import colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
from colorcet import cm

from rasterio import open
from rasterio.plot import reshape_as_image
from cartopy.crs import Orthographic, Geodetic
import cartopy.feature as cfeature

from utils_basic import EASTMIN_WHOLE as min_east_array, EASTMAX_WHOLE as max_east_array, NORTHMIN_WHOLE as min_north_array, NORTHMAX_WHOLE as max_north_array
from utils_basic import HYDRO_DEPTHS as depth_dict, GEO_COMPONENTS as components
from utils_basic import SPECTROGRAM_DIR as dir_spec, MT_DIR as dir_mt
from utils_basic import CENTER_LONGITUDE as lon, CENTER_LATITUDE as lat
from utils_basic import IMAGE_DIR as dir_img
from utils_basic import str2timestamp
from utils_basic import power2db
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_time_slice_from_geo_stft
from utils_plot import format_db_ylabels, format_freq_xlabels, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for plotting the 3C spectra in Fig. 2 in Liu et al. (2025a) for presentation purposes.")
parser.add_argument("--station_spec", type=str, help="Station whose 3C spectra will be plotted.")
parser.add_argument("--time_window", type=str, help="Time window for the 3C spectra.")
parser.add_argument("--window_length_stft", type=float, default=300.0, help="Window length in seconds for computing the STFT.")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between consecutive windows for computing the STFT.")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type=float, default=15.0, help="Maximum mean dB value for excluding noise windows.")

parser.add_argument("--min_db", type=float, default=-20.0, help="Minimum dB value for the color scale.")
parser.add_argument("--max_db", type=float, default=80.0, help="Maximum dB value for the color scale.")
parser.add_argument("--min_arrow_db", type=float, default=0.0, help="Minimum dB value for the arrow color scale.")
parser.add_argument("--subarray_label_size", type=float, default=15.0, help="Font size of the subarray labels.")
parser.add_argument("--freq_label_size", type=float, default=15.0, help="Font size of the frequency labels.")
parser.add_argument("--axis_label_size", type=float, default=12.0, help="Font size of the axis labels.")
parser.add_argument("--tick_label_size", type=float, default=12.0, help="Font size of the tick labels.")
parser.add_argument("--title_size", type=float, default=15.0, help="Font size of the title.")
parser.add_argument("--legend_size", type=float, default=12.0, help="Font size of the legend.")
parser.add_argument("--arrow_gap", type=float, default=5.0, help="Gap between the arrow and the text.")
parser.add_argument("--arrow_length", type=float, default=10.0, help="Length of the arrow.")
parser.add_argument("--arrow_width", type=float, default=0.01, help="Width of the arrow.")
parser.add_argument("--arrow_head_width", type=float, default=5.0, help="Width of the arrow head.")
parser.add_argument("--arrow_head_length", type=float, default=5.0, help="Length of the arrow head.")

parser.add_argument("--color_geo", type=str, default="gold", help="Color of the geophone markers.")
parser.add_argument("--color_borehole", type=str, default="violet", help="Color of the borehole markers.")
parser.add_argument("--color_hydro", type=str, default="violet", help="Color of the hydrophone markers.")
parser.add_argument("--color_highlight", type=str, default="crimson", help="Color of the highlighted geophone markers.")
parser.add_argument("--color_missing", type=str, default="gray", help="Color of the missing resonance frequencies.")
parser.add_argument("--color_water", type=str, default="deepskyblue", help="Color of the water.")

parser.add_argument("--subarray_a_label_x", type=float, default=-5.0, help="X-coordinate of the subarray A label.")
parser.add_argument("--subarray_a_label_y", type=float, default=-40.0, help="Y-coordinate of the subarray A label.")
parser.add_argument("--subarray_b_label_x", type=float, default=-60.0, help="X-coordinate of the subarray B label.")
parser.add_argument("--subarray_b_label_y", type=float, default=70.0, help="Y-coordinate of the subarray B label.")

# Parse the command line arguments
args = parser.parse_args()

station_spec = args.station_spec
time_window = args.time_window
window_length_stft = args.window_length_stft
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

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

min_db = args.min_db
max_db = args.max_db
min_arrow_db = args.min_arrow_db

color_geo = args.color_geo
color_borehole = args.color_borehole
color_hydro = args.color_hydro
color_highlight = args.color_highlight
color_missing = args.color_missing
color_water = args.color_water

subarray_a_label_x = args.subarray_a_label_x
subarray_a_label_y = args.subarray_a_label_y
subarray_b_label_x = args.subarray_b_label_x
subarray_b_label_y = args.subarray_b_label_y

# Constants
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

### Plot the sum of the 3C spectra ###
print(f"Reading the 3C spectra of station {station_spec} for time window {time_window}...")
suffix_spec = get_spectrogram_file_suffix(window_length_stft, overlap)

filename = f"whole_deployment_daily_geo_stft_{station_spec}_{suffix_spec}.h5"
filepath = join(dir_spec, filename)

psd_dict = read_time_slice_from_geo_stft(filepath, time_window, db = False)

# Read the resonance frequencies
print(f"Getting the resonance frequencies of station {station_spec}...")
filename = f"stationary_harmonic_series_PR02549_base2.csv"
filepath = join(dir_spec, filename)

harmonic_df = read_csv(filepath)
time_window = str2timestamp(time_window)
suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)
mode_marker_dfs = []
for mode_name in harmonic_df["mode_name"]:
    
    if mode_name.startswith("MH"):
        mode_marker_df = DataFrame({"mode_name": [mode_name], "frequency": [nan]})
    else:
        filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
        filepath = join(dir_spec, filename)

        resonance_df = read_hdf(filepath)

        freq = resonance_df.loc[(resonance_df["station"] == station_spec) & (resonance_df["time"] == time_window), "frequency"].values[0]
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
print(f"Plotting the 3C spectra of station {station_spec}...")

# Compute the total spectrum
spec_total = psd_dict["1"] + psd_dict["2"] + psd_dict["Z"]
spec_total = power2db(spec_total)

freqax = psd_dict["freqs"]

# Generate the figure and axes
fig = figure(figsize = (fig_width, fig_height))
ax_spec = fig.add_axes([margin_x, margin_y, 1.0 - 2 * margin_x, 1.0 - 2 * margin_y])

# Plot the total spectrum
ax_spec.plot(freqax, spec_total, color = "black", linewidth = linewidth_spec, zorder = 2)

# Plot the resonance frequencies
flag = False
for mode_order in mode_marker_df["mode_order"].unique():
    mode_name = mode_marker_df.loc[mode_marker_df["mode_order"] == mode_order, "mode_name"].values[0]
    print(f"Plotting mode {mode_name}...")
    freq_resonance = mode_marker_df.loc[mode_marker_df["mode_order"] == mode_order, "frequency"].values[0]

    # Use interpolation to get the power at the resonance frequency
    power = interp1d(freqax, spec_total)(freq_resonance)
    power = max(power, min_arrow_db)


    if mode_name.startswith("MH"):
        ax_spec.annotate(f"?", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + 2 * arrow_length),
                    color = color_missing, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                    arrowprops=dict(facecolor=color_missing, edgecolor=color_missing, shrink=0.05, width=arrow_width, headwidth=arrow_head_width, headlength=arrow_head_length))
    else:
        if not flag:
            ax_spec.annotate(f"Mode {mode_order}", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                        color = color_highlight, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                        arrowprops=dict(facecolor=color_highlight, edgecolor=color_highlight, shrink=0.05, width=arrow_width, headwidth=arrow_head_width, headlength=arrow_head_length))
            flag = True
        else:
            ax_spec.annotate(f"{mode_order}", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                        color = color_highlight, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                        arrowprops=dict(facecolor=color_highlight, edgecolor=color_highlight, shrink=0.05, width=arrow_width, headwidth=arrow_head_width, headlength=arrow_head_length))
   

                    
# Set the axis limits
ax_spec.set_xlim(freq_min, freq_max)
ax_spec.set_ylim(min_db, max_db)


# Set the x-axis labels
format_freq_xlabels(ax_spec, 
                    plot_axis_label = True, 
                        major_tick_spacing = major_freq_spacing, 
                        axis_label_size = axis_label_size, 
                        tick_label_size = tick_label_size, 
                        major_tick_length=major_tick_length, 
                        minor_tick_length=minor_tick_length, 
                        tick_width = tick_width)

# Set the y-axis labels
format_db_ylabels(ax_spec, 
                    plot_axis_label = True, 
                    major_tick_spacing = major_db_spacing, 
                    axis_label_size = axis_label_size, 
                    tick_label_size = tick_label_size, 
                    major_tick_length=major_tick_length, 
                    minor_tick_length=minor_tick_length, 
                    tick_width = tick_width)

# Set the title
starttime = time_window - Timedelta(window_length_stft / 2, unit = "s")
endtime = time_window + Timedelta(window_length_stft / 2, unit = "s")
ax_spec.set_title(f"{station_spec}, total power spectrum, {starttime:%Y-%m-%d %H:%M:%S} - {endtime:%H:%M:%S}", fontsize = title_size, fontweight = "bold")

### Save the figure ###
print("Saving the figure...")
figname = "liu_2025a_specs_only.png"
save_figure(fig, figname)