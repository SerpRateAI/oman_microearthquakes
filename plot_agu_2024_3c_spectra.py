# Plot the example 3C spectra of a geophone station for the AGU 2024 iPoster

from os.path import join
from numpy import isnan, nan
from argparse import ArgumentParser
from json import loads
from pandas import DataFrame, Timedelta
from pandas import concat, read_csv, read_hdf
from matplotlib.pyplot import subplots

from utils_basic import EASTMIN_WHOLE as min_east_array, EASTMAX_WHOLE as max_east_array, NORTHMIN_WHOLE as min_north_array, NORTHMAX_WHOLE as max_north_array
from utils_basic import HYDRO_DEPTHS as depth_dict, GEO_COMPONENTS as components
from utils_basic import SPECTROGRAM_DIR as dir_spec, PLOTTING_DIR as dir_plot
from utils_basic import CENTER_LONGITUDE as lon, CENTER_LATITUDE as lat
from utils_basic import IMAGE_DIR as dir_img
from utils_basic import get_geophone_coords, get_borehole_coords, str2timestamp
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_time_slice_from_geo_stft
from utils_plot import component2label, format_east_xlabels, format_db_ylabels, format_freq_xlabels, format_north_ylabels, format_depth_ylabels, get_geo_component_color, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Input parameters for plotting the station maps, hydrophone depth profiles, and 3C spectra of a geophone station for Liu et al. (2025a).")
parser.add_argument("--station_spec", type=str, help="Station whose 3C spectra will be plotted.")
parser.add_argument("--time_window", type=str, help="Time window for the 3C spectra.")
parser.add_argument("--window_length", type=float, default=300.0, help="Window length in seconds for computing the STFT.")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between consecutive windows for computing the STFT.")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type=float, default=10.0, help="Maximum mean dB value for excluding noise windows.")

parser.add_argument("--color_highlight", type=str, default="crimson", help="Color of the highlighted geophone markers.")
parser.add_argument("--color_missing", type=str, default="gray", help="Color of the missing resonance frequencies.")

# Parse the command line arguments
args = parser.parse_args()

station_spec = args.station_spec
time_window = args.time_window
window_length = args.window_length
overlap = args.overlap
min_prom = args.min_prom
min_rbw = args.min_rbw
max_mean_db = args.max_mean_db

color_highlight = args.color_highlight
color_missing = args.color_missing

# Constants
fig_width = 12.0
fig_height = 12.0
spec_gap = 0.05

axis_offset = 0.1

min_depth = 0.0
max_depth = 450.0

hydro_min = -0.5
hydro_max = 1.5

freq_min = 0.0
freq_max = 200.0

min_db = -40.0
max_db = 55.0
min_arrow_db = 0.0

water_level = 15.0
water_amp = 2.5
water_period = 0.2

linewidth_marker = 1.0
linewidth_coast = 0.2
linewidth_water = 2.0
linewidth_spec = 1.0
linewidth_arrow = 1.0

station_size = 100.0
borehole_size = 100.0
hydro_size = 100.0

station_font_size = 12.0
station_label_x = 7.0
station_label_y = 7.0

borehole_font_size = 12.0
borehole_label_x = 40.0
borehole_label_y = -40.0

location_font_size = 12.0

water_font_size = 12.0

major_dist_spacing = 25.0
major_depth_spacing = 50.0
major_freq_spacing = 50.0
major_db_spacing = 20.0

axis_label_size = 12.0
tick_label_size = 12.0
title_size = 16.0
component_label_size = 14.0
freq_label_size = 12.0

legend_size = 12.0

major_tick_length = 5.0
minor_tick_length = 2.0
tick_width = 1.0

frame_width = 1.0

arrow_gap = 5.0
arrow_length = 10.0
arrow_width = 1.0
arrow_headwidth = 5.0
arrow_headlength = 5.0

subplot_label_size = 18.0
subplot_offset_x = -0.04
subplot_offset_y = 0.02

### Plot the spectra of the 3C geophone station ###
# Create the figure and axes
fig, axes = subplots(3, 1, figsize = (fig_width, fig_height))

# Read the spectra
print(f"Reading the 3C spectra of station {station_spec} for time window {time_window}...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

filename = f"whole_deployment_daily_geo_stft_{station_spec}_{suffix_spec}.h5"
filepath = join(dir_spec, filename)

psd_dict = read_time_slice_from_geo_stft(filepath, time_window)

# Read the resonance frequencies
print(f"Getting the resonance frequencies of station {station_spec}...")
filename = f"stationary_harmonic_series_PR02549_base2.csv"
filepath = join(dir_spec, filename)

suffix_peak = get_spec_peak_file_suffix(min_prom, min_rbw, max_mean_db)

harmonic_df = read_csv(filepath)
mode_names = []
freqs_resonance = []
time_window = str2timestamp(time_window)
mode_marker_dfs = []
for mode_name in harmonic_df["mode_name"]:
    
    if mode_name.startswith("MH"):
        mode_marker_df = DataFrame({"mode_name": [mode_name], "frequency": [nan], "power_1": [min_arrow_db], "power_2": [min_arrow_db], "power_z": [min_arrow_db]})
    else:
        filename = f"stationary_resonance_properties_geo_{mode_name}_{suffix_spec}_{suffix_peak}.h5"
        filepath = join(dir_spec, filename)

        resonance_df = read_hdf(filepath)

        mode_marker_df = resonance_df.loc[(resonance_df["station"] == station_spec) & (resonance_df["time"] == time_window)]
        mode_marker_df = mode_marker_df[["frequency", "power_1", "power_2", "power_z"]]

        mode_marker_df["mode_name"] = mode_name
    
    mode_marker_df["mode_order"] = harmonic_df.loc[harmonic_df["mode_name"] == mode_name, "mode_order"].values[0]

    # Handle the exception for mode 15
    if mode_marker_df["mode_order"].values[0] == 15:
        mode_marker_df["power_2"] = mode_marker_df["power_1"]

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

print(mode_marker_df)
mode_marker_df.reset_index(drop = True, inplace = True)

# Plot the spectra and the resonance frequencies
print(f"Plotting the 3C spectra of station {station_spec}...")
for i, component in enumerate(components):
    print(f"Plotting component {component}...")
    ax = axes[i]

    spec = psd_dict[component]
    freqax = psd_dict["freqs"]

    color = get_geo_component_color(component)
    ax.plot(freqax, spec, color = color, linewidth = linewidth_spec, zorder = 2)

    # Plot the resonance frequencies
    flag = False
    for mode_order in mode_marker_df["mode_order"].unique():
        print(mode_order)
        mode_name = mode_marker_df.loc[mode_marker_df["mode_order"] == mode_order, "mode_name"].values[0]
        print(f"Plotting mode {mode_name}...")
        freq_resonance = mode_marker_df.loc[mode_marker_df["mode_order"] == mode_order, "frequency"].values[0]
        power = mode_marker_df.loc[mode_marker_df["mode_order"] == mode_order, f"power_{component.lower()}"].values[0]
        power = max(power, min_arrow_db)

        if i == 1:
            if mode_name.startswith("MH"):
                print((freq_resonance, power + arrow_gap))
                ax.annotate(f"?", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                            color = color_missing, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                            arrowprops=dict(facecolor=color_missing, edgecolor=color_missing, shrink=0.05, width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength))
            else:
                if not flag:
                    ax.annotate(f"Mode {mode_order}", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                                color = color_highlight, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                                arrowprops=dict(facecolor=color_highlight, edgecolor=color_highlight, shrink=0.05, width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength))
                    flag = True
                else:
                    ax.annotate(f"{mode_order}", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                                color = color_highlight, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                                arrowprops=dict(facecolor=color_highlight, edgecolor=color_highlight, shrink=0.05, width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength))
        else:
            if mode_name.startswith("MH"):
                ax.annotate(f"", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                            color = color_missing, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                            arrowprops=dict(facecolor=color_missing, edgecolor=color_missing, shrink=0.05, width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength))
            else:
                ax.annotate(f"", xy=(freq_resonance, power + arrow_gap), xytext=(freq_resonance, power + arrow_gap + arrow_length),
                            color = color_highlight, fontsize = freq_label_size, fontweight = "bold", va = "bottom", ha = "center",
                            arrowprops=dict(facecolor=color_highlight, edgecolor=color_highlight, shrink=0.05, width=arrow_width, headwidth=arrow_headwidth, headlength=arrow_headlength))
            
     
    # Set the axis limits
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(min_db, max_db)

    # Plot the component label
    label = component2label(component)
    ax.text(0.01, 0.97, label, fontsize = component_label_size, fontweight = "bold", verticalalignment = "top", horizontalalignment = "left", transform = ax.transAxes)

    # Set the x-axis labels
    if i == 2:
        format_freq_xlabels(ax,
                            plot_axis_label = True, plot_tick_label = True,
                            major_tick_spacing = major_freq_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
    else:
        format_freq_xlabels(ax,
                            plot_axis_label = False, plot_tick_label = False,
                            major_tick_spacing = major_freq_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
    
    # Set the y-axis labels
    format_db_ylabels(ax,
                      plot_axis_label = True, plot_tick_label = True,
                      major_tick_spacing = major_db_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size, major_tick_length=major_tick_length, minor_tick_length=minor_tick_length, tick_width = tick_width)
    
    # Set the title
    if i == 0:
        starttime = time_window - Timedelta(window_length / 2, unit = "s")
        endtime = time_window + Timedelta(window_length / 2, unit = "s")
        ax.set_title(f"3C spectra of {station_spec} showing all modes of the harmonic series", fontsize = title_size, fontweight = "bold")


# Save the figure
save_figure(fig, "agu_2024_3c_spectra.png")