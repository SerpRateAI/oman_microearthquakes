"""
Plot the spectrograms of a stationary resonance recorded on all locations of a station
"""

# Imports
from os.path import join
from argparse import ArgumentParser
from pandas import date_range, read_csv
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from utils_basic import SPECTROGRAM_DIR as dirpath, STARTTIME_HYDRO as starttime, ENDTIME_HYDRO as endtime, HYDRO_LOCATIONS as loc_dict
from utils_basic import get_mode_order
from utils_spec import get_spectrogram_file_suffix, read_hydro_stft
from utils_plot import HYDRO_PSD_LABEL as hydro_psd_label   
from utils_plot import add_colorbar, add_horizontal_scalebar, format_datetime_xlabels, format_freq_ylabels, save_figure

# Inputs
# Parse the arguments
parser = ArgumentParser(description = "Plot the spectrograms of a stationary resonance recorded on all locations of a station")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--station", type = str, help = "Station name")
parser.add_argument("--min_db", type = float, help = "Minimum dB")
parser.add_argument("--max_db", type = float, help = "Maximum dB")

parser.add_argument("--figwidth", type = float, help = "Figure width", default = 15.0)
parser.add_argument("--figheight", type = float, help = "Figure height", default = 15.0)
parser.add_argument("--window_length", type = float, help = "Window length in seconds", default = 300.0)
parser.add_argument("--overlap", type = float, help = "Overlap in seconds", default = 0.0)
parser.add_argument("--colormap_name", type = str, help = "Colormap name", default = "inferno")
parser.add_argument("--major_time_spacing", type = str, help = "Major time spacing", default = "30d")
parser.add_argument("--num_minor_time_ticks", type = int, help = "Number of minor time ticks", default = 6)
parser.add_argument("--major_freq_spacing", type = float, help = "Major frequency spacing", default = 0.5)
parser.add_argument("--num_minor_freq_ticks", type = int, help = "Number of minor frequency ticks", default = 5)
parser.add_argument("--location_label_x", type = float, help = "Location label x", default = 0.01)
parser.add_argument("--location_label_y", type = float, help = "Location label y", default = 0.94)
parser.add_argument("--location_label_fontsize", type = int, help = "Location label fontsize", default = 10)
parser.add_argument("--tick_label_size", type = int, help = "Tick label size", default = 10)
parser.add_argument("--cax_x", type = float, help = "Colorbar x", default = 0.02)
parser.add_argument("--cax_y", type = float, help = "Colorbar y", default = 0.15)
parser.add_argument("--cax_width", type = float, help = "Colorbar width", default = 0.1)
parser.add_argument("--cax_height", type = float, help = "Colorbar height", default = 0.03)
parser.add_argument("--cax_pad_x", type = float, help = "Colorbar pad x", default = 0.02)
parser.add_argument("--cax_pad_y", type = float, help = "Colorbar pad y", default = 0.01)
parser.add_argument("--cax_label_offset", type = float, help = "Colorbar label offset", default = -3.0)
parser.add_argument("--supertitle_fontsize", type = int, help = "Supertitle fontsize", default = 14)

args = parser.parse_args()
mode_name = args.mode_name
station_to_plot = args.station
min_db = args.min_db
max_db = args.max_db

figwidth = args.figwidth
figheight = args.figheight
window_length = args.window_length
overlap = args.overlap
colormap_name = args.colormap_name  
major_time_spacing = args.major_time_spacing
num_minor_time_ticks = args.num_minor_time_ticks
major_freq_spacing = args.major_freq_spacing
num_minor_freq_ticks = args.num_minor_freq_ticks
location_label_x = args.location_label_x
location_label_y = args.location_label_y
location_label_fontsize = args.location_label_fontsize
tick_label_size = args.tick_label_size
cax_x = args.cax_x
cax_y = args.cax_y
cax_width = args.cax_width
cax_height = args.cax_height
cax_pad_x = args.cax_pad_x
cax_pad_y = args.cax_pad_y
cax_label_offset = args.cax_label_offset
supertitle_fontsize = args.supertitle_fontsize

###
# Read the inputs
###
# Read the frequency range
filename = f"stationary_resonance_freq_ranges_hydro.csv"
inpath = join(dirpath, filename)
freq_range_df = read_csv(inpath)

min_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["min_freq_plot"].values[0]
max_freq = freq_range_df[freq_range_df["mode_name"] == mode_name]["max_freq_plot"].values[0]

# Read the spectrograms
print(f"Reading the spectrograms of {mode_name}...")
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

filename = f"whole_deployment_daily_hydro_stft_{station_to_plot}_{suffix_spec}.h5"
inpath = join(dirpath, filename)

locations = loc_dict[station_to_plot]
stream = read_hydro_stft(inpath,
                        starttime = starttime, endtime = endtime,
                        min_freq = min_freq, max_freq = max_freq,
                        psd_only = True)

###
# Plot the spectrograms
###
print(f"Plotting the spectrograms of {mode_name}...")
num_loc = len(locations)
fig, axes = subplots(num_loc, 1, figsize = (figwidth, figheight))

cmap = colormaps[colormap_name]
cmap.set_bad(color='darkgray')
for i, location in enumerate(locations):
    print(f"Plotting the spectrogram of {location}...")
    ax = axes[i]
    trace = stream.select(locations = location)[0]
    trace.to_db()
    psd_mat = trace.psd_mat
    freqax = trace.freqs
    timeax = trace.times

    mappable = ax.pcolormesh(timeax, freqax, psd_mat, shading = "auto", cmap = cmap, vmin = min_db, vmax = max_db)

    ax.set_xlim(starttime, endtime)
    ax.set_ylim(min_freq, max_freq)

    if i == num_loc - 1:
        format_datetime_xlabels(ax,
                                plot_axis_label = True, plot_tick_label = True,
                                major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                                date_format = "%Y-%m-%d")
    else:
        format_datetime_xlabels(ax,
                                plot_axis_label = False, plot_tick_label = False,
                                major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks)
    
    format_freq_ylabels(ax,
                        plot_axis_label = True, plot_tick_label = True,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)
    
    # Add the location label
    ax.text(location_label_x, location_label_y, f"{station_to_plot}.{location}",
            fontsize = location_label_fontsize, fontweight = "bold", transform = ax.transAxes,
            va = "top", ha = "left",
            bbox = dict(facecolor = "white", alpha = 1.0, edgecolor = "black"))
    
    # Add the colorbar
    if i == num_loc - 1:
        ax.add_patch(Rectangle((0.0, 0.0), cax_x + cax_width + cax_pad_x, cax_y + cax_height + 2 * cax_pad_y, transform=ax.transAxes, 
                      facecolor='white', edgecolor='black'))

        cax = ax.inset_axes([cax_x, cax_y, cax_width, cax_height])
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize = tick_label_size)
        cbar.ax.set_xlabel(hydro_psd_label, fontsize = tick_label_size, ha = "left", va = "top")
        cbar.ax.xaxis.set_label_coords(0.0, cax_label_offset)

# Add the supertitle
mode_order = get_mode_order(mode_name)
fig.suptitle(f"Mode {mode_order}", fontsize = supertitle_fontsize, fontweight = "bold", y = 0.9)

# Save the figure
figname = f"stationary_resonance_hydro_spec_all_locations_{mode_name}_{station_to_plot}.png"
save_figure(fig, figname)

