"""
Plot examples of resonance spectrograms
"""
#--------------------------------------------------------------------------------------------------
# Import libraries
#--------------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from os.path import join
from numpy import array
from pandas import read_csv
from scipy.signal import ShortTimeFFT
from obspy import UTCDateTime
from matplotlib.pyplot import subplots, Axes, Figure
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
from matplotlib.patches import Rectangle

from utils_basic import SPECTROGRAM_DIR as dirpath
from utils_spec import TraceSTFT, read_geo_stft, get_spectrogram_file_suffix
from utils_plot import format_datetime_xlabels, format_freq_ylabels, save_figure, GEO_PSD_LABEL as colorbar_label

#--------------------------------------------------------------------------------------------------
# Helpers
#--------------------------------------------------------------------------------------------------
def plot_spectrogram(ax: Axes, trace_stft: TraceSTFT, min_freq: float, max_freq: float, 
                     min_db: float = -10.0, max_db: float = 10.0,
                     station_label: str | None = None,
                     plot_time_ticks: bool = True,
                     plot_freq_ticks: bool = True,
                     major_time_tick_spacing: str = "1m", num_minor_time_ticks: int = 4,
                     major_freq_tick_spacing: float = 10.0, num_minor_freq_ticks: int = 5):
    """
    Plot a spectrogram of a trace.
    """

    # Convert to dB if needed
    trace_stft.to_db()
    
    # Get the PSD matrix
    psd_mat = trace_stft.psd_mat
    freqax = trace_stft.freqs
    timeax = trace_stft.times

    # Get the color map
    cmap = colormaps["inferno"]
    cmap.set_bad(color="darkgray")

    # Plot the spectrogram
    mappable = ax.pcolormesh(timeax, freqax, psd_mat, vmin=min_db, vmax=max_db, cmap=cmap)

    ax.set_xlim(timeax[0], timeax[-1])
    ax.set_ylim(min_freq, max_freq)

    # Add a station label if provided
    if station_label is not None:
        ax.text(0.02, 0.98, station_label, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=14, bbox=dict(facecolor="white", alpha=1.0, edgecolor="black"))

    # Format the x-axis labels
    if plot_time_ticks:
        format_datetime_xlabels(ax,
                                major_tick_spacing = major_time_tick_spacing, num_minor_ticks = num_minor_time_ticks, rotation = 15, ha = "right", va = "top")
    else:
        format_datetime_xlabels(ax,
                                plot_tick_label = False, plot_axis_label = False,
                                major_tick_spacing = major_time_tick_spacing, num_minor_ticks = num_minor_time_ticks, rotation = 15, ha = "right", va = "top")
        
    # Format the y-axis labels
    if plot_freq_ticks:
        format_freq_ylabels(ax,
                            major_tick_spacing = major_freq_tick_spacing, num_minor_ticks = num_minor_freq_ticks)
    else:
        format_freq_ylabels(ax,
                            plot_tick_label = False, plot_axis_label = False,
                            major_tick_spacing = major_freq_tick_spacing, num_minor_ticks = num_minor_freq_ticks)

    return mappable
                                
# Add the colorbar with a frame
def add_colorbar_with_frame(fig: Figure, ax: Axes, mappable: ScalarMappable, label: str, 
                            tick_label_size: float = 10,
                            cax_x: float = 0.05, cax_y: float = 0.15, cax_width: float = 0.4, cax_height: float = 0.03,
                            cax_pad_x: float = 0.05, cax_pad_y: float = 0.01):
    
    ax.add_patch(Rectangle((0.0, 0.0), cax_x + cax_width + cax_pad_x, cax_y + cax_height + 2 * cax_pad_y, transform=ax.transAxes, 
                        facecolor='white', edgecolor='black'))

    cax = ax.inset_axes([cax_x, cax_y, cax_width, cax_height])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize = tick_label_size)
    cbar.ax.set_xlabel(label, fontsize = tick_label_size)

#--------------------------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------------------------


# Input arguments
parser = ArgumentParser(description="Plot examples of resonance spectrograms")
parser.add_argument("--starttime1", type=str, help="Start time of the first example.")
parser.add_argument("--endtime1", type=str, help="End time of the first example.")
parser.add_argument("--starttime2", type=str, help="Start time of the second example.")
parser.add_argument("--endtime2", type=str, help="End time of the second example.")
parser.add_argument("--stations", nargs="+", type=str, help="Stations to plot.")

parser.add_argument("--min_db", type=float, default=-10.0, help="Minimum dB value for the first example.")
parser.add_argument("--max_db", type=float, default=10.0, help="Maximum dB value for the first example.")

parser.add_argument("--min_freq1", type=float, default=0.0, help="Minimum frequency in Hz for the first example.")
parser.add_argument("--max_freq1", type=float, default=100.0, help="Maximum frequency in Hz for the first example.")
parser.add_argument("--min_freq2", type=float, default=0.0, help="Minimum frequency in Hz for the second example.")
parser.add_argument("--max_freq2", type=float, default=100.0, help="Maximum frequency in Hz for the second example.")

parser.add_argument("--major_time_tick_spacing", type=str, default="1m", help="Major time tick spacing for the x-axis.")
parser.add_argument("--num_minor_time_ticks", type=int, default=4, help="Number of minor time ticks between major time ticks.")
parser.add_argument("--major_freq_tick_spacing1", type=float, default=10.0, help="Major frequency tick spacing for the first example.")
parser.add_argument("--num_minor_freq_ticks1", type=int, default=5, help="Number of minor frequency ticks between major frequency ticks for the first example.")
parser.add_argument("--major_freq_tick_spacing2", type=float, default=10.0, help="Major frequency tick spacing for the second example.")
parser.add_argument("--num_minor_freq_ticks2", type=int, default=5, help="Number of minor frequency ticks between major frequency ticks for the second example.")

parser.add_argument("--figwidth", type=float, default=10.0, help="Figure width in inches.")
parser.add_argument("--figheight", type=float, default=12.0, help="Figure height in inches.")
parser.add_argument("--hspace", type=float, default=0.1, help="Horizontal space between subplots.")
parser.add_argument("--panel_label_x", type=float, default=-0.05, help="X offset of the panel label.")
parser.add_argument("--panel_label_y", type=float, default=1.05, help="Y offset of the panel label.")
parser.add_argument("--panel_label_size", type=float, default=14, help="Size of the panel label.")

parser.add_argument("--window_length", type=float, default=1.0, help="Window length in seconds for computing the spectrogram.")
parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between consecutive windows for computing the spectrogram.")
parser.add_argument("--min_prom", type=float, default=15.0, help="Minimum prominence of a spectral peak.")
parser.add_argument("--min_rbw", type=float, default=15.0, help="Minimum reverse bandwidth of a spectral peak.")
parser.add_argument("--max_mean_db", type=float, default=15.0, help="Maximum mean dB value for excluding noise windows.")


args = parser.parse_args()
starttime1 = args.starttime1
endtime1 = args.endtime1
starttime2 = args.starttime2
endtime2 = args.endtime2
stations = args.stations
min_db = args.min_db
max_db = args.max_db
min_freq1 = args.min_freq1
max_freq1 = args.max_freq1
min_freq2 = args.min_freq2
max_freq2 = args.max_freq2
major_time_tick_spacing = args.major_time_tick_spacing
num_minor_time_ticks = args.num_minor_time_ticks
major_freq_tick_spacing1 = args.major_freq_tick_spacing1
num_minor_freq_ticks1 = args.num_minor_freq_ticks1
major_freq_tick_spacing2 = args.major_freq_tick_spacing2
num_minor_freq_ticks2 = args.num_minor_freq_ticks2
window_length = args.window_length
overlap = args.overlap
figwidth = args.figwidth
figheight = args.figheight
hspace = args.hspace
panel_label_x = args.panel_label_x
panel_label_y = args.panel_label_y
panel_label_size = args.panel_label_size

# Read the data
suffix_spec = get_spectrogram_file_suffix(window_length, overlap)

# Read the spectrograms
## First example
print("Reading the data for the first example...")
stft_dict1 = {}

for station in stations:
    filename = f"whole_deployment_daily_geo_stft_{station}_{suffix_spec}.h5"
    inpath = join(dirpath, filename)
    
    # Get the three components
    stream_stft = read_geo_stft(inpath, starttime = starttime1, endtime = endtime1, psd_only = True)
    trace_stft = stream_stft.get_total_psd()

    stft_dict1[station] = trace_stft.copy()

## Second example
print("Reading the data for the second example...")
stft_dict2 = {}

for station in stations:
    filename = f"whole_deployment_daily_geo_stft_{station}_{suffix_spec}.h5"
    inpath = join(dirpath, filename)
    
    # Get the three components
    stream_stft = read_geo_stft(inpath, starttime = starttime2, endtime = endtime2, psd_only = True)
    trace_stft = stream_stft.get_total_psd()

    stft_dict2[station] = trace_stft.copy()


# Plot the spectrograms
## Generate the figure
num_sta = len(stations)
fig, axes = subplots(num_sta, 2, figsize=(figwidth, figheight))
fig.subplots_adjust(hspace=hspace)

## Plot the spectrograms
print("Plotting the spectrograms...")
for i, station in enumerate(stations):
    if i == num_sta - 1:
        plot_time_ticks = True
    else:
        plot_time_ticks = False

    mappable = plot_spectrogram(axes[i, 0], stft_dict1[station], min_freq1, max_freq1, 
                     min_db = min_db, max_db = max_db, station_label = station, 
                     plot_time_ticks = plot_time_ticks, major_time_tick_spacing = major_time_tick_spacing, num_minor_time_ticks = num_minor_time_ticks,
                     major_freq_tick_spacing = major_freq_tick_spacing1, num_minor_freq_ticks = num_minor_freq_ticks1)

    _ = plot_spectrogram(axes[i, 1], stft_dict2[station], min_freq2, max_freq2, 
                    min_db = min_db, max_db = max_db, station_label = station, 
                     plot_time_ticks = plot_time_ticks, major_time_tick_spacing = major_time_tick_spacing, num_minor_time_ticks = num_minor_time_ticks,
                     major_freq_tick_spacing = major_freq_tick_spacing2, num_minor_freq_ticks = num_minor_freq_ticks2)
    
## Add the colorbar
add_colorbar_with_frame(fig, axes[0, 0], mappable, colorbar_label)

# Add the panel labels
axes[0, 0].text(panel_label_x, panel_label_y, "(a)",
                fontsize = panel_label_size, fontweight = "bold",
                transform = axes[0, 0].transAxes,
                ha = "right", va = "bottom", zorder = 1)

axes[0, 1].text(panel_label_x, panel_label_y, "(b)",
                fontsize = panel_label_size, fontweight = "bold",
                transform = axes[0, 1].transAxes,
                ha = "right", va = "bottom", zorder = 1)

# Save the figure
print("Saving the figure...")
save_figure(fig, "resonance_spectrogram_examples.png", dpi=300)