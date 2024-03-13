# Functions and classes for plotting
from matplotlib.pyplot import subplots
import matplotlib.dates as mdates
from utils_basic import GEO_STATIONS, GEO_COMPONENTS, VELOCITY_UNIT, days_to_timestamps
from pandas import Timedelta

## Function to make a plot the 3C seismograms of all stations in a given time window
def plot_3c_seismograms(stream, stations=GEO_STATIONS, xdim_per_comp=25, ydim_per_sta=0.3, 
                        scale=1e-4, linewidth=0.5, major_tick_spacing=5, minor_tick_spacing=1,
                        station_label=True, station_label_size=12, axis_label_size=12, tick_label_size=12, title_size=15, scale_bar_amp=100):

    stations_to_plot = stations
    numsta = len(stations_to_plot)
    fig, axes = subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(3 * xdim_per_comp, ydim_per_sta * numsta))

    ## Loop over the stations
    for i, station in enumerate(stations_to_plot):

        for component in GEO_COMPONENTS:
            try:
                trace = stream.select(station=station, channel=f"*{component}")[0]
                data = trace.data * scale + i
            except:
                print(f"Could not find {station}.GH{component}")
                continue

            ### Convert the time axis to Pandas Timestamps
            timeax = trace.times("matplotlib")
            timeax = days_to_timestamps(timeax)

            if component == "Z":
                axes[0].plot(timeax, data, color="black", linewidth=linewidth)
            elif component == "1":
                axes[1].plot(timeax, data, color="forestgreen", linewidth=linewidth)
            elif component == "2":
                axes[2].plot(timeax, data, color="royalblue", linewidth=linewidth)

            dur = timeax[-1] - timeax[0]

            ### Add station labels
            if station_label:
                axes[0].text(timeax[0] - dur / 50, i, station, fontsize=station_label_size, verticalalignment="center", horizontalalignment="right")

    ## Plot the scale bar
    plot_amp_scale_bar(axes[0], timeax[0] + dur / 50, -1, scale_bar_amp, scale)
    axes[0].text(timeax[0] + 2 * dur / 50, -1, f"{scale_bar_amp} {VELOCITY_UNIT}", fontsize=station_label_size, verticalalignment="center", horizontalalignment="left")

    ## Set the x-axis limits
    axes[0].set_xlim([timeax[0], timeax[-1]])
    axes[1].set_xlim([timeax[0], timeax[-1]])
    axes[2].set_xlim([timeax[0], timeax[-1]])

    ## Format x lables
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%dT%H:%M:%S'))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%dT%H:%M:%S'))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%m-%dT%H:%M:%S'))

    for label in axes[0].get_xticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('top')
        label.set_horizontalalignment('right')
        label.set_rotation(10)

    for label in axes[1].get_xticklabels():
        label.set_fontsize(tick_label_size)
        label.set_verticalalignment('top')
        label.set_horizontalalignment('right')
        label.set_rotation(10)

    for label in axes[2].get_xticklabels():
        label.set_fontsize(tick_label_size)
        label.set_verticalalignment('top')
        label.set_horizontalalignment('right')
        label.set_rotation(10)
    
    axes[0].set_xlabel("Time (UTC)", fontsize=axis_label_size)
    axes[1].set_xlabel("Time (UTC)", fontsize=axis_label_size)
    axes[2].set_xlabel("Time (UTC)", fontsize=axis_label_size)

    ## Set the y-axis limits
    axes[0].set_ylim([-1.5, numsta])
    axes[1].set_ylim([-1.5, numsta])
    axes[2].set_ylim([-1.5, numsta])

    ## Set tick label size
    axes[0].tick_params(axis="x", which="major", labelsize=tick_label_size)
    axes[1].tick_params(axis="x", which="major", labelsize=tick_label_size)
    axes[2].tick_params(axis="x", which="major", labelsize=tick_label_size)

    axes[0].tick_params(axis="y", which="major", labelsize=tick_label_size)
    axes[1].tick_params(axis="y", which="major", labelsize=tick_label_size)
    axes[2].tick_params(axis="y", which="major", labelsize=tick_label_size)

    ## Set titles
    axes[0].set_title("Up", fontsize=title_size, fontweight="bold")
    axes[1].set_title("North", fontsize=title_size, fontweight="bold")
    axes[2].set_title("East", fontsize=title_size, fontweight="bold")

    ## Set x label spacing to 60 seconds
    axes[0].xaxis.set_major_locator(mdates.SecondLocator(interval=major_tick_spacing))
    axes[1].xaxis.set_major_locator(mdates.SecondLocator(interval=major_tick_spacing))
    axes[2].xaxis.set_major_locator(mdates.SecondLocator(interval=major_tick_spacing))

    axes[0].xaxis.set_minor_locator(mdates.SecondLocator(interval=minor_tick_spacing))
    axes[1].xaxis.set_minor_locator(mdates.SecondLocator(interval=minor_tick_spacing))
    axes[2].xaxis.set_minor_locator(mdates.SecondLocator(interval=minor_tick_spacing))

    ## Turn off the y-axis labels and ticks
    axes[0].set_yticks([])
    axes[1].set_yticks([])
    axes[2].set_yticks([])

    axes[0].set_yticklabels([])
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    
    fig.patch.set_alpha(0.0)

    return fig, axes

## Function to generate an amplitude scale bar
def plot_amp_scale_bar(ax, x, y, amplitude, scale, unit=VELOCITY_UNIT, linewidth=1, fontsize=12):
    ax.errorbar(x, y, yerr=amplitude * scale/2, xerr=None, capsize=2.5, color='black', fmt='-', linewidth=linewidth)

    return ax