# Functions and classes for plotting
from pandas import Timedelta
from numpy import nditer
from matplotlib.pyplot import subplots, Circle
from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import DateFormatter

from utils_basic import GEO_STATIONS, GEO_COMPONENTS, days_to_timestamps, get_timeax_from_trace, get_unique_stations


VELOCITY_LABEL = f"Velocity (nm s$^{{-1}}$)"
VELOCITY_LABEL_SHORT = "Vel. (nm s$^{{-1}}$)"
X_SLOW_LABEL = "East slowness (s km$^{-1}$)"
Y_SLOW_LABEL = "North slowness (s km$^{-1}$)"

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
    axes[0].text(timeax[0] + 2 * dur / 50, -1, VELOCITY_LABEL, fontsize=station_label_size, verticalalignment="center", horizontalalignment="left")

    ## Set the x-axis limits
    axes[0].set_xlim([timeax[0], timeax[-1]])
    axes[1].set_xlim([timeax[0], timeax[-1]])
    axes[2].set_xlim([timeax[0], timeax[-1]])

    ## Format x lables
    axes[0].xaxis.set_major_formatter(DateFormatter('%m-%dT%H:%M:%S'))
    axes[1].xaxis.set_major_formatter(DateFormatter('%m-%dT%H:%M:%S'))
    axes[2].xaxis.set_major_formatter(DateFormatter('%m-%dT%H:%M:%S'))

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
    axes[0].xaxis.set_major_locator(SecondLocator(interval=major_tick_spacing))
    axes[1].xaxis.set_major_locator(SecondLocator(interval=major_tick_spacing))
    axes[2].xaxis.set_major_locator(SecondLocator(interval=major_tick_spacing))

    axes[0].xaxis.set_minor_locator(SecondLocator(interval=minor_tick_spacing))
    axes[1].xaxis.set_minor_locator(SecondLocator(interval=minor_tick_spacing))
    axes[2].xaxis.set_minor_locator(SecondLocator(interval=minor_tick_spacing))

    ## Turn off the y-axis labels and ticks
    axes[0].set_yticks([])
    axes[1].set_yticks([])
    axes[2].set_yticks([])

    axes[0].set_yticklabels([])
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    
    fig.patch.set_alpha(0.0)

    return fig, axes

## Function to plot the 3C seismograms and spectrograms computed using STFT of a list of stations
def plot_3c_waveforms_and_spectrograms(stream, specdict, 
                                       linewidth=0.1, xdim_per_comp=10, ydim_per_sta=3, ylim_wf=(-50, 50), dbmin=-50, dbmax=0, ylim_spec=(0, 200), 
                                       major_tick_spacing=5, minor_tick_spacing=1, station_label_size=12, axis_label_size=12, tick_label_size=12, title_size=15):
    components = GEO_COMPONENTS
    vel_label = VELOCITY_LABEL_SHORT

    stations = get_unique_stations(stream)
    numsta = len(stations)

    fig, axes = subplots(2 * numsta, 3, figsize=(xdim_per_comp * 3, ydim_per_sta * numsta * 2), sharex=True)

    for i, component in enumerate(components):
        for j, station in enumerate(stations):
            trace = stream.select(station=station, component=component)[0]

            ## Plot the waveforms
            signal = trace.data
            timeax_wf = get_timeax_from_trace(trace)

            ax = axes[2 * j, i]
            color = get_geo_component_color(component)
            ax.plot(timeax_wf, signal, color=color, linewidth=linewidth)

            ax.set_ylim(ylim_wf)

            if j == 0:
                title = component_to_title(component)
                ax.set_title(f"{title}", fontsize=15, fontweight="bold")

            if i == 0:
                ax.set_ylabel(vel_label, fontsize=tick_label_size)
                ax.text(0.01, 0.98, f"{station}", fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top") 


            ## Plot the spectrograms
            timeax_spec, freqax, spec = specdict[(station, component)]
            ax = axes[2 * j + 1, i]
            ax.pcolormesh(timeax_spec, freqax, spec, cmap="inferno", vmin=dbmin, vmax=dbmax)

            ax.set_ylim(ylim_spec)

            if j == numsta - 1:
                ax.set_xlabel("Time (UTC)", fontsize=axis_label_size)

            if i == 0:
                ax.set_ylabel("Freq. (Hz)", fontsize=axis_label_size)

    ax = axes[0, 0]
    ax.set_xlim(timeax_wf[0], timeax_wf[-1])
    ax.xaxis.set_major_locator(SecondLocator(interval=major_tick_spacing))
    ax.xaxis.set_minor_locator(SecondLocator(interval=minor_tick_spacing))

    format_utc_xlabels(axes)

    fig.patch.set_alpha(0.0)

    return fig, axes

## Function to plot the beamforming results
def plot_beam_images(xslow, yslow, beamdict, vmin=0, vmax=1):
    numcomp = len(beamdict)
    xslow = xslow * 1000
    yslow = yslow * 1000

    fig, axes = subplots(1, numcomp, figsize=(15, 5), sharex=True, sharey=True)

    for i, component in enumerate(beamdict.keys()):
        bimage = beamdict[component]
        ax = axes[i]
        cmap = ax.pcolor(xslow, yslow, bimage, cmap="inferno", vmin=vmin, vmax=vmax)

        if i == 0:
            ax.set_xlabel(X_SLOW_LABEL, fontsize=12)
            ax.set_ylabel(Y_SLOW_LABEL, fontsize=12)

        ax.set_title(component_to_title(component), fontsize=15, fontweight="bold")
        ax.set_aspect("equal")
        
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    ### Plot the reference slowness
    for ax in axes:
        ax.add_patch(Circle((0, 0), 1, edgecolor='limegreen', facecolor='none', linestyle="--"))
        ax.add_patch(Circle((0, 0), 2, edgecolor='limegreen', facecolor='none', linestyle="--"))
        ax.add_patch(Circle((0, 0), 3, edgecolor='limegreen', facecolor='none', linestyle="--"))
        ax.axhline(0, color="limegreen", linestyle="--")
        ax.axvline(0, color="limegreen", linestyle="--")

    ### Add the colorbar
    caxis = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cmap, cax=caxis)
    cbar.set_label("Normalized power", fontsize=12)
    
    return fig, axes
    
## Function to generate an amplitude scale bar
def plot_amp_scale_bar(ax, x, y, amplitude, scale, linewidth=1):
    ax.errorbar(x, y, yerr=amplitude * scale/2, xerr=None, capsize=2.5, color='black', fmt='-', linewidth=linewidth)

    return ax

## Function to format the x labels in UTC time
def format_utc_xlabels(axes, major_tick_spacing=60, minor_tick_spacing=15, tick_label_size=12):
    numrow = axes.shape[0]
    numcol = axes.shape[1]

    for i in range(numcol):
        ax = axes[numrow-1, i]
        ax.xaxis.set_major_formatter(DateFormatter('%m-%dT%H:%M:%S'))
        ax.xaxis.set_major_locator(SecondLocator(interval=major_tick_spacing))
        ax.xaxis.set_minor_locator(SecondLocator(interval=minor_tick_spacing))

        for label in ax.get_xticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('top')
            label.set_horizontalalignment('right')
            label.set_rotation(10)

    return axes

### Function for getting the color for the three geophone components
def get_geo_component_color(component):
    if component == "Z":
        color = "black"
    elif component == "1":
        color = "forestgreen"
    elif component == "2":
        color = "royalblue"
    else:
        raise ValueError("Invalid component name!")

    return color

### Function to convert component names to subplot titles
def component_to_title(component):
    if component == "Z":
        title = "Up"
    elif component == "1":
        title = "North"
    elif component == "2":
        title = "East"
    else:
        raise ValueError("Invalid component name!")

    return title