# Functions and classes for plotting
from os.path import join
from pandas import Timestamp, Timedelta
from numpy import arctan, abs, amax, angle, column_stack, cos, sin, linspace, pi, radians
from numpy.linalg import norm
from scipy.stats import gmean
from matplotlib.pyplot import subplots, Circle
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib.dates import DateFormatter, DayLocator, HourLocator, MinuteLocator, SecondLocator, MicrosecondLocator
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MultipleLocator

from utils_basic import GEO_STATIONS, GEO_COMPONENTS, PM_COMPONENT_PAIRS, WAVELET_COMPONENT_PAIRS, ROOTDIR_GEO, FIGURE_DIR, HYDRO_LOCATIONS
from utils_basic import days_to_timestamps, get_geophone_coords, get_datetime_axis_from_trace, get_unique_stations, hour2sec, timestamp_to_utcdatetime, utcdatetime_to_timestamp, sec2day
from utils_wavelet import mask_cross_phase

GROUND_VELOCITY_UNIT = "nm s$^{-1}$"
GROUND_VELOCITY_LABEL = f"Velocity (nm s$^{{-1}}$)"
GROUND_VELOCITY_LABEL_SHORT = "Vel. (nm s$^{{-1}}$)"
WAVE_VELOCITY_UNIT = "m s$^{-1}$"
WAVE_SLOWNESS_UNIT = "s km$^{-1}$"
PRESSURE_UNIT = "Pa"
APPARENT_VELOCITY_UNIT = "m s$^{-1}$"
X_SLOW_LABEL = "East slowness (s km$^{-1}$)"
Y_SLOW_LABEL = "North slowness (s km$^{-1}$)"

######
# Classes
######

## Class for storing the windowed particle-motion data
class ParticleMotionData:
    def __init__(self, starttime, endtime, data_dict):
        if not isinstance(starttime, Timestamp):
            raise ValueError("Invalid start time format!")

        if not isinstance(endtime, Timestamp):
            raise ValueError("Invalid end time format!")
        
        if not isinstance(data_dict, dict):
            raise ValueError("Invalid data format!")
        
        if ("2", "1") not in data_dict:
            raise ValueError("Data for the 1-2 component pair is missing!")
        if ("1", "Z") not in data_dict:
            raise ValueError("Data for the Z-1 component pair is missing!")
        if ("2", "Z") not in data_dict:
            raise ValueError("Data for the Z-2 component pair is missing!")
        
        self.starttime = starttime
        self.endtime = endtime
        self.midtime = starttime + (endtime - starttime) / 2
        self.data_dict = data_dict

    def get_data_for_plot(self, component_pair):
        data = self.data_dict[component_pair]
        data1 = data[:, 0]
        data2 = data[:, 1]
        midtime = self.midtime

        data1_plot = [midtime + Timedelta(seconds=reltime) for reltime in data1]
        data2_plot = data2

        return data1_plot, data2_plot

######
# Functions
######

###### Functions for plotting waveforms ######

## Function to make a plot the 3C seismograms of all geophone stations in a given time window
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
    add_scalebar(axes[0], timeax[0] + dur / 50, -1, scale_bar_amp, scale)
    axes[0].text(timeax[0] + 2 * dur / 50, -1, f"{scale_bar_amp} {GROUND_VELOCITY_UNIT}", fontsize=station_label_size, verticalalignment="center", horizontalalignment="left")

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

## Plot the hydrophone waveforms of one borehole in a given time window
def plot_windowed_hydro_waveforms(stream, station_df, 
                                station_to_plot = "A00", locations_to_plot = None, normalize=False,
                                scale = 1e-3, xdim = 10, ydim_per_loc = 2, 
                                linewidth_wf = 1, linewidth_sb = 0.5,
                                major_time_spacing = 5.0, minor_time_spacing = 1.0, major_depth_spacing=50.0, minor_depth_spacing=10.0,
                                station_label_size = 12, axis_label_size = 12, tick_label_size = 12,
                                depth_lim = (0.0, 420.0), location_label_offset = (0.05, -15), 
                                scalebar_offset = (0.2, -15), scalebar_length = 0.2, scalebar_label_size=12):
    unit = PRESSURE_UNIT

    ### Get the number of locations
    if locations_to_plot is None:
        locations_to_plot = HYDRO_LOCATIONS
    
    numloc = len(locations_to_plot)
    
    ### Generate the figure and axes
    fig, ax  = subplots(1, 1, figsize=(xdim, ydim_per_loc * numloc))

    ### Plot each location
    for location in locations_to_plot:
        #### Plot the trace
        try:
            trace = stream.select(station=station_to_plot, location=location)[0]
        except:
            print(f"Could not find {station_to_plot}.{location}")
            continue

        data = trace.data
        depth = station_df.loc[(station_df["station"] == station_to_plot) & (station_df["location"] == location), "depth"].values[0]
        if normalize:
            data = data / amax(abs(data))
        
        data = -data * scale + depth
        timeax = get_datetime_axis_from_trace(trace)

        ax.plot(timeax, data, color="darkviolet", linewidth=linewidth_wf)

        #### Add the location label
        offset_x = location_label_offset[0]
        offset_y = location_label_offset[1]
        label = f"{station_to_plot}.{location}"
        ax.text(timeax[0] + Timedelta(seconds=offset_x), depth + offset_y, label, fontsize=station_label_size, fontweight="bold", ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

    ### Set axis limits
    ax.set_xlim([timeax[0], timeax[-1]])
    ax.set_ylim(depth_lim)

    ### Reverse the y-axis
    ax.invert_yaxis()

    ### Plot the scale bar
    if not normalize:
        max_depth = ax.get_ylim()[0]
        if major_time_spacing > 1.0:
            scalebar_coord = (timeax[0] + Timedelta(seconds=scalebar_offset[0]), max_depth + scalebar_offset[1])
        else:
            scalebar_coord = (timeax[0] + scalebar_offset[0], max_depth + scalebar_offset[1])

        add_scalebar(ax, scalebar_coord, scalebar_length, scale, linewidth=linewidth_sb)
        ax.annotate(f"{scalebar_length} {unit}", scalebar_coord, xytext=(0.5, 0), textcoords="offset fontsize", fontsize=scalebar_label_size, ha="left", va="center")


    ### Format the x-axis labels
    ax  = format_datetime_xlabels(ax, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, axis_label_size=axis_label_size, tick_label_size=tick_label_size)

    ### Format the y-axis labels
    ax = format_depth_ylabels(ax, major_tick_spacing=major_depth_spacing, minor_tick_spacing=minor_depth_spacing, axis_label_size=axis_label_size, tick_label_size=tick_label_size)

    return fig, ax     

## Plot the hydrophone waveforms of one borehole in a given time window
def plot_cascade_zoom_in_hydro_waveforms(stream, station_df, window_dict,
                                station_to_plot = "A00", locations_to_plot = None, 
                                scale = 1e-3, xdim_per_win = 7, ydim_per_loc = 2, 
                                linewidth_wf = 1, linewidth_sb = 0.5,
                                major_time_spacing = 5.0, minor_time_spacing = 1.0, major_depth_spacing=50.0, minor_depth_spacing=10.0,
                                station_label_size = 12, axis_label_size = 12, tick_label_size = 12,
                                depth_lim = (0.0, 420.0), location_label_offset = (2, -15),
                                box_y_offset = 10.0, linewidth_box = 1.5,
                                scalebar_offset = (2, -15), scalebar_length = 0.2, scalebar_label_size=12):      

    unit = PRESSURE_UNIT
    min_depth = depth_lim[0]
    max_depth = depth_lim[1]

    ### Get the window parameters
    starttimes = window_dict["starttimes"]
    durations = window_dict["durations"]
    major_time_spacings = window_dict["major_time_spacings"]
    minor_time_spacings = window_dict["minor_time_spacings"]
    
    ### Determine if the numbers of windows are consistent
    num_windows = len(starttimes)
    if not all([len(starttimes) == num_windows, len(durations) == num_windows, len(major_time_spacings) == num_windows, len(minor_time_spacings) == num_windows]):
        raise ValueError("Inconsistent number of windows!")

    ### Convert the start times to Timestamp objects
    for i, starttime in enumerate(starttimes):
        if not isinstance(starttime, Timestamp):
            if isinstance(starttime, str):
                starttimes[i] = Timestamp(starttime, tz="UTC")
            else:
                raise ValueError("Invalid start time format!")
            
    starttime0 = starttimes[0]

    ### Get the number of locations
    if locations_to_plot is None:
        locations_to_plot = HYDRO_LOCATIONS
    
    numloc = len(locations_to_plot)
    
    ### Generate the figure and axes
    num_windows = len(starttimes)
    fig, axes  = subplots(1, num_windows, figsize=(xdim_per_win * num_windows, ydim_per_loc * numloc), sharey=True)

    ### Loop over each window
    for i, starttime in enumerate(starttimes):
        ax = axes[i]
        endtime = starttime + Timedelta(seconds=durations[i])
        duration = durations[i]
        major_time_spacing = major_time_spacings[i]
        minor_time_spacing = minor_time_spacings[i]

        ### Loop over each location
        for location in locations_to_plot:
            #### Plot the trace
            try:
                trace = stream.select(station=station_to_plot, location=location)[0]
            except:
                print(f"Could not find {station_to_plot}.{location}")
                continue

            data = trace.data
            depth = station_df.loc[(station_df["station"] == station_to_plot) & (station_df["location"] == location), "depth"].values[0]
            data = -data * scale + depth

            if major_time_spacing > 1.0:
                timeax = get_datetime_axis_from_trace(trace)
            else:
                timeax = trace.times()
                rel_time_start = (starttime - starttime0).total_seconds()
                timeax = timeax - rel_time_start

            ax.plot(timeax, data, color="darkviolet", linewidth=linewidth_wf)

            #### Add the location label
            if i == 0:
                offset_x = location_label_offset[0]
                offset_y = location_label_offset[1]
                label = f"{station_to_plot}.{location}"

                if major_time_spacing > 1.0:
                    ax.text(timeax[0] + Timedelta(seconds=offset_x), depth + offset_y, label, fontsize=station_label_size, fontweight="bold", ha="left", va="top", bbox=dict(facecolor="white", alpha=1))
                else:
                    ax.text(timeax[0] + offset_x, depth + offset_y, label, fontsize=station_label_size, fontweight="bold", ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        ### Set axis limits
        if major_time_spacing > 1.0:
            ax.set_xlim(starttime, endtime)
        else:
            ax.set_xlim(0, duration)

        ax.set_ylim(depth_lim)

        ### Reverse the y-axis
        ax.invert_yaxis()

        ### Plot the scale bar
        if i == 0:
            max_depth = ax.get_ylim()[0]
            if major_time_spacing > 1.0:
                scalebar_coord = (timeax[0] + Timedelta(seconds=scalebar_offset[0]), max_depth + scalebar_offset[1])
            else:
                scalebar_coord = (timeax[0] + scalebar_offset[0], max_depth + scalebar_offset[1])

            add_scalebar(ax, scalebar_coord, scalebar_length, scale, linewidth=linewidth_sb)
            ax.annotate(f"{scalebar_length} {unit}", scalebar_coord, xytext=(0.5, 0), textcoords="offset fontsize", fontsize=scalebar_label_size, ha="left", va="center")

        ### Format the x-axis labels
        if major_time_spacing > 1.0:
            ax  = format_datetime_xlabels(ax, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, axis_label_size=axis_label_size, tick_label_size=tick_label_size)
        else:
            ax  = format_rel_time_xlabels(ax, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, axis_label_size=axis_label_size, tick_label_size=tick_label_size)

        ### Format the y-axis labels
        if i == 0:
            ax = format_depth_ylabels(ax, major_tick_spacing=major_depth_spacing, minor_tick_spacing=minor_depth_spacing, axis_label_size=axis_label_size, tick_label_size=tick_label_size)

        ### Plot the box
        if i > 0:
            box_height = max_depth - min_depth - 2 * box_y_offset
            box = Rectangle((starttime, min_depth + box_y_offset), Timedelta(seconds=duration), box_height, edgecolor="crimson", facecolor="none", linestyle="-", linewidth=linewidth_box)
            ax = axes[i - 1]
            ax.add_patch(box)

    return fig, axes

###### Functions for plotting STFT power spectrograms ######   

# Function to plot the 3C seismograms and spectrograms computed using STFT of a list of stations
def plot_3c_waveforms_and_stfts(stream, specdict, 
                                       linewidth=0.1, xdim_per_comp=10, ydim_per_sta=3, ylim_wf=(-50, 50), dbmin=-50, dbmax=0, ylim_spec=(0, 200), 
                                       major_tick_spacing=5, minor_tick_spacing=1, station_label_size=12, axis_label_size=12, tick_label_size=12, title_size=15):
    components = GEO_COMPONENTS
    vel_label = GROUND_VELOCITY_LABEL_SHORT

    stations = get_unique_stations(stream)
    numsta = len(stations)

    fig, axes = subplots(2 * numsta, 3, figsize=(xdim_per_comp * 3, ydim_per_sta * numsta * 2), sharex=True)

    for i, component in enumerate(components):
        for j, station in enumerate(stations):
            trace = stream.select(station=station, component=component)[0]

            # Plot the waveforms
            signal = trace.data
            timeax_wf = get_datetime_axis_from_trace(trace)

            ax = axes[2 * j, i]
            color = get_geo_component_color(component)
            ax.plot(timeax_wf, signal, color=color, linewidth=linewidth)

            ax.set_ylim(ylim_wf)

            if j == 0:
                title = component_to_label(component)
                ax.set_title(f"{title}", fontsize=title_size, fontweight="bold")

            if i == 0:
                ax.set_ylabel(vel_label, fontsize=tick_label_size)
                ax.text(0.01, 0.98, f"{station}", fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top") 


            # Plot the spectrograms
            timeax_spec, freqax, spec = specdict[(station, component)]
            ax = axes[2 * j + 1, i]
            ax.pcolormesh(timeax_spec, freqax, spec, cmap="inferno", vmin=dbmin, vmax=dbmax)

            ax.set_ylim(ylim_spec)

            if j == numsta - 1:
                ax.set_xlabel("Time (UTC)", fontsize=axis_label_size)

            if i == 0:
                ax.set_ylabel("Freq. (Hz)", fontsize=axis_label_size)

    format_datetime_xlabels(axes)

    fig.patch.set_alpha(0.0)

    return fig, axes

# Plot the 3C long-term geophone spectrograms of a station computed using STFT
def plot_long_term_geo_stft_spectrograms(stream_spec,
                            xdim = 15, ydim_per_comp= 5, 
                            freq_lim=(0, 490), dbmin=-30, dbmax=0,
                            component_label_x = 0.01, component_label_y = 0.96,
                            datetime_format = "%Y-%m-%d",
                            major_time_spacing=24, minor_time_spacing=6, 
                            major_freq_spacing=100, minor_freq_spacing=20,
                            component_label_size=15, axis_label_size=12, tick_label_size=10, title_size=15,
                            time_tick_rotation=15, time_tick_va="top", time_tick_ha="right",
                            plot_total_psd = False, **kwargs):
    # Convert the power to dB
    stream_spec.to_db()

    # Extract the data
    trace_spec_z = stream_spec.select(component = "Z")[0]
    trace_spec_1 = stream_spec.select(component = "1")[0]
    trace_spec_2 = stream_spec.select(component = "2")[0]

    timeax = trace_spec_z.times
    freqax = trace_spec_z.freqs

    data_z = trace_spec_z.data
    data_1 = trace_spec_1.data
    data_2 = trace_spec_2.data

    # Plot the spectrograms
    if plot_total_psd:
        fig, axes = subplots(4, 1, figsize=(xdim, 3 * ydim_per_comp), sharex=True, sharey=True)
    else:
        fig, axes = subplots(3, 1, figsize=(xdim, 3 * ydim_per_comp), sharex=True, sharey=True)

    ax = axes[0]
    power_color = ax.pcolormesh(timeax, freqax, data_z, cmap = "inferno", vmin = dbmin, vmax = dbmax)
    label = component_to_label("Z")
    ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, tick_label_size = tick_label_size)

    ax = axes[1]
    power_color = ax.pcolormesh(timeax, freqax, data_1, cmap = "inferno", vmin = dbmin, vmax = dbmax)
    label = component_to_label("1")
    ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, tick_label_size = tick_label_size)

    ax = axes[2]
    power_color = ax.pcolormesh(timeax, freqax, data_2, cmap = "inferno", vmin = dbmin, vmax = dbmax)
    label = component_to_label("2")
    ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, tick_label_size = tick_label_size)

    if plot_total_psd:
        ax = axes[3]
        trace_total = kwargs["total_psd_trace"]
        trace_total.to_db()
        data_total = trace_total.data
        power_color = ax.pcolormesh(timeax, freqax, data_total, cmap = "inferno", vmin = dbmin, vmax = dbmax)
        label = "Total"
        ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
        format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, tick_label_size = tick_label_size)
    
    ax.set_ylim(freq_lim)

    major_time_spacing = hour2sec(major_time_spacing) # Convert hours to seconds
    minor_time_spacing = hour2sec(minor_time_spacing) # Convert hours to seconds
    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing, tick_label_size = tick_label_size, datetime_format = datetime_format, rotation = time_tick_rotation, vertical_align=time_tick_va, horizontal_align=time_tick_ha)

    # Add the colorbar
    bbox = axes[-1].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    cbar = add_power_colorbar(fig, power_color, position, tick_spacing=10, tick_label_size=tick_label_size)
    cbar.set_label("Power spectral density (dB)")

    station = trace_spec_z.station
    fig.suptitle(station, fontsize = title_size, fontweight = "bold", y = 0.9)

    return fig, axes, cbar

# Plot the long-term spectrograms of all locations of a hydrophone station computed using STFT
def plot_long_term_hydro_stft_spectrograms(stream_spec,
                            xdim = 15, ydim_per_loc= 5, 
                            freq_lim=(0, 490), dbmin=-30, dbmax=0,
                            component_label_x = 0.01, component_label_y = 0.96,
                            datetime_format = "%Y-%m-%d",
                            major_time_spacing=24, minor_time_spacing=6, 
                            major_freq_spacing=100, minor_freq_spacing=20,
                            component_label_size=15, axis_label_size=12, tick_label_size=10, title_size=15,
                            time_tick_rotation=15, time_tick_va="top", time_tick_ha="right"):
    # Convert the power to dB
    stream_spec.to_db()

    # Get all locations
    locations = stream_spec.get_locations()
    num_loc = len(locations)

    # Plot the spectrograms
    fig, axes = subplots(num_loc, 1, figsize=(xdim, 3 * ydim_per_loc), sharex=True, sharey=True)

    for i, location in enumerate(locations):
        trace_spec = stream_spec.select(location = location)[0]
        timeax = trace_spec.times
        freqax = trace_spec.freqs
        data = trace_spec.data
        
        ax = axes[i]
        power_color = ax.pcolormesh(timeax, freqax, data, cmap = "inferno", vmin = dbmin, vmax = dbmax)
        label = location
        ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
        format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, tick_label_size = tick_label_size)
    
    ax.set_ylim(freq_lim)

    major_time_spacing = hour2sec(major_time_spacing) # Convert hours to seconds
    minor_time_spacing = hour2sec(minor_time_spacing) # Convert hours to seconds
    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing, tick_label_size = tick_label_size, datetime_format = datetime_format, rotation = time_tick_rotation, vertical_align=time_tick_va, horizontal_align=time_tick_ha)

    # Add the colorbar
    bbox = axes[num_loc - 1].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    cbar = add_power_colorbar(fig, power_color, position, tick_spacing=10, tick_label_size=tick_label_size)
    cbar.set_label("Power spectral density (dB)")

    station = trace_spec.station
    fig.suptitle(station, fontsize = title_size, fontweight = "bold", y = 0.9)

    return fig, axes, cbar

# Plot the total PSD of a geophone station and the detected spectral peaks
def plot_geo_total_psd_and_peaks(trace_total, peak_df,
                            xdim = 15, ydim_per_row = 5, 
                            freq_lim=(0, 490), dbmin=-30, dbmax=0, rbwmin=0.1, rbwmax=0.5,
                            marker_size = 5,
                            panel_label_x = 0.01, panel_label_y = 0.96,
                            datetime_format = "%Y-%m-%d",
                            major_time_spacing=24, minor_time_spacing=6, 
                            major_freq_spacing=100, minor_freq_spacing=20,
                            panel_label_size=15, axis_label_size=12, tick_label_size=10, title_size=15,
                            time_tick_rotation=15, time_tick_va="top", time_tick_ha="right"):

    # Convert the power to dB
    trace_total.to_db()

    # Generate the figure and axes
    fig, axes = subplots(3, 1, figsize=(xdim, 3 * ydim_per_row), sharex=True, sharey=True)

    # Plot the total PSD
    ax = axes[0]

    color_norm = Normalize(vmin=dbmin, vmax=dbmax)

    timeax = trace_total.times
    freqax = trace_total.freqs
    power = trace_total.data
    power_color = ax.pcolormesh(timeax, freqax, power, cmap = "inferno", norm = color_norm)

    label = "Total PSD"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the spectral peak power
    ax = axes[1]

    peak_times = peak_df["time"]
    peak_freqs = peak_df["frequency"]
    peak_powers = peak_df["power"]

    ax.set_facecolor("lightgray")
    ax.scatter(peak_times, peak_freqs, c = peak_powers, s = marker_size, cmap = "inferno", norm = color_norm, edgecolors = None)
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the power colorbar
    bbox = ax.get_position()
    position = [bbox.x0 + bbox.width + 0.02, bbox.y0 , 0.01, bbox.height]
    power_cbar = add_power_colorbar(fig, power_color, position, tick_spacing=10, tick_label_size=tick_label_size, orientation = "vertical")

    # Plot the spectral peak reversed bandwidths
    ax = axes[2]

    peak_rbw = peak_df["reverse_bandwidth"]
    ax.set_facecolor("lightgray")
    rbw_color = ax.scatter(peak_times, peak_freqs, c = peak_rbw, s = marker_size, cmap = "viridis", norm=LogNorm(vmin = rbwmin, vmax = rbwmax))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, minor_tick_spacing = minor_freq_spacing, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the reversed-bandwidth colorbar
    bbox = ax.get_position()
    position = [bbox.x0 + bbox.width + 0.02, bbox.y0 , 0.01, bbox.height]
    qf_cbar = add_rbw_colorbar(fig, rbw_color, position, tick_label_size=tick_label_size, orientation = "vertical")

    # Format the x-axis labels
    major_time_spacing = hour2sec(major_time_spacing) # Convert hours to seconds
    minor_time_spacing = hour2sec(minor_time_spacing) # Convert hours to seconds
    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, minor_tick_spacing = minor_time_spacing, 
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size, datetime_format = datetime_format, rotation = time_tick_rotation, vertical_align=time_tick_va, horizontal_align=time_tick_ha)

    # Set the y-axis limits
    ax.set_ylim(freq_lim)

    # Set the title
    station = trace_total.station
    fig.suptitle(station, fontsize = title_size, fontweight = "bold", y = 0.9)

    return fig, axes

###### Functions for plotting CWT spectra ######

## Function to plot the 3C seismograms and spectrograms computed using CWT of a list of stations
def plot_3c_waveforms_and_cwts(stream, specs, 
                                 xdim_per_comp=10, ydim_per_sta=3, ylim_wf=(-50, 50), ylim_freq=(0.0, 500.0), dbmin=0.0, dbmax=30.0, 
                                 linewidth_wf=0.2, major_time_spacing=5.0, minor_time_spacing=1.0, major_freq_spacing=20, minor_freq_spacing=5, 
                                 station_label_x=0.02, station_label_y=0.92, station_label_size=15, axis_label_size=12, tick_label_size=12, title_size=15):
    components = GEO_COMPONENTS
    vel_label = GROUND_VELOCITY_LABEL_SHORT
    
    stations = specs.get_stations()
    numsta = len(stations)
    
    fig, axes = subplots(2 * numsta, 3, figsize=(xdim_per_comp * 3, ydim_per_sta * numsta * 2), sharex=True)
    
    for i, component in enumerate(components):
        for j, station in enumerate(stations):
                spec = specs.get_spectra_by_station_component(station, component)[0]
                freqax = spec.freqs
                timeax = spec.times
    
                ### Plot the waveforms

                ax = axes[2 * j, i]
                trace = stream.select(station=station, component=component)[0]
                signal = trace.data
                timeax_wf = get_datetime_axis_from_trace(trace)
    
                ax = axes[2 * j, i]
                color = get_geo_component_color(component)
                ax.plot(timeax_wf, signal, color=color, linewidth=linewidth_wf)

                if i == 0:
                    ax.text(station_label_x, station_label_y, f"{station}", fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))
                    ax.set_ylabel(vel_label, fontsize=axis_label_size)

                #### Set the waveform limits
                ax.set_ylim(ylim_wf)

                #### Plot the compoent name
                if j == 0:
                    title = component_to_label(component)
                    ax.set_title(f"{title}", fontsize=title_size, fontweight="bold")

                #### Set the y tick labels
                ax.yaxis.set_tick_params(labelsize=tick_label_size)

                ### Plot the power spectrogram
                    
                #### Get the power spectrum in dB
                power = spec.get_power()
                ax = axes[2 * j + 1, i]
                power_color = ax.pcolormesh(timeax, freqax, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

                #### Set the frequency limits
                ax.set_ylim(ylim_freq)

                #### Set the frequency tick spacing
                ax.yaxis.set_major_locator(MultipleLocator(major_freq_spacing))
                ax.yaxis.set_minor_locator(MultipleLocator(minor_freq_spacing))

                #### Set the frequency axis label
                if i == 0:
                    ax.set_ylabel("Freq. (Hz)", fontsize=axis_label_size)

                #### Set the y tick labels
                ax.yaxis.set_tick_params(labelsize=tick_label_size)
                
    ### Format the x-axis labels
    format_datetime_xlabels(axes, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

    ### Plot the colorbar at the bottom
    cax = fig.add_axes([0.35, 0.0, 0.3, 0.02])
    cbar = fig.colorbar(power_color, cax=cax, orientation="horizontal", label="Power (dB)")
    cbar.locator = MultipleLocator(base=10)
    cbar.update_ticks()

    return fig, axes


## Function to plot the amplitude of the CWTs of a stream
def plot_cwt_powers(specs,
                    xdim_per_comp=10, ydim_per_sta=3, freqlim=(0.0, 200), dbmin=0.0, dbmax=30, 
                    major_time_spacing="5S", minor_time_spacing="1S", major_freq_spacing=20, minor_freq_spacing=5,
                    station_label_size=15, axis_label_size=12, tick_label_size=12, title_size=15):
    stations = specs.get_stations()
    numsta = len(stations)

    fig, axes = subplots(numsta, 3, figsize=(xdim_per_comp * 3, ydim_per_sta * numsta), sharex=True, sharey=True)
    for i, component in enumerate(GEO_COMPONENTS):
        for j, station in enumerate(stations):
            spec = specs.get_spectra(station, compoent=component)[0]
            freqax = spec.freqs
            timeax = spec.times

            ### Get the power spectrum in dB
            power = spec.get_power()
            
            ### Plot the power
            ax = axes[j, i]
            power_color = ax.pcolormesh(timeax, freqax, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

            # ### Plot the noise window
            # box = Rectangle((noise_window[0], noise_window[2]), noise_window[1] - noise_window[0], noise_window[3] - noise_window[2], edgecolor="white", facecolor="none", linestyle=":", linewidth=linewidth_box)
            # ax.add_patch(box)

            if j == 0:
                title = component_to_label(component)
                ax.set_title(f"{title}", fontsize=title_size, fontweight="bold")

            if i == 0:
                ax.text(0.01, 0.98, f"{station}", color="white", fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top")
    
    ax = axes[0, 0]
    ax.set_ylim(freqlim)
    
    format_datetime_xlabels(axes, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)
    format_freq_ylabels(axes, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

    ### Add the colorbar
    caxis = fig.add_axes([0.35, -0.03, 0.3, 0.03])
    cbar = fig.colorbar(power_color, cax=caxis, orientation="horizontal", label="Power (dB)")

    return fig, axes, cbar

## Function for plotting the geophone waveforms and CWT spectrograms in successively zomed-in time-frequency windows
def plot_cascade_zoom_in_geo_waveforms_and_cwt_powers(stream, specs, window_dict, min_freq_filter, max_freq_filter,
                                stations_to_plot=None, component_to_plot=None,
                                xdim_per_sta=10, ydim_per_win=4, ylim_wf=(-50, 50), major_velocity_spacing=20, minor_velocity_spacing=5, dbmin=0.0, dbmax=30.0, 
                                linewidth_wf=0.2, linewidth_box=2,
                                station_label_x=0.02, station_label_y=0.95, station_label_size=15,
                                filter_label_x=0.02, filter_label_y=0.05, filter_label_size=15,
                                axis_label_size=12, tick_label_size=12):
    ### Get the window parameters
    starttimes = window_dict["starttimes"]
    durations = window_dict["durations"]
    min_freqs = window_dict["min_freqs"]
    max_freqs = window_dict["max_freqs"]
    major_time_spacings = window_dict["major_time_spacings"]
    minor_time_spacings = window_dict["minor_time_spacings"]
    major_freq_spacings = window_dict["major_freq_spacings"]
    minor_freq_spacings = window_dict["minor_freq_spacings"]

    ### Determine if the numbers of windows are consistent
    num_windows = len(starttimes)
    if not all([len(starttimes) == num_windows, len(durations) == num_windows, len(min_freqs) == num_windows, len(max_freqs) == num_windows]):
        raise ValueError("Inconsistent number of windows!")
    
    ### Convert the start times to Timestamp objects
    for i, starttime in enumerate(starttimes):
        if not isinstance(starttime, Timestamp):
            if isinstance(starttime, str):
                starttimes[i] = Timestamp(starttime, tz="UTC")
            else:
                raise ValueError("Invalid start time format!")
    
    ### Get the number of stations
    if stations_to_plot is None:
        stations_to_plot = get_unique_stations(stream)

    numsta = len(stations_to_plot)

    ### Get the component to plot
    if component_to_plot is None:
        component_to_plot = stream[0].stats.component

    ### Generate the figure and axes
    num_rows = num_windows + 1
    fig, axes = subplots(num_rows, numsta, figsize=(xdim_per_sta * numsta, ydim_per_win * num_rows))

    ### Plotting
    #### Loop over the stations
    for i, station in enumerate(stations_to_plot):
        ##### Plot the waveform
        trace = stream.select(station=station, component=component_to_plot)[0]
        waveform = trace.data
        timeax = get_datetime_axis_from_trace(trace)
        color = get_geo_component_color(component_to_plot)
        label = component_to_label(component_to_plot)

        ax = axes[0, i]
        ax.plot(timeax, waveform, color=color, linewidth=linewidth_wf)
        ax.text(station_label_x, station_label_y, f"{station}.{label}", fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))
        
        if i == 0:
            filter_label = f"Bandpass: {min_freq_filter}-{max_freq_filter} Hz"
            ax.text(filter_label_x, filter_label_y, f"{filter_label}", fontsize=filter_label_size, transform=ax.transAxes, ha="left", va="bottom")
        
        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(ylim_wf)

        major_time_spacing = major_time_spacings[0]
        minor_time_spacing = minor_time_spacings[0]
        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)
        
        if i == 0:
            format_vel_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_velocity_spacing, minor_tick_spacing=minor_velocity_spacing, tick_label_size=tick_label_size)
        else:
            format_vel_ylabels(ax, label=False, major_tick_spacing=major_velocity_spacing, minor_tick_spacing=minor_velocity_spacing, tick_label_size=tick_label_size)

        ##### Plot the CWT powers
        for j, (starttime, duration, min_freq, max_freq) in enumerate(zip(starttimes, durations, min_freqs, max_freqs)):
            spec = specs.get_spectra(station, component=component_to_plot)[0]
            freqax = spec.freqs
            timeax = spec.times

            power = spec.get_power()
            ax = axes[j + 1, i]
            power_color = ax.pcolormesh(timeax, freqax, power, cmap="inferno", vmin=dbmin, vmax=dbmax)
            endtime = starttime + Timedelta(seconds=duration)
            ax.set_xlim(starttime, endtime)
            ax.set_ylim(min_freq, max_freq)

            major_time_spacing = major_time_spacings[j]
            minor_time_spacing = minor_time_spacings[j]
            major_freq_spacing = major_freq_spacings[j]
            minor_freq_spacing = minor_freq_spacings[j]

            if j < num_windows - 1:
                format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing)
            else:
                format_datetime_xlabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

            if i > 0:
                format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
            else:
                format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)


            if j > 0 and j < num_windows:
                box = Rectangle((starttime, min_freq), Timedelta(seconds=duration), max_freq - min_freq, edgecolor="turquoise", facecolor="none", linestyle="-", linewidth=linewidth_box)
                ax = axes[j, i]
                ax.add_patch(box)

    #### Plot the power colorbar
    position = [0.42, 0.01, 0.15, 0.02]
    cbar = add_power_colorbar(fig, power_color, position, orientation="horizontal")

    return fig, axes, cbar

## Function for plotting the hydrophone waveforms and CWT spectrograms in successively zomed-in time-frequency windows
def plot_cascade_zoom_in_hydro_waveforms_and_cwt_powers(stream, specs, window_dict, min_freq_filter, max_freq_filter,
                                station_to_plot=None, locations_to_plot=None, component_to_plot=None,
                                xdim_per_sta=10, ydim_per_win=4, ylim_wf=(-5, 5), major_pressure_spacing=5e-3, minor_pressure_spacing=1e-3, dbmin=0.0, dbmax=30.0, 
                                linewidth_wf=0.2, linewidth_box=2,
                                station_label_x=0.02, station_label_y=0.95, station_label_size=15,
                                filter_label_x=0.02, filter_label_y=0.05, filter_label_size=15,
                                axis_label_size=12, tick_label_size=12):
    ### Get the window parameters
    starttimes = window_dict["starttimes"]
    durations = window_dict["durations"]
    min_freqs = window_dict["min_freqs"]
    max_freqs = window_dict["max_freqs"]
    major_time_spacings = window_dict["major_time_spacings"]
    minor_time_spacings = window_dict["minor_time_spacings"]
    major_freq_spacings = window_dict["major_freq_spacings"]
    minor_freq_spacings = window_dict["minor_freq_spacings"]

    ### Determine if the numbers of windows are consistent
    num_windows = len(starttimes)
    if not all([len(starttimes) == num_windows, len(durations) == num_windows, len(min_freqs) == num_windows, len(max_freqs) == num_windows]):
        raise ValueError("Inconsistent number of windows!")
    
    ### Convert the start times to Timestamp objects
    for i, starttime in enumerate(starttimes):
        if not isinstance(starttime, Timestamp):
            if isinstance(starttime, str):
                starttimes[i] = Timestamp(starttime, tz="UTC")
            else:
                raise ValueError("Invalid start time format!")
    
    ### Get the stations and locations to plot
    if station_to_plot is None:
        station_to_plot = stream[0].stats.station

    if locations_to_plot is None:
        locations_to_plot = HYDRO_LOCATIONS

    numloc = len(locations_to_plot)

    ### Generate the figure and axes
    num_rows = num_windows + 1
    fig, axes = subplots(num_rows, numloc, figsize=(xdim_per_sta * numloc, ydim_per_win * num_rows))

    ### Plotting
    #### Loop over the locations
    for i, location in enumerate(locations_to_plot):
        ##### Plot the waveform
        trace = stream.select(station=station_to_plot, location=location)[0]
        waveform = trace.data
        timeax = get_datetime_axis_from_trace(trace)
        color = "darkviolet"
        label = trace.stats.location

        ax = axes[0, i]
        ax.plot(timeax, waveform, color=color, linewidth=linewidth_wf)
        ax.text(station_label_x, station_label_y, f"{station_to_plot}.{label}", fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))
        
        if i == 0:
            filter_label = f"Bandpass: {min_freq_filter}-{max_freq_filter} Hz"
            ax.text(filter_label_x, filter_label_y, f"{filter_label}", fontsize=filter_label_size, transform=ax.transAxes, ha="left", va="bottom")
        
        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(ylim_wf)

        major_time_spacing = major_time_spacings[0]
        minor_time_spacing = minor_time_spacings[0]
        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)
        
        if i == 0:
            format_pressure_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_pressure_spacing, minor_tick_spacing=minor_pressure_spacing, tick_label_size=tick_label_size)
        else:
            format_pressure_ylabels(ax, label=False, major_tick_spacing=major_pressure_spacing, minor_tick_spacing=minor_pressure_spacing, tick_label_size=tick_label_size)

        ##### Plot the CWT powers
        for j, (starttime, duration, min_freq, max_freq) in enumerate(zip(starttimes, durations, min_freqs, max_freqs)):
            spec = specs.get_spectra(station_to_plot, location=location)[0]
            freqax = spec.freqs
            timeax = spec.times

            power = spec.get_power()
            ax = axes[j + 1, i]
            power_color = ax.pcolormesh(timeax, freqax, power, cmap="inferno", vmin=dbmin, vmax=dbmax)
            endtime = starttime + Timedelta(seconds=duration)
            ax.set_xlim(starttime, endtime)
            ax.set_ylim(min_freq, max_freq)

            major_time_spacing = major_time_spacings[j]
            minor_time_spacing = minor_time_spacings[j]
            major_freq_spacing = major_freq_spacings[j]
            minor_freq_spacing = minor_freq_spacings[j]

            if j < num_windows - 1:
                format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing)
            else:
                format_datetime_xlabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

            if i > 0:
                format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
            else:
                format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)


            if j > 0 and j < num_windows:
                box = Rectangle((starttime, min_freq), Timedelta(seconds=duration), max_freq - min_freq, edgecolor="turquoise", facecolor="none", linestyle="-", linewidth=linewidth_box)
                ax = axes[j, i]
                ax.add_patch(box)

    #### Plot the power colorbar
    position = [0.42, 0.01, 0.15, 0.02]
    cbar = add_power_colorbar(fig, power_color, position, orientation="horizontal")

    return fig, axes, cbar


## Function to plot the power, phase, and coherence of the station cross-spectra of a station pair
def plot_cwt_cross_station_spectra(cross_specs, station_pair,
                                    cohe_threshold=0.8, power_threshold=40.0, xdim_per_col=10, ydim_per_row=3.5, 
                                    freqlim=(0.0, 200), dbmin=0.0, dbmax=50.0, cohemin=0.5, cohemax=1.0,
                                    major_time_spacing=5.0, minor_time_spacing=1.0, major_freq_spacing=10, minor_freq_spacing=2, 
                                    component_label_x=0.02, component_label_y=0.95, compoent_label_size=13,
                                    threshold_label_x=0.02, threshold_label_y=0.95, threshold_label_size=13,
                                    axis_label_size=12, tick_label_size=10, title_size=15):
    components = GEO_COMPONENTS

    station1 = station_pair[0]
    station2 = station_pair[1]

    if station1 > station2:
        station1, station2 = station2, station1

    fig, axes = subplots(4, 3, figsize=(xdim_per_col * 3, ydim_per_row * 4), sharex=True, sharey=True)
    for i, component in enumerate(components):
        cross_spec = cross_specs.get_spectra(station1, station2, component=component)[0]

        ### Get the power spectra in dB
        power = cross_spec.get_power()
        phase = cross_spec.get_phase()

        freqs = cross_spec.freqs
        
        ### Plot the power
        timeax = cross_spec.times
        ax = axes[0, i]
        power_color = ax.pcolormesh(timeax, freqs, power, cmap="inferno", vmin=dbmin, vmax=dbmax)


        label = component_to_label(component)
        ax.text(component_label_x, component_label_y, label, fontsize=compoent_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
            
        ### Plot the phase
        ax = axes[1, i]
        cmap = colormaps["twilight_shifted"]
        cmap.set_bad(color="darkgray")
        phase_color = ax.pcolormesh(timeax, freqs, phase, cmap=cmap, vmin=-pi, vmax=pi)

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

        ### Plot the coherence
        coherence = cross_spec.coherence

        if coherence is None:
            raise ValueError("Coherence is not computed!")
        
        ax = axes[2, i]
        cohe_color = ax.pcolormesh(timeax, freqs, coherence, cmap="viridis", vmin=cohemin, vmax=cohemax)

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

        ### Plot the masked phase
        ax = axes[3, i]
        masked_phase = mask_cross_phase(phase, coherence, power, cohe_threshold=cohe_threshold, power_threshold=power_threshold)
        ax.pcolormesh(timeax, freqs, masked_phase, cmap=cmap, vmin=-pi, vmax=pi)

        label = f"Coherence > {cohe_threshold}, Power > {power_threshold}"
        if i == 0:
            ax.text(threshold_label_x, threshold_label_y, label, fontsize=threshold_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)


    ### Set the frequency limits
    ax = axes[0, 0]
    ax.set_ylim(freqlim)

    ### Plot the power, phase, and coherence colorbars
    bbox = axes[3, 0].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    power_cbar = add_power_colorbar(fig, power_color, position, orientation="horizontal")

    bbox = axes[3, 1].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    phase_cbar = add_phase_colorbar(fig, phase_color, position, orientation="horizontal")

    bbox = axes[3, 2].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    cohe_cbar = add_coherence_colorbar(fig, cohe_color, position, orientation="horizontal")

    ### Plot the super title
    fig.suptitle(f"{station1}-{station2}", fontsize=title_size, fontweight="bold", y=0.9)

    return fig, axes, power_cbar, phase_cbar, cohe_cbar

## Function to plot the power, phase, and coherence of the component cross-spectra of a station
def plot_cwt_cross_component_spectra(cross_specs, station, 
                                    cohe_threshold=0.8, power_threshold=40.0, xdim_per_col=10, ydim_per_row=3.5, 
                                    freqlim=(0.0, 200), dbmin=0.0, dbmax=50.0, cohemin=0.5, cohemax=1.0,
                                    major_time_spacing=5.0, minor_time_spacing=1.0, major_freq_spacing=10, minor_freq_spacing=2, 
                                    component_label_x=0.02, component_label_y=0.95, compoent_label_size=13,
                                    threshold_label_x=0.02, threshold_label_y=0.95, threshold_label_size=13,
                                    axis_label_size=12, tick_label_size=10, title_size=15):
    
    fig, axes = subplots(4, 3, figsize=(xdim_per_col * 3, ydim_per_row * 4), sharex=True, sharey=True)
    
    component_pairs = WAVELET_COMPONENT_PAIRS

    for i, (component1, component2) in enumerate(component_pairs):
        cross_spec = cross_specs.get_spectra(station, component1, component2)[0]

        ### Get the power spectra in dB
        power = cross_spec.get_power()
        phase = cross_spec.get_phase()

        ### Normalize the power and convert to dB
        freqs = cross_spec.freqs
        
        ### Plot the power
        timeax = cross_spec.times
        ax = axes[0, i]
        power_color = ax.pcolormesh(timeax, freqs, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

        label = f"{component_to_label(component1)}-{component_to_label(component2)}"
        ax.text(component_label_x, component_label_y, label, fontsize=compoent_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

        ### Plot the phase
        ax = axes[1, i]
        cmap = colormaps["twilight_shifted"]
        cmap.set_bad(color="darkgray")
        phase_color = ax.pcolormesh(timeax, freqs, phase, cmap=cmap, vmin=-pi, vmax=pi)

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

        ### Plot the coherence
        coherence = cross_spec.coherence

        if coherence is None:
            raise ValueError("Coherence is not computed!")
        
        ax = axes[2, i]
        cohe_color = ax.pcolormesh(timeax, freqs, coherence, cmap="viridis", vmin=cohemin, vmax=cohemax)

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

        ### Plot the masked phase
        ax = axes[3, i]
        masked_phase = mask_cross_phase(phase, coherence, power, cohe_threshold=cohe_threshold, power_threshold=power_threshold)
        ax.pcolormesh(timeax, freqs, masked_phase, cmap=cmap, vmin=-pi, vmax=pi)

        if i == 0:
            label = f"Coherence > {cohe_threshold}, Power > {power_threshold} dB"
            ax.text(threshold_label_x, threshold_label_y, label, fontsize=threshold_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        ax.set_xlabel("Time (UTC)", fontsize=axis_label_size)

        format_datetime_xlabels(ax, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

    ### Set the frequency and time limits
    ax = axes[0, 0]
    ax.set_xlim([timeax[0], timeax[-1]])
    ax.set_ylim(freqlim)

    ### Plot the colorbars
    bbox = axes[3, 0].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    power_cbar = add_power_colorbar(fig, power_color, position, orientation="horizontal")

    bbox = axes[3, 1].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    phase_cbar = add_phase_colorbar(fig, phase_color, position, orientation="horizontal")

    bbox = axes[3, 2].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    cohe_cbar = add_coherence_colorbar(fig, cohe_color, position, orientation="horizontal")

    ### Plot the super title
    fig.suptitle(f"{station}", fontsize=title_size, fontweight="bold", y=0.9)
    
    return fig, axes, power_cbar, phase_cbar, cohe_cbar

## Plot frequency-phase-difference pairs extracted from the cross-spectra
def plot_freq_phase_pairs(freq_phi_dict, station_pair, freq_lim=(0, 200),
                          reference_vels=[150, 300, 600], linewidth=1,
                          major_freq_spacing=10, minor_freq_spacing=2, major_phase_spacing=1, minor_phase_spacing=0.5,
                          title_size=15, axis_label_size=12, tick_label_size=12):
    unit = WAVE_VELOCITY_UNIT
    components = GEO_COMPONENTS

    loc_df = get_geophone_coords()
    station1 = station_pair[0]
    station2 = station_pair[1]

    if station1 > station2:
        station1, station2 = station2, station1

    ### Compute the reference differential travel times
    east1 = loc_df.loc[loc_df["name"] == station1, "east"].values[0]
    north1 = loc_df.loc[loc_df["name"] == station1, "north"].values[0]
    east2 = loc_df.loc[loc_df["name"] == station2, "east"].values[0]
    north2 = loc_df.loc[loc_df["name"] == station2, "north"].values[0]

    distance = norm([east1 - east2, north1 - north2])
    difftime_dict = {}
    for i, vel in enumerate(reference_vels):
        difftime_pos = distance / vel
        difftime_neg = -distance / vel
        difftime_dict[f"{vel:.0f}"] = [difftime_pos, difftime_neg]

    fig, axes = subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for i, component in enumerate(components):
        freq_phi_pairs = freq_phi_dict[(station1, station2, component)]
        freqs = freq_phi_pairs[:, 0]
        phases = freq_phi_pairs[:, 1]
        color = get_geo_component_color(component)
        title = component_to_label(component)

        ### Plot the frequency-phase pairs
        ax = axes[i]
        ax.scatter(freqs, phases, color=color, marker="+", s=10)

        ### Plot the reference lines
        freqmin = freq_lim[0]
        freqmax = freq_lim[1]
        for j, (vel, difftimes) in enumerate(difftime_dict.items()):
            for difftime in difftimes:
                x1 = freqmin
                y1 = 0
                x2 = freqmax
                y2 = difftime * (freqmax-freqmin) * 2 * pi

                ax.plot([x1, x2], [y1, y2], color="crimson", linestyle=":", linewidth=linewidth)
                
                if (i == 0) and (difftime > 0):
                    rotation = arctan((y2 - y1) / (x2 - x1) * (freqmax - freqmin) / 2 / pi) * 180 / pi
                    label_y = pi / 2
                    label_x = freqmin + label_y / difftime / 2 / pi
                    if j == 0:
                        ax.text(label_x, label_y, f"{vel} {unit}", fontsize=tick_label_size, color="crimson", ha="left", va="bottom", rotation=rotation, rotation_mode="anchor")
                    else:
                        ax.text(label_x, label_y, f"{vel}", fontsize=tick_label_size, color="crimson", ha="left", va="bottom", rotation=rotation, rotation_mode="anchor")

        ax.set_xlim([freqmin, freqmax])
        ax.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax.set_xlabel("Frequency (Hz)", fontsize=axis_label_size)
        ax.set_title(title, fontsize=title_size, fontweight="bold")

    ax = axes[0]
    
    ax.set_xlim([freqmin, freqmax])
    ax.set_ylim([-pi, pi])
   
    ax.xaxis.set_major_locator(MultipleLocator(base=major_freq_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(base=minor_freq_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(base=major_phase_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(base=minor_phase_spacing))
    ax.xaxis.set_major_locator(MultipleLocator(base=major_freq_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(base=minor_freq_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(base=major_phase_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(base=minor_phase_spacing))

    
    ax.set_ylabel("Phase (rad)", fontsize=axis_label_size)

    fig.suptitle(f"{station1}-{station2}", fontsize=title_size, fontweight="bold")

    return fig, axes

## Plot the waveforms, CWT spectrograms, and cross-correlation functioins of a station pair
def plot_waveforms_cwt_and_xcorr(stream, specs, cc_dict, station_pair,
                                 xdim_per_col=10, ydim_per_row=3.0,
                                 freq_lim=(0, 200), wf_lim=(-40, 40), lag_time_lim=(-0.1, 0.1), dbmin=0, dbmax=30,
                                 linewidth_wf=0.2, linewidth_cc=1,
                                 station_label_x=0.02, station_label_y=0.92, station_label_size=15,
                                 axis_label_size=12, tick_label_size=10, title_size=15,
                                 major_time_spacing=10, minor_time_spacing=1, major_freq_spacing=10, minor_freq_spacing=2, major_vel_spacing=20, minor_vel_spacing=5, dbspacing=10,
                                 cbar_y_offset=0.08, cbar_width=0.01):

    components = GEO_COMPONENTS

    ### Generate the figure and axes
    fig, axes = subplots(5, 3, figsize=(xdim_per_col * 3, ydim_per_row * 4))

    ### Plot the waveforms of station1
    station1 = station_pair[0]
    for i, component in enumerate(components):
        trace = stream.select(station=station1, component=component)[0]
        waveform = trace.data
        timeax = get_datetime_axis_from_trace(trace)
        color = get_geo_component_color(component)

        ax = axes[0, i]
        ax.plot(timeax, waveform, color=color, linewidth=linewidth_wf)

        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(wf_lim)

        if i == 0:
            ax.text(station_label_x, station_label_y, station1, fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        title = component_to_label(component)
        ax.set_title(title, fontsize=title_size, fontweight="bold")

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_vel_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_vel_spacing, minor_tick_spacing=minor_vel_spacing, tick_label_size=tick_label_size)
        else:
            format_vel_ylabels(ax, label=False, major_tick_spacing=major_vel_spacing, minor_tick_spacing=minor_vel_spacing, tick_label_size=tick_label_size)

    ### Plot the power spectra of station1
    for i, component in enumerate(components):
        spec = specs.get_spectra(station1, component=component)[0]
        power = spec.get_power()
        freqs = spec.freqs
        timeax = spec.times

        ax = axes[1, i]
        power_color = ax.pcolormesh(timeax, freqs, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(freq_lim)

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

    ### Plot the waveforms of station2
    station2 = station_pair[1]
    for i, component in enumerate(components):
        trace = stream.select(station=station2, component=component)[0]
        waveform = trace.data
        timeax = get_datetime_axis_from_trace(trace)
        color = get_geo_component_color(component)

        ax = axes[2, i]
        ax.plot(timeax, waveform, color=color, linewidth=linewidth_wf)

        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(wf_lim)

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

        if i == 0:
            ax.text(station_label_x, station_label_y, station2, fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        if i == 0:
            format_vel_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_vel_spacing, minor_tick_spacing=minor_vel_spacing, tick_label_size=tick_label_size)
        else:
            format_vel_ylabels(ax, label=False, major_tick_spacing=major_vel_spacing, minor_tick_spacing=minor_vel_spacing, tick_label_size=tick_label_size)

    ### Plot the power spectra of station2
    for i, component in enumerate(components):
        spec = specs.get_spectra(station2, component=component)[0]
        power = spec.get_power()
        freqs = spec.freqs
        timeax = spec.times

        ax = axes[3, i]
        power_color = ax.pcolormesh(timeax, freqs, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(freq_lim)

        if i == 0:
            format_freq_ylabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)
        else:
            format_freq_ylabels(ax, label=False, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

        format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

    ### Plot the cross-correlation functions
    for i, component in enumerate(components):
        cc = cc_dict[(station1, station2, component)]
        numpts = len(cc)
        sampling_rate = stream[0].stats.sampling_rate
        sampling_int = 1 / sampling_rate

        min_time_cc = -(numpts - 1) / 2 * sampling_int
        max_time_cc = (numpts - 1) / 2 * sampling_int
        timeax = linspace(min_time_cc, max_time_cc, numpts)

        ax = axes[4, i]
        color = get_geo_component_color(component)
        ax.plot(timeax, cc, color=color, linewidth=linewidth_cc)

        ax.set_xlim(lag_time_lim)
        ax.set_ylim([-1, 1])

        if i == 0:
            ax.text(station_label_x, station_label_y, f"{station1}-{station2}", fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        ax.set_xlabel("Time lag (s)", fontsize=axis_label_size)
        ax.set_ylabel("CC", fontsize=axis_label_size)

    ### Add the poewr colorbar
    bbox = axes[4, 1].get_position()
    position = [bbox.x0, bbox.y0 - cbar_y_offset, bbox.width, cbar_width]
    add_power_colorbar(fig, power_color, position, orientation="horizontal", tick_spacing=dbspacing)

    return fig, axes

## Function to plot the beamforming results
def plot_beam_images(xslow, yslow, beamdict, 
                     vmin=0, vmax=1, reference_vels=[150, 300, 600], slow_spacing=2, subarray="A", station_selection="inner",
                     axis_label_size=12, tick_label_size=12, title_size=15, label_size=10, label_angle=90):
    
    xslow_label = X_SLOW_LABEL
    yslow_label = Y_SLOW_LABEL
    vel_unit = WAVE_VELOCITY_UNIT
    
    numcomp = len(beamdict)
    xslow = xslow * 1000
    yslow = yslow * 1000

    fig, axes = subplots(1, numcomp, figsize=(15, 5), sharex=True, sharey=True)

    for i, component in enumerate(beamdict.keys()):
        bimage = beamdict[component]
        ax = axes[i]
        cmap = ax.pcolor(xslow, yslow, bimage, cmap="inferno", vmin=vmin, vmax=vmax)

        if i == 0:
            ax.set_xlabel(xslow_label, fontsize=tick_label_size)
            ax.set_ylabel(yslow_label, fontsize=tick_label_size)

        ax.set_title(component_to_label(component), fontsize=title_size, fontweight="bold")
        ax.set_aspect("equal")
        
        for j, vel in enumerate(reference_vels):
            slow = 1 / vel * 1000
            ax.add_patch(Circle((0, 0), slow, edgecolor='turquoise', facecolor='none', linestyle="--"))
            ax.add_patch(Circle((0, 0), slow, edgecolor='turquoise', facecolor='none', linestyle="--"))
            ax.add_patch(Circle((0, 0), slow, edgecolor='turquoise', facecolor='none', linestyle="--"))

            if i == 0:
                x = slow * cos(radians(label_angle))
                y = slow * sin(radians(label_angle))
                if j == 0:
                    ax.text(x, y, f"{vel} {vel_unit}", fontsize=label_size, fontweight="bold", color="turquoise", va="bottom", ha="left")
                else:
                    ax.text(x, y, f"{vel}", fontsize=label_size, fontweight="bold", color="turquoise", va="bottom", ha="left")

    ### Plot the reference apparent velocities and labels
    for ax in axes:
        
        ax.axhline(0, color="turquoise", linestyle="--")
        ax.axvline(0, color="turquoise", linestyle="--")

    ### Set the axis ticks and labels
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(slow_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(slow_spacing))

    ### Add the colorbar
    bbox = axes[-1].get_position()
    position = [bbox.x0 + bbox.width + 0.01, bbox.y0, 0.01, bbox.height]
    caxis = fig.add_axes(position)
    cbar = fig.colorbar(cmap, cax=caxis)
    cbar.set_label("Normalized power", fontsize=12)

    ### Add the title
    if subarray == "A":
        if station_selection == "inner":
            title = "Array A, inner"
        else:
            title = "Array A, all"
    else:
        if station_selection == "inner":
            title = "Array B, inner"
        else:
            title = "Array B, all"

    fig.suptitle(title, fontsize=title_size, fontweight="bold")
    
    return fig, axes

## Function for plotting the 3C particle-motion variation with time
def plot_3c_particle_motion_with_time(stream, specs, pm_data_list,
                             linewidth_wf=0.2, linewidth_pm=0.2, freqmin=50, freqmax=100, ymax_wf=1.05, ymax_pm=1.05, dbmin=0.0, dbmax=30,
                             major_time_spacing=10, minor_time_spacing=1, major_freq_spacing=10, minor_freq_spacing=2, dbspacing=10,
                            component_label_size=10, axis_label_size=10, tick_label_size=10, title_size=15):
    components = GEO_COMPONENTS
    component_pairs = PM_COMPONENT_PAIRS

    ## Compute the correct aspect ratio for the plot
    num_windows = len(pm_data_list)
    starttime_plot = timestamp_to_utcdatetime(pm_data_list[0].starttime)
    endtime_plot = timestamp_to_utcdatetime(pm_data_list[-1].endtime)
    stream_plot = stream.slice(starttime_plot, endtime_plot)
    dur_plot = endtime_plot - starttime_plot

    ax_aspect_plot = 1 / num_windows

    ## Generate the figure and axes
    fig, axes = subplots(7, 1, figsize=(12, 13))

    ## Plot the CWT spectrogram
    station = stream[0].stats.station
    ax = axes[0]
    spec = specs.get_spectra(station, component="Z")[0]
    power = spec.get_power(reference_type="mean")
    timeax = spec.times
    freqs = spec.freqs
    power_color = ax.pcolormesh(timeax, freqs, power, shading="auto", cmap="inferno", vmin=dbmin, vmax=dbmax)

    cbar_coord = [0.91, 0.8, 0.01, 0.07]
    cbar = add_power_colorbar(fig, power_color, cbar_coord, orientation="vertical", dbspacing=dbspacing)

    ax.text(0.01, 0.85, component_to_label("Z"), fontsize=component_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))
    
    ax.set_ylim(freqmin, freqmax)
    ax.yaxis.set_major_locator(MultipleLocator(major_freq_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_freq_spacing))

    ax.set_ylabel("Freq. (Hz)", fontsize=axis_label_size)
    ax.yaxis.set_tick_params(labelsize=tick_label_size)

    ax_aspect_phys = sec2day(dur_plot) / (freqmax -  freqmin) * ax_aspect_plot
    ax.set_aspect(ax_aspect_phys)

    format_datetime_xlabels(ax, label=False, axis_label_size=axis_label_size, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)
    format_freq_ylabels(ax, abbreviation=True, axis_label_size=axis_label_size, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing, tick_label_size=tick_label_size)

    ## Plot the waveforms
    for i, component in enumerate(components):
        trace = stream_plot.select(station=station, component=component)[0]
        data = trace.data
        timeax = get_datetime_axis_from_trace(trace)
        color = get_geo_component_color(component)

        ## Plot the waveform
        ax = axes[i + 1]
        ax.plot(timeax, data, color=color, linewidth=linewidth_wf)

        ## Label the component
        ax.text(0.01, 0.85, component_to_label(component), fontsize=component_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

        ## Set the axis limits
        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(-ymax_wf, ymax_wf)

        ## Compute and set the aspect ratio
        ax_aspect_phys = sec2day(dur_plot) / 2 / ymax_wf * ax_aspect_plot
        ax.set_aspect(ax_aspect_phys)

        if i < len(components) - 1:
            format_datetime_xlabels(ax, label=False, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)
        else:
            format_datetime_xlabels(ax, axis_label_size=axis_label_size, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

    ## Plot the particle motions
    for i, component_pair in enumerate(component_pairs):

        ax = axes[i + 4]
        for pm_data in pm_data_list:
            data1, data2 = pm_data.get_data_for_plot(component_pair)
            ax.plot(data1, data2, color="lightgray", linewidth=linewidth_pm)

        ## Compute and set the aspect ratio
        ax_aspect_phys = sec2day(dur_plot) / 2 / ymax_pm * ax_aspect_plot
        ax.set_aspect(ax_aspect_phys)
        
        ## Set the axis limits
        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(-ymax_pm, ymax_pm)

        ## Set the labels
        component1 = component_pair[0]
        component2 = component_pair[1]

        label1 = component_to_label(component1)
        label2 = component_to_label(component2)

        ax.set_xlabel(label1, fontsize=axis_label_size)
        ax.set_ylabel(label2, fontsize=axis_label_size)

        ax.xaxis.set_ticklabels([])


    # ## Plot the polarization analysis
    # timeax = get_datetime_axis_from_trace(stream_plot[0])
    # window_length = pol_params.window_length
    # timeax_pol = timeax[window_length - 1:]

    # ### Strike and dip
    # ax_strike = axes[7]

    # ax_strike.plot(timeax_pol, pol_params.strike, "mediumpurple", label="Strike")
    # ax_dip = ax_strike.twinx()
    # ax_dip.plot(timeax_pol, pol_params.dip, "teal", label="Dip ($^\circ$)")

    # ax_strike.set_ylim(0, 180)
    # ax_strike.set_ylabel("Strike ($^\circ$)", color="mediumpurple")

    # ax_dip.set_ylim(-90, 90)
    # ax_dip.set_ylabel("Dip ($^\circ$)", color="teal")

    # ax_strike.yaxis.set_major_locator(MultipleLocator(30))
    # ax_strike.spines['left'].set_color('mediumpurple')
    # ax_strike.tick_params(axis='y', colors='mediumpurple')

    # ax_dip.yaxis.set_major_locator(MultipleLocator(30))
    # ax_dip.spines['right'].set_color('teal')
    # ax_dip.tick_params(axis='y', colors='teal')

    # ### Ellipticity
    # ax_ellip = axes[8]

    # ax_ellip.plot(timeax_pol, pol_params.ellipticity, "black", label="Ellipticity")
    # ax_ellip.set_ylim(0, 1)
    # ax_ellip.set_ylabel("Ellipticity")

    # ### Strength and planarity
    # ax_strength = axes[9]

    # ax_strength.plot(timeax_pol, pol_params.strength, "mediumpurple", label="Strength")
    # ax_planar = ax_strength.twinx()
    # ax_planar.plot(timeax_pol, pol_params.planarity, "teal", label="Planarity")

    # ax_strength.set_ylim(0, 1.05)
    # ax_planar.set_ylim(0, 1.05)

    # ax_strength.set_ylabel("Strength", color="mediumpurple")
    # ax_planar.set_ylabel("Planarity", color="teal")

    # ax_strength.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax_strength.spines['left'].set_color('mediumpurple')
    # ax_strength.tick_params(axis='y', colors='mediumpurple')

    # ax_planar.yaxis.set_major_locator(MultipleLocator(0.2))
    # ax_planar.spines['right'].set_color('teal')
    # ax_planar.tick_params(axis='y', colors='teal')

    # ## Set the x axis limits
    # axes[-1].set_xlim(starttime_plot, endtime_plot)

    # ## Format the x axis
    # # ax = axes[-1]
    # # format_datetime_xlabels(ax, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, tick_label_size=tick_label_size)

    ## Set the title
    title = f"{station}"
    fig.suptitle(title, y=0.89, fontsize=title_size, fontweight="bold")

    return fig, axes, cbar

## Function for getting windowed particle-motion data
def get_windowed_pm_data(stream_in, window_length):
    ### Determine if all three components are present
    components = GEO_COMPONENTS
    component_pairs = PM_COMPONENT_PAIRS

    if len(stream_in) < 3:
        raise ValueError("All three components are required!")
    else:
        for component in components:
            if len(stream_in.select(component=component)) == 0:
                raise ValueError(f"Component {component} is missing!")
            
    ### Divide the stream into windows
    starttime = stream_in[0].stats.starttime
    endtime = stream_in[0].stats.endtime
    window_start = starttime
    window_end = starttime + window_length
    windowed_streams = []
    while window_end <= endtime:
        stream = stream_in.slice(window_start, window_end)
        windowed_streams.append(stream)
        window_start += window_length
        window_end += window_length

    ### Extract the particle motion for each window
    pm_data_list = []
    for stream in windowed_streams:
        starttime = stream[0].stats.starttime
        starttime = Timestamp(starttime.datetime, tz="UTC")
        endtime = stream[0].stats.endtime
        endtime = Timestamp(endtime.datetime, tz="UTC")

        data_dict = {}
        for pair in component_pairs:
            component1 = pair[0]
            component2 = pair[1]

            trace1 = stream.select(component=component1)[0]
            trace2 = stream.select(component=component2)[0]
            data1 = trace1.data
            data2 = trace2.data

            data_dict[pair] = column_stack((data1, data2))

        pm_data = ParticleMotionData(starttime, endtime, data_dict)
        pm_data_list.append(pm_data)

    return pm_data_list



## Function to format the x labels in datetime
def format_datetime_xlabels(ax, label=True, datetime_format = '%m-%dT%H:%M:%S', major_tick_spacing = 60, minor_tick_spacing = 15, axis_label_size = 12, tick_label_size = 12, rotation = 0, vertical_align = "top", horizontal_align = "center"):
    if label:
        ax.set_xlabel("Time (UTC)", fontsize=axis_label_size)


    ax.xaxis.set_major_formatter(DateFormatter(datetime_format))

    ax.xaxis.set_major_locator(MultipleLocator(sec2day(major_tick_spacing)))
    ax.xaxis.set_minor_locator(MultipleLocator(sec2day(minor_tick_spacing)))

    for label in ax.get_xticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment(vertical_align)
        label.set_horizontalalignment(horizontal_align)
        label.set_rotation(rotation)

    return ax

## Function format the x labels in relative time
def format_rel_time_xlabels(ax, label=True, major_tick_spacing=60, minor_tick_spacing=15, axis_label_size=12, tick_label_size=12):
    if label:
        ax.set_xlabel("Time (s)", fontsize=axis_label_size)

    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))

    for label in ax.get_xticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('top')
        label.set_horizontalalignment('center')

    return ax

## Function to format the y labels in depth
def format_depth_ylabels(ax, label=True, major_tick_spacing=50, minor_tick_spacing=10, axis_label_size=12, tick_label_size=12):
    if label:
        ax.set_ylabel("Depth (m)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('center')
        label.set_horizontalalignment('right')

    return ax


## Function to format the y labels in frequency
def format_freq_ylabels(ax, label=True, abbreviation=False, major_tick_spacing=20, minor_tick_spacing=5, axis_label_size=12, tick_label_size=12):
    if label:
        if abbreviation:
            ax.set_ylabel("Freq. (Hz)", fontsize=axis_label_size)
        else:
            ax.set_ylabel("Frequency (Hz)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('center')
        label.set_horizontalalignment('right')

    return ax

## Function to format the ylabel in ground velocity
def format_vel_ylabels(ax, label=True, abbreviation=False, major_tick_spacing=100, minor_tick_spacing=50, axis_label_size=12,  tick_label_size=12):
    if label:
        if abbreviation:
            ax.set_ylabel(f"{GROUND_VELOCITY_LABEL_SHORT}", fontsize=axis_label_size)
        else:
            ax.set_ylabel(f"{GROUND_VELOCITY_LABEL}", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('center')
        label.set_horizontalalignment('right')

    return ax

## Function to format the ylabel in pressure
def format_pressure_ylabels(ax, label=True, abbreviation=False, major_tick_spacing=5, minor_tick_spacing=1, axis_label_size=12,  tick_label_size=12):
    if label:
        if abbreviation:
            ax.set_ylabel(f"Press. (Pa)", fontsize=axis_label_size)
        else:
            ax.set_ylabel(f"Pressure (Pa)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('center')
        label.set_horizontalalignment('right')

    return ax

# Add a power colorbar to the plot
def add_power_colorbar(fig, color, position, orientation="horizontal", tick_spacing=5, axis_label_size=12, tick_label_size=12):
    cax = fig.add_axes(position)
    cbar = fig.colorbar(color, cax=cax, orientation=orientation)
    cbar.set_label("Power (dB)", fontsize=axis_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)
    cbar.locator = MultipleLocator(tick_spacing)
    cbar.update_ticks() 

    return cbar

# Add a phase colorbar to the plot
def add_phase_colorbar(fig, color, position, orientation="horizontal", tick_spacing=pi/2, axis_label_size=12, tick_label_size=12):
    cax = fig.add_axes(position)
    cbar = fig.colorbar(color, cax=cax, orientation=orientation)
    cbar.set_label("Phase (rad)", fontsize=axis_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)
    cbar.locator = MultipleLocator(tick_spacing)
    cbar.update_ticks() 

    return cbar

# Add a coherence colorbar to the plot
def add_coherence_colorbar(fig, color, position, orientation="horizontal", tick_spacing=0.1, axis_label_size=12, tick_label_size=12):
    cax = fig.add_axes(position)
    cbar = fig.colorbar(color, cax=cax, orientation=orientation)
    cbar.set_label("Coherence", fontsize=axis_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)
    cbar.locator = MultipleLocator(tick_spacing)
    cbar.update_ticks() 

    return cbar

# Add a reversed-bandwidth colorbar to the plot
def add_rbw_colorbar(fig, color, position, orientation="horizontal", axis_label_size=12, tick_label_size=12):
    cax = fig.add_axes(position)
    cbar = fig.colorbar(color, cax=cax, orientation=orientation)
    cbar.set_label("$Q/f_0$ (Hz$^{-1}$)", fontsize=axis_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)
    #cbar.locator = MultipleLocator(tick_spacing)
    cbar.update_ticks() 

## Function to generate a scale bar
def add_scalebar(ax, coordinates, amplitude, scale, linewidth=1):
    x = coordinates[0]
    y = coordinates[1]
    ax.errorbar(x, y, yerr=amplitude * scale/2, xerr=None, capsize=2.5, color='black', fmt='-', linewidth=linewidth)

    return ax

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
def component_to_label(component):
    if component == "Z":
        title = "Up"
    elif component == "1":
        title = "North"
    elif component == "2":
        title = "East"
    else:
        raise ValueError("Invalid component name!")

    return title

### Function to convert the frequency band to a label
def freq_band_to_label(freqmin, freqmax):
    if freqmin is not None and freqmax is not None:
        label = f"Bandpass {freqmin}-{freqmax} Hz"
    elif freqmin is not None and freqmax is None:
        label = f"Highpass {freqmin} Hz"
    elif freqmin is None and freqmax is not None:
        label = f"Lowpass {freqmax} Hz"
    else:
        label = "Raw data"

    return label

### Function for saving a figure
def save_figure(fig, filename, outdir=FIGURE_DIR, dpi=300):
    fig.patch.set_alpha(0)

    outpath = join(outdir, filename)

    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {outpath}")