# Functions and classes for plotting
from os.path import join
from pandas import Timestamp, Timedelta
from pandas import date_range, merge
from numpy import arctan, array, abs, amax, angle, column_stack, cos, sin, linspace, log, pi, radians
from numpy.linalg import norm
from scipy.stats import gmean

from matplotlib.pyplot import figure, subplots, get_cmap, Circle
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib.dates import DateFormatter, DayLocator, HourLocator, MinuteLocator, SecondLocator, MicrosecondLocator, DateLocator
from matplotlib.dates import num2date, date2num
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter, MaxNLocator
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

import colorcet as cc

from utils_basic import GEO_STATIONS, GEO_COMPONENTS, STARTTIME_GEO, ENDTIME_GEO, STARTTIME_HYDRO, ENDTIME_HYDRO, ROOTDIR_GEO, FIGURE_DIR, HYDRO_LOCATIONS
from utils_basic import EASTMIN_WHOLE , EASTMAX_WHOLE, NORTHMIN_WHOLE, NORTHMAX_WHOLE, HYDRO_DEPTHS
from utils_basic import days_to_timestamps, get_borehole_coords, get_geophone_coords, get_datetime_axis_from_trace, get_geo_sunrise_sunset_times, get_unique_stations, hour2sec, timestamp_to_utcdatetime, str2timestamp, sec2day
# from utils_wavelet import mask_cross_phase

HYDRO_COLOR = "tab:purple"
GROUND_VELOCITY_UNIT = "nm s$^{-1}$"
GROUND_VELOCITY_LABEL = f"Velocity (nm s$^{{-1}}$)"
GROUND_VELOCITY_LABEL_SHORT = "Vel. (nm s$^{{-1}}$)"
WAVE_VELOCITY_UNIT = "m s$^{-1}$"
WAVE_SLOWNESS_UNIT = "s km$^{-1}$"
PRESSURE_UNIT = "Pa"
PRESSURE_LABEL = "Pressure (mPa)"
APPARENT_VELOCITY_LABEL = "Velocity (m s$^{-1}$)"
APPARENT_VELOCITY_LABEL_SHORT = "Vel. (m s$^{-1}$)"
HYDRO_PSD_LABEL = "PSD (mPa$^2$ Hz$^{-1}$, dB)"
GEO_PSD_LABEL = "PSD (nm$^2$ s$^{-2}$ Hz$^{-1}$, dB)"
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

# Plot the hydrophone waveforms of one borehole in a given time window
def plot_windowed_hydro_waveforms(stream,
                                locations_to_plot = None, 
                                normalize=False,
                                scale = 1e-3, xdim = 10, ydim_per_loc = 2, 
                                linewidth_wf = 1,
                                date_format = "%Y-%m-%d %H:%M:%S",
                                major_time_spacing = "1s", num_minor_time_ticks = 5, major_depth_spacing = 50.0, num_minor_depth_ticks = 5,
                                va = "top", ha = "center", rotation = 0,
                                station_label_size = 12, axis_label_size = 12, tick_label_size = 12,
                                depth_lim = (0.0, 420.0), location_label_offset = (0.02, -15),
                                plot_pa_scalebar = False, plot_time_scalebar = False,
                                **kwargs):
    pa_unit = PRESSURE_UNIT
    loc_dict = HYDRO_LOCATIONS
    depth_dict = HYDRO_DEPTHS
    station_to_plot = stream[0].stats.station

    # Determine if the scalebar parameters are given
    if plot_pa_scalebar or plot_time_scalebar:
        scalebar_label_size = kwargs["scalebar_label_size"]
        scalebar_width = kwargs["scalebar_width"]

        if plot_pa_scalebar:
            pa_scalebar_offset_x = kwargs["pa_scalebar_offset_x"]
            pa_scalebar_offset_y = kwargs["pa_scalebar_offset_y"]
            pa_scalebar_length = kwargs["pa_scalebar_length"]
        
        if plot_time_scalebar:
            time_scalebar_offset_x = kwargs["time_scalebar_offset_x"]
            time_scalebar_offset_y = kwargs["time_scalebar_offset_y"]
            time_scalebar_length = kwargs["time_scalebar_length"]

    # Get the number of locations
    if locations_to_plot is None:
        locations_to_plot = loc_dict[station_to_plot]
    
    numloc = len(locations_to_plot)
    
    # Generate the figure and axes
    fig, ax  = subplots(1, 1, figsize = (xdim, ydim_per_loc * numloc))

    # Plot each location
    for location in locations_to_plot:
        print(f"Plotting {station_to_plot}.{location}...")
        # Plot the trace
        try:
            trace = stream.select(station=station_to_plot, location=location)[0]
        except:
            print(f"Could not find {station_to_plot}.{location}")
            continue

        data = trace.data
        depth = depth_dict[location]
        if normalize:
            data = data / amax(abs(data))
        
        data = -data * scale + depth
        timeax = get_datetime_axis_from_trace(trace)

        ax.plot(timeax, data, color="darkviolet", linewidth=linewidth_wf)

        # Add the location label
        offset_x = location_label_offset[0]
        offset_y = location_label_offset[1]
        label = f"{station_to_plot}.{location}"
        ax.text(timeax[0] + Timedelta(seconds=offset_x), depth + offset_y, label, fontsize=station_label_size, fontweight="bold", ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

    ### Set axis limits
    ax.set_xlim([timeax[0], timeax[-1]])
    ax.set_ylim(depth_lim)

    ### Reverse the y-axis
    ax.invert_yaxis()

    # Plot the scale bar
    if not normalize:
        if plot_pa_scalebar:
            if isinstance(pa_scalebar_offset_x, str):
                pa_scalebar_offset_x = Timedelta(pa_scalebar_offset_x)
            max_depth = ax.get_ylim()[0]

            scalebar_coord = (timeax[0] + pa_scalebar_offset_x, max_depth - pa_scalebar_offset_y)
            label_offsets = (0.25 * pa_scalebar_offset_x, 0.0)

            add_vertical_scalebar(ax, scalebar_coord, pa_scalebar_length, scale, label_offsets, label_unit = pa_unit, fontsize = scalebar_label_size, linewidth = scalebar_width)

    if plot_time_scalebar:
        if isinstance(time_scalebar_offset_x, str):
            time_scalebar_offset_x = Timedelta(time_scalebar_offset_x)
        max_depth = ax.get_ylim()[0]

        scalebar_coord = (timeax[0] + time_scalebar_offset_x, max_depth - time_scalebar_offset_y)
        label_offsets = (Timedelta(0.0), 10.0)

        add_horizontal_scalebar(ax, scalebar_coord, time_scalebar_length, 1.0, label_offsets, label_unit = "ms", fontsize = scalebar_label_size, linewidth = scalebar_width)

    # Format the x-axis labels
    ax  = format_datetime_xlabels(ax,
                                  date_format = date_format, 
                                  major_tick_spacing=major_time_spacing, num_minor_ticks=num_minor_time_ticks,
                                  va = va, ha = ha, rotation = rotation,
                                  axis_label_size=axis_label_size, tick_label_size=tick_label_size)

    # Format the y-axis labels
    ax = format_depth_ylabels(ax, 
                              major_tick_spacing=major_depth_spacing, num_minor_ticks=num_minor_depth_ticks, 
                              axis_label_size=axis_label_size, tick_label_size=tick_label_size)

    return fig, ax     

## Plot the hydrophone waveforms of one borehole in a given time window
def plot_cascade_zoom_in_hydro_waveforms(stream, station_df, window_dict,
                                station_to_plot = "A00", locations_to_plot = None, 
                                scale = 1e-3, xdim_per_win = 7, ydim_per_loc = 2, 
                                linewidth_wf = 1, linewidth_sb = 0.5,
                                major_time_spacing = 5.0, num_minor_time_ticks= 5, major_depth_spacing=50.0, num_minor_depth_ticks = 10,
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

            #add_scalebar(ax, scalebar_coord, scalebar_length, scale, linewidth=linewidth_sb)
            ax.annotate(f"{scalebar_length} {unit}", scalebar_coord, xytext=(0.5, 0), textcoords="offset fontsize", fontsize=scalebar_label_size, ha="left", va="center")

        ### Format the x-axis labels
        if major_time_spacing > 1.0:
            ax  = format_datetime_xlabels(ax, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, axis_label_size=axis_label_size, tick_label_size=tick_label_size)
        else:
            ax  = format_rel_time_xlabels(ax, major_tick_spacing=major_time_spacing, minor_tick_spacing=minor_time_spacing, axis_label_size=axis_label_size, tick_label_size=tick_label_size)

        ### Format the y-axis labels
        if i == 0:
            ax = format_depth_ylabels(ax, major_tick_spacing=major_depth_spacing, num_minor_ticks = num_minor_depth_ticks, axis_label_size=axis_label_size, tick_label_size=tick_label_size)

        ### Plot the box
        if i > 0:
            box_height = max_depth - min_depth - 2 * box_y_offset
            box = Rectangle((starttime, min_depth + box_y_offset), Timedelta(seconds=duration), box_height, edgecolor="crimson", facecolor="none", linestyle="-", linewidth=linewidth_box)
            ax = axes[i - 1]
            ax.add_patch(box)

    return fig, axes


###### Functions for plotting particle motions ######

###### Functions for plotting stationary-resonance properties vs time ######
# Property must contains at least four columns  "time", "station", "frequency", and property_name of the property to plot
# Station names are the indices of coord_df
# Stations will be plotted from south to north
def plot_stationary_resonance_properties_vs_time(property_name, property_df,
                                                 width = 10, height = 6,
                                                 sun_df = None, coord_df = None,
                                                 day_night_shading = True,
                                                 marker_size = 1.0,
                                                 scale_factor = 5.0,
                                                 station_label_offset = "2h",
                                                 station_label_size = 6.0,
                                                 ylim_offset = 0.5,
                                                 direction_label_offset = 0.1,
                                                 cbar_offset = 0.01,
                                                 axis_label_size = 8,
                                                 tick_label_size = 6,
                                                 title_size = 10,
                                                 **kwargs):
    starttime = STARTTIME_GEO
    endtime = ENDTIME_GEO

    print(property_name)

    # Determine if an axis is given
    if "axis" in kwargs:
        ax = kwargs["axis"]
        fig = ax.get_figure()
    else:
        fig, ax = subplots(1, 1, figsize=(width, height))

    # Load the station coordinates
    if coord_df is None:
        coord_df = get_geophone_coords()

    # Load the day-night time
    if day_night_shading and sun_df is None:
        sun_df = get_geo_sunrise_sunset_times()

    # Sort the stations from south to north
    coord_df = coord_df.sort_values(by="north")

    # Define the color scales
    if property_name == "frequency":

        if "min_freq" not in kwargs and "max_freq" not in kwargs:
            vmax = property_df["frequency"].max()
            vmin = property_df["frequency"].min()
        else:
            vmax = kwargs["max_freq"]
            vmin = kwargs["min_freq"]

        cmap = "coolwarm"
        cbar_label = "Frequency (Hz)"

    elif property_name == "total_power":

        if "min_power" not in kwargs and "max_power" not in kwargs:
            vmax = property_df["total_power"].max()
            vmin = property_df["total_power"].min()
        else:
            vmax = kwargs["max_power"]
            vmin = kwargs["min_power"]

        cmap = "inferno"
        cbar_label = GEO_PSD_LABEL

    elif property_name == "quality_factor":
            
        if "min_qf" not in kwargs and "max_qf" not in kwargs:
            vmax = property_df["quality_factor"].max()
            vmin = property_df["quality_factor"].min()
        else:
            vmax = kwargs["max_qf"]
            vmin = kwargs["min_qf"]

        cmap = "viridis"
        cbar_label = "Quality factor"

    elif "phase_diff" in property_name:
        vmax = 360.0
        vmin = 0.0

        cmap = cc.cm.CET_C9
        cbar_label = "Phase difference (deg)"

    elif "amp_ratio" in property_name:

        if "min_amp_rat" not in kwargs and "max_amp_rat" not in kwargs:
            vmax = property_df[property_name].max()
            vmin = property_df[property_name].min()
        else:
            vmax = kwargs["max_amp_rat"]
            vmin = kwargs["min_amp_rat"]

        cmap = "magma"
        cbar_label = "Amplitude ratio"

    elif "dip" in property_name:
        vmax = 90.0
        vmin = 0.0
        cmap = "cividis"
        cbar_label = "Dip (deg)"

    elif "strike" in property_name:
        vmax = 360.0
        vmin = 0.0
        cmap = "twilight"
        cbar_label = "Strike (deg)"

    # Plot the property vs time
    station_label_time = starttime + Timedelta(station_label_offset)
    i = 0
    for index, _ in coord_df.iterrows():
        station = index
        print(f"Plotting {station}...")

        # Get the properties of all time points
        property_sta_df = property_df.loc[property_df["station"] == station]
        times = property_sta_df["time"]
        freqs = property_sta_df["frequency"]
        mean_freq_plot = freqs.mean()
        
        if property_name != "frequency":
            properties = property_sta_df[property_name]
        else:
            properties = freqs

        # Plot the property vs time
        freqs_to_plot = (freqs - mean_freq_plot) * scale_factor + i

        # Plot the power vs time dots
        mappable = ax.scatter(times, freqs_to_plot, 
                              c = properties, s = marker_size,
                              cmap = cmap, vmin = vmin, vmax = vmax,
                              edgecolors = None, linewidths = 0,
                              zorder = 2)

        # Plot the station labels
        ax.text(station_label_time, i, station, color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "left")

        # Update the counter
        i += 1

    # Add the day-night shading
    print("Adding the day-night shading...")
    ax, _, _ = add_day_night_shading(ax, sun_df)

    # Plot the direction labels
    min_y = -1 - ylim_offset
    max_y = len(coord_df) + ylim_offset

    direction_label_time = station_label_time
    north_label_y = max_y - direction_label_offset
    south_label_y = min_y + direction_label_offset

    ax.text(direction_label_time, north_label_y, "North", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "top", horizontalalignment = "left")
    ax.text(direction_label_time, south_label_y, "South", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "bottom", horizontalalignment = "left")

    # Turn off the y ticks
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("")
    ax.yaxis.label.set_visible(False)

    # Set the axis labels and limits
    ax.set_xlim(starttime, endtime)
    ax.set_ylim(min_y, max_y)

    # Format the x-axis labels
    format_datetime_xlabels(ax, 
                            major_tick_spacing = "1d", num_minor_ticks = 4,
                            date_format = "%Y-%m-%d", 
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size,
                            va = "top", ha = "right", rotation = 30)


    # Add a vertical colorbar on the right
    print("Adding the colorbar...")
    bbox = ax.get_position()
    position = [bbox.x1 + cbar_offset, bbox.y0, cbar_offset / 2, bbox.height]
    cbar = add_colorbar(fig, position, cbar_label,
                        mappable = mappable, orientation = "vertical", axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Add the title
    if "title" in kwargs:
        title = kwargs["title"]
        ax.set_title(title, fontsize = title_size, fontweight = "bold")

    return fig, ax, cbar
        

###### Functions for plotting STFT power spectrograms and related data ######   

# Plot the 3C seismograms and spectrograms computed using STFT of a list of stations
def plot_geo_3c_waveforms_and_stfts(stream_wf, stream_spec,
                            xdim_per_col = 8, ydim_per_row= 4,
                            starttime = None, endtime = None,
                            min_freq = 0, max_freq = 200,
                            min_wf = -100, max_wf = 100,
                            linewidth=0.1,
                            dbmin=-30, dbmax=0,
                            component_label_x = 0.02, component_label_y = 0.96,
                            date_format = "%Y-%m-%d %H:%M:%S",
                            major_time_spacing="1min", num_minor_time_ticks=5,
                            major_vel_spacing=100, minor_vel_spacing=50,
                            major_freq_spacing=50, num_minor_freq_ticks = 5,
                            component_label_size=15, axis_label_size=12, tick_label_size=10, title_size=15,
                            time_tick_rotation=5, time_tick_va="top", time_tick_ha="right"):
    
    # Convert the power to dB
    stream_spec.to_db()

    # Get the list of stations
    stations_wf = get_unique_stations(stream_wf)
    stations_spec = stream_spec.get_stations()

    if stations_wf != stations_spec:
        raise ValueError("Inconsistent stations between the waveforms and spectrograms!")
    
    stations = stations_wf
    num_sta = len(stations)

    # Generate the figure and axes
    fig, axes = subplots(6, num_sta, figsize=(xdim_per_col * num_sta, 6 * ydim_per_row), sharex=True)

    # Loop over the stations
    for i, station in enumerate(stations):
        # Loop over the components
        for j, component in enumerate(GEO_COMPONENTS):
            # Plot the waveform
            trace_wf = stream_wf.select(station=station, component=component)[0]
            waveform = trace_wf.data
            timeax_wf = get_datetime_axis_from_trace(trace_wf)
            color = get_geo_component_color(component)

            ax = axes[ 2 * j, i]
            ax.plot(timeax_wf, waveform, color=color, linewidth=linewidth)
            
            if j == 0:
                title = station
                ax.set_title(title, fontsize=title_size, fontweight="bold")
            
            if i == 0:
                label = component2label(component)
                ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize=component_label_size, fontweight="bold", ha="left", va="top", bbox=dict(facecolor='white', alpha=1.0))
                format_vel_ylabels(ax, major_tick_spacing=major_vel_spacing, minor_tick_spacing=minor_vel_spacing, tick_label_size=tick_label_size)
            else:
                format_vel_ylabels(ax, major_tick_spacing=major_vel_spacing, minor_tick_spacing=minor_vel_spacing, tick_label_size=tick_label_size, label=False)

            ax.set_ylim((min_wf, max_wf))

            # Plot the spectrogram
            trace_spec = stream_spec.select(station=station, component=component)[0]
            timeax_spec = trace_spec.times
            freqax = trace_spec.freqs
            spec = trace_spec.data

            ax = axes[2 * j + 1, i]
            quadmesh = ax.pcolormesh(timeax_spec, freqax, spec, cmap="inferno", vmin=dbmin, vmax=dbmax)

            if i == 0:
                label = component2label(component)
                ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize=component_label_size, fontweight="bold", ha="left", va="top", bbox=dict(facecolor='white', alpha=1.0))

            if j == 2:
                format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                                        axis_label_size = axis_label_size, tick_label_size = tick_label_size, date_format = date_format,
                                        rotation = time_tick_rotation, vertical_align=time_tick_va, horizontal_align=time_tick_ha)
            if i == 0:
                format_freq_ylabels(ax, major_tick_spacing=major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, 
                                    axis_label_size = axis_label_size, tick_label_size=tick_label_size)
            else:
                format_freq_ylabels(ax, major_tick_spacing=major_freq_spacing, minor_tick_spacing=minor_freq_spacing,
                                    axis_label_size = axis_label_size, tick_label_size=tick_label_size, label=False)

            ax.set_ylim((min_freq, max_freq))

    # Set the axis limits
    if starttime is None:
        starttime = timeax_wf[0]
    elif isinstance(starttime, str):
        starttime = Timestamp(starttime)

    if endtime is None:
        endtime = timeax_wf[-1]
    elif isinstance(endtime, str):
        endtime = Timestamp(endtime)

    axes[0, 0].set_xlim(starttime, endtime)

    # Add the colorbar
    bbox = axes[-1, -1].get_position()
    position = [bbox.x0 + bbox.width + 0.01, bbox.y0, 0.005, bbox.height]
    cbar = add_quadmeshbar(fig, quadmesh, position, tick_spacing=10, tick_label_size=tick_label_size, orientation="vertical")

    return fig, axes, cbar

# Plot the 3C geophone spectrograms of a station computed using STFT
def plot_geo_stft_spectrograms(stream_spec,
                                xdim = 15, ydim_per_comp= 5, 
                                freq_lim=(0, 490), dbmin=-30, dbmax=0,
                                component_label_x = 0.01, component_label_y = 0.96,
                                date_format = "%Y-%m-%d",
                                major_time_spacing="1d", num_minor_time_ticks=4,
                                major_freq_spacing=100, num_minor_freq_ticks=5,
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

    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color="lightgray")

    ax = axes[0]
    quadmesh = ax.pcolormesh(timeax, freqax, data_z, cmap = cmap, vmin = dbmin, vmax = dbmax)
    label = component2label("Z")
    ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)  

    ax = axes[1]
    quadmesh = ax.pcolormesh(timeax, freqax, data_1, cmap = cmap, vmin = dbmin, vmax = dbmax)
    label = component2label("1")
    ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, tick_label_size = tick_label_size)

    ax = axes[2]
    quadmesh = ax.pcolormesh(timeax, freqax, data_2, cmap = cmap, vmin = dbmin, vmax = dbmax)
    label = component2label("2")
    ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, tick_label_size = tick_label_size)

    if plot_total_psd:
        ax = axes[3]
        trace_total = kwargs["total_psd_trace"]
        trace_total.to_db()
        data_total = trace_total.data
        quadmesh = ax.pcolormesh(timeax, freqax, data_total, cmap = cmap, vmin = dbmin, vmax = dbmax)
        label = "Total"
        ax.text(component_label_x, component_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
        format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, tick_label_size = tick_label_size)
    
    ax.set_ylim(freq_lim)

    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, tick_label_size = tick_label_size, date_format = date_format
, rotation = time_tick_rotation, vertical_align=time_tick_va, horizontal_align=time_tick_ha)

    # Add the colorbar
    bbox = axes[-1].get_position()
    position = [bbox.x0, bbox.y0 - 0.07, bbox.width, 0.01]
    cbar = add_quadmeshbar(fig, quadmesh, position, tick_spacing=10, tick_label_size=tick_label_size)
    cbar.set_label("Power spectral density (dB)")

    station = trace_spec_z.station
    fig.suptitle(station, fontsize = title_size, fontweight = "bold", y = 0.9)

    return fig, axes, cbar

# Plot the hydrophone spectrograms of all locations of a station computed using STFT
def plot_hydro_stft_spectrograms(stream_spec,
                            axes = None,
                            starttime = None, endtime = None,
                            min_freq = None, max_freq = None,
                            dbmin=-80, dbmax=-50,
                            location_label_x = 0.02, location_label_y = 0.96,
                            date_format = "%Y-%m-%d",
                            major_time_spacing = "24h", num_minor_time_ticks = 4,
                            major_freq_spacing = 10.0, num_minor_freq_ticks = 4,
                            component_label_size=12, axis_label_size=12, tick_label_size=10, title_size=12,
                            time_tick_rotation=30, time_tick_va="top", time_tick_ha="right",
                            title = None,
                            **kwargs):
    
    # Convert the power to dB
    stream_spec.to_db()

    # Get all locations
    locations = stream_spec.get_locations()
    num_loc = len(locations)

    # Generate the figure and axes if not provided
    if axes is None:
        width = kwargs["width"]
        row_height = kwargs["column_height"]
        fig, axes = subplots(num_loc, 1, figsize=(width, num_loc * row_height), sharex=True, sharey=True)
    else:
        if len(axes) != num_loc:
            print(len(axes), num_loc)
            raise ValueError("Inconsistent number of axes and locations!")

    # Plot the spectrograms
    cmap = get_quadmeshmap()

    for i, location in enumerate(locations):
        trace_spec = stream_spec.select(locations = location)[0]
        timeax = trace_spec.times
        freqax = trace_spec.freqs
        data = trace_spec.data
        
        ax = axes[i]
        mappable = ax.pcolormesh(timeax, freqax, data, cmap = cmap, vmin = dbmin, vmax = dbmax)

        station = trace_spec.station
        label = f"{station}.{location}"
        ax.text(location_label_x, location_label_y, label, transform=ax.transAxes, fontsize = component_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
        format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    if starttime is None:
        starttime = timeax[0]
    
    if endtime is None:
        endtime = timeax[-1]

    if min_freq is None:
        min_freq = freqax[0]

    if max_freq is None:
        max_freq = freqax[-1]

    ax.set_xlim(starttime, endtime)
    ax.set_ylim((min_freq, max_freq))

    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, tick_label_size = tick_label_size, date_format
 = date_format
, rotation = time_tick_rotation, axis_label_size = axis_label_size ,va = time_tick_va, ha = time_tick_ha)

    if title is not None:
        axes[0].set_title(title, fontsize = title_size, fontweight = "bold")

    if axes is None:
        return fig, axes, mappable
    else:
        return axes, mappable

# Plot the total PSD of a geophone station and the detected spectral peaks
def plot_geo_total_psd_and_peaks(trace_total, peak_df,
                            xdim = 15, ydim_per_row = 5, 
                            freq_lim=(0, 490), dbmin=-30, dbmax=0, rbwmin=0.1, rbwmax=0.5,
                            marker_size = 5,
                            panel_label_x = 0.01, panel_label_y = 0.96,
                            date_format = "%Y-%m-%d",
                            major_time_spacing=24, num_minor_time_ticks=6,
                            major_freq_spacing=100, num_minor_freq_ticks=5,
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
    quadmesh = ax.pcolormesh(timeax, freqax, power, cmap = "inferno", norm = color_norm)

    label = "Total PSD"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the spectral peak power
    ax = axes[1]

    peak_times = peak_df["time"]
    peak_freqs = peak_df["frequency"]
    peak_powers = peak_df["power"]

    ax.set_facecolor("lightgray")
    ax.scatter(peak_times, peak_freqs, c = peak_powers, s = marker_size, cmap = "inferno", norm = color_norm, edgecolors = None)
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the power colorbar
    bbox = ax.get_position()
    position = [bbox.x0 + bbox.width + 0.02, bbox.y0 , 0.01, bbox.height]
    #power_cbar = add_quadmeshbar(fig, quadmesh, position, tick_spacing=10, tick_label_size=tick_label_size, orientation = "vertical")

    # Plot the spectral peak reversed bandwidths
    ax = axes[2]

    peak_rbw = peak_df["reverse_bandwidth"]
    ax.set_facecolor("lightgray")
    rbw_color = ax.scatter(peak_times, peak_freqs, c = peak_rbw, s = marker_size, cmap = "viridis", norm=LogNorm(vmin = rbwmin, vmax = rbwmax))
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the reverse-bandwidth colorbar
    bbox = ax.get_position()
    position = [bbox.x0 + bbox.width + 0.02, bbox.y0 , 0.01, bbox.height]
    qf_cbar = add_rbw_colorbar(fig, rbw_color, position, tick_label_size=tick_label_size, orientation = "vertical")

    # Format the x-axis labels
    major_time_spacing = hour2sec(major_time_spacing) # Convert hours to seconds
    minor_time_spacing = hour2sec(minor_time_spacing) # Convert hours to seconds
    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size, date_format
 = date_format
, rotation = time_tick_rotation, vertical_align=time_tick_va, horizontal_align=time_tick_ha)

    # Set the y-axis limits
    ax.set_ylim(freq_lim)

    # Set the title
    station = trace_total.station
    fig.suptitle(station, fontsize = title_size, fontweight = "bold", y = 0.9)

    return fig, axes

# Plot array spectral-peak bin counts
def plot_array_spec_peak_counts(count_df,
                            size_scale = 5,
                            example_counts = array([5, 20, 35]),
                            xdim = 15, ydim = 5, 
                            starttime = STARTTIME_GEO, endtime = ENDTIME_GEO, min_freq = 0.0, max_freq = 500.0,
                            date_format = "%Y-%m-%d",
                            major_time_spacing="24h", num_minor_time_ticks = 6,
                            major_freq_spacing=100, num_minor_freq_ticks=5,
                            panel_label_size=15, axis_label_size=12, tick_label_size=10, title_size=15,
                            time_tick_rotation=15, time_tick_va="top", time_tick_ha="right"):

    # Trim the counts to the specified time range
    starttime = str2timestamp(starttime)
    endtime = str2timestamp(endtime)

    count_df = count_df.loc[(count_df["time"] >= starttime) & (count_df["time"] <= endtime)]

    # Calculate the marker sizes
    counts = count_df["count"]
    marker_sizes = (counts - counts.min()) / (counts.max() - counts.min()) * size_scale
    # print(marker_sizes.min())
    # print(len(count_df["time"]), len(count_df["frequency"]), len(marker_sizes))
    # print(len(count_df["time"]))
    # print(len(count_df["frequency"]))
    # print(len(marker_sizes))
    
    # Plot the detection counts
    fig, ax = subplots(1, 1, figsize=(xdim, ydim))
    ax.scatter(count_df["time"], count_df["frequency"], s = marker_sizes, facecolors = "lightgray", edgecolors = "black", alpha = 0.5, linewidth = 0.1)

    # Plot the example counts for the legend
    for count in example_counts:
        #marker_size = (log(count) - log(counts.min())) / (log(counts.max()) - log(counts.min())) * size_scale
        marker_size = (count - counts.min()) / (counts.max() - counts.min()) * size_scale
        ax.scatter([], [], s = marker_size, facecolors = "lightgray", edgecolors = "black", linewidth = 0.1, label = f"{count}")

    # Add the legend
    ax.legend(title = "Counts", fontsize = tick_label_size, title_fontsize = axis_label_size, loc = "upper right", framealpha = 1.0, edgecolor = "black")

    # Set the x-axis limits
    ax.set_xlim([count_df["time"].min(), count_df["time"].max()])
    
    # Set the y-axis limits
    ax.set_ylim([min_freq, max_freq])

    # Format the frequency axis
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Format the time axis
    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size, date_format = date_format, 
                            rotation = time_tick_rotation, va=time_tick_va, ha=time_tick_ha)


    # Add the title
    ax.set_title("Array detection counts", fontsize = title_size, fontweight = "bold")

    return fig, ax

# Plot the array spectral peak times and frequencies with a detection number above the threshold
def plot_array_spec_peak_times_and_freqs_in_multi_rows(count_df,
                                                       marker_size = 5,
                                                       starttime = STARTTIME_HYDRO, endtime = ENDTIME_HYDRO,
                                                       min_freq = None, max_freq = None,
                                                       num_rows = 4, column_width = 10.0, row_height = 2.0,
                                                       major_time_spacing = "15d", num_minor_time_ticks = 3,
                                                       major_freq_spacing = 0.5, num_minor_freq_ticks = 5):

    # Plotting
    print("Plotting the spectrograms...")
    fig, axes = subplots(num_rows, 1, figsize=(column_width, row_height * num_rows), sharey = True)

    # Plot each time window
    windows = date_range(starttime, endtime, periods = num_rows + 1)

    for i in range(num_rows):
        starttime = windows[i]
        endtime = windows[i + 1]
        count_df_window = count_df.loc[(count_df["time"] >= starttime) & (count_df["time"] <= endtime)]

        ax = axes[i]
        ax.scatter(count_df_window["time"], count_df_window["frequency"], s = marker_size, facecolors = "lightgray", edgecolors = "black", alpha = 0.5, linewidth = 0.1)

        format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks, date_format = "%Y-%m-%d")
        

        # if i < num_rows - 1:       
        #     add_horizontal_scalebar(ax, scalebar_coord, "1d", 1.0, color = color_ref, plot_label = False)
        # else:
        #     add_horizontal_scalebar(ax, scalebar_coord, "1d", 1.0, color = color_ref, plot_label = True,
        #                             label = "1d", label_offsets = scalebar_label_offsets, fontsize = 10, fontweight = "bold")


        if min_freq is not None and max_freq is not None:
            ax.set_ylim([min_freq, max_freq])
    
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)   

    return fig, axes

# Plot the total PSD of a geophone station, the power and reverse bandwidth of the spectral peaks detected from it, and the array detection counts
def plot_geo_total_psd_peaks_and_array_counts(trace_total, peak_df, count_df,
                            size_scale = 5, marker_size = 1,
                            example_counts = array([5, 20, 35]),
                            xdim = 15, ydim_per_row = 5,
                            freq_lim=(0, 490), dbmin=-30, dbmax=0, rbwmin=0.1, rbwmax=0.5,
                            date_format = "%Y-%m-%d",
                            major_time_spacing="1d", num_minor_time_ticks=4,
                            major_freq_spacing=100, num_minor_freq_ticks=5,
                            panel_label_x = 0.01, panel_label_y = 0.96, panel_label_size=12, 
                            axis_label_size=12, tick_label_size=10, title_size=15,
                            time_tick_rotation=15, time_tick_va="top", time_tick_ha="right"):
    # Convert the power to dB
    trace_total.to_db()

    # Generate the figure and axes
    fig, axes = subplots(4, 1, figsize=(15, 15), sharex=True, sharey=True)

    # Plot the total PSD
    ax = axes[0]

    cmap_power = colormaps["inferno"].copy()
    cmap_power.set_bad(color='lightgray')
    norm_power = Normalize(vmin=dbmin, vmax=dbmax)

    timeax = trace_total.times
    freqax = trace_total.freqs
    power = trace_total.data
    ax.set_facecolor("lightgray")
    quadmesh = ax.pcolormesh(timeax, freqax, power, cmap = cmap_power, norm = norm_power)

    label = "Total PSD"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))

    station = trace_total.station
    ax.set_title(station, fontsize = title_size, fontweight = "bold")

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the spectral peak power
    ax = axes[1]

    peak_times = peak_df["time"]
    peak_freqs = peak_df["frequency"]
    peak_powers = peak_df["power"]

    ax.set_facecolor("lightgray")
    ax.scatter(peak_times, peak_freqs, c = peak_powers, s = marker_size, cmap = cmap_power, norm = norm_power, edgecolors = None)

    label = "Peak power"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the power colorbar
    bbox = ax.get_position()
    position = [bbox.x0 + bbox.width + 0.02, bbox.y0 , 0.01, bbox.height]
    power_cbar = add_quadmeshbar(fig, quadmesh, position, tick_spacing=10, tick_label_size=tick_label_size, orientation = "vertical")

    # Plot the spectral peak reverse bandwidths
    ax = axes[2]

    peak_rbw = peak_df["reverse_bandwidth"]
    ax.set_facecolor("lightgray")

    cmap_rbw = colormaps["viridis"].copy()
    cmap_rbw.set_bad(color='lightgray')
    rbw_color = ax.scatter(peak_times, peak_freqs, c = peak_rbw, s = marker_size, cmap = cmap_rbw, norm = LogNorm(vmin = rbwmin, vmax = rbwmax))

    label = "Peak width"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the reverse-bandwidth colorbar
    bbox = ax.get_position()
    position = [bbox.x0 + bbox.width + 0.02, bbox.y0 , 0.01, bbox.height]
    rbw_cbar = add_rbw_colorbar(fig, rbw_color, position, tick_label_size=tick_label_size, orientation = "vertical")

    # Plot the array detection counts
    ax = axes[3]

    # Calculate the marker sizes
    counts = count_df["count"]
    marker_sizes = (counts - counts.min()) / (counts.max() - counts.min()) * size_scale
    
    # Plot the detection counts
    ax.scatter(count_df["time"], count_df["frequency"], s = marker_sizes, facecolors = "lightgray", edgecolors = "black", alpha = 0.5, linewidth = 0.1)

    # Plot the example counts for the legend
    for count in example_counts:
        marker_size = (count - counts.min()) / (counts.max() - counts.min()) * size_scale
        ax.scatter([], [], s = marker_size, facecolors = "lightgray", edgecolors = "black", linewidth = 0.1, label = f"{count}")

    # Add the legend
    ax.legend(title = "Counts", fontsize = tick_label_size, title_fontsize = axis_label_size, loc = "upper right", framealpha = 1.0, edgecolor = "black")

    # Format the frequency axis
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Format the time axis
    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size, date_format = date_format
                            , rotation = time_tick_rotation, vertical_align=time_tick_va, horizontal_align=time_tick_ha)
    
    # Add the title
    ax.set_title("Array bin counts", fontsize = title_size, fontweight = "bold")
    
    # Set the x-axis limits
    ax.set_xlim([count_df["time"].min(), count_df["time"].max()])
    
    # Set the y-axis limits
    ax.set_ylim(freq_lim)

    return fig, axes, power_cbar, rbw_cbar

# Plot the geophone total psd, the spectral-peak powers, array detection counts, and the binarized array spectrogram
def plot_geo_total_psd_to_bin_array_spectrogram(trace_total, peak_df, count_df, bin_array_dict,
                                            size_scale = 5, marker_size = 1,
                                            example_counts = array([10, 20, 30]),
                                            xdim = 15, ydim_per_row = 5,
                                            min_freq = 0.0, max_freq = 500.0,
                                            starttime = None, endtime = None,
                                            dbmin=-30, dbmax=0,
                                            date_format = "%Y-%m-%d %H:%M:%S",
                                            major_time_spacing="15min", num_minor_time_ticks=3,
                                            major_freq_spacing=50.0, num_minor_freq_ticks=5,
                                            panel_label_x = 0.01, panel_label_y = 0.96, panel_label_size=12,
                                            axis_label_size=12, tick_label_size=10, title_size=15,
                                            time_tick_rotation=5, time_tick_va="top", time_tick_ha="right"):

    # Define the axes
    fig, axes = subplots(4, 1, figsize=(xdim, 3 * ydim_per_row), sharex=True, sharey=True)

    # Plot the total PSD
    ax = axes[0]

    cmap_power = colormaps["inferno"].copy()
    cmap_power.set_bad(color='lightgray')
    norm_power = Normalize(vmin=dbmin, vmax=dbmax)

    trace_total.to_db()
    station = trace_total.station
    timeax = trace_total.times
    freqax = trace_total.freqs
    power = trace_total.data

    ax.set_facecolor("lightgray")
    quadmesh = ax.pcolormesh(timeax, freqax, power, cmap = cmap_power, norm = norm_power)

    label = f"{station}, total PSD"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Add the colorbar
    bbox = ax.get_position()
    position = [bbox.x1 + 0.02, bbox.y0 , 0.01, bbox.height]
    power_cbar = add_colorbar(fig, quadmesh, "Power (dB)", position, orientation = "vertical")

    # Plot the spectral peak power
    ax = axes[1]

    peak_times = peak_df["time"]
    peak_freqs = peak_df["frequency"]
    peak_powers = peak_df["power"]

    ax.set_facecolor("lightgray")
    ax.scatter(peak_times, peak_freqs, c = peak_powers, s = marker_size, cmap = cmap_power, norm = norm_power, edgecolors = None)

    label = f"{station}, peak power"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the array spectral-peak counts
    ax = axes[2]

    # Calculate the marker sizes
    counts = count_df["count"]
    marker_sizes = (counts - counts.min()) / (counts.max() - counts.min()) * size_scale

    # Plot the detection counts
    ax.scatter(count_df["time"], count_df["frequency"], s = marker_sizes, facecolors = "lightgray", edgecolors = "black", alpha = 0.5, linewidth = 0.1)

    label = "Array spectral-peak counts"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))


    # Plot the example counts for the legend
    for count in example_counts:
        example_size = (count - counts.min()) / (counts.max() - counts.min()) * size_scale
        ax.scatter([], [], s = example_size, facecolors = "lightgray", edgecolors = "black", linewidth = 0.1, label = f"{count}")
    
    # Add the legend
    ax.legend(title = "Counts", fontsize = tick_label_size, title_fontsize = axis_label_size, loc = "upper right", framealpha = 1.0, edgecolor = "black")

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the binarized array spectrogram
    ax = axes[3]

    timeax = bin_array_dict["times"]
    freqax = bin_array_dict["freqs"]
    data = bin_array_dict["data"]

    ax.set_facecolor("blue")
    ax.pcolormesh(timeax, freqax, data, cmap = "binary_r")

    label = "Binarized array spectrogram"
    ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)  

    # Format the time axis
    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size, date_format = date_format,
                            rotation = time_tick_rotation, va = time_tick_va, ha = time_tick_ha)

    # Set the x-axis limits
    if starttime is None:
        starttime = timeax.min()
    elif not isinstance(starttime, Timestamp):
        starttime = Timestamp(starttime)
    
    if endtime is None:
        endtime = timeax.max()
    elif not isinstance(endtime, Timestamp):
        endtime = Timestamp(endtime)

    ax.set_xlim((starttime, endtime))

    # Set the y-axis limits
    ax.set_ylim((min_freq, max_freq))

    return fig, axes, power_cbar


# Plot the comparison between the spectral-peak array counts and the peaks found from the stacked spectrograms
def plot_array_peak_counts_vs_stacked_peaks(count_df, peak_df,
                            size_scale = 5, marker_size = 1,
                            example_counts = array([5, 20, 35]),
                            xdim = 15, ydim_per_row = 5,
                            starttime = None, endtime = None,
                            min_freq = 0.0, max_freq = 500.0,
                            dbmin=-30, dbmax=0, rbwmin=0.1, rbwmax=0.5,
                            date_format = "%Y-%m-%d",
                            major_time_spacing="1d", num_minor_time_ticks=4,
                            major_freq_spacing=100,num_minor_freq_ticks=5,
                            panel_label_x = 0.01, panel_label_y = 0.96, panel_label_size=12, 
                            axis_label_size=12, tick_label_size=10, title_size=15,
                            time_tick_rotation=15, time_tick_va="top", time_tick_ha="right"):

    # Generate the figure and axes
    fig, axes = subplots(3, 1, figsize=(15, 10), sharex=True, sharey=True)

    # Plot the array detection counts
    ax = axes[0]

    # Calculate the marker sizes
    counts = count_df["count"]
    marker_sizes = (counts - counts.min()) / (counts.max() - counts.min()) * size_scale

    # Plot the detection counts
    ax.scatter(count_df["time"], count_df["frequency"], s = marker_sizes, facecolors = "lightgray", edgecolors = "black", alpha = 0.5, linewidth = 0.1)

    # Plot the example counts for the legend
    for count in example_counts:
        example_size = (count - counts.min()) / (counts.max() - counts.min()) * size_scale
        ax.scatter([], [], s = example_size, facecolors = "lightgray", edgecolors = "black", linewidth = 0.1, label = f"{count}")

    # Add the legend
    ax.legend(title = "Counts", fontsize = tick_label_size, title_fontsize = axis_label_size, loc = "upper right", framealpha = 1.0, edgecolor = "black")

    # Format the frequency axis
    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the stacked spectrogram peaks color-coded by power
    ax = axes[1]

    peak_times = peak_df["time"]
    peak_freqs = peak_df["frequency"]
    peak_powers = peak_df["power"]

    ax.set_facecolor("lightgray")
    quadmesh = ax.scatter(peak_times, peak_freqs, c = peak_powers, s = marker_size, cmap = "inferno", edgecolors = None)

    # label = "Peak power"
    # ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the power colorbar
    bbox = ax.get_position()
    position = [bbox.x0 + bbox.width + 0.02, bbox.y0 , 0.01, bbox.height]
    #power_cbar = add_quadmeshbar(fig, quadmesh, position, tick_spacing=10, tick_label_size=tick_label_size, orientation = "vertical")

    # Plot the stacked spectrogram peaks color-coded by reverse bandwidth
    ax = axes[2]

    peak_rbw = peak_df["reverse_bandwidth"]
    ax.set_facecolor("lightgray")

    cmap_rbw = colormaps["viridis"].copy()
    cmap_rbw.set_bad(color='lightgray')
    rbw_color = ax.scatter(peak_times, peak_freqs, c = peak_rbw, s = marker_size, cmap = cmap_rbw, norm = LogNorm(vmin = rbwmin, vmax = rbwmax))

    # label = "Peak width"
    # ax.text(panel_label_x, panel_label_y, label, transform=ax.transAxes, fontsize = panel_label_size, fontweight = "bold", ha = "left", va = "top", bbox=dict(facecolor='white', alpha=1.0))

    format_freq_ylabels(ax, major_tick_spacing = major_freq_spacing, num_minor_freq_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    # Plot the reverse-bandwidth colorbar
    bbox = ax.get_position()
    position = [bbox.x0 + bbox.width + 0.02, bbox.y0 , 0.01, bbox.height]
    rbw_cbar = add_rbw_colorbar(fig, rbw_color, position, tick_label_size=tick_label_size, orientation = "vertical")

    # Format the time axis
    format_datetime_xlabels(ax, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                            axis_label_size = axis_label_size, tick_label_size = tick_label_size, date_format = date_format
                            , rotation = time_tick_rotation, vertical_align=time_tick_va, horizontal_align=time_tick_ha)

    # Set the x-axis limits
    if starttime is None:
        starttime = peak_df["time"].min()
    elif isinstance(starttime, str):
        starttime = Timestamp(starttime)

    if endtime is None:
        endtime = peak_df["time"].max()
    elif isinstance(endtime, str):
        endtime = Timestamp(endtime)
        
    ax.set_xlim((starttime, endtime))
    
    # Set the y-axis limits
    ax.set_ylim((min_freq, max_freq))

    return fig, axes, power_cbar, rbw_cbar

# Plot the cumulative fraction of the total observation time of the spectral peaks
def plot_cum_freq_fractions(count_df,
                        xtick_labels = True,
                        ytick_labels = True,
                        xdim = 15, ydim = 5,
                        log_scale = True,
                        min_freq = 0.0, max_freq = 500.0,
                        linewdith = 1.0, marker_size = 10.0,
                        major_freq_spacing = 100, num_minor_freq_ticks = 5,
                        axis_label_size = 12, tick_label_size = 10,
                        **kwargs):

    # Determin if an axis object is provided
    if "axis" in kwargs:
        ax = kwargs["axis"]
    else:
        fig, ax = subplots(1, 1, figsize=(xdim, ydim))

    # Plot the cumulative frequency counts with stems and markers
    if log_scale:
        ax.set_yscale("log")
        count_df.loc[count_df["fraction"] == 0, "fraction"] = 1e-10

    freqs = count_df["frequency"]
    fracs = count_df["fraction"]

    if ytick_labels:
        ax.set_ylabel("Fraction", fontsize = axis_label_size)

    markerline, stemlines, _ = ax.stem(freqs, fracs)
    markerline.set_markersize(marker_size)
    markerline.set_markerfacecolor("black")
    markerline.set_markeredgecolor("black")
    stemlines.set_linewidth(linewdith)
    stemlines.set_color("black")

    # Set the x-axis limits
    ax.set_xlim((min_freq, max_freq))

    # Format the frequency axis
    format_freq_xlabels(ax, 
                        label = xtick_labels,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, 
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    if "axis" in kwargs:
        return ax
    else:
        return fig, ax
          
###### Functions for plotting CWT spectra ######

## Function to plot the 3C seismograms and spectrograms computed using CWT of a list of stations
def plot_3c_waveforms_and_cwts(stream, specs, 
                                 xdim_per_comp=10, ydim_per_sta=3, ylim_wf=(-50, 50), ylim_freq=(0.0, 500.0), dbmin=0.0, dbmax=30.0, 
                                 linewidth_wf=0.2, major_time_spacing=5.0, num_minor_time_ticks = 5, major_freq_spacing=20, minor_freq_spacing=5, 
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
                    title = component2label(component)
                    ax.set_title(f"{title}", fontsize=title_size, fontweight="bold")

                #### Set the y tick labels
                ax.yaxis.set_tick_params(labelsize=tick_label_size)

                ### Plot the power spectrogram
                    
                #### Get the power spectrum in dB
                power = spec.get_power()
                ax = axes[2 * j + 1, i]
                quadmesh = ax.pcolormesh(timeax, freqax, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

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
    format_datetime_xlabels(axes, major_tick_spacing=major_time_spacing, num_minor_ticks=num_minor_time_ticks, tick_label_size=tick_label_size)

    ### Plot the colorbar at the bottom
    cax = fig.add_axes([0.35, 0.0, 0.3, 0.02])
    cbar = fig.colorbar(quadmesh, cax=cax, orientation="horizontal", label="Power (dB)")
    cbar.locator = MultipleLocator(base=10)
    cbar.update_ticks()

    return fig, axes


## Function to plot the amplitude of the CWTs of a stream
def plot_cwt_powers(specs,
                    xdim_per_comp=10, ydim_per_sta=3, freqlim=(0.0, 200), dbmin=0.0, dbmax=30, 
                    major_time_spacing="5s", num_minor_tick_ticks=5, major_freq_spacing=20, num_minor_freq_ticks = 5,
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
            quadmesh = ax.pcolormesh(timeax, freqax, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

            # ### Plot the noise window
            # box = Rectangle((noise_window[0], noise_window[2]), noise_window[1] - noise_window[0], noise_window[3] - noise_window[2], edgecolor="white", facecolor="none", linestyle=":", linewidth=linewidth_box)
            # ax.add_patch(box)

            if j == 0:
                title = component2label(component)
                ax.set_title(f"{title}", fontsize=title_size, fontweight="bold")

            if i == 0:
                ax.text(0.01, 0.98, f"{station}", color="white", fontsize=station_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top")
    
    ax = axes[0, 0]
    ax.set_ylim(freqlim)
    
    format_datetime_xlabels(axes, major_tick_spacing=major_time_spacing, num_minor_ticks = num_minor_tick_ticks, tick_label_size=tick_label_size)
    format_freq_ylabels(axes, major_tick_spacing=major_freq_spacing, num_minor_ticks=num_minor_freq_ticks, tick_label_size=tick_label_size, axis_label_size=axis_label_size)

    ### Add the colorbar
    caxis = fig.add_axes([0.35, -0.03, 0.3, 0.03])
    cbar = fig.colorbar(quadmesh, cax=caxis, orientation="horizontal", label="Power (dB)")

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
        label = component2label(component_to_plot)

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
            quadmesh = ax.pcolormesh(timeax, freqax, power, cmap="inferno", vmin=dbmin, vmax=dbmax)
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
    cbar = add_quadmeshbar(fig, quadmesh, position, orientation="horizontal")

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
            quadmesh = ax.pcolormesh(timeax, freqax, power, cmap="inferno", vmin=dbmin, vmax=dbmax)
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
    cbar = add_quadmeshbar(fig, quadmesh, position, orientation="horizontal")

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
        quadmesh = ax.pcolormesh(timeax, freqs, power, cmap="inferno", vmin=dbmin, vmax=dbmax)


        label = component2label(component)
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
    power_cbar = add_quadmeshbar(fig, quadmesh, position, orientation="horizontal")

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
        quadmesh = ax.pcolormesh(timeax, freqs, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

        label = f"{component2label(component1)}-{component2label(component2)}"
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
    power_cbar = add_quadmeshbar(fig, quadmesh, position, orientation="horizontal")

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
        title = component2label(component)

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

        title = component2label(component)
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
        quadmesh = ax.pcolormesh(timeax, freqs, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

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
        quadmesh = ax.pcolormesh(timeax, freqs, power, cmap="inferno", vmin=dbmin, vmax=dbmax)

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
    add_quadmeshbar(fig, quadmesh, position, orientation="horizontal", tick_spacing=dbspacing)

    return fig, axes

# Plot the 3C beamforming results
# Input slownesses are in s/m and will be converted to s/km
def plot_beam_images(xslowax, yslowax, bimage_dict,
                     plot_global_max = True, plot_local_max = False,
                     vmin=0, vmax=1, reference_vels=[500, 1500, 3500],           
                     marker_color = "deepskyblue", marker_size_max = 20.0, marker_size_local_max = 10.0,
                     major_slow_spacing=0.5, num_minor_slow_ticks=5,
                     axis_label_size=10, tick_label_size=10, title_size=14, label_size=10, label_angle=90):
    
    xslow_label = X_SLOW_LABEL
    yslow_label = Y_SLOW_LABEL
    vel_unit = WAVE_VELOCITY_UNIT
    
    numcomp = len(bimage_dict)
    xslowax = xslowax * 1000
    yslowax = yslowax * 1000

    fig, axes = subplots(1, numcomp, figsize=(15, 5), sharex=True, sharey=True)

    for i, component in enumerate(bimage_dict.keys()):
        bimage = bimage_dict[component].bimage
        ax = axes[i]
        cmap = ax.pcolor(xslowax, yslowax, bimage, cmap="inferno", vmin=vmin, vmax=vmax)

        if i == 0:
            ax.set_xlabel(xslow_label, fontsize=axis_label_size)
            ax.set_ylabel(yslow_label, fontsize=axis_label_size)

        ax.set_title(component2label(component), fontsize=title_size, fontweight="bold")
        ax.set_aspect("equal")

        # Plot the local maxima
        if plot_local_max:
            xslows_local_max = bimage_dict[component].xslows_local_max * 1000
            yslows_local_max = bimage_dict[component].yslows_local_max * 1000

            for xslow_local_max, yslow_local_max in zip(xslows_local_max, yslows_local_max):
                ax.scatter(xslow_local_max, yslow_local_max, color=marker_color, s=marker_size_local_max, marker="D", alpha=0.5)

        # Plot the reference slownesses and the crosshair
        if i == 0:
            plot_reference_slownesses(ax, reference_vels, 
                                      plot_label = True,
                                      linewidth=1.0, label_angle=label_angle, label_size=label_size)
        else:
            plot_reference_slownesses(ax, reference_vels, 
                                      plot_label = False,
                                      linewidth=1.0, label_angle=label_angle, label_size=label_size)

        # Plot the global maxima
        if plot_global_max:
            xslow_global_max = bimage_dict[component].xslow_global_max * 1000
            yslow_global_max = bimage_dict[component].yslow_global_max * 1000
            
            ax.scatter(xslow_global_max, yslow_global_max, color=marker_color, s=marker_size_max, marker="D", edgecolor="black", linewidth=0.5)

    ### Set the axis ticks and labels
    for i, ax in enumerate(axes):
        if i == 0:
            format_slowness_xlabels(ax,
                                    major_tick_spacing=major_slow_spacing, num_minor_ticks=num_minor_slow_ticks, 
                                    axis_label_size=axis_label_size, tick_label_size=tick_label_size)
            format_slowness_ylabels(ax,
                                    major_tick_spacing=major_slow_spacing, num_minor_ticks=num_minor_slow_ticks, 
                                    axis_label_size=axis_label_size, tick_label_size=tick_label_size)
        else:
            format_slowness_xlabels(ax, label=False,
                                    major_tick_spacing=major_slow_spacing, num_minor_ticks=num_minor_slow_ticks, 
                                    axis_label_size=axis_label_size, tick_label_size=tick_label_size)
            format_slowness_ylabels(ax, label=False,
                                    major_tick_spacing=major_slow_spacing, num_minor_ticks=num_minor_slow_ticks, 
                                    axis_label_size=axis_label_size, tick_label_size=tick_label_size)
            
    ### Add the colorbar
    bbox = axes[-1].get_position()
    position = [bbox.x0 + bbox.width + 0.01, bbox.y0, 0.01, bbox.height]
    caxis = fig.add_axes(position)
    cbar = fig.colorbar(cmap, cax=caxis)
    cbar.set_label("Normalized power", fontsize=12)

    # ### Add the title
    # if subarray == "A":
    #     if station_selection == "inner":
    #         title = "Array A, inner"
    #     else:
    #         title = "Array A, all"
    # elif subarray == "B":
    #     if station_selection == "inner":
    #         title = "Array B, inner"
    #     else:
    #         title = "Array B, all"
    # else:
    #     title = "Whole array"

    # fig.suptitle(title, fontsize=title_size, fontweight="bold")
    
    return fig, axes

# Plot reference slownesses on the beamforming images
def plot_reference_slownesses(ax, vels,
                              plot_label = True,
                              color = "deepskyblue", linewidth = 1.0, label_angle=90, label_size=10):
        
        vel_unit = WAVE_VELOCITY_UNIT

        # Plot the reference slownesses
        for i, vel in enumerate(vels):
            slow = 1 / vel * 1000
            ax.add_patch(Circle((0, 0), slow, edgecolor=color, facecolor='none', linestyle="--", linewidth=linewidth))
            ax.add_patch(Circle((0, 0), slow, edgecolor=color, facecolor='none', linestyle="--", linewidth=linewidth))
            ax.add_patch(Circle((0, 0), slow, edgecolor=color, facecolor='none', linestyle="--", linewidth=linewidth))

            if plot_label:
                x = slow * cos(radians(label_angle))
                y = slow * sin(radians(label_angle))

                if i == 0:
                    label = f"{vel} {vel_unit}"
                else:
                    label = f"{vel}"

                ax.text(x, y, label, fontsize=label_size, fontweight="bold", color=color, va="top", ha="left")

        # Plot the crosshair
        ax.axhline(0, color=color, linestyle="--", linewidth=linewidth)
        ax.axvline(0, color=color, linestyle="--", linewidth=linewidth)

        return ax

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
    quadmesh = ax.pcolormesh(timeax, freqs, power, shading="auto", cmap="inferno", vmin=dbmin, vmax=dbmax)

    cbar_coord = [0.91, 0.8, 0.01, 0.07]
    cbar = add_quadmeshbar(fig, quadmesh, cbar_coord, orientation="vertical", dbspacing=dbspacing)

    ax.text(0.01, 0.85, component2label("Z"), fontsize=component_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))
    
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
        ax.text(0.01, 0.85, component2label(component), fontsize=component_label_size, fontweight="bold", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=1))

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

        label1 = component2label(component1)
        label2 = component2label(component2)

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

# Function for getting windowed particle-motion data
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

# Plot the projection of the particle-motion data on the three planes
def plot_pm_projections(signal_1, signal_2, signal_z,
                        linewidth = 0.5, linecolor = "gray",
                        panel_width = 5.0, panel_height = 5.0, panel_gap = 1.0,
                        **kwargs):

    if "axes" in kwargs:
        axes = kwargs["axes"]
        fig = axes[0].get_figure()

        if len(axes) != 3:
            raise ValueError("Three axes are required!")
    else:
        fig, axes = subplots(1, 3, figsize=(panel_width * 3 + 2 * panel_gap, panel_height))

    # Normalize the signals by the maximum value of the three components
    max_val = max([amax(abs(signal_1)), amax(abs(signal_2)), amax(abs(signal_z))])
    signal_1 = signal_1 / max_val
    signal_2 = signal_2 / max_val
    signal_z = signal_z / max_val

    # Plot the 1-2 projection
    ax = axes[0]
    ax.plot(signal_2, signal_1, color=linecolor, linewidth=linewidth)
    ax.set_xlabel(component2label("2"))
    ax.set_ylabel(component2label("1"))
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_aspect("equal")


    # Plot the 1-Z projection
    ax = axes[1]
    ax.plot(signal_1, signal_z, color=linecolor, linewidth=linewidth)
    ax.set_xlabel(component2label("1"))
    ax.set_ylabel(component2label("Z"))
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_aspect("equal")

    # Plot the 2-Z projection
    ax = axes[2]
    ax.plot(signal_2, signal_z, color=linecolor, linewidth=linewidth)
    ax.set_xlabel(component2label("2"))
    ax.set_ylabel(component2label("Z"))
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_aspect("equal")

    return fig, axes

###### Functions for plotting FFT PSDs ######
def plot_all_geo_fft_psd_n_maps(stream_fft, coord_df, min_freq, max_freq, 
                                factor = 0.05, base = -10, psd_plot_hw_ratio = 2.0, psd_plot_width = 4.0, panel_gap = 0.2,
                                linewidth_psd = 1.0, marker_size = 10, linewidth = 0.5, 
                                station_label_offset = 0.005, station_label_size = 8, direction_label_offset = 0.5,
                                axis_label_size = 12, tick_label_size = 10, title_size = 14,
                                major_freq_spacing = 0.2, num_minor_freq_ticks = 10,
                                stations_highlight = {"A16": (3.0, -3.0), "A01": (-3.0, 3.0), "A19": (3.0, -3.0), "B01": (-3.0, 3.0), "B20": (3.0, -3.0)}):
    
    max_north = NORTHMAX_WHOLE
    min_north = NORTHMIN_WHOLE
    max_east = EASTMAX_WHOLE
    min_east = EASTMIN_WHOLE

    ### Sort the stations by their S-N coordinates
    coord_df = coord_df.sort_values(by = "north", ascending = True)

    ### Generate the figure and axes ###
    print("Plotting the frequency vs time for each station...")
    print("Generating the figure and axes...")
    map_hw_ratio = (max_north - min_north) / (max_east - min_east)
    width_ratios = [1, 1, 1, psd_plot_hw_ratio / map_hw_ratio]

    gs = GridSpec(1, 4, width_ratios=width_ratios)

    fig_width = psd_plot_width * (3 + psd_plot_hw_ratio / map_hw_ratio + 3 * panel_gap)
    fig_height = psd_plot_width * psd_plot_hw_ratio

    fig = figure(figsize = (fig_width, fig_height))

    ### Plot the 3D PSDs for each station ###
    ax_psd_z = fig.add_subplot(gs[0, 0])
    ax_psd_1 = fig.add_subplot(gs[0, 1])
    ax_psd_2 = fig.add_subplot(gs[0, 2])

    i = 0
    for index, _ in coord_df.iterrows():
        station = index
        print(f"Plotting {station}...")

        # Get the 3C PSD of the station
        trace_fft_z = stream_fft.select(stations=station, components="Z")[0]
        trace_fft_1 = stream_fft.select(stations=station, components="1")[0]
        trace_fft_2 = stream_fft.select(stations=station, components="2")[0]

        trace_fft_z.to_db()
        trace_fft_1.to_db()
        trace_fft_2.to_db()

        psd_z = trace_fft_z.psd
        psd_1 = trace_fft_1.psd
        psd_2 = trace_fft_2.psd
        freqax = trace_fft_z.freqs

        psd_z_to_plot = psd_z[(freqax >= min_freq) & (freqax <= max_freq)]
        psd_1_to_plot = psd_1[(freqax >= min_freq) & (freqax <= max_freq)]
        psd_2_to_plot = psd_2[(freqax >= min_freq) & (freqax <= max_freq)]
        freqax_to_plot = freqax[(freqax >= min_freq) & (freqax <= max_freq)]

        # Plot the frequency vs time curves
        psd_z_to_plot = (psd_z_to_plot - base) * factor + i
        psd_1_to_plot = (psd_1_to_plot - base) * factor + i
        psd_2_to_plot = (psd_2_to_plot - base) * factor + i

        ax_psd_z.plot(freqax_to_plot, psd_z_to_plot, color = get_geo_component_color("Z"), linewidth = linewidth_psd)
        ax_psd_1.plot(freqax_to_plot, psd_1_to_plot, color = get_geo_component_color("1"), linewidth = linewidth_psd)
        ax_psd_2.plot(freqax_to_plot, psd_2_to_plot, color = get_geo_component_color("2"), linewidth = linewidth_psd)

        # Plot the station labels
        station_label_x = min_freq + station_label_offset
        ax_psd_z.text(station_label_x, i, station, color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "center", horizontalalignment = "left")

        # Update the counter
        i += 1

    # Plot the direction labels
    min_y = -1
    max_y = len(coord_df) + 1

    direction_label_x = station_label_x
    north_label_y = max_y - direction_label_offset
    south_label_y = min_y + direction_label_offset

    ax_psd_z.text(direction_label_x, north_label_y, "North", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "bottom", horizontalalignment = "left")
    ax_psd_z.text(direction_label_x, south_label_y, "South", color = "black", fontsize = station_label_size, fontweight = "bold", verticalalignment = "top", horizontalalignment = "left")

    # Turn off the y ticks
    ax_psd_z.set_yticks([])
    ax_psd_1.set_yticks([])
    ax_psd_2.set_yticks([])

    # Set the axis labels and limits
    ax_psd_z.set_xlim(min_freq, max_freq)
    ax_psd_1.set_xlim(min_freq, max_freq)
    ax_psd_2.set_xlim(min_freq, max_freq)

    format_freq_xlabels(ax_psd_z, 
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    format_freq_xlabels(ax_psd_1,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)
    
    format_freq_xlabels(ax_psd_2,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks,
                        axis_label_size = axis_label_size, tick_label_size = tick_label_size)


    ax_psd_z.set_ylim(min_y, max_y)
    ax_psd_1.set_ylim(min_y, max_y)
    ax_psd_2.set_ylim(min_y, max_y)

    # Set the axis titles
    ax_psd_z.set_title(component2label("Z"), fontsize = title_size, fontweight = "bold")
    ax_psd_1.set_title(component2label("1"), fontsize = title_size, fontweight = "bold")
    ax_psd_2.set_title(component2label("2"), fontsize = title_size, fontweight = "bold")

    # Add a station map on the right
    ax_map = fig.add_subplot(gs[0, 3])
    add_station_map(ax_map,
                    stations_highlight = stations_highlight,
                    axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    return fig, [ax_psd_z, ax_psd_1, ax_psd_2, ax_map]

###### Functions for plotting maps ######
# Plot the station triads while ensuring that each edge is plotted only once
def plot_station_triads(ax, coord_df, triad_df, linewidth = 1.0, linecolor = "gray", zorder = 1, **kwargs):

    if "triads_to_plot" in kwargs:
        triads_to_plot_df = kwargs["triads_to_plot"]
        triad_df = merge(triad_df, triads_to_plot_df, on = ["station1", "station2", "station3"], how = "inner")

    edges_plotted = set()
    for _, row in triad_df.iterrows():
        edges = set([tuple(sorted([row["station1"], row["station2"]])), tuple(sorted([row["station2"], row["station3"]])), tuple(sorted([row["station3"], row["station1"]]))])

        for edge in edges:
            if edge not in edges_plotted:
                station1, station2 = edge
                east1, north1 = coord_df.loc[station1, ["east", "north"]]
                east2, north2 = coord_df.loc[station2, ["east", "north"]]

                ax.plot([east1, east2], [north1, north2], color = linecolor, linewidth = linewidth, zorder = zorder)
                edges_plotted.add(edge)

    return ax

###### Functions for adding elements to the plots ######

# Add day-night shading to the plot
def add_day_night_shading(ax, sun_df=None, day_color="white", night_color="lightgray"):
    if sun_df is None:
        sun_df = get_geo_sunrise_sunset_times()

    for i in range(len(sun_df) - 1):
        sunrise = sun_df.iloc[i]["sunrise"]
        sunset = sun_df.iloc[i]["sunset"]

        if i == 0:
            day_handle = ax.axvspan(sunrise, sunset, color=day_color, alpha = 0.5, zorder=0, edgecolor=None, label="Day")
        else:
            ax.axvspan(sunrise, sunset, color=day_color, alpha = 0.5, zorder=0, edgecolor=None)

        surise_next = sun_df.iloc[i + 1]["sunrise"]

        if i == 0:
            night_handle = ax.axvspan(sunset, surise_next, color=night_color, alpha = 0.5, zorder=0, edgecolor=None, label="Night")
        else:
            ax.axvspan(sunset, surise_next, color=night_color, alpha = 0.5, zorder=0, edgecolor=None)

    return ax, day_handle, night_handle

# Add a station map to the plot
# Make sure that the axis has the correct aspect ratio!
def add_station_map(ax, 
                    geo_df = None,
                    bh_df = None,
                    stations_highlight = {},
                    label_boreholes = True,
                    station_size = 30, station_label_size = 6, edge_width = 0.5,
                    borehole_size = 30, borehole_label_size = 6,
                    borehole_color = "darkviolet", station_color = "lightgray", hightlight_color = "crimson",
                    bh_label_offset_x = 20.0, bh_label_offset_y = -20.0,
                    axis_label_size = 10, tick_label_size = 8):

    if geo_df is None:
        geo_df = get_geophone_coords()

    if bh_df is None:
        bh_df = get_borehole_coords()
    
    # Plot each station and its label
    for station, row in geo_df.iterrows():
        east = row["east"]
        north = row["north"]

        if station in stations_highlight.keys():
            offset_x = stations_highlight[station][0]
            offset_y = stations_highlight[station][1]
            ha, va = get_label_alignments(offset_x, offset_y)
            
            ax.scatter(east, north, s = station_size, marker = "^", color = station_color, edgecolor = hightlight_color, linewidths = edge_width, zorder = 1)
            ax.text(east + offset_x, north + offset_y, station, fontsize = station_label_size, color = hightlight_color, ha = ha, va = va, zorder = 2)
        else:
            ax.scatter(east, north, s = station_size, marker = "^", color = station_color, edgecolor = "black", linewidths = edge_width, zorder = 1)
           
    # Plot the boreholes
    for borehole, row in bh_df.iterrows():
        east = row["east"]
        north = row["north"]

        ax.scatter(east, north, s = borehole_size, marker = "o", color = borehole_color, edgecolor = "black", linewidths = edge_width, zorder = 1)
        if label_boreholes:
            ha, va = get_label_alignments(bh_label_offset_x, bh_label_offset_y)
            ax.annotate(borehole, (east, north), xytext = (bh_label_offset_x, bh_label_offset_y), textcoords = "offset points", fontsize = borehole_label_size, color = borehole_color, ha = ha, va = va, arrowprops = dict(arrowstyle = "-", color = "black"), zorder = 2)

    # Set the aspect ratio
    ax.set_aspect("equal")

    # Format the axes
    ax.set_xlim(EASTMIN_WHOLE, EASTMAX_WHOLE)
    ax.set_ylim(NORTHMIN_WHOLE, NORTHMAX_WHOLE)

    format_east_xlabels(ax, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
    format_north_ylabels(ax, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

    return ax

# Add a colorbar to the plot
def add_colorbar(fig, position, label, 
                 orientation="vertical", axis_label_size=12, tick_label_size=10, max_num_ticks=5,
                 frameon=True, **kwargs):
    
    if "mappable" in kwargs:
        color = kwargs["mappable"]
    elif "cmap" in kwargs and "norm" in kwargs:
        color = ScalarMappable(cmap=kwargs["cmap"], norm=kwargs["norm"])
    else:
        raise ValueError("No colorbar object found!")
    
    # Add the colorbar axis
    cax = fig.add_axes(position, zorder=10)  # Ensures it's above the subplot

    # Now, draw the color bar
    cbar = fig.colorbar(color, cax=cax, orientation=orientation)

    cbar.locator = MaxNLocator(nbins=max_num_ticks)
    cbar.set_label(label, fontsize=axis_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)
    cbar.update_ticks()

    return cbar

# Add a vertical scale bar
# Coordinates are in fractions of the axis
def add_vertical_scalebar(ax, coordinates, length, scale, label_offsets, label_unit = GROUND_VELOCITY_UNIT, color = "black", linewidth = 1.0, fontsize = 12, fontweight = "bold"):
    xlim = ax.get_xlim()
    xmin = xlim[0]
    xmax = xlim[1]
    xdim = xmax - xmin

    ylim = ax.get_ylim()
    ymin = ylim[0]
    ymax = ylim[1]
    ydim = ymax - ymin

    scale_x = coordinates[0] * xdim + xmin
    scale_y = coordinates[1] * ydim + ymin

    label_x = scale_x + label_offsets[0] * xdim
    label_y = scale_y + label_offsets[1] * ydim


    ax.errorbar(scale_x, scale_y, yerr=length * scale/2, xerr=None, capsize=2.5, color=color, fmt='-', linewidth=linewidth)
    ax.text(label_x, label_y, f"{length} {label_unit}", fontsize=fontsize, color=color, ha='left', va='center')
    
    return ax

# Add a horizontal scale bar
# Coordinates are in fractions of the axis
def add_horizontal_scalebar(ax, coordinates, length, scale, 
                            color = "black", linewidth = 1.0, plot_label = True, plot_bbox = False,
                            **kwargs):
    xlim = ax.get_xlim()
    xmin = xlim[0]
    xmax = xlim[1]

    if isinstance(length, str):
        length = Timedelta(length)
        xmin = Timestamp(num2date(xmin))
        xmax = Timestamp(num2date(xmax))
    
    xdim = xmax - xmin

    ylim = ax.get_ylim()
    ymin = ylim[0]
    ymax = ylim[1]
    ydim = ymax - ymin

    scale_x = coordinates[0] * xdim + xmin
    scale_y = coordinates[1] * ydim + ymin

    # Handle the case where the x-axis is in datetime                  
    ax.errorbar(scale_x, scale_y, xerr=length * scale/2, yerr=None, capsize=2.5, color=color, fmt='-', linewidth=linewidth)

    if plot_label:
        label = kwargs.get("label", "")
        label_offsets = kwargs.get("label_offsets", (0.1, 0.1))
        fontsize = kwargs.get("fontsize", 12)
        fontweight = kwargs.get("fontweight", "normal")

        label_x = scale_x + label_offsets[0] * xdim
        label_y = scale_y + label_offsets[1] * ydim
        ax.text(label_x, label_y, f"{label}", fontsize=fontsize, color=color, ha='left', va='center', fontweight=fontweight)

    if plot_bbox:
        ax.add_patch(Rectangle((scale_x - length * scale / 2, scale_y - length * scale / 2), length * scale, length * scale, fill=False, edgecolor=color, linewidth=linewidth))
    
    return ax

###### Get colormaps for the plots ######
def get_quadmeshmap(**kwargs):
    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color='darkgray')

    if "min_db" in kwargs and "max_db" in kwargs:
        min_db = kwargs["min_db"]
        max_db = kwargs["max_db"]
        norm = Normalize(vmin=min_db, vmax=max_db)

        return cmap, norm
    else:
        return cmap
    
###### Format the axis labels of the plots ######

# Format the x labels in datetime
def format_datetime_xlabels(ax, 
                            plot_axis_label=True, plot_tick_label=True,
                            date_format = '%Y-%m-%d %H:%M:%S', 
                            major_tick_spacing = "24h", num_minor_ticks = 5,
                            axis_label_size = 12, tick_label_size = 10, rotation = 0,
                            major_tick_length = 5, minor_tick_length = 2.5, tick_width = 1,
                            va = "top", ha = "center"):


    # Convert to time deltas
    major_tick_spacing = Timedelta(major_tick_spacing)
  
    major_locator = timedelta_to_locator(major_tick_spacing)

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))
    ax.xaxis.set_major_formatter(DateFormatter(date_format))

    ax.tick_params(axis='x', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='x', which='minor', length=minor_tick_length, width=tick_width)

    if plot_tick_label:
        for label in ax.get_xticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment(va)
            label.set_horizontalalignment(ha)
            label.set_rotation(rotation)
    else:
        ax.set_xticklabels([])

    if plot_axis_label:
        ax.set_xlabel("Time (UTC)", fontsize=axis_label_size)

    return ax

# Format the x labels in frequency
def format_freq_xlabels(ax,
                        plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing=100.0, num_minor_ticks=5,
                        axis_label_size=12, tick_label_size=10,
                        major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    
    if plot_axis_label:
        ax.set_xlabel("Frequency (Hz)", fontsize=axis_label_size)

    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if plot_tick_label:
        for label in ax.get_xticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('top')
            label.set_horizontalalignment('center')
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis='x', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='x', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the x labels in relative time
def format_rel_time_xlabels(ax, label=True, 
                            major_tick_spacing=60, minor_tick_spacing=15, 
                            axis_label_size=12, tick_label_size=10,
                            major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if label:
        ax.set_xlabel("Time (s)", fontsize=axis_label_size)

    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))

    for label in ax.get_xticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('top')
        label.set_horizontalalignment('center')

    ax.tick_params(axis='x', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='x', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the x labels in easting
def format_east_xlabels(ax, 
                        plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing=50, num_minor_ticks=5,
                        axis_label_size=12, tick_label_size=10,
                        major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if plot_axis_label:
        ax.set_xlabel("East (m)", fontsize=axis_label_size)

    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if plot_tick_label:
        for label in ax.get_xticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('top')
            label.set_horizontalalignment('center')
    else:
        ax.set_xticklabels([])

    ax.tick_params(axis='x', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='x', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the y labels in northing
def format_north_ylabels(ax, 
                        plot_axis_label=True, plot_tick_label=True,
                        major_tick_spacing=50, num_minor_ticks=5,
                        axis_label_size=12, tick_label_size=10,
                        major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if plot_axis_label:
        ax.set_ylabel("North (m)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the y labels in depth
def format_depth_ylabels(ax, 
                        plot_axis_label=True, plot_tick_label=True,
                         major_tick_spacing=50, num_minor_ticks=5,
                         axis_label_size=12, tick_label_size=10,
                         major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if plot_axis_label:
        ax.set_ylabel("Depth (m)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))
    
    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

## Function to format the y labels in frequency
def format_freq_ylabels(ax, 
                        plot_axis_label=True, plot_tick_label=True,
                        abbreviation=False, 
                        major_tick_spacing=20, num_minor_ticks=4,
                        axis_label_size=12, tick_label_size=10,
                        major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if plot_axis_label:
        if abbreviation:
            ax.set_ylabel("Freq. (Hz)", fontsize=axis_label_size)
        else:
            ax.set_ylabel("Frequency (Hz)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the y labels in decibels
def format_db_ylabels(ax, sensor="geo",
                      plot_axis_label=True, plot_tick_label=True,
                      major_tick_spacing=10, num_minor_ticks=5,
                      axis_label_size=12, tick_label_size=10,
                      major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    
    if plot_axis_label:
        if sensor == "geo":
            ax.set_ylabel(GEO_PSD_LABEL, fontsize=axis_label_size)
        else:
            ax.set_ylabel(HYDRO_PSD_LABEL, fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

## Function to format the ylabel in ground velocity
def format_vel_ylabels(ax, label=True, 
                       abbreviation=False, 
                       major_tick_spacing=100, minor_tick_spacing=50, 
                       axis_label_size=12,  tick_label_size=10,
                       major_tick_length=5, minor_tick_length=2.5, tick_width=1):
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

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the y axis label in normalized amplitude
def format_norm_amp_ylabels(ax, 
                            plot_axis_label=True, plot_tick_label=True,
                            abbreviation=False, 
                            major_tick_spacing=0.5, num_minor_ticks=5,
                            axis_label_size=12, tick_label_size=12,
                            major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if plot_axis_label:
        if abbreviation:
            ax.set_ylabel("Norm. amp.", fontsize=axis_label_size)
        else:
            ax.set_ylabel("Normalized amplitude", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the ylabel in pressure
def format_pressure_ylabels(ax, label=True, 
                            abbreviation=False, 
                            major_tick_spacing=0.5, num_minor_ticks=5,
                            axis_label_size=12, tick_label_size=12,
                            major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if label:
        if abbreviation:
            ax.set_ylabel(f"Press. (mPa)", fontsize=axis_label_size)
        else:
            ax.set_ylabel(f"Pressure (mPa)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('center')
        label.set_horizontalalignment('right')

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the ylabel in pressure PSD for hydrophones
def format_hydro_psd_ylabels(ax, 
                                plot_axis_label=True, plot_tick_label=True,
                                major_tick_spacing=10, num_minor_ticks=5,
                                axis_label_size=12, tick_label_size=10,
                                major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    
    if plot_axis_label:
        ax.set_ylabel(HYDRO_PSD_LABEL, fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the ylabel in normalized PSD
def format_norm_psd_ylabels(ax, 
                            plot_axis_label=True, plot_tick_label=True,
                            major_tick_spacing=10, num_minor_ticks=5,
                            axis_label_size=12, tick_label_size=10,
                            major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    
    if plot_axis_label:
        ax.set_ylabel("Normalized PSD (dB)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the ylabel in coherence
def format_coherence_ylabels(ax, 
                              plot_axis_label=True, plot_tick_label=True,
                              major_tick_spacing=0.5, num_minor_ticks=5,
                              axis_label_size=12, tick_label_size=10,
                              major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    
    if plot_axis_label:
        ax.set_ylabel("Coherence", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))   

    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the ylabel in phase difference
def format_phase_diff_ylabels(ax, 
                              plot_axis_label=True, plot_tick_label=True,
                              abbreviation=False,
                              major_tick_spacing=pi/2, num_minor_ticks=3,
                              axis_label_size=12, tick_label_size=10,
                              major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    
    ax.set_ylim(-pi, pi)
    
    if plot_axis_label:
        if abbreviation:
            ax.set_ylabel("Phase diff. (rad)", fontsize=axis_label_size)
        else:
            ax.set_ylabel("Phase difference (rad)", fontsize=axis_label_size)
    
    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(pi_format_func))

    if plot_tick_label:
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_size) 
            label.set_verticalalignment('center')
            label.set_horizontalalignment('right')
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return ax

# Format the xlabel in slowness in s/km
def format_slowness_xlabels(ax, label=True,
                            major_tick_spacing=1.0, num_minor_ticks=5,
                            axis_label_size=10, tick_label_size=10,
                            major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if label:
        ax.set_xlabel("East slowness (s km$^{-1}$)", fontsize=axis_label_size)

    ax.xaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.xaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    for label in ax.get_xticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('top')
        label.set_horizontalalignment('center')

    ax.tick_params(axis='x', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='x', which='minor', length=minor_tick_length, width=tick_width)

    return

# Format the ylabel in slowness in s/km
def format_slowness_ylabels(ax, label=True,
                            major_tick_spacing=1.0, num_minor_ticks=5,
                            axis_label_size=10, tick_label_size=10,
                            major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if label:
        ax.set_ylabel("North slowness (s km$^{-1}$)", fontsize=axis_label_size)

    ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
    ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('center')
        label.set_horizontalalignment('right')

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return

# Format the y label in apparent velocity
def format_app_vel_ylabels(ax,
                          abbreviation=False,
                          plot_axis_label=True, plot_tick_label=True,
                          major_tick_spacing=500.0, num_minor_ticks=5,
                          axis_label_size=12, tick_label_size=10,
                          major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if plot_axis_label:
        if abbreviation:
            ax.set_ylabel(f"{APPARENT_VELOCITY_LABEL_SHORT}", fontsize=axis_label_size)
        else:
            ax.set_ylabel(f"{APPARENT_VELOCITY_LABEL}", fontsize=axis_label_size)

    if plot_tick_label:
        ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
        ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))
    else:
        ax.set_yticklabels([])

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('center')
        label.set_horizontalalignment('right')

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return

# Format the y label in back azimuth
def format_back_azi_ylabels(ax,
                            abbreviation=False,
                            plot_axis_label=True, plot_tick_label=True,
                            major_tick_spacing=45.0, num_minor_ticks=3,
                            axis_label_size=12, tick_label_size=10,
                            major_tick_length=5, minor_tick_length=2.5, tick_width=1):
    if plot_axis_label:
        if abbreviation:
            ax.set_ylabel("Back azi. (deg)", fontsize=axis_label_size)
        else:
            ax.set_ylabel("Back azimuth (deg)", fontsize=axis_label_size)

    if plot_tick_label:
        ax.yaxis.set_major_locator(MultipleLocator(major_tick_spacing))
        ax.yaxis.set_minor_locator(AutoMinorLocator(num_minor_ticks))
    else:
        ax.set_yticklabels([])

    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_size) 
        label.set_verticalalignment('center')
        label.set_horizontalalignment('right')

    ax.tick_params(axis='y', which='major', length=major_tick_length, width=tick_width)
    ax.tick_params(axis='y', which='minor', length=minor_tick_length, width=tick_width)

    return


###### Functions for handling colors ######
# Create a cunstomized colormap using a portion of an existing colormap
def get_cmap_segment(cmap, begin_portion, end_portion, num = 256):
    cmap_in = get_cmap(cmap)
    cmap_segment = cmap_in(linspace(begin_portion, end_portion, num))
    cmap_out = LinearSegmentedColormap.from_list("customized_cmap", cmap_segment, N=num)

    return cmap_out
    

# Create a categorical colormap with more categories by interpolating a segment of an existing colormap
def get_interp_cat_colors(cmap, begin_index, end_index, num_cat):
    # Extract the segment of the colormap
    cmap_in = get_cmap(cmap)
    colors_in = [cmap_in(i) for i in range(begin_index, end_index + 1)]
    
    # Interpolate the colors
    custom_cmap = LinearSegmentedColormap.from_list("customized_cmap", colors_in, N=num_cat)
    colors_out = [custom_cmap(i) for i in range(num_cat)]

    return colors_out

###### Basic utility functions ######

# Convert a Timedelta to a datetime locator
def timedelta_to_locator(time_delta):
    if time_delta.days > 0:
        locator = DayLocator(interval=time_delta.days)  # Use DayLocator for days
    elif time_delta.seconds >= 3600:
        locator = HourLocator(interval=time_delta.seconds // 3600)  # Use HourLocator for hours
    elif time_delta.seconds >= 60:
        locator = MinuteLocator(interval=time_delta.seconds // 60)  # Use MinuteLocator for minutes
    elif time_delta.seconds >= 1:
        locator = SecondLocator(interval=time_delta.seconds)  # Use SecondLocator for seconds
    elif time_delta.microseconds >= 1:
        locator = MicrosecondLocator(interval=time_delta.microseconds)  # Use MicrosecondLocator for microseconds


    return locator

# Function for getting the color for the three geophone components
def get_geo_component_color(component):
    if component == "Z":
        color = "tab:blue"
    elif component == "1":
        color = "tab:orange"
    elif component == "2":
        color = "tab:green"
    else:
        raise ValueError("Invalid component name!")

    return color

# Function to convert component names to subplot titles
def component2label(component):
    if component == "Z":
        title = "Up"
    elif component == "1":
        title = "North"
    elif component == "2":
        title = "East"
    else:
        raise ValueError("Invalid component name!")

    return title

# Function to convert the frequency band to a label
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

# Get STFT parameter labels
def get_stft_param_labels(window_length, freq_interval):
    if freq_interval < 1:
        label = f"Window length: {window_length:.0f} s, Freq. sampling: {freq_interval:.3} Hz"
    else:
        label = f"Window length: {window_length:.0f} s, Freq. sampling: {freq_interval:.0f} Hz"

    return label

# Get label alignment from the x and y offsets
def get_label_alignments(x_offset, y_offset):
    if x_offset > 0 and y_offset > 0:
        va = "bottom"
        ha = "left"
    elif x_offset > 0 and y_offset < 0:
        va = "top"
        ha = "left"
    elif x_offset < 0 and y_offset > 0:
        va = "bottom"
        ha = "right"
    elif x_offset < 0 and y_offset < 0:
        va = "top"
        ha = "right"

    return ha, va

# Function for saving a figure
def save_figure(fig, filename, outdir=FIGURE_DIR, dpi=300):
    fig.patch.set_alpha(0)

    outpath = join(outdir, filename)

    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {outpath}")

# Format a phase difference in radians to a fraction of pi
def pi_format_func(value, tick_number):
    # Convert value to multiples of π
    if value == 0:
        return "0"
    elif value == pi:
        return "π"
    elif value == -pi:
        return "-π"
    elif value >0:
        return f"π/{pi / value:.0f}"
    else:
        return f"-π/{pi / abs(value):.0f}"
