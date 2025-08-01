"""
Compute the rolling kurtosis of the hammer waveforms and estimate the arrival times from the results
"""
from argparse import ArgumentParser
from pathlib import Path
from numpy import arange, std, isnan, amax, where, nan
from pandas import Timedelta, DataFrame
from obspy import read
from matplotlib.pyplot import subplots
from utils_basic import (
                         GEO_COMPONENTS as components,
                         PICK_DIR as dirpath_pick,
                         DETECTION_DIR as dirpath_detection,
                         SAMPLING_RATE as sampling_rate,
                         get_unique_stations
                         )

from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_plot import get_geo_component_color, save_figure
from utils_loc import process_arrival_info
from utils_snuffler import read_time_windows
from utils_sta_lta import compute_sta_lta

#-------------------------------------------------------------------------------------------------- 
# Define the functions
#--------------------------------------------------------------------------------------------------
"""
Plot the 3-C waveforms of a station.
"""
def plot_station_waveforms(ax, waveform_dict,
                            linewidth = 1.0,
                            max_abs_amplitude = 0.3):

    for component, waveform in waveform_dict.items():
        num_pts = len(waveform)
        timeax = arange(num_pts) / sampling_rate
        color = get_geo_component_color(component)
        ax.plot(timeax, waveform, linewidth=linewidth, label=component, color=color, zorder=1)
    
    ax.set_ylim(-max_abs_amplitude, max_abs_amplitude)

    return ax

"""
Plot the average CF of a station.
"""
def plot_station_cf(ax, cf,
                        linewidth = 3.0,
                        max_abs_amplitude = 3.0):

    num_pts = len(cf)
    timeax = arange(num_pts) / sampling_rate
    ax.plot(timeax, cf, linewidth=linewidth, linestyle = ":", color="crimson", zorder=1)
    
    ax.set_ylim(-max_abs_amplitude, max_abs_amplitude)

    return ax

"""
Get the trigger time from the STA/LTA characteristic function
"""
def get_trigger_time(cf, threshold, timeax, get_uncertainty = True):
    indices = where(cf > threshold)[0]

    if len(indices) == 0:
        if get_uncertainty:
            return nan, nan
        else:
            return nan
    else:
        i_trigger = indices[0]
        trigger_time = timeax[i_trigger]

        if get_uncertainty:
            noise_std = std(cf[:i_trigger])
            slope_before = (cf[i_trigger] - cf[i_trigger - 1]) / (timeax[i_trigger] - timeax[i_trigger - 1])
            slope_after = (cf[i_trigger + 1] - cf[i_trigger]) / (timeax[i_trigger + 1] - timeax[i_trigger])
            trigger_time_std = noise_std / (slope_before + slope_after) * 2
            return trigger_time, trigger_time_std
        else:
            return trigger_time

#--------------------------------------------------------------------------------------------------
# Define the main function
#--------------------------------------------------------------------------------------------------

def main():
    # Get the command line arguments
    parser = ArgumentParser()
    parser.add_argument("--hammer_id", type=str)
    parser.add_argument("--window_length_sta", type=float, default=0.02)
    parser.add_argument("--window_length_lta", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--buffer_begin", type=float, default=0.1)
    parser.add_argument("--buffer_end", type=float, default=0.3)

    args = parser.parse_args()

    # Get the input parameters
    hammer_id = args.hammer_id
    window_length_sta = args.window_length_sta
    window_length_lta = args.window_length_lta
    threshold = args.threshold
    buffer_begin = args.buffer_begin
    buffer_end = args.buffer_end

    # Load the hammer waveforms
    print(f"Loading the hammer waveforms for Hammer {hammer_id}")
    filename = f"hammer_windows_{hammer_id}.txt"
    filepath = Path(dirpath_pick) / filename
    arrival_df = read_time_windows(filepath)
    starttime = arrival_df["starttime"].min()

    starttime_waveform = starttime - Timedelta(seconds = buffer_begin)
    endtime_waveform = starttime + Timedelta(seconds = buffer_end)

    stations = arrival_df["station"].to_list()

    # Read the hammer waveforms
    print(f"Reading the hammer waveforms")
    stream = read_and_process_windowed_geo_waveforms(starttime_waveform, endtime=endtime_waveform, stations = stations)

    # Compute the rolling kurtosis for the vertical component only
    print(f"Computing the STA/LTA")
    cf_dict = {}
    for station in stations:
        stream_sta = stream.select(station = station)
        for i, component in enumerate(components):
            stream_comp = stream_sta.select(component = component)
            waveform =stream_comp[0].data
            cf = compute_sta_lta(waveform, sampling_rate, window_length_sta, window_length_lta)
            if i == 0:
                cf_sta = cf
            else:
                cf_sta += cf
        cf_dict[station] = cf_sta / len(components)

    # Estimate the arrival times
    print(f"Estimating the arrival times")
    arrival_time_dict = {}
    for station in stations:
        cf_sta = cf_dict[station]
        timeax = arange(len(cf_sta)) / sampling_rate
        arrival_time, arrival_time_std = get_trigger_time(cf_sta, threshold, timeax, get_uncertainty = True)
        arrival_time_dict[station] = (arrival_time, arrival_time_std)

    # Plot the waveforms, kurtosis, and arrival times
    print(f"Plotting the waveforms, STA/LTA, and arrival times")
    fig, axes = subplots(len(stations), 1, figsize=(10, 15), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    for i_station, station in enumerate(stations):
        stream_sta = stream.select(station = station)
        waveform_sta_dict = {}
        max_amp = 0
        ax = axes[i_station]

        for component in components:
            stream_comp = stream_sta.select(component = component)
            waveform = stream_comp[0].data
            max_amp_comp = amax(abs(waveform))
            max_amp = max(max_amp, max_amp_comp)
            waveform_sta_dict[component] = waveform
            
        ## Normalize the traces
        for component in components:
            waveform_sta_dict[component] = waveform_sta_dict[component] / max_amp

        plot_station_waveforms(ax, waveform_sta_dict)

        ax.set_xlim([0, timeax[-1]])
        ax.text(0.0, 0.0, station, fontsize=12, va = "bottom", ha = "left", fontweight = "bold")

        ## Plot the STA/LTA on a twin axis
        ax_waveform = axes[i_station]
        ax_sta_lta = ax_waveform.twinx()

        cf_sta = cf_dict[station]
        plot_station_cf(ax_sta_lta, cf_sta)
    
        ## Plot the arrival times
        arrival_time, arrival_time_std = arrival_time_dict[station]
        color = "crimson"
        ax_waveform.axvline(arrival_time, color=color, linewidth = 3.0)
        ax_waveform.axvline(arrival_time + arrival_time_std, color=color, linewidth = 2.0, linestyle = "--")
        ax_waveform.axvline(arrival_time - arrival_time_std, color=color, linewidth = 2.0, linestyle = "--")

        fig.suptitle(f"STA/LTA, Hammer {hammer_id}", fontsize=16, fontweight = "bold", y = 0.9)

    # Save the mean pick times
    output_dicts = []
    for station in stations:
        arrival_time, arrival_time_std = arrival_time_dict[station]
        if isnan(arrival_time):
            continue
        else:
            arrival_time_abs = starttime + Timedelta(seconds = arrival_time)
            uncertainty = arrival_time_std
        output_dict = {
            "station": station,
            "arrival_time": arrival_time_abs,
            "uncertainty": uncertainty
        }
        output_dicts.append(output_dict)

    # Save the arrival times
    print(f"Saving the arrival times")
    output_df = DataFrame(output_dicts)
    output_df.sort_values(by = "arrival_time", inplace = True)
    outpath = Path(dirpath_detection) / f"hammer_arrivals_sta_lta_{hammer_id}.csv"
    output_df.to_csv(outpath, index=False, na_rep="nan")
    print(f"Saved the arrival times to {outpath}")

    # Save the figure
    figname = f"hammer_sta_lta_arrivals_{hammer_id}.png"
    save_figure(fig, figname, dpi=300)

if __name__ == "__main__":
    main()