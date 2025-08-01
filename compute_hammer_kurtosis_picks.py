"""
Compute the rolling kurtosis of the hammer waveforms and estimate the arrival times from the results
"""
from argparse import ArgumentParser
from pathlib import Path
from numpy import arange, mean, std, isnan, amax
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
from utils_kurtosis import rolling_kurtosis, get_arrival_time
from utils_loc import process_arrival_info
from utils_snuffler import read_time_windows

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

# """
# Plot the 3-C kurtosis of a station.
# """
# def plot_station_kurtosis(ax, kurtosis_dict,
#                           linewidth = 3.0,
#                           max_abs_amplitude = 7.0):

#     for component, kurtosis in kurtosis_dict.items():
#         num_pts = len(kurtosis)
#         timeax = arange(num_pts) / sampling_rate
#         color = get_geo_component_color(component)
#         ax.plot(timeax, kurtosis, linewidth=linewidth, linestyle = ":", label=component, color=color, zorder=1)
    
#     ax.set_ylim(-max_abs_amplitude, max_abs_amplitude)

#     return ax

#--------------------------------------------------------------------------------------------------
# Define the main function
#--------------------------------------------------------------------------------------------------

def main():
    # Get the command line arguments
    parser = ArgumentParser()
    parser.add_argument("--hammer_id", type=str)
    parser.add_argument("--window_length_kurtosis", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--buffer_begin", type=float, default=0.05)
    parser.add_argument("--buffer_end", type=float, default=0.1)

    args = parser.parse_args()

    # Get the input parameters
    hammer_id = args.hammer_id
    window_length_kurtosis = args.window_length_kurtosis
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
    print(f"Computing the rolling kurtosis")
    kurtosis_dict = {}
    for station in stations:
        stream_z = stream.select(station = station, component = "Z")
        waveform = stream_z[0].data
        kurtosis = rolling_kurtosis(waveform, window_length_kurtosis)

        kurtosis_dict[station] = kurtosis
    
    # Estimate the arrival times
    print(f"Estimating the arrival times")
    arrival_time_dict = {}
    for station in stations:
        arrival_time_sta_dict = {}

        kurtosis = kurtosis_dict[station]
        timeax = arange(len(kurtosis)) / sampling_rate
        arrival_time = get_arrival_time(kurtosis, timeax, threshold)
        arrival_time_dict[station] = arrival_time

    # Plot the waveforms, kurtosis, and arrival times
    print(f"Plotting the waveforms, kurtosis, and arrival times")
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

        ## Plot the kurtosis on a twin axis
        ax_waveform = axes[i_station]
        ax_kurtosis = ax_waveform.twinx()

        kurtosis = kurtosis_dict[station]
        num_pts = len(kurtosis)
        timeax = arange(num_pts) / sampling_rate
        color = get_geo_component_color("Z")
        ax_kurtosis.plot(timeax, kurtosis, linewidth=3.0, linestyle = ":", label=component, color=color, zorder=1)
    
        ax_kurtosis.set_ylim(-7.0, 7.0)
        
        ## Plot the arrival times
        arrival_time = arrival_time_dict[station]
        ax_waveform.axvline(arrival_time, color=color, linewidth = 3.0)

        fig.suptitle(f"Kurtosis, Hammer {hammer_id}", fontsize=16, fontweight = "bold", y = 0.9)

    # Save the mean pick times
    output_dicts = []
    for station in stations:
        arrival_time  = arrival_time_dict[station]
        if isnan(arrival_time):
            continue
        else:
            arrival_time_abs = starttime + Timedelta(seconds = arrival_time)

        output_dict = {
            "station": station,
            "arrival_time": arrival_time_abs,
        }
        output_dicts.append(output_dict)

    # Save the arrival times
    print(f"Saving the arrival times")
    output_df = DataFrame(output_dicts)
    output_df.sort_values(by = "arrival_time", inplace = True)
    outpath = Path(dirpath_detection) / f"hammer_arrivals_kurtosis_picks_{hammer_id}.csv"
    output_df.to_csv(outpath, index=False, na_rep="nan")
    print(f"Saved the arrival times to {outpath}")

    # Save the figure
    figname = f"hammer_kurtosis_picks_{hammer_id}.png"
    save_figure(fig, figname, dpi=300)

if __name__ == "__main__":
    main()