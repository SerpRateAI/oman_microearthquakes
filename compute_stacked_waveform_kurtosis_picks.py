"""
Compute the rolling kurtosis of the stacked waveforms and estimate the arrival times from the results
"""
from argparse import ArgumentParser
from pathlib import Path
from numpy import arange, mean, std, isnan, nan
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
from utils_plot import get_geo_component_color, save_figure
from utils_kurtosis import rolling_kurtosis, get_arrival_time
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

"""
Plot the 3-C kurtosis of a station.
"""
def plot_station_kurtosis(ax, kurtosis_dict,
                          linewidth = 3.0,
                          max_abs_amplitude = 7.0):

    for component, kurtosis in kurtosis_dict.items():
        num_pts = len(kurtosis)
        timeax = arange(num_pts) / sampling_rate
        color = get_geo_component_color(component)
        ax.plot(timeax, kurtosis, linewidth=linewidth, linestyle = ":", label=component, color=color, zorder=1)
    
    ax.set_ylim(-max_abs_amplitude, max_abs_amplitude)

    return ax

#--------------------------------------------------------------------------------------------------
# Define the main function
#--------------------------------------------------------------------------------------------------

def main():
    # Get the command line arguments
    parser = ArgumentParser()
    parser.add_argument("--template_id", type=str)
    parser.add_argument("--window_length_kurtosis", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=4.0)

    args = parser.parse_args()

    # Get the input parameters
    template_id = args.template_id
    window_length_kurtosis = args.window_length_kurtosis
    threshold = args.threshold

    # Load the stacked waveforms
    print(f"Loading the stacked waveforms for Template {template_id}")
    filename = f"matched_waveform_stack_template{template_id}.mseed"
    filepath = Path(dirpath_detection) / filename
    stream = read(filepath)
    starttime = stream[0].stats.starttime

    # Get the unique stations
    print(f"Getting the unique stations")
    stations = get_unique_stations(stream)

    # Compute the kurtosis and estimate the arrival times
    print(f"Computing the kurtosis and estimating the arrival times")
    kurtosis_dict = {}
    for station in stations:
        stream_sta = stream.select(station = station)
        kurtosis_sta_dict = {}
        for component in components:
            stream_comp = stream_sta.select(component = component)
            waveform =stream_comp[0].data
            kurtosis_sta_dict[component] = rolling_kurtosis(waveform, window_length_kurtosis)
        kurtosis_dict[station] = kurtosis_sta_dict

    # Estimate the arrival times
    arrival_time_dict = {}
    for station in stations:
        arrival_time_sta_dict = {}
        for component in components:
            kurtosis_sta_dict = kurtosis_dict[station]
            kurtosis = kurtosis_sta_dict[component]
            timeax = arange(len(kurtosis)) / sampling_rate
            arrival_time = get_arrival_time(kurtosis, timeax, threshold)
            arrival_time_sta_dict[component] = arrival_time

        ## Compute the mean arrival time
        arrival_time_mean = mean(list(arrival_time_sta_dict.values()))
        arrival_time_std = std(list(arrival_time_sta_dict.values()))
        arrival_time_sta_dict["mean"] = arrival_time_mean
        arrival_time_sta_dict["std"] = arrival_time_std
        arrival_time_dict[station] = arrival_time_sta_dict

    # Plot the waveforms, kurtosis, and arrival times
    print(f"Plotting the waveforms, kurtosis, and arrival times")
    fig, axes = subplots(len(stations), 1, figsize=(10, 15), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    for i_station, station in enumerate(stations):
        stream_sta = stream.select(station = station)
        waveform_sta_dict = {}
        for component in components:
            stream_comp = stream_sta.select(component = component)
            waveform =stream_comp[0].data
            waveform_sta_dict[component] = waveform
            ax = axes[i_station]

        plot_station_waveforms(ax, waveform_sta_dict)
        
        ax.set_xlim([0, timeax[-1]])
        ax.text(0.0, 0.0, station, fontsize=12, va = "bottom", ha = "left", fontweight = "bold")

        ## Plot the kurtosis on a twin axis
        ax_waveform = axes[i_station]
        ax_kurtosis = ax_waveform.twinx()

        kurtosis_sta_dict = kurtosis_dict[station]
        plot_station_kurtosis(ax_kurtosis, kurtosis_sta_dict)

        ## Plot the arrival times
        arrival_time_sta_dict = arrival_time_dict[station]
        for component in components:
            arrival_time = arrival_time_sta_dict[component]
            color = get_geo_component_color(component)
            ax_waveform.axvline(arrival_time, color=color, linewidth = 3.0)

        ## Plot the mean arrival time
        arrival_time_mean = arrival_time_sta_dict["mean"]
        color = "crimson"
        ax_waveform.axvline(arrival_time_mean, color=color, linewidth = 3.0)

        fig.suptitle(f"Kurtosis, Stacked Waveforms, Template {template_id}", fontsize=16, fontweight = "bold", y = 0.9)

    # Save the mean pick times
    output_dicts = []
    for station in stations:
        ## Get the absolute time of the mean pick
        arrival_time  = arrival_time_dict[station]["mean"]
        if isnan(arrival_time):
            continue
        else:
            arrival_time_abs = starttime + Timedelta(seconds = arrival_time)
            uncertainty = arrival_time_dict[station]["std"]

        output_dict = {
            "station": station,
            "arrival_time": arrival_time_abs,
            "uncertainty": uncertainty
        }
        output_dicts.append(output_dict)

    # Save the output dictionary
    output_df = DataFrame(output_dicts)
    output_df.sort_values(by = "arrival_time", inplace = True)
    outpath = Path(dirpath_pick) / f"template_arrivals_kurtosis_stack_{template_id}.csv"
    output_df.to_csv(outpath, index=False, na_rep="nan")
    print(f"Saved the mean pick times to {outpath}")

    # Save the figure
    print(f"Saving the figure")
    save_figure(fig, f"stacked_waveform_kurtosis_picks_template{template_id}.png")

if __name__ == "__main__":
    main()