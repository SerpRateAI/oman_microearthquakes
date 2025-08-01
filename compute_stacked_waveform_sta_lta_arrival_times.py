"""
Compute the STA/LTA method to compute the characteristic function of the stacked waveforms and estimate the arrival times from the results
"""
from argparse import ArgumentParser
from pathlib import Path
from numpy import arange, mean, std, isnan, nan, where
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
from utils_sta_lta import compute_sta_lta
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
            print(f"Noise std: {noise_std}")
            slope_before = (cf[i_trigger] - cf[i_trigger - 1]) / (timeax[i_trigger] - timeax[i_trigger - 1])
            print(f"Slope before: {slope_before}")
            slope_after = (cf[i_trigger + 1] - cf[i_trigger]) / (timeax[i_trigger + 1] - timeax[i_trigger])
            print(f"Slope after: {slope_after}")
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
    parser.add_argument("--template_id", type=str)
    parser.add_argument("--window_length_sta", type=float, default=0.02)
    parser.add_argument("--window_length_lta", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=4.0)

    args = parser.parse_args()

    # Get the input parameters
    template_id = args.template_id
    window_length_sta = args.window_length_sta
    window_length_lta = args.window_length_lta
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
    arrival_time_dict = {}
    for station in stations:
        print(f"Estimating the arrival times for station {station}")
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
        for component in components:
            stream_comp = stream_sta.select(component = component)
            waveform =stream_comp[0].data
            waveform_sta_dict[component] = waveform
            ax = axes[i_station]

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

        fig.suptitle(f"STA/LTA, Stacked Waveforms, Template {template_id}", fontsize=16, fontweight = "bold", y = 0.9)

    # Save the arrival times
    output_dicts = []
    for station in stations:
        arrival_time, arrival_time_std = arrival_time_dict[station]
        if isnan(arrival_time):
            continue
        else:
            arrival_time_abs = starttime + Timedelta(seconds = arrival_time)
            uncertainty = arrival_time_std
        output_dicts.append({
            "station": station,
            "arrival_time": arrival_time_abs,
            "uncertainty": uncertainty
        })

    output_df = DataFrame(output_dicts)
    output_df.sort_values(by = "arrival_time", inplace = True)
    outpath = Path(dirpath_detection) / f"template_arrivals_sta_lta_stack_{template_id}.csv"
    output_df.to_csv(outpath, index=False, na_rep="nan")
    print(f"Saved the mean pick times to {outpath}")

    # Save the figure
    print(f"Saving the figure")
    save_figure(fig, f"stacked_waveform_sta_lta_arrivals_template{template_id}.png")

if __name__ == "__main__":
    main()