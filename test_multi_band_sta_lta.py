"""
Test estimating the STA/LTA arrival times and uncertainties using data filtered in multiple frequency bands
"""
#--------------------------------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from pathlib import Path
from numpy import arange, mean, std, isnan, nan, where, logspace, log10, amax
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

from utils_basic import GEO_COMPONENTS as components
from utils_plot import get_geo_component_color, save_figure
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
        print(amax(cf))
        print(cf[i_trigger])

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
# Main
#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    template_id = "0114040016"
    station = "A10"

    min_freq_filter = 50.0
    max_freq_filter = 500.0

    window_length_sta = 0.005
    window_length_lta = 0.05

    num_freq_elements = 24
    num_freq_bands = 3

    threshold = 2.5

    freq_elements = logspace(log10(min_freq_filter), log10(max_freq_filter), num_freq_elements)

    # Generate the frequency bands
    num_elements_per_band = num_freq_elements // num_freq_bands - 1
    freq_bands = []
    for i_band in range(num_freq_bands):
        freq_band = (freq_elements[i_band * num_elements_per_band], freq_elements[i_band * num_elements_per_band + num_elements_per_band - 1])
        freq_bands.append(freq_band)

    # Load the stacked waveforms
    print(f"Loading the stacked waveforms")
    filename = f"matched_waveform_stack_template{template_id}.mseed"
    filepath = Path(dirpath_detection) / filename
    stream = read(filepath, station = station)
    stream = stream.select(station = station)
    print(stream)
    starttime = stream[0].stats.starttime

    # Filter the waveforms and compute the STA-LTA
    cf_dict = {}
    arrival_time_dict = {}
    stream_filtered_dict = {}
    for i_band, freq_band in enumerate(freq_bands):
        print(f"Filtering the waveforms for frequency band {i_band + 1} of {num_freq_bands}")
        stream_filtered = stream.copy()
        stream_filtered.filter("bandpass", freqmin = freq_band[0], freqmax = freq_band[1])
        stream_filtered.normalize()
        stream_filtered_dict[freq_band] = stream_filtered
    
        for i_component, component in enumerate(components):
            stream_comp = stream_filtered.select(component = component)
            waveform =stream_comp[0].data
            cf = compute_sta_lta(waveform, sampling_rate, window_length_sta, window_length_lta)
            if i_component == 0:
                cf_band = cf
            else:
                cf_band += cf

        cf_band = cf_band / len(components)
        cf_dict[freq_band] = cf_band

        # Get the arrival time
        timeax = arange(len(cf_band)) / sampling_rate
        arrival_time = get_trigger_time(cf_band, threshold, timeax, get_uncertainty = False)
        arrival_time_dict[freq_band] = arrival_time

    # Plot the results
    # Plot the waveforms, characteristic function, and arrival times
    print(f"Plotting the waveforms, STA/LTA, and arrival times")
    fig, axes = subplots(len(freq_bands), 1, figsize=(10, 15), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    for i_band, freq_band in enumerate(freq_bands):
        stream = stream_filtered_dict[freq_band]
        waveform_sta_dict = {}
        for component in components:
            stream_comp = stream.select(component = component)
            waveform =stream_comp[0].data
            waveform_sta_dict[component] = waveform
            ax = axes[i_band]

        plot_station_waveforms(ax, waveform_sta_dict)
        
        ax.set_xlim([0, timeax[-1]])
        ax.text(0.0, 0.1, f"{freq_band[0]:.1f} - {freq_band[1]:.1f} Hz", fontsize=12, va = "bottom", ha = "left", fontweight = "bold")

        ## Plot the STA/LTA on a twin axis
        ax_waveform = axes[i_band]
        ax_cf = ax_waveform.twinx()

        cf_band = cf_dict[freq_band]

        plot_station_cf(ax_cf, cf_band)

        ## Plot the arrival times
        arrival_time = arrival_time_dict[freq_band]
        color = "crimson"
        ax_waveform.axvline(arrival_time, color=color, linewidth = 3.0)

    fig.suptitle(f"STA/LTA, Stacked Waveforms, Template {template_id}, {station}", fontsize=16, fontweight = "bold", y = 0.9)

    # Save the figure
    print(f"Saving the figure")
    save_figure(fig, f"test_multi_band_sta_lta_template{template_id}_{station}.png")

