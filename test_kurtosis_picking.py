"""
Test picking the arrival time of impulsive events using rolling kurtosis
"""

# Import modules
from __future__ import annotations
from typing import Literal
from pathlib import Path

from numpy import asarray, full, nan, arange, ndarray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis
from pandas import Timedelta
from obspy import read
from matplotlib.pyplot import subplots

from utils_basic import PICK_DIR as dirpath_pick, ROOTDIR_GEO as dirpath_geo, DETECTION_DIR as dirpath_detection, GEO_COMPONENTS as components, SAMPLING_RATE as sampling_rate
from utils_snuffler import read_time_windows
from utils_cont_waveform import load_waveform_slice
from utils_plot import get_geo_component_color, save_figure

# from plot_template_event_waveforms_and_arrival_times import plot_station_waveforms

#--------------------------------------------------------------------------------------------------
# Define the helper functions
#--------------------------------------------------------------------------------------------------

"""
rolling_kurtosis.py

Compute rolling (windowed) kurtosis of a 1-D time series using SciPy only
(no Pandas). The window is specified in SECONDS, and the sampling rate fs
defaults to 1000 Hz. By default, returns *excess* kurtosis (normal → 0) with
the small-sample bias correction (bias=False).

Author: Tianze Liu
"""

def rolling_kurtosis(
    x: ndarray,
    win_sec: float,
    *,
    fs: float = 1000.0,
    fisher: bool = True,
    bias: bool = False,
    mode: Literal["same", "valid"] = "same",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ndarray:
    """
    Rolling kurtosis of a 1-D array.

    Parameters
    ----------
    x : ndarray
        1-D input signal.
    win_sec : float
        Window length in seconds (must be > 0 and correspond to >= 4 samples).
    fs : float, default 1000.0
        Sampling rate in Hz (must be > 0).
    fisher : bool, default True
        If True, return *excess* kurtosis (normal → 0).
        If False, return Pearson kurtosis (normal → 3).
    bias : bool, default False
        If False, apply the small-sample correction (unbiased under normality).
        If True, use the biased (population) estimator).
    mode : {"same", "valid"}, default "same"
        "same": return an array the same length as x, centered; edges are NaN.
        "valid": return only the values where the full window fits,
                 length = len(x) - window + 1.
    nan_policy : {"propagate", "omit", "raise"}, default "propagate"
        How to handle NaNs within a window (as in scipy.stats.kurtosis).

    Returns
    -------
    ndarray
        Rolling kurtosis array. If mode="same", length equals len(x) with
        NaNs padding both ends; if "valid", length equals len(x)-window+1.

    Notes
    -----
    - Requires NumPy >= 1.20 for sliding_window_view.
    - For even window sizes, the window "center" is placed at index i + window//2.
    """
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if win_sec <= 0:
        raise ValueError("win_sec must be positive.")

    # Convert seconds to samples
    window = int(round(win_sec * fs))
    if window < 4:
        raise ValueError(
            f"win_sec × fs yields window={window} < 4; increase win_sec or fs."
        )

    x = asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    n = x.size
    if n < window:
        # No full window fits; return empty for 'valid' and NaN-padded for 'same'
        return full(n if mode == "same" else 0, nan, dtype=float)

    # Create a (num_windows, window) view without copying
    W = sliding_window_view(x, window_shape=window)  # shape: (n - window + 1, window)

    # Compute kurtosis along the window axis
    k_valid = kurtosis(W, axis=1, fisher=fisher, bias=bias, nan_policy=nan_policy)

    if mode == "same":
        k = full(n, nan, dtype=float)
        centers = arange(k_valid.size) + (window // 2)
        k[centers] = k_valid
        return k
    elif mode == "valid":
        return k_valid
    else:
        raise ValueError("mode must be 'same' or 'valid'.")
    
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
    # Define the input parameters
    template_id = "0114040016"
    min_freq_filter = 20.0
    window_length_kurtosis = 0.03 # seconds
    window_length_begin = 0.05 # seconds
    window_length_end = 0.1 # seconds
    window_length_plot = window_length_end + window_length_begin

    # Load the arrival time windows 
    filepath = Path(dirpath_pick) / f"template_arrivals_{template_id}.txt"
    pick_df = read_time_windows(filepath)
    starttime = pick_df["starttime"].min()

    starttime_waveform = starttime - Timedelta(seconds=window_length_begin)
    endtime_waveform = starttime + Timedelta(seconds=window_length_end)
    stations = pick_df["station"].unique()

    # Compute the kurtosis for the template event 
    ## Load the template-event waveform
    filename = f"preprocessed_data_min{min_freq_filter:.0f}hz.h5"
    hdf5_path = str(Path(dirpath_geo) / filename)
    waveform_dict = {}
    for station in stations:
        waveform_sta_dict = load_waveform_slice(hdf5_path, station, starttime_waveform, endtime=endtime_waveform, normalize=True)
        waveform_dict[station] = waveform_sta_dict

    ## Compute kurtosis
    kurtosis_dict = {}
    for station in stations:
        waveform_sta_dict = waveform_dict[station]
        kurtosis_sta_dict = {}
        for component in components:
            kurtosis_sta_dict[component] = rolling_kurtosis(waveform_sta_dict[component], window_length_kurtosis)
        kurtosis_dict[station] = kurtosis_sta_dict

    ## Plot the waveforms and kurtosis
    ### Generate the axes
    num_sta = len(stations)
    fig, axes = subplots(num_sta, 1, figsize=(10, 15), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    ### Plot the waveforms
    for i_station, station in enumerate(stations):
        for component in components:
            waveform_sta_dict = waveform_dict[station]
            ax = axes[i_station]
            plot_station_waveforms(ax, waveform_sta_dict)
            
            ax.set_xlim([0, window_length_plot])
            ax.text(0.0, 0.0, station, fontsize=12, va = "bottom", ha = "left", fontweight = "bold")


    ### Plot the kurtosis on a twin axis
    for i_station, station in enumerate(stations):
        for component in components:
            kurtosis_sta_dict = kurtosis_dict[station]
            ax_waveform = axes[i_station]
            ax_kurtosis = ax_waveform.twinx()

            plot_station_kurtosis(ax_kurtosis, kurtosis_sta_dict)

    fig.suptitle(f"Kurtosis, Template Event, {template_id}", fontsize=16, fontweight = "bold", y = 0.9)

    ### Save the figure
    save_figure(fig, "kurtosis_test_template.png")

    # Compute the kurtosis for the stacked waveforms
    ## Load the waveforms
    filename = f"matched_waveform_stack_template{template_id}.mseed"
    filepath = Path(dirpath_detection) / filename
    stream = read(filepath)

    ## Compute kurtosis
    kurtosis_dict = {}
    for station in stations:
        stream_sta = stream.select(station = station)
        kurtosis_sta_dict = {}
        for component in components:
            stream_comp = stream_sta.select(component = component)
            waveform =stream_comp[0].data
            kurtosis_sta_dict[component] = rolling_kurtosis(waveform, window_length_kurtosis)
        kurtosis_dict[station] = kurtosis_sta_dict

    ## Plot the waveforms and kurtosis
    ### Generate the axes
    fig, axes = subplots(num_sta, 1, figsize=(10, 15), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    ### Plot the waveforms
    for i_station, station in enumerate(stations):
        stream_sta = stream.select(station = station)
        waveform_sta_dict = {}
        for component in components:
            stream_comp = stream_sta.select(component = component)
            waveform =stream_comp[0].data
            waveform_sta_dict[component] = waveform
            ax = axes[i_station]
            plot_station_waveforms(ax, waveform_sta_dict)
            
            ax.set_xlim([0, window_length_plot])
            ax.text(0.0, 0.0, station, fontsize=12, va = "bottom", ha = "left", fontweight = "bold")

    
    ### Plot the kurtosis on a twin axis
    for i_station, station in enumerate(stations):
        for component in components:
            kurtosis_sta_dict = kurtosis_dict[station]
            ax_waveform = axes[i_station]
            ax_kurtosis = ax_waveform.twinx()

            plot_station_kurtosis(ax_kurtosis, kurtosis_sta_dict)

    fig.suptitle(f"Kurtosis, Stacked Waveforms", fontsize=16, fontweight = "bold", y = 0.9)

    ### Save the figure
    save_figure(fig, "kurtosis_test_stack.png")

if __name__ == "__main__":
    main()