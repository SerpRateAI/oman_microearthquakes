"""
Compute 3-component average STA/LTA characteristic function.

Usage:
    python compute_sta_lta_3c.py
"""

# ---- Explicit imports ----
from numpy import zeros, float64, ndarray
from numba import njit, prange
from os.path import join
from time import time
from pandas import Timedelta, Timestamp, DataFrame
from typing import List, Tuple
from argparse import ArgumentParser
from utils_basic import get_geophone_days, get_freq_limits_string
from utils_basic import ROOTDIR_GEO as dirpath_data, GEO_COMPONENTS as components, GEO_STATIONS as stations, DETECTION_DIR as dirpath_detections, PICK_DIR as dirpath_pick, NETWORK as network
from utils_basic import geo_component2channel
from utils_cont_waveform import load_day_long_waveform_from_hdf, DayLongWaveform
from utils_sta_lta import Snippets, pick_triggers, merge_snippets, get_sta_lta_suffix
from utils_snuffler import write_time_windows

parser = ArgumentParser()
parser.add_argument("--station", type=str, required=True)
parser.add_argument("--sta_window_sec", type=float, default=0.01)
parser.add_argument("--lta_window_sec", type=float, default=0.2)
parser.add_argument("--thr_on", type=float, default=4.0)
parser.add_argument("--thr_off", type=float, default=1.0)
parser.add_argument("--buffer_start", type=float, default=0.01)
parser.add_argument("--buffer_end", type=float, default=0.005)
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=None)
parser.add_argument("--test", action="store_true", help="Test mode: only process one station and one day")
parser.add_argument("--write_snuffler_picks", action="store_true", help="Write the trigger windows to a Snuffler picker file")

args = parser.parse_args()
station = args.station
sta_window_sec = args.sta_window_sec
lta_window_sec = args.lta_window_sec
thr_on = args.thr_on
thr_off = args.thr_off
min_freq = args.min_freq_filter
max_freq = args.max_freq_filter
test = args.test
buffer_start = args.buffer_start
buffer_end = args.buffer_end
write_snuffler_picks = args.write_snuffler_picks

@njit
def _sta_lta_single(x, sta_n, lta_n, use_square):
    """
    Compute STA/LTA characteristic function for one 1-D trace.
    """
    n = x.size
    cf = zeros(n, dtype=float64)
    buf = zeros(lta_n, dtype=float64)
    sta_sum = 0.0
    lta_sum = 0.0

    for i in range(n):
        v = x[i]
        y = v * v if use_square else (v if v >= 0.0 else -v)

        idx = i % lta_n
        y_out_lta = buf[idx]
        buf[idx] = y

        lta_sum += y
        if i >= lta_n:
            lta_sum -= y_out_lta

        sta_sum += y
        if i >= sta_n:
            sta_sum -= buf[(i - sta_n) % lta_n]

        if i >= lta_n:
            sta = sta_sum / float(sta_n)
            lta = lta_sum / float(lta_n)
            if lta < 1e-16:
                lta = 1e-16
            cf[i] = sta / lta

    return cf

# ---- STA/LTA helper: single-component ----
def assemble_input_data(day_long_waveform: DayLongWaveform):
    """
    Assemble the input data for the STA/LTA characteristic function.
    """
    
    trace_mat = zeros((3, day_long_waveform.num_pts))
    for i, component in enumerate(components):
        waveform = day_long_waveform.get_component(component)
        trace_mat[i, :] = waveform

    print(f"Assembled the input data for the STA/LTA characteristic function")

    return trace_mat


# ---- 3-component average STA/LTA ----
@njit(parallel=True)
def compute_sta_lta_3c(trace_mat: ndarray, sampling_rate: float, sta_window_sec: float, lta_window_sec: float, use_square: bool = False):
    """
    Compute average STA/LTA CF for 3-component data.

    Parameters
    ----------
    trace_mat : ndarray, shape (3, n_samples)
        3-component input traces.
    sampling_rate : float
        Samples per second (Hz).
    sta_window_sec, lta_window_sec : float
        Short- and long-term window lengths in seconds.
    use_square : bool, optional
        Use squared amplitudes if True.

    Returns
    -------
    cf_avg : ndarray, shape (n_samples,)
        Average STA/LTA characteristic function across the 3 components.
    """
    n_traces, n_samples = trace_mat.shape
    if n_traces != 3:
        raise ValueError("Expected exactly 3 components (shape = (3, n_samples))")

    sta_n = int(round(sta_window_sec * sampling_rate))
    lta_n = int(round(lta_window_sec * sampling_rate))
    if sta_n <= 0 or lta_n <= 0 or sta_n >= lta_n:
        raise ValueError("Require 0 < STA < LTA (in samples)")

    cf_sum = zeros(n_samples, dtype=float64)
    for k in prange(3):
        cf_sum += _sta_lta_single(trace_mat[k], sta_n, lta_n, use_square)

    return cf_sum / 3.0

# Function to extract the snippets
def extract_snippets(triggers: List[Tuple[Timestamp, Timestamp]], day_long_waveform: DayLongWaveform, buffer_start: float, buffer_end: float):
    snippets = Snippets(day_long_waveform.station)

    for trigger in triggers:
        starttime = trigger[0]
        endtime = trigger[1]
        starttime = starttime - Timedelta(seconds=buffer_start)
        endtime = endtime + Timedelta(seconds=buffer_end)
        snippet = day_long_waveform.slice(starttime, endtime)
        snippets.append(snippet)

    return snippets

# Function to assemble the output
def assemble_csv_output(snippets: Snippets):
    output_dicts = []
    for snippet in snippets:
        id = snippet.id
        starttime = snippet.starttime
        endtime = snippet.get_endtime()
        output_dict = {"id": id, "starttime": starttime, "endtime": endtime}
        output_dicts.append(output_dict)

    output_df = DataFrame(output_dicts)
    return output_df

# ---- Example entry point ----
def main():
    """
    Example usage: load 3-component data, compute STA/LTA average, save result.
    """
    # Loop through the days
    freq_str = get_freq_limits_string(min_freq, max_freq)
    filename = f"preprocessed_data_{freq_str}.h5"
    filepath = join(dirpath_data, filename)

    print(f"Loading the day-long waveform for {station}")


    days = get_geophone_days(timestamp=False)
    if test:
        days = days[1:2]
    else:
        days = days

    trigger_windows_all = []
    snippets_all = []

    for day in days:
        print("--------------------------------")
        print(f"Processing {day}...")
        print("--------------------------------")
        print(f"STA window length: {sta_window_sec} seconds")
        print(f"LTA window length: {lta_window_sec} seconds")
        print("")

        clock1 = time()
        day_long_waveform = load_day_long_waveform_from_hdf(filepath, station, day)
        if day_long_waveform is None:
            print(f"No data found for station {station} on {day}")
            continue

        print(f"Loaded the day-long waveform for {day}")
        starttime = day_long_waveform.starttime
        sampling_rate = day_long_waveform.sampling_rate
        trace_mat = assemble_input_data(day_long_waveform)

        print("Computing the STA/LTA characteristic function...")
        cf = compute_sta_lta_3c(trace_mat, sampling_rate, sta_window_sec, lta_window_sec, use_square=False)

        # Find the trigger windows
        print(f"Picking the trigger windows")
        trigger_dict = pick_triggers(cf, on_thresh=thr_on, off_thresh=thr_off, sampling_rate=sampling_rate, starttime=starttime)
        trigger_windows = trigger_dict["event_windows"]
        if trigger_windows is None:
            print("Find no triggers. Skipping the day...")
            continue
        else:
            print(f"Detected {len(trigger_windows)} triggers.")

        # # Remove the hammer signals
        # print(f"Removing the hammer signals..")
        # trigger_windows = remove_hammer_signals(trigger_windows)
        # print(f"Number of trigger windows after removing the hammer signals: {len(trigger_windows)}")

        # Save the trigger windows
        print(f"Saving the trigger windows...")
        trigger_windows_all.extend(trigger_windows)

        # Extract the snippets and save
        snippets = extract_snippets(trigger_windows, day_long_waveform, buffer_start, buffer_end)
        snippets_all.append(snippets)

        clock2 = time()
        print(f"Time taken for processing {day}: {clock2 - clock1} seconds")

    # Merge the snippets
    print(f"Merging the snippets...")
    snippets_all = merge_snippets(snippets_all)

    # Set the sequential IDs for the snippets
    print(f"Setting the sequential IDs for the snippets...")
    snippets_all.set_sequential_ids()

    # Assemble the output
    print(f"Assembling the output")
    output_df = assemble_csv_output(snippets_all)

    # Write the trigger windows to a Snuffler picker file
    if write_snuffler_picks:
        print(f"Writing the trigger windows to a Snuffler picker file")
        suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
        filepath = join(dirpath_pick, f"sta_lta_detections_{freq_str}_{suffix}_{station}.txt")
        starttimes = [trigger_window[0] for trigger_window in trigger_windows_all]
        endtimes = [trigger_window[1] for trigger_window in trigger_windows_all]
        seed_ids = [f"{network}.{station}..BH1"] * len(starttimes)
        write_time_windows(filepath, starttimes, endtimes, seed_ids)
        print(f"Saved the trigger windows to {filepath}")

    # Save the time windows to a CSV file
    print(f"Saving the time windows to a CSV file")
    suffix = get_sta_lta_suffix(sta_window_sec, lta_window_sec, thr_on, thr_off)
    filepath = join(dirpath_detections, f"sta_lta_detections_{freq_str}_{suffix}_{station}.csv")
    output_df.to_csv(filepath, index=False)
    print(f"Saved the time windows to {filepath}")

    # Save the snippets to an HDF5 file

    print(f"Saving the snippets to an HDF5 file")
    filepath = join(dirpath_detections, f"snippets_sta_lta_{freq_str}_{suffix}_{station}.h5")
    snippets_all.to_hdf(filepath)
    print(f"Saved the snippets to {filepath}")

if __name__ == "__main__":
    main()