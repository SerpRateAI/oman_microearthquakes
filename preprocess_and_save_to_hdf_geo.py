"""
Preprocess all geophone MSEED files and save them to an HDF5 file
A highpass Butterworth filter will be applied to the data
"""

#------
# Import
#------
from pathlib import Path
from argparse import ArgumentParser
from time import time
from h5py import File
from obspy import Trace

from utils_basic import GEO_STATIONS as stations, GEO_COMPONENTS as components, ROOTDIR_GEO as dirpath
from utils_basic import get_geophone_days
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_cont_waveform import save_day_long_trace_to_hdf

#------
# Main
#------

if __name__ == "__main__":
    # Get the command line arguments
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--days", type=str, nargs="+", default=get_geophone_days())
    parser.add_argument("--day_test", type=str, default="2020-01-12")
    parser.add_argument("--min_freq", type=float, default=20.0)
    parser.add_argument("--max_freq", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    test = args.test
    day_test = args.day_test
    days = args.days
    min_freq = args.min_freq
    max_freq = args.max_freq
    overwrite = args.overwrite

    # If test is True, only process the test day
    if test:
        days = [day_test]
        print(f"Running in test mode for day {day_test} only...")
    else:
        print(f"Running for days: {days}")

    # Print the corner frequency
    print(f"Corner frequency for highpass filter: {min_freq} Hz")
    if max_freq is not None:
        print(f"Corner frequency for lowpass filter: {max_freq} Hz")

    # Create the HDF5 file
    if max_freq is not None:
        hdf5_path = Path(dirpath) / f"preprocessed_data_min{min_freq:.0f}hz_max{max_freq:.0f}hz.h5"
    else:
        hdf5_path = Path(dirpath) / f"preprocessed_data_min{min_freq:.0f}hz.h5"

    # Preprocess the data and save it to HDF5
    for day in days:
        print("-"*100)
        print(f"Processing day {day}...")
        print("-"*100)

        for station in stations:
            print("-"*100)
            print(f"Processing station {station}...")
            print("-"*100)

            # Read the data and preprocess it
            print(f"Reading and preprocessing data for station {station}...")
            start_time = time()
            stream = read_and_process_day_long_geo_waveforms(day, 
                                                             stations = [station], 
                                                             filter = True, 
                                                             filter_type = "butter",
                                                             min_freq = min_freq,
                                                             max_freq = max_freq)
            end_time = time()
            print(f"Time taken: {end_time - start_time} seconds")

            if stream is None:
                print(f"No data found for station {station} on {day}!")
                continue

            # Save the processed stream 
            print(f"Saving the results...")
            start_time = time()
            for component in components:
                trace = stream.select(component = component)[0]
                save_day_long_trace_to_hdf(trace, hdf5_path, overwrite=overwrite)
            end_time = time()
            print(f"Time taken: {end_time - start_time} seconds")

            print(f"Done processing station {station}...")
            print("")

        print(f"Done processing day {day}...")
        print("")

