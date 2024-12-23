# Compute the Fourier coefficients and the cross-component ratios for a given harmonic series

# Imports
from os.path import join
from numpy import isnan
from pandas import DataFrame
from pandas import concat, read_csv, read_hdf
from time import time
from multiprocessing import Pool

from utils_basic import GEO_STATIONS as stations, SPECTROGRAM_DIR as indir
from utils_basic import get_geophone_days, get_geo_metadata, timestamp2utcdatetime
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_spec import get_geo_3c_fft, get_start_n_end_from_center, get_stationary_resonance_cross_comp_pha_amp_diffs

# Inputs
base_name = "PR02549"
base_mode = 2

window_length = 60.0

num_processes = 32 # Number of processors to use for parallel processing

# Read the harmonic series information
print(f"Reading the information of the harmonic series with {base_name} as the base mode {base_mode}...")
filename = f"stationary_harmonic_series_{base_name}_base{base_mode}.csv"
inpath = join(indir, filename)
harmonic_series_df = read_csv(inpath)
print("")

print("Reading and concatenating the properties of each mode...")
resonance_dfs = []
for _, row in harmonic_series_df.iterrows():
    mode_name = row["name"]
    freq = row["observed_freq"]

    if isnan(freq):
        continue
    
    filename = f"stationary_resonance_properties_{mode_name}_geo.h5"
    inpath = join(indir, filename)
    resonance_df = read_hdf(inpath, key="properties")
    # print(resonance_df.columns)
    
    resonance_df["mode_name"] = mode_name
    resonance_dfs.append(resonance_df)

resonance_df = concat(resonance_dfs, ignore_index = True)


# Get the days of the geophone deployment
print("Getting the days of the geophone deployment...")
days = get_geophone_days()

# Get the metadata of the geophone stations
print("Getting the metadata of the geophone stations...")
metadata = get_geo_metadata()

# Compute the cross spectra for each station
diff_dicts = []
# stations = ["A01"]
for station in stations:
    clock1 = time()
    print(f"Processing {station}...")
    resonance_sta_df = resonance_df[resonance_df["station"] == station]

    for day in days:
        print(f"Processing {day}...")
        print("Reading the waveforms of the day...")
        stream = read_and_process_day_long_geo_waveforms(day, stations = station, metadata = metadata)

        if stream is None:
            continue

        print("Filtering the resonance data for the day...")
        resonance_day_df = resonance_sta_df[resonance_sta_df["time"].dt.strftime("%Y-%m-%d") == day]
        unique_times = resonance_day_df["time"].unique()
        print(f"{len(unique_times)} unique time windows to compute for {station} on {day}.")

        print("Slicing the waveform and resonance data for each unique time window...")
        args = []
        for centertime in unique_times:
            # print(type(centertime))
            starttime, endtime = get_start_n_end_from_center(centertime, window_length)
            starttime = timestamp2utcdatetime(starttime)
            endtime = timestamp2utcdatetime(endtime)
            stream_slice = stream.slice(starttime = starttime, endtime = endtime)
            resonance_slice_df = resonance_day_df[resonance_day_df["time"] == centertime]

            if resonance_slice_df.empty:
                continue

            if stream_slice is None:
                continue

            args.append((centertime, window_length, resonance_slice_df, stream_slice))

    
        print("Computing the phase and amplitude differences for each unique time window with mulitprocessing...")
        with Pool(processes=num_processes) as pool:                
                # Map the function directly to the centertimes with multiprocessing
                results = pool.starmap(get_stationary_resonance_cross_comp_pha_amp_diffs, args)

        print("Assembling the results...")
        for result in results:
            diff_dicts.extend(result)

    clock2 = time()
    print(f"Processing {station} took {clock2 - clock1:.2f} seconds.")
    print("")          

# Assemble the results into a data frame
print("Assembling the results into a data frame...")
diff_dicts = [diff_dict for diff_dict in diff_dicts if diff_dict is not None]
diff_df = DataFrame(diff_dicts)
diff_df["window_length"] = window_length
diff_df = diff_df[["mode_name", "station", "time", "window_length", "frequency",
                   "amplitude_z", "amplitude_1", "amplitude_2",
                    "phase_z", "phase_1", "phase_2",
                   "pha_diff_z_1", "pha_diff_z_2", "pha_diff_1_2",
                   "amp_rat_z_1", "amp_rat_z_2", "amp_rat_1_2",
                   "peak_power_z", "peak_power_1", "peak_power_2",
                   "peak_prom_z", "peak_prom_1", "peak_prom_2"]]

# Save the results for each mode separately
print("Saving the results to CSV format...")
for mode_name in harmonic_series_df["name"]:

    if mode_name.startswith("MH"):
        continue

    filename = f"stationary_resonance_phase_amplitude_diff_{mode_name}_geo.csv"
    outpath = join(indir, filename)

    diff_mode_df = diff_df[diff_df["mode_name"] == mode_name].copy()
    diff_mode_df.drop(columns = ["mode_name"], inplace = True)
    diff_mode_df.sort_values(by = ["station", "time"], inplace = True)
    diff_mode_df.to_csv(outpath, index = False)

    print(f"Results saved to {outpath}.")

print("Saving the results to HDF format...")
for mode_name in harmonic_series_df["name"]:

    if mode_name.startswith("MH"):
        continue

    filename = f"stationary_resonance_phase_amplitude_diff_{mode_name}_geo.h5"
    outpath = join(indir, filename)

    diff_mode_df = diff_df[diff_df["mode_name"] == mode_name].copy()
    diff_mode_df.drop(columns = ["mode_name"], inplace = True)
    diff_mode_df.sort_values(by = ["station", "time"], inplace = True)
    diff_mode_df.to_hdf(outpath, key = "properties", mode = "w")

    print(f"Results saved to {outpath}.")


