# Compute the Fourier coefficients of one harmonic series for all time windows using STFT
# Results are saved in the form of amplitudes and phases of the Fourier coefficients

# Imports
from os.path import join
from numpy import isnan, abs, angle
from pandas import read_csv, read_hdf, concat
from time import time

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations, GEO_COMPONENTS as components
from utils_basic import get_geophone_days, get_geo_metadata
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_torch import get_daily_geo_spectrograms

# Inputs
# Harmonic series information
base_name = "PR02549"
base_mode = 2

# STFT parameters
window_length = 60.0 # IN SECONDS
num_process_resample = 32 # Number of processes while resampling along the time axis in parallel

# Measuring peak properties
peak_bandwidth = 0.5 # Bandwidth for measuring the peak prominence

# Load the station metadata
metadata = get_geo_metadata()


# Get the geophone deployment days
days = get_geophone_days()

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
    
    filename = f"stationary_resonance_geo_summary_{mode_name}.h5"
    inpath = join(indir, filename)
    resonance_df = read_hdf(inpath, key="properties")
    resonance_df.reset_index(inplace = True)
    
    resonance_df["mode_name"] = mode_name
    resonance_dfs.append(resonance_df)

resonance_df = concat(resonance_dfs, ignore_index = True)

# Compute the frequency intervals
freq_interval = 1.0 / window_length

# Loop over stations
output_dicts = []
for station in stations:
    print("######")
    print(f"Processing {station}...")
    print("######")
    print("")
    
    # Loop over days
    num_days = len(days)
    for i, day in enumerate(days):
        # Start the clock
        clock1 = time()
        print(f"### Processing {day} for {station}... ###")

        # Read and preprocess the data
        stream_day = read_and_process_day_long_geo_waveforms(day, metadata, stations = station)
        if stream_day is None:
            print(f"{day} is skipped.")
            continue

        # Compute the spectrogram
        print("Computing the spectrograms...")
        stream_spec = get_daily_geo_spectrograms(stream_day, 
                                                  window_length = window_length, resample_in_parallel = True, num_process_resample = num_process_resample)
        
        # # Filter the resonance data for the day
        # print("Getting the harmonic peaks for the day...")
        # resonance_day_df = resonance_df[resonance_df["time"].dt.strftime("%Y-%m-%d") == day]

        # # Extract the properties of the harmonic peaks
        # print("Extracting the properties of the harmonic peaks...")
        # for _, row in resonance_day_df.iterrows():
        #     time_window = row["time"]
        #     freq = row["frequency"]
        #     mode_name = row["mode_name"]
            
        #     comp_dict = {}
        #     for component in components:
        #         trace_spec = stream_spec.select(components = component)[0]
        #         coeff, psd, prominence = trace_spec.get_peak_coeff_n_psd(time_window, freq, peak_bandwidth = peak_bandwidth)

        #         if coeff is None:
        #             break

        #         amplitude = abs(coeff)
        #         phase = angle(coeff, deg = True)
        #         comp_dict[component] = {"amplitude": amplitude, "phase": phase, "psd": psd, "prominence": prominence}

        #     if coeff is None:
        #         print(f"Skipping {time_window} for {station} on {day}.")
        #         continue

        #     output_dict = {"station": station, "time": time_window, "mode_name": mode_name, "frequency": freq}
        #     for component in components:
        #         output_dict[f"amplitude_{component.lower()}"] = comp_dict[component]["amplitude"]
        #         output_dict[f"phase_{component.lower()}"] = comp_dict[component]["phase"]
        #         output_dict[f"psd_{component.lower()}"] = comp_dict[component]["psd"]
        #         output_dict[f"prominence_{component.lower()}"] = comp_dict[component]["prominence"]

        #     output_dicts.append(output_dict)

        # Stop the clock
        clock2 = time()
        elapse = clock2 - clock1
        print("Done.")
        print(f"Elapsed time: {elapse} s")
        print("")

