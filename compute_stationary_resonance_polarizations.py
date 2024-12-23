# Compute the polarization of a stationary resonance as function of time on all geophone stations 

### Import the required libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import exp, deg2rad
from multiprocessing import Pool
from pandas import DataFrame
from pandas import read_hdf

from utils_basic import SPECTROGRAM_DIR as indir, COMPONENT_PAIRS as comp_pairs
from utils_pol import get_pol_from_fourier_coeffs

# Define a wrapper function for parallel processing
def wrapper_get_pol_from_fourier_coeffs(in_dict):
    station = in_dict["station"]
    time = in_dict["time"]
    window_length = in_dict["window_length"]
    freq = in_dict["frequency"]
    amplitude_z = in_dict["amplitude_z"]
    phase_z = in_dict["phase_z"]
    amplitude_1 = in_dict["amplitude_1"]
    phase_1 = in_dict["phase_1"]
    amplitude_2 = in_dict["amplitude_2"]
    phase_2 = in_dict["phase_2"]
    peak_prom_z = in_dict["peak_prom_z"]
    peak_prom_1 = in_dict["peak_prom_1"]
    peak_prom_2 = in_dict["peak_prom_2"]
    peak_prom_threshold = in_dict["peak_prom_threshold"]

    if peak_prom_z < peak_prom_threshold or peak_prom_1 < peak_prom_threshold or peak_prom_2 < peak_prom_threshold:
        print(f"Peak prominence below threshold at station {station} and time {time.strftime('%Y-%m-%d %H:%M:%S')}. Skipping...")
        return None
    
    coeff_z = amplitude_z * exp(1j * deg2rad(phase_z))
    coeff_1 = amplitude_1 * exp(1j * deg2rad(phase_1))
    coeff_2 = amplitude_2 * exp(1j * deg2rad(phase_2))

    _, strike_major, dip_major, strike_minor, dip_minor, ellip = get_pol_from_fourier_coeffs(coeff_z, coeff_1, coeff_2)
    out_dict = {"station": station, "time": time, "window_length": window_length, "frequency": freq,
                "major_strike": strike_major, "major_dip": dip_major, "minor_strike": strike_minor, "minor_dip": dip_minor, "ellipticity": ellip}
    
    return out_dict


### Inputs ###
# Command line arguments
parser = ArgumentParser(description = "Compute the polarization of a stationary resonance as function of time on all geophone stations")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--peak_prom_threshold", type = float, default = 10.0, help = "Peak prominence threshold")
parser.add_argument("--num_processes", type = int, default = 1, help = "Number of processes to use for parallel processing")


# Parse the command line arguments
args = parser.parse_args()
mode_name = args.mode_name
peak_prom_threshold = args.peak_prom_threshold
num_processes = args.num_processes

print(f"Computing the polarization of {mode_name} at all stations as a function of time...")
print(f"Peak prominence threshold: {peak_prom_threshold}")

### Read the data ###
print(f"Reading the properties of {mode_name}...")
filename = f"stationary_resonance_phase_amplitude_diff_{mode_name}_geo.h5"
inpath = join(indir, filename)
resonance_df = read_hdf(inpath, key = "properties")

print(f"Assembling the input dictionaries for parallel processing...")
in_dicts = []
for i, row in resonance_df.iterrows():
    station = row["station"]
    time = row["time"]
    window_length = row["window_length"]
    freq = row["frequency"]
    amplitude_z = row["amplitude_z"]
    phase_z = row["phase_z"]
    amplitude_1 = row["amplitude_1"]
    phase_1 = row["phase_1"]
    amplitude_2 = row["amplitude_2"]
    phase_2 = row["phase_2"]
    peak_prom_z = row["peak_prom_z"]
    peak_prom_1 = row["peak_prom_1"]
    peak_prom_2 = row["peak_prom_2"]

    in_dict = {"station": station, "time": time, "window_length": window_length, "frequency": freq,
                "amplitude_z": amplitude_z, "phase_z": phase_z, "amplitude_1": amplitude_1, "phase_1": phase_1, "amplitude_2": amplitude_2, "phase_2": phase_2,
                "peak_prom_z": peak_prom_z, "peak_prom_1": peak_prom_1, "peak_prom_2": peak_prom_2, "peak_prom_threshold": peak_prom_threshold}
    in_dicts.append(in_dict)

### Compute the polarization in parallel ###
print(f"Computing the polarization using {num_processes} processes...")
if num_processes > 1:
    with Pool(num_processes) as pool:
        out_dicts = pool.map(wrapper_get_pol_from_fourier_coeffs, in_dicts)
else:
    out_dicts = []
    for in_dict in in_dicts:
        out_dict = wrapper_get_pol_from_fourier_coeffs(in_dict)
        out_dicts.append(out_dict)

out_dicts = [out_dict for out_dict in out_dicts if out_dict is not None]
print("Done!")

# Convert the results to a DataFrame
out_df = DataFrame(out_dicts)

### Save the results ###
print("Saving the results to HDF5 format...")
filename = f"stationary_resonance_polarization_{mode_name}.h5"
outpath = join(indir, filename)
out_df.to_hdf(outpath, key = "properties", mode = "w")
print(f"Results saved to {outpath}")

print("Save the results to CSV format...")
filename = f"stationary_resonance_polarization_{mode_name}.csv"
outpath = join(indir, filename)
out_df.to_csv(outpath, index = True)
print(f"Results saved to {outpath}")

