# Find the hydrophone windows with high stationary resonance power in the time period that overlaps with geophones

# Imports
from os.path import join
from argparse import ArgumentParser
from numpy import isnan
from scipy.signal import find_peaks
from pandas import DataFrame
from pandas import read_hdf
from json import dumps

from utils_basic import SPECTROGRAM_DIR as indir, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo
from utils_spec import get_spectrogram_file_suffix, read_hydro_power_spectrograms

# Function for processing each time window
def process_time_window(time_window, freq_peak_geo, freqax, power_spec_dict, prom_threshold, max_mean_power, freq_pts_threshold = 1):
    print("Geophone peak frequency: ", freq_peak_geo)
    locations_out = []
    for location, power_spec in power_spec_dict.items():
        # Deterimine if the mean power is above the threshold
        mean_power = power_spec.mean()

        if mean_power > max_mean_power or any(isnan(power_spec)):
            continue
        
        # Find the peaks on the hydrophone spectrum
        inds_peak, _ = find_peaks(power_spec, prominence = prom_threshold)
        freqs_peak_hydro = freqax[inds_peak]
        freq_interval = freqax[1] - freqax[0]

        if len(freqs_peak_hydro) == 0:
            continue

        # Get the peak frequencies closest to the geophone peak frequency
        freq_diffs = abs(freqs_peak_hydro - freq_peak_geo)
        i_peak = freq_diffs.argmin()
        min_freq_diff = freq_diffs[i_peak]

        if min_freq_diff <= freq_pts_threshold * freq_interval:
            locations_out.append(location)

    if len(locations_out) > 1:
        result_dict = {"time": time_window, "locations": locations_out}
        return result_dict
    else:
        return None

# Inputs
# Command line arguments
parser = ArgumentParser(description = "Find the hydrophone windows with high stationary resonance power in the time period that overlaps with geophones")

parser.add_argument("--station", type = str, help = "Hydrohopne station to process")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--window_length", type = float, help = "Length of the window in seconds")
parser.add_argument("--overlap", type = float, help = "Overlap fraction")
parser.add_argument("--prom_threshold", type = float, help = "Prominence threshold for peak detection on the hydrophone spectrograms")
parser.add_argument("--freq_pts_threshold", type = int, default = 1, help = "Number of frequency points around the geophone peak frequency")
parser.add_argument("--freq_window_width", type = float, help = "Width of the frequency window in Hz around the geophone peak frequency")
parser.add_argument("--max_mean_power", type = float, help = "Maximum mean power in dB for excluding noisy hydrophone windows")
parser.add_argument("--num_processes", type = int, help = "Number of processes")

# Parse the command line arguments
args = parser.parse_args()
station = args.station
mode_name = args.mode_name
window_length = args.window_length
overlap = args.overlap
prom_threshold = args.prom_threshold
freq_pts_threshold = args.freq_pts_threshold
max_mean_power = args.max_mean_power
freq_window_width = args.freq_window_width
num_processes = args.num_processes

# Print the arguments
print(f"Station: {station}")
print(f"Mode name: {mode_name}")
print(f"Window length: {window_length}")
print(f"Overlap: {overlap}")
print(f"Prominence threshold: {prom_threshold}")
print(f"Number of frequency points around the geophone peak frequency: {freq_pts_threshold}")
print(f"Frequency window width: {freq_window_width}")
print(f"Maximum mean power: {max_mean_power}")
print(f"Number of processes: {num_processes}")

# Read the geophone mean peak properties
print("Reading the geophone mean peak properties...")
filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
inpath = join(indir, filename)
geo_properties_df = read_hdf(inpath, key = "properties")
mean_freq_whole = geo_properties_df["frequency"].mean()

# Read the hydrophone spectrograms in the frequcency window around the geophone peak frequency
print("Reading the hydrophone spectrograms in the frequency window around the geophone peak frequency...")
min_freq = mean_freq_whole - freq_window_width / 2
max_freq = mean_freq_whole + freq_window_width / 2

suffix = get_spectrogram_file_suffix(window_length, overlap)
filename = f"whole_deployment_daily_hydro_power_spectrograms_{station}_{suffix}.h5"
inpath = join(indir, filename)
stream_spec = read_hydro_power_spectrograms(inpath, starttime = starttime_geo, endtime = endtime_geo, 
                                             min_freq = min_freq, max_freq = max_freq)
stream_spec.to_db()

# Process each time window
print("Processing each time window...")
locations = stream_spec.get_locations()

# Process each time window in serial
result_dicts = []
if num_processes == 1:
    for time_window in geo_properties_df.index:
        print(f"Processing time window {time_window.strftime('%Y-%m-%d %H:%M:%S')}...")
        freq_peak_geo = geo_properties_df.loc[time_window, "frequency"]
        
        # Get the power spectrograms of every hydrophone locations
        power_spec_dict = {}
        for location in locations:
            trace_spec = stream_spec.select(locations = location)[0]
            psd_mat = trace_spec.data
            freqax = trace_spec.freqs
            timeax = trace_spec.times

            psd_spec = psd_mat[:, timeax == time_window].flatten()
            power_spec_dict[location] = psd_spec

        # Get the locations with high power peaks
        result_dict = process_time_window(time_window, freq_peak_geo, freqax, power_spec_dict, prom_threshold, max_mean_power)
        if result_dict is not None:
            result_dicts.append(result_dict)

num_windows = len(result_dicts)
print(f"Number of windows with more than one locations having high prominance peaks: {num_windows}")

if num_windows == 0:
    print("No windows with more than one locations having high prominance peaks found!")
    exit()

# Convert the result_dicts to a DataFrame
result_df = DataFrame(result_dicts)

# Add the column with the number of locations at each time window
result_df["num_locations"] = result_df["locations"].apply(lambda x: len(x))
result_df = result_df[["time", "num_locations", "locations"]]
result_df.sort_values(by = ["num_locations", "time"], ascending = [False, True], inplace = True)
result_df.reset_index(drop = True, inplace = True)

# Save the result to a CSV file
result_df["locations"] = result_df["locations"].apply(dumps)

filename = f"stationary_resonance_hydro_windows_{station}_{mode_name}.csv"
outpath = join(indir, filename)
result_df.to_csv(outpath, index=True)
print(f"Result saved to {outpath}!")

# Save the result to an HDF5 file
filename = f"stationary_resonance_hydro_windows_{station}_{mode_name}.h5"
outpath = join(indir, filename)
result_df.to_hdf(outpath, key = "windows")
print(f"Result saved to {outpath}!")




