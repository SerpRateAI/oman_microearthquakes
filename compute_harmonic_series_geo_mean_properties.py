# Compute the mean frequency of the stationary resonance for each time window on the geophone stations

# Imports
from os.path import join
from pandas import concat, read_csv, read_hdf

from utils_basic import GEO_STATIONS as stations, SPECTROGRAM_DIR as indir

# Inputs
base_name = "PR02549"
base_mode = 2

# Read the harmonic series information
print(f"Reading the information of the harmonic series with {base_name} as the base mode {base_mode}...")
filename = f"stationary_harmonic_series_{base_name}_base{base_mode}.csv"
inpath = join(indir, filename)
harmonic_series_df = read_csv(inpath)
print("")

# Compute the mean properties of each mode
for mode_name in harmonic_series_df["name"]:
    if mode_name.startswith("MH"):
        print(f"Skipping the missing harmonic {mode_name}...")
        continue

    print(f"Computing the mean properties of the stationary resonance {mode_name}...")

    # Read the data
    print(f"Reading the properties of {mode_name}...")
    filename = f"stationary_resonance_properties_{mode_name}_geo.h5"
    inpath = join(indir, filename)

    resonance_df = read_hdf(inpath, key="properties")
    print(f"{len(resonance_df)} rows read from {inpath}.")

    # Find the time windows with the required number of stations recording non-na values
    print(f"Finding the time windows when the frequencies at all stations are equal")
    filtered_resonance_df = resonance_df.groupby("time").filter(lambda x: x["frequency"].nunique() == 1)

    # Group filtered_resonance_df by time and compute the mean frequency of each group while ignoring NaNs
    print("Computing the source frequency for each time window...")
    source_freq_sr = filtered_resonance_df.groupby("time")["frequency"].first()
    source_freq_sr.name = "frequency"
    mean_power_sr = filtered_resonance_df.groupby("time")["power"].mean()
    mean_power_sr.name = "mean_power"
    mean_qf_sr = filtered_resonance_df.groupby("time")["quality_factor"].mean()
    mean_qf_sr.name = "mean_quality_factor"
    num_sta_sr = filtered_resonance_df.groupby("time")["station"].count()
    num_sta_sr.name = "num_stations"
    # Assemble source_freq_sr, mean_power_sr, and num_sta_sr into a data frame with time as the index
    result_df = concat([source_freq_sr, mean_power_sr, mean_qf_sr, num_sta_sr], axis=1)

    print(f"{len(result_df)} time windows with source frequency estimated.")

    # Save the results
    print("Saving the results to CSV format...")
    filename = f"stationary_resonance_geo_mean_properties_{mode_name}.csv"
    outpath = join(indir, filename)
    result_df.to_csv(outpath, index = True)
    print(f"Results saved to {outpath}.")

    print("Saving the results to HDF format...")
    filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
    outpath = join(indir, filename)
    result_df.to_hdf(outpath, key = "properties", mode = "w")
    print(f"Results saved to {outpath}.")