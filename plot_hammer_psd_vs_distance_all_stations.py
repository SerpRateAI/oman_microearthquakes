"""
Plot the PSD of hammer shots at a certian frequency recorded at a station vs distance to the shots.
"""

###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import sqrt, interp, log10, linspace, exp, pi    
from pandas import read_csv
from matplotlib.pyplot import subplots

from utils_basic import MT_DIR as dirpath_mt, LOC_DIR as dirpath_loc
from utils_basic import get_geophone_coords
from utils_plot import save_figure

parser = ArgumentParser(description = "Plot the PSD of hammer shots at a certian frequency recorded at a station vs distance to the shots.")

parser.add_argument("--freq_target", type = float, help = "The frequency of the PSD to plot", default = 150.0)
parser.add_argument("--marker_size", type = float, help = "The size of the markers", default = 50)
parser.add_argument("--quality_factor", type = float, help = "The quality factor for the refeence curve", default = 50.0)
parser.add_argument("--velocity", type = float, help = "The velocity of the wave in m/s", default = 500.0)

args = parser.parse_args()

freq_target = args.freq_target
marker_size = args.marker_size
quality_factor = args.quality_factor
velocity = args.velocity
###
# Load the data
###

# Load the station locations
station_df = get_geophone_coords()
# Load the hammer locations
filename = "hammer_locations.csv"
filepath = join(dirpath_loc, filename)
hammer_df = read_csv(filepath, dtype = {"hammer_id": str})

###
# Extract the PSD at the target frequency for each hammer

# Create the plot
fig, ax = subplots()
for station, row_sta in station_df.iterrows():
    print(f"Plotting the PSD of hammer shots at {freq_target:.0f} Hz recorded at {station}...")
    psds_target = []
    distances = []
    east_sta = station_df.loc[station, "east"]
    north_sta = station_df.loc[station, "north"]

    for _, row_hammer in hammer_df.iterrows():
        hammer_id = row_hammer["hammer_id"]
        east_hammer = row_hammer["east"]
        north_hammer = row_hammer["north"]

        # Load the MT auto-spectra
        filename = f"hammer_mt_aspecs_{hammer_id}_{station}.csv"
        filepath = join(dirpath_mt, filename)
        psd_df = read_csv(filepath)

        # Extract the PSD at the target frequency
        freqs = psd_df["frequency"]
        psds = psd_df["aspec_total"]

        # Interpolate the PSD at the target frequency
        psd = interp(freq_target, freqs, psds)

        distance = sqrt((east_hammer - east_sta)**2 + (north_hammer - north_sta)**2)

        psds_target.append(psd)
        distances.append(distance)

    if row_sta["subarray"] == "A":
        ax.scatter(distances, psds_target, s = marker_size, color = "blue", label = "Subarray A")
    else:
        ax.scatter(distances, psds_target, s = marker_size, color = "orange", label = "Subarray B")

ax.set_xlim(0, 210)
ax.set_ylim(-10, 110)

ax.set_xlabel("Distance to station (m)")
ax.set_ylabel(f"PSD (dB)")
ax.set_title(f"Hammer PSD at {freq_target:.0f} Hz vs distance to station")


###
# Plot the reference curves for geometric spreading

# Define the reference frequencies
distances_ref = linspace(10.0, 450.0, 100)

# Compute the reference curves
power_decay_ref1 = 110.0 - 10.0 * log10(distances_ref)
power_decay_ref2 = 110.0 - 10.0 * log10(distances_ref) - 20 * log10(exp(2 * pi * freq_target * distances_ref / quality_factor / velocity / 2.0))


ax.plot(distances_ref, power_decay_ref1, color = "black", linewidth = 1.5, linestyle = "--", label = "GS")
ax.plot(distances_ref, power_decay_ref2, color = "black", linewidth = 1.5, label = f"GS and Q = {quality_factor}")

# Add the legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), loc = "upper right")

###
# Save the plot
save_figure(fig, f"hammer_psd_vs_distance_{freq_target:.0f}hz_all_stations.png")