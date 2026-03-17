"""
This script computes the hammer power decay rate for each station.
"""
###
# Import the necessary libraries
###

from os.path import join
from argparse import ArgumentParser
from numpy import sqrt, interp, array, abs
from pandas import read_csv, DataFrame
from matplotlib.pyplot import subplots, close
from sklearn.linear_model import QuantileRegressor
from utils_basic import MT_DIR as dirpath_mt, LOC_DIR as dirpath_loc, MT_DIR as dirpath_mt, GEO_STATIONS as stations
from utils_basic import get_geophone_coords
from utils_plot import save_figure

parser = ArgumentParser(description = "Plot the PSD of hammer shots at a certian frequency recorded at a station vs distance to the shots.")

parser.add_argument("--freq_target", type = float, help = "The frequency to compute the power decay rate for")
args = parser.parse_args()

freq_target = args.freq_target

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

# Compute the power decay rate for each station
power_decay_rates = []
normalized_residuals = []
for station in stations:
    print(f"Computing the power decay rate for {station} at {freq_target:.0f} Hz...")
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

    # Fit a linear regression to the data
    x = array(distances).reshape(-1, 1)
    y_obs = array(psds_target)
    model = QuantileRegressor(quantile=0.5, alpha=0)
    model.fit(x, y_obs)

    y_pred = model.predict(x)
    residuals = y_obs - y_pred
    mean_norm_res = abs(residuals / y_obs).mean()

    power_decay_rates.append(model.coef_[0])
    normalized_residuals.append(mean_norm_res)
    print("Slope:", model.coef_[0])
    print("Mean normalized absolute residual:", mean_norm_res)

    print(f"The power decay rate for {station} at {freq_target:.0f} Hz is {model.coef_[0]:.2f} dB/m")


    # Plot the data and the regression line
    fig, ax = subplots(1, 1, figsize = (10, 10))
    ax.scatter(x, y_obs, label = "Observations", color = "skyblue", s = 100)
    ax.plot(x, y_pred, label = "Regression", color = "crimson", linewidth = 1.5)
    ax.set_xlabel("Distance (m)", fontsize = 12)
    ax.set_ylabel("PSD (dB)", fontsize = 12)
    ax.set_title(f"Power decay rate for {station} at {freq_target:.0f} Hz", fontsize = 14, fontweight = "bold")
    ax.legend(fontsize = 12)

    close()
    save_figure(fig, f"hammer_power_decay_rate_{station}_{freq_target:.0f}hz.png")

# Save the results
filename = f"hammer_power_decay_rate_{freq_target:.0f}hz.csv"
filepath = join(dirpath_mt, filename)
results_df = DataFrame({"station": stations, "power_decay_rate": power_decay_rates, "normalized_residual": normalized_residuals})
results_df.to_csv(filepath, index = False)
print(f"Saved the results to {filepath}")