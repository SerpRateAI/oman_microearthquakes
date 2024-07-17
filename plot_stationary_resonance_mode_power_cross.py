# Plot the power correlation between two stationary-resonance modes recorded on all the geophone stations

# Imports
from os.path import join
from numpy import column_stack, corrcoef, isnan, mean, std
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_plot import add_colorbar, save_figure

# Inputs
# Data
name1 = "SR25a"
name2 = "SR76a"

min_num_sta = 9

# Plotting
num_row = 6
num_col = 6

power_window = 40.0

label_size = 12

label_x = 0.01
label_y = 0.99

xmin_norm = -3.0
xmax_norm = 3.0

ymin_norm = -3.0
ymax_norm = 3.0

cmap_name = "viridis"
cmin = 0.2
cmax = 1.0

title_size = 15

cbar_width = 0.01
cbar_offset = 0.02

# Read the data
filename1 = f"stationary_resonance_properties_{name1}_geo.h5"
filename2 = f"stationary_resonance_properties_{name2}_geo.h5"

inpath1 = join(indir, filename1)
inpath2 = join(indir, filename2)

resonance_df1 = read_hdf(inpath1, key="properties")
resonance_df2 = read_hdf(inpath2, key="properties")

sta_raw_dict = {}
sta_norm_dict = {}
sta_corr_dict = {}
for station in stations:
    print(f"Extracting the power for {name1} and {name2} at {station}...")
    resonance_sta_df1 = resonance_df1.loc[resonance_df1["station"] == station]
    resonance_sta_df2 = resonance_df2.loc[resonance_df2["station"] == station]

    resonance_sta_df1 = resonance_sta_df1.reindex(resonance_sta_df2.index, method="nearest", limit=1)

    power1_in = resonance_sta_df1["power"].values
    power2_in = resonance_sta_df2["power"].values

    # Remove the NaN values
    power1_raw = power1_in[~isnan(power1_in) & ~isnan(power2_in)]
    power2_raw = power2_in[~isnan(power1_in) & ~isnan(power2_in)]

    # Compute the correlation coefficient
    power_corr = corrcoef(power1_raw, power2_raw)[0, 1]
    sta_corr_dict[station] = power_corr

    # Remove the mean and divide by the standard deviation
    power1_norm = (power1_raw - mean(power1_raw)) / std(power1_raw)
    power2_norm = (power2_raw - mean(power2_raw)) / std(power2_raw)

    sta_raw_dict[station] = column_stack((power1_raw, power2_raw))
    sta_norm_dict[station] = column_stack((power1_norm, power2_norm))

# Plot the raw power distribution
if num_row * num_col != len(stations):
    raise ValueError("The number of rows and columns must match the number of stations!")

fig, axes = subplots(num_row, num_col, figsize=(20, 20))
for i, (station, power_corr) in enumerate(sta_raw_dict.items()):
    print(f"Plotting the power correlation for {station}...")
    ax = axes.flatten()[i]
    ax.scatter(power_corr[:, 0], power_corr[:, 1], s=10, c='black', alpha=0.2)

    power1_mean = mean(power_corr[:, 0])
    power2_mean = mean(power_corr[:, 1])

    xmin = power1_mean - power_window / 2
    xmax = power1_mean + power_window / 2

    ymin = power2_mean - power_window / 2
    ymax = power2_mean + power_window / 2

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    ax.text(label_x, label_y, station, fontsize=label_size, fontweight = "bold", ha='left', va='top', transform=ax.transAxes)

    # Plot the reference line with a slope of 1
    ax.plot([xmin, xmax], [ymin, ymax], color = "crimson", linestyle = "--", linewidth = 1.0)

    if i % num_col == 0:
        ax.set_ylabel(f"{name2} power (dB)", fontsize=label_size)

    if i >= num_col * (num_row - 1):
        ax.set_xlabel(f"{name1} power (dB)", fontsize=label_size)

    ax.set_aspect('equal', 'box')

fig.suptitle(f"Cross plots of the observed power of {name1} and {name2} on all geophone stations", fontsize=title_size, fontweight='bold', y=0.9)

# Save the plot
figname = f"stationary_resonance_power_plots_raw_geo_{name1}_{name2}.png"
save_figure(fig, figname)

# Plot the normalized power distribution
fig, axes = subplots(num_row, num_col, figsize=(20, 20))

for i, (station, power_corr) in enumerate(sta_norm_dict.items()):
    print(f"Plotting the normalized power correlation for {station}...")
    corr = sta_corr_dict[station]
    ax = axes.flatten()[i]
    ax.set_facecolor("lightgray")
    mappable = ax.scatter(power_corr[:, 0], power_corr[:, 1], s = 10, c = [corr] * power_corr.shape[0], cmap = cmap_name, vmin = cmin, vmax = cmax)

    ax.set_xlim(xmin_norm, xmax_norm)
    ax.set_ylim(xmin_norm, xmax_norm)
    
    ax.text(label_x, label_y, station, fontsize=label_size, fontweight = "bold", ha='left', va='top', transform=ax.transAxes)

    # ax.plot([xmin, xmax], [ymin, ymax], color = "crimson", linestyle = "--", linewidth = 1.0)

    if i % num_col == 0:
        ax.set_ylabel(f"{name2} norm. power", fontsize=label_size)

    if i >= num_col * (num_row - 1):
        ax.set_xlabel(f"{name1} norm. power", fontsize=label_size)

    ax.set_aspect('equal', 'box')

# Add a colorbar to the right of the plot
ax_top = axes[0, 0]
ax_bottom = axes[num_row - 1, num_col - 1]

bbox_top = ax_top.get_position()
bbox_bottom = ax_bottom.get_position()

cbar_left = bbox_bottom.x1 + cbar_offset
cbar_bottom = bbox_bottom.y0
cbar_height = bbox_top.y1 - bbox_bottom.y0
cbar_position = [cbar_left, cbar_bottom, cbar_width, cbar_height]

cbar = add_colorbar(fig, mappable, "Correlation coefficient", cbar_position, orientation = "vertical", major_tick_spacing = 0.2)

fig.suptitle(f"Cross plots of the normalized power of {name1} and {name2} on all geophone stations", fontsize=title_size, fontweight='bold', y=0.9)

# Save the plot
figname = f"stationary_resonance_power_plots_norm_geo_{name1}_{name2}.png"
save_figure(fig, figname)