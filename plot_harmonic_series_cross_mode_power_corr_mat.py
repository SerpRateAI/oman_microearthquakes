# Plot the power correlation matrix for the stationary resonances with harmonic relations at all geophone stations

# Imports
from os.path import join
from numpy import column_stack, corrcoef, isnan, mean, nan, std, zeros
from pandas import DataFrame
from pandas import read_csv, read_hdf
from matplotlib.pyplot import subplots, get_cmap

from utils_basic import SPECTROGRAM_DIR as indir, GEO_STATIONS as stations
from utils_plot import add_colorbar, save_figure

# Inputs
# Data
base_name = "PR02549"
base_number = 2

cmap_name = "viridis"
cmin = 0.2
cmax = 1.0

cbar_width = 0.01
cbar_offset = 0.02

# Plotting
dim = 4

num_col = 6
num_row = 6

title_size = 12

# Read the list of stationary resonances
filename_harmo = f"stationary_harmonic_series_{base_name}_base{base_number}.csv"
inpath = join(indir, filename_harmo)
harmonic_series_df = read_csv(inpath)
harmonic_series_df = harmonic_series_df.loc[harmonic_series_df["detected"]]

# Read the stationary resonance properties
mode_dict = {}
for mode_name, row in harmonic_series_df.iterrows():
    # Read the data
    filename = f"stationary_resonance_properties_{mode_name}_geo.h5"
    inpath = join(indir, filename)

    property_df = read_hdf(inpath, key="properties")
    mode_dict[mode_name] = property_df

# Make the plot for each station
cmap = get_cmap(cmap_name)
cmap.set_bad('lightgray')

if num_col * num_row != len(stations):
    raise ValueError("Number of columns and rows do not match the number of stations!")

fig, axes = subplots(num_row, num_col, figsize=(dim * num_col, dim * num_row))
mean_corrs = []
for sta_ind, station in enumerate(stations):
    print(f"Working on {station}...")
    num_modes = len(mode_dict)
    corr_mat = zeros((num_modes, num_modes))
    ax = axes.flatten()[sta_ind]

    print("Computing the power correlation matrix...")
    mean_corr = 0.0
    for i, (name1, df1) in enumerate(mode_dict.items()):
        for j, (name2, df2) in enumerate(mode_dict.items()):
            df1 = df1.loc[df1["station"] == station]
            df2 = df2.loc[df2["station"] == station]

            df2 = df2.reindex(df1.index, method = "nearest", limit = 1)

            power1_raw = df1["power"].values
            power2_raw = df2["power"].values

            power1 = power1_raw[~isnan(power1_raw) & ~isnan(power2_raw)]
            power2 = power2_raw[~isnan(power1_raw) & ~isnan(power2_raw)]

            if len(power1) == 0:
                print(f"Time series length is 0 for {name1} and {name2} at {station}! Skipped.")
                corr_mat[i, j] = nan
                continue

            power_corr = corrcoef(power1, power2)
            corr_mat[i, j] = power_corr[0, 1]

            if i != j:
                mean_corr += power_corr[0, 1]

    mean_corr /= num_modes * (num_modes - 1)
    mean_corrs.append(mean_corr)
    print(f"Mean correlation: {mean_corr:.2f}")

    print("Plotting the power correlation matrix...")
    mappable = ax.matshow(corr_mat, cmap=cmap, vmin=cmin, vmax=cmax)
    ax.set_aspect('equal')
    ax.set_title(f"{station}", fontsize=title_size, fontweight='bold', pad=5)

# Add colorbar
ax = axes.flatten()[-1]
bbox = ax.get_position()
cbar_x = bbox.x1 + cbar_offset
cbar_y = bbox.y0
cbar_width = cbar_width
cbar_height = bbox.y1 - bbox.y0
position = (cbar_x, cbar_y, cbar_width, cbar_height)

cbar = add_colorbar(fig, position, "CC coefficient", 
                    mappable = mappable,
                    major_tick_spacing = 0.2)

print("Saving the figure...")
filename = f"harmonic_resonance_power_corr_mat_{base_name}.png"
save_figure(fig, filename)

# Construct and save the mean correlation values
mean_corr_df = DataFrame(mean_corrs, index=stations, columns=["mean_corr"])
outdir = indir
outpath = join(outdir, f"harmonic_resonance_geo_power_mean_corr_{base_name}.csv")
mean_corr_df.to_csv(outpath)
print(f"Mean correlations saved to {outpath}")




