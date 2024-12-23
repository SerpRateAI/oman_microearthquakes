# Plot the geophone and hydrophone spectra around the stationary resonance peak frequency for a time window

# Import the necessary libraries
from os.path import join
from argparse import ArgumentParser
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir, HYDRO_LOCATIONS as hydro_loc_dict, GEO_COMPONENTS as components
from utils_basic import power2db, get_geophone_coords, str2timestamp, time2suffix
from utils_spec import get_spectrogram_file_suffix, read_time_slice_from_hydro_power_spectrograms, read_time_slice_from_geo_power_spectrograms
from utils_plot import HYDRO_COLOR as hydro_color
from utils_plot import format_db_ylabels, format_freq_xlabels, get_geo_component_color, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description = "Plot the geophone and hydrophone spectra around the stationary resonance peak frequency for a time window")
parser.add_argument("--station_hydro", type = str, help = "Hydrophone station to plot")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--window_length", type = float, help = "Length of the window in seconds")
parser.add_argument("--overlap", type = float, help = "Overlap fraction")
parser.add_argument("--center_time", type = str, help = "Center time of the time window")
parser.add_argument("--freq_window_width", type = float, help = "Width of the frequency window in Hz around the geophone peak frequency", default = 1.0)
parser.add_argument("--min_db_geo", type = float, help = "Minimum power level for the geophone spectrogram in dB", default = -20.0)
parser.add_argument("--max_db_geo", type = float, help = "Maximum power level for the geophone spectrogram in dB", default = 10.0)
parser.add_argument("--min_db_hydro", type = float, help = "Minimum power level for the hydrophone spectrogram in dB", default = -100.0)
parser.add_argument("--max_db_hydro", type = float, help = "Maximum power level for the hydrophone spectrogram in dB", default = -60.0)

# Parse the command line arguments
args = parser.parse_args()
station_hydro = args.station_hydro
mode_name = args.mode_name
window_length = args.window_length
overlap = args.overlap
center_time = str2timestamp(args.center_time)
freq_window_width = args.freq_window_width
min_db_geo = args.min_db_geo
max_db_geo = args.max_db_geo
min_db_hydro = args.min_db_hydro
max_db_hydro = args.max_db_hydro


print(f"Hydrophone station: {station_hydro}")
print(f"Mode name: {mode_name}")
print(f"Center time: {center_time}")
print(f"Window length: {window_length}")
print(f"Overlap: {overlap}")
print(f"Frequency window width: {freq_window_width}")

if station_hydro == "A00":
    station_geo = "A01"
elif station_hydro == "B00":
    station_geo = "B01"
else:
    raise ValueError("The hydrophone station is not recognized!")

# Constants
panel_width = 10.0
panel_height = 3.0
linewidth = 1.5
x_station_label = 0.01
y_station_label = 0.97
station_label_size = 14
y_suptitle = 0.90

### Read the data ###
print("Reading the peak frequency of the stationary resonance for the time window...")
filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
inpath = join(indir, filename)
geo_properties_df = read_hdf(inpath, key = "properties")
freq_reson = geo_properties_df.loc[center_time, "frequency"]
print(f"The peak frequency of the stationary resonance is {freq_reson:.3f} Hz.")

print("Reading the geophone spectrograms...")
suffix = get_spectrogram_file_suffix(window_length, overlap)
filename = f"whole_deployment_daily_geo_spectrograms_{station_geo}_{suffix}.h5"
inpath = join(indir, filename)
power_geo_dict = read_time_slice_from_geo_power_spectrograms(inpath, center_time)

print("Reading the hydrophone spectrograms...")
filename = f"whole_deployment_daily_hydro_power_spectrograms_{station_hydro}_{suffix}.h5"
inpath = join(indir, filename)
power_hydro_dict = read_time_slice_from_hydro_power_spectrograms(inpath, center_time)

### Plotting ###
print("Plotting the geophone and hydrophone spectra...")
locations = hydro_loc_dict[station_hydro]
num_loc = len(locations)

fig, axes = subplots(nrows = num_loc + 1, ncols = 1, figsize = (panel_width, panel_height * (num_loc + 1)), sharex = True)

# Plot the geophone spectra
print("Plotting the 3C geophone spectra...")
ax = axes[0]
for component in components:
    freqax = power_geo_dict["freqs"]
    power_spec = power2db(power_geo_dict[component])
    color = get_geo_component_color(component)

    ax.plot(freqax, power_spec, label = component, linewidth = linewidth)
    ax.set_ylim(min_db_geo, max_db_geo)

    format_db_ylabels(ax,
                     major_tick_spacing = 20.0, num_minor_ticks = 10)

ax.text(x_station_label, y_station_label, station_geo, transform = ax.transAxes, fontsize = station_label_size, fontweight = "bold", va = "top", ha = "left")
ax.axvline(x = freq_reson, color = "crimson", linestyle = "--", linewidth = linewidth)

# Plot the hydrophone spectra
print("Plotting the hydrophone spectra of all locations...")
for i, location in enumerate(locations):
    ax = axes[i + 1]
    freqax = power_hydro_dict["freqs"]
    power_spec = power2db(power_hydro_dict[location])

    ax.plot(freqax, power_spec, label = location, color = hydro_color, linewidth = linewidth)
    ax.set_ylim(min_db_hydro, max_db_hydro)

    format_db_ylabels(ax,
                      major_tick_spacing = 20.0, num_minor_ticks = 10)

    ax.axvline(x = freq_reson, color = "crimson", linestyle = "--", linewidth = linewidth)
    
    ax.text(x_station_label, y_station_label, f"{station_hydro}.{location}", transform = ax.transAxes, fontsize = station_label_size, fontweight = "bold", va = "top", ha = "left")

    if i == num_loc - 1:
        min_freq = freq_reson - freq_window_width / 2
        max_freq = freq_reson + freq_window_width / 2
        ax.set_xlim(min_freq, max_freq)

# Format the x-axis
format_freq_xlabels(axes[-1],
                    major_tick_spacing = 0.1, num_minor_ticks = 5)

# Set the title
title = f"{mode_name}, {station_hydro}, {center_time.strftime('%Y-%m-%d %H:%M:%S')}, {window_length:.1f} s"
fig.suptitle(title, fontsize = 14, fontweight = "bold", y = y_suptitle)

# Save the figure
print("Saving the figure...")
time_suffix = time2suffix(center_time)
figname = f"stationary_resonance_geo_n_hydro_spectra_{mode_name}_{station_geo}_{station_hydro}_{time_suffix}.png"
save_figure(fig, figname)




