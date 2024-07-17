# Extract and plot the hydrophone and geophone spectra at a specific time
# All hydrophones are plotted, whereas only one geophone is plotted for reference

# Imports
from os.path import join
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from numpy import linspace

from utils_basic import SPECTROGRAM_DIR as indir, GEO_COMPONENTS as components, HYDRO_LOCATIONS as loc_dict, HYDRO_DEPTHS as depth_dict
from utils_basic import time2suffix
from utils_spec import get_spectrogram_file_suffix, read_time_from_geo_spectrograms, read_time_from_hydro_spectrograms, power2db
from utils_plot import save_figure, get_geo_component_color, component_to_label , format_depth_ylabels, format_freq_xlabels

# Inputs
# Data
time_to_plot = "2020-01-13T20:00:00"
geo_station_a = "A01"
geo_station_b = "B01"

window_length = 60.0
overlap = 0.0
downsample = False
downsample_factor = 60

major_noise_freqs = linspace(40.0, 200.0, 17)
minor_noise_freqs = linspace(60.0, 200.0, 141)
peak_freqs = [12.75, 25.5, 38.25]

# Plotting
hydro_base = -80.0
geo_base = -40.0

hydro_scale = 0.6
geo_scale = 0.01

linewidth_spec = 0.5
peak_linewidth = 1.0
major_noise_linewidth = 0.5
minor_noise_linewidth = 0.25

ymin_geo = -0.5
ymax_geo = 2.75

ymin_hydro = 0.0
ymax_hydro = 425.0   

min_freq = 0.0
max_freq = 50.0

axis_label_size = 12.0
tick_label_size = 10.0

component_label_x = 5.0
component_label_y = 0.1
component_label_size = 12.0

location_label_x = 5.0
location_label_y = 0.1
location_label_size = 12.0

freq_label_x = 1.0
freq_label_y = 0.1
freq_label_size = 10.0

station_label_x = 5.0
hydro_station_label_y = 0.2
geo_station_label_y = 0.2
station_label_size = 12.0

major_freq_spacing = 50.0
num_minor_freq_ticks = 5

major_depth_spacing = 100.0
num_minor_depth_ticks = 5


# Read the spectra at the specific time
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample, downsample_factor = downsample_factor)

print(f"Reading the hydrophone spectra at {time_to_plot}...")

filename = f"whole_deployment_daily_hydro_spectrograms_A00_{suffix_spec}.h5"
inpath = join(indir, filename)
hydro_dict_a = read_time_from_hydro_spectrograms(inpath, time_to_plot)

filename = f"whole_deployment_daily_hydro_spectrograms_B00_{suffix_spec}.h5"
inpath = join(indir, filename)
hydro_dict_b = read_time_from_hydro_spectrograms(inpath, time_to_plot)

print(f"Reading the geophone spectra at {time_to_plot}...")

filename = f"whole_deployment_daily_geo_spectrograms_{geo_station_a}_{suffix_spec}.h5"
inpath = join(indir, filename)
geo_dict_a = read_time_from_geo_spectrograms(inpath, time_to_plot)

filename = f"whole_deployment_daily_geo_spectrograms_{geo_station_b}_{suffix_spec}.h5"
inpath = join(indir, filename)
geo_dict_b = read_time_from_geo_spectrograms(inpath, time_to_plot)

# Plotting
# Generate the figure and axes
fig = figure(figsize = (12, 12))
gs = GridSpec(2, 2, height_ratios = [1, 2])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Plot the geophone spectra
# Station A
for i, component in enumerate(components):
    freqax = geo_dict_a["frequencies"]
    spec = power2db(geo_dict_a[component])
    spec_to_plot = (spec - geo_base) * geo_scale + i
    color = get_geo_component_color(component)
    ax1.plot(freqax, spec_to_plot, color = color, linewidth = linewidth_spec, zorder = 1)

    label = component_to_label(component)
    ax1.text(component_label_x, component_label_y + i, label, fontsize = component_label_size, fontweight = "bold", va = "top", ha = "left", bbox = dict(facecolor = "white", alpha = 1.0), zorder = 3)

for freq in peak_freqs:
    ax1.axvline(freq, color = "darkorange", linestyle = "--", linewidth = peak_linewidth, zorder = 0)
    ax1.text(freq + freq_label_x, ymin_geo + freq_label_y, f"{freq:.2f}", color = "darkorange", fontsize = freq_label_size, va = "bottom", ha = "left", zorder = 3)

ax1.set_title(geo_station_a, fontsize = station_label_size, fontweight = "bold")
ax1.set_xlim(min_freq, max_freq)
ax1.set_ylim(ymin_geo, ymax_geo)
ax1.set_yticks([])

format_freq_xlabels(ax1, 
                    label = False, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Station B
for i, component in enumerate(components):
    freqax = geo_dict_b["frequencies"]
    spec = power2db(geo_dict_b[component])
    spec_to_plot = (spec - geo_base) * geo_scale + i
    color = get_geo_component_color(component)
    ax2.plot(freqax, spec_to_plot, color = color, linewidth = linewidth_spec, zorder = 1)

    label = component_to_label(component)
    ax2.text(component_label_x, component_label_y + i, label, fontsize = component_label_size, fontweight = "bold", va = "top", ha = "left", bbox = dict(facecolor = "white", alpha = 1.0), zorder = 3)

for freq in peak_freqs:
    ax2.axvline(freq, color = "darkorange", linestyle = "--", linewidth = peak_linewidth, zorder = 2)

ax2.set_title(geo_station_b, fontsize = station_label_size, fontweight = "bold")
ax2.set_xlim(min_freq, max_freq)
ax2.set_ylim(ymin_geo, ymax_geo)
ax2.set_yticks([])

format_freq_xlabels(ax2, label = False, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Plot the hydrophone spectra
# Station A
locations_a = loc_dict["A00"]
for i, location in enumerate(locations_a):
    depth = depth_dict[location]
    freqax = hydro_dict_a["frequencies"]
    spec = power2db(hydro_dict_a[location])
    spec_to_plot = -(spec - hydro_base) * hydro_scale + depth
    ax3.plot(freqax, spec_to_plot, color = "darkviolet", linewidth = linewidth_spec, zorder = 1)

    ax3.text(location_label_x, location_label_y + depth, location, fontsize = location_label_size, fontweight = "bold", va = "top", ha = "left", bbox = dict(facecolor = "white", alpha = 1.0), zorder = 3)

for freq in minor_noise_freqs:
    ax3.axvline(freq, color = "lightgray", linestyle = "--", linewidth = minor_noise_linewidth, zorder = 0)

for freq in major_noise_freqs:
    ax3.axvline(freq, color = "darkgray", linestyle = "--", linewidth = major_noise_linewidth, zorder = 1)

for freq in peak_freqs:
    ax3.axvline(freq, color = "darkorange", linestyle = "--", linewidth = peak_linewidth, zorder = 2)

#ax3.text(station_label_x, ymin_hydro + hydro_station_label_y, "A00", fontsize = station_label_size, fontweight = "bold", va = "bottom", ha = "left", bbox = dict(facecolor = "white", alpha = 1.0), zorder = 3)
ax3.set_title("A00", fontsize = station_label_size, fontweight = "bold")

ax3.set_xlim(min_freq, max_freq)
ax3.set_ylim(ymin_hydro, ymax_hydro)
ax3.invert_yaxis()

format_freq_xlabels(ax3, label = True, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
format_depth_ylabels(ax3, label = True, major_tick_spacing = major_depth_spacing, num_minor_ticks = num_minor_depth_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Station B
locations_b = loc_dict["B00"]
for i, location in enumerate(locations_b):
    depth = depth_dict[location]
    freqax = hydro_dict_b["frequencies"]
    spec = power2db(hydro_dict_b[location])
    spec_to_plot = -(spec - hydro_base) * hydro_scale + depth
    ax4.plot(freqax, spec_to_plot, color = "darkviolet", linewidth = linewidth_spec, zorder = 1)

    ax4.text(location_label_x, location_label_y + depth, location, fontsize = location_label_size, fontweight = "bold", va = "top", ha = "left", bbox = dict(facecolor = "white", alpha = 1.0), zorder = 3) 


for freq in minor_noise_freqs:
    ax4.axvline(freq, color = "lightgray", linestyle = "--", linewidth = minor_noise_linewidth, zorder = 0)

for freq in major_noise_freqs:
    ax4.axvline(freq, color = "darkgray", linestyle = "--", linewidth = major_noise_linewidth, zorder = 1)

for freq in peak_freqs:
    ax4.axvline(freq, color = "darkorange", linestyle = "--", linewidth = peak_linewidth, zorder = 2)


#ax4.text(station_label_x, ymin_hydro + hydro_station_label_y, "B00", fontsize = station_label_size, fontweight = "bold", va = "bottom", ha = "left", bbox = dict(facecolor = "white", alpha = 1.0), zorder = 2)
ax4.set_title("B00", fontsize = station_label_size, fontweight = "bold")

ax4.set_xlim(min_freq, max_freq)
ax4.set_ylim(ymin_hydro, ymax_hydro)
ax4.invert_yaxis()

format_freq_xlabels(ax4, label = True, major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)
format_depth_ylabels(ax4, label = False, major_tick_spacing = major_depth_spacing, num_minor_ticks = num_minor_depth_ticks, axis_label_size = axis_label_size, tick_label_size = tick_label_size)

# Add the super title
fig.suptitle(f"{time_to_plot}", fontsize = 16.0, fontweight = "bold")

# Save the figure
fig.tight_layout()
suffix_time = time2suffix(time_to_plot)
figname = f"spec_time_slice_on_hydro_n_geo_{suffix_spec}_{suffix_time}.png"
save_figure(fig, figname)



