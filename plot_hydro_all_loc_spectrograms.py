# Plot the hydrophone spectrograms of all locations in a borehole

# Imports
from os.path import join
from matplotlib.pyplot import get_cmap, subplots

from utils_basic import HYDRO_LOCATIONS as loc_dict, STARTTIME_GEO as starttime_geo, ENDTIME_GEO as endtime_geo, STARTTIME_HYDRO as starttime_hydro, ENDTIME_HYDRO as endtime_hydro, SPECTROGRAM_DIR as indir
from utils_basic import time2suffix
from utils_spec import read_hydro_spectrograms, get_spectrogram_file_suffix
from utils_plot import add_colorbar, format_datetime_xlabels, format_freq_ylabels, save_figure


# Inputs
# Data
station = "A00"

window_length = 300.0
overlap = 0.0
downsample = False
downsample_factor = 60
prom_threshold = 10
rbw_threshold = 3.0

starttime = starttime_geo
endtime = endtime_geo

min_freq = 0.0
max_freq = 5.0

# Plotting
row_height = 2
width = 10

min_db = -60.0
max_db = -40.0

loc_label_x = 0.02
loc_label_y = 0.9
loc_label_size = 12

date_format = "%Y-%m-%d"
major_time_spacing = "1d"
num_minor_time_ticks = 4

major_freq_spacing = 1.0
num_minor_freq_ticks = 5

cbar_width = 0.01
cbar_offset_x = 0.02

# Read the hydrophone spectrograms
suffix_spec = get_spectrogram_file_suffix(window_length, overlap, downsample)
filename_in = f"whole_deployment_daily_hydro_spectrograms_{station}_{suffix_spec}.h5"

print("Reading the hydrophone spectrograms...")
inpath = join(indir, filename_in)
stream = read_hydro_spectrograms(inpath, min_freq = min_freq, max_freq = max_freq, starttime = starttime_geo, endtime = endtime_geo)

# Plot the hydrophone spectrograms
print("Plotting the hydrophone spectrograms...")
num_loc = len(loc_dict[station])
fig, axes = subplots(num_loc, 1, figsize = (width, row_height * num_loc), sharex = True, sharey = True)
cmap = get_cmap("inferno")
cmap.set_bad("gray")

for i, location in enumerate(loc_dict[station]):
    trace = stream.select(locations = location)[0]
    trace.to_db()
    timeax = trace.times
    freqax = trace.freqs
    data = trace.data
    
    ax = axes[i]
    quadmesh = ax.pcolormesh(timeax, freqax, data, cmap = cmap, shading = "auto", vmin = min_db, vmax = max_db)

    ax.text(loc_label_x, loc_label_y, location, transform = ax.transAxes, fontsize = loc_label_size, fontweight = "bold", va = "top", ha = "left", bbox = {"facecolor": "white", "alpha": 1.0, "pad": 5})

    if i < num_loc - 1:
        format_datetime_xlabels(ax,
                                label = False,
                                date_format = date_format, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                                va = "top", ha = "right", rotation = 45)
    else:
        format_datetime_xlabels(ax,
                                label = True,
                                date_format = date_format, major_tick_spacing = major_time_spacing, num_minor_ticks = num_minor_time_ticks,
                                va = "top", ha = "right", rotation = 45)

    format_freq_ylabels(ax,
                        label = True,
                        major_tick_spacing = major_freq_spacing, num_minor_ticks = num_minor_freq_ticks)

# Add the colorbar 
bbox = axes[-1].get_position()
cbar_pos = [bbox.x1 + cbar_offset_x, bbox.y0, cbar_width, bbox.height]
cbar = add_colorbar(fig, quadmesh, "dB", cbar_pos, orientation = "vertical")

# Save the figure
print("Saving the figure...")
suffix_start = time2suffix(starttime)
suffix_end = time2suffix(endtime)

figname = f"hydro_spectrograms_{station}_{suffix_spec}_{suffix_start}to{suffix_end}_freq{min_freq:.2f}to{max_freq:.2f}hz.png"
save_figure(fig, figname)
    

