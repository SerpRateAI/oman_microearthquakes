# Plot the horizontal particle motions of a stationary resonance in a time window at all stations
#
### Import the required libraries ###
from os.path import join
from argparse import ArgumentParser
from pandas import read_hdf
from matplotlib.pyplot import subplots

from utils_basic import SPECTROGRAM_DIR as indir
from utils_basic import EASTMIN_WHOLE as min_east_whole, EASTMAX_WHOLE as max_east_whole, NORTHMIN_WHOLE as min_north_whole, NORTHMAX_WHOLE as max_north_whole
from utils_basic import EASTMIN_A_INNER as min_east_a_inner, EASTMAX_A_INNER as max_east_a_inner, NORTHMIN_A_INNER as min_north_a_inner, NORTHMAX_A_INNER as max_north_a_inner
from utils_basic import EASTMIN_B_INNER as min_east_b_inner, EASTMAX_B_INNER as max_east_b_inner, NORTHMIN_B_INNER as min_north_b_inner, NORTHMAX_B_INNER as max_north_b_inner
from utils_basic import INNER_STATIONS_A as inner_stations_a, INNER_STATIONS_B as inner_stations_b
from utils_basic import get_geophone_coords, str2timestamp, time2suffix
from utils_preproc import read_and_process_windowed_geo_waveforms
from utils_spec import get_start_n_end_from_center
from utils_plot import add_vertical_scalebar, format_east_xlabels, format_north_ylabels, save_figure

### Inputs ###
# Command line arguments
parser = ArgumentParser(description = "Plot the horizontal particle motions of a stationary resonance in a time window at all stations")
parser.add_argument("--mode_name", type = str, help = "Mode name")
parser.add_argument("--center_time", type = str, help = "Center time of the time window")
parser.add_argument("--window_length", type = float, help = "Length of the time window in seconds")
parser.add_argument("--scale_factor_whole", type = float, help = "Scale factor for the whole array")
parser.add_argument("--scale_factor_a_inner", type = float, help = "Scale factor for the A inner stations")
parser.add_argument("--scale_factor_b_inner", type = float, help = "Scale factor for the B inner stations")
parser.add_argument("--scalebar_length_whole", type = float, help = "Length of the scalebar for the whole array in nm/s")
parser.add_argument("--scalebar_length_a_inner", type = float, help = "Length of the scalebar for the A inner stations in nm/s")
parser.add_argument("--scalebar_length_b_inner", type = float, help = "Length of the scalebar for the B inner stations in nm/s")

# Constants
map_width = 10.0
line_width = 0.1
marker_size_whole = 10
marker_size_inner = 20

x_scalebar = 0.05
y_scalebar = 0.05

x_label_offset = 0.02
y_label_offset = 0.00

x_station_label_offset = 0.0
y_station_label_offset = 10.0

# Parse the command line arguments
args = parser.parse_args()
mode_name = args.mode_name
center_time = str2timestamp(args.center_time)
window_length = args.window_length
scale_factor_whole = args.scale_factor_whole
scale_factor_a_inner = args.scale_factor_a_inner
scale_factor_b_inner = args.scale_factor_b_inner
scalebar_length_whole = args.scalebar_length_whole
scalebar_length_a_inner = args.scalebar_length_a_inner
scalebar_length_b_inner = args.scalebar_length_b_inner

print(f"Center time: {center_time}")
print(f"Window length: {window_length:.1f} s")

### Read the data ###
# Read the geophoen coordinates
print("Reading the geophone coordinates...")
coords_df = get_geophone_coords()

# Read the average properties of the mode
print(f"Reading the average properties of {mode_name}...")
filename = f"stationary_resonance_geo_mean_properties_{mode_name}.h5"
inpath = join(indir, filename)
properties_df = read_hdf(inpath, key = "properties")

freq_reson = properties_df.loc[center_time, "frequency"]
print(f"Resonant frequency: {freq_reson:.2f} Hz")
qf_reson = properties_df.loc[center_time, "mean_quality_factor"]
print(f"Mean quality factor: {qf_reson:.2f}")

qf_filt = qf_reson / 2
print(f"Quality factor for peak filtering: {qf_filt:.2f}")

# Read and process the waveforms
print("Reading and processing the waveforms...")
starttime, endtime = get_start_n_end_from_center(center_time, window_length)
stream = read_and_process_windowed_geo_waveforms(starttime, 
                                                 endtime = endtime,
                                                 filter = True, filter_type = "peak",
                                                 zero_phase = True,
                                                 freq = freq_reson, qf = qf_filt)

### Plot the data ###
# Plot the whole array
print("Plotting the horizontal particle motions of the whole array...")
map_height = map_width * (max_north_whole - min_north_whole) / (max_east_whole - min_east_whole)
fig, ax = subplots(1, 1, figsize = (map_width, map_height))

for station, row in coords_df.iterrows():
    print(f"Plotting {station}...")
    east_sta = row["east"]
    north_sta = row["north"]

    trace_east = stream.select(station = station, component = "2")[0]
    trace_north = stream.select(station = station, component = "1")[0]

    pm_east = trace_east.data * scale_factor_whole + east_sta
    pm_north = trace_north.data * scale_factor_whole + north_sta

    ax.plot(pm_east, pm_north, color = "lightgray", linewidth = line_width, zorder = 1)
    ax.scatter(east_sta, north_sta, color = "crimson", s = marker_size_whole, zorder = 2)

ax.set_xlim(min_east_whole, max_east_whole)
ax.set_ylim(min_north_whole, max_north_whole)
ax.set_aspect("equal")

format_east_xlabels(ax,
                    major_tick_spacing = 50.0, num_minor_ticks = 5)

format_north_ylabels(ax,
                     major_tick_spacing = 50.0, num_minor_ticks = 5)

# Add the scalebar
print("Adding the scalebar...")
add_vertical_scalebar(ax, (x_scalebar, y_scalebar), scalebar_length_whole, scale_factor_whole, (x_label_offset, y_label_offset))

ax.set_title(f"{mode_name}, {center_time.strftime('%Y-%m-%d %H:%M:%S')}, {window_length:.1f} s", fontsize = 14, fontweight = "bold")

# Save the figure
print("Saving the figure...")
time_suffix = time2suffix(center_time)
filename = f"stationary_resonance_horizontal_pm_{mode_name}_{time_suffix}_{window_length:.1f}s.png"

save_figure(fig, filename)

# Plot the area around the A inner stations
map_height = map_width * (max_north_a_inner - min_north_a_inner) / (max_east_a_inner - min_east_a_inner)
fig, ax = subplots(1, 1, figsize = (map_width, map_height))

print("Plotting the horizontal particle motions of the A inner stations...")
for station, row in coords_df.iterrows():
    if station not in inner_stations_a:
        continue

    print(f"Plotting {station}...")
    east_sta = row["east"]
    north_sta = row["north"]

    trace_east = stream.select(station = station, component = "2")[0]
    trace_north = stream.select(station = station, component = "1")[0]

    pm_east = trace_east.data * scale_factor_a_inner + east_sta
    pm_north = trace_north.data * scale_factor_a_inner + north_sta

    ax.plot(pm_east, pm_north, color = "lightgray", linewidth = line_width, zorder = 1)
    ax.scatter(east_sta, north_sta, color = "crimson", s = marker_size_inner, zorder = 2)
    ax.annotate(station, xy = (east_sta, north_sta), xytext = (x_station_label_offset, y_station_label_offset),
                va = "bottom", ha = "center",
                textcoords = "offset points", fontsize = 12, color = "crimson")

ax.set_xlim(min_east_a_inner, max_east_a_inner)
ax.set_ylim(min_north_a_inner, max_north_a_inner)
ax.set_aspect("equal")

format_east_xlabels(ax,
                    major_tick_spacing = 10.0, num_minor_ticks = 5)

format_north_ylabels(ax,
                    major_tick_spacing = 10.0, num_minor_ticks = 5)

# Add the scalebar
print("Adding the scalebar...")
add_vertical_scalebar(ax, (x_scalebar, y_scalebar), scalebar_length_a_inner, scale_factor_a_inner, (x_label_offset, y_label_offset))

ax.set_title(f"{mode_name}, {center_time.strftime('%Y-%m-%d %H:%M:%S')}, {window_length:.1f} s", fontsize = 14, fontweight = "bold")

# Save the figure
print("Saving the figure...")
time_suffix = time2suffix(center_time)
filename = f"stationary_resonance_horizontal_pm_{mode_name}_{time_suffix}_{window_length:.1f}s_a_inner.png"

save_figure(fig, filename)

# Plot the area around the B inner stations
map_height = map_width * (max_north_b_inner - min_north_b_inner) / (max_east_b_inner - min_east_b_inner)
fig, ax = subplots(1, 1, figsize = (map_width, map_height))

print("Plotting the horizontal particle motions of the B inner stations...")
for station, row in coords_df.iterrows():
    if station not in inner_stations_b:
        continue

    print(f"Plotting {station}...")
    east_sta = row["east"]
    north_sta = row["north"]

    trace_east = stream.select(station = station, component = "2")[0]
    trace_north = stream.select(station = station, component = "1")[0]

    pm_east = trace_east.data * scale_factor_b_inner + east_sta
    pm_north = trace_north.data * scale_factor_b_inner + north_sta

    ax.plot(pm_east, pm_north, color = "lightgray", linewidth = line_width, zorder = 1)
    ax.scatter(east_sta, north_sta, color = "crimson", s = marker_size_inner, zorder = 2)
    ax.annotate(station, xy = (east_sta, north_sta), xytext = (x_station_label_offset, y_station_label_offset), 
                textcoords = "offset points", fontsize = 10, color = "crimson")
    
ax.set_xlim(min_east_b_inner, max_east_b_inner)
ax.set_ylim(min_north_b_inner, max_north_b_inner)
ax.set_aspect("equal")

format_east_xlabels(ax,
                    major_tick_spacing = 10.0, num_minor_ticks = 5)

format_north_ylabels(ax,
                    major_tick_spacing = 10.0, num_minor_ticks = 5)

# Add the scalebar
print("Adding the scalebar...")
add_vertical_scalebar(ax, (x_scalebar, y_scalebar), scalebar_length_b_inner, scale_factor_b_inner, (x_label_offset, y_label_offset))

ax.set_title(f"{mode_name}, {center_time.strftime('%Y-%m-%d %H:%M:%S')}, {window_length:.1f} s", fontsize = 14, fontweight = "bold")

# Save the figure
print("Saving the figure...")
time_suffix = time2suffix(center_time)
filename = f"stationary_resonance_horizontal_pm_{mode_name}_{time_suffix}_{window_length:.1f}s_b_inner.png"

save_figure(fig, filename)
    







                                                 

