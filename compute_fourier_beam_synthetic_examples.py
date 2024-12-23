# Compute synthetic examples for Fourier beam-forming

### Import necessary libraries ###
from os.path import join
from argparse import ArgumentParser
from numpy import exp, cos, sin, deg2rad, pi
from numpy.random import uniform
from pandas import DataFrame
from matplotlib.pyplot import subplots

from utils_basic import get_geophone_coords
from utils_basic import INNER_STATIONS_A as inner_stations_a, INNER_STATIONS_B as inner_stations_b
from utils_basic import MIDDLE_STATIONS_A as middle_stations_a, MIDDLE_STATIONS_B as middle_stations_b
from utils_basic import OUTER_STATIONS_A as outer_stations_a, OUTER_STATIONS_B as outer_stations_b
from utils_array import get_fourier_beam_image, get_slowness_axes, select_stations
from utils_plot import add_colorbar, format_slowness_xlabels, format_slowness_ylabels, plot_reference_slownesses, save_figure

###### Functions ######
### Compute the synthetic Fourier coefficients for one mode ###
# slowness is in s/m and azimuth is in degrees clockwise from north
def get_synthetic_fourier_coefficients(freq, slow, azimuth, array_selection, array_aperture):
    # Select the stations
    stations, _ = select_stations(array_selection, array_aperture)

    # Get the station coordinates
    coord_df = get_geophone_coords()
    coord_df = coord_df.loc[stations]

    # Generate synthetic Fourier coefficients
    xslow = slow * sin(deg2rad(azimuth))
    yslow = slow * cos(deg2rad(azimuth))
    sta_coeff_dicts = []
    for station, coord in coord_df.iterrows():
        x = coord["east"]
        y = coord["north"]

        # Generate synthetic Fourier coefficients
        time_shift = x * xslow + y * yslow
        coeff = exp(-2j * pi * freq * time_shift)

        # Store the synthetic Fourier coefficients
        sta_coeff_dicts.append({"station": station, "x": x, "y": y, "peak_coeff": coeff, "peak_power": abs(coeff) ** 2})

    # Convert the dictionary to a DataFrame
    sta_coeff_df = DataFrame(sta_coeff_dicts)

    return sta_coeff_df

###### Main script ######
### Inputs ###
# Command-line arguments
parser = ArgumentParser(description="Compute synthetic examples for Fourier beam-forming consisting of two wave types.")
parser.add_argument("--freq", type=float, help="Frequency in Hz.")
parser.add_argument("--vel_app1", type=float, help="Apparent velocity of wave type 1 in m/s.")
parser.add_argument("--azimuth1", type=float, help="Azimuth 1 in degrees.")
parser.add_argument("--vel_app2", type=float, help="Apparent velocity of wave type 2 in m/s.")
parser.add_argument("--azimuth2", type=float, help="Azimuth 2 in degrees.")
parser.add_argument("--array_selection", type=str, help="Selection of the subarray: A or B.")
parser.add_argument("--array_aperture", type=str, help="Aperture of the array: small, medium, or large.")

args = parser.parse_args()
freq = args.freq
vel_app1 = args.vel_app1
azimuth1 = args.azimuth1
vel_app2 = args.vel_app2
azimuth2 = args.azimuth2
array_selection = args.array_selection
array_aperture = args.array_aperture

slow1 = 1.0 / vel_app1 # Slowness of wave type 1 in s/m
slow2 = 1.0 / vel_app2 # Slowness of wave type 2 in s/m

# Constants
amplitude = 0.5 # Amplitude of the first mode in the homogeneous mixture model
min_vel_app = 500.0 # Minimum apparent velocity in m/s
num_vel_app = 51 # Number of apparent velocities along each axis
vels_ref = [500, 1500, 3500] # Reference apparent velocities in m/s

marker_size = 50 # Marker size for the true and estiamted slownesses
marker_line_width = 1.0 # Line width for the true and estimated slownesses
color_est = "deepskyblue" # Color for the estimated slownesses
colorbar_offset = 0.02 # Offset of the colorbar from the right edge of the last subplot
colorbar_width = 0.01 # Width of the colorbar

# Print the inputs
print("###")
print("Computing synthetic examples for Fourier beam-forming consisting of two wave types.")
print(f"Frequency: {freq:.1f} Hz.")
print(f"Slowness 1: {slow1:.1f} s/m.")
print(f"Azimuth 1: {azimuth1:.1f} degrees.")
print(f"Slowness 2: {slow2:.1f} s/m.")
print(f"Azimuth 2: {azimuth2:.1f} degrees.")
print(f"Selection of the subarray: {array_selection}.")
print(f"Aperture of the array: {array_aperture}.")
print("###")
print("")

### Define the x and y slow axes ###
xslowax, yslowax = get_slowness_axes(min_vel_app, num_vel_app)

### Compute the case with wave type 1 only ###
# Compute the synthetic Fourier coefficients for wave type 1
sta_coeff1_df = get_synthetic_fourier_coefficients(freq, slow1, azimuth1, array_selection, array_aperture)

# Compute the Fourier beam
bimage1 = get_fourier_beam_image(freq, sta_coeff1_df, xslowax, yslowax)

### Compute the case with wave type 2 only ###
# Compute the synthetic Fourier coefficients for wave type 2
sta_coeff2_df = get_synthetic_fourier_coefficients(freq, slow2, azimuth2, array_selection, array_aperture)

# Compute the Fourier beam
bimage2 = get_fourier_beam_image(freq, sta_coeff2_df, xslowax, yslowax)

### Compute the case with the homogeneous mixture model ###
# Compute the synthetic Fourier coefficients for the homogeneous mixture model
sta_coeff_hom_df = sta_coeff1_df.copy()
sta_coeff_hom_df["peak_coeff"] = amplitude * sta_coeff1_df["peak_coeff"] + (1 - amplitude) * sta_coeff2_df["peak_coeff"]

# Compute the Fourier beam
bimage_hom = get_fourier_beam_image(freq, sta_coeff_hom_df, xslowax, yslowax)

### Compute the case with the inhomogeneous mixture model ###
# Compute the synthetic Fourier coefficients for the inhomogeneous mixture model
amplitudes = uniform(0.0, 1.0, len(sta_coeff1_df))
sta_coeff_het_df = sta_coeff1_df.copy()
sta_coeff_het_df["peak_coeff"] = amplitudes * sta_coeff1_df["peak_coeff"] + (1 - amplitudes) * sta_coeff2_df["peak_coeff"]

# Compute the Fourier beam
bimage_het = get_fourier_beam_image(freq, sta_coeff_het_df, xslowax, yslowax)

### Plot the Fourier beams of the four cases ###
# Convert the slownesses t s/km
xslowax *= 1000.0
yslowax *= 1000.0

xslow1 = slow1 * sin(deg2rad(azimuth1)) * 1000.0
yslow1 = slow1 * cos(deg2rad(azimuth1)) * 1000.0

xslow2 = slow2 * sin(deg2rad(azimuth2)) * 1000.0
yslow2 = slow2 * cos(deg2rad(azimuth2)) * 1000.0

# Generate the figure
fig, axes = subplots(2, 2, figsize = (12, 12), sharex = True, sharey = True)

# Plot the case with Wave Type 1 only
ax = axes[0, 0]
power_image = bimage1.bimage

coherence = bimage1.coherence
xslow_est = bimage1.xslow_global_max * 1000.0
yslow_est = bimage1.yslow_global_max * 1000.0

mappable = ax.pcolor(xslowax, yslowax, power_image, 
                     cmap = "inferno", vmin = 0.0, vmax = 1.0)

plot_reference_slownesses(ax, vels_ref,
                          plot_label = True,
                          color = "deepskyblue")


ax.scatter(xslow1, yslow1, s = marker_size, color = "white", marker = "D", edgecolor = "black", linewidth = marker_line_width)
ax.scatter(xslow_est, yslow_est, s = marker_size, color = color_est, marker = "D", edgecolor = "black", linewidth = marker_line_width)

ax.set_title(f"Wave type 1, coherence = {coherence:.2f}", fontsize = 12, fontweight = "bold")

# Plot the case with Wave Type 2 only
ax = axes[0, 1]
power_image = bimage2.bimage
coherence = bimage2.coherence
xslow_est = bimage2.xslow_global_max * 1000.0
yslow_est = bimage2.yslow_global_max * 1000.0

mappable = ax.pcolor(xslowax, yslowax, power_image, 
                     cmap = "inferno", vmin = 0.0, vmax = 1.0)

plot_reference_slownesses(ax, vels_ref,
                          plot_label = False,
                          color = "deepskyblue")
ax.scatter(xslow2, yslow2, s = marker_size, color = "white", marker = "D", edgecolor = "black", linewidth = marker_line_width)
ax.scatter(xslow_est, yslow_est, s = marker_size, color = color_est, marker = "D", edgecolor = "black", linewidth = marker_line_width)

ax.set_title(f"Wave type 2, coherence = {coherence:.2f}", fontsize = 12, fontweight = "bold")

# Plot the case with the homogeneous mixture model
ax = axes[1, 0]
power_image = bimage_hom.bimage

coherence = bimage_hom.coherence
xslow_est = bimage_hom.xslow_global_max * 1000.0
yslow_est = bimage_hom.yslow_global_max * 1000.0

mappable = ax.pcolor(xslowax, yslowax, power_image, 
                     cmap = "inferno", vmin = 0.0, vmax = 1.0)

plot_reference_slownesses(ax, vels_ref,
                          plot_label = False,
                          color = "deepskyblue")

ax.scatter(xslow1, yslow1, s = marker_size, color = "white", marker = "D", edgecolor = "black", linewidth = marker_line_width)
ax.scatter(xslow2, yslow2, s = marker_size, color = "white", marker = "D", edgecolor = "black", linewidth = marker_line_width)
ax.scatter(xslow_est, yslow_est, s = marker_size, color = color_est, marker = "D", edgecolor = "black", linewidth = marker_line_width)


ax.set_title(f"1-to-1 mixture, coherence = {coherence:.2f}", fontsize = 12, fontweight = "bold")

# Set the common x and y labels
format_slowness_xlabels(ax)
format_slowness_ylabels(ax)

# Plot the case with the hetreogeneous mixture model
ax = axes[1, 1]
power_image = bimage_het.bimage

coherence = bimage_het.coherence
xslow_est = bimage_het.xslow_global_max * 1000.0
yslow_est = bimage_het.yslow_global_max * 1000.0

mappable = ax.pcolor(xslowax, yslowax, power_image, 
                     cmap = "inferno", vmin = 0.0, vmax = 1.0)

plot_reference_slownesses(ax, vels_ref,
                          plot_label = False,
                          color = "deepskyblue")

ax.scatter(xslow1, yslow1, s = marker_size, color = "white", marker = "D", edgecolor = "black", linewidth = marker_line_width)
ax.scatter(xslow2, yslow2, s = marker_size, color = "white", marker = "D", edgecolor = "black", linewidth = marker_line_width)
ax.scatter(xslow_est, yslow_est, s = marker_size, color = color_est, marker = "D", edgecolor = "black", linewidth = marker_line_width)

ax.set_title(f"Heterogeneous mixture, coherence = {coherence:.2f}", fontsize = 12, fontweight = "bold")

### Add the colorbar ###
bbox = ax.get_position()
position = [bbox.x1 + colorbar_offset, bbox.y0, colorbar_width, bbox.height]
add_colorbar(fig, position, "Normalized power", mappable = mappable)

### Add the super title ###
fig.suptitle(f"Synthetic examples, frequency = {freq:.1f} Hz, Array {array_selection.upper()}, {array_aperture} aperture", fontsize = 12, fontweight = "bold", y = 0.93)

### Save the figure ###
figname = f"fourier_beam_synthetic_examples_freq{freq:.0f}hz_{array_selection.lower()}_{array_aperture}.png"
save_figure(fig, figname)







