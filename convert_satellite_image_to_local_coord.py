"""
Convert a satellite image to local coordinates and adjust the white balance
"""

# Import the required libraries
from os.path import join
from argparse import ArgumentParser
from rasterio import open as rasterio_open, band
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from numpy import amax,clip, empty, uint16

from utils_basic import IMAGE_DIR as rootdir, CENTER_LATITUDE as lat0, CENTER_LONGITUDE as lon0

# def adjust_white_balance(image_array, red_gain=1.0, green_gain=1.0, blue_gain=1.0):
#     """
#     Adjust the white balance of a NumPy array representing an RGB image.

#     Parameters:
#         image_array (np.ndarray): Input image array in the format (height, width, channels).
#         red_gain (float): Gain for the red channel.
#         green_gain (float): Gain for the green channel.
#         blue_gain (float): Gain for the blue channel.

#     Returns:
#         np.ndarray: White-balanced image array.
#     """
#     # Separate channels
#     red_channel = image_array[0] * red_gain
#     green_channel = image_array[1] * green_gain
#     blue_channel = image_array[2] * blue_gain

#     # Stack adjusted channels back
#     adjusted_image = np.stack([
#         clip(red_channel, 0, 255),
#         clip(green_channel, 0, 255),
#         clip(blue_channel, 0, 255)
#     ])

#     return adjusted_image

"""
Main function
"""

### Inputs ###
# Command-line arguments
parser = ArgumentParser(description = "Convert a satellite image to local coordinates and adjust the white balance.")
parser.add_argument("--dirname_sat", type = str, help = "Directory name of the satellite image.")
parser.add_argument("--filename_in", type = str, help = "Filename of the input satellite image.")
parser.add_argument("--filename_out", type = str, help = "Filename of the output satellite image.")
parser.add_argument("--scale_factor", type = float, default = 100, help = "Scaling factor for the satellite image.")

args = parser.parse_args()

### Read the inputs ###
dirname_sat = args.dirname_sat
dirpath = join(rootdir, dirname_sat)
filename_in = args.filename_in
filename_out = args.filename_out
scale_factor = args.scale_factor

### Process the satellite image ###

# Satellite image path
inpath = join(dirpath, filename_in)

# Define the local Cartesian CRS (example: UTM zone centered at the region of interest)
local_crs = CRS.from_proj4(f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# Open the GeoTIFF file
with rasterio_open(inpath) as src:
    # Check if the input file has at least three bands
    if src.count < 3:
        raise ValueError("The input GeoTIFF file does not contain at least 3 bands required for RGB extraction.")
    
    # Extract the indices of the R, G, and B bands
    # Assuming the band order is B, G, R, NIR
    rgb_band_indices = [1, 2, 3]
    
    # Calculate transform and new dimensions in the local CRS
    transform, width, height = calculate_default_transform(
        src.crs, local_crs, src.width, src.height, *src.bounds
    )

    # Define metadata for the output file
    kwargs = src.meta.copy()
    kwargs.update({
        "count": 3,  # Only R, G, B bands
        "dtype": "uint8",
        "crs": local_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    # Output path
    outpath = join(rootdir, filename_out)

    # Write the reprojected RGB raster
    with rasterio_open(outpath, "w", **kwargs) as dst:
        for i, band_index in enumerate(rgb_band_indices, start=1):  # Start with band 1 in the output
            # Create an empty array for the destination band
            dest_array = empty((height, width), dtype=uint16)

            # Read the source band
            src_band = src.read(band_index)
            print(amax(src_band))
            src_band = src_band * scale_factor
            

            # Reproject the source band into the destination array
            reproject(
                source=src_band,  # Read the R, G, or B band
                destination=dest_array,      # Allocate the reprojected band
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=local_crs,
                resampling=Resampling.bilinear
            )

            # Normalize pixel values to 0–255
            # dest_array = clip(dest_array, 0, 65535)  # Ensure values are within valid range
            # dest_array = (dest_array / 65535.0 * 255).astype("uint8")  # Scale to 0–255 range
            
            # Set the color interpretation for the destination band
            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

            # Write the reprojected band to the destination file
            dst.write(dest_array, indexes=i)

print(f"Reprojected GeoTIFF with RGB bands saved to {outpath}")
