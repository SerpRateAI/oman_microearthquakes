# Convert the satellite image to local coordinate system

# Import the required libraries
from os.path import join

from rasterio import open as rasterio_open, band
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

from utils_basic import IMAGE_DIR as indir, CENTER_LATITUDE as lat0, CENTER_LONGITUDE as lon0

# Sattelite image path
dirpath = join(indir, "spot_2019-12-06/PROD_SPOT7_001/VOL_SPOT7_001_A/IMG_SPOT7_PMS_001_A")
filename_in = "IMG_SPOT7_PMS_201912060614294_ORT_9f15aaaf-6d3d-4468-c42a-3631dd99069d_R1C1.TIF"

filename_out = "spot_2019-12-06_local.tif"

# Path to the input GeoTIFF file
inpath = join(dirpath, filename_in)

# Define the local Cartesian CRS (example: UTM zone centered at the region of interest)
local_crs = CRS.from_proj4(f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# Open the GeoTIFF file
with rasterio_open(inpath) as src:
    # Calculate transform and new dimensions in the local CRS
    transform, width, height = calculate_default_transform(
        src.crs, local_crs, src.width, src.height, *src.bounds
    )

    # Define metadata for the output file
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": local_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    # Write the reprojected and resampled raster
    outpath = join(indir, filename_out)
    with rasterio_open(outpath, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):  # Iterate over raster bands
            reproject(
                source=band(src, i),
                destination=band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=local_crs,
                resampling=Resampling.bilinear  # Use bilinear interpolation
            )

print(f"Reprojected GeoTIFF saved to {outpath}")