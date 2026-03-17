from os.path import join, basename
from argparse import ArgumentParser
from numpy import empty_like, zeros, isfinite, percentile, clip, uint8, float32, issubdtype, integer, where
from numpy import amax
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.enums import ColorInterp

from utils_basic import IMAGE_DIR as dirpath, CENTER_LATITUDE as lat0, CENTER_LONGITUDE as lon0

parser = ArgumentParser(description="Plot the map of the stations whose phase differences are derived and the respective observations for the AGU 2024 iPoster")
parser.add_argument("--filename_in", type=str, help="Input filename")
parser.add_argument("--filename_out", type=str, help="Output filename")
args = parser.parse_args()

filename_in = args.filename_in
filename_out = args.filename_out

src_crs = "EPSG:4326"  # change if needed
dst_crs = rasterio.crs.CRS.from_string(
    f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
)

path_in = join(dirpath, filename_in)
path_out = join(dirpath, filename_out)

with rasterio.open(path_in) as src:
    transform, width, height = calculate_default_transform(
        src.crs or src_crs, dst_crs, src.width, src.height, *src.bounds, resolution=1.0
    )
    profile = src.profile.copy()
    profile.update(crs=dst_crs, transform=transform, width=width, height=height)

    with rasterio.open(path_out, "w", **profile) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs or src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
        print(f"Rotated image saved to {path_out}")