"""
Utility functions for handling satellite images
"""

from os.path import join
from utils_basic import LOC_DIR as dirpath_loc, IMAGE_DIR as dirpath_img
from pandas import read_csv
from rasterio import open as rasterio_open
from rasterio.warp import reproject, Resampling
from rasterio.plot import reshape_as_image


MAXAR_IMAGE_FILENAME = "maxar_2019-09-17_local.tif"

# Load the high-resolution Maxar image
def load_maxar_image():
    # Load the satellite image
    inpath = join(dirpath_img, MAXAR_IMAGE_FILENAME)
    with rasterio_open(inpath) as src:
        # Read the image in RGB format
        rgb_band = src.read([1, 2, 3])

        # Reshape the image
        image = reshape_as_image(rgb_band)

        # Extract the extent of the image
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    return image, extent