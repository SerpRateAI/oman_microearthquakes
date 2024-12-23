# Merge multiple DEM files into a single one
# Imports
from os.path import join
from rasterio import open
from rasterio.merge import merge
from matplotlib.pyplot import subplots

from utils_basic import ROOTDIR_MAP as indir
from utils_plot import save_figure

# Inputs
filenames = ["n22_e058_1arc_v3.tif", "n23_e058_1arc_v3.tif"]

figwidth = 15.0

# Read and merge the DEM files
print("Reading and merging the DEM files...")
inpaths = [join(indir, filename) for filename in filenames]

# Open all the files as datasets
datasets = [open(file) for file in inpaths]

# Merge the datasets into a single array
mosaic, out_trans = merge(datasets)

# Get the metadata of the first dataset
out_meta = datasets[0].meta.copy()

# Update the metadata
out_meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans})

# Close the datasets
for dataset in datasets:
    dataset.close()

# Save the merged DEM
outpath = join(indir, "merged_dem.tif")
with open(outpath, "w", **out_meta) as dest:
    dest.write(mosaic)
    
print(f"Saved the merged DEM to {outpath}")