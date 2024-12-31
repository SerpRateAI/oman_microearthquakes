"""
Compute delauney triangles of geophone station
"""

### Imports ###
from os.path import join
from numpy import column_stack, sqrt
from scipy.spatial import Delaunay
from pandas import DataFrame
from matplotlib.pyplot import subplots

from utils_basic import MT_DIR as outdir
from utils_basic import get_geophone_coords

from utils_plot import save_figure

### Load the station coordinates ###
coord_df = get_geophone_coords()
easts = coord_df["east"].values
norths = coord_df["north"].values

### Build the Delauney triangles ###
coords = column_stack((easts, norths))
tria = Delaunay(coords)
simplices = tria.simplices

### Extract and save the station pairs ###
num_simp = simplices.shape[0]

# Loop over all simplices
result_dicts = []
for i in range(num_simp):
    inds_simp = simplices[i, :]

    station1 = coord_df.index[inds_simp[0]]
    station2 = coord_df.index[inds_simp[1]]
    east1 = coord_df.loc[station1, "east"]
    east2 = coord_df.loc[station2, "east"]
    north1 = coord_df.loc[station1, "north"]
    north2 = coord_df.loc[station2, "north"]

    distance = sqrt((east1 - east2) ** 2 + (north1 - north2) ** 2)

    # Place the northern station first
    if north1 < north2:
        station1, station2 = station2, station1

    print((station1, station2, distance))
    result_dicts.append({"station1": station1, "station2": station2, "distance": distance})

    station1 = coord_df.index[inds_simp[1]]
    station2 = coord_df.index[inds_simp[2]]
    east1 = coord_df.loc[station1, "east"]
    east2 = coord_df.loc[station2, "east"]
    north1 = coord_df.loc[station1, "north"]
    north2 = coord_df.loc[station2, "north"]

    distance = sqrt((east1 - east2) ** 2 + (north1 - north2) ** 2)

    # Place the southern station first
    if north1 < north2:
        station1, station2 = station2, station1

    print((station1, station2, distance))
    result_dicts.append({"station1": station1, "station2": station2, "distance": distance})   

result_df = DataFrame(result_dicts)

# Drop duplicates based on the sorted_stations column
result_df = result_df.drop_duplicates(subset=["station1", "station2"])

# Sort by distance and reset the indices
result_df.sort_values(by = "distance", ascending = True, inplace = True)
result_df.reset_index(drop = True, inplace = True)


filepath = join(outdir, "delaunay_station_pairs.csv")
result_df.to_csv(filepath)
print(f"Station pairs saved to {filepath}")

### Plot the simplices
fig, ax = subplots(1, 1)

ax.scatter(easts, norths, 5, marker = "^")
ax.triplot(easts, norths, simplices)

ax.set_aspect("equal")

figname = "station_delaunay_triangles.png"
save_figure(fig, figname)

