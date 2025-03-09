"""
Compute the station pairs that form the Delaunay triangles
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
triad = Delaunay(coords)
simplices = triad.simplices

### Extract and save the station pairs ###
num_simp = simplices.shape[0]

# Loop over all simplices
pair_dicts = []
triad_dicts = []
for i in range(num_simp):
    inds_simp = simplices[i, :]

    # Get the rows corresponding to the stations
    station1 = coord_df.index[inds_simp[0]]
    station2 = coord_df.index[inds_simp[1]]
    station3 = coord_df.index[inds_simp[2]]

    simp_df = coord_df.loc[[station1, station2, station3], :]

    # Sort the stations by the northern coordinate
    simp_df.sort_values(by = "north", ascending = False, inplace = True)

    # Compute the center coordinates of the triangle
    center_east = simp_df["east"].mean()
    center_north = simp_df["north"].mean()

    # Save the triad information
    station1 = simp_df.index[0]
    station2 = simp_df.index[1]
    station3 = simp_df.index[2]

    triad_dicts.append({"station1": station1, "station2": station2, "station3": station3, 
                        "east": center_east, "north": center_north})

    # Save the information of the first pair
    distance = sqrt((simp_df.loc[station1, "east"] - simp_df.loc[station2, "east"]) ** 2 + 
                    (simp_df.loc[station1, "north"] - simp_df.loc[station2, "north"]) ** 2)
    
    pair_dicts.append({"station1": station1, "station2": station2, "distance": distance})

    # Save the information of the second pair
    distance = sqrt((simp_df.loc[station2, "east"] - simp_df.loc[station3, "east"]) ** 2 + 
                    (simp_df.loc[station2, "north"] - simp_df.loc[station3, "north"]) ** 2)
    
    pair_dicts.append({"station1": station2, "station2": station3, "distance": distance})

triad_df = DataFrame(triad_dicts)
pair_df = DataFrame(pair_dicts)

# Drop duplicates based on the sorted_stations column
pair_df = pair_df.drop_duplicates(subset=["station1", "station2"])

# Sort by distance and reset the indices
pair_df.sort_values(by = "distance", ascending = True, inplace = True)
pair_df.reset_index(drop = True, inplace = True)

### Save the results ###
filepath = join(outdir, "delaunay_station_pairs.csv")
pair_df.to_csv(filepath)
print(f"Station pairs saved to {filepath}")

filepath = join(outdir, "delaunay_station_triads.csv")
triad_df.to_csv(filepath)
print(f"Station triads saved to {filepath}")

### Plot the simplices
fig, ax = subplots(1, 1)

# Plot the stations
ax.scatter(easts, norths, 5, marker = "^")

# Plot the simplices
ax.triplot(easts, norths, simplices)

ax.set_aspect("equal")

figname = "station_delaunay_triads.png"
save_figure(fig, figname)

