"""
Compute the apparent velocities for the P phases of a hammer shot on the geophone triads
"""

###
# Import the necessary libraries
###
from os.path import join
from argparse import ArgumentParser
from numpy import rad2deg
from pandas import read_csv, DataFrame, Timedelta

from utils_basic import LOC_DIR as dirpath_loc, PICK_DIR as dirpath_pick
from utils_basic import get_geophone_coords, get_geophone_pairs, get_geophone_triads
from utils_snuffler import read_normal_markers
from utils_loc import get_triad_app_vel, get_dist_mat_inv

###
# Input parameters
###
parser = ArgumentParser()
parser.add_argument('--hammer_id', type=str, required=True)

args = parser.parse_args()
hammer_id = args.hammer_id

###
# Read the Input files
###
print("Reading the station information...")
station_df = get_geophone_coords()
pair_df = get_geophone_pairs()
triad_df = get_geophone_triads()

# Read the hammer location
inpath = join(dirpath_loc, 'hammer_locations.csv')
hammer_loc_df = read_csv(inpath)

# Read the hammer arrival times
filename = f"hammer_{hammer_id}_p_geo.txt"
inpath = join(dirpath_pick, filename)
hammer_pick_df = read_normal_markers(inpath)

###
# Compute the differential arrival times between station pairs
###
print("Computing the differential arrival times between station pairs...")
diff_time_dicts =[]
for _, row in pair_df.iterrows():
    station1 = row['station1']
    station2 = row['station2']

    try:
        atime1 = hammer_pick_df.loc[hammer_pick_df['station'] == station1, 'time'].values[0]
        atime2 = hammer_pick_df.loc[hammer_pick_df['station'] == station2, 'time'].values[0]
    except:
        print(f"No pick found for station {station1} or {station2}. The station pair is skipped.")
        continue

    atime_diff = Timedelta(atime2 - atime1)
    atime_diff = atime_diff.total_seconds()

    diff_time_dicts.append({
        'station1': station1,
        'station2': station2,
        'diff_time': atime_diff
    })
    print(f"The differential arrival time between {station1} and {station2} is {atime_diff} seconds.")

diff_time_df = DataFrame(diff_time_dicts)
# print(diff_time_df)

###
# Compute the apparent velocities for the triads
###
print("Computing the apparent velocities for the triads...")
app_vel_dicts = []
for _, row in triad_df.iterrows():
    station1 = row['station1']
    station2 = row['station2']
    station3 = row['station3']
    north_triad = row["north"]
    east_triad = row["east"]

    try:
        diff_time_12 = diff_time_df.loc[ (diff_time_df['station1'] == station1) & (diff_time_df['station2'] == station2), 'diff_time' ].values[0]
        diff_time_23 = diff_time_df.loc[ (diff_time_df['station1'] == station2) & (diff_time_df['station2'] == station3), 'diff_time' ].values[0]
    except:
        print(f"The station triad {station1}-{station2}-{station3} is missing at least one station pair. The station triad is skipped.")
        continue

    east1, north1 = station_df.loc[station1, 'east'], station_df.loc[station1, 'north']
    east2, north2 = station_df.loc[station2, 'east'], station_df.loc[station2, 'north']
    east3, north3 = station_df.loc[station3, 'east'], station_df.loc[station3, 'north']

    if station1 == 'A07' and station2 == 'A01' and station3 == 'A06':
        print(east1, north1, east2, north2, east3, north3)
        print(diff_time_12, diff_time_23)
    dist_mat_inv = get_dist_mat_inv(east1, north1, east2, north2, east3, north3)

    vel_app, back_azi, vel_app_east, vel_app_north = get_triad_app_vel(diff_time_12, diff_time_23, dist_mat_inv)

    app_vel_dicts.append({
        'station1': station1,
        'station2': station2,
        'station3': station3,
        'north': north_triad,
        'east': east_triad,
        'vel_app': vel_app,
        'back_azi': rad2deg(back_azi),
        'vel_app_east': vel_app_east,
        'vel_app_north': vel_app_north
    })

app_vel_df = DataFrame(app_vel_dicts)

###
# Write the output
###
filename = f"hammer_station_triad_app_vels_p_{hammer_id}.csv"
outpath = join(dirpath_loc, filename)
app_vel_df.to_csv(outpath, index=False)
print(f"Output written to {outpath}")