"""
Count the number of station triads for each time window and component
"""


### Import packages ###
from os.path import join
from argparse import ArgumentParser
from pandas import concat, read_csv

from utils_basic import GEO_COMPONENTS as components, MT_DIR as dirname

### Input arguments ###
parser = ArgumentParser()
parser.add_argument("mode_name", type = str, help = "Name of the mode", default = "PR02549")
parser.add_argument("window_length_mt", type = float, help = "Window length in seconds", default = 900.0)
parser.add_argument("min_cohe", type = float, help = "Minimum coherence", default = 0.85)

args = parser.parse_args()
mode_name = args.mode_name
window_length_mt = args.window_length_mt
min_cohe = args.min_cohe

### Load data ###
# List of station triads
filename = "delaunay_station_triads.csv"
filepath = join(dirname, filename)

triad_df = read_csv(filepath)

# Apparent velocity for each station triad
app_vel_dfs = []
for _, row in triad_df.iterrows():
    station1, station2, station3 = row["station1"], row["station2"], row["station3"]
    
    filename = f"station_triad_app_vels_{mode_name}_{station1}_{station2}_{station3}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
    filepath = join(dirname, filename)

    app_vel_df = read_csv(filepath, parse_dates = ["time"])

    app_vel_dfs.append(app_vel_df)

app_vel_df = concat(app_vel_dfs)

### Count the number of station triads for each time window and component ###
for i, component in enumerate(components):
    # Select the apparent velocity for the component
    app_vel_comp_df = app_vel_df[["time", f"vel_app_{component.lower()}"]].copy()

    # Remove the nan values
    app_vel_comp_df.dropna(inplace = True)

    # Group by time window and count the number of station triads
    num_triads_comp_df = app_vel_comp_df.groupby("time").agg(count=('time', 'size'))
    num_triads_comp_df.rename(columns={'count': f'num_triad_{component.lower()}'}, inplace=True)
    
    # Merge the data frames of all components
    if i == 0:
        num_triads_df = num_triads_comp_df.copy()
    else:
        num_triads_df = num_triads_df.merge(num_triads_comp_df, on = "time", how = "outer")
        num_triads_df.fillna(0, inplace = True)

### Save the data ###
filename = f"num_triads_time_window_{mode_name}_mt_win{window_length_mt:.0f}s_min_cohe{min_cohe:.2f}.csv"
filepath = join(dirname, filename)

# Convert columns to int before saving
num_triads_df["num_triad_1"] = num_triads_df["num_triad_1"].astype(int)
num_triads_df["num_triad_2"] = num_triads_df["num_triad_2"].astype(int) 
num_triads_df["num_triad_z"] = num_triads_df["num_triad_z"].astype(int)

num_triads_df.to_csv(filepath, index = True)

print(f"Saved to {filepath}.")
