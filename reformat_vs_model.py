"""
Reformat the Vs model to the format accepted by Cake
"""

#-----------
# Imports
#-----------
from pathlib import Path
from pandas import read_csv
from numpy import array, savetxt
from utils_basic import VEL_MODEL_DIR as dirpath

#-----------
# Parameters
#-----------
filename = "vs_profile_a.csv"
filename_out = "vs_1d_a.nd"

#-----------
# Main
#-----------
filepath = Path(dirpath) / filename
model_df = read_csv(filepath)

model_df["depth"] = model_df["depth"] / 1000
model_df.rename(columns={"depth": "depth", "velocity": "vs"}, inplace=True)
model_df["vs"] = model_df["vs"] / 1000
model_df["vp"] = model_df["vs"] * 2
model_df["density"] = 1.0
model_df = model_df[["depth", "vp", "vs", "density"]]
model_mat = model_df.to_numpy()

filepath_out = Path(dirpath) / filename_out
savetxt(filepath_out, model_mat, fmt="%.5f")
print(f"Saved {filepath_out}")






