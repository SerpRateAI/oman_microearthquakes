"""
This script is used to get the picks for every hammer from a file containing all the picks
"""

from os.path import join
from utils_basic import PICK_DIR as dirpath_pick, LOC_DIR as dirpath_loc
from utils_snuffler import parse_hammer_picks

# Parse the picks
filepath = join(dirpath_pick, "hammer_p_picks.txt")
hammer_dict = parse_hammer_picks(filepath)
print(f"In total, there are {len(hammer_dict)} hammer shots")

# Write the picks for each hammer
for hammer_id, pick_df in hammer_dict.items():
    print(f"Working on hammer {hammer_id}...")
    filepath = join(dirpath_pick, f"hammer_p_picks_{hammer_id}.txt")
    pick_df.to_csv(filepath, index=False)
    print(f"Wrote the picks for hammer {hammer_id} to {filepath}")

# Write the list of hammers
filepath = join(dirpath_loc, "hammers.csv")
with open(filepath, "w") as f:
    f.write("hammer_id\n")
    for hammer_id in hammer_dict.keys():
        f.write(f"{hammer_id}\n")

print(f"Wrote the list of hammers to {filepath}")