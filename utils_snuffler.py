# Functions and classes for handleing Snuffler output files
from pandas import read_csv, to_datetime

# Read a list of normal picks
def read_normal_markers(inpath):
        
    pick_df = read_csv(inpath, sep=" ", skiprows=1, skipinitialspace=True, header=None)
    pick_df.drop([2], axis=1, inplace=True)
    pick_df.columns = ["date", "time_of_day", "seed_id"]
    pick_df["time"] = to_datetime(pick_df["date"] + " " + pick_df["time_of_day"])

    pick_df["station"] = pick_df["seed_id"].str[3:6]
    pick_df.drop(["date", "time_of_day", "seed_id"], axis=1, inplace=True)
    
    return pick_df

# Parse the picks into different hammer shots
def parse_hammer_picks(inpath, max_gap_time: float = 1.0):
    pick_df = read_normal_markers(inpath)

    # Sort the picks by time
    pick_df.sort_values(by="time", inplace=True)

    # Loop through the picks
    hammer_dict = {}
    for i, row in pick_df.iterrows():
        if i == 0:
            hammer_id = row["time"].strftime("%H%M%S")
            pick_time_previous = row["time"]
            i_begin = 0
        else:
            pick_time_current = row["time"]
            if (pick_time_current - pick_time_previous).total_seconds() > max_gap_time:
                i_end = i
                hammer_dict[hammer_id] = pick_df.iloc[i_begin:i_end].copy()
                hammer_id = row["time"].strftime("%H%M%S")
                i_begin = i

            pick_time_previous = pick_time_current

    return hammer_dict

def read_phase_markers(inpath):
    
    pick_df = read_csv(inpath, sep=" ", skiprows=1, skipinitialspace=True, header=None)
    pick_df.drop([0, 3, 5, 6, 7, 9, 6, 10], axis=1, inplace=True)
    pick_df.columns = ["date", "time_of_day", "seed_id", "phase"]

    pick_df["time"] = to_datetime(pick_df["date"] + " " + pick_df["time_of_day"])

    pick_df["station"] = pick_df["seed_id"].str[3:6]
    pick_df.drop(["date", "time_of_day"], axis=1, inplace=True)

    return pick_df