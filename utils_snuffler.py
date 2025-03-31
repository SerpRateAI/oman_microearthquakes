# Functions and classes for handleing Snuffler output files
from pandas import read_csv, to_datetime

def read_normal_markers(inpath):
        
    pick_df = read_csv(inpath, sep=" ", skiprows=1, skipinitialspace=True, header=None)
    pick_df.drop([2], axis=1, inplace=True)
    pick_df.columns = ["date", "time_of_day", "seed_id"]
    pick_df["time"] = to_datetime(pick_df["date"] + " " + pick_df["time_of_day"])

    pick_df["station"] = pick_df["seed_id"].str[3:6]
    pick_df.drop(["date", "time_of_day", "seed_id"], axis=1, inplace=True)
    
    return pick_df

def read_phase_markers(inpath):
    
    pick_df = read_csv(inpath, sep=" ", skiprows=1, skipinitialspace=True, header=None)
    pick_df.drop([0, 3, 5, 6, 7, 9, 6, 10], axis=1, inplace=True)
    pick_df.columns = ["date", "time_of_day", "seed_id", "phase"]

    pick_df["time"] = to_datetime(pick_df["date"] + " " + pick_df["time_of_day"])

    pick_df["station"] = pick_df["seed_id"].str[3:6]
    pick_df.drop(["date", "time_of_day"], axis=1, inplace=True)

    return pick_df