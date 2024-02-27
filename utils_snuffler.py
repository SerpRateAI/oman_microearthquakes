# Functions and classes for handleing Snuffler output files
from pandas import read_csv, to_datetime

def read_normal_markers(inpath):
        
        pickdf = read_csv(inpath, sep=" ", skiprows=1, skipinitialspace=True, header=None)
        pickdf.drop([2], axis=1, inplace=True)
        pickdf.columns = ["date", "time_of_day", "seed_id"]
        pickdf["time"] = to_datetime(pickdf["date"] + " " + pickdf["time_of_day"])

        pickdf["station"] = pickdf["seed_id"].str[3:6]
        pickdf.drop(["date", "time_of_day", "seed_id"], axis=1, inplace=True)
    
        return pickdf

def read_phase_markers(inpath):
    
    pickdf = read_csv(inpath, sep=" ", skiprows=1, skipinitialspace=True, header=None)
    pickdf.drop([0, 3, 5, 6, 7, 9, 6, 10], axis=1, inplace=True)
    pickdf.columns = ["date", "time_of_day", "seed_id", "phase"]

    pickdf["time"] = pickdf["date"] + " " + pickdf["time_of_day"]
    pickdf["time"] = to_datetime(pickdf["time"])

    pickdf["station"] = pickdf["seed_id"].str[3:6]
    pickdf.drop(["date", "time_of_day"], axis=1, inplace=True)

    return pickdf