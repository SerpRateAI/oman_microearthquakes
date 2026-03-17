
# Download the broadband data from the Grand Coulee station
# Imports
from os import makedirs
from os.path import join
from argparse import ArgumentParser
from time import time
from pandas import date_range
from pandas import Timestamp
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from utils_basic import COULEE_DATA_DIR as outdir

### Inputs ###
# Command line arguments
parser = ArgumentParser(description="Plot the map of the stations whose phase differences are derived and the respective observations for the AGU 2024 iPoster")
parser.add_argument("--station", type=str, default="GRCO2", help="Station")
parser.add_argument("--starttime", type=Timestamp, help="Start time", default="2022-10-04 00:00:00")
parser.add_argument("--endtime", type=Timestamp, help="End time", default="2022-10-05 00:00:00")

# Parse the command line arguments
args = parser.parse_args()

station = args.station
starttime = args.starttime
endtime = args.endtime

# Constants
client_name = "IRIS"
network = "RE"

# Get the client
client = Client(client_name)

# Generate the root directory
makedirs(outdir, exist_ok = True)

# Get the date range with one day intervals
dates = date_range(start = starttime, end = endtime, freq = "D")

# Get the data
print("Acquiring the data...")
for i, date in enumerate(dates):
    print(f"Acquiring the data for {date}...")
    clock1 = time()
    starttime = UTCDateTime(date)
    endtime = starttime + 24.0 * 3600.0


    print("Downloading the waveforms...")
    try:
        stream = client.get_waveforms(network = network, station = station, location = "*", channel = "HH*", starttime = starttime, endtime = endtime)
    except Exception as e:
        print(f"Failed to get the waveforms for {date} with error: {e}. The data will be skipped.")
        continue
    
    # if len(stream) != 3:
    #     print(f"Abnormal number of traces for {date}. The data will be skipped.")
    #     continue

    print("Waveforms successfully obtained!")

    # Save the data
    print("Saving the data...")
    outdir_date = join(outdir, date.strftime("%Y-%m-%d"))
    makedirs(outdir_date, exist_ok = True)
    print(f"Directory created at {outdir_date}.")

    for trace in stream:
        outpath = join(outdir_date, f"{trace.id}.mseed")
        trace.write(outpath, format = "MSEED")
        print(f"Data saveed to {outpath}.")

    clock2 = time()
    print(f"Data acquisition for {date} completed in {clock2 - clock1} seconds.")