
# Download the broadband data from the GFZ data center
# Imports
from os import makedirs
from os.path import join
from time import time
from pandas import date_range
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from utils_basic import GFZ_DATA_DIR as outdir

# Inputs
network = "5H"
station = "COO03"
starttime = "2015-11-23 00:00:00"
endtime = "2016-02-09 23:59:59"

# Get the client
client = Client("GEOFON")

# Get the date range with one day intervals
dates = date_range(starttime, endtime, freq = "D")

# Get the data
print("Acquiring the data...")
for i, date in enumerate(dates):
    print(f"Acquiring the data for {date}...")
    clock1 = time()
    starttime = UTCDateTime(date)
    endtime = starttime + 24.0 * 3600.0


    print("Downloading the waveforms...")
    try:
        stream = client.get_waveforms(network = network, station = station, location = "*", channel = "*", starttime = starttime, endtime = endtime)
    except Exception as e:
        print(f"Failed to get the waveforms for {date} with error: {e}. The data will be skipped.")
        continue
    
    if len(stream) != 3:
        print(f"Abnormal number of traces for {date}. The data will be skipped.")
        continue

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

