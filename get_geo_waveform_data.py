# Get the geophone waveform data from SAGE DMC

## Import libraries
from obspy.clients.fdsn import Client
from obspy.io.sac.sactrace import SACTrace
from obspy import read_inventory
from obspy import UTCDateTime

from os import makedirs, walk, rmdir
from os.path import join, exists
from re import search
from glob import glob
import time

from utils_basic import NETWORK, WINDOW_LENGTH_GEO, STARTTIME_GEO, ENDTIME_GEO, CENTER_LONGITUDE, CENTER_LATITUDE, MIN_LONGITUDE, MAX_LONGITUDE, MIN_LATITUDE, MAX_LATITUDE, DELTA

## Inputs
rootdir = "/Volumes/OmanData/data/geophones_new"
write_over = True
length_window = 3600 # in seconds

## Find the start time for each time window
starttime0 = UTCDateTime(STARTTIME_GEO)
endtime0 = UTCDateTime(ENDTIME_GEO)

numwin = round((endtime0 - starttime0) / length_window)
starttimes_twin = []
print("#############################################")
print('In total ',format(numwin, 'd'),' time periods for each station to download')
print("#############################################")
print("\n")
for ind in range(numwin):
	starttime = starttime0 + ind * length_window
	starttimes_twin.append(starttime)
	
## Get the station inventory
print('Reading the station inventory...')
inpath = join(rootdir, "station_metadata.xml")

if not exists(inpath):
	print(f"The station metadata does not exist! Quit!")
	raise

inventory = read_inventory(inpath, format="STATIONXML")
stations = inventory.get_contents()['stations']

for i, station in enumerate(stations):
    station = search(r'\((.*?)\)', station).group(1)
    stations[i] = station

## Get the data for each time window and each station
client = Client("IRIS")
network = NETWORK

## Loop over the time windows
for starttime in starttimes_twin:
    endtime = starttime + length_window
    timestr = starttime.strftime("%Y-%m-%d-%H")
    
    print(f"Getting the data for {timestr}...")
    print("\n")

    ### Loop over the stations
    for station in stations:
        print(f"Getting the data for station {station} at {timestr}...")
        print("\n")

        ### Check if the data already exists
        if not write_over:
            filename = f"{network}.{station}..mseed"
            path = join(rootdir, timestr, filename)

            if exists(path):
                print(f"The data for station {station} at {timestr} already exists! Skip!")
                print("\n")
                continue

        ### Get the waveform data
        try:
            stream = client.get_waveforms(network, station, "*", "GH*", starttime, endtime)
        except Exception as exception:
            print(exception)
            print(f"Failed to get the data for station {station} at {timestr}! Skip!")
            print("\n")
            continue

        ### Check the number of channels
        if len(stream) != 3:
            print(f"Incorrect number of channels! Skip the station {station} at {timestr}!")
            print("\n")
            continue 

        print("Channels acquired:")
        for trace in stream:
            id = trace.id
            print(id)

        ### Reverse the polarity of the vertical component so that positive is upward
        print("Reversing the polarity of the vertical component...")
        for trace in stream:
            if trace.stats.channel == "GHZ":
                trace.data = -trace.data

        ### Save the data
        outdir = join(rootdir, timestr)
        if not exists(outdir):
            makedirs(outdir)
            print(f"Create the directory {outdir}...")

        filename = f"{network}.{station}..mseed"
        outpath = join(outdir, filename)
        stream.write(outpath, format="MSEED")

        print(f"Data saved to {outpath}!")
        print("\n")