"""
Get the station metadata
"""

###
# Import modules
###

from argparse import ArgumentParser
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from os.path import join

args = ArgumentParser

###
# Inputs
###

parser = ArgumentParser(description="Download the metadata for an experiment")
parser.add_argument("--network", type=str, help="Network code")
parser.add_argument("--channels", type=str, nargs="+", help="Channel codes")
parser.add_argument("--starttime", type=str, help="Start time")
parser.add_argument("--endtime", type=str, help="End time")
parser.add_argument("--dirpath_out", type=str, help="Output directory")

parser.add_argument("--client", type=str, help="Client name", default="IRIS")

args = parser.parse_args()
network = args.network
channels = args.channels
starttime = args.starttime
endtime = args.endtime
dirpath_out = args.dirpath_out

client_name = args.client

###
# Define the client
###


client = Client(client_name)
channels = ",".join(channels)

###
# Get the metadata
###
starttime = UTCDateTime(starttime)
endtime = UTCDateTime(endtime)

# Fetch station metadata
print(f"Downloading the station metadata for {network}...")
while True:
    try:
        inventory = client.get_stations(network=network, starttime=starttime, endtime=endtime, channel=channels, level="response")
        break
    except KeyboardInterrupt:
        raise
    except Exception as exception:
        print(exception)
        continue
print("Succeeded!")

###
# Save the metadata
###
outpath = join(dirpath_out, "metadata.xml")
inventory.write(outpath, format="STATIONXML")
print(f"The results are written to {outpath}.")







