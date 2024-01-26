# Get inventory data for a network in a time period

from obspy.clients.fdsn import Client
from obspy import read_inventory, UTCDateTime

from os.path import join, exists

# The network name
ntwknm = '7F'

# The client name
clntnm = 'IRIS'

# The output directory and file names
dirnm_in = '/Volumes/OmanData/geophones_no_prefilt/data'
dirnm_out = '/Volumes/OmanData/geophones_no_prefilt/data'

# The latitude and longitude limits
stla_min = 22.7
stla_max = 23.2
stlo_min = 58.5
stlo_max = 59

# Get the time range
print('Reading the epoch table...')
path_in = join(dirnm_in, 'TimeRange.dat')
with open(path_in, 'r') as fp:
    lines = fp.readlines()

line = lines[0]
fields = line.split()
btime = fields[0]
etime = fields[1].rstrip()

filenm_out = 'Inventory_'+ntwknm+'_'+btime+'_'+etime+'.xml'

# Get the inventory from the data center
print('Getting the inventory for '+ntwknm+' from '+clntnm+' for the period '+btime+' to '+etime+'...')

client = Client(clntnm)

btime = UTCDateTime(btime)
etime = UTCDateTime(etime)

n = 0
while True:
    try:
        inv = client.get_stations(starttime=btime, endtime=etime, network=ntwknm, channel='GH*', level='response', minlatitude=stla_min, maxlatitude=stla_max, minlongitude=stlo_min, maxlongitude=stlo_max)
        n = n+1
        print('The '+str(n)+'th attempt succeed. Writing the inventory to file...')
        break
    except KeyboardInterrupt:
        raise
    except Exception as e:
        n = n+1
        print(e)

# Write the inventory to file
path_out = join(dirnm_out, filenm_out)
inv.write(path_out, 'STATIONXML')

print(path_out+' is written.')

    
