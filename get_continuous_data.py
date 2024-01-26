from obspy.clients.fdsn import Client
from obspy.io.sac.sactrace import SACTrace
from obspy import read_inventory
from obspy import UTCDateTime

from os import makedirs, walk, rmdir
from os.path import join, exists
import time

# Network name
ntwknm = '7F'

# Client name
clntnm = 'IRIS'

# Input and output directory name
dirnm_in = '/Volumes/OmanData/geophones_no_prefilt/data'
dirnm_out = '/Volumes/OmanData/geophones_no_prefilt/data'

# Time window length for each file in second
winlen = 3600

# Parameters for removing instrument response
taper_fraction = 0.001

# Formats of the input files
invfmt = 'STATIONXML'

# # Pause time in secs after downloading each event
# t_ps_st = 30
# t_ps_epc = 600

client = Client(clntnm)

# Get the time range
print('Reading the time-range file...')
path_in = join(dirnm_in, 'TimeRange.dat')
with open(path_in) as fp:
	lines = fp.readlines()

line = lines[0]

fields = line.split()
btime_str = fields[0]
etime_str = fields[1].rstrip()


# Compute the time period for each SAC file
btime = UTCDateTime(btime_str)
etime = UTCDateTime(etime_str)

numwin = round((etime-btime)/winlen)
list_bwin = []
print('In total ',format(numwin, 'd'),' time periods for each station to download')
for ind in range(numwin):
	btime_win = btime+ind*winlen
	list_bwin.append(btime_win)

# Print the openning remarks
t = time.localtime()
time_now = time.strftime("%D:%H:%M:%S", t)
print(time_now)
print('Downloading data for '+ntwknm+' for the time range '+btime_str+' to '+etime_str+' from '+clntnm+'...')

# Get the station inventory
print('Reading the station inventory...')
invnm = 'Inventory_'+ntwknm+'_'+btime_str+'_'+etime_str+'.xml'
path_inv = join(dirnm_in, invnm)

if not exists(path_inv):
	print(path_inv+' does not exist! Quit.')
	raise

inv = read_inventory(path_inv, format=invfmt)
cont = inv.get_contents()
list_seedid = cont['channels']

# Extract station names from the inventory
list_stnm = []
for seedid in list_seedid:
	fields = seedid.split('.')
	stnm = fields[1]
	list_stnm.append(stnm)

# Remove duplicating station names
list_stnm = list(dict.fromkeys(list_stnm));

numst = len(list_stnm)
print('In total ',format(numst,'d'),' stations to download')

# Get the waveforms for each event in the epoch
for btime_win in list_bwin:
		
	# Start of the clock
	start = time.time()

	# Time-window name
	twinnm = btime_win.strftime('%Y-%m-%d-%H-%M-%S')
	print(twinnm)

	# Find the list of stations already downloaded in the event folder and make the folder if it does not exist
	list_stnm_pre = []
	dirnm_twin = join(dirnm_out, twinnm)

	if exists(dirnm_twin):
		for root, dirs, files in walk(dirnm_twin):
			for filenm in files:	
				if '.SAC' in filenm:
					fields = filenm.split('.')
				stnm = fields[1]
				list_stnm_pre.append(stnm)
	else:
		makedirs(dirnm_twin)
		print(dirnm_twin+' is created.')

	# Get the waveforms for each station that recorded the time window
	for stnm in list_stnm:
		# Print the station name
		print('Acquiring data for '+stnm+'...')

		# Check if the station is already downloaded
		if stnm in list_stnm_pre:
			print(stnm+' is already downloaded. Skipped.')
			continue

		list_seedid_z = [seed_id for seed_id in list_seedid if stnm in seed_id and 'GHZ' in seed_id]
		seedid_z = list_seedid_z[0]

		# Get the station coordinate
		try:
			print('Looking for channel metadata for '+seedid_z+'...')
			coord = inv.get_coordinates(seedid_z, btime_win)
			print('Channel metadata found.')
		except Exception as err:
			print(err)
			print('No coordinates information found for '+stnm+' Skipped.')
			continue

		stlo = coord['longitude']
		stla = coord['latitude']
		stel = coord['elevation']

		# Get the waveforms for the station
		print('Acquiring the waveform data...')
		try:
			etime_win = btime_win+winlen-1
			stream = client.get_waveforms(ntwknm, stnm, '*', 'GH*', btime_win, etime_win)
			print('Waveform data found.')
		except (KeyboardInterrupt, SystemExit):
			raise
		except Exception as err:
			print(str(err))
			continue

		numtrs = stream.count()

		if numtrs < 3:
			print('Abnormal trace number. Skipped')
			continue

		# Remove instrument response
		try:
			stream = stream.remove_response(inventory=inv, taper_fraction=taper_fraction)
			print('Removing the instrument response...')
		except (KeyboardInterrupt, SystemExit):
			raise
		except Exception as err:
			print(str(err))
			continue


		# Save traces to SAC files
		print('Writing SAC files...')
		for trace in stream:
			chnm = trace.stats.channel
			lcnm = trace.stats.location
			btime_trc = trace.stats.starttime
			seed_id_trc = ntwknm+'.'+stnm+'.'+lcnm+'.'+chnm

			# Get the component orientations
			ori = inv.get_orientation(seed_id_trc, btime_trc)
			cmpaz = ori['azimuth']

			# The original conversion was wrong for the vertical component. Fixed by Tianze Liu on 2023-12-18
			cmpinc = ori['dip']+90 # Dip is measured from horizontal downward, whereas incident is measured from vertical upward!
			
			# Change the data unit from m/s to nm/s and the unit name
			# Added by Tianze Liu on 2023-11-27
			trace.data = trace.data * 1e9
	
			# Set SAC headers
			sac = SACTrace.from_obspy_trace(trace)
			sac.stlo = stlo
			sac.stla = stla
			sac.stel = stel
			sac.nzyear = btime_win.year
			sac.nzjday = btime_win.julday
			sac.nzhour = btime_win.hour
			sac.nzmin = btime_win.minute
			sac.nzsec = btime_win.second
			sac.nzmsec = btime_win.microsecond/1000
			sac.o = 0
			sac.b = 0
			sac.idep = 'ivel'
			sac.iztype = 'io'
			sac.cmpaz = cmpaz
			sac.cmpinc = cmpinc

			# Write SAC file
			sacnm = ntwknm+'.'+stnm+'.'+twinnm+'.'+lcnm+'.'+chnm+'.SAC'
			path_sac = join(dirnm_twin, sacnm)
			sac.write(path_sac)
			print(path_sac+' is written')

	# Make the SAC file and station list
	list_stnm_sac = []

	for root, dirs, files in walk(dirnm_twin):
		for filenm in files:
			if 'GHZ' in filenm:
				fields = filenm.split('.')
				stnm = fields[1]
				list_stnm_sac.append(stnm)

	if not list_stnm_sac:
		rmdir(dirnm_twin)
		print('No data for '+twinnm+'. The folder is deleted.')
	else:
		path_lst = join(dirnm_twin, 'stations.lst')
		with open(path_lst, 'w') as fp:
			fp.writelines('%s\n' % stnm for stnm in list_stnm_sac)

	# Stop the clock
	end = time.time()
	print('It took '+str(end-start)+' s to download the data for '+twinnm)
