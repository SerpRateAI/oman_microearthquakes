# Basic classes and functions for the project

## Import libraries
from os.path import join
from numpy import sqrt, mean, square, amin
from pandas import Timestamp, Timedelta, to_datetime, read_csv

## Constants

ROOTDIR = "/Volumes/OmanData/data"
ROOTDIR_GEO = "/Volumes/OmanData/data/geophones"
ROOTDIR_HYDRO = "/Volumes/OmanData/data/hydrophones"
ROOTDIR_HAMMER = "/Volumes/OmanData/data/hammer"
PATH_GEO_METADATA = join(ROOTDIR_GEO, "station_metadata.xml")

CENTER_LONGITUDE = 58.70034
CENTER_LATITUDE =  22.881751
MIN_LONGITUDE = 58.6
MAX_LONGITUDE = 58.8
MIN_LATITUDE = 22.8
MAX_LATITUDE = 23.0

DELTA = 0.001

NETWORK = "7F"

HYDRO_STATIONS = ["A00", "B00"]
HYDRO_LOCATIONS = ["01", "02", "03", "04", "05", "06"]

GEO_COMPONENTS = ["Z", "1", "2"]
GEO_STATIONS_A = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19"]
GEO_STATIONS_B = ["B01", "B02", "B03", "B04", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20"]
GEO_STATIONS = GEO_STATIONS_A + GEO_STATIONS_B

INNER_STATIONS_A = ["A01", "A02", "A03", "A04", "A05", "A06"]
INNER_STATIONS_B = ["B01", "B02", "B03", "B04", "B06"]
INNER_STATIONS = INNER_STATIONS_A + INNER_STATIONS_B

BROKEN_CHANNELS = ["A18.GH1"]

EASTMIN_WHOLE = -115
EASTMAX_WHOLE = 65
NORTHMIN_WHOLE = -100
NORTHMAX_WHOLE = 105

EASTMIN_A = -20
EASTMAX_A = 65
NORTHMIN_A = -100
NORTHMAX_A = 15

EASTMIN_B = -115
EASTMAX_B = 65
NORTHMIN_B = 15
NORTHMAX_B = 105

DAYS_PATH = join(ROOTDIR_GEO, "days.csv")
NIGHTS_PATH = join(ROOTDIR_GEO, "nights.csv")
STATIONS_PATH = join(ROOTDIR_GEO, "stations.csv")

WINDOW_LENGTH_GEO = 3600 # in seconds
STARTTIME_GEO = Timestamp("2020-01-10T03:00:00Z", tz="UTC")
ENDTIME_GEO = Timestamp("2020-02-02T23:59:59Z", tz="UTC")
STARTTIME_HYDRO = Timestamp("2019-05-01T05:00:00Z", tz="UTC")
ENDTIME_HYDRO = Timestamp("2020-02-03T09:59:59Z", tz="UTC")
HAMMER_DATE = "2020-01-25"

COUNTS_TO_VOLT = 419430 # Divide this number to get from counts to volts for the hydrophone data
DB_VOLT_TO_MPASCAL = -165.0 # Divide this number in dB to get from volts to microPascals

## Classes

### Functions to get unique stations from a stream
def get_unique_stations(stream):
    stations = list(set([trace.stats.station for trace in stream]))

    return stations

### Function for converting an array of days since the Unix epoch to an array of Pandas Timestamp objects
def days_to_timestamps(days):
    timestamps = to_datetime(days, unit="D", origin="unix", utc=True)

    return timestamps

### Function for getting the timeax consisting of Pandas Timestamp objects from a Trace object
def get_timeax_from_trace(trace):
    timeax = trace.times("matplotlib")
    timeax = days_to_timestamps(timeax)

    return timeax

### Funcion to convert an array of relative times in seconds to an array of Pandas Timestamp objects using a given start time
def reltimes_to_timestamps(reltimes, starttime):
    if not isinstance(starttime, Timestamp):
        try:
            starttime = Timestamp(starttime, tz="UTC")
        except:
            raise ValueError("Invalid start time format!")
        
    timestamps = [starttime + Timedelta(seconds=reltime) for reltime in reltimes]

    return timestamps

### Function to convert UTC time to local time
def utc_to_local(utc_time, timezone="Asia/Muscat"):
    local_time = utc_time.tz_convert(timezone)

    return local_time

### Function to convert local time to UTC time
def local_to_utc(local_time):
    utc_time = local_time.tz_convert("UTC")

    return utc_time

### Function to get the geophone station locations
def get_geophone_locs():
    inpath = join(ROOTDIR, "stations.csv")
    stadf = read_csv(inpath, index_col=0)

    return stadf

### Function for saving a figure
def save_figure(fig, filename, outdir=ROOTDIR, dpi=300):
    fig.patch.set_alpha(0)

    outpath = join(outdir, filename)

    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {outpath}")