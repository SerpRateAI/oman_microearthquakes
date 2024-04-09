# Basic classes and functions for the project

## Import libraries
from os.path import join
from numpy import amax, log10
from scipy.stats import gmean
from pandas import Timestamp, Timedelta, to_datetime, read_csv
from obspy import UTCDateTime

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

SAMPLING_RATE_GEO = 1000.0


NETWORK = "7F"

HYDRO_STATIONS = ["A00", "B00"]
HYDRO_LOCATIONS = ["01", "02", "03", "04", "05", "06"]

GEO_COMPONENTS = ["Z", "1", "2"]
PM_COMPONENT_PAIRS = [("2", "1"), ("1", "Z"), ("2", "Z")]
WAVELET_COMPONENT_PAIRS = [("1", "2"), ("Z", "1"), ("Z", "2")]

GEO_STATIONS_A = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A13", "A14", "A15", "A16", "A17", "A19"]
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
    stations.sort()

    return stations

### Function to convert from ObsPy UTCDateTime objects to Pandas Timestamp objects
def utcdatetime_to_timestamp(utcdatetime):
    timestamp = to_datetime(utcdatetime.datetime)

    return timestamp

### Function to convert from Pandas Timestamp objects to ObsPy UTCDateTime objects
def timestamp_to_utcdatetime(timestamp):
    utcdatetime = UTCDateTime(to_datetime(timestamp).to_pydatetime())

    return utcdatetime

### Function for converting an array of days since the Unix epoch to an array of Pandas Timestamp objects
def days_to_timestamps(days):
    timestamps = to_datetime(days, unit="D", origin="unix", utc=True)

    return timestamps

### Function for getting the timeax consisting of Pandas Timestamp objects from a Trace object
def get_datetime_axis_from_trace(trace):
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

### Function to convert power to decibels
### If reference is "mean", the reference is the geometric mean of the power values
def power2db(power, reference_type="mean", **kwargs):
    if reference_type == "mean":
        reference = gmean(power, axis=None)
    elif reference_type == "max":
        reference = amax(power)
    elif reference_type == "custom":
        reference = kwargs["reference"]
    else:
        raise ValueError("Invalid reference type!")
    
    power = power / reference
    db = 10 * log10(power)

    return db

### Function to get the geophone station coordinates
def get_geophone_locs():
    inpath = join(ROOTDIR, "geo_stations.csv")
    sta_df = read_csv(inpath, index_col=0)

    return sta_df

### Function to get the hydrophone station coordinates
def get_hydrophone_coords():
    inpath = join(ROOTDIR, "hydro_stations.csv")
    sta_df = read_csv(inpath, index_col=0, dtype = {"location": str})

    return sta_df

### Function to convert seconds to days
def sec2day(seconds):
    days = seconds / 86400

    return days

### Function to convert a time string from input format to filename format
def time2suffix(input):
    if isinstance(input, Timestamp):
        timestamp = input
    elif isinstance(input, str): 
        timestamp = Timestamp(input)
    else:
        raise TypeError("Invalid input type!")
    
    output = timestamp.strftime("%Y%m%dT%H%M%S")

    return output

### Function to generate a suffix for output filename from the input frequency limits
def freq2suffix(freqmin, freqmax):
    if freqmin is not None and freqmax is not None:
        suffix = f"bp{freqmin:.0f}-{freqmax:.0f}hz"
    elif freqmin is not None and freqmax is None:
        suffix = f"hp{freqmin:.0f}hz"
    elif freqmin is None and freqmax is not None:
        suffix = f"lp{freqmax:.0f}hz"
    else:
        suffix = "no_filter"

    return suffix

### Function to generate a suffix for output filename from the normalization option
def norm2suffix(normalize):
    if normalize:
        suffix = "normalized"
    else:
        suffix = "no_norm"
    
    return suffix
    


