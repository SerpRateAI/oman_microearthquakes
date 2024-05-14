# Basic classes and functions for the project

## Import libraries
from os.path import join
from numpy import amax, array, log10
from scipy.stats import gmean
from pandas import Timestamp, Timedelta, DatetimeIndex
from pandas import Series
from pandas import date_range, to_datetime, read_csv
from obspy import UTCDateTime, read_inventory

## Constants

POWER_FLOOR = 1e-50

ROOTDIR_GEO = "/fp/projects01/ec332/data/geophones"
ROOTDIR_HYDRO = "/fp/projects01/ec332/data/hydrophones"
ROOTDIR_HAMMER = "/Volumes/OmanData/data/hammer"
SPECTROGRAM_DIR = "/fp/projects01/ec332/data/spectrograms"
FIGURE_DIR = "/fp/projects01/ec332/data/figures"

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
BROKEN_LOCATIONS = ["A00.01", "A00.02"]

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
STARTTIME_GEO = Timestamp("2020-01-10T00:00:00")
ENDTIME_GEO = Timestamp("2020-02-01T23:59:59")
STARTTIME_HYDRO = Timestamp("2019-05-01T05:00:00")
ENDTIME_HYDRO = Timestamp("2020-02-03T09:59:59")
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

### Function to convert UTC time to local time
def utc_to_local(utc_time, timezone="Asia/Muscat"):
    local_time = utc_time.tz_convert(timezone)

    return local_time

### Function to convert local time to UTC time
def local_to_utc(local_time):
    utc_time = local_time.tz_convert("UTC")

    return utc_time

### Function to convert power to decibels
def power2db(power, reference_type=None, **kwargs):
    if reference_type is None:
        reference = 1.0
    elif reference_type == "mean":
        reference = gmean(power, axis=None)
    elif reference_type == "max":
        reference = amax(power)
    elif reference_type == "custom":
        reference = kwargs["reference"]
    else:
        raise ValueError("Invalid reference type!")
    
    power = power / reference
    power[ power < POWER_FLOOR ] = POWER_FLOOR
    db = 10 * log10(power)

    return db

######
# Functions for loading substantial data files
######

# Get the geophone metadata
def get_geo_metadata():
    inv = read_inventory(PATH_GEO_METADATA)

    return inv
    
# Get the geophone station coordinates
def get_geophone_coords():
    inpath = join(ROOTDIR, "geo_stations.csv")
    sta_df = read_csv(inpath, index_col=0)

    return sta_df

### Function to get the hydrophone station coordinates
def get_hydrophone_coords():
    inpath = join(ROOTDIR, "hydro_stations.csv")
    sta_df = read_csv(inpath, index_col=0, dtype = {"location": str})

    return sta_df

### Function to get the days of the geophone deployment
def get_geophone_days():
    inpath = join(ROOTDIR_GEO, "days.csv")
    days_df = read_csv(inpath)
    days = days_df["day"].values

    return days

######
# Functions for handling times
######

# Convert seconds to days
def sec2day(seconds):
    days = seconds / 86400

    return days

# Convert days to seconds
def day2sec(days):
    seconds = days * 86400

    return seconds

# Convert hours to seconds
def hour2sec(hours):
    seconds = hours * 3600

    return seconds

# Get the begin and end of a day
def get_day_begin_and_end(day):
    if not isinstance(day, Timestamp):
        day = Timestamp(day)

    starttime = day.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    endtime = day.replace(hour = 23, minute = 59, second = 59, microsecond = 999999)

    return starttime, endtime

# Function to convert year-month-day to day of the year
def to_day_of_year(date):
    if not isinstance(date, Timestamp):
        if isinstance(date, str):
            date = Timestamp(date)
        else:
            raise TypeError("Invalid input type!")

    day_of_year = str(date.day_of_year).zfill(3)

    return day_of_year

# Function to convert DateTimeIndex objects to a list of integers representing nanoseconds since the Unix epoch
def datetime2int(datetimes):
    if not isinstance(datetimes, DatetimeIndex):
        if isinstance(datetimes, list):
            datetimes = DatetimeIndex(datetimes)
        else:
            raise TypeError("Invalid input type!")

    datetimes = datetimes.astype('int64')
    datetimes = datetimes.to_numpy()

    return datetimes

# Function to convert a list of integers representing nanoseconds since the Unix epoch to a DateTimeIndex object
def int2datetime(ints):
    if not isinstance(ints, array):
        ints = array(ints)

    datetimes = DatetimeIndex(ints)

    return datetimes

# Convert an array of relative times in seconds to a Pandas DatetimeIndex objects using a given start time
def reltimes_to_timestamps(reltimes, starttime):
    if not isinstance(starttime, Timestamp):
        try:
            starttime = Timestamp(starttime, tz="UTC")
            starttime = starttime.round("ms") # Round to the closest millisecond
        except:
            raise ValueError("Invalid start time format!")
        
    timestamps = [starttime + Timedelta(seconds=reltime) for reltime in reltimes]
    timestamps = DatetimeIndex(timestamps)

    return timestamps

# Assemble a time axis of DateTimeIndex type from integers representing nanoseconds since the Unix epoch
def assemble_timeax_from_ints(starttime, num_time, time_step):
    starttime = to_datetime(starttime, unit='ns') 
    timeax = date_range(start=starttime, periods=num_time, freq=f'{time_step}ns')

    return timeax

######
# Functions for handling file names
######

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

# Convert a time string from input format to filename format
def time2suffix(input):
    if isinstance(input, Timestamp):
        timestamp = input
    elif isinstance(input, str): 
        timestamp = Timestamp(input)
    else:
        raise TypeError("Invalid input type!")
    
    output = timestamp.strftime("%Y%m%dT%H%M%S")

    return output

# Convert a day string to a filename suffix
def day2suffix(day):
    suffix = day.replace("-", "")

    return suffix

    


