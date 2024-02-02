# Basic classes and functions for the project

## Import libraries
from os.path import join
from numpy import sqrt, mean, square, amin
from pandas import Timestamp, to_datetime

## Constants
ROOTDIR = "/Volumes/OmanData/geophones_no_prefilt/data"
INNER_STATIONS = ["A01", "A02", "A03", "A04", "A05", "A06", "B01", "B02", "B03", "B04", "B05", "B06"]
DAYS_PATH = join(ROOTDIR, "days.csv")
NIGHTS_PATH = join(ROOTDIR, "nights.csv")
STATIONS_PATH = join(ROOTDIR, "stations.csv")
STARTTIME = Timestamp("2020-01-10T00:00:00Z", tz="UTC")
ENDTIME = Timestamp("2020-02-2T23:59:59Z", tz="UTC")


## Classes

## Functions

### Function for converting an array of days since the Unix epoch to an array of Pandas Timestamp objects
def days_to_timestamps(days):
    timestamps = to_datetime(days, unit="D", origin="unix", utc=True)

    return timestamps

### Function to convert UTC time to local time
def utc_to_local(utc_time, timezone="Asia/Muscat"):
    local_time = utc_time.tz_convert(timezone)

    return local_time

### Function to convert local time to UTC time
def local_to_utc(local_time):
    utc_time = local_time.tz_convert("UTC")

    return utc_time