# Basic classes and functions for the project

## Import libraries
from os.path import join
from numpy import sqrt, mean, square, amin
from pandas import Timestamp, to_datetime

## Constants
ROOTDIR = "/Volumes/OmanData/geophones_no_prefilt/data"
DELTA = 0.001
CENTER_LONGITUDE = 58.70034
CENTER_LATITUDE =  22.881751

INNER_STATIONS = ["A01", "A02", "A03", "A04", "A05", "A06", "B01", "B02", "B03", "B04", "B06"]
INNER_STATIONS_A = ["A01", "A02", "A03", "A04", "A05", "A06"]
INNER_STATIONS_B = ["B01", "B02", "B03", "B04", "B06"]

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

DAYS_PATH = join(ROOTDIR, "days.csv")
NIGHTS_PATH = join(ROOTDIR, "nights.csv")
STATIONS_PATH = join(ROOTDIR, "stations.csv")
STARTTIME = Timestamp("2020-01-10T00:00:00Z", tz="UTC")
ENDTIME = Timestamp("2020-02-2T23:59:59Z", tz="UTC")
HAMMER_DATE = "2020-01-25"

VELOCITY_UNIT = "nm s$^{-1}$"
DISPLACEMENT_UNIT = "nm"

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