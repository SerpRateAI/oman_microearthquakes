# Basic classes and functions for the project

## Import libraries
from os.path import join
from numpy import datetime64
from numpy import angle, amax, array, exp, log10, mean, sqrt, ndarray
from scipy.stats import gmean
from scipy.signal import hilbert
from pandas import Timestamp, Timedelta, DatetimeIndex
from pandas import Series
from pandas import date_range, to_datetime, read_csv, read_excel, to_timedelta
from obspy import UTCDateTime, read_inventory

## Constants

POWER_FLOOR = 1e-50
NUM_SEONCDS_IN_DAY = 86400
BAR = 1e5 # Average barometric pressure

ROOTDIR_OTHER = "/proj/mazu/tianze.liu/oman/other_observables"
ROOTDIR_GEO = "/proj/mazu/tianze.liu/oman/geophones"
ROOTDIR_BROADBAND = "/proj/mazu/tianze.liu/oman/gfz_data"
ROOTDIR_HYDRO = "/proj/mazu/tianze.liu/oman/hydrophones"
ROOTDIR_MAP = "/proj/mazu/tianze.liu/oman/maps"
SPECTROGRAM_DIR = "/proj/mazu/tianze.liu/oman/spectrograms"
BEAM_DIR = "/proj/mazu/tianze.liu/oman/beams"
PLOTTING_DIR = "/proj/mazu/tianze.liu/oman/plotting"
FIGURE_DIR = "/proj/mazu/tianze.liu/oman/figures"
PICK_DIR = "/proj/mazu/tianze.liu/oman/snuffler_picks"
GFZ_DATA_DIR = "/proj/mazu/tianze.liu/oman/gfz_data"
NORCAL_DATA_DIR = "/proj/mazu/tianze.liu/oman/norcal_data"
DHOFAR_DATA_DIR = "/proj/mazu/tianze.liu/oman/dhofar_data"
IMAGE_DIR = "/proj/mazu/tianze.liu/oman/satellite_images"
MT_DIR = "/proj/mazu/tianze.liu/oman/multitaper"
PHYS_DIR = "/proj/mazu/tianze.liu/oman/physical_models"
VEL_MODEL_DIR = "/proj/mazu/tianze.liu/oman/velocity_models"
LOC_DIR = "/proj/mazu/tianze.liu/oman/locations"
TIME_DIR = "/proj/mazu/tianze.liu/oman/times"

PATH_GEO_METADATA = join(ROOTDIR_GEO, "station_metadata.xml")
PATH_BROADBAND_METADATA = join(ROOTDIR_BROADBAND, "station_metadata.xml")

CENTER_LONGITUDE = 58.70034
CENTER_LATITUDE =  22.881751
MIN_LONGITUDE = 58.6
MAX_LONGITUDE = 58.8
MIN_LATITUDE = 22.8
MAX_LATITUDE = 23.0

SAMPLING_RATE_GEO = 1000.0

NETWORK = "7F"

HYDRO_STATIONS = ["A00", "B00"]
HYDRO_LOCATIONS = {"A00": ["03", "04", "05", "06"], "B00": ["01", "02", "03", "04", "05", "06"]}
HYDRO_DEPTHS = {"01": 30.0, "02": 100.0, "03": 170.0, "04": 240.0, "05": 310.0, "06": 380.0}
                
ALL_COMPONENTS = ["Z", "1", "2", "H"]
GEO_COMPONENTS = ["Z", "1", "2"]
BROADBAND_COMPONENTS = ["Z", "N", "E"]
#PM_COMPONENT_PAIRS = [("2", "1"), ("1", "Z"), ("2", "Z")]
PHASE_DIFF_COMPONENT_PAIRS = [("1", "2"), ("1", "Z"), ("2", "Z")]

GEO_STATIONS_A = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A13", "A14", "A15", "A16", "A17", "A19"]
GEO_STATIONS_B = ["B01", "B02", "B03", "B04", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20"]
GEO_STATIONS = GEO_STATIONS_A + GEO_STATIONS_B

INNER_STATIONS_A = ["A01", "A02", "A03", "A04", "A05", "A06"]
INNER_STATIONS_B = ["B01", "B02", "B03", "B04", "B06"]
INNER_STATIONS = INNER_STATIONS_A + INNER_STATIONS_B

MIDDLE_STATIONS_A = ["A07", "A08", "A09", "A10", "A11"]
MIDDLE_STATIONS_B = ["B07", "B08", "B09", "B10", "B11", "B12"]
MIDDLE_STATIONS = MIDDLE_STATIONS_A + MIDDLE_STATIONS_B

OUTER_STATIONS_A = ["A13", "A14", "A15", "A16", "A17"]
OUTER_STATIONS_B = ["B13", "B14", "B15", "B16", "B17", "B18"]
OUTER_STATIONS = OUTER_STATIONS_A + OUTER_STATIONS_B

CORE_STATIONS = INNER_STATIONS + MIDDLE_STATIONS + OUTER_STATIONS
CORE_STATIONS_A = INNER_STATIONS_A + MIDDLE_STATIONS_A + OUTER_STATIONS_A
CORE_STATIONS_B = INNER_STATIONS_B + MIDDLE_STATIONS_B + OUTER_STATIONS_B

BROKEN_CHANNELS = ["A18.GH1"]
BROKEN_LOCATIONS = ["A00.01", "A00.02"]

EASTMIN_WHOLE = -115
EASTMAX_WHOLE = 100
NORTHMIN_WHOLE = -110
NORTHMAX_WHOLE = 105

EASTMIN_A = -25
EASTMAX_A = 100
NORTHMIN_A = -110
NORTHMAX_A = 15

EASTMIN_B = -115
EASTMAX_B = 65
NORTHMIN_B = 15
NORTHMAX_B = 105

EASTMIN_A_INNER = 15.0
EASTMAX_A_INNER = 35.0
NORTHMIN_A_INNER = -70.0
NORTHMAX_A_INNER = -50.0

EASTMIN_B_INNER = -36.0
EASTMAX_B_INNER = -16.0
NORTHMIN_B_INNER = 52.0
NORTHMAX_B_INNER = 72.0

SAMPLING_RATE = 1000.0

# DAYS_PATH = join(ROOTDIR_GEO, "days.csv")
# NIGHTS_PATH = join(ROOTDIR_GEO, "nights.csv")get_broadband_metadata
SUNRISE_SUNSET_PATH = join(TIME_DIR, "sunrise_sunset_times.csv")
STATIONS_PATH = join(ROOTDIR_GEO, "stations.csv")

WINDOW_LENGTH_GEO = 3600 # in seconds
STARTTIME_GEO = Timestamp("2020-01-10T00:00:00", tz="UTC")
ENDTIME_GEO = Timestamp("2020-02-01T23:59:59", tz="UTC")
STARTTIME_HYDRO = Timestamp("2019-05-01T05:00:00", tz="UTC")
ENDTIME_HYDRO = Timestamp("2020-02-03T09:59:59", tz="UTC")
HAMMER_STARTTIME = Timestamp("2020-01-25T06:00:00", tz="UTC")
HAMMER_ENDTIME = Timestamp("2020-01-25T15:00:00", tz="UTC")

COUNTS_TO_VOLT = 419430 # Divide this number to get from counts to volts for the hydrophone data
DB_VOLT_TO_MPASCAL = -165.0 # Divide this number in dB to get from volts to microPascals

BOREHOLE_DEPTH = 382.0 # Depth of the borehole in meters
WATER_TABLE = 15.0 # Depth to the wavter table in meters
WATER_HEIGHT = BOREHOLE_DEPTH - WATER_TABLE # Height of the water column in meters

## Classes

### Functions to get unique stations from a stream
def get_unique_stations(stream):
    stations = list(set([trace.stats.station for trace in stream]))
    stations.sort()

    return stations

### Function to get the unique locations from a stream
def get_unique_locations(stream):
    locations = list(set([trace.stats.location for trace in stream]))
    locations.sort()

    return locations

### Function to convert from ObsPy UTCDateTime objects to Pandas Timestamp objects
def utcdatetime_to_timestamp(utcdatetime):
    timestamp = to_datetime(utcdatetime.datetime, utc=True)

    return timestamp

### Function to convert from Pandas Timestamp objects to ObsPy UTCDateTime objects
def timestamp_to_utcdatetime(timestamp):
    utcdatetime = UTCDateTime(to_datetime(timestamp, utc = True).to_pydatetime())

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

# Power to decibels
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

# Powers in decibel to amplitude ratios
def powers2amplitude_ratio(power1, power2):
    amp_rat = 10 ** ((power2 - power1) / 20)

    return amp_rat

######
# Functions for loading substantial data files
######

# Get the geophone metadata
def get_geo_metadata():
    inv = read_inventory(PATH_GEO_METADATA)

    return inv

# Get the broadband metadata
def get_broadband_metadata():
    inv = read_inventory(PATH_BROADBAND_METADATA)

    return inv

# Get the geophone station coordinates
def get_geophone_coords():
    inpath = join(LOC_DIR, "geo_stations.csv")
    sta_df = read_csv(inpath, index_col=0)
    sta_df.set_index("name", inplace=True, drop=True)
    
    return sta_df

# Get the delaunay geophone pairs
def get_geophone_pairs():
    inpath = join(LOC_DIR, "delaunay_station_pairs.csv")
    pair_df = read_csv(inpath)

    return pair_df

# Get the delaunay geophone triads
def get_geophone_triads():
    inpath = join(LOC_DIR, "delaunay_station_triads.csv")
    triad_df = read_csv(inpath)

    return triad_df


# Get the indices of the stations forming the delaunay triads
def get_station_triad_indices(coords_df, triad_df):
    ind_mat = []
    for _, row in triad_df.iterrows():
        ind_mat.append([coords_df.index.get_loc(row["station1"]), coords_df.index.get_loc(row["station2"]), coords_df.index.get_loc(row["station3"])])

    ind_mat = array(ind_mat)
    return ind_mat

# Function to get the hydrophone station coordinates
def get_borehole_coords():
    inpath = join(LOC_DIR, "boreholes.csv")
    bh_df = read_csv(inpath, index_col=0)
    bh_df.set_index("name", inplace=True)

    return bh_df

# Get the days of the geophone deployment
def get_geophone_days(timestamp = False):
    inpath = join(ROOTDIR_GEO, "days.csv")
    days_df = read_csv(inpath)
    days = days_df["day"].values

    if timestamp:
        days = to_datetime(days, utc=True)

    return days

# Get the days of the hydrophone deployment
def get_hydrophone_days(timestamp = False):
    inpath = join(ROOTDIR_HYDRO, "hydrophone_days.csv")
    days_df = read_csv(inpath)
    days = days_df["day"].values

    if timestamp:
        days = to_datetime(days, utc=True)

    return days

# Function to get the sunrise and sunset times for the hydrophone deployment
def get_sunrise_sunset_times():
    inpath = SUNRISE_SUNSET_PATH
    times_df = read_csv(inpath, parse_dates=["day", "sunrise", "sunset"])

    return times_df

# Determine if a time is during the day or night
def is_daytime(time, times_df):
    begin_of_day = time.replace(hour=0, minute=0, second=0, microsecond=0)
    # print(begin_of_day)
    # print(type(times_df.loc[begin_of_day, "sunrise"]))
    sunrise = times_df.loc[begin_of_day, "sunrise"]
    sunset = times_df.loc[begin_of_day, "sunset"]

    if time >= sunrise and time <= sunset:
        return True
    else:
        return False

# Get the temperature and barometric data
def get_baro_temp_data():
    inpath = join(ROOTDIR_OTHER, "baro_temp_log.xlsx")
    baro_temp_df = read_excel(inpath)

    # Change the column names
    baro_temp_df.columns = ["time", "elapsed_seconds", "pressure", "temperature"]

    # Set the time zone to Gulf Standard Time
    baro_temp_df["time"] = baro_temp_df["time"].dt.tz_localize("Asia/Muscat")

    # Convet to UTC
    baro_temp_df["time"] = baro_temp_df["time"].dt.tz_convert("UTC")

    # Set the time column as the index
    baro_temp_df.set_index("time", inplace=True)

    return baro_temp_df

# Get the tidal strain data
def get_tidal_strain_data():
    inpath = join(ROOTDIR_OTHER, "earthtide_oman_waves-all_tidalarealstrain.xlsx")
    tidal_strain_df = read_excel(inpath)

    # Change the column names
    tidal_strain_df = tidal_strain_df[["UTC", "Signal [nstr]"]]
    tidal_strain_df.columns = ["time", "strain"]

    # Convert the time column to a Pandas Timestamp object
    tidal_strain_df["time"] = to_datetime(tidal_strain_df["time"], utc=True)

    # Set the time column as the index
    tidal_strain_df.set_index("time", inplace=True)

    return tidal_strain_df

# Get mode order
def get_mode_order(mode_name, base_mode_name = "PR02549", base_mode_order = 2):
    filename = f"stationary_harmonic_series_{base_mode_name}_base{base_mode_order:d}.csv"

    inpath = join(SPECTROGRAM_DIR, filename)
    harmonic_df = read_csv(inpath)

    mode_order = harmonic_df.loc[harmonic_df["mode_name"] == mode_name, "mode_order"].values[0]

    return mode_order

######
# Functions for handling times
######

# Convert a string to a Pandas Timestamp object
def str2timestamp(input):
    if isinstance(input, Timestamp):
        timestamp = input

        return timestamp
    else:
        timestamp = to_datetime(input, utc=True)

    return timestamp


# Convert a Pandas Timestamp object to a ObsPy UTCDateTime object
def timestamp2utcdatetime(timestamp):
    utcdatetime = UTCDateTime(timestamp.to_pydatetime())

    return utcdatetime

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
    if not isinstance(ints, ndarray):
        ints = array(ints)

    datetimes = DatetimeIndex(ints)
    datetimes = datetimes.tz_localize('UTC')

    return datetimes

# Convert an array of relative times in seconds to a Pandas DatetimeIndex objects using a given start time
def reltimes_to_datetimes(reltimes, starttime):
    if not isinstance(starttime, Timestamp):
        try:
            starttime = Timestamp(starttime, tz="UTC")
            starttime = starttime.round("ms") # Round to the closest millisecond
        except:
            raise ValueError("Invalid start time format!")
        
    datetimes = starttime + to_timedelta(reltimes, unit = "s")

    return datetimes

# Assemble a time axis of DateTimeIndex type from integers representing nanoseconds since the Unix epoch
def assemble_timeax_from_ints(starttime, num_time, time_step):
    starttime = to_datetime(starttime, unit='ns', utc=True)
    timeax = date_range(start=starttime, periods=num_time, freq=f'{time_step}ns')

    return timeax

###### Reading CSV files ######
def convert_boolean(value):
    if value.lower() in ['true', 'yes']:
        return True
    elif value.lower() in ['false', 'no']:
        return False
    
    return value

###### Handling file names and paths ######

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
    
    output = timestamp.strftime("%Y%m%d%H%M%S")

    return output

# Convert a day string to a filename suffix
def day2suffix(day):
    suffix = day.replace("-", "")

    return suffix

### Functions for handling data frames
def is_subset_df(df1, df2):
    return df1.merge(df2).shape[0] == df1.shape[0]

# Convert a string to a list
def str2list(input):
    if isinstance(input, str):
        output = [input]
    else:
        output = input

    return output

### Functions for handling angles ###

# Compute the differences between a set of angles while accounting for the periodicity nature
# The input angles are in radians!
def get_angles_diff(angles1, angles2):
    comps1 = exp(1j * angles1)
    comps2 = exp(1j * angles2)

    diffs = comps1 / comps2
    angles_diff = angle(diffs)

    return angles_diff

# Compute the mean of a set of angles while accounting for the periodicity nature
# The input angles are in radians!
def get_angles_mean(angles, axis = 0):
    angle_mean = angle(mean(exp(1j * angles), axis = axis))

    return angle_mean
    
# Compute the standard deviation of a set of angles while accounting for the periodicity nature
# The input angles are in radians!
def get_angles_std(angles, axis = 0):
    angle_mean = get_angles_mean(angles, axis = axis)
    angles_diff = get_angles_diff(angles, angle_mean)
    angle_std = sqrt(mean(angles_diff ** 2, axis = axis))

    return angle_std
