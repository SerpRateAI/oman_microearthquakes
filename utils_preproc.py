# Function and classes for reading and preprocessing waveforms

from os.path import join
from numpy import abs, array, mean, amax, amin, argmax
from numpy.linalg import norm
from scipy.stats import linregress
from scipy.signal import find_peaks
from obspy import read, read_inventory, UTCDateTime, Stream
from obspy.signal.cross_correlation import correlate_template
from pandas import to_datetime, Timestamp, Timedelta, DataFrame

from utils_cc import TemplateEventWaveforms, TemplateEvent, Matches, MatchedEvent, MatchWaveforms, MatchedEventWaveforms
from utils_basic import COUNTS_TO_VOLT, DB_VOLT_TO_MPASCAL, ROOTDIR_GEO, ROOTDIR_HYDRO, PATH_GEO_METADATA, GEO_STATIONS, GEO_COMPONENTS, HYDRO_STATIONS, HYDRO_LOCATIONS, BROKEN_CHANNELS, BROKEN_LOCATIONS

from utils_basic import get_geo_metadata, timestamp_to_utcdatetime, to_day_of_year

# Read and preprocess day-long geophone waveforms
def read_and_process_day_long_geo_waveforms(day, metadat = None, freqmin=None, freqmax=None, stations=None, components=None, zerophase=False, corners=4, normalize=False, decimate=False, decimate_factor=None, all_components=True):
    # Check if metadat is specified
    if metadat is None:
        metadat = get_geo_metadata()

    # Read the waveforms
    print(f"Reading the waveforms for {day}")
    stations_to_read = get_geo_stations_to_read(stations)
    channels_to_read = get_geo_channels_to_read(components)

    stream_in = Stream()
    for station in stations_to_read:
        stream_station = Stream()

        for channel in channels_to_read:
            stream = read_day_long_geo_waveforms(day, station, channel)
            
            if stream is None:
                print(f"Warning: No data found for {station}.{channel}!")
                if all_components:
                    break
                else:
                    continue
            else:
                stream_station += stream
        
        if len(stream_station) == len(channels_to_read):
            stream_in += stream_station
        else:
            print(f"Warning: Not all components read for {station}! The station is skipped.")
            continue

    # Process the waveforms
    print("Preprocessing the waveforms...")
    if len(stream_in) < 3 and all_components:
        return None
    else:
        stream_proc = preprocess_geo_stream(stream_in, metadat, freqmin, freqmax, zerophase=zerophase, corners=corners, normalize=normalize, decimate=decimate, decimate_factor=decimate_factor)
        return stream_proc

# Read and preprocess day-long hydrophone waveforms
def read_and_process_day_long_hydro_waveforms(day, freqmin=None, freqmax=None, stations=None, locations=None, zerophase=False, corners=4, normalize=False, decimate=False, decimate_factor=None):

    # Read the waveforms
    print(f"Reading the waveforms for {day}")
    stations_to_read = get_hydro_stations_to_read(stations)
    locations_to_read = get_hydro_locations_to_read(locations)

    stream_in = Stream()
    for station in stations_to_read:
        stream_station = Stream()

        for location in locations_to_read:
            stream = read_day_long_hydro_waveforms(day, station, location)
            
            if stream is None:
                print(f"Warning: No data is read for {station}.{location}!")
                continue
            else:
                stream_station += stream

        stream_in += stream_station

    # Process the waveforms
    print("Preprocessing the waveforms...")
    stream_proc = preprocess_hydro_stream(stream_in, freqmin, freqmax, zerophase=zerophase, corners=corners, normalize=normalize, decimate=decimate, decimate_factor=decimate_factor)

    return stream_proc

## Read and preprocess geophone waveforms in a time window
## Window length is in second
def read_and_process_windowed_geo_waveforms(starttime, metdat, endtime = None, dur = None, freqmin=None, freqmax=None, stations=None, components=None, zerophase=False, corners=4, normalize=False, decimate=False, decimate_factor=10, all_components=True):

    # Check if endtime and dur are specified at the same time
    if endtime is not None and dur is not None:
        raise ValueError("Error: endtime and dur cannot be specified at the same time!")
    
    if not isinstance(starttime, Timestamp):
        if type(starttime) is str:
            starttime = Timestamp(starttime, tz="UTC")
        else:
            raise TypeError("Error: starttime must be a string or Pandas Timestamp object!")

    # Define the time window
    if endtime is not None:
        if not isinstance(endtime, Timestamp):
            if type(endtime) is str:
                endtime = Timestamp(endtime, tz="UTC")
            else:
                raise TypeError("Error: endtime must be a string or Pandas Timestamp object!")
    elif dur is not None:
        if dur is None:
            raise ValueError("Error: Either endtime or dur must be specified!")
        else:
            endtime = starttime + Timedelta(seconds=dur)


    ### Read the waveforms
    stations_to_read = get_geo_stations_to_read(stations)
    channels_to_read = get_geo_channels_to_read(components)

    stream_in = Stream()
    for station in stations_to_read:
        stream_station = Stream()

        for channel in channels_to_read:
            stream = read_geo_waveforms_in_timewindow(starttime, endtime, station, channel)
            
            if stream is None:
                print(f"Warning: No data found for {station}.{channel}!")
                if all_components:
                    break
                else:
                    continue
            else:
                stream_station += stream
        
        if len(stream_station) == len(channels_to_read):
            stream_in += stream_station
        else:
            print(f"Warning: Not all components read for {station}! The station is skipped.")
            continue

    ### Process the waveforms
    if len(stream_in) < 3 and all_components:
        return None
    else:
        stream_proc = preprocess_geo_stream(stream_in, metdat, freqmin, freqmax, zerophase=zerophase, corners=corners, normalize=normalize, decimate=decimate, decimate_factor=decimate_factor)
        return stream_proc

## Read and preprocess hydrophone waveforms in a time window
## Window length is in second
def read_and_process_windowed_hydro_waveforms(starttime, dur, freqmin=None, freqmax=None, stations=None, locations=None, zerophase=False, corners=4, normalize=False, decimate=False, decimate_factor=10):
    if not isinstance(starttime, Timestamp):
        if type(starttime) is str:
            starttime = Timestamp(starttime, tz="UTC")
        else:   
            raise TypeError("Error: starttime must be a Pandas Timestamp object!")
    
    ### Read the waveforms
    stations_to_read = get_hydro_stations_to_read(stations)
    locations_to_read = get_hydro_locations_to_read(locations)

    stream_in = Stream()
    endtime = starttime + Timedelta(seconds=dur)
    for station in stations_to_read:
        for location in locations_to_read:
            stream = read_hydro_waveforms_in_timewindow(starttime, endtime, station, location)

            if stream is not None:
                stream_in += stream

    ### Process the waveforms
    stream_proc = preprocess_hydro_stream(stream_in, freqmin, freqmax, zerophase=zerophase, corners=corners, normalize=normalize, decimate=decimate, decimate_factor=decimate_factor)

    return stream_proc
    

## Read and preprocess waveforms in a set of defined time windows
def read_and_process_picked_geo_waveforms(pickdf, freqmin=None, freqmax=None, stations=None, components=None, begin=-0.01, end=0.2, reference="common", zerophase=False, corners=4, normalize=False, decimate=False, decimate_factor=10):
    
    ### Check if pickdf is a pandas dataframe
    if not isinstance(pickdf, DataFrame):
        raise TypeError("Error: pickdf must be a pandas DataFrame object!")

    ### Read the waveforms
    stations_to_read = get_geo_stations_to_read(stations)
    channels_to_read = get_geo_channels_to_read(components)

    stream_in = Stream()
    for row in pickdf.iterrows():
        station = row["station"]

        if station in stations_to_read:
            for channel in channels_to_read:
                
                if reference == "individual":
                    picktime = row["time"]
                    starttime = picktime + Timedelta(seconds=begin)
                    endtime = picktime + Timedelta(seconds=end)
                elif reference == "common":
                    starttime = pickdf["time"].min() + Timedelta(seconds=begin)
                    endtime = pickdf["time"].min() + Timedelta(seconds=end)
                else:
                    raise ValueError("Error: reference must be either 'individual' or 'common'!")

                stream = read_geo_waveforms_in_timewindow(starttime, endtime, station, channel)
                if stream is not None:
                    stream_in += stream
    
    ### Process the waveforms
    stream_proc = preprocess_geo_stream(stream_in, freqmin, freqmax, zerophase=zerophase, corners=corners, normalize=normalize, decimate=decimate, decimate_factor=decimate_factor)
    
    return stream_proc


## Read and process the template waveforms
# def read_and_process_template_waveforms(template, freqmin, freqmax, rootdir=ROOTDIR_GEO, stations=None, components=None, begin=0.01, end=0.2, reference="common", taper=0.001, zerophase=False, corners=4):

#     ### Check if template is a Template object
#     if not isinstance(template, Template):
#         raise TypeError("Error: template must be a Template object!")
    
#     ### Find the day of the starttime
#     day = template.first_start_time.strftime("%Y-%m-%d")

#     ### Get the list of components to read
#     if components is None:
#         components_to_read = GEO_COMPONENTS
#     else:
#         if type(components) is not list:
#             if type(components) is str:
#                 components_to_read = [components]
#             else:
#                 raise TypeError("Error: component must be a list of strings!")
#         else:
#             components_to_read = components

#     ### Get the list of stations to read
#     if stations is None:
#         stations = GEO_STATIONS
#     else:
#         if type(stations) is not list:
#             if type(stations) is str:
#                 stations_to_read = [stations]
#             else:
#                 raise TypeError("Error: stations must be a list of strings!")
#         else:
#             stations_to_read = stations

#     stream_in = Stream()
#     for station in template.stations:
#         if station in stations_to_read:
#             if reference == "common":
#                 starttime = template.first_start_time + Timedelta(seconds=begin)
#                 endtime = template.first_start_time + Timedelta(seconds=end)
#             elif reference == "individual":
#                 starttime = template.get_start_time_for_station(station)
#                 starttime = starttime + begin
#                 endtime = starttime + end
#             else:
#                 raise ValueError("Error: reference must be either 'individual' or 'common'!")
            
#             starttime = UTCDateTime(to_datetime(starttime).to_pydatetime()) # Convert to python datetime object first and then to UTCDateTime object
#             endtime = UTCDateTime(to_datetime(endtime).to_pydatetime()) # Convert to python datetime object first and then to UTCDateTime object    starttime = UTCDateTime(to_datetime(starttime).to_pydatetime())
#             for component in components_to_read:
#                 pattern = join(rootdir, day, f"*{station}*GH{component}.mseed")
#                 try:
#                     stream_in += read(pattern, starttime=starttime, endtime=endtime)
#                 except FileNotFoundError:
#                     print(f"Warning: {pattern} not found!")
#                     continue

#                 stream_in += stream

#     ### Process the waveforms
#     stream_proc = preprocess_stream(stream_in, freqmin, freqmax, taper=taper, zerophase=zerophase, corners=corners)

#     ### Trim the waveforms
#     stream_trim = Stream()
#     for station in stations_active:
#         if reference == "individual":
#             starttime =  UTCDateTime(template.get_start_time_for_station(station))
#         elif reference == "common":
#             starttime = starttime0
#         else:
#             raise ValueError("Error: reference must be either 'individual' or 'common'!")

#         starttime_trim = starttime + begin
#         endtime_trim = starttime + end
#         stream = stream_proc.select(station=station)
#         stream.trim(starttime_trim, endtime_trim)
#         stream_trim += stream

#     ### Store the data in a TemplateEventWaveforms object
#     tempwaveforms = TemplateEventWaveforms(template, stations_active, components_active, stream_trim)

#     return tempwaveforms

# ## Read and process the waveforms of a list of matches
# def read_and_process_match_waveforms(matches, freqmin, freqmax, ROOTDIR_GEO=ROOTDIR_GEO, names=None, stations=None, components=None, begin=0.01, end=0.2, reference="individual", taper=0.001, zerophase=False, corners=4):
    
#     ### Time window for reading the long waveforms for preprocessing
#     begin_read = begin - 1
#     end_read = end + 3

#     ### Determine if matches is a MatchWaveforms object
#     if type(matches) is not Matches:
#         raise TypeError("Error: matches must be a Matches object!")

#     ### Keep only the matches with the specified names
#     if names is not None:
#         matches_active = matches.get_matches_by_names(names)
#     else:
#         matches_active = matches

#     ### Determine the components to be processed
#     if components is None:
#             components_active = ["Z", "1", "2"]
#     else:
#         if type(components) is not list:
#             if type(components) is str:
#                 components_active = [components]
#             else:
#                 print("Error: components must be a list of strings!")
#                 raise TypeError
#         else:
#             components_active = components

#     ### Process each matched event
#     match_waveforms = MatchWaveforms()
#     for i, match in enumerate(matches_active):
#         ### Determine the stations to be processed
#         if stations is None:
#             stations_in  = match.stations
#         else:
#             if type(stations) is not list:
#                 if type(stations) is str:
#                     stations = [stations]
#                 else:
#                     print("Error: stations must be a list of strings!")
#                     raise TypeError
            
#             stations_in = list(set(stations) & set(match.stations))

#         #### Read the long waveforms
#         starttime0 = UTCDateTime(match.first_start_time)
#         timewin = starttime0.strftime("%Y-%m-%d-%H-00-00")

#         stream_in = Stream()
#         for station in stations_in:
            
#             for component in components_active:
#                 pattern = join(ROOTDIR_GEO, timewin, f"*{station}*{component}.SAC")
#                 try:
#                     stream_in += read(pattern, starttime=starttime0+begin_read, endtime=starttime0+end_read)
#                 except FileNotFoundError:
#                     print(f"Warning: {pattern} not found!")
#                     continue

#         ### Process the waveforms
#         stream_proc = preprocess_stream(stream_in, freqmin, freqmax, taper=taper, zerophase=zerophase, corners=corners)

#         ### Trim the waveforms
#         stream_trim = Stream()
#         for station in stations_in:

#             if reference == "individual":
#                 starttime = match.get_info_for_station(station, entries=["start_time"])["start_time"]
#                 starttime_trim = UTCDateTime(starttime) + begin
#                 endtime_trim = UTCDateTime(starttime) + end
#             elif reference == "common":
#                 starttime_trim = starttime0 + begin
#                 endtime_trim = starttime0 + end

#             stream = stream_proc.select(station=station)
#             stream.trim(starttime_trim, endtime_trim)
#             stream_trim += stream

#         ### Store the data in a MatchedEventWaveforms object and append it to the MatchWaveforms object
#         match_waveforms.append(MatchedEventWaveforms(match, stations_in, components_active, stream_trim))

#         print(f"Processed {i+1} of {len(matches_active)} matches!")

    
#     print(f"In total, {len(match_waveforms)} matches processed!")
#     return match_waveforms

## Get the list of geophone stations to read
def get_geo_stations_to_read(stations):
    if stations is None:
        stations_to_read = GEO_STATIONS
    else:
        if type(stations) is not list:
            if type(stations) is str:
                stations_to_read = [stations]
            else:
                raise TypeError("Error: stations must be a list of strings!")
        else:
            stations_to_read = stations

    return stations_to_read

## Get the list of geophone components to read
def get_geo_channels_to_read(components):
    if components is None:
        components_to_read = GEO_COMPONENTS
    else:
        if type(components) is not list:
            if type(components) is str:
                components_to_read = [components]
            else:
                raise TypeError("Error: component must be a list of strings!")
        else:
            components_to_read = components

    channels_to_read = [f"GH{component}" for component in components_to_read]

    return channels_to_read

## Get the list of hydrophone stations to read
def get_hydro_stations_to_read(stations):
    if stations is None:
        stations_to_read = HYDRO_STATIONS
    else:
        if type(stations) is not list:
            if type(stations) is str:
                stations_to_read = [stations]
            else:
                raise TypeError("Error: stations must be a list of strings!")
        else:
            stations_to_read = stations

    return stations_to_read

## Get the list of hydrophone locations to read
def get_hydro_locations_to_read(locations):
    if locations is None:
        locations_to_read = HYDRO_LOCATIONS
    else:
        if type(locations) is not list:
            if type(locations) is str:
                locations_to_read = [locations]
            else:
                raise TypeError("Error: locations must be a list of strings!")
        else:
            locations_to_read = locations

    return locations_to_read

# Read the day-long geophone waveforms recorded on specific channels of a specific station
def read_day_long_geo_waveforms(day, station, channel, rootdir=ROOTDIR_GEO):
        
            # Check if the channel is broken
            if f"{station}.{channel}" in BROKEN_CHANNELS:
                print(f"Warning: {station}.{channel} is broken!")
                return None
    
            # Read the waveforms
            pattern = join(rootdir, day, f"7F.{station}*{channel}.mseed")
            try:
                stream = read(pattern)
            except Exception as e:
                print(f"An error occurred: {e}")
                return None
    
            return stream

# Read the day-long hydrophone waveforms recorded on a specific location of a specific station
def read_day_long_hydro_waveforms(day, station, location, rootdir=ROOTDIR_HYDRO):
        
            # Check if the location is broken
            if f"{station}.{location}" in BROKEN_LOCATIONS:
                print(f"Warning: {station}.{location} is broken!")
                return None
    
            # Read the waveforms
            pattern = join(rootdir, day, f"7F.{station}.{location}.GDH.mseed")
            try:
                stream = read(pattern)
            except FileNotFoundError:
                print(f"Warning: {pattern} not found!")
                return None
            
            return stream

## Read the geophone waveforms in a timewindow recorded on specific channels of a specific station
def read_geo_waveforms_in_timewindow(starttime, endtime, station, channel, rootdir=ROOTDIR_GEO):
    
        ### Find the day of the starttime
        day = starttime.strftime("%Y-%m-%d")

        ### Check if the channel is broken
        if f"{station}.{channel}" in BROKEN_CHANNELS:
            print(f"Warning: {station}.{channel} is broken!")
            return None
    
        ### Read the waveforms
        pattern = join(rootdir, day, f"*{station}*{channel}.mseed")
        starttime = timestamp_to_utcdatetime(starttime)
        endtime = timestamp_to_utcdatetime(endtime)
        try:    
            stream = read(pattern, starttime=starttime, endtime=endtime)
        except:
            print(f"Warning: {pattern} not found!")
            return None
    
        return stream

# ## Read the hydrophone waveforms in a timewindow recorded on specific locations of a specific station
def read_hydro_waveforms_in_timewindow(starttime, endtime, station, location, rootdir=ROOTDIR_HYDRO):

        ### Find the day of the starttime
        day = starttime.strftime("%Y-%m-%d")

        ### Read the waveforms
        pattern = join(rootdir, day, f"7F.{station}.{location}.GDH.mseed")
        starttime = timestamp_to_utcdatetime(starttime)
        endtime = timestamp_to_utcdatetime(endtime)
        try:
            stream = read(pattern, starttime=starttime, endtime=endtime)
        except FileNotFoundError:
            print(f"Warning: {pattern} not found!")
            return None
        
        return stream


# Preprocess a stream object consisting of geophone data by removing the sensitivity, detrending, tapering, and filtering
# The output data will be in nm/s
def preprocess_geo_stream(stream_in, metadat, freqmin, freqmax, corners=4, zerophase=False, normalize=False, decimate=False, decimate_factor=10, path_meta=PATH_GEO_METADATA):

    # Remove the sensitivity
    stream_out = stream_in
    stream_out.remove_sensitivity(inventory=metadat)

    # Reverse the polarity of the vertical components
    stream_out = correct_geo_polarity(stream_out)

    # Convert from m/s to nm/s
    for trace in stream_out:
        trace.data *= 1e9

    # Detrend, taper, and filter
    if freqmin is not None or freqmax is not None:
        stream_out.detrend("linear")
        stream_out.taper(0.001)

    if freqmin is not None and freqmax is not None:
        stream_out.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=zerophase ,corners=corners)
    elif freqmin is not None and freqmax is None:
        stream_out.filter("highpass", freq=freqmin, zerophase=zerophase ,corners=corners)
    elif freqmin is None and freqmax is not None:
        stream_out.filter("lowpass", freq=freqmax, zerophase=zerophase ,corners=corners)

    # Decimate the waveforms
    if decimate:
        stream_out.decimate(decimate_factor)

    # Normalize the waveforms
    if normalize:
        stream_out.normalize(global_max=True)

    return stream_out

## Preprocess a stream object consisting of hydrophone data by removing the sensitivity, detrending, tapering, and filtering
## The output data will be in Pa
## Given that the instrument response of the hydrophone data from SAGE is wrong, we use a customized response-removal procedure
def preprocess_hydro_stream(stream_in, freqmin, freqmax, corners=4, zerophase=False, normalize=False, decimate=False, decimate_factor=None):

    ### Remove the sensitivity
    stream_out = remove_hydro_response(stream_in)

    ### Correct the polarity of the hydrophone at Locatin 04
    stream_out = correct_hydro_polarity(stream_out)

    ### Detrend, taper, and filter
    if freqmin is not None or freqmax is not None:
        stream_out.detrend("linear")
        stream_out.taper(0.001)

    if freqmin is not None and freqmax is not None:
        stream_out.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=zerophase ,corners=corners)
    elif freqmin is not None and freqmax is None:
        stream_out.filter("highpass", freq=freqmin, zerophase=zerophase ,corners=corners)
    elif freqmin is None and freqmax is not None:
        stream_out.filter("lowpass", freq=freqmax, zerophase=zerophase ,corners=corners)

    ### Decimate the waveforms
    if decimate:
        stream_out.decimate(decimate_factor)

    ### Normalize the waveforms
    if normalize:
        stream_out.normalize(global_max=True)

    return stream_out


## Correct the polarity of the vertical-component geophone data
def correct_geo_polarity(stream_in):
    stream_out = stream_in.copy()

    for trace in stream_out:
        if trace.stats.channel[-1] == "Z":
            trace.data = -1 * trace.data

    return stream_out
                        
## Remove the instrument response from a stream object consisting of hydrophone data using the instrument response information from Rob Sohn
def remove_hydro_response(stream_in):
    stream_out = stream_in.copy()

    for trace in stream_out:
        trace.data = trace.data / COUNTS_TO_VOLT # converts from counts to volts
        trace.data = trace.data / (10 ** (DB_VOLT_TO_MPASCAL / 20.)) # converts from volts to micropascals
        trace.data = trace.data / 1e6 # converts from micro pascals to pascals

    return stream_out

## Correct the polarity of the hydrophone at Location 04
def correct_hydro_polarity(stream_in):
    stream_out = stream_in.copy()

    for trace in stream_out:
        if trace.stats.station == "B00" and trace.stats.location == "04":
            trace.data = -1 * trace.data

    return stream_out