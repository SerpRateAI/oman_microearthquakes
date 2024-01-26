# Function and classes for reading and preprocessing waveforms
from os.path import join
from numpy import abs, array, mean, amax, amin, argmax
from numpy.linalg import norm
from scipy.stats import linregress
from scipy.signal import find_peaks
from obspy import read, UTCDateTime, Stream
from obspy.signal.cross_correlation import correlate_template
from utils_cc import TemplateEventWaveforms, Matches, MatchedEvent, MatchWaveforms, MatchedEventWaveforms
from pandas import to_datetime

ROOTDIR = "/Volumes/OmanData/geophones_no_prefilt/data"

## Read and preprocess waveforms in a set of defined time windows
def read_and_process_windowed_waveforms(pickdf, freqmin, freqmax, rootdir=ROOTDIR, stations=None, begin=-0.01, end=0.2, reference="individual", taper=0.001, zerophase=False, corners=4):
    
    ### Read the hour-long waveforms
    starttime0 = pickdf["time"].min()
    starttime0 = UTCDateTime(to_datetime(starttime0).to_pydatetime()) # Convert to python datetime object first and then to UTCDateTime object
    timewin = starttime0.strftime("%Y-%m-%d-%H-00-00")

    if stations is None:
        stations = pickdf["station"].tolist()
    else:
        if type(stations) is not list:
            if type(stations) is str:
                stations = [stations]
            else:
                raise TypeError("Error: stations must be a list of strings!")

    stream_in = Stream()
    for station in stations:
        pattern = join(rootdir, timewin, f"*{station}*.SAC")
        stream_in += read(pattern)

    ### Process the waveforms
    stream_proc = preprocess_stream(stream_in, freqmin, freqmax, taper=taper, zerophase=zerophase, corners=corners)

    ### Trim the waveforms
    stream_trim = Stream()
    for station in stations:
        print(station)
        if reference == "individual":
            starttime =  pickdf[pickdf["station"] == station]["time"].values[0]
            starttime = UTCDateTime(to_datetime(starttime).to_pydatetime()) # Convert to python datetime object first and then to UTCDateTime object
        elif reference == "common":
            starttime = starttime0
        else:
            raise ValueError("Error: reference must be either 'individual' or 'common'!")

        # print(starttime.strftime("%Y-%m-%d %H:%M:%S"))
        starttime_trim = starttime + begin
        endtime_trim = starttime + end
        stream = stream_proc.select(station=station)
        #print((starttime_trim.strftime("%Y-%m-%d %H:%M:%S"), endtime_trim.strftime("%Y-%m-%d %H:%M:%S")))
        stream.trim(starttime_trim, endtime_trim)
        stream_trim += stream

    stream_out = stream_trim
    
    return stream_out


## Read and process the template waveforms
def read_and_process_template_waveforms(template, freqmin, freqmax, rootdir=ROOTDIR, stations=None, components=None, begin=0.01, end=0.2, reference="individual", taper=0.001, zerophase=False, corners=4):

    ### Read the hour-long waveforms
    starttime0 = UTCDateTime(template.first_start_time)
    timewin = starttime0.strftime("%Y-%m-%d-%H-00-00")

    if stations is None:
        stations_active = template.stations
    else:
        if type(stations) is not list:
            if type(stations) is str:
                stations_active = [stations]
            else:
                raise TypeError("Error: stations must be a list of strings!")
        else:
            stations_active = stations

    stream_in = Stream()
    for station in stations_active:
        if components is None:
            components_active = ["Z", "1", "2"]
        else:
            if type(components) is not list:
                if type(components) is str:
                    components_active = [components]
                else:
                    raise TypeError("Error: component must be a list of strings!")
            else:
                components_active = components
            
        print(components_active)
        print(station)
        print(rootdir)
        print(timewin)
        for component in components_active:
            print(component)
            pattern = join(rootdir, timewin, f"*{station}*{component}.SAC")
            stream_in += read(pattern)

    ### Process the waveforms
    stream_proc = preprocess_stream(stream_in, freqmin, freqmax, taper=taper, zerophase=zerophase, corners=corners)

    ### Trim the waveforms
    stream_trim = Stream()
    for station in stations_active:
        if reference == "individual":
            starttime =  UTCDateTime(template.get_start_time_for_station(station))
        elif reference == "common":
            starttime = starttime0
        else:
            raise ValueError("Error: reference must be either 'individual' or 'common'!")

        starttime_trim = starttime + begin
        endtime_trim = starttime + end
        stream = stream_proc.select(station=station)
        stream.trim(starttime_trim, endtime_trim)
        stream_trim += stream

    ### Store the data in a TemplateEventWaveforms object
    tempwaveforms = TemplateEventWaveforms(template, stations_active, components_active, stream_trim)

    return tempwaveforms

## Read and process the waveforms of a list of matches
def read_and_process_match_waveforms(matches, freqmin, freqmax, rootdir=ROOTDIR, names=None, stations=None, components=None, begin=0.01, end=0.2, reference="individual", taper=0.001, zerophase=False, corners=4):
    
    ### Time window for reading the long waveforms for preprocessing
    begin_read = begin - 1
    end_read = end + 3

    ### Determine if matches is a MatchWaveforms object
    if type(matches) is not Matches:
        raise TypeError("Error: matches must be a Matches object!")

    ### Keep only the matches with the specified names
    if names is not None:
        matches_active = matches.get_matches_by_names(names)
    else:
        matches_active = matches

    ### Determine the components to be processed
    if components is None:
            components_active = ["Z", "1", "2"]
    else:
        if type(components) is not list:
            if type(components) is str:
                components_active = [components]
            else:
                print("Error: components must be a list of strings!")
                raise TypeError
        else:
            components_active = components

    ### Process each matched event
    match_waveforms = MatchWaveforms()
    for i, match in enumerate(matches_active):
        ### Determine the stations to be processed
        if stations is None:
            stations_in  = match.stations
        else:
            if type(stations) is not list:
                if type(stations) is str:
                    stations = [stations]
                else:
                    print("Error: stations must be a list of strings!")
                    raise TypeError
            
            stations_in = list(set(stations) & set(match.stations))

        #### Read the long waveforms
        starttime0 = UTCDateTime(match.first_start_time)
        timewin = starttime0.strftime("%Y-%m-%d-%H-00-00")

        stream_in = Stream()
        for station in stations_in:
            
            for component in components_active:
                pattern = join(rootdir, timewin, f"*{station}*{component}.SAC")
                try:
                    stream_in += read(pattern, starttime=starttime0+begin_read, endtime=starttime0+end_read)
                except FileNotFoundError:
                    print(f"Warning: {pattern} not found!")
                    continue

        ### Process the waveforms
        stream_proc = preprocess_stream(stream_in, freqmin, freqmax, taper=taper, zerophase=zerophase, corners=corners)

        ### Trim the waveforms
        stream_trim = Stream()
        for station in stations_in:

            if reference == "individual":
                starttime = match.get_info_for_station(station, entries=["start_time"])["start_time"]
                starttime_trim = UTCDateTime(starttime) + begin
                endtime_trim = UTCDateTime(starttime) + end
            elif reference == "common":
                starttime_trim = starttime0 + begin
                endtime_trim = starttime0 + end

            stream = stream_proc.select(station=station)
            stream.trim(starttime_trim, endtime_trim)
            stream_trim += stream

        ### Store the data in a MatchedEventWaveforms object and append it to the MatchWaveforms object
        match_waveforms.append(MatchedEventWaveforms(match, stations_in, components_active, stream_trim))

        print(f"Processed {i+1} of {len(matches_active)} matches!")

    
    print(f"In total, {len(match_waveforms)} matches processed!")
    return match_waveforms

## Preprocess a stream object
def preprocess_stream(stream_in, freqmin, freqmax, corners=4, zerophase=False, taper=0.001):

    stream_out = stream_in.copy()
    stream_out.detrend("linear")
    stream_out.taper(0.001)

    if freqmin is not None and freqmax is not None:
        stream_out.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=zerophase ,corners=corners)
    elif freqmin is not None and freqmax is None:
        stream_out.filter("highpass", freq=freqmin, zerophase=zerophase ,corners=corners)
    elif freqmin is None and freqmax is not None:
        stream_out.filter("lowpass", freq=freqmax, zerophase=zerophase ,corners=corners)

    return stream_out

