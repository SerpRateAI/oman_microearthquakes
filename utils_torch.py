# Classes and functions for operations using PyTorch
from torch import Tensor
from torch import abs, hann_window, square, stft, sum, tensor
from numpy import linspace

from utils_basic import power2db, reltimes_to_timestamps
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_spec import StreamSTFTPSD, TraceSTFTPSD
from utils_spec import downsample_stft_stream_freq

######
# Functions
######

# Compute hourly spectrograms of a geophone station from a day-long stream object
# Window length is in SECONDS!
# Hours with missing data on ANY of the components are skipped
def get_hourly_geo_spectrograms_for_a_day(stream_day, window_length = 1.0, overlap = 0.0, cuda = False, downsample = False, downsample_factor = None):
    if downsample and downsample_factor is None:
        raise ValueError("The downsample factor is not set!")

    station = stream_day[0].stats.station
    starttime_day = stream_day[0].stats.starttime
    starttime_day = starttime_day.replace(hour=0, minute=0, second=0, microsecond=0)
    stream_spec_out = StreamSTFTPSD()
    stream_spec_ds_out = StreamSTFTPSD()

    for hour in range(24):
        starttime_hour = starttime_day + hour * 3600.0
        endtime_hour = starttime_hour + 3600.0
        stream_hour = stream_day.slice(starttime = starttime_hour, endtime = endtime_hour)

        if stream_hour is None:
            print(f"No data found for {station} between {starttime_hour} and {endtime_hour}! Skipped.")
            continue
        elif stream_hour[0].stats.npts < 3600001:
            print(f"The length of the data is less than 1 hour for {station} between {starttime_hour} and {endtime_hour}! Skipped.")
            continue

        print(f"Computing the spectrograms for starttime {starttime_hour}...")
        stream_spec = get_stream_spectrograms(stream_hour, window_length, overlap=overlap, cuda = cuda)
        stream_spec_out.extend(stream_spec)
        
        if downsample:
            print(f"Downsampling the spectrograms...")
            stream_spec_ds = downsample_stft_stream_freq(stream_spec, factor = downsample_factor)
            stream_spec_ds_out.extend(stream_spec_ds)    

    return stream_spec_out, stream_spec_ds_out

# Compute hourly spectrograms for ALL locations of a hydrophone station from a day-long stream object
# Window length is in SECONDS!
# Hours with missing data on ANY of the locations are skipped
def get_hourly_hydro_spectrograms_for_a_day(stream_day, window_length = 1.0, overlap = 0.0, cuda = False, downsample = False, downsample_factor = None):
    if downsample and downsample_factor is None:
        raise ValueError("The downsample factor is not set!")

    num_loc_in = len(stream_day)
    
    station = stream_day[0].stats.station
    starttime_day = stream_day[0].stats.starttime
    starttime_day = starttime_day.replace(hour=0, minute=0, second=0, microsecond=0)
    stream_spec_out = StreamSTFTPSD()
    stream_spec_ds_out = StreamSTFTPSD()

    # Loop over the hours of the day
    for hour in range(24):
        starttime_hour = starttime_day + hour * 3600.0
        endtime_hour = starttime_hour + 3600.0
        stream_hour = stream_day.slice(starttime = starttime_hour, endtime = endtime_hour)
        skip_hour = False

        # Examine the intergrity of the data of the hour
        if stream_hour is None:
            print(f"No data found for {station} between {starttime_hour} and {endtime_hour}! The hour is skipped.")
            continue
        elif len(stream_hour) != num_loc_in:
            print(f"Not all locations are availalbe for the hour! The hour is skipped for all locations.")
            continue
        else:
            for trace in stream_hour:
                location = trace.stats.location
                numpts = trace.stats.npts
                if numpts < 3600001:
                    print(f"Data of {station}.{location} is shorter than an hour. The hour is skipped for all locations.")
                    skip_hour = True
                    break

        if skip_hour == True:
            continue

        # Compute the spectrograms for the hour
        print(f"Computing the spectrograms for starttime {starttime_hour}...")
        
        stream_spec = get_stream_spectrograms(stream_hour, window_length, overlap=overlap, cuda = cuda)
        stream_spec_out.extend(stream_spec)

        # Downsample the spectrograms
        if downsample:
            print(f"Downsampling the spectrograms...")
            stream_spec_ds = downsample_stft_stream_freq(stream_spec, factor = downsample_factor)
            stream_spec_ds_out.extend(stream_spec_ds)    

    return stream_spec_out, stream_spec_ds_out
        
# Compute a day-long spectrogram of a geophone station and return BOTH the original and downsampled spectrograms
# Window length is in SECONDS!
def get_daily_geo_spectrograms(stream_day, window_length = 60.0, overlap = 0.0, cuda = False, downsample = False, downsample_factor = None):
    if downsample and downsample_factor is None:
        raise ValueError("The downsample factor is not set!")
    
    # Compute the spectrograms
    print(f"Computing the spectrograms...")
    stream_spec = get_stream_spectrograms(stream_day, window_length, overlap=overlap, cuda = cuda)

    # Downsample the spectrograms
    if downsample:
        print(f"Downsampling the spectrograms...")
        stream_spec_ds = downsample_stft_stream_freq(stream_spec, factor = downsample_factor)
    else:
        stream_spec_ds = None
    
    return stream_spec, stream_spec_ds

# Compute a day-long spectrogram of ALL locations of a hydrophone station and return BOTH the original and downsampled spectrograms
# Window length is in SECONDS!

def get_daily_hydro_spectrograms(stream_day, window_length = 60.0, overlap = 0.0, cuda = False, downsample = False, downsample_factor = None):
    if downsample and downsample_factor is None:
        raise ValueError("The downsample factor is not set!")
    
    # Compute the spectrograms
    print(f"Computing the spectrograms...")
    stream_spec = get_stream_spectrograms(stream_day, window_length, overlap=overlap, cuda = cuda)

    # Downsample the spectrograms
    if downsample:
            stream_spec_ds = downsample_stft_stream_freq(stream_spec, factor = downsample_factor)
    else:
        stream_spec_ds = None
    
    return stream_spec, stream_spec_ds

# Compute a day-long spectrogram of ALL locations of a hydrophone station and return BOTH the original and downsampled spectrograms
# Window length is in SECONDS!

def get_daily_hydro_spectrograms(stream_day, window_length = 60.0, overlap = 0.0, cuda = False, downsample = False, downsample_factor = None):
    if downsample and downsample_factor is None:
        raise ValueError("The downsample factor is not set!")
    
    # Compute the spectrograms
    print(f"Computing the spectrograms...")
    stream_spec = get_stream_spectrograms(stream_day, window_length, overlap=overlap, cuda = cuda)

    # Pad the two ends of the spectrograms with NaNs
    print(f"Padding the spectrograms...")
    stream_spec.pad_to_length(length = "day")

    # Downsample the spectrograms
    if downsample:
        print(f"Downsampling the spectrograms...")
        stream_spec_ds = downsample_stft_stream_freq(stream_spec, factor = downsample_factor)
        stream_spec_ds = downsample_stft_stream_freq(stream_spec, factor = downsample_factor)
    else:
        stream_spec_ds = None
    
    return stream_spec, stream_spec_ds
    
# Compute the spectrograms in PSD of a stream using STFT
def get_stream_spectrograms(stream, window_length = 1.0, overlap = 0.5, cuda = False):
    stream_spec = StreamSTFTPSD()

    for trace in stream:
        trace_spec = get_trace_spectrogram(trace, window_length, overlap, cuda = cuda)
        stream_spec.append(trace_spec)

    return stream_spec

# Compute the spectrogram in PSD of a trace using STFT
def get_trace_spectrogram(trace, window_length = 1.0, overlap = 0.0, cuda = False):
    signal = trace.data
    station = trace.stats.station
    location = trace.stats.location
    component = trace.stats.component
    sampling_rate = trace.stats.sampling_rate
    numpts = trace.stats.npts
    starttime = trace.stats.starttime
    starttime = starttime.datetime

    num_fft = int(window_length * sampling_rate)
    window = hann_window(num_fft)
    hop_length = int(window_length * sampling_rate * (1 - overlap))
    signal_tsr = tensor(signal)

    # Determine if GPU is used
    if cuda:
        signal_tsr.to("cuda")
        
    stft_tsr = stft(signal_tsr, num_fft, hop_length = hop_length, window = window, return_complex = True)

    # Normalize the STFT
    stft_tsr[1:-1] =  2 * stft_tsr[1:-1] / num_fft

    # Convert to PSD
    power_tsr = square(abs(stft_tsr)) / 2
    enbw = get_window_enbw(window, sampling_rate)
    psd_tsr = power_tsr / enbw

    if cuda:
        psd_tsr.to("cpu")
        
    psd = psd_tsr.numpy()

    # Assemble the output TraceSTFTPSD object
    num_freq = psd.shape[0]
    nyfreq = sampling_rate / 2
    freqax = linspace(0, nyfreq, num_freq)
    
    num_time = psd.shape[1]
    timeax = linspace(0, (numpts - 1) / sampling_rate, num_time)
    timeax = reltimes_to_timestamps(timeax, starttime)

    time_label = timeax[0].replace(minute = 0, second = 0, microsecond = 0).strftime("%Y%m%d%H%M%S")

    trace_spec = TraceSTFTPSD(station, location, component, time_label, timeax, freqax, psd, overlap = overlap, db = False)

    return trace_spec
    
# Get the effective noise bandwidth of a window function
def get_window_enbw(window, sampling_rate):
    if not isinstance(window, Tensor):
        raise ValueError("Invalid start time format!")

    enbw = sampling_rate * sum(square((abs(window)))) / square(abs(sum(window)))

    return enbw

