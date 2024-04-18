# Classes and functions for operations using PyTorch
from torch import Tensor
from torch import abs, hann_window, square, stft, sum, tensor
from numpy import linspace

from utils_basic import power2db, reltimes_to_timestamps
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_spec import StreamSTFTPSD, TraceSTFTPSD
from utils_spec import downsample_stft_spec, downsample_stft_freqax

######
# Functions
######

# Compute hourly spectrograms of a geophone station from a day-long stream object
def get_hourly_geo_spectrograms_for_a_day(stream_day, window_length = 1.0, overlap = 0.5, cuda = False, downsample = False, downsample_factor = None):
    if downsample and downsample_factor is None:
        raise ValueError("The downsample factor is not set!")

    station = stream_day[0].stats.station
    starttime_day = stream_day[0].stats.starttime
    endttime_day = stream_day[0].stats.endtime
    stream_spec_out = StreamSTFTPSD()
    stream_spec_ds_out = StreamSTFTPSD()

    starttime = starttime_day
    while starttime < endttime_day:
        endtime = starttime + 3600.0
        stream_hour = stream_day.slice(starttime = starttime, endtime = endtime)

        print(f"Computing the spectrograms for start time {starttime}...")
        stft_dict = get_stream_spectrograms(stream_hour, window_length, overlap=overlap, cuda = cuda)
        timeax = stft_dict[(station, "Z")][0]
        freqax = stft_dict[(station, "Z")][1]
        
        hour_spec_z = stft_dict[(station, "Z")][2]
        hour_spec_1 = stft_dict[(station, "1")][2]
        hour_spec_2 = stft_dict[(station, "2")][2]
        
        if downsample:
            print(f"Downsampling the spectrograms...")

            freqax_ds = downsample_stft_freqax(freqax, downsample_factor)
        
            hour_spec_z_ds = downsample_stft_spec(day_mat_z, downsample_factor)
            hour_spec_1_ds = downsample_stft_spec(day_mat_1, downsample_factor)
            hour_spec_2_ds = downsample_stft_spec(day_mat_2, downsample_factor)          

        # The original spectrograms
        trace_spec_z = TraceSTFTPSD(station, None, "Z", timeax, freqax, hour_spec_z)
        trace_spec_1 = TraceSTFTPSD(station, None, "1", timeax, freqax, hour_spec_1)
        trace_spec_2 = TraceSTFTPSD(station, None, "2", timeax, freqax, hour_spec_2)

        stream_spec = StreamSTFTPSD([trace_spec_z, trace_spec_1, trace_spec_2])
        stream_spec_out.extend(stream_spec)

        # The downsampled spectrograms
        if downsample:
            trace_spec_z = TraceSTFTPSD(station, None, "Z", timeax, freqax_ds, hour_spec_z_ds)
            trace_spec_1 = TraceSTFTPSD(station, None, "1", timeax, freqax_ds, hour_spec_1_ds)
            trace_spec_2 = TraceSTFTPSD(station, None, "2", timeax, freqax_ds, hour_spec_2_ds)

            stream_spec_ds = StreamSTFTPSD([trace_spec_z, trace_spec_1, trace_spec_2])
            stream_spec_ds_out.extend(stream_spec_ds)

        # Update the starttime
        starttime = endtime

    return stream_spec_out, stream_spec_ds_out

        
# Compute a day-long spectrogram of a station and return BOTH the original and downsampled spectrograms
# Window length is in SECONDS!
def get_day_long_spectrograms(station, day, metadat):
    print("######")
    print(f"Computing spectrograms for {station} on {day}...")
    print("######")

    print(f"Reading and processing the waveforms...")
    stream = read_and_process_day_long_geo_waveforms(day, metadat, stations = [station])

    if stream is None:
        raise ValueError("Error: No data found for the day!")

    print(f"Computing the spectrograms...")
    stft_dict = get_stream_spectrograms(stream, window_length, overlap=overlap, cuda = cuda)

    print(f"Downsampling the spectrograms...")
    timeax = stft_dict[(station, "Z")][0]
    freqax = stft_dict[(station, "Z")][1]
    freqax_ds = downsample_stft_freqax(freqax, downsample_factor)

    day_mat_z = stft_dict[(station, "Z")][2]
    day_mat_1 = stft_dict[(station, "1")][2]
    day_mat_2 = stft_dict[(station, "2")][2]

    day_mat_z_ds = downsample_stft_spec(day_mat_z, downsample_factor)
    day_mat_1_ds = downsample_stft_spec(day_mat_1, downsample_factor)
    day_mat_2_ds = downsample_stft_spec(day_mat_2, downsample_factor)

    if db:
        day_mat_z = power2db(day_mat_z)
        day_mat_1 = power2db(day_mat_1)
        day_mat_2 = power2db(day_mat_2)
        
        day_mat_z_ds = power2db(day_mat_z_ds)
        day_mat_1_ds = power2db(day_mat_1_ds)
        day_mat_2_ds = power2db(day_mat_2_ds)

    # The original spectrograms
    trace_spec_z = TraceSTFTPSD(station, None, "Z", timeax, freqax, day_mat_z, db = db)
    trace_spec_1 = TraceSTFTPSD(station, None, "1", timeax, freqax, day_mat_1, db = db)
    trace_spec_2 = TraceSTFTPSD(station, None, "2", timeax, freqax, day_mat_2, db = db)

    stream_spec = StreamSTFTPSD([trace_spec_z, trace_spec_1, trace_spec_2])
    
    # The downsampled spectrograms
    trace_spec_z = TraceSTFTPSD(station, None, "Z", timeax, freqax_ds, day_mat_z_ds, db = db)
    trace_spec_1 = TraceSTFTPSD(station, None, "1", timeax, freqax_ds, day_mat_1_ds, db = db)
    trace_spec_2 = TraceSTFTPSD(station, None, "2", timeax, freqax_ds, day_mat_2_ds, db = db)

    stream_spec_ds = StreamSTFTPSD([trace_spec_z, trace_spec_1, trace_spec_2])

    print(f"Done processing day {day}.")
    
    return stream_spec, stream_spec_ds
    
# Compute the spectrograms in PSD of a stream using STFT
def get_stream_spectrograms(stream, window_length = 1.0, overlap = 0.5, cuda = False):
    specdict = {}
    for trace in stream:
        station = trace.stats.station
        component = trace.stats.component

        timeax, freqax, psd = get_trace_spectrogram(trace, window_length, overlap, cuda = cuda)
        specdict[(station, component)] = (timeax, freqax, psd)

    return specdict

# Compute the spectrogram in PSD of a trace using STFT
def get_trace_spectrogram(trace, window_length = 1.0, overlap = 0.5, cuda = False):
    signal = trace.data
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

    num_freq = psd.shape[0]
    nyfreq = sampling_rate / 2
    freqax = linspace(0, nyfreq, num_freq)

    num_time = psd.shape[1]
    timeax = linspace(0, (numpts - 1) / sampling_rate, num_time)
    timeax = reltimes_to_timestamps(timeax, starttime)

    return timeax, freqax, psd
    
# Get the effective noise bandwidth of a window function
def get_window_enbw(window, sampling_rate):
    if not isinstance(window, Tensor):
        raise ValueError("Invalid start time format!")

    enbw = sampling_rate * sum(square((abs(window)))) / square(abs(sum(window)))

    return enbw

