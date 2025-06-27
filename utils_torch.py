# Classes and functions for operations using PyTorch
from torch import Tensor
from torch import abs, hann_window, square, stft, sum, tensor, atan2, rad2deg
from numpy import angle, linspace, round
from pandas import date_range
from pandas import Timedelta, Timestamp

from utils_basic import SPECTROGRAM_DIR
from utils_basic import power2db, reltimes_to_datetimes
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_spec import StreamSTFTPSD, TraceSTFTPSD, StreamSTFT, TraceSTFT
from utils_spec import downsample_stft_stream_freq

######
# Functions
######
    
# # Compute hourly spectrograms of a geophone station from a day-long stream object
# # Window length is in SECONDS!
# # Hours with missing data on ANY of the components are skipped
# def get_hourly_geo_spectrograms_for_a_day(stream_day, window_length = 1.0, overlap = 0.0, cuda = False, downsample = False, downsample_factor = None):
#     if downsample and downsample_factor is None:
#         raise ValueError("The downsample factor is not set!")

#     station = stream_day[0].stats.station
#     starttime_day = stream_day[0].stats.starttime
#     starttime_day = starttime_day.replace(hour=0, minute=0, second=0, microsecond=0)
#     stream_spec_out = StreamSTFTPSD()
#     stream_spec_ds_out = StreamSTFTPSD()

#     for hour in range(24):
#         starttime_hour = starttime_day + hour * 3600.0
#         endtime_hour = starttime_hour + 3600.0
#         stream_hour = stream_day.slice(starttime = starttime_hour, endtime = endtime_hour)

#         print(f"Computing the spectrograms for starttime {starttime_hour}...")
        
#         if stream_hour is None:
#             print(f"No data found for {station} between {starttime_hour} and {endtime_hour}! Skipped.")
#             continue
#         elif stream_hour[0].stats.npts < 3600001:
#             print(f"The length of the data is less than 1 hour for {station} between {starttime_hour} and {endtime_hour}! Skipped.")
#             continue

#         stream_spec = get_stream_spectrograms(stream_hour, window_length, overlap=overlap, cuda = cuda)
#         stream_spec_out.extend(stream_spec)
        
#         if downsample:
#             print(f"Downsampling the spectrograms...")
#             stream_spec_ds = downsample_stft_stream_freq(stream_spec, factor = downsample_factor)
#             stream_spec_ds_out.extend(stream_spec_ds)    

#     return stream_spec_out, stream_spec_ds_out

# # Compute hourly spectrograms for ALL locations of a hydrophone station from a day-long stream object
# # Window length is in SECONDS!
# # Hours with missing data on ANY of the locations are skipped
# def get_hourly_hydro_spectrograms_for_a_day(stream_day, window_length = 1.0, overlap = 0.0, cuda = False, downsample = False, downsample_factor = None):
#     if downsample and downsample_factor is None:
#         raise ValueError("The downsample factor is not set!")

#     num_loc_in = len(stream_day)
    
#     station = stream_day[0].stats.station
#     starttime_day = stream_day[0].stats.starttime
#     starttime_day = starttime_day.replace(hour=0, minute=0, second=0, microsecond=0)
#     stream_spec_out = StreamSTFTPSD()
#     stream_spec_ds_out = StreamSTFTPSD()

#     # Loop over the hours of the day
#     for hour in range(24):
#         starttime_hour = starttime_day + hour * 3600.0
#         endtime_hour = starttime_hour + 3600.0
#         stream_hour = stream_day.slice(starttime = starttime_hour, endtime = endtime_hour)
#         skip_hour = False

#         print(f"Computing the spectrograms for starttime {starttime_hour}...")

#         # Examine the intergrity of the data of the hour
#         if stream_hour is None:
#             print(f"No data found for {station} between {starttime_hour} and {endtime_hour}! The hour is skipped.")
#             continue
#         elif len(stream_hour) != num_loc_in:
#             print(f"Not all locations are availalbe for the hour! The hour is skipped for all locations.")
#             continue
#         else:
#             for trace in stream_hour:
#                 location = trace.stats.location
#                 numpts = trace.stats.npts
#                 if numpts < 3600001:
#                     print(f"Data of {station}.{location} is shorter than an hour. The hour is skipped for all locations.")
#                     skip_hour = True
#                     break

#         if skip_hour == True:
#             continue

#         # Compute the spectrograms for the hour
#         stream_spec = get_stream_spectrograms(stream_hour, window_length, overlap=overlap, cuda = cuda)
#         stream_spec_out.extend(stream_spec)

#         # Downsample the spectrograms
#         if downsample:
#             print(f"Downsampling the spectrograms...")
#             stream_spec_ds = downsample_stft_stream_freq(stream_spec, factor = downsample_factor)
#             stream_spec_ds_out.extend(stream_spec_ds)    

#     return stream_spec_out, stream_spec_ds_out

# Compute a day-long spectrogram of a geophone station
# Window length is in SECONDS!
# The new version of the function returns a StreamSTFT object containing both the complex coefficients and the PSD, 2024-09-17
# Downsample option is no longer available
# def get_daily_geo_spectrograms(stream_day, window_length = 60.0, overlap = 0.0, cuda = False, resample_in_parallel = False, **kwargs):

#     if resample_in_parallel and "num_process_resample" not in kwargs:
#         raise ValueError("Error: Number of processes is not given!")
    
#     # Compute the spectrograms
#     print(f"Computing the spectrograms...")
#     stream_spec = get_stream_spectrograms(stream_day, window_length = window_length, overlap = overlap, cuda = cuda)
    
#     # Pad and resample the spectrograms to the begin and end of the day
#     print(f"Resampling the spectrograms to the begin and end of the day...")
#     if resample_in_parallel:
#         num_process = kwargs["num_process_resample"]
#         stream_spec.resample_to_day(parallel = True, num_process = num_process)
#     else:
#         stream_spec.resample_to_day(parallel = False)

#     # Set the time labels
#     stream_spec.set_time_labels(block_type = "daily")

#     return stream_spec

# Compute the STFT with both power and phase information of a day-long stream object
# Window length is in SECONDS!
def get_daily_stft(stream_day, window_length = 60.0, overlap = 0.0):
        
        # Compute the spectrograms
        print(f"Computing the spectrograms...")
        stream_stft = get_stream_stft(stream_day, window_length = window_length, overlap = overlap)
        # print(type(stream_spec[0].psd_mat[0, 0]))
    
        # # Pad and resample the spectrograms to the begin and end of the day
        # print(f"Resampling the spectrograms to the begin and end of the day...")
        # stream_stft.resample_to_day(num_process = num_process)
    
        # Set the time labels
        stream_stft.set_time_labels(block_type = "daily")
        
        return stream_stft

# # Compute a day-long STFT with both power and phase information of ALL locations of a hydrophone station
# # Window length is in SECONDS!
# def get_daily_hydro_stft(stream_day, window_length = 60.0, overlap = 0.0):
    
#         # Compute the spectrograms
#         print(f"Computing the spectrograms...")
#         stream_stft = get_stream_stft(stream_day, window_length = window_length, overlap = overlap)
#         # print(type(stream_spec[0].psd_mat[0, 0]))
    
#         # # Pad and resample the spectrograms to the begin and end of the day
#         # print(f"Resampling the spectrograms to the begin and end of the day...")
#         # stream_stft.resample_to_day(num_process = num_process)
    
#         # Set the time labels
#         stream_stft.set_time_labels(block_type = "daily")
        
#         return stream_stft

# Compute a day-long spectrogram of ALL locations of a hydrophone station
# Window length is in SECONDS!
# The new version of the function returns a StreamSTFT object containing both the complex coefficients and the PSD, 2024-09-29
# Downsample option is no longer available
def get_daily_hydro_stft_psd(stream_day, window_length = 60.0, overlap = 0.0, cuda = False, num_process = 1):

    
    # Compute the spectrograms
    print(f"Computing the spectrograms...")
    stream_stft = get_stream_stft(stream_day, window_length = window_length, overlap = overlap)
    # print(type(stream_spec[0].psd_mat[0, 0]))

    # Pad and resample the spectrograms to the begin and end of the day
    print(f"Resampling the spectrograms to the begin and end of the day...")
    stream_stft.resample_to_day(num_process = num_process, psd_only = True)

    # Set the time labels
    stream_stft.set_time_labels(block_type = "daily")
    
    return stream_stft

# Compute the spectrogram and return the complex coefficients of a stream using STFT
def get_stream_stft(stream, window_length = 1.0, overlap = 0.0):
    #stream_spec = StreamSTFTPSD()
    stream_stft = StreamSTFT()

    for trace in stream:
        trace_spec = get_trace_stft(trace, window_length, overlap)
        stream_stft.append(trace_spec)

    return stream_stft

# Compute the spectrogram in PSD of a trace using STFT
# The new version of the function returns a TraceSTFT object containing both the complex coefficients and the PSD, 2024-09-17
def get_trace_stft(trace, window_length = 1.0, overlap = 0.0):
    signal = trace.data
    station = trace.stats.station
    location = trace.stats.location
    component = trace.stats.component
    sampling_rate = trace.stats.sampling_rate
    starttime = trace.stats.starttime
    starttime = starttime.datetime

    num_fft = int(round(window_length * sampling_rate))
    window = hann_window(num_fft)
    hop_length = int(round(window_length * sampling_rate * (1 - overlap)))
    signal_tsr = tensor(signal)
    coeff_tsr = stft(signal_tsr, num_fft, hop_length = hop_length, window = window, return_complex = True, center = False)

    # Get the phase in degrees
    pha_tsr = atan2(coeff_tsr.imag, coeff_tsr.real)
    pha_tsr = rad2deg(pha_tsr)
    pha_mat = pha_tsr.numpy()

    # Get the correct scaling factor for the PSD
    psd_tsr = square(abs(coeff_tsr)) / sampling_rate
    psd_tsr[1:-1, :] =  2 * psd_tsr[1:-1, :]
    window_power = sum(square(window))
    psd_tsr = psd_tsr / window_power
    # psd_tsr = psd_tsr.real
    # enbw = get_window_enbw(window, sampling_rate)
    psd_mat = psd_tsr.numpy()

    # Assemble the output TraceSTFT object
    num_freq = psd_mat.shape[0]
    freq_interval = sampling_rate / num_fft
    freqax = linspace(0, (num_freq - 1) * freq_interval, num_freq)
    
    num_time = psd_mat.shape[1]
    time_interval = hop_length / sampling_rate
    # timeax = linspace(0, (num_time - 1) * time_interval, num_time)
    starttime_left = Timestamp(starttime)
    starttime_center = starttime_left + Timedelta(seconds = window_length / 2)
    timeax = date_range(start = starttime_center, periods = num_time, freq = f"{time_interval}s")

    trace_stft = TraceSTFT(station, location, component, "", timeax, freqax, psd_mat, pha_mat, overlap = overlap, db = False)
    # print(type(trace_spec.psd_mat[0, 0]))
    #trace_spec = TraceSTFTPSD(station, location, component, "", timeax, freqax, psd, overlap = overlap, db = False)

    return trace_stft
    
# Get the effective noise bandwidth of a window function
def get_window_enbw(window, sampling_rate):
    if not isinstance(window, Tensor):
        raise ValueError("Invalid input type for the window function!")

    enbw = sampling_rate * sum(square((abs(window)))) / square(abs(sum(window)))

    return enbw

