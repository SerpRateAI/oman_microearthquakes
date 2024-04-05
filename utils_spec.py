# Functions and classes for spectrum analysis   
from scipy.fft import fft, fftfreq
from scipy.signal import iirfilter, sosfilt, freqz
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from numpy import amax, abs, pi, cumsum
from multitaper import MTSpec

from utils_basic import reltimes_to_timestamps, power2db

## Function to compute the spectrogram of a stream using STFT
## Returns spectrogram in dB
## A wrapper for the get_trace_spectrogram_stft function
def get_stream_spectrogram_stft(stream, window_length=1.0, overlap=0.5):
    specdict = {}
    for trace in stream:
        station = trace.stats.station
        component = trace.stats.component

        timeax, freqax, spec = get_trace_spectrogram_stft(trace, window_length, overlap)
        specdict[(station, component)] = (timeax, freqax, spec)

    return specdict

## Function to compute the spectrogram of a trace using STFT
## Returns spectrogram in dB
def get_trace_spectrogram_stft(trace, window_length=1.0, overlap=0.5):
    signal = trace.data
    sampling_rate = trace.stats.sampling_rate
    starttime = trace.stats.starttime
    starttime = starttime.datetime

    window = hann(int(window_length * sampling_rate))
    hop = int(window_length * sampling_rate * (1 - overlap))
    stft = ShortTimeFFT(window, hop, sampling_rate, scale_to="psd")
    freqax = stft.f
    timeax = stft.t(len(signal))
    timeax = reltimes_to_timestamps(timeax, starttime)

    spec = stft.spectrogram(signal, detr="linear")
    spec = spec / amax(spec)
    spec = power2db(spec)

    return timeax, freqax, spec

## Function to calculate the frequency response of a filter 
### Currently only support Butterworth filters
def get_filter_response(freqmin, freqmax, samprat, numpts, order=4):
    
    if freqmin >= freqmax:
        raise ValueError("Error: freqmin must be smaller than freqmax!")
    
    nyquist = samprat / 2
    low = freqmin / nyquist
    high = freqmax / nyquist
    b, a = iirfilter(order, [low, high], btype="bandpass", ftype="butter")
    omegaax, resp = freqz(b, a, worN=numpts)

    resp = abs(resp)
    resp = resp / amax(resp)
    freqax = omegaax * nyquist / pi

    return freqax, resp

## Get the time axis consisting of Pandas Timestamp objects from a Trace object
def get_timeax_from_trace(trace):
    timeax = trace.times("timestamp")

## Function for calculating the VELOCITY PSD of a VELOCITY signal using multitaper method
def get_vel_psd_mt(vel, sampling_rate=1000.0, nw=2):    
    mt = MTSpec(vel, nw=nw, dt=1/sampling_rate)
    freqax, psd = mt.rspec()
    
    return freqax, psd

## Function for calculating the DISPLACEMENT PSD of a VELOCITY using multitaper method
def get_disp_psd_mt(vel, sampling_rate=1000.0, nw=2):
    disp = vel2disp(vel, sampling_rate)
    
    mt = MTSpec(disp, nw=nw, dt=1/sampling_rate)
    freqax, psd = mt.rspec()
    
    return freqax, psd
    
## Convert velocity to displacement
def vel2disp(vel, sampling_rate=1000.0):
    disp = cumsum(vel) / sampling_rate

    return disp