# Functions and classes for spectrum analysis   
from scipy.fft import fft, fftfreq
from scipy.signal import iirfilter, sosfilt, freqz
from scipy.signal import periodogram
from numpy import amax, abs, pi, hanning, cumsum
from multitaper import MTSpec

## Function to calculate the spectrum of a signal
def get_data_spectrum(data, samprat, taper=0.01):
   
    numpts = len(data)
    
    # Apply taper to the data
    taper_length = int(taper * numpts)
    taper_window = hanning(taper_length)
    data[:taper_length] *= taper_window
    data[-taper_length:] *= taper_window[::-1]
    
    freqax = fftfreq(numpts, d=1/samprat)
    spec = fft(data)
    spec = spec[1:int(numpts/2)]
    freqax = freqax[1:int(numpts/2)]
    spec = abs(spec)
    spec = spec / amax(spec)
    
    return freqax, spec

## Function to calculate the power spectral density of a signal
def get_data_psd(data, samprat, taper=0.01):
        
        numpts = len(data)
        
        # Apply taper to the data
        taper_length = int(taper * numpts)
        taper_window = hanning(taper_length)
        data[:taper_length] *= taper_window
        data[-taper_length:] *= taper_window[::-1]
        
        freqax, psd = periodogram(data, fs=samprat)
        
        return freqax, psd

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