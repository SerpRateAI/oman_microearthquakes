# Functions and classes for spectrum analysis   
from scipy.fft import fft, fftfreq
from scipy.signal import iirfilter, sosfilt, freqz
from numpy import amax, abs, pi

## Function to calculate the spectrum of a signal
def get_data_spectrum(data, samprat, taper=0.01):
   
    numpts = len(data)
    freqax = fftfreq(numpts, d=1/samprat)
    spec = fft(data)
    spec = spec[1:int(numpts/2)]
    freqax = freqax[1:int(numpts/2)]
    spec = abs(spec)
    spec = spec / amax(spec)
    
    return freqax, spec

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
    