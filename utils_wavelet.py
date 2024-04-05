# Functions and classes for wavelet analysis
from numpy import arange, array, angle, column_stack, exp, geomspace, linspace, meshgrid, nan, ones, sum, tile, zeros_like
from scipy.stats import gmean
from scipy.signal import convolve, convolve2d
from pywt import wavelist, scale2frequency, cwt

from utils_basic import SAMPLING_RATE_GEO, GEO_COMPONENTS, get_timeax_from_trace, get_unique_stations, power2db
from utils_basic import WAVELET_COMPONENT_PAIRS as component_pairs

## Class for storing the wavelet specgtra of multiple traces
class WaveletSpectra:
    def __init__(self):
        self.spectra = []

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, index):
        return self.spectra[index]

    def __setitem__(self, index, value):
        self.spectra[index] = value

    def __delitem__(self, index):
        del self.spectra[index]

    def append(self, spectrum):
        if not isinstance(spectrum, WaveletSpectrum):
            raise TypeError("Invalid spectrum format!")
        
        self.spectra.append(spectrum)

    def extend(self, spectra):
        if not isinstance(spectra, WaveletSpectra):
            raise TypeError("Invalid spectra format!")
        
        self.spectra.extend(spectra)

    def remove(self, spectrum):
        self.spectra.remove(spectrum)

    def clear(self):
        self.spectra.clear()

    def get_stations(self):
        stations = list(set([spectrum.station for spectrum in self.spectra]))
        stations.sort()

        return stations
    
    def get_spectra(self, station, component=None, location=None):
        if component is None:
            if location is None:
                spectra = [spectrum for spectrum in self.spectra if spectrum.station == station]
            else:
                spectra = [spectrum for spectrum in self.spectra if spectrum.station == station and spectrum.location == location]
        else:
            if location is None:
                spectra = [spectrum for spectrum in self.spectra if spectrum.station == station and spectrum.component == component]
            else:
                spectra = [spectrum for spectrum in self.spectra if spectrum.station == station and spectrum.component == component and spectrum.location == location]

        return spectra

## Class for storing the station cross-spectra
class StationCrossSpectra:
    def __init__(self):
        self.spectra = []

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, index):
        return self.spectra[index]

    def __setitem__(self, index, value):
        self.spectra[index] = value

    def __delitem__(self, index):
        del self.spectra[index]

    def append(self, spectrum):
        if not isinstance(spectrum, StationCrossSpectrum):
            raise TypeError("Invalid spectrum format!")
        
        self.spectra.append(spectrum)

    def extend(self, spectra):
        if not isinstance(spectra, StationCrossSpectra):
            raise TypeError("Invalid spectra format!")
        
        self.spectra.extend(spectra)

    def remove(self, spectrum):
        self.spectra.remove(spectrum)

    def clear(self):
        self.spectra.clear()

    def get_spectra(self, station1, station2, component=None):
        ## Ensure station1 is alphabetically before station2
        if station1 > station2:
            station1, station2 = station2, station1

        if component is None:
            spectra = [spectrum for spectrum in self.spectra if spectrum.station1 == station1 and spectrum.station2 == station2]
        else:
            spectra = [spectrum for spectrum in self.spectra if spectrum.station1 == station1 and spectrum.station2 == station2 and spectrum.component == component]

        return spectra
    
## Class for storing the component cross-spectra
class ComponentCrossSpectra:
    def __init__(self):
        self.spectra = []

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, index):
        return self.spectra[index]

    def __setitem__(self, index, value):
        self.spectra[index] = value

    def __delitem__(self, index):
        del self.spectra[index]

    def append(self, spectrum):
        if not isinstance(spectrum, ComponentCrossSpectrum):
            raise TypeError("Invalid spectrum format!")
        
        self.spectra.append(spectrum)

    def extend(self, spectra):
        if not isinstance(spectra, ComponentCrossSpectra):
            raise TypeError("Invalid spectra format!")
        
        self.spectra.extend(spectra)

    def remove(self, spectrum):
        self.spectra.remove(spectrum)

    def clear(self):
        self.spectra.clear()

    def get_spectra(self, station, component1, component2):
        spectra = [spectrum for spectrum in self.spectra if spectrum.station == station and spectrum.component1 == component1 and spectrum.component2 == component2]

        return spectra

## Class for storing the wavelet spectrum and associated parameters of a trace
class WaveletSpectrum:
    def __init__(self, station, location, component, times, freqs, scales, data, wavelet):
        self.station = station
        self.location = location
        self.component = component
        self.times = times
        self.freqs = freqs
        self.scales = scales
        self.data = data
        self.wavelet = wavelet

    def get_power(self, db=True, reference_type="mean"):
        power = abs(self.data) ** 2
        if db:
            power = power2db(power, reference_type=reference_type)
        
        return power
    
    def get_phase(self):
        phase = angle(self.data)
        
        return phase

## Class for storing the station cross-spectrum and associated parameters of a pair of traces
class StationCrossSpectrum:
    def __init__(self, station1, station2, component, times, freqs, scales, data, coherence, wavelet):
        self.station1 = station1
        self.station2 = station2
        self.component = component
        self.times = times
        self.freqs = freqs
        self.scales = scales
        self.data = data
        self.coherence = coherence
        self.wavelet = wavelet

    def get_power(self, db=True, reference_type="mean"):
        power = abs(self.data) ** 2
        if db:
            power = power2db(power, reference_type=reference_type)
        
        return power
    
    def get_phase(self):
        phase = angle(self.data)
        
        return phase
    
## Class for storing the component cross-spectrum and associated parameters of a pair of traces
class ComponentCrossSpectrum:
    def __init__(self, station, component1, component2, times, freqs, scales, data, coherence, wavelet):
        self.station = station
        self.component1 = component1
        self.component2 = component2
        self.times = times
        self.freqs = freqs
        self.scales = scales
        self.data = data
        self.coherence = coherence
        self.wavelet = wavelet

    def get_power(self, db=True, reference_type="mean"):
        power = abs(self.data) ** 2
        if db:
            power = power2db(power, reference_type=reference_type)
        
        return power
    
    def get_phase(self):
        phase = angle(self.data)
        
        return phase
    
## Function for getting the station cross-spectra of all possible traces in a stream
def get_stream_cross_station_spectra(stream, wavelet="cmorl", bandwidth=10.0, center_freq=1.0, scales=range(1, 128), boxcar_window=3, coherence=True):
    ### Get the station list
    stations = get_unique_stations(stream)
    stations.sort() # Sort the stations alphabetically

    ### Compute the spectra
    specs = get_stream_cwt(stream, wavelet, bandwidth, center_freq, scales)

    ### Compute the cross-spectra
    cross_specs = StationCrossSpectra()
    for i, station1 in enumerate(stations):
        for j, station2 in enumerate(stations):
            if i < j:
                print(f"Computing cross-spectra between stations {station1} and {station2}...")
                for component in GEO_COMPONENTS:
                    spec1 = specs.get_spectra(station1, component=component)[0]
                    spec2 = specs.get_spectra(station2, component=component)[0]

                    freqs = spec1.freqs
                    timeax = spec1.times
                    scales = spec1.scales

                    #### Compute the cross spectrum and coherence
                    cross_mat, cross_cohe = get_cross_spectrum(spec1, spec2, boxcar_window=boxcar_window, coherence=coherence)

                    cross_spec_dict = {"station1": station1, "station2": station2, "component": component, "times": timeax, "freqs": freqs, "scales": scales, "data": cross_mat, "coherence": cross_cohe, "wavelet": wavelet}
                    cross_spec = StationCrossSpectrum(**cross_spec_dict)
                    cross_specs.append(cross_spec)

    return specs, cross_specs

## Function for getting the component cross-spectra of all possible traces in a stream
def get_stream_cross_component_spectra(stream, wavelet="cmorl", bandwidth=10.0, center_freq=1.0, scales=range(1, 128), boxcar_window=3, coherence=True):
    ### Get the station list
    stations = get_unique_stations(stream)

    ### Compute the spectra
    specs = get_stream_cwt(stream, wavelet, bandwidth, center_freq, scales)

    ### Compute the cross-spectra
    cross_specs = ComponentCrossSpectra()
    for station in stations:
        stream_station = stream.select(station=station)
        if len(stream_station) < 3:
            print(f"Skipping station {station} due to insufficient number of components!")
            continue

        ## Loop over the component pairs
        for component_pair in component_pairs:
            component1 = component_pair[0]
            component2 = component_pair[1]

            spec1 = specs.get_spectra(station, component=component1)[0]
            spec2 = specs.get_spectra(station, component=component2)[0]

            freqs = spec1.freqs
            timeax = spec1.times
            scales = spec1.scales

            cross_mat, cross_cohe = get_cross_spectrum(spec1, spec2, boxcar_window=boxcar_window, coherence=coherence)

            cross_spec_dict = {"station": station, "component1": component1, "component2": component2, "times": timeax, "freqs": freqs, "scales": scales, "data": cross_mat, "coherence": cross_cohe, "wavelet": wavelet}
            cross_spec = ComponentCrossSpectrum(**cross_spec_dict)
            cross_specs.append(cross_spec)

    return specs, cross_specs

## Function for getting the continous wavelet transform of a stream
## Wrapper for get_trace_cwt()
def get_stream_cwt(stream, wavelet="cmorl", bandwidth=10.0, center_freq=1.0, scales=range(1, 128)):
    cwt_specs = WaveletSpectra()
    for trace in stream:
        cwt_spec = get_trace_cwt(trace, wavelet, bandwidth, center_freq, scales)
        cwt_specs.append(cwt_spec)
        
    return cwt_specs

## Function for getting the continuous wavelet transform of a trace
def get_trace_cwt(trace, wavelet="cmor", bandwidth=10.0, center_freq=1.0, scales=range(1, 128)):
    station  = trace.stats.station
    component = trace.stats.component
    location = trace.stats.location
    signal = trace.data
    timeax = get_timeax_from_trace(trace)
    sampling_rate = trace.stats.sampling_rate

    wavelet = f"{wavelet}{bandwidth}-{center_freq}"
    cwt_mat, freqs = cwt(signal, scales, wavelet, sampling_period=1 / sampling_rate)

    cwt_dict = {"station": station, "location":location, "component": component, "times": timeax, "freqs": freqs, "scales": scales, "data": cwt_mat, "wavelet": wavelet}
    cwt_spec = WaveletSpectrum(**cwt_dict)

    return cwt_spec

## Function for computing the cross spectrum and coherence of a pair of CWT spectra
## Smoothing function is a Gaussian along the time axis that scales with scale and a boxcar along the frequency (scale) axis according to Mao et al. (2020)
def get_cross_spectrum(spec1, spec2, boxcar_window=3, coherence=False):
    ### Compute the cross-spectrum
    cross_mat = spec1.data * spec2.data.conj()

    if coherence:
        ### Convert the width of the boxcar window to the closest odd integer
        if boxcar_window % 2 == 0:
            boxcar_window += 1
        
        ### Get the scale normalization factors
        scale_factors = get_scale_factors(spec1)

        ### Smooth the power
        power1 = spec1.get_power(db=False)
        power2 = spec2.get_power(db=False)

        power1 = power1 / scale_factors
        power2 = power2 / scale_factors

        scales = spec1.scales
        smoothed_power1 = smooth_along_freq(power1, boxcar_window)
        smoothed_power1 = smooth_along_time(smoothed_power1, scales)

        smoothed_power2 = smooth_along_freq(power2, boxcar_window)
        smoothed_power2 = smooth_along_time(smoothed_power2, scales)

        ## Smooth the cross-spectrum
        cross_mat = cross_mat / scale_factors
        smoothed_cross = smooth_along_freq(cross_mat, boxcar_window)
        smoothed_cross = smooth_along_time(smoothed_cross, scales)

        ## Compute the coherence
        cross_cohe = abs(smoothed_cross) ** 2 / (smoothed_power1 * smoothed_power2)

        return cross_mat, cross_cohe
    else:
        return cross_mat, None
    
## Function for extracting the frequency-phase pairs from a time-frequency image with high coherence and power
def extract_freq_phase_pairs(cross_specs, freqmin=None, freqmax=None, cohe_threshold=0.8, power_threshold=-20):
    freq_phi_dict = {}
    for cross_spec in cross_specs:
        station1 = cross_spec.station1
        station2 = cross_spec.station2
        component = cross_spec.component
        freqs = cross_spec.freqs
        timeax = cross_spec.times
        power = cross_spec.get_power()
        phase = cross_spec.get_phase()
        coherence = cross_spec.coherence

        if freqmin is None:
            freqmin = freqs.min()
        
        if freqmax is None:
            freqmax = freqs.max()

        freqs_window = freqs[(freqs > freqmin) & (freqs < freqmax)]
        power_window = power[(freqs > freqmin) & (freqs < freqmax), :]
        phase_window = phase[(freqs > freqmin) & (freqs < freqmax), :]
        cohe_window = coherence[(freqs > freqmin) & (freqs < freqmax), :]
        _, freq_grid = meshgrid(timeax, freqs_window)

        freqs_extract = freq_grid[(power_window > power_threshold) & (cohe_window > cohe_threshold)]
        phases_extract = phase_window[(power_window > power_threshold) & (cohe_window > cohe_threshold)]
        freq_phi_pairs = column_stack((freqs_extract, phases_extract))

        freq_phi_dict[(station1, station2, component)] = freq_phi_pairs

    return freq_phi_dict
    
## Function for smoothing the spectrum along the time axis
def smooth_along_time(data, scales):
    smoothed_data = zeros_like(data)
    for i, scale in enumerate(scales):
        kernel = get_gauss_kernel(scale)
        smoothed_data[i, :] = convolve(data[i, :], kernel, mode="same")

    return smoothed_data

## Function for smoothing the spectrum along the frequency (scale) axis
def smooth_along_freq(data, boxcar_window):
    kernel = get_boxcar_kernel(boxcar_window)
    smoothed_data = convolve2d(data, kernel, mode="same")

    return smoothed_data

## Function for constructing the 1D Gaussian kernel along the time axis
def get_gauss_kernel(scale):
    window_length = round(4 * scale)
    center_index = round(window_length / 2)
    x = arange(window_length)
    kernel = exp(-(x-center_index) ** 2 / (2 * scale ** 2))
    kernel = kernel / sum(kernel)

    return kernel

## Function for constructing the 2D boxcar kernel along the frequency (scale) axis
def get_boxcar_kernel(boxcar_window):
    kernel = ones((boxcar_window, 1)) / boxcar_window

    return kernel

## Get the scale normalization factors
def get_scale_factors(spec):
    scales = spec.scales
    num_times = spec.data.shape[1]
    scale_factors = tile(scales, (num_times, 1))
    scale_factors = scale_factors.T

    return scale_factors

## Function for masking the phase of the cross-spectra
## Threshold is power in dB
def mask_cross_phase(phase, coherence, power, cohe_threshold=0.8, power_threshold=45):
    phase[(coherence < cohe_threshold) | (power < power_threshold)] = nan

    return phase

## Get the frequencies corresponding to the scales
def get_cwt_freqs(scales, wavelet, bandwidth, center_freq, sampling_rate = SAMPLING_RATE_GEO):
    wavelet = f"{wavelet}{bandwidth}-{center_freq}"
    freqs = scale2frequency(wavelet, scales) * sampling_rate

    return freqs

## Get the noise level of a time-frequency spectrum
def get_noise_power(freqs, cwt_power, noise_window):
    freq_bool = (freqs >= noise_window[0]) & (freqs <= noise_window[1])
    noise_mat = cwt_power[freq_bool, :]
    noise_power = gmean(noise_mat, axis=None)

    return noise_power

## Function for getting the scales and frequencies of the chosen wavelet
def get_scales_and_freqs(wavelet, bandwidth, center_freq, min_scale, max_scale, num_scales=128, sampling_rate=SAMPLING_RATE_GEO):
    scales = geomspace(min_scale, max_scale, num_scales )
    freqs = get_cwt_freqs(scales, wavelet, bandwidth, center_freq, sampling_rate)

    return scales, freqs