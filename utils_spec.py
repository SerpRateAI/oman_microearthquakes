# Functions and classes for spectrum analysis
from os.path import join
from scipy.fft import fft, fftfreq
from scipy.signal import iirfilter, sosfilt, freqz
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from numpy import abs, amax, array, column_stack, concatenate, cumsum, delete, load, pi, savez
from multitaper import MTSpec

from utils_basic import GEO_COMPONENTS
from utils_basic import SPECTROGRAM_DIR 
from utils_basic import reltimes_to_timestamps, power2db
from utils_preproc import read_and_process_day_long_geo_waveforms

# Classes

## Class for storing the STFT traces of multiple traces
class StreamSTFTPSD:
    def __init__(self, traces = None):
        if traces is not None:
            if not isinstance(traces, list):
                raise TypeError("Invalid TraceSTFTPSD objects!")
            else:
                self.traces = traces
        else:
            self.traces = []

    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, index):
        return self.traces[index]
    
    def __setitem__(self, index, value):
        self.traces[index] = value

    def __delitem__(self, index):
        del self.traces[index]
    
    def append(self, trace_spec):
        if not isinstance(trace_spec, TraceSTFTPSD):
            raise TypeError("Invalid TraceSTFTPSD object!")

        self.traces.append(trace_spec)

    def extend(self, stream_spec):
        if not isinstance(stream_spec, StreamSTFTPSD):
            raise TypeError("Invalid StreamSTFTPSD object!")

        self.traces.extend(stream_spec.traces)

    def select(self, station = None, location = None, component = None, starttime = None):
        traces = []
        if station is not None:
            if location is not None:
                if component is not None:
                    if starttime is not None:
                        traces = [trace for trace in self.traces if trace.station == station and trace.location == location and trace.component == component and starttime == starttime]
                    else:
                        traces = [trace for trace in self.traces if trace.station == station and trace.location == location and trace.component == component]
                else:
                    if starttime is not None:
                        traces = [trace for trace in self.traces if trace.station == station and trace.location == location and trace.starttime == starttime]
                    else:
                        traces = [trace for trace in self.traces if trace.station == station and trace.location == location]
            else:
                if component is not None:
                    if starttime is not None:
                        traces = [trace for trace in self.traces if trace.station == station and trace.component == component and trace.starttime == starttime]
                    else:
                        traces = [trace for trace in self.traces if trace.station == station and trace.component == component]
                else:
                    if starttime is not None:
                        traces = [trace for trace in self.traces if trace.station == station and trace.starttime == starttime]
                    else:
                        traces = [trace for trace in self.traces if trace.station == station]
        else:
            if location is not None:
                if component is not None:
                    if starttime is not None:
                        traces = [trace for trace in self.traces if trace.location == location and trace.component == component and trace.starttime == starttime]
                    else:
                        traces = [trace for trace in self.traces if trace.location == location and trace.component == component]
                else:
                    if starttime is not None:
                        traces = [trace for trace in self.traces if trace.location == location and trace.starttime == starttime]
                    else:
                        traces = [trace for trace in self.traces if trace.location == location]           
            else:
                if component is not None:
                    if starttime is not None:
                        traces = [trace for trace in self.traces if trace.component == component and trace.starttime == starttime]
                    else:
                        traces = [trace for trace in self.traces if trace.component == component]
                else:
                    if starttime is not None:
                        traces = [trace for trace in self.traces if trace.starttime == starttime]
                    else:
                        traces = self.traces
                        
        stream = StreamSTFTPSD(traces)
        return stream
    
    def get_stations(self):
        stations = list(set([trace.station for trace in self.traces]))
        stations.sort()

        return stations

    def get_starttimes(self):
        starttimes = list(set([trace.starttime for trace in self.traces]))
        starttimes.sort()

        return starttimes
    
    def to_db(self):
        for trace in self.traces:
            trace.to_db()
        
## Class for storing the STFT spectrum and associated parameters of a trace
class TraceSTFTPSD:
    def __init__(self, station, location, component, times, freqs, data, db=False):
        self.station = station
        self.location = location
        self.component = component
        self.times = times
        self.freqs = freqs
        self.data = data
        self.db = db

        starttime = times[0]
        self.starttime = starttime.strftime("%Y%m%d%H%M%S")

    def to_db(self):
        if self.db:
            return
        else:
            self.data = power2db(self.data)
            self.db = True

## Funcion for computing the spectrogram of a station during the entire deployment period
## Window length is in MINUTES!
def get_whole_deployment_spectrogram(station, days, window_length = 15.0, overlap = 0.5, downsample_factor = 1000, db=False):
    print("######")
    print(f"Computing spectrograms for {station}...")
    print("######")

    timeax_period = array([])
    window_length = window_length * 60 # Convert to seconds

    for i, day in enumerate(days):
        print(f"Processing day {day}...")

        print(f"Reading the waveforms...")
        stream = read_and_process_day_long_geo_waveforms(day, stations = [station])

        if stream is None:
            print(f"No data for day {day}!")
            continue

        print(f"Computing the spectrograms...")
        stft_dict = get_stream_spectrogram_stft(stream, window_length, overlap=overlap, db=db)

        print(f"Downsampling the spectrograms...")
        timeax_day = stft_dict[(station, "Z")][0]
        if i == 0:
            freqax = stft_dict[(station, "Z")][1]
            freqax_ds = downsample_stft_freqax(freqax, downsample_factor)

        day_mat_z = stft_dict[(station, "Z")][2]
        day_mat_1 = stft_dict[(station, "1")][2]
        day_mat_2 = stft_dict[(station, "2")][2]

        day_mat_z_ds = downsample_stft_spec(day_mat_z, downsample_factor)
        day_mat_1_ds = downsample_stft_spec(day_mat_1, downsample_factor)
        day_mat_2_ds = downsample_stft_spec(day_mat_2, downsample_factor)

        if i > 0:
            if timeax_day[0] <= timeax_period[-1]:
                timeax_period = delete(timeax_period, -1)
                period_mat_z = delete(period_mat_z, -1, axis=1)
                period_mat_1 = delete(period_mat_1, -1, axis=1)
                period_mat_2 = delete(period_mat_2, -1, axis=1)

        timeax_period = concatenate([timeax_period, timeax_day])

        if i == 0:
            period_mat_z = day_mat_z_ds
            period_mat_1 = day_mat_1_ds
            period_mat_2 = day_mat_2_ds
        else:
            period_mat_z = column_stack([period_mat_z, day_mat_z_ds])
            period_mat_1 = column_stack([period_mat_1, day_mat_1_ds])
            period_mat_2 = column_stack([period_mat_2, day_mat_2_ds])

        print(f"Done processing day {day}.")

    if db:
        period_mat_z = power2db(period_mat_z)
        period_mat_1 = power2db(period_mat_1)
        period_mat_2 = power2db(period_mat_2)

    spec_z = TraceSTFTPSD(station, None, "Z", timeax_period, freqax_ds, period_mat_z)
    spec_1 = TraceSTFTPSD(station, None, "1", timeax_period, freqax_ds, period_mat_1)
    spec_2 = TraceSTFTPSD(station, None, "2", timeax_period, freqax_ds, period_mat_2)

    specs = StreamSTFTPSD([spec_z, spec_1, spec_2])

    return specs


# Compute the spectrogram in PSD of a stream using STFT
# A wrapper for the get_trace_spectrogram_stft function
def get_stream_spectrogram_stft(stream, window_length=1.0, overlap=0.5, db=True):
    specdict = {}
    for trace in stream:
        station = trace.stats.station
        component = trace.stats.component

        timeax, freqax, spec = get_trace_spectrogram_stft(trace, window_length, overlap, db=db)
        specdict[(station, component)] = (timeax, freqax, spec)

    return specdict

# Compute the spectrogram in PSD of a trace using STFT
def get_trace_spectrogram_stft(trace, window_length=1.0, overlap=0.5, db=True):
    signal = trace.data
    sampling_rate = trace.stats.sampling_rate
    starttime = trace.stats.starttime
    starttime = starttime.datetime

    window = hann(int(window_length * sampling_rate))
    hop = int(window_length * sampling_rate * (1 - overlap))
    stft = ShortTimeFFT(window, hop, sampling_rate, scale_to="psd")
    freqax = array(stft.f)
    timeax = stft.t(len(signal))
    timeax = reltimes_to_timestamps(timeax, starttime)

    psd = stft.spectrogram(signal, detr="linear")

    if db:
        psd = power2db(psd)

    return timeax, freqax, psd

## Downsample a STFT spectrogram by averaging over a given number of points along the frequency axis
## Row axis is the frequency axis and column axis is the time axis!
def downsample_stft_spec(spec, factor=1000):
    numpts = spec.shape[0]
    numrows = numpts // factor
    numcols = spec.shape[1]

    spec = spec[:numrows * factor, :]
    spec = spec.reshape((numrows, factor, numcols))
    spec = spec.mean(axis=1)

    return spec

## Function to down sample the frequency axis of a spectrogram
def downsample_stft_freqax(freqax, factor=1000):
    numpts = len(freqax)
    numrows = numpts // factor

    freqax = freqax[:numrows * factor]
    freqax = freqax.reshape((numrows, factor))
    freqax = freqax.mean(axis=1)

    return freqax

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

# Save the 3C spectrogram data of a geophone station in a time range to a .npz file
def save_geo_spectrograms(stream_spec, filename, outdir = SPECTROGRAM_DIR):
    if len(stream_spec) != 3:
        raise ValueError("Error: Invalid number of spectrogram data!")

    if stream_spec[0].starttime != stream_spec[1].starttime or stream_spec[1].starttime != stream_spec[2].starttime:
        raise ValueError("Error: The spectrograms do not have the save start time!")

    if stream_spec[0].station != stream_spec[1].station or stream_spec[1].station != stream_spec[2].station:
        raise ValueError("Error: The spectrograms do not belong to the same station!")
    
    trace_spec_z = stream_spec.select(component="Z")[0]
    trace_spec_1 = stream_spec.select(component="1")[0]
    trace_spec_2 = stream_spec.select(component="2")[0]

    station = trace_spec_z.station
    timeax = trace_spec_z.times
    freqax = trace_spec_z.freqs
    data_z = trace_spec_z.data
    data_1 = trace_spec_1.data
    data_2 = trace_spec_2.data

    try:
        outpath = join(outdir, filename)
        savez(outpath, station = station, times = timeax, freqs = freqax, spectrogram_z = data_z, spectrogram_1 = data_1, spectrogram_2 = data_2)
        print(f"Spectrogram data saved to {outpath}.")
    except Exception as e:
        print(f"Error saving the spectrogram data: {e}")

# Read the spectrogram data from a .npz file
def read_geo_spectrograms(inpath):
    try:
        data = load(inpath, allow_pickle=True)
        station = data["station"]
        timeax = data["times"]
        freqax = data["freqs"]
        spec_z = data["spectrogram_z"]
        spec_1 = data["spectrogram_1"]
        spec_2 = data["spectrogram_2"]

        trace_spec_z = TraceSTFTPSD(station, None, "Z", timeax, freqax, spec_z)
        trace_spec_1 = TraceSTFTPSD(station, None, "1", timeax, freqax, spec_1)
        trace_spec_2 = TraceSTFTPSD(station, None, "2", timeax, freqax, spec_2)

        stream_spec = StreamSTFTPSD([trace_spec_z, trace_spec_1, trace_spec_2])

        return stream_spec
    except Exception as e:
        print(f"Error reading the spectrogram data: {e}")
        return None
    

# Calculate the VELOCITY PSD of a VELOCITY signal using multitaper method
def get_vel_psd_mt(vel, sampling_rate=1000.0, nw=2):    
    mt = MTSpec(vel, nw=nw, dt=1/sampling_rate)
    freqax, psd = mt.rspec()
    
    return freqax, psd

# Calculate the DISPLACEMENT PSD of a VELOCITY using multitaper method
def get_disp_psd_mt(vel, sampling_rate=1000.0, nw=2):
    disp = vel2disp(vel, sampling_rate)
    
    mt = MTSpec(disp, nw=nw, dt=1/sampling_rate)
    freqax, psd = mt.rspec()
    
    return freqax, psd
    
# Convert velocity to displacement
def vel2disp(vel, sampling_rate=1000.0):
    disp = cumsum(vel) / sampling_rate

    return disp
