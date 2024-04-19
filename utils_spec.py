# Functions and classes for spectrum analysis
from os.path import join
from scipy.fft import fft, fftfreq
from scipy.signal import iirfilter, sosfilt, freqz
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from numpy import abs, amax, array, column_stack, concatenate, convolve, cumsum, delete, load, linspace, ones, pi, savez
from pandas import Series, to_datetime
from h5py import File
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

    def __iter__(self):
        return iter(self.traces)
    
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

# Stitch spectrograms of multiple time periods together
# Output window length is in SECOND!
def stitch_spectrograms(stream_spec, average_window, overlap = 0.5):
    if len(stream_spec) < 2:
        raise ValueError("Error: Insufficient number of StreamSTFTPSD objects!")

    station = stream_spec[0].station
    component = stream_spec[0].component
    location = stream_spec[0].location
    freqax = stream_spec[0].freqs

    # Stich the spectrograms together
    for i, trace_spec in enumerate(stream_spec):
        if trace_spec.station != station or trace_spec.component != component or trace_spec.location != location:
            raise ValueError("Error: Inconsistent station, component or location in the StreamSTFTPSD objects!")
        
        if i == 0:
            timeax_out = trace_spec.times
            data_out = trace_spec.data
        else:
            timeax_in = trace_spec.times
            data_in = trace_spec.data

            if timeax_out[-1] >= timeax_in[0]:
                timeax_out = delete(timeax_out, -1)
                data_out = delete(data_out, -1, axis=1)

            data_out = column_stack([data_out, data_in])
            timeax_out = concatenate([timeax_out, timeax_in])

    # Average the spectrogram over the given window
    if average_window is not None:
        data_out = moving_average_2d(data_out, average_window, axis=1)

        timeax_sr = Series(timeax_out)
        timeax_sr = timeax_sr.astype('int64')
        timeax_out = timeax_sr.to_numpy()
        timeax_out = moving_average(timeax_out, average_window)
        timeax_sr = Series(timeax_out)
        timeax_sr = timeax_sr.astype('datetime64[ns]')
        timeax_out = timeax_sr.to_numpy()

    # Save the stitched spectrogram to a new TraceSTFTPSD object
    trace_spec = TraceSTFTPSD(station, location, component, timeax_out, freqax, data_out)

    return trace_spec

# Moving window average of a 1D array
def moving_average(data, window_length):
    if data.ndim != 1:
        raise ValueError("Error: data must be a 1D numpy array")
    
    window = ones(window_length) / window_length
    data = convolve(data, window, mode="same")

    return data

# Moving window average of a 2D array
def moving_average_2d(data, window_length, axis=0):
    if data.ndim != 2:
        raise ValueError("Error: data must be a 2D numpy array")
    
    numrows, numcols = data.shape
    window = ones(window_length) / window_length

    if axis == 0:
        for i in range(numcols):
            data[:, i] = convolve(data[:, i], window, mode="same")
    elif axis == 1:
        for i in range(numrows):
            data[i, :] = convolve(data[i, :], window, mode="same")

    return data

# Downsample a STFT spectrogram by averaging over a given number of points along the frequency axis
# Row axis is the frequency axis and column axis is the time axis!
def downsample_stft_spec(spec, factor=1000):
    numpts = spec.shape[0]
    numrows = numpts // factor
    numcols = spec.shape[1]

    spec = spec[:numrows * factor, :]
    spec = spec.reshape((numrows, factor, numcols))
    spec = spec.mean(axis=1)

    return spec

# Function to down sample the frequency axis of a spectrogram
def downsample_stft_freqax(freqax, factor=1000):
    numpts = len(freqax)
    numrows = numpts // factor

    freqax = freqax[:numrows * factor]
    freqax = freqax.reshape((numrows, factor))
    freqax = freqax.mean(axis=1)

    return freqax

# Function to calculate the frequency response of a filter 
# Currently only support Butterworth filters
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

# # Save the 3C spectrogram data of a geophone station in a time range to a .npz file
# def save_geo_spectrograms(stream_spec, filename, outdir = SPECTROGRAM_DIR):
#     if len(stream_spec) != 3:
#         raise ValueError("Error: Invalid number of spectrogram data!")

#     if stream_spec[0].starttime != stream_spec[1].starttime or stream_spec[1].starttime != stream_spec[2].starttime:
#         raise ValueError("Error: The spectrograms do not have the save start time!")

#     if stream_spec[0].station != stream_spec[1].station or stream_spec[1].station != stream_spec[2].station:
#         raise ValueError("Error: The spectrograms do not belong to the same station!")
    
#     trace_spec_z = stream_spec.select(component="Z")[0]
#     trace_spec_1 = stream_spec.select(component="1")[0]
#     trace_spec_2 = stream_spec.select(component="2")[0]

#     station = trace_spec_z.station
#     timeax = trace_spec_z.times
#     freqax = trace_spec_z.freqs
#     data_z = trace_spec_z.data
#     data_1 = trace_spec_1.data
#     data_2 = trace_spec_2.data

#     try:
#         outpath = join(outdir, filename)
#         savez(outpath, station = station, times = timeax, freqs = freqax, spectrogram_z = data_z, spectrogram_1 = data_1, spectrogram_2 = data_2)
#         print(f"Spectrogram data saved to {outpath}.")
#     except Exception as e:
#         print(f"Error saving the spectrogram data: {e}")

# Save the 3C spectrogram data of a geophone station to a HDF5 file
def save_geo_spectrograms(stream_spec, filename, outdir = SPECTROGRAM_DIR):
    components = GEO_COMPONENTS
    # Verify the StreamSTFTPSD object
    if len(stream_spec) != 3:
        raise ValueError("Error: Invalid number of spectrogram data!")

    if stream_spec[0].starttime != stream_spec[1].starttime or stream_spec[1].starttime != stream_spec[2].starttime:
        raise ValueError("Error: The spectrograms do not have the save start time!")

    if stream_spec[0].station != stream_spec[1].station or stream_spec[1].station != stream_spec[2].station:
        raise ValueError("Error: The spectrograms do not belong to the same station!")
    
    # Extract the data from the StreamSTFTPSD object
    trace_spec_z = stream_spec.select(component="Z")[0]
    trace_spec_1 = stream_spec.select(component="1")[0]
    trace_spec_2 = stream_spec.select(component="2")[0]

    station = trace_spec_z.station
    timeax = trace_spec_z.times
    freqax = trace_spec_z.freqs
    data_z = trace_spec_z.data
    data_1 = trace_spec_1.data
    data_2 = trace_spec_2.data

    # Convert the time axis to integer
    timeax = Series(timeax)
    timeax = timeax.astype('int64')
    timeax = timeax.to_numpy()

    # Save the data
    outpath = join(outdir, filename)
    try:
        with File(outpath, 'w') as file:
            # Create the group for storing the headers and data
            header_group = file.create_group('headers')
            data_group = file.create_group('data')

            # Save the header information
            # Encode the strings
            station_encode = station.encode("utf-8")
            components_encode = [component.encode("utf-8") for component in components]
            
            header_group.create_dataset('station', data = station_encode)
            header_group.create_dataset('components', data = components_encode)
            header_group.create_dataset('locations', data = [])
            header_group.create_dataset('starttime', data = timeax[0])
            header_group.create_dataset('time_interval', data = timeax[1] - timeax[0])
            header_group.create_dataset('frequency_interval', data = freqax[1] - freqax[0])

            # Save the spectrogram data
            data_group.create_dataset('psd_z', data = data_z)
            data_group.create_dataset('psd_1', data = data_1)
            data_group.create_dataset('psd_2', data = data_2)
    except Exception as e:
        print(f"Error saving the spectrogram data: {e}")

# Read the spectrogram data from an HDF5 file
def read_geo_spectrograms(inpath):
    try:
        with File(inpath, 'r') as file:
            # Read the header information
            header_group = file["headers"]
            station = header_group["station"][()]
            time_interval = header_group["time_interval"][()]
            freq_interval = header_group["frequency_interval"][()]
            starttime = header_group["starttime"][()]

            components = header_group["components"][:]
            locations = header_group["locations"][:]

            # Decode the strings
            station = station.decode("utf-8")

            # Read the spectrogram data
            data_group = file["data"]
            data_z = data_group["psd_z"][:]
            data_1 = data_group["psd_1"][:]
            data_2 = data_group["psd_2"][:]

            # Create the StreamSTFTPSD object
            num_time = data_z.shape[1]
            timeax = Series(range(num_time)) * time_interval + starttime
            timeax = to_datetime(timeax, unit='ns')
            timeax = timeax.to_list()
            print(type(timeax[0]))

            num_freq = data_z.shape[0]
            freqax = linspace(0, (num_freq - 1) * freq_interval, num_freq)

            trace_spec_z = TraceSTFTPSD(station, None, "Z", timeax, freqax, data_z)
            trace_spec_1 = TraceSTFTPSD(station, None, "1", timeax, freqax, data_1)
            trace_spec_2 = TraceSTFTPSD(station, None, "2", timeax, freqax, data_2)

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
