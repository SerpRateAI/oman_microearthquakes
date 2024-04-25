# Functions and classes for spectrum analysis
from os.path import join
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, iirfilter, sosfilt, freqz
from scipy.signal.windows import hann
from numpy import abs, amax, array, column_stack, concatenate, convolve, cumsum, delete, load, linspace, nan, ones, pi, savez
from pandas import Series, DataFrame
from pandas import to_datetime, date_range
from h5py import File, special_dtype
from multitaper import MTSpec

from utils_basic import GEO_COMPONENTS
from utils_basic import SPECTROGRAM_DIR 
from utils_basic import assemble_timeax_from_ints, datetime2int, reltimes_to_timestamps, power2db
from utils_preproc import read_and_process_day_long_geo_waveforms

######
# Classes
######

# Class for storing the STFT data of multiple traces
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

    def select(self, station = None, location = None, component = None, time_label = None):
        traces = []
        if station is not None:
            if location is not None:
                if component is not None:
                    if time_label is not None:
                        traces = [trace for trace in self.traces if trace.station == station and trace.location == location and trace.component == component and time_label == time_label]
                    else:
                        traces = [trace for trace in self.traces if trace.station == station and trace.location == location and trace.component == component]
                else:
                    if time_label is not None:
                        traces = [trace for trace in self.traces if trace.station == station and trace.location == location and trace.time_label == time_label]
                    else:
                        traces = [trace for trace in self.traces if trace.station == station and trace.location == location]
            else:
                if component is not None:
                    if time_label is not None:
                        traces = [trace for trace in self.traces if trace.station == station and trace.component == component and trace.time_label == time_label]
                    else:
                        traces = [trace for trace in self.traces if trace.station == station and trace.component == component]
                else:
                    if time_label is not None:
                        traces = [trace for trace in self.traces if trace.station == station and trace.time_label == time_label]
                    else:
                        traces = [trace for trace in self.traces if trace.station == station]
        else:
            if location is not None:
                if component is not None:
                    if time_label is not None:
                        traces = [trace for trace in self.traces if trace.location == location and trace.component == component and trace.time_label == time_label]
                    else:
                        traces = [trace for trace in self.traces if trace.location == location and trace.component == component]
                else:
                    if time_label is not None:
                        traces = [trace for trace in self.traces if trace.location == location and trace.time_label == time_label]
                    else:
                        traces = [trace for trace in self.traces if trace.location == location]           
            else:
                if component is not None:
                    if time_label is not None:
                        traces = [trace for trace in self.traces if trace.component == component and trace.time_label == time_label]
                    else:
                        traces = [trace for trace in self.traces if trace.component == component]
                else:
                    if time_label is not None:
                        traces = [trace for trace in self.traces if trace.time_label == time_label]
                    else:
                        traces = self.traces
                        
        stream = StreamSTFTPSD(traces)
        return stream
    
    # Get the list of stations in the stream
    def get_stations(self):
        stations = list(set([trace.station for trace in self.traces]))
        stations.sort()

        return stations

    # Get the list of locations in the stream
    def get_locations(self):
        locations = list(set([trace.location for trace in self.traces]))
        locations.sort()

        return locations

    # Get the list of time labels in the stream
    def get_time_labels(self):
        time_labels = list(set([trace.time_label for trace in self.traces]))
        time_labels.sort()

        return time_labels
    
    # Convert the power spectrograms to dB
    def to_db(self):
        for trace in self.traces:
            trace.to_db()

    # Trim the spectrograms to a given time range
    def trim(self, starttime = None, endtime = None):
        for trace in self.traces:
            trace.trim(starttime, endtime)

    # Trim the spectrograms to the begin and end of the day
    def trim_to_day(self):
        for trace in self.traces:
            trace.trim_to_day()

    # Get the total power spectrogram of the geophone station
    def get_total_power(self):
        if len(self.traces) != 3:
            raise ValueError("Error: Invalid number of components!")
        
        if self.traces[0].station != self.traces[1].station or self.traces[1].station != self.traces[2].station:
            raise ValueError("Error: Inconsistent station names!")
        
        if self.traces[0].db or self.traces[1].db or self.traces[2].db:
            raise ValueError("Error: The power spectrograms must be in linear scale!")

        data_total = self.traces[0].data + self.traces[1].data + self.traces[2].data
        trace_total = self.traces[0].copy()
        trace_total.data = data_total
        trace_total.component = None

        return trace_total
        
# Class for storing the STFT data and associated parameters of a trace
class TraceSTFTPSD:
    def __init__(self, station, location, component, time_label, times, freqs, data, overlap=0, db=False):
        self.station = station
        self.location = location
        self.component = component
        self.time_label = time_label
        self.times = times
        self.freqs = freqs
        self.overlap = overlap
        self.data = data
        self.db = db

    # Convert the power spectrogram to dB
    def to_db(self):
        if self.db:
            return
        else:
            self.data = power2db(self.data)
            self.db = True

    # Make a deep copy of the object
    def copy(self):
        return TraceSTFTPSD(self.station, self.location, self.component, self.time_label, self.times, self.freqs, self.data, self.db)

    # Trim the spectrogram to a given time range
    def trim(self, starttime = None, endtime = None):
        timeax = self.times
    
        if starttime is not None:
            start_index = timeax.searchsorted(starttime)
        else:
            start_index = 0

        if endtime is not None:
            end_index = timeax.searchsorted(endtime)
        else:
            end_index = len(timeax)

        self.times = timeax[start_index:end_index]
        self.data = self.data[:, start_index:end_index]

    # Trim the spectrogram to the begin and end of the day
    def trim_to_day(self):
        timeax = self.times
        num_time = len(timeax)
        ref_time = timeax[num_time // 2]
        start_of_day = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = ref_time.replace(hour=23, minute=59, second=59, microsecond=999999)

        self.trim(start_of_day, end_of_day)

    # Find spectral peaks satisfying the given criteria
    # The power threshold is in dB!
    def find_spectral_peaks(self, power_threshold, qf_threshold, freqmin = None, freqmax = None):
        # Convert to dB
        self.to_db()

        # Trim the data to the given frequency range
        if freqmin is None:
            freqmin = self.freqs[0]

        if freqmax is None:
            freqmax = self.freqs[-1]

        freq_inds = (self.freqs >= freqmin) & (self.freqs <= freqmax)
        freqax = self.freqs[freq_inds]
        data = self.data[freq_inds, :]

        # Find the spectral peaks
        peak_freqs = []
        peak_times = []
        peak_powers = []
        peak_qfs = []
        for i, time in enumerate(self.times):
            power = data[:, i]
            peak_inds, _ = find_peaks(power, height=power_threshold)

            for j in peak_inds:
                freq = freqax[j]
                qf = get_quality_factor(freqax, power, freq)
                if qf >= qf_threshold:
                    peak_freqs.append(freq)
                    peak_times.append(time)
                    peak_powers.append(power[j])
                    peak_qfs.append(qf)
        
        peak_df = DataFrame({"frequency": peak_freqs, "time": peak_times, "power": peak_powers, "quality_factor": peak_qfs})

        return peak_df

######
# Functions
######

###### Functions for handling STFT spectrograms ######

# Find spectral peaks in the spectrograms of a geophone station
# A Pandas DataFrame containing the frequency, time, power, and quality factor of each peak is returned
# The power threshold is in dB!
def find_geo_station_spectral_peaks(stream_spec, power_threshold = 0.0, qf_threshold = 200.0, freqmin = None, freqmax = None):
    # Verify the StreamSTFTPSD object
    if len(stream_spec) != 3:
        raise ValueError("Error: Invalid number of components!")

    # Compute the total power spectrogram
    trace_spec_total = stream_spec.get_total_power()

    # Find the spectral peaks in each component
    peak_df = trace_spec_total.find_spectral_peaks(power_threshold, qf_threshold, freqmin = freqmin, freqmax = freqmax)

    return peak_df, trace_spec_total

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

# Downsample an STFT stream object along the frequency axis
def downsample_stft_stream_freq(stream_in, factor=1000):
    stream_out = StreamSTFTPSD()
    for trace_in in stream_in:
        trace_out = downsample_stft_trace_freq(trace_in, factor)
        stream_out.append(trace_out)

    return stream_out

# Downsample an STFT trace object along the frequency axis
def downsample_stft_trace_freq(trace_in, factor=1000):
    trace_out = trace_in.copy()

    freqax = trace_in.freqs
    freqax_ds = downsample_stft_freqax(freqax, factor)
    data = trace_in.data
    data_ds = downsample_stft_data_freq(data, factor)

    trace_out.freqs = freqax_ds
    trace_out.data = data_ds

    return trace_out

# Downsample an STFT data matrix along the frequency axis by averaging over a given number of points along the frequency axis
def downsample_stft_data_freq(data, factor=1000):
    numpts = data.shape[0]
    numrows = numpts // factor
    numcols = data.shape[1]

    data = data[:numrows * factor, :]
    data = data.reshape((numrows, factor, numcols))
    data = data.mean(axis=1)

    return data

# Downsample the frequency axis of an STFT data matrix
def downsample_stft_freqax(freqax, factor=1000):
    numpts = len(freqax)
    numrows = numpts // factor

    freqax = freqax[:numrows * factor]
    freqax = freqax.reshape((numrows, factor))
    freqax = freqax.mean(axis=1)

    return freqax

# Assemble the name of a spectrogram file
def assemble_spec_filename(range_type, sensor_type, time_label, station, window_length, overlap, downsample, **kwargs):
    if downsample:
        downsample_factor = kwargs["downsample_factor"]
        filename = f"{range_type}_{sensor_type}_spectrograms_{time_label}_{station}_{window_length:.0f}s_{overlap:.1f}_downsample{downsample_factor:d}.h5"
    else:
        filename = f"{range_type}_{sensor_type}_spectrograms_{time_label}_{station}_{window_length}s_{overlap}.h5"

    return filename

# Save the 3C spectrogram data of a geophone station to a HDF5 file
def save_geo_spectrograms(stream_spec, filename, outdir = SPECTROGRAM_DIR):
    components = GEO_COMPONENTS
    # Verify the StreamSTFTPSD object
    if len(stream_spec) != 3:
        raise ValueError("Error: Invalid number of components!")

    if stream_spec[0].time_label != stream_spec[1].time_label or stream_spec[1].time_label != stream_spec[2].time_label:
        raise ValueError("Error: The spectrograms do not have the save start time!")

    if stream_spec[0].station != stream_spec[1].station or stream_spec[1].station != stream_spec[2].station:
        raise ValueError("Error: The spectrograms do not belong to the same station!")
    
    # Extract the data from the StreamSTFTPSD object
    trace_spec_z = stream_spec.select(component="Z")[0]
    trace_spec_1 = stream_spec.select(component="1")[0]
    trace_spec_2 = stream_spec.select(component="2")[0]

    station = trace_spec_z.station
    time_label = trace_spec_z.time_label
    timeax = trace_spec_z.times
    freqax = trace_spec_z.freqs
    overlap = trace_spec_z.overlap
    data_z = trace_spec_z.data
    data_1 = trace_spec_1.data
    data_2 = trace_spec_2.data

    # Convert the time axis to integer
    timeax = datetime2int(timeax)

    # Save the data
    outpath = join(outdir, filename)
    with File(outpath, 'w') as file:
        # Create the group for storing the headers and data
        header_group = file.create_group('headers')
    
        # Save the header information with encoding
        header_group.create_dataset('station', data=station.encode("utf-8"))
        header_group.create_dataset('time_label', data=time_label.encode("utf-8"))
        header_group.create_dataset('components', data=[component.encode("utf-8") for component in components])

        header_group.create_dataset("start_time", data=timeax[0])
        header_group.create_dataset('time_interval', data=timeax[1] - timeax[0])
        header_group.create_dataset('frequency_interval', data=freqax[1] - freqax[0])
        header_group.create_dataset('overlap', data=overlap)
    
        # Create the group for storing each component's spectrogram data
        data_group = file.create_group('data')
        for component in components:
            comp_group = data_group.create_group(component)
            if component == "Z":
                comp_group.create_dataset("psd", data=data_z)
            elif component == "1":
                comp_group.create_dataset("psd", data=data_1)
            elif component == "2":
                comp_group.create_dataset("psd", data=data_2)

        print(f"Spectrograms saved to {outpath}")

# Save all locations of a hydrophone station to a HDF5 file
# Each location has its own time axis!
def save_hydro_spectrograms(stream_spec, filename, outdir = SPECTROGRAM_DIR):
    # Verify the StreamSTFTPSD object
    locations = stream_spec.get_locations()
    if len(stream_spec) != len(locations):
        raise ValueError("Error: Number of locations is inconsistent with spectrograms!")

    outpath = join(outdir, filename)
    # Save the data
    with File(outpath, 'w') as file:
        # Create the group for storing each location's spectrogram data
        data_group = file.create_group('data')  
        for i, location in enumerate(locations):
            trace_spec = stream_spec.select(location=location)[0]
            timeax = trace_spec.times
            freqax = trace_spec.freqs
            overlap = trace_spec.overlap

            # Convert the time axis to integer
            timeax = datetime2int(timeax)
            
            if i == 0:
                time_label = trace_spec.time_label
                station = trace_spec.station
            else:
                if time_label != trace_spec.time_label:
                    raise ValueError("Error: The spectrograms do not have the same start time!")
                
                if station != trace_spec.station:
                    raise ValueError("Error: The spectrograms do not belong to the same station!")

            data = trace_spec.data
            loc_group = data_group.create_group(location)
            loc_group.create_dataset("psd", data=data)
            loc_group.create_dataset('start_time', data=timeax[0])
            loc_group.create_dataset('time_interval', data=timeax[1] - timeax[0])

        # Create the group for storing the headers
        header_group = file.create_group('headers')
    
        # Save the header information with encoding
        header_group.create_dataset('station', data=station.encode("utf-8"))
        header_group.create_dataset('time_label', data=time_label.encode("utf-8"))
        header_group.create_dataset('locations', data=[location.encode("utf-8") for location in locations])

        header_group.create_dataset('frequency_interval', data=freqax[1] - freqax[0])
        header_group.create_dataset('overlap', data=overlap)

    print(f"Spectrograms saved to {outpath}")

# Read the geophone spectrograms from an HDF5 file
def read_geo_spectrograms(inpath):
    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        station = header_group["station"][()]
        time_label = header_group["time_label"][()]
        components = header_group["components"][:]
        
        starttime = header_group["start_time"][()]
        time_interval = header_group["time_interval"][()]
        freq_interval = header_group["frequency_interval"][()]
        overlap = header_group["overlap"][()]
        
        # Decode the strings
        station = station.decode("utf-8")
        time_label = time_label.decode("utf-8")
        components = [component.decode("utf-8") for component in components]

        # Read the spectrogram data
        data_group = file["data"]
        stream_spec = StreamSTFTPSD()
        for component in components:
            comp_group = data_group[component]
            data = comp_group["psd"][:]

            num_freq = data.shape[0]
            freqax = linspace(0, (num_freq - 1) * freq_interval, num_freq)

            num_time = data.shape[1]
            timeax = assemble_timeax_from_ints(starttime, num_time, time_interval)
        
            trace_spec = TraceSTFTPSD(station, None, component, time_label, timeax, freqax, data)
            stream_spec.append(trace_spec)

        return stream_spec

# Read the hydrophone spectrograms of ALL locations of one stations from an HDF5 file
# Each location has its own time axis!
def read_hydro_spectrograms(inpath):
    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        station = header_group["station"][()]
        time_label = header_group["time_label"][()]
        locations = header_group["locations"][:]

        freq_interval = header_group["frequency_interval"][()]
        overlap = header_group["overlap"][()]

        # Decode the strings
        station = station.decode("utf-8")
        time_label = time_label.decode("utf-8")
        locations = [location.decode("utf-8") for location in locations]

        # Read the spectrogram data
        data_group = file["data"]
        stream_spec = StreamSTFTPSD()
        for location in locations:
            loc_group = data_group[location]
            data = loc_group["psd"][:]
            starttime = loc_group["start_time"][()]
            time_interval = loc_group["time_interval"][()]

            num_freq = data.shape[0]
            freqax = linspace(0, (num_freq - 1) * freq_interval, num_freq)
            
            num_time = data.shape[1]
            timeax = assemble_timeax_from_ints(starttime, num_time, time_interval)
            
            trace_spec = TraceSTFTPSD(station, location, "H", time_label, timeax, freqax, data)
            stream_spec.append(trace_spec)

        return stream_spec    
    

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

###### Basic functions ######

# Get the quality factor of a peak in a power spectrum
def get_quality_factor(freqax, power, freq0, db=False):
    if not db:
        power = power2db(power)

    # Find the peak point
    ipeak = freqax.searchsorted(freq0)
    power0 = power[ipeak]
    power3db = power0 - 3.0

    # Find the points where power drops below the 3 dB level
    for i in range(ipeak, len(power)):
        if power[i] < power3db:
            break
    freq_high = freqax[i]

    for i in range(ipeak, 0, -1):
        if power[i] < power3db:
            break
    freq_low = freqax[i]

    quality_factor = freq0 / (freq_high - freq_low)

    return quality_factor




    return quality_factor
