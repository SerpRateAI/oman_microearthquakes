# Functions and classes for spectrum analysis
from os.path import join
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, iirfilter, sosfilt, freqz
from scipy.signal.windows import hann
from numpy import abs, amax, array, column_stack, concatenate, convolve, cumsum, delete, load, linspace, nan, ones, pi, savez
from pandas import Series, DataFrame, Timedelta, Timestamp
from pandas import date_range, concat
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
    
    # Append a TraceSTFTPSD object to the stream
    def append(self, trace_spec):
        if not isinstance(trace_spec, TraceSTFTPSD):
            raise TypeError("Invalid TraceSTFTPSD object!")

        self.traces.append(trace_spec)

    # Delete the TraceSTFTPSD objects contained in the input StreamSTFTPSD object
    def delete(self, stream_spec):
        if not isinstance(stream_spec, StreamSTFTPSD):
            raise TypeError("Invalid StreamSTFTPSD object!")

        self.traces = [trace for trace in self.traces if trace not in stream_spec.traces]

    # Extend the stream with another StreamSTFTPSD object
    def extend(self, stream_spec):
        if not isinstance(stream_spec, StreamSTFTPSD):
            raise TypeError("Invalid StreamSTFTPSD object!")

        self.traces.extend(stream_spec.traces)

    # Select traces based on station, location, component, and time label
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
    
    # Stitch the spectrograms of multiple time periods together in place
    def stitch(self, stations = None, locations = None, components = None, fill_value = nan):
        # Determine the stations, locations, and components to stitch
        if stations is None:
            stations = self.get_stations()
        
        if locations is None:
            locations = self.get_locations()

        if components is None:
            components = GEO_COMPONENTS

        # Stitch the spectrograms of each station, location, and component
        for station in stations:
            for location in locations:
                for component in components:
                    stream_to_stitch = self.select(station=station, location=location, component=component)
                    if len(stream_to_stitch) > 0:
                        # Stitch the spectrograms
                        trace_spec = stitch_spectrograms(stream_to_stitch, fill_value)
                        self.append(trace_spec)

                        # Delete the original traces
                        self.delete(stream_to_stitch)
    
    # Sort the traces by start time
    def sort_by_time(self):
        self.traces.sort(key=lambda trace: trace.times[0])
    
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

        if self.traces[0].data.shape != self.traces[1].data.shape or self.traces[1].data.shape != self.traces[2].data.shape:
            raise ValueError("Error: Inconsistent data matrix shapes!")

        data_total = self.traces[0].data + self.traces[1].data + self.traces[2].data
        trace_total = self.traces[0].copy()
        trace_total.data = data_total
        trace_total.component = None

        return trace_total

    # Verify if the traces have the same station name, time labels, and the three components (what called a block)
    def verify_geo_block(self):
        if len(self.traces) != 3:
            raise ValueError("Error: Invalid number of components!")
        
        if self.traces[0].station != self.traces[1].station or self.traces[1].station != self.traces[2].station:
            raise ValueError("Error: Inconsistent station names!")
        
        if self.traces[0].time_label != self.traces[1].time_label or self.traces[1].time_label != self.traces[2].time_label:
            raise ValueError("Error: Inconsistent time labels!")
        
        if self.traces[0].component != "Z" or self.traces[1].component != "1" or self.traces[2].component != "2":
            raise ValueError("Error: Invalid component names!")
        
    # Verify if the traces in the stream have the same staiton name, location, and component
    def verify_id(self):
        if len(self.traces) < 2:
            raise ValueError("Error: No traces in the stream!")

        station = self.traces[0].station
        location = self.traces[0].location
        component = self.traces[0].component

        for trace in self.traces:
            if trace.station != station or trace.location != location or trace.component != component:
                raise ValueError("Error: Inconsistent station, location, or component names!")
        
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

        self.num_times = len(self.times)
        self.num_freqs = len(self.freqs)

        self.time_interval = self.times[1] - self.times[0]
        self.freq_interval = self.freqs[1] - self.freqs[0]

        self.start_time = self.times[0]

    def __eq__(self, other):
        id_flag = self.station == other.station and self.location == other.location and self.component == other.component
        time_flag = self.num_times == other.num_times and self.time_interval == other.time_interval and self.start_time == other.start_time
        freq_flag = self.num_freqs == other.num_freqs and self.freq_interval == other.freq_interval
        
        return id_flag and time_flag and freq_flag

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
    def find_spectral_peaks(self, prom_threshold, rbw_threshold, freqmin = None, freqmax = None):
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
        peak_rbws = []
        for i, time in enumerate(self.times):
            power = data[:, i]
            peak_inds, _ = find_peaks(power, prominence = prom_threshold)

            for j in peak_inds:
                freq = freqax[j]
                _, rbw = get_quality_factor(freqax, power, freq)
                if rbw >= rbw_threshold:
                    peak_freqs.append(freq)
                    peak_times.append(time)
                    peak_powers.append(power[j])
                    peak_rbws .append(rbw)
        
        peak_df = DataFrame({"frequency": peak_freqs, "time": peak_times, "power": peak_powers, "reverse_bandwidth": peak_rbws})

        return peak_df

######
# Functions
######

###### Functions for handling STFT spectrograms ######

# Find spectral peaks in the spectrograms of a geophone station
# A Pandas DataFrame containing the frequency, time, power, and quality factor of each peak is returned
# The power threshold is in dB!
def find_geo_station_spectral_peaks(stream_spec, rbw_threshold = 0.2, prom_threshold = 5, freqmin = None, freqmax = None):
    # Verify the StreamSTFTPSD object
    if len(stream_spec) != 3:
        raise ValueError("Error: Invalid number of components!")

    # Compute the total power spectrogram
    trace_spec_total = stream_spec.get_total_power()

    # Find the spectral peaks in each component
    peak_df = trace_spec_total.find_spectral_peaks(prom_threshold, rbw_threshold, freqmin = freqmin, freqmax = freqmax)

    return peak_df, trace_spec_total

# Stitch spectrograms of multiple time periods together
# Output window length is in SECOND!
def stitch_spectrograms(stream_spec, fill_value = nan):
    if len(stream_spec) < 2:
        raise ValueError("Error: Insufficient number of StreamSTFTPSD objects!")

    # Verify if the StreamSTFTPSD objects have the same station, component, and location
    stream_spec.verify_id()

    # Sort the traces by start time
    stream_spec.sort_by_time()

    # Stich the spectrograms together
    station = stream_spec[0].station
    location = stream_spec[0].location
    component = stream_spec[0].component
    time_label = stream_spec[0].time_label
    freqax = stream_spec[0].freqs
    for i, trace_spec in enumerate(stream_spec):
        if i == 0:
            timeax_out = trace_spec.times
            data_out = trace_spec.data
        else:
            timeax_in = trace_spec.times
            data_in = trace_spec.data
            time_intverval = timeax_in[1] - timeax_in[0]

            # Check if the time axes overlap; if so, remove the last point of the first spectrogram
            if timeax_out[-1] >= timeax_in[0]:
                timeax_out = delete(timeax_out, -1)
                data_out = delete(data_out, -1, axis=1)

            # Check if the time axes are continuous
            if timeax_in[0] - timeax_out[-1] > time_intverval:
                # Fill the gap with the fill value
                print(f"Warning: Time axes are not continuous between {timeax_out[-1]} and {timeax_in[0]} for {station}.{location}.{component}!")
                num_fill = int((timeax_in[0] - timeax_out[-1]) / time_intverval)
                fill_data = fill_value * ones((data_out.shape[0], num_fill))
                data_out = column_stack([data_out, fill_data])
                timeax_out = timeax_out.append(date_range(start = timeax_out[-1] + time_intverval, periods = num_fill, freq = time_intverval))
                print(f"Filled {num_fill} points between them.")
            
            # Stitch the spectrograms
            data_out = column_stack([data_out, data_in])
            timeax_out = timeax_out.append(timeax_in)

    # Save the stitched spectrogram to a new TraceSTFTPSD object
    trace_spec_out = TraceSTFTPSD(station, location, component, time_label, timeax_out, freqax, data_out)

    return trace_spec_out

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
def assemble_spec_filename(range_type, block_type, sensor_type, station, window_length, overlap, downsample, **kwargs):
    if block_type == "day":
        block_type = "daily"
    elif block_type == "hour":
        block_type = "hourly"
    
    if time_label is None:
        if downsample:
            downsample_factor = kwargs["downsample_factor"]
            filename = f"{range_type}_{block_type}_{sensor_type}_spectrograms_{station}_window{window_length:.0f}s_overlap{overlap:.1f}_downsample{downsample_factor:d}.h5"
        else:
            filename = f"{range_type}_{block_type}_{sensor_type}_spectrograms_{station}_window{window_length:.0f}s_overlap{overlap:.1f}.h5"
    else:
        if downsample:
            downsample_factor = kwargs["downsample_factor"]
            filename = f"{range_type}_{block_type}_{sensor_type}_spectrograms_{station}_{time_label}_window{window_length:.0f}s_overlap{overlap:.1f}_downsample{downsample_factor:d}.h5"
        else:
            filename = f"{range_type}_{block_type}_{sensor_type}_spectrograms_{station}_{time_label}_window{window_length:.0f}s_overlap{overlap:.1f}.h5"

    return filename

# Create a spectrogram file for a geophone station and save the header information
def create_geo_spectrogram_file(station, range_type = "whole_deployment", block_type = "day", window_length = 60.0, overlap = 0.0, freq_interval = 1.0, downsample = False, outdir = SPECTROGRAM_DIR, **kwargs):

    if not downsample:
        filename = assemble_spec_filename(range_type, block_type, "geo", None, station, window_length, overlap, downsample)
    else:
        downsample_factor = kwargs['downsample_factor']
        filename = assemble_spec_filename(range_type, block_type, "geo", None, station, window_length, overlap, downsample, downsample_factor = downsample_factor)

    # Create the output file
    outpath = join(outdir, filename)
    file = File(outpath, 'w')
    
    # Create the group for storing the headers
    header_group = file.create_group('headers')

    # Save the header information with encoding
    header_group.create_dataset('station', data=station.encode("utf-8"))
    header_group.create_dataset('window_length', data=window_length)
    header_group.create_dataset('overlap', data=overlap)
    header_group.create_dataset('block_type', data=block_type.encode("utf-8"))
    header_group.create_dataset('frequency_interval', data=freq_interval)

    # Create the group for storing the spectrogram data (blocks)
    data_group = file.create_group('data')

    print(f"Created spectrogram file {outpath}")

    return file

# Save one geophone spectrogram data block to an opened HDF5 file
def write_geo_spectrogram_block(file, stream_spec, outdir = SPECTROGRAM_DIR, close_file = True):
    components = GEO_COMPONENTS

    # Verify the StreamSTFTPSD object
    stream_spec.verify_geo_block()
    
    # Extract the data from the StreamSTFTPSD object
    trace_spec_z = stream_spec.select(component="Z")[0]
    trace_spec_1 = stream_spec.select(component="1")[0]
    trace_spec_2 = stream_spec.select(component="2")[0]

    # Verify the station name
    if trace_spec_z.station != file["headers"]["station"][()].decode("utf-8"):
        raise ValueError("Error: Inconsistent station name!")

    time_label = trace_spec_z.time_label
    timeax = trace_spec_z.times
    data_z = trace_spec_z.data
    data_1 = trace_spec_1.data
    data_2 = trace_spec_2.data

    # Convert the time axis to integer
    timeax = datetime2int(timeax)

    # Create a new block
    data_group = file["data"]
    block_group = data_group.create_group(time_label)
    block_group.create_dataset("start_time", data=timeax[0])

    # Create the group for storing each component's spectrogram data
    comp_group = block_group.create_group("components")
    for component in components:
        if component == "Z":
            data = data_z
        elif component == "1":
            data = data_1
        elif component == "2":
            data = data_2
        
        comp_group.create_dataset(component, data=data)

    print(f"Spectrogram block {time_label} is saved")

    # Close the file
    if close_file:
        file.close()

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

# Finish writing a geophone spectrogram file by writing the list of time labels and close the file
def finish_geo_spectrogram_file(file, time_labels):
    file["headers"].create_dataset('time_labels', data=[time_label.encode("utf-8") for time_label in time_labels])
    print("Time labels are saved.")
    
    file.close()
    print("The spectrogram file is closed.")
    
# Read specific geophone-spectrogram data blocks from an HDF5 file 
def read_geo_spectrograms(inpath, time_labels_to_read = None):
    components = GEO_COMPONENTS
    
    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        
        station = header_group["station"][()]
        station = station.decode("utf-8")

        time_labels = header_group["time_labels"][:]
        time_labels = [time_label.decode("utf-8") for time_label in time_labels]
        
        freq_interval = header_group["frequency_interval"][()]
        
        window_length = header_group["window_length"][()]
        overlap = header_group["overlap"][()]
        
        # Read the time labels
        if time_labels_to_read is None:
            time_labels_to_read = time_labels
            
        data_group = file["data"]
        stream_spec = StreamSTFTPSD()
        for time_label in time_labels_to_read:
            block_group = data_group[time_label]
            starttime = block_group["start_time"][()]
            starttime = Timestamp(starttime, unit='ns')

            comp_group = block_group["components"]
            for component in components:
                data = comp_group[component][:]
    
                num_freq = data.shape[0]
                freqax = linspace(0, (num_freq - 1) * freq_interval, num_freq)
    
                num_time = data.shape[1]
                time_interval = Timedelta(seconds = window_length * (1 - overlap))
                timeax = date_range(start = starttime, periods = num_time, freq = time_interval)
            
                trace_spec = TraceSTFTPSD(station, "", component, time_label, timeax, freqax, data)
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

# Get the quality factor and resonance width of a peak in a power spectrum
# The input power must be in dB!
def get_quality_factor(freqax, power_in_db, freq0):
    
    # Find the peak point
    ipeak = freqax.searchsorted(freq0)
    power0 = power_in_db[ipeak]
    power3db = power0 - 3.0

    # Find the points where power drops below the 3 dB level
    for i in range(ipeak, len(power_in_db)):
        if power_in_db[i] < power3db:
            break
    freq_high = freqax[i]

    for i in range(ipeak, 0, -1):
        if power_in_db[i] < power3db:
            break
    freq_low = freqax[i]

    bandwidth =  freq_high - freq_low
    rbw = 1 / bandwidth
    quality_factor = freq0 * rbw

    return quality_factor, rbw




    return quality_factor
