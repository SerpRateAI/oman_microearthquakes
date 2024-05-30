# Functions and classes for spectrum analysis
from os.path import join, splitext
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, iirfilter, sosfilt, freqz
from scipy.signal.windows import hann
from scipy.interpolate import RegularGridInterpolator
from numpy import abs, amax, array, column_stack, concatenate, convolve, cumsum, delete, interp, load, linspace, amax, meshgrid, amin, nan, ones, pi, savez, zeros
from pandas import Series, DataFrame, DatetimeIndex, Timedelta, Timestamp
from pandas import concat, cut, date_range, read_csv, read_hdf, to_datetime
from h5py import File, special_dtype
from multitaper import MTSpec
from multiprocessing import Pool

from utils_basic import GEO_COMPONENTS
from utils_basic import SPECTROGRAM_DIR 
from utils_basic import assemble_timeax_from_ints, datetime2int, int2datetime, reltimes_to_timestamps, power2db
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
                    if len(stream_to_stitch) > 1:
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

    # Trim the spectrograms to a given frequency range
    def trim_freq(self, freqmin = None, freqmax = None):
        for trace in self.traces:
            trace.trim_freq(freqmin = freqmin, freqmax = freqmax)

    # Trim the spectrograms to a given time range
    def trim_time(self, starttime = None, endtime = None):
        for trace in self.traces:
            trace.trim(starttime = starttime, endtime = endtime)

    # Trim the spectrograms to the begin and end of the day
    def trim_to_day(self):
        for trace in self.traces:
            trace.trim_to_day()

    # Resample the spectrograms to a given time axis
    def resample_time(self, timeax_out):
        for trace in self.traces:
            trace.resample_time(timeax_out)

    # Resample the spectrograms to the begin and end of the day
    def resample_to_day(self, parallel = False, **kwargs):         
        for trace in self.traces:
            trace.resample_to_day(parallel, **kwargs)

    # Set the time label of the traces
    def set_time_labels(self, block_type = "daily"):
        for trace in self.traces:
            trace.set_time_label(block_type)

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

    # Verify if the geophone traces have the same station name, time labels, and the three components ("a block")
    def verify_geo_block(self):
        if len(self.traces) != 3:
            raise ValueError("Error: Invalid number of components!")
        
        if self.traces[0].station != self.traces[1].station or self.traces[1].station != self.traces[2].station:
            raise ValueError("Error: Inconsistent station names!")
        
        if self.traces[0].time_label != self.traces[1].time_label or self.traces[1].time_label != self.traces[2].time_label:
            raise ValueError("Error: Inconsistent time labels!")
        
        if self.traces[0].component != "Z" or self.traces[1].component != "1" or self.traces[2].component != "2":
            raise ValueError("Error: Invalid component names!")

    # Verify if the hydrophone traces have the same station name, time labels, and the number of locations  ("a block")
    def verify_hydro_block(self, locations):
        station = self.traces[0].station
        time_label = self.traces[0].time_label

        for i, trace in enumerate(self.traces):
            if trace.station != station or trace.time_label != time_label:
                raise ValueError("Error: Inconsistent station names or time labels!")

            if trace.location != locations[i]:
                raise ValueError("Error: Inconsistent location names!")

        
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
    def trim_time(self, starttime = None, endtime = None):
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

    # Trim the spectrogram to a given frequency range
    def trim_freq(self, freqmin = None, freqmax = None):
        freqax = self.freqs

        if freqmin is not None:
            min_index = freqax.searchsorted(freqmin)
        else:
            min_index = 0

        if freqmax is not None:
            max_index = freqax.searchsorted(freqmax)
        else:
            max_index = len(freqax)

        self.freqs = freqax[min_index:max_index]
        self.data = self.data[min_index:max_index, :]

    # Trim the spectrogram to the begin and end of the day
    def trim_to_day(self):
        timeax = self.times
        num_time = len(timeax)
        ref_time = timeax[num_time // 2]
        start_of_day = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = ref_time.replace(hour=23, minute=59, second=59, microsecond=999999)

        self.trim_time(start_of_day, end_of_day)

    # Pad and resample the spectrogram to a given time axis
    def resample_time(self, timeax_out, parallel = False, **kwargs):
        if parallel and "num_process" not in kwargs:
            raise ValueError("Error: Number of processes is not given!")
        
        # Convert the input and output time axes to integers
        timeax_in = self.times
        timeax_in = datetime2int(timeax_in)

        timeax_out = datetime2int(timeax_out)

        # Resample the spectrogram
        data_in = self.data
        if parallel:
            num_process = kwargs["num_process"]
            data_out = resample_stft_time_in_parallel(timeax_in, timeax_out, data_in, num_process)
        else:
            data_out = resample_stft_time(timeax_in, timeax_out, data_in)

        timeax_out = int2datetime(timeax_out)
        self.times = timeax_out
        self.data = data_out

    # Pad and resample the spectrogram to the begin and end of the day
    def resample_to_day(self, parallel = False, **kwargs):
        if parallel and "num_process" not in kwargs:
            raise ValueError("Error: Number of processes is not given!")
        
        timeax = self.times
        num_time = len(timeax)
        ref_time = timeax[num_time // 2]
        start_of_day = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = ref_time.replace(hour=23, minute=59, second=59, microsecond=999999)

        timeax_out = date_range(start=start_of_day, end=end_of_day, freq=self.time_interval)
        self.resample_time(timeax_out, parallel, **kwargs)

    # Set the time label of the trace
    def set_time_label(self, block_type):
        num_times = self.num_times
        timeax = self.times
        
        ref_time = timeax[num_times // 2]
        if block_type == "daily":
            start_of_block = ref_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif block_type == "hourly":
            start_of_block = ref_time.replace(minute=0, second=0, microsecond=0)
        else:
            raise ValueError("Error: Invalid block type!")

        time_label = start_of_block.strftime("%Y%m%d%H%M%S%f")
        self.time_label = time_label

######
# Functions
######

###### Functions for handling STFT spectrograms ######
# Concatenate multiple StreamSTFTPSD objects
def concat_stream_spec(streams):
    traces = []
    for stream in streams:
        traces.extend(stream.traces)

    stream_concat = StreamSTFTPSD(traces)

    return stream_concat

# Find spectral peaks in the spectrograms of a geophone station
# The function runs in parallel using the given number of processes
# A Pandas DataFrame containing the frequency, time, power, and quality factor of each peak is returned
# The power threshold is in dB!
def find_geo_station_spectral_peaks(stream_spec, num_process, prom_threshold = 5, rbw_threshold = 0.2, min_freq = None, max_freq = None):
    # Verify the StreamSTFTPSD object
    if len(stream_spec) != 3:
        raise ValueError("Error: Invalid number of components!")

    # Compute the total power spectrogram
    trace_spec_total = stream_spec.get_total_power()

    # Find the spectral peaks in the total power spectrogram
    peak_df = find_trace_spectral_peaks(trace_spec_total, num_process, prom_threshold, rbw_threshold, min_freq, max_freq)

    return peak_df, trace_spec_total

# Find spectral peaks satisfying the given criteria
# The function runs in parallel using the given number of processes
# The power threshold is in dB!
def find_trace_spectral_peaks(trace_spec, num_process, prom_threshold = 5, rbw_threshold = 0.2, min_freq = None, max_freq = None):
    # Convert the trace to dB
    trace_spec.to_db()
    
    # Trim the data to the given frequency range
    if min_freq is None:
        min_freq = trace_spec.freqs[0]

    if max_freq is None:
        max_freq = trace_spec.freqs[-1]

    freq_inds = (trace_spec.freqs >= min_freq) & (trace_spec.freqs <= max_freq)
    freqax = trace_spec.freqs[freq_inds]
    data = trace_spec.data[freq_inds, :]

    timeax = trace_spec.times

    # Divide the data and time axis into chunks for parallel processing
    num_chunks = num_process
    chunk_size = data.shape[1] // num_chunks
    data_chunks = [data[:, i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    time_chunks = [timeax[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    # Construct the arguments for the parallel processing
    args = [(time_chunk, freqax, data_chunk, prom_threshold, rbw_threshold) for data_chunk, time_chunk in zip(data_chunks, time_chunks)]

    # Find the spectral peaks in parallel
    print(f"Finding the spectral peaks in {num_process} processes...")
    with Pool(num_process) as pool:
        results = pool.starmap(find_spectral_peaks, args)

    # Concatenate the results
    results = [df for df in results if not df.empty]
    peak_df = concat(results, ignore_index=True)
    

    return peak_df

# Find the times and frequencies of spectral peaks satisfying the given prominence and reverse bandwidth criteria
def find_spectral_peaks(timeax, freqax, power, prominence_threshold, rbw_threshold):
    peak_freqs = []
    peak_times = []
    peak_powers = []
    peak_rbws = []
    for i, peak_time in enumerate(timeax):
        power_column = power[:, i]
        freq_inds, _ = find_peaks(power_column, prominence = prominence_threshold)

        for j in freq_inds:
            peak_freq = freqax[j]
            peak_power = power_column[j]
            _, rbw = get_quality_factor(freqax, power_column, peak_freq)

            if rbw >= rbw_threshold:
                peak_freqs.append(peak_freq)
                peak_times.append(peak_time)
                peak_powers.append(peak_power)
                peak_rbws.append(rbw)

    peak_df = DataFrame({"frequency": peak_freqs, "time": peak_times, "power": peak_powers, "reverse_bandwidth": peak_rbws})

    return peak_df

# Update the spectral-peak group count
def update_spectral_peak_group_counts(peak_df, counts_to_update = None):
    count_df = peak_df.groupby(['time', 'frequency']).size().reset_index(name='count')
    
    if counts_to_update is None:
        counts_updated_df = count_df.copy()
    else:     
        counts_updated_df = concat([counts_to_update, count_df]).groupby(['time', 'frequency']).sum().reset_index()
        
    return counts_updated_df

# Get the file-name suffix for the spectrograms
def get_spectrogram_file_suffix(window_length, overlap, downsample, **kwargs):
    suffix = f"window{window_length:.0f}s_overlap{overlap:.1f}"
    if downsample:
        downsample_factor = kwargs["downsample_factor"]
        suffix = f"{suffix}_downsample{downsample_factor:d}"

    return suffix
    
# Get the file-name suffix for the spectral peaks
def get_spec_peak_file_suffix(prom_threshold, rbw_threshold, min_freq = None, max_freq = None):
    if min_freq is None and max_freq is None:
        suffix = f"prom{prom_threshold:.0f}db_rbw{rbw_threshold:.1f}"
    elif min_freq is not None and max_freq is None:
        suffix = f"prom{prom_threshold:.0f}db_rbw{rbw_threshold:.1f}_freq{min_freq:.0f}to500hz"
    elif min_freq is None and max_freq is not None:
        suffix = f"prom{prom_threshold:.0f}db_rbw{rbw_threshold:.1f}_freq0to{max_freq:.0f}hz"
    else:
        suffix = f"prom{prom_threshold:.0f}db_rbw{rbw_threshold:.1f}_freq{min_freq:.0f}to{max_freq:.0f}hz"

    return suffix

# # Group the spectral-peak detections into regular time and frequency bins
# def group_spectral_peaks_regular_bins(peak_df, time_bin_edges, freq_bin_edges):

#     # Group the spectral peaks
#     peak_df['time_bin'] = cut(peak_df["time"], time_bin_edges, include_lowest=True, right=False)
#     peak_df['freq_bin'] = cut(peak_df["frequency"], freq_bin_edges, include_lowest=True, right=False)
    
#     grouped = peak_df.groupby(['time_bin', 'freq_bin'], observed = False).size().unstack(fill_value=0)
#     bin_counts = grouped.values
#     bin_counts = bin_counts.T

#     return bin_counts

# # Convert the spectral-peak bin counts to a DataFrame
# def bin_counts_to_df(time_bin_centers, freq_bin_centers, counts, count_threshold = 1):
#     # Convert the time bin centers to nano-second integers
#     time_bin_centers = [time.value for time in time_bin_centers]

#     # Create the 2D meshgrid of time and frequency bin centers
#     time_mesh, freq_mesh = meshgrid(time_bin_centers, freq_bin_centers)

#     # Flatten the 2D meshgrid
#     time_mesh = time_mesh.flatten()
#     freq_mesh = freq_mesh.flatten()
#     counts = counts.flatten()

#     # Create the DataFrame
#     count_df = DataFrame({"time": time_mesh, "frequency": freq_mesh, "count": counts})

#     # Convert the time column to Timestamp
#     count_df["time"] = to_datetime(count_df["time"])

#     # Remove the rows with counts below the threshold
#     count_df = count_df.loc[count_df["count"] >= count_threshold]
#     count_df.reset_index(drop = True, inplace = True)

#     return count_df


# Stitch spectrograms of multiple time periods together
# Output window length is in SECOND!
def stitch_spectrograms(stream_spec, fill_value = nan):
    if len(stream_spec) == 1:
        raise ValueError("Error: Only one trace in the stream. Nothing to stitch!")

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

# A wrapper function for resampling a STFT spectrogram to a new time axis using nearest-neighbor interpolation in parallel
def resample_stft_time_in_parallel(timeax_in, timeax_out, data_in, num_process):

    # Divide the input data along the frequency axis into chunks for parallel processing
    num_chunks = num_process
    chunk_size = data_in.shape[0] // num_chunks + 1
    data_chunks = [data_in[i * chunk_size:(i + 1) * chunk_size, :] for i in range(num_chunks)]

    # Construct the arguments for the parallel processing
    args = [(timeax_in, timeax_out, data_chunk) for data_chunk in data_chunks]

    # Resample the spectrogram in parallel
    with Pool(num_process) as pool:
        results = pool.starmap(resample_stft_time, args)

    # Concatenate the results
    data_out = concatenate(results, axis=0)

    return data_out

# Resample a STFT spectrogram to a new time axis using nearest-neighbor interpolation
def resample_stft_time(timeax_in, timeax_out, data_in):
    num_freqs = data_in.shape[0]
    data_out = zeros((num_freqs, len(timeax_out)))

    for i in range(num_freqs):
        row = data_in[i, :]
        interpolator = RegularGridInterpolator((timeax_in,), row, bounds_error=False, fill_value=nan, method='nearest')
        data_out[i, :] = interpolator(timeax_out)

    return data_out


# # Find the times and frequencies of spectral peaks satisfying the given prominence and reverse bandwidth criteria
# def find_spectral_peaks(timeax, freqax, power, prominence_threshold, rbw_threshold):
#     peak_freqs = []
#     peak_times = []
#     peak_powers = []
#     peak_rbws = []
#     for i, peak_time in enumerate(timeax):
#         power_column = power[:, i]
#         freq_inds, _ = find_peaks(power_column, prominence = prominence_threshold)

#         for j in freq_inds:
#             peak_freq = freqax[j]
#             peak_power = power_column[j]
#             _, rbw = get_quality_factor(freqax, power_column, peak_freq)

#             if rbw >= rbw_threshold:
#                 peak_freqs.append(peak_freq)
#                 peak_times.append(peak_time)
#                 peak_powers.append(peak_power)
#                 peak_rbws.append(rbw)

#     peak_df = DataFrame({"frequency": peak_freqs, "time": peak_times, "power": peak_powers, "reverse_bandwidth": peak_rbws})

#     return peak_df

# Get the time and frequency indices of spectral-peak counts
def get_spec_peak_time_freq_inds(counts_df, timeax, freqax):
    num_time = len(timeax)
    time_inds = []
    freq_inds = []
    for i, row in counts_df.iterrows():
        time = row["time"]
        freq = row["frequency"]
        time_ind = timeax.searchsorted(time)
        freq_ind = freqax.searchsorted(freq)

        # Exclude the time and frequency indices that are out of the range
        if time_ind == num_time or time_ind == 0 or freq_ind == len(freqax) or freq_ind == 0:
            continue

        time_inds.append(time_ind)
        freq_inds.append(freq_ind)

    return time_inds, freq_inds

# Create a spectrogram file for a geophone station and save the header information
def create_geo_spectrogram_file(station, window_length = 60.0, overlap = 0.0, freq_interval = 1.0, downsample = False, outdir = SPECTROGRAM_DIR, **kwargs):
    suffix = get_spectrogram_file_suffix(window_length, overlap, downsample, **kwargs)
    filename = f"whole_deployment_daily_geo_spectrograms_{station}_{suffix}.h5"

    # Create the output file
    outpath = join(outdir, filename)
    file = File(outpath, 'w')
    
    # Create the group for storing the headers
    header_group = file.create_group('headers')

    # Save the header information with encoding
    header_group.create_dataset('station', data=station.encode("utf-8"))
    header_group.create_dataset('window_length', data=window_length)
    header_group.create_dataset('overlap', data=overlap)
    header_group.create_dataset('block_type', data="daily")
    header_group.create_dataset('frequency_interval', data=freq_interval)

    # Create the group for storing the spectrogram data (blocks)
    file.create_group('data')

    print(f"Created spectrogram file {outpath}")

    return file

# Save one geophone spectrogram data block to an opened HDF5 file
def write_geo_spectrogram_block(file, stream_spec, close_file = True):
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
    block_group.create_dataset("num_times", data=len(timeax))
    block_group.create_dataset("num_freqs", data=len(trace_spec_z.freqs))

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

# Finish writing a geophone spectrogram file by writing the list of time labels and close the file
def finish_geo_spectrogram_file(file, time_labels):
    file["headers"].create_dataset('time_labels', data=[time_label.encode("utf-8") for time_label in time_labels])
    print("Time labels are saved.")
    
    file.close()
    print("The spectrogram file is closed.")
    
# Read specific segments of geophone-spectrogram data from an HDF5 file 
def read_geo_spectrograms(inpath, time_labels = None, starttime = None, endtime = None, min_freq = 0.0, max_freq = 500.0):
    components = GEO_COMPONENTS

    # Determine if both start and end times are provided
    if (starttime is not None and endtime is None) or (starttime is None and endtime is not None):
        raise ValueError("Error: Both start and end times must be provided!")
    
    # Determine if the input time information is redundant
    if starttime is not None and endtime is not None and time_labels is not None:
        raise ValueError("Error: Time labels and start/end times cannot be given at the same time!")

    # Convert the time labels to a list
    if not isinstance(time_labels, list):
        if isinstance(time_labels, str):
            time_labels = [time_labels]
    
    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        
        station = header_group["station"][()]
        station = station.decode("utf-8")

        time_labels_in = header_group["time_labels"][:]
        time_labels_in = [time_label.decode("utf-8") for time_label in time_labels_in]
        
        freq_interval = header_group["frequency_interval"][()]

        min_freq_index = int(min_freq / freq_interval)
        max_freq_index = int(max_freq / freq_interval)
        
        time_interval = get_spec_time_interval(header_group)
        data_group = file["data"]
        
        if starttime is None and endtime is None:
            # Option 1: Read the spectrograms of specific time labels
            if time_labels is None:
                time_labels = time_labels_in
                
            stream_spec = StreamSTFTPSD()
            for time_label in time_labels:
                try:
                    block_group = data_group[time_label]
                except KeyError:
                    print(f"Warning: Time label {time_label} does not exist!")
                    return None

                timeax = get_spec_block_timeax(block_group, time_interval)

                comp_group = block_group["components"]
                for component in components:
                    data = comp_group[component][min_freq_index:max_freq_index, :]
        
                    num_freq = data.shape[0]
                    freqax = linspace(min_freq, (num_freq - 1) * freq_interval, num_freq)
                
                    trace_spec = TraceSTFTPSD(station, "", component, time_label, timeax, freqax, data)
                    stream_spec.append(trace_spec)
        else:
            # Option 2: Read the spectrograms of a specific time range
            # Convert the start and end times to Timestamp objects
            if isinstance(starttime, str):
                starttime_to_read = Timestamp(starttime)
            
            if isinstance(endtime, str):
                endtime_to_read = Timestamp(endtime)
            
            # Check the start time is greater than the end time
            if starttime_to_read > endtime_to_read:
                raise ValueError("Error: Start time must be less than the end time!")

            # Get the start and end times of each block
            block_timing_df = get_block_timings(data_group, time_labels_in, time_interval)

            # Find the blocks that overlap with the given time range
            overlap_df = get_overlapping_blocks(block_timing_df, starttime_to_read, endtime_to_read)
                
            # Read the spectrograms of the overlapping blocks
            stream_spec = StreamSTFTPSD()
            for _, row in overlap_df.iterrows():
                time_label = row["time_label"]
                block_group = data_group[time_label]

                starttime_block = row["start_time"]
                num_times = row["num_times"]

                comp_group = block_group["components"]
                for component in components:
                    # Find the start and end indices of the time axis
                    start_index, end_index = get_data_block_time_indices(starttime_block, num_times, time_interval, starttime_to_read, endtime_to_read)

                    # Slice the data matrix
                    data = comp_group[component][min_freq_index:max_freq_index, start_index:end_index]

                    # Create the frequency axis
                    freqax = linspace(min_freq, max_freq, data.shape[0])
        
                    # Create the time axis
                    starttime_out = max([starttime_to_read, starttime_block])
                    timeax = date_range(start = starttime_out, periods = data.shape[1], freq = time_interval)

                    # Create the TraceSTFTPSD object
                    trace_spec = TraceSTFTPSD(station, "", component, time_label, timeax, freqax, data)
                    stream_spec.append(trace_spec)
    
    # Stitch the spectrograms if there are multiple blocks
    stream_spec.stitch()

    return stream_spec

# Create a hydrophone spectrogram file and save the header information
def create_hydro_spectrogram_file(station, locations, window_length = 60.0,  overlap = 0.0, freq_interval = 1.0, downsample = False, outdir = SPECTROGRAM_DIR, **kwargs):
    suffix = get_spectrogram_file_suffix(window_length, overlap, downsample, **kwargs)
    filename = f"whole_deployment_daily_hydro_spectrograms_{station}_{suffix}.h5"

    # Create the output file
    outpath = join(outdir, filename)
    file = File(outpath, 'w')
    
    # Create the group for storing the headers
    header_group = file.create_group('headers')

    # Save the header information with encoding
    header_group.create_dataset('station', data=station.encode("utf-8"))
    header_group.create_dataset('locations', data=[location.encode("utf-8") for location in locations])
    header_group.create_dataset('window_length', data=window_length)
    header_group.create_dataset('overlap', data=overlap)
    header_group.create_dataset('block_type', data="daily")
    header_group.create_dataset('frequency_interval', data=freq_interval)

    # Create the group for storing the spectrogram data (blocks)
    file.create_group('data')

    print(f"Created spectrogram file {outpath}")

    return file

# Save one hydrophone spectrogram data block to an opened HDF5 file
def write_hydro_spectrogram_block(file, stream_spec, locations, close_file = True):
    # Verify the StreamSTFTPSD object
    stream_spec.verify_hydro_block(locations)

    # Verify the station name
    station = file["headers"]["station"][()].decode("utf-8")
    if stream_spec[0].station != station:
        raise ValueError("Error: Inconsistent station name!")
    
    # Extract the data from the StreamSTFTPSD object
    time_label = stream_spec[0].time_label
    timeax = stream_spec[0].times

    # Convert the time axis to integer
    timeax = datetime2int(timeax)

    # Create a new block
    data_group = file["data"]
    block_group = data_group.create_group(time_label)
    block_group.create_dataset("start_time", data=timeax[0])
    block_group.create_dataset("num_times", data=len(timeax))
    block_group.create_dataset("num_freqs", data=len(stream_spec[0].freqs))

    # Create the group for storing each location's spectrogram data
    loc_group = block_group.create_group("locations")
    for i, location in enumerate(locations):
        trace_spec = stream_spec[i]
        data = trace_spec.data
        loc_group.create_dataset(location, data=data)

    print(f"Spectrogram block {time_label} is saved")

    # Close the file
    if close_file:
        file.close()

# Finish writing a hydrophone spectrogram file by writing the list of time labels and close the file
def finish_hydro_spectrogram_file(file, time_labels):
    file["headers"].create_dataset('time_labels', data=[time_label.encode("utf-8") for time_label in time_labels])
    print("Time labels are saved.")
    
    file.close()
    print("The spectrogram file is closed.")

# Read specific segments of hydrophone-spectrogram data from an HDF5 file
def read_hydro_spectrograms(inpath, time_labels = None, starttime = None, endtime = None, min_freq = 0.0, max_freq = 500.0):

    # Deterimine if both start and end times are provided
    if (starttime is not None and endtime is None) or (starttime is None and endtime is not None):
        raise ValueError("Error: Both start and end times must be provided!")
    
    # Determine if the input time information is redundant
    if starttime is not None and endtime is not None and time_labels is not None:
        raise ValueError("Error: Time labels and start/end times cannot be given at the same time!")
    


    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        
        station = header_group["station"][()]
        station = station.decode("utf-8")

        time_labels_in = header_group["time_labels"][:]
        time_labels_in = [time_label.decode("utf-8") for time_label in time_labels_in]


        locations = header_group["locations"][:]
        locations = [location.decode("utf-8") for location in locations]

        freq_interval = header_group["frequency_interval"][()]

        min_freq_index = int(min_freq / freq_interval)
        max_freq_index = int(max_freq / freq_interval)
        
        time_interval = get_spec_time_interval(header_group)
        data_group = file["data"]
        
        if starttime is None and endtime is None:
            # Option 1: Read the spectrograms of specific time labels
            # If the input time labels are not provided, read only the first time label
            if time_labels is None:
                time_labels = time_labels_in[:1]
            else:
                # Convert the time labels to a list
                if not isinstance(time_labels, list):
                    if isinstance(time_labels, str):
                        time_labels = [time_labels]
                
            stream_spec = StreamSTFTPSD()
            for time_label in time_labels:
                print(f"Reading spectrogram block {time_label}...")
                try:
                    block_group = data_group[time_label]
                except KeyError:
                    print(f"Warning: Time label {time_label} does not exist!")
                    return None

                timeax = get_spec_block_timeax(block_group, time_interval)

                loc_group = block_group["locations"]
                for location in locations:
                    data = loc_group[location][min_freq_index:max_freq_index, :]
        
                    num_freq = data.shape[0]
                    freqax = linspace(min_freq, max_freq, num_freq)
                
                    trace_spec = TraceSTFTPSD(station, location, "", time_label, timeax, freqax, data)
                    stream_spec.append(trace_spec)
        else:
            # Option 2: Read the spectrograms of a specific time range
            # Convert the start and end times to Timestamp objects
            if isinstance(starttime, str):
                starttime_to_read = Timestamp(starttime)
            
            if isinstance(endtime, str):
                endtime_to_read = Timestamp(endtime)
            
            # Check the start time is greater than the end time
            if starttime_to_read > endtime_to_read:
                raise ValueError("Error: Start time must be less than the end time!")

            # Get the start and end times of each block
            block_timing_df = get_block_timings(data_group, time_labels_in, time_interval)

            # Find the blocks that overlap with the given time range
            overlap_df = get_overlapping_blocks(block_timing_df, starttime_to_read, endtime_to_read)

            # Read the spectrograms of the overlapping blocks
            stream_spec = StreamSTFTPSD()
            for _, row in overlap_df.iterrows():
                time_label = row["time_label"]
                block_group = data_group[time_label]

                starttime_block = row["start_time"]
                num_times = row["num_times"]

                loc_group = block_group["locations"]
                for location in locations:
                    # Find the start and end indices of the time axis
                    start_index, end_index = get_data_block_time_indices(starttime_block, num_times, time_interval, starttime_to_read, endtime_to_read)

                    # Slice the data matrix
                    data = loc_group[location][min_freq_index:max_freq_index, start_index:end_index]

                    # Create the frequency axis
                    freqax = linspace(min_freq, max_freq, data.shape[0])
        
                    # Create the time axis
                    starttime_out = max([starttime_to_read, starttime_block])
                    timeax = date_range(start = starttime_out, periods = data.shape[1], freq = time_interval)

                    # Create the TraceSTFTPSD object
                    trace_spec = TraceSTFTPSD(station, location, "", time_label, timeax, freqax, data)
                    stream_spec.append(trace_spec)

    # Stitch the spectrograms if there are multiple blocks
    stream_spec.stitch()

    return stream_spec

# Read the power at a specific time from a geophone spectrogram file
def read_time_from_geo_spectrograms(inpath, time_out, components = GEO_COMPONENTS):
    # Convert string to Timestamp
    if isinstance(time_out, str):
        time_out = Timestamp(time_out)

    power_dict = {}
    

    with File(inpath, 'r') as file:
        header_group = file["headers"]

        # Get the time labels and start times of each block
        time_labels = get_spec_time_labels(header_group)
        starttimes = to_datetime(time_labels, format = "%Y%m%d%H%M%S%f")

        # Get the time and frequency interval
        time_interval = get_spec_time_interval(header_group)
        freq_interval = header_group["frequency_interval"][()]

        # Get the time label of the block that contains the time_out
        index = starttimes.searchsorted(time_out)

        if index == 0 or index == len(starttimes):
            raise ValueError("Error: The time is out of the range of the spectrogram file!")
        
        time_label = time_labels[index - 1]

        # Read the time slice of the spectrogram
        block_group = file["data"][time_label]
        timeax = get_spec_block_timeax(block_group, time_interval)
        freqax = get_spec_block_freqax(block_group, freq_interval)

        time_index = timeax.searchsorted(time_out)

        power_dict["frequencies"] = freqax

        comp_group = block_group["components"]
        for component in components:
            data = comp_group[component][:, time_index]
            power_dict[component] = data

    return power_dict

# Read the power at a specific frequency from a geophone spectrogram file
def read_freq_from_geo_spectrograms(inpath, freq_out, components = GEO_COMPONENTS):
    power_dict = {}
    power_dict["times"] = []
    for component in components:
        power_dict[component] = []

    with File(inpath, 'r') as file:
        header_group = file["headers"]

        # Get the time interval
        time_interval = get_spec_time_interval(header_group)

        # Get the frequency index
        freq_interval = header_group["frequency_interval"][()]
        freq_index = int(round(freq_out / freq_interval))

        # Read each block
        time_labels = file["headers"]["time_labels"][:]

        for time_label in time_labels:
            block_group = file["data"][time_label]
            timeax = get_spec_block_timeax(block_group, time_interval)
            power_dict["times"].append(Series(timeax))

            # Read every component
            comp_group = block_group["components"]
            for component in components:
                data = comp_group[component][freq_index, :]

                power_dict[component].append(data)

    # Remove the last point of the time axis if it is the same as the first point of the next block
    num_time_labels = len(time_labels)
    for i in range(num_time_labels - 1):
        timeax_prev = power_dict["times"][i]
        timeax_next = power_dict["times"][i + 1]
        if timeax_prev.iloc[-1] >= timeax_next.iloc[0]:
            timeax_prev = timeax_prev.iloc[:-1]
            power_dict["times"][i] = timeax_prev
            for component in components:
                power_dict[component][i] = power_dict[component][i][:-1]

    # Concatenate the time axis and power data
    power_dict["times"] = concat(power_dict["times"])
    power_dict["times"] = DatetimeIndex(power_dict["times"])
    
    for component in components:
        power_dict[component] = concatenate(power_dict[component])

    return power_dict

# Read the power at a specific time from a hydrophone spectrogram file
def read_time_from_hydro_spectrograms(inpath, time_out, locations = None):
    # Convert string to Timestamp
    if isinstance(time_out, str):
        time_out = Timestamp(time_out)

    power_dict = {}

    with File(inpath, 'r') as file:
        header_group = file["headers"]

        # Get the time labels and start times of each block
        time_labels = get_spec_time_labels(header_group)
        starttimes = to_datetime(time_labels, format = "%Y%m%d%H%M%S%f")

        # Get the time and frequency interval
        time_interval = get_spec_time_interval(header_group)
        freq_interval = header_group["frequency_interval"][()]

        # Get the time label of the block that contains the time_out
        index = starttimes.searchsorted(time_out)

        if index == 0 or index == len(starttimes):
            raise ValueError("Error: The time is out of the range of the spectrogram file!")
        
        time_label = time_labels[index - 1]

        # Read the time slice of each location
        block_group = file["data"][time_label]
        timeax = get_spec_block_timeax(block_group, time_interval)
        freqax = get_spec_block_freqax(block_group, freq_interval)
        
        power_dict["frequencies"] = freqax

        time_index = timeax.searchsorted(time_out)

        loc_group = block_group["locations"]

        if locations is None:
            locations = get_hydro_spec_locations(header_group)

        for location in locations:
            data = loc_group[location][:, time_index]
            power_dict[location] = data

    return power_dict

# Read the headers of a geophone spectrogram file and return them in a dictionary
def read_geo_spec_headers(inpath):
    header_dict = {}
    with File(inpath, 'r') as file:
        header_group = file["headers"]
        
        station = header_group["station"][()]
        station = station.decode("utf-8")

        block_type = header_group["block_type"][()]
        block_type.decode("utf-8")

        window_length = header_group["window_length"][()]
        overlap = header_group["overlap"][()]
        freq_interval = header_group["frequency_interval"][()]

        time_labels = header_group["time_labels"][:]
        time_labels = [time_label.decode("utf-8") for time_label in time_labels]

    header_dict["station"] = station
    header_dict["block_type"] = block_type
    header_dict["window_length"] = window_length
    header_dict["overlap"] = overlap
    header_dict["frequency_interval"] = freq_interval
    header_dict["time_labels"] = time_labels

    return header_dict

# Get the time labels of a geophone spectrogram file from the headers
def get_spec_time_labels(header_group):
    time_labels = header_group["time_labels"][:]
    time_labels = [time_label.decode("utf-8") for time_label in time_labels]

    return time_labels

# Get the time interval of a geophone spectrogram file
def get_spec_time_interval(header_group):
        window_length = header_group["window_length"][()]
        overlap = header_group["overlap"][()]
        time_interval = Timedelta(seconds = window_length * (1 - overlap))

        return time_interval

# Get the frequency axis from the headers of a spectrogram file
def get_spec_block_freqax(block_group, freq_interval):
    num_freqs = block_group["num_freqs"][()]
    freqax = linspace(0, (num_freqs - 1) * freq_interval, num_freqs)

    return freqax


# Get the time axis of a geophone spectrogram block
def get_spec_block_timeax(block_group, time_interval):
    starttime = block_group["start_time"][()]
    starttime = Timestamp(starttime, unit='ns')
    num_times = block_group["num_times"][()]
    endtime = starttime + (num_times - 1) * time_interval
    timeax = date_range(start = starttime, end = endtime, periods = num_times)

    return timeax

# Get the timing of a list of spectrogram blocks
def get_block_timings(data_group, time_interval, time_labels):
    block_starttimes = []
    block_endtimes = []
    block_numtimes = []
    for time_label in time_labels:
        block_group = data_group[time_label]
        starttime = block_group["start_time"][()]
        num_times = block_group["num_times"][()]
        starttime = Timestamp(starttime, unit='ns')
        endtime = starttime + (num_times - 1) * time_interval
        block_starttimes.append(starttime)
        block_endtimes.append(endtime)
        block_numtimes.append(num_times)

    block_timing_df = DataFrame({"time_label": time_labels, "start_time": block_starttimes, "end_time": block_endtimes, "num_times": block_numtimes})

    return block_timing_df

# Get the blocks overlapping with a given time range
def get_overlapping_blocks(block_timing_df, starttime, endtime):
    overlap_df = block_timing_df.loc[(block_timing_df["start_time"] <= endtime) & (block_timing_df["end_time"] >= starttime)]
    if len(overlap_df) == 0:
        print("Warning: No data is read!")

    return overlap_df

# Get the data time indices for slicing a spectrogram data block
def get_data_block_time_indices(starttime_block, num_times, time_interval, starttime_to_read, endtime_to_read):
    start_index = amax([0, int((starttime_to_read - starttime_block) / time_interval)])
    end_index = amin([num_times, int((endtime_to_read - starttime_block) / time_interval)])

    return start_index, end_index

# Get the locations of a hydrophone spectrogram file from the headers
def get_hydro_spec_locations(header_group):
    locations = header_group["locations"][:]
    locations = [location.decode("utf-8") for location in locations]

    return locations


# # Read the hydrophone spectrograms of ALL locations of one stations from an HDF5 file
# # Each location has its own time axis!
# def read_hydro_spectrograms(inpath):
#     with File(inpath, 'r') as file:
#         # Read the header information
#         header_group = file["headers"]
#         station = header_group["station"][()]
#         time_label = header_group["time_label"][()]
#         locations = header_group["locations"][:]

#         freq_interval = header_group["frequency_interval"][()]
#         overlap = header_group["overlap"][()]

#         # Decode the strings
#         station = station.decode("utf-8")
#         time_label = time_label.decode("utf-8")
#         locations = [location.decode("utf-8") for location in locations]

#         # Read the spectrogram data
#         data_group = file["data"]
#         stream_spec = StreamSTFTPSD()
#         for location in locations:
#             loc_group = data_group[location]
#             data = loc_group["psd"][:]
#             starttime = loc_group["start_time"][()]
#             time_interval = loc_group["time_interval"][()]

#             num_freq = data.shape[0]
#             freqax = linspace(0, (num_freq - 1) * freq_interval, num_freq)
            
#             num_time = data.shape[1]
#             timeax = assemble_timeax_from_ints(starttime, num_time, time_interval)
            
#             trace_spec = TraceSTFTPSD(station, location, "H", time_label, timeax, freqax, data)
#             stream_spec.append(trace_spec)

#         return stream_spec    

# Read detected spectral peaks from a CSV or HDF file
def read_spectral_peaks(inpath, **kwargs):
    # If the file format is not given, infer it from the file extension
    if "file_format" in kwargs:
        file_format = kwargs["file_format"]
    else:
        _, file_format = splitext(inpath)
        file_format = file_format.replace(".", "")

    # print(file_format)
        
    if file_format == "csv":
        peak_df = read_csv(inpath, index_col = 0, parse_dates = ["time"])
    elif file_format == "h5" or file_format == "hdf":
        peak_df = read_hdf(inpath, key = "peaks")
    else:
        raise ValueError("Error: Invalid file format!")

    return peak_df

# Save the detected spectral peaks to a CSV or HDF file
def save_spectral_peaks(peak_df, file_stem, file_format, outdir = SPECTROGRAM_DIR):
    if file_format == "csv":
        outpath = join(outdir, f"{file_stem}.csv")
        peak_df.to_csv(outpath, date_format = "%Y-%m-%d %H:%M:%S.%f")
    elif file_format == "h5" or file_format == "hdf":
        outpath = join(outdir, f"{file_stem}.h5")
        peak_df.to_hdf(outpath, key = "peaks", mode = "w")
    else:
        raise ValueError("Error: Invalid file format!")

    print(f"Results saved to {outpath}")

# Read spectral peak bin counts from a CSV or HDF file
def read_spectral_peak_counts(inpath, **kwargs):
    # If the file format is not given, infer it from the file extension
    if "file_format" in kwargs:
        file_format = kwargs["file_format"]
    else:
        _, file_format = splitext(inpath)
        file_format = file_format.replace(".", "")

    # print(file_format)

    if file_format == "csv":
        count_df = read_csv(inpath, index_col = 0, parse_dates = ["time"])
    elif file_format == "h5" or file_format == "hdf":
        count_df = read_hdf(inpath, key = "counts")
    else:
        raise ValueError("Error: Invalid file format!")

    return count_df

# Save spectral-peak bin counts to a CSV or HDF file
def save_spectral_peak_counts(count_df, file_stem, file_format, outdir = SPECTROGRAM_DIR):
    if file_format == "csv":
        outpath = join(outdir, f"{file_stem}.csv")
        count_df.to_csv(outpath, date_format = "%Y-%m-%d %H:%M:%S.%f")
    elif file_format == "h5" or file_format == "hdf":
        outpath = join(outdir, f"{file_stem}.h5")
        count_df.to_hdf(outpath, key = "counts", mode = "w")
    else:
        raise ValueError("Error: Invalid file format!")

    print(f"Results saved to {outpath}")

# Read a binary spectrogram file
def read_binary_spectrogram(inpath, starttime, endtime, min_freq, max_freq):
    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        start_time = header_group["start_time"][()]
        min_freq = header_group["min_freq"][()]
        num_times = header_group["num_times"][()]
        num_freqs = header_group["num_freqs"][()]
        time_interval = header_group["time_interval"][()]
        freq_interval = header_group["freq_interval"][()]

        # Create the time axis
        timeax = assemble_timeax_from_ints(start_time, num_times, time_interval)
        start_time_index = timeax.searchsorted(starttime)
        end_time_index = timeax.searchsorted(endtime)
        timeax = timeax[start_time_index:end_time_index]

        # Create the frequency axis
        freqax = linspace(min_freq, min_freq + (num_freqs - 1) * freq_interval, num_freqs)
        min_freq_index = freqax.searchsorted(min_freq)
        max_freq_index = freqax.searchsorted(max_freq)
        freqax = freqax[min_freq_index:max_freq_index]

        # Read the spectrogram data
        data_group = file["data"]
        data = data_group["detections"][min_freq_index:max_freq_index, start_time_index:end_time_index]

        # Construct the output dictionary
        out_dict = {"times": timeax, "freqs": freqax, "data": data}

    return out_dict

# Save a binary spectrogram file
def save_binary_spectrogram(timeax, freqax, data, outpath):
    timeax = datetime2int(timeax)
    with File(outpath, 'w') as file:
        header_group = file.create_group('headers')
        header_group.create_dataset('start_time', data=timeax[0])
        header_group.create_dataset('min_freq', data=freqax[0])
        header_group.create_dataset('num_times', data=len(timeax))
        header_group.create_dataset('num_freqs', data=len(freqax))
        header_group.create_dataset('time_interval', data=timeax[1] - timeax[0])
        header_group.create_dataset('freq_interval', data=freqax[1] - freqax[0])

        data_group = file.create_group('data')
        data_group.create_dataset('detections', data=data)

    print(f"Binary spectrogram saved to {outpath}")


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

# Get the time label for a spectrogram block from a time string
def string_to_time_label(time_str):
    time_label = Timestamp(time_str).strftime("%Y%m%d%H%M%S%f")

    return time_label
