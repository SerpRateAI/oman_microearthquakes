# Functions and classes for spectrum analysis
from os.path import join, splitext
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, iirfilter, sosfilt, freqz
from scipy.signal.windows import hann
from scipy.interpolate import RegularGridInterpolator
from numpy import abs, amax, arange, array, column_stack, concatenate, convolve, cumsum, delete, interp, load, log10, linspace, amax, median, amin, nan, ones, pi, savez, zeros
from pandas import Series, DataFrame, DatetimeIndex, IntervalIndex, Timedelta, Timestamp
from pandas import concat, crosstab, cut, date_range, read_csv, read_hdf, to_datetime
from matplotlib.pyplot import subplots, get_cmap
from h5py import File, special_dtype
from multitaper import MTSpec
from multiprocessing import Pool
from skimage.morphology import remove_small_objects

from utils_basic import ALL_COMPONENTS, GEO_COMPONENTS
from utils_basic import SPECTROGRAM_DIR, STARTTIME_GEO, ENDTIME_GEO, STARTTIME_HYDRO, ENDTIME_HYDRO
from utils_basic import WATER_HEIGHT
from utils_basic import assemble_timeax_from_ints, convert_boolean, datetime2int, int2datetime, power2db, reltimes_to_timestamps, str2timestamp
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
    def select(self, stations = None, locations = None, components = None, time_labels = None):
        traces = []
        if stations is not None:
            if locations is not None:
                if components is not None:
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.station in stations and trace.location in locations and trace.component in components and trace.time_label in time_labels]
                    else:
                        traces = [trace for trace in self.traces if trace.station in stations and trace.location in locations and trace.component in components]
                else:
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.station in stations and trace.location in locations and trace.time_label in time_labels]
                    else:
                        traces = [trace for trace in self.traces if trace.station in stations and trace.location in locations]
            else:
                if components is not None:
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.station in stations and trace.component in components and trace.time_label in time_labels]
                    else:
                        traces = [trace for trace in self.traces if trace.station in stations and trace.component in components]
                else:
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.station in stations and trace.time_label in time_labels]
                    else:
                        traces = [trace for trace in self.traces if trace.station in stations]
        else:
            if locations is not None:
                if components is not None:
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.location in locations and trace.component in components and trace.time_label in time_labels]
                    else:
                        traces = [trace for trace in self.traces if trace.location in locations and trace.component in components]
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.location in locations and trace.time_label in time_labels]
                    else:
                        traces = [trace for trace in self.traces if trace.location in locations]
                else:
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.location in locations and trace.time_label in time_labels]
                    else:
                        traces = [trace for trace in self.traces if trace.location in locations]        
            else:
                if components is not None:
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.component in components and trace.time_label in time_labels]
                    else:
                        traces = [trace for trace in self.traces if trace.component in components]
                else:
                    if time_labels is not None:
                        traces = [trace for trace in self.traces if trace.time_label in time_labels]
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
            components = ALL_COMPONENTS

        # Stitch the spectrograms of each station, location, and component
        for station in stations:
            for location in locations:
                for component in components:
                    stream_to_stitch = self.select(stations=station, locations=location, components=component)
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
            trace.trim_time(starttime = starttime, endtime = endtime)

    # Trim the spectrograms to the begin and end of the day
    def trim_to_day(self):
        for trace in self.traces:
            trace.trim_to_day()

    # Slice the spectrograms to a given time range without modifying the original traces
    def slice_time(self, starttime = None, endtime = None):
        traces_sliced = []
        for trace in self.traces:
            trace_sliced = trace.copy()
            trace_sliced.trim_time(starttime = starttime, endtime = endtime)
            traces_sliced.append(trace_sliced)
        
        stream_sliced = StreamSTFTPSD(traces_sliced)

        return stream_sliced


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

        if self.num_times > 1:
            self.time_interval = self.times[1] - self.times[0]
        else:
            self.time_interval = None

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
            time_label = start_of_block.strftime("day_%Y%m%d")
        elif block_type == "hourly":
            start_of_block = ref_time.replace(minute=0, second=0, microsecond=0)
            time_label = start_of_block.strftime("hour_%Y%m%d%H")
        else:
            raise ValueError("Error: Invalid block type!")

        self.time_label = time_label

    # Plot the power spectrogram in a given axis
    def plot(self, ax, min_db, max_db):
        if self.db:
            data = self.data
        else:
            data = power2db(self.data)

        timeax = self.times
        freqax = self.freqs
        
        cmap = get_cmap("inferno")
        cmap.set_bad(color='darkgray')

        quadmesh = ax.pcolormesh(timeax, freqax, data, shading = "auto", cmap = cmap, vmin = min_db, vmax = max_db)
        ax.set_xlim(timeax[0], timeax[-1])
        ax.set_ylim(freqax[0], freqax[-1])
        
        return quadmesh

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
    print("Concatenating the results...")
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

# Get the file-name suffix for stationary-resonance detections
def get_stationary_resonance_file_suffix(frac_threshold, prom_frac_threshold):
    suffix = f"frac{frac_threshold:.1f}_prom{prom_frac_threshold:.1f}"

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
    for _, row in counts_df.iterrows():
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

    if downsample and "downsample_factor" not in kwargs:
        raise ValueError("Error: Downsample factor is not given!")

    # Create the output file
    outpath = join(outdir, filename)
    file = File(outpath, 'w')
    
    # Create the group for storing the headers
    header_group = file.create_group('headers')

    # Save the header information
    header_group.attrs['station'] = station
    header_group.attrs['window_length'] = window_length
    header_group.attrs['overlap'] = overlap
    header_group.attrs['block_type'] = "daily"
    header_group.attrs['frequency_interval'] = freq_interval
    header_group.attrs['downsample'] = downsample

    if downsample:
        header_group.attrs['downsample_factor'] = kwargs["downsample_factor"]

    file.create_group('data')

    print(f"Created spectrogram file {outpath}")

    return file

# Save one geophone spectrogram data block to an opened HDF5 file
def write_geo_spectrogram_block(file, stream_spec, close_file = True):
    # Verify the StreamSTFTPSD object
    stream_spec.verify_geo_block()
    
    # Extract the data from the StreamSTFTPSD object
    trace_spec_z = stream_spec.select(components="Z")[0]
    trace_spec_1 = stream_spec.select(components="1")[0]
    trace_spec_2 = stream_spec.select(components="2")[0]

    # Verify the station name
    if trace_spec_z.station != file["headers"].attrs["station"]:
        raise ValueError("Error: Inconsistent station name!")

    time_label = trace_spec_z.time_label
    timeax = trace_spec_z.times
    data_z = trace_spec_z.data
    data_1 = trace_spec_1.data
    data_2 = trace_spec_2.data

    # Convert the time axis to integer
    timeax = datetime2int(timeax)

    # Create a new block and write the data
    data_group = file["data"]
    block_group = data_group.create_group(time_label)
    block_group.attrs["start_time"] = timeax[0]
    block_group.attrs["num_times"] = len(timeax)
    block_group.attrs["num_freqs"] = len(trace_spec_z.freqs)
    
    block_group.create_dataset("data_z", data=data_z)
    block_group.create_dataset("data_1", data=data_1)
    block_group.create_dataset("data_2", data=data_2)

    print(f"Spectrogram block {time_label} is saved")

    # Close the file
    if close_file:
        file.close()

# Finish writing a geophone spectrogram file by writing the list of time labels and close the file
def finish_geo_spectrogram_file(file, time_labels):
    file["data"].create_dataset('time_labels', data=[time_label.encode("utf-8") for time_label in time_labels])
    print("Time labels are saved.")
    
    file.close()
    print("The spectrogram file is closed.")
    
# Read specific segments of geophone-spectrogram data from an HDF5 file 
def read_geo_spectrograms(inpath, time_labels = None, starttime = None, endtime = None, min_freq = 0.0, max_freq = 500.0, components = GEO_COMPONENTS):

    if min_freq is None:
        min_freq = 0.0
    
    if max_freq is None:
        max_freq = 500.0

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
        data_group = file["data"]
        
        station = header_group.attrs["station"]

        time_labels_in = data_group["time_labels"][:]
        time_labels_in = [time_label.decode("utf-8") for time_label in time_labels_in]
        
        freq_interval = header_group.attrs["frequency_interval"]
        # print(f"Frequency interval is {freq_interval}")

        min_freq_index = int(min_freq / freq_interval)
        max_freq_index = int(max_freq / freq_interval)
        
        time_interval = get_spec_time_interval(header_group)
        
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

                for component in components:
                    if component == "Z":
                        data = block_group["data_z"][min_freq_index:max_freq_index, :]
                    elif component == "1":
                        data = block_group["data_1"][min_freq_index:max_freq_index, :]
                    elif component == "2":
                        data = block_group["data_2"][min_freq_index:max_freq_index, :]
        
                    freqax = linspace(min_freq, max_freq, data.shape[0])
                
                    trace_spec = TraceSTFTPSD(station, "", component, time_label, timeax, freqax, data)
                    stream_spec.append(trace_spec)
        else:
            # Option 2: Read the spectrograms of a specific time range
            # Convert the start and end times to Timestamp objects
            if isinstance(starttime, str):
                starttime_to_read = Timestamp(starttime, tz = "UTC")
            else:
                starttime_to_read = starttime
            
            if isinstance(endtime, str):
                endtime_to_read = Timestamp(endtime, tz = "UTC")
            else:
                endtime_to_read = endtime
            
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

                # Find the start and end indices of the time axis
                start_index, end_index = get_data_block_time_indices(starttime_block, num_times, time_interval, starttime_to_read, endtime_to_read)

                for component in components:
                    # Slice the data matrix
                    if component == "Z":
                        data = block_group["data_z"][min_freq_index:max_freq_index, start_index:end_index + 1]
                    elif component == "1":
                        data = block_group["data_1"][min_freq_index:max_freq_index, start_index:end_index + 1]
                    elif component == "2":
                        data = block_group["data_2"][min_freq_index:max_freq_index, start_index:end_index + 1]

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

# Read the frequency and time axes of a geophone spectrogram file for specific ranges
def read_geo_spec_axes(inpath, starttime = None, endtime = None, min_freq = 0.0, max_freq = 500.0):
    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        data_group = file["data"]

        time_labels = data_group["time_labels"][:]
        time_labels = [time_label.decode("utf-8") for time_label in time_labels]
        
        freq_interval = header_group.attrs["frequency_interval"]
        min_freq_index = int(min_freq / freq_interval)
        max_freq_index = int(max_freq / freq_interval)
        freqax = linspace(min_freq, max_freq, max_freq_index - min_freq_index + 1)  

        time_interval = get_spec_time_interval(header_group)
        data_group = file["data"]
        
        if starttime is None and endtime is None:
            # Option 1: Read the axes of the whole deployment
            timeaxes = []
            for time_label in time_labels:
                block_group = data_group[time_label]
                timeax_block = get_spec_block_timeax(block_group, time_interval)
                timeaxes.append(timeax_block)

            timeax = DatetimeIndex(concatenate(timeaxes))
        else:
            # Option 2: Read the axes of a specific time range
            # Convert the start and end times to Timestamp objects
            if isinstance(starttime, str):
                starttime_to_read = Timestamp(starttime)
            
            if isinstance(endtime, str):
                endtime_to_read = Timestamp(endtime)
            
            # Check the start time is greater than the end time
            if starttime_to_read > endtime_to_read:
                raise ValueError("Error: Start time must be less than the end time!")

            # Get the start and end times of each block
            block_timing_df = get_block_timings(data_group, time_labels, time_interval)

            # Find the blocks that overlap with the given time range
            overlap_df = get_overlapping_blocks(block_timing_df, starttime_to_read, endtime_to_read)
                
            # Read the axes of the overlapping blocks
            timeaxes = []
            for _, row in overlap_df.iterrows():
                starttime_block = row["start_time"]
                num_times = row["num_times"]

                # Find the start and end indices of the time axis
                start_index, end_index = get_data_block_time_indices(starttime_block, num_times, time_interval, starttime_to_read, endtime_to_read)

                # Create the time axis
                starttime_out = max([starttime_to_read, starttime_block])
                timeax = date_range(start = starttime_out, periods = end_index - start_index + 1, freq = time_interval)
                timeaxes.append(timeax)

            timeax = DatetimeIndex(concatenate(timeaxes))

    return timeax, freqax

# Create a hydrophone spectrogram file and save the header information
def create_hydro_spectrogram_file(station, locations, window_length = 60.0,  overlap = 0.0, freq_interval = 1.0, downsample = False, outdir = SPECTROGRAM_DIR, **kwargs):
    suffix = get_spectrogram_file_suffix(window_length, overlap, downsample, **kwargs)
    filename = f"whole_deployment_daily_hydro_spectrograms_{station}_{suffix}.h5"

    if downsample and "downsample_factor" not in kwargs:
        raise ValueError("Error: Downsample factor is not given!")

    # Create the output file
    outpath = join(outdir, filename)
    file = File(outpath, 'w')
    
    # Create the group for storing the headers
    header_group = file.create_group('headers')

    # Save the header information with encoding
    header_group.attrs['station'] = station
    header_group.create_dataset('locations', data=[location.encode("utf-8") for location in locations])
    header_group.attrs['window_length'] = window_length
    header_group.attrs['overlap'] = overlap
    header_group.attrs['block_type'] = "daily"
    header_group.attrs['frequency_interval'] = freq_interval
    header_group.attrs['downsample'] = downsample
    if downsample:
        header_group.attrs['downsample_factor'] = kwargs["downsample_factor"]

    # Create the group for storing the spectrogram data (blocks)
    file.create_group('data')

    print(f"Created spectrogram file {outpath}")

    return file

# Save one hydrophone spectrogram data block to an opened HDF5 file
def write_hydro_spectrogram_block(file, stream_spec, locations, close_file = True):
    # Verify the StreamSTFTPSD object
    stream_spec.verify_hydro_block(locations)

    # Verify the station name
    station = file["headers"].attrs["station"]
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
    block_group.attrs["start_time"] = timeax[0]
    block_group.attrs["num_times"] = len(timeax)
    block_group.attrs["num_freqs"] = len(stream_spec[0].freqs)

    for location in locations:
        trace_spec = stream_spec.select(locations=location)[0]
        data = trace_spec.data
        block_group.create_dataset(f"data_{location}", data=data)

    print(f"Spectrogram block {time_label} is saved")

    # Close the file
    if close_file:
        file.close()

# Finish writing a hydrophone spectrogram file by writing the list of time labels and close the file
def finish_hydro_spectrogram_file(file, time_labels):
    file["data"].create_dataset('time_labels', data=[time_label.encode("utf-8") for time_label in time_labels])
    print("Time labels are saved.")
    
    file.close()
    print("The spectrogram file is closed.")

# Read specific segments of hydrophone-spectrogram data from an HDF5 file
def read_hydro_spectrograms(inpath, time_labels = None, locations = None, starttime = None, endtime = None, min_freq = 0.0, max_freq = 500.0):
    if min_freq is None:
        min_freq = 0.0
    
    if max_freq is None:
        max_freq = 500.0

    # Convert the locations to a list
    if locations is not None:
        if not isinstance(locations, list):
            if isinstance(locations, str):
                locations = [locations]
        else:
            raise ValueError("Error: Invalid location format!")

    # Deterimine if both start and end times are provided
    if (starttime is not None and endtime is None) or (starttime is None and endtime is not None):
        raise ValueError("Error: Both start and end times must be provided!")
    
    # Determine if the input time information is redundant
    if starttime is not None and endtime is not None and time_labels is not None:
        raise ValueError("Error: Time labels and start/end times cannot be given at the same time!")
    
    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        data_group = file["data"]
        
        station = header_group.attrs["station"]

        time_labels_in = data_group["time_labels"][:]
        time_labels_in = [time_label.decode("utf-8") for time_label in time_labels_in]

        if locations is None:
            locations = header_group["locations"][:]
            locations = [location.decode("utf-8") for location in locations]

        freq_interval = header_group.attrs["frequency_interval"]

        min_freq_index = int(min_freq / freq_interval)
        max_freq_index = int(max_freq / freq_interval)
        
        time_interval = get_spec_time_interval(header_group)
        
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

                for location in locations:
                    try:
                        data = block_group[f"data_{location}"][min_freq_index:max_freq_index, :]
                    except KeyError:
                        print(f"Warning: Location {location} does not exist for time label {time_label}!")
                        continue
        
                    num_freq = data.shape[0]
                    freqax = linspace(min_freq, max_freq, num_freq)
                
                    trace_spec = TraceSTFTPSD(station, location, "H", time_label, timeax, freqax, data)
                    stream_spec.append(trace_spec)
        else:
            # Option 2: Read the spectrograms of a specific time range
            # Convert the start and end times to Timestamp objects
            if isinstance(starttime, str):
                starttime_to_read = Timestamp(starttime, tz = "UTC")
            else:
                starttime_to_read = starttime
            
            if isinstance(endtime, str):
                endtime_to_read = Timestamp(endtime, tz = "UTC")
            else:
                endtime_to_read = endtime
            
            # Verify the start time is greater than the end time
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

                # Find the start and end indices of the time axis
                start_index, end_index = get_data_block_time_indices(starttime_block, num_times, time_interval, starttime_to_read, endtime_to_read)
                for location in locations:
                    # Slice the data matrix
                    try:
                        data = block_group[f"data_{location}"][min_freq_index:max_freq_index, start_index:end_index + 1]
                    except KeyError:
                        print(f"Warning: Location {location} does not exist for time label {time_label}!")
                        continue

                    # Create the frequency axis
                    freqax = linspace(min_freq, max_freq, data.shape[0])
        
                    # Create the time axis
                    starttime_out = max([starttime_to_read, starttime_block])
                    timeax = date_range(start = starttime_out, periods = data.shape[1], freq = time_interval)

                    # Create the TraceSTFTPSD object
                    trace_spec = TraceSTFTPSD(station, location, "H", time_label, timeax, freqax, data)
                    stream_spec.append(trace_spec)

    # Stitch the spectrograms if there are multiple blocks
    stream_spec.stitch()

    return stream_spec

# Read the power at a specific time from a geophone spectrogram file
def read_time_from_geo_spectrograms(inpath, time_out, components = GEO_COMPONENTS):
    # Convert string to Timestamp
    if isinstance(time_out, str):
        time_out = Timestamp(time_out, tz = "UTC")

    power_dict = {}
    

    with File(inpath, 'r') as file:
        header_group = file["headers"]

        # Get the time labels and start times of each block
        time_labels = get_spec_time_labels(header_group)
        starttimes = to_datetime(time_labels, format = "%Y%m%d%H%M%S%f", utc = True)

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
        time_out = Timestamp(time_out, tz = "UTC")

    power_dict = {}

    with File(inpath, 'r') as file:
        header_group = file["headers"]

        # Get the time labels and start times of each block
        time_labels = get_spec_time_labels(header_group)
        starttimes = to_datetime(time_labels, format = "%Y%m%d%H%M%S%f", utc = True)

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
        
        station = header_group.attrs["station"]
        window_length = header_group.attrs["window_length"]
        overlap = header_group.attrs["overlap"]
        block_type = header_group.attrs["block_type"]
        freq_interval = header_group.attrs["frequency_interval"]
        downsample = header_group.attrs["downsample"]

        header_dict["station"] = station
        header_dict["block_type"] = block_type
        header_dict["window_length"] = window_length
        header_dict["overlap"] = overlap
        header_dict["frequency_interval"] = freq_interval

        if downsample:
            downsample_factor = header_group.attrs["downsample_factor"]
            header_dict["downsample_factor"] = downsample_factor

    return header_dict

# Read the headers of a hydrophone spectrogram file and return them in a dictionary
def read_hydro_spec_headers(inpath):
    header_dict = {}
    with File(inpath, 'r') as file:
        header_group = file["headers"]
        
        station = header_group.attrs["station"]
        window_length = header_group.attrs["window_length"]
        overlap = header_group.attrs["overlap"]
        block_type = header_group.attrs["block_type"]
        freq_interval = header_group.attrs["frequency_interval"]
        downsample = header_group.attrs["downsample"]

        locations = header_group["locations"][:]
        locations = [location.decode("utf-8") for location in locations]

        header_dict["station"] = station
        header_dict["block_type"] = block_type
        header_dict["window_length"] = window_length
        header_dict["overlap"] = overlap
        header_dict["frequency_interval"] = freq_interval
        header_dict["locations"] = locations

        if downsample:
            downsample_factor = header_group.attrs["downsample_factor"]
            header_dict["downsample_factor"] = downsample_factor

    return header_dict

# Read the time labels, begin times, and end times of all blocks in a spectrogram file
def read_spec_block_timings(inpath):
    with File(inpath, 'r') as file:
        header_group = file["headers"]
        data_group = file["data"]

        time_labels = get_spec_time_labels(data_group)
        time_interval = get_spec_time_interval(header_group)

        block_timing_df = get_block_timings(data_group, time_labels, time_interval)

    return block_timing_df

# Read the time labels of a hydrophone spectrogram file
def read_spec_time_labels(inpath):
    with File(inpath, 'r') as file:
        time_labels = file["data"]["time_labels"][:]
        time_labels = [time_label.decode("utf-8") for time_label in time_labels]

    return time_labels

# Append a new header variable to an opened spectrogram file
def append_header_variable(file, var_name, var_value):
    header_group = file["headers"]
    header_group.create_dataset(var_name, data=var_value)
    print(f"Appended variable {var_name} to the header.")

# Get the time labels of a geophone spectrogram file from the headers
def get_spec_time_labels(data_group):
    time_labels = data_group["time_labels"][:]
    time_labels = [time_label.decode("utf-8") for time_label in time_labels]

    return time_labels

# Get the time interval of a geophone spectrogram file
def get_spec_time_interval(header_group):
        window_length = header_group.attrs["window_length"]
        overlap = header_group.attrs["overlap"]
        time_interval = Timedelta(seconds = window_length * (1 - overlap))

        return time_interval

# Get the frequency axis from the headers of a spectrogram file
def get_spec_block_freqax(block_group, freq_interval):
    num_freqs = block_group["num_freqs"][()]
    freqax = linspace(0, (num_freqs - 1) * freq_interval, num_freqs)

    return freqax

# Get the time axis of a geophone spectrogram block
def get_spec_block_timeax(block_group, time_interval):
    starttime = block_group.attrs["start_time"]
    starttime = Timestamp(starttime, unit='ns', tz = "UTC")
    num_times = block_group.attrs["num_times"]
    endtime = starttime + (num_times - 1) * time_interval
    timeax = date_range(start = starttime, end = endtime, periods = num_times)

    return timeax

# Get the timing of a list of spectrogram blocks
def get_block_timings(data_group, time_labels, time_interval):
    block_starttimes = []
    block_endtimes = []
    block_numtimes = []
    for time_label in time_labels:
        block_group = data_group[time_label]
        starttime = block_group.attrs["start_time"]
        num_times = block_group.attrs["num_times"]
        starttime = Timestamp(starttime, unit='ns', tz = "UTC")
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
    start_index = amax([0, int(round((starttime_to_read - starttime_block) / time_interval))])
    end_index = amin([num_times, int(round((endtime_to_read - starttime_block) / time_interval)) + 1])

    return start_index, end_index

# Get the locations of a hydrophone spectrogram file from the headers
def get_hydro_spec_locations(header_group):
    locations = header_group["locations"][:]
    locations = [location.decode("utf-8") for location in locations]

    return locations

# # Read detected spectral peaks from a CSV or HDF file
# def read_spectral_peaks(inpath, **kwargs):
#     # If the file format is not given, infer it from the file extension
#     if "file_format" in kwargs:
#         file_format = kwargs["file_format"]
#     else:
#         _, file_format = splitext(inpath)
#         file_format = file_format.replace(".", "")

#     # print(file_format)
        
#     if file_format == "csv":
#         peak_df = read_csv(inpath, index_col = 0, parse_dates = ["time"])
#     elif file_format == "h5" or file_format == "hdf":
#         peak_df = read_hdf(inpath, key = "peaks")
#     else:
#         raise ValueError("Error: Invalid file format!")

#     peak_df["time"] = peak_df["time"].dt.tz_localize("UTC")
 
#     return peak_df

# Extract a stationary resonance from the spectral-peak data of a geophone station
def extract_stationary_resonance(peak_df, array_count_df, timeax, freqax, min_patch_size = 3):
    # Create the boolean array with each True value representing a peak
    print("Creating the boolean array representing the peaks")
    peak_array = array([[False] * len(timeax)] * len(freqax))
    peak_row_ind_array = zeros(peak_array.shape, dtype = int)

    # print(data.shape)
    # print(peak_array.shape)

    for index, row in peak_df.iterrows():
        time = row["time"]
        freq = row["frequency"]

        # print(timeax[0].tzinfo)
        # print(time.tzinfo)

        time_index = timeax.searchsorted(time)
        freq_index = freqax.searchsorted(freq)
        peak_array[freq_index, time_index] = True
        peak_row_ind_array[freq_index, time_index] = index

    # Remove isolated peaks
    print("Removing isolated peaks...")
    peak_array_clean = remove_small_objects(peak_array, min_size = min_patch_size, connectivity = 2)

    # Recover the peaks from the cleaned array
    print("Recovering the peaks from the cleaned array...")
    peak_clean_indices = peak_array_clean.nonzero()
    peak_clean_row_inds = peak_row_ind_array[peak_clean_indices]

    peak_clean_df = peak_df.loc[peak_clean_row_inds]
    print(f"{peak_clean_df.shape[0]:d} peaks are left after removing isolated peaks.")

    # Group the peaks by time
    print("Grouping the peaks by time...")
    peak_group_by_time = peak_clean_df.groupby("time")
    print(f"In total {len(peak_group_by_time):d} time groups of peaks.")

    # Extract the peaks using the array counts as a guide
    print("Extracting the stationary resonance peaks...")
    resonance_peaks = []
    for time, group in peak_group_by_time:
        if len(group) == 1:
            resonance_peaks.append(group)
        else:
            freq_peak = get_peak_freq_w_max_detection(array_count_df, time)
            if freq_peak is None:
                continue
            else:
                # print(f"Time: {time}, Peak frequency: {freq_peak}")
                max_detect_peak = group.loc[group["frequency"] == freq_peak]
                if len(max_detect_peak) == 0:
                    # print("No peak is found!")
                    continue
                else:
                    # print(type(max_detect_peak))
                    resonance_peaks.append(max_detect_peak)
            
    # Construct a data frame using the elements of resonance_peaks
    resonance_df = concat(resonance_peaks, ignore_index=True)
    print(f"Found {len(resonance_peaks):d} stationary resonance peaks.")

    # Sort the data frame by time
    resonance_df.sort_values("time", inplace = True)
    resonance_df.set_index("time", inplace = True)

    return resonance_df

# Get the lower and upper bounds of a fraction peak
# Find the first local minima on the left and right of a peak that is more than height_factor * prominence lower than the peak in log scale
def get_stationary_resonance_freq_intervals(stationary_resonance_df, peak_dict, count_df, height_factor = 0.8):
    # Convert the fractions log scale
    freqax = count_df["frequency"].values
    fracs = count_df["fraction"].values
    log_fracs = log10(fracs)

    # Find the upper and lower bounds for each peak
    freqs_lower = []
    freqs_upper = []
    for i, i_freq_peak in enumerate(stationary_resonance_df["freq_index"]):
        prominence = peak_dict["prominences"][i]
        i_left_base = peak_dict["left_bases"][i]
        i_right_base = peak_dict["right_bases"][i]

        freq_lower, freq_upper = get_fraction_peak_bounds(freqax, log_fracs, prominence, i_freq_peak, i_left_base, i_right_base, height_factor)
        freqs_lower.append(freq_lower)
        freqs_upper.append(freq_upper)

    stationary_resonance_df["lower_freq_bound"] = freqs_lower
    stationary_resonance_df["upper_freq_bound"] = freqs_upper

    return stationary_resonance_df

# Get the spectral peak with the maximum number of array detection at a specific time
def get_peak_freq_w_max_detection(count_df, time):
    time_slice_df = count_df.loc[count_df["time"] == time]

    if len(time_slice_df) == 0:
        return None
    else:
        peak = time_slice_df.loc[time_slice_df["count"].idxmax()]
        freq = peak["frequency"]

        return freq

# Read the geophone spectral peaks from an HDF file
def read_geo_spec_peaks(inpath, time_labels = None, starttime = None, endtime = None, min_freq = 0.0, max_freq = 200.0):
    # Read the time labels, start times, and end times of all blocks
    block_timing_df = read_hdf(inpath, key = "block_timings")

    # Convert the start and end times to Timestamp objects
    if starttime is not None:
        starttime = str2timestamp(starttime)
    
    if endtime is not None:
        endtime = str2timestamp(endtime)

    # Read the spectral peaks
    pead_dfs = []
    if time_labels is None and starttime is None and endtime is None:
        print("Reading all spectral peaks...")
        for time_label in block_timing_df["time_label"]:
            peak_df = read_hdf(inpath, key = time_label, index_col = 0, parse_dates = ["time"])
            peak_df = peak_df.loc[(peak_df["frequency"] >= min_freq) & (peak_df["frequency"] <= max_freq)]
            print(f"Time label {time_label}: {len(peak_df)} peaks")
            pead_dfs.append(peak_df)

    elif time_labels is not None and starttime is None and endtime is None:
        print("Reading spectral peaks for the given time labels...")
        for time_label in time_labels:
            try:
                peak_df = read_hdf(inpath, key = time_label, index_col = 0, parse_dates = ["time"])
                print(f"Time label {time_label}: {len(peak_df)} peaks")
                peak_df = peak_df.loc[(peak_df["frequency"] >= min_freq) & (peak_df["frequency"] <= max_freq)]
                pead_dfs.append(peak_df)
            except KeyError:
                print(f"Warning: Time label {time_label} does not exist!")
        
    elif time_labels is None and starttime is not None and endtime is not None:
        print(f"Reading spectral peaks between {starttime} and {endtime}...")
        for _, row in block_timing_df.iterrows():
            time_label = row["time_label"]
            starttime_block = row["start_time"]
            endtime_block = row["end_time"]

            if starttime_block <= endtime and endtime_block >= starttime:
                peak_df = read_hdf(inpath, key = time_label, index_col = 0, parse_dates = ["time"])
                peak_df = peak_df.loc[(peak_df["time"] >= starttime) & (peak_df["time"] <= endtime) & (peak_df["frequency"] >= min_freq) & (peak_df["frequency"] <= max_freq)]
                print(f"Time label {time_label}: {len(peak_df)} peaks")
                pead_dfs.append(peak_df)

    else:
        raise ValueError("Error: Redundant input time information!")

    if len(pead_dfs) == 0:
        print("Warning: No spectral peaks are read!")
        return None
    else:
        peak_df = concat(pead_dfs, ignore_index = True)

        return peak_df

# Read the hydrophone spectral peaks from an HDF file
def read_hydro_spec_peaks(inpath, time_labels = None, locations = None):
    if time_labels is None:
        time_label_sr = read_hdf(inpath, key = "time_label")
        time_labels_to_read = [time_label_sr.values[0]]
    else:
        if isinstance(time_labels, str):
            time_labels_to_read = [time_labels]
        else:
            time_labels_to_read = time_labels

    if locations is None:
        locations_sr = read_hdf(inpath, key = "location")
        locations_to_read = locations_sr.values[0]
    else:
        if isinstance(locations, str):
            locations_to_read = [locations]
        else:
            locations_to_read = locations

    peak_dfs = []
    for time_label in time_labels_to_read:
        peak_df = read_hdf(inpath, key = time_label, index_col = 0, parse_dates = ["time"])
        print(len(peak_df.loc[peak_df["location"] == "03"]))
        print(len(peak_df.loc[peak_df["location"] == "04"]))
        print(len(peak_df.loc[peak_df["location"] == "05"]))
        print(len(peak_df.loc[peak_df["location"] == "06"]))
        peak_df = peak_df.loc[peak_df["location"].isin(locations_to_read)]
        peak_dfs.append(peak_df)
    
    peak_df = concat(peak_dfs)

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


# Read spectral peak counts from an HDF5 file
# The counts are organized by time labels
def read_spec_peak_array_counts(inpath, time_labels = None, starttime = None, endtime = None, min_freq = 0.0, max_freq = 200.0):
    # Read the block timings
    block_timing_df = read_hdf(inpath, key = "block_timings")

    # Convert the start and end times to Timestamp objects
    if starttime is not None:
        starttime = str2timestamp(starttime)

    if endtime is not None:
        endtime = str2timestamp(endtime)

    
    count_dfs = []
    if time_labels is None and starttime is None and endtime is None:
        print("Reading all spectral peak counts...")
        for time_label in block_timing_df["time_label"]:
            count_df = read_hdf(inpath, key = time_label, index_col = 0, parse_dates = ["time"])
            count_df = count_df.loc[(count_df["frequency"] >= min_freq) & (count_df["frequency"] <= max_freq)]
            print(f"Time label {time_label}: {len(count_df)} counts")
            count_dfs.append(count_df)

    elif time_labels is not None and starttime is None and endtime is None:
        print("Reading spectral peak counts for the given time labels...")
        for time_label in time_labels:
            try:
                count_df = read_hdf(inpath, key = time_label, index_col = 0, parse_dates = ["time"])
                count_df = count_df.loc[(count_df["frequency"] >= min_freq) & (count_df["frequency"] <= max_freq)]
                print(f"Time label {time_label}: {len(count_df)} counts")
                count_dfs.append(count_df)
            except KeyError:
                print(f"Warning: Time label {time_label} does not exist!")

    elif time_labels is None and starttime is not None and endtime is not None:
        print(f"Reading spectral peak counts between {starttime} and {endtime}...")
        for _, row in block_timing_df.iterrows():
            time_label = row["time_label"]
            starttime_block = row["start_time"]
            endtime_block = row["end_time"]

            if starttime_block <= endtime and endtime_block >= starttime:
                count_df = read_hdf(inpath, key = time_label, index_col = 0, parse_dates = ["time"])
                count_df = count_df.loc[(count_df["time"] >= starttime) & (count_df["time"] <= endtime) & (count_df["frequency"] >= min_freq) & (count_df["frequency"] <= max_freq)]
                print(f"Time label {time_label}: {len(count_df)} counts")
                count_dfs.append(count_df)

    if len(count_dfs) == 0:
        print("Warning: No spectral peak counts are read!")
        return None
    else:
        count_df = concat(count_dfs)
    
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
def read_binary_spectrogram(inpath, starttime = STARTTIME_GEO, endtime = ENDTIME_GEO, min_freq = 0.0, max_freq = 500.0):
    with File(inpath, 'r') as file:
        # Read the header information
        header_group = file["headers"]
        start_time = header_group.attrs["start_time"]
        min_freq = header_group.attrs["min_freq"]
        num_times = header_group.attrs["num_times"]
        num_freqs = header_group.attrs["num_freqs"]
        time_interval = header_group.attrs["time_interval"]
        freq_interval = header_group.attrs["freq_interval"]

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
        header_group.attrs["start_time"] = timeax[0]
        header_group.attrs["min_freq"] = freqax[0]
        header_group.attrs["num_times"] = len(timeax)
        header_group.attrs["num_freqs"] = len(freqax)
        header_group.attrs["time_interval"] = timeax[1] - timeax[0]
        header_group.attrs["freq_interval"] = freqax[1] - freqax[0]

        data_group = file.create_group('data')
        data_group.create_dataset('detections', data=data)

    print(f"Binary array spectrogram saved to {outpath}")

# Compute the probablistic power spectral density of the frequency-power data in a DataFrame
def get_prob_psd(psd_df, min_freq = None, max_freq = None, freq_bin_width = 0.1, min_db = None, max_db = None, db_bin_width = 1.0):
    # If the frequency and power ranges are not given, use the minimum and maximum values in the DataFrame
    if min_freq is None:
        min_freq = psd_df["frequency"].min()
    
    if max_freq is None:
        max_freq = psd_df["frequency"].max()

    if min_db is None:
        min_db = psd_df["power"].min()

    if max_db is None:
        max_db = psd_df["power"].max()

    # Define the bins
    bin_edges_db = arange(min_db, max_db + db_bin_width, db_bin_width)
    bin_edges_freq = arange(min_freq, max_freq + freq_bin_width, freq_bin_width)

    # print(f"Number of bins for power: {len(bin_edges_db) - 1}")
    # print(f"Number of bins for frequency: {len(bin_edges_freq) - 1}")

    # Ensure consistent use of `cut` parameters
    psd_df["power_bin"] = cut(psd_df["power"], bin_edges_db, right=False, include_lowest=True)
    psd_df["frequency_bin"] = cut(psd_df["frequency"], bin_edges_freq, right=False, include_lowest=True)

    # Create contingency tables
    count_table_df = crosstab(psd_df["frequency_bin"], psd_df["power_bin"])

    # Reindex the contingency tables
    interval_index_db = IntervalIndex(psd_df["power_bin"].cat.categories)
    interval_index_freq = IntervalIndex(psd_df["frequency_bin"].cat.categories)
    count_table_df = count_table_df.reindex(index=interval_index_freq, columns=interval_index_db, fill_value=0)

    # Convert to percentage
    percent_table_df = count_table_df / count_table_df.sum().sum() * 100

    # Convert to probability density
    percent_table_df = percent_table_df / (freq_bin_width * db_bin_width)

    # Transpose to make the frequency bins the columns
    percent_table_df = percent_table_df.T

    return bin_edges_freq, bin_edges_db, percent_table_df

###### Handling harmonic data tables ######
def read_harmonic_data_table(inpath):
    harmo_df = read_csv(inpath, index_col = "name", converters = {"detected": convert_boolean})

    return harmo_df

###### Multi-taper spectral analysis ######

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

###### Borehole organpipe modes ######
# Calculate the predicted frequencies of the organpipe modes
def get_organpipe_freqs(orders, sound_speed = 1500.0):
    water_height = WATER_HEIGHT

    freqs = []
    for order in orders:
        wavelength = water_height / order * 4
        freq = sound_speed / wavelength
        freqs.append(freq)

    return freqs


###### Basic functions ######
# Get the upper and lower bounds of a fraction peak
def get_fraction_peak_bounds(freqax, log_fracs, prominence, i_peak, i_left_base, i_right_base, height_factor):
        log_peak_frac = log_fracs[i_peak]

        # Get the lower and upper bounds of the peak
        # print(f"Peak index: {i_peak}")
        # print(f"Looking for the left bound...")
        i = i_peak - 1
        while i >= i_left_base:
            # print(f"i: {i}, log_fracs[i]: {log_fracs[i]}")
            if log_fracs[i] >= log_fracs[i + 1] and log_fracs[i] < log_peak_frac - height_factor * prominence:
                break
            i -= 1

        freq_lower = freqax[i]

        i = i_peak + 1
        while i <= i_right_base:
            # print(f"i: {i}, log_fracs[i]: {log_fracs[i]}")
            if log_fracs[i] >= log_fracs[i - 1] and log_fracs[i] < log_peak_frac - height_factor * prominence:
                break
            i += 1

        freq_upper = freqax[i]

        return freq_lower, freq_upper

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

# Determine if the input time information is redundant
def check_time_input_redundancy(time_labels, starttime, endtime):
    if starttime is not None and endtime is not None and time_labels is not None:
        raise ValueError("Error: Time labels and start/end times cannot be given at the same time!")
