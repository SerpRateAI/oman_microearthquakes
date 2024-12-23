# Functions and classes for array processing
from obspy import Stream, Trace
from obspy.core.util import AttribDict
from numpy import amax, amin, arctan2, cos,  exp, inf, linspace, meshgrid, pi, sqrt, zeros, unravel_index
from numpy.linalg import norm
from pandas import Timedelta, Timestamp
from skimage.feature import peak_local_max
from multiprocessing import Pool
from h5py import File
from skimage.feature import peak_local_max

from utils_basic import GEO_COMPONENTS
from utils_basic import GEO_STATIONS_A as stations_a, GEO_STATIONS_B as stations_b
from utils_basic import INNER_STATIONS_A as inner_stations_a, INNER_STATIONS_B as inner_stations_b 
from utils_basic import MIDDLE_STATIONS_A as middle_stations_a, MIDDLE_STATIONS_B as middle_stations_b
from utils_basic import OUTER_STATIONS_A as outer_stations_a, OUTER_STATIONS_B as outer_stations_b
from utils_basic import utcdatetime_to_timestamp
from utils_plot import plot_beam_images
from utils_spec import get_peak_power_n_prominence, get_start_n_end_from_center

### Classes ###
### Class for storing the 3C stacked Fourier beam images of a collection of time windows
class FourierBeamImageStack:
    def __init__(self, num_stack, bimage_dict, xslowax, yslowax, starttimes, dur):
        self.num_stack = num_stack
        self.bimage_obj_dict = bimage_dict
        self.xslowax = xslowax
        self.yslowax = yslowax
        self.starttimes = starttimes
        self.dur = dur

    def plot_beam_images(self, plot_global_max = True, plot_local_max = False):
        fig, axes = plot_beam_images(self.xslowax, self.yslowax, self.bimage_dict,
                                     plot_global_max = plot_global_max, plot_local_max = plot_local_max)

        return fig, axes

### Class for storing the 3C Fourier beam-forming parameters and results of a collection of time windows
class FourierBeamCollection:
    def __init__(self, beam_windows):
        self.beam_windows = beam_windows
        
        self.starttimes = [beam_window.starttime for beam_window in beam_windows]

        self.num_windows = len(beam_windows)
        self.freq = beam_windows[0].freq
        self.dur = beam_windows[0].dur

    def __len__(self):
        return self.num_windows

    def get_stacked_beam_images(self, min_coherence = None):        
        xslowax = self.beam_windows[0].xslowax
        yslowax = self.beam_windows[0].yslowax

        bimage_stack_dict = {"1":zeros((len(yslowax), len(xslowax))), "2":zeros((len(yslowax), len(xslowax))), "Z":zeros((len(yslowax), len(xslowax)))}
        num_stack = 0
        for beam_window in self.beam_windows:
            if min_coherence is not None:
                cohe_dict = beam_window.get_coherences()
                if any([cohe_dict[component] < min_coherence for component in GEO_COMPONENTS]):
                    continue

            for component in GEO_COMPONENTS:
                bimage = beam_window.bimage_dict[component].bimage
                bimage_stack_dict[component] += bimage

            num_stack += 1

        print(f"Number of beam images included in the stack: {num_stack}")

        bimage_obj_dict = {}
        for component in GEO_COMPONENTS:
            bimage_stack_dict[component] = bimage_stack_dict[component] / amax(bimage_stack_dict[component])
            xslow_global_max, yslow_global_max = get_global_maxima(xslowax, yslowax, bimage_stack_dict[component])
            vel_app, azimuth = slowness2velapp(xslow_global_max, yslow_global_max)
            bimage_obj = FourierBeamImage([], None, None, bimage_stack_dict[component], xslow_global_max, yslow_global_max, vel_app, azimuth, [], [])
            bimage_obj_dict[component] = bimage_obj

        # bimage_obj_dict = {}
        # num_stack = 0
        # for component in GEO_COMPONENTS:
        #     bimage_stack = zeros((len(yslowax), len(xslowax)))
        #     for beam_window in self.beam_windows:
        #         if min_coherence is not None:
        #             cohe_dict = beam_window.get_coherences()
        #             print(cohe_dict)
        #             if any([cohe_dict[component] < min_coherence for component in GEO_COMPONENTS]):
        #                 continue
                            
        #         bimage = beam_window.bimage_dict[component].bimage
        #         bimage_stack += bimage
        #         num_stack += 1

        #     bimage_stack = bimage_stack / amax(bimage_stack)
        #     xslow_global_max, yslow_global_max = get_global_maxima(xslowax, yslowax, bimage_stack)
        #     vel_app, azimuth = slowness2velapp(xslow_global_max, yslow_global_max)

        #     bimage_obj = FourierBeamImage([], None, None, bimage_stack, xslow_global_max, yslow_global_max, vel_app, azimuth, [], [])
        #     bimage_obj_dict[component] = bimage_obj

        # print(f"Number of beam images included in the stack: {num_stack}")

        return xslowax, yslowax, bimage_obj_dict, num_stack

    def get_all_local_maxima(self):
        local_max_dict = {}
        for component in GEO_COMPONENTS:
            xslow_local_max = []
            yslow_local_max = []

            for beam_window in self.beam_windows:
                xslow_local_max.extend(beam_window.bimage_dict[component].xslows_local_max)
                yslow_local_max.extend(beam_window.bimage_dict[component].yslows_local_max)
            
            local_max_dict[component] = (xslow_local_max, yslow_local_max)

        return local_max_dict

    def to_hdf(self, outpath):
        with File(outpath, "w") as file:
            file.attrs["freq"] = self.freq
            file.attrs["dur"] = self.dur.total_seconds()

            file.create_dataset("start_times", data = [beam_window.starttime.strftime("%Y-%m-%dT%H:%M:%S") for beam_window in self.beam_windows])
            file.create_dataset("x_slow_axis", data = self.beam_windows[0].xslowax)
            file.create_dataset("y_slow_axis", data = self.beam_windows[0].yslowax)

            for _, beam_window in enumerate(self.beam_windows):
                starttime_str = beam_window.starttime.strftime("%Y-%m-%dT%H:%M:%S")
                window_group = file.create_group(starttime_str)

                window_group.attrs["starttime"] = beam_window.starttime.value
                
                for component in GEO_COMPONENTS:
                    bimage = beam_window.bimage_dict[component]
                    component_group = window_group.create_group(f"beam_image_{component.lower()}")
                    
                    component_group.attrs["mean_peak_power"] = bimage.mean_peak_power
                    component_group.attrs["coherence"] = bimage.coherence
                    component_group.attrs["x_slow_global_max"] = bimage.xslow_global_max
                    component_group.attrs["y_slow_global_max"] = bimage.yslow_global_max
                    component_group.attrs["vel_app"] = bimage.vel_app
                    component_group.attrs["azimuth"] = bimage.azimuth

                    component_group.create_dataset("stations", data = [station.encode("utf-8") for station in bimage.stations])
                    component_group.create_dataset("beam_image", data = bimage.bimage)
                    component_group.create_dataset("x_slows_local_max", data = bimage.xslows_local_max)
                    component_group.create_dataset("y_slows_local_max", data = bimage.yslows_local_max)

        print(f"Data saved to {outpath}.")  

    
### Class for storing the 3C Fourier beam-forming parameters and results of one time window
### Slownesses and apparent velocities are in m/s and azimuths are in degrees ###
class FourierBeamWindow:
    def __init__(self, freq, starttime, xslowax, yslowax, bimage_dict, **kwargs):
        if "endtime" in kwargs:
            endtime = kwargs["endtime"]
            dur = endtime - starttime
        elif "dur" in kwargs:
            dur = kwargs["dur"]

            if isinstance(dur, float):
                dur = Timedelta(seconds = dur)

        self.freq = freq
        self.starttime = starttime
        self.dur = dur
        self.endtime = starttime + dur
        self.xslowax = xslowax
        self.yslowax = yslowax
        self.bimage_dict = bimage_dict

    def plot_beam_images(self):
        xslowax = self.xslowax
        yslowax = self.yslowax

        fig, axes = plot_beam_images(xslowax, yslowax, self.bimage_dict)

        return fig, axes

    def get_coherences(self):
        coherence_dict = {}
        for component in GEO_COMPONENTS:
            bimage = self.bimage_dict[component]
            coherence_dict[component] = bimage.coherence

        return coherence_dict

        
### Class for storing the Fourier beam-forming image of one component for one time window
class FourierBeamImage:
    def __init__(self, stations, mean_peak_power, coherence, bimage, xslow_global_max, yslow_global_max, vel_app, azimuth, xslows_local_max, yslows_local_max):   
        self.stations = stations
        self.mean_peak_power = mean_peak_power
        self.coherence = coherence

        self.bimage = bimage
        self.xslow_global_max = xslow_global_max
        self.yslow_global_max = yslow_global_max

        self.vel_app = vel_app
        self.azimuth = azimuth
        self.xslows_local_max = xslows_local_max
        self.yslows_local_max = yslows_local_max

### Class for storing the 3C time-domain beamforming parameters and results of a time window
class BeamWindow:
    def __init__(self, starttime, xslowax, yslowax, beamdict, **kwargs):
        if "endtime" in kwargs:
            endtime = kwargs["endtime"]
            dur = endtime - starttime
        elif "dur" in kwargs:
            dur = kwargs["dur"]

            if isinstance(dur, float):
                dur = Timedelta(seconds = dur)

        self.starttime = starttime
        self.dur = dur
        self.endtime = starttime + dur
        self.xslowax = xslowax
        self.yslowax = yslowax
        self.beamdict = beamdict

    def plot_beam_images(self):
        xslowax = self.xslowax
        yslowax = self.yslowax

        fig, axes = plot_beam_images(xslowax, yslowax, self.beamdict)

        return fig, axes

### Class for storing the time-domain beam image of one component for one time window
class BeamImage:
    def __init__(self, stations, bimage, xslow_global_max, yslow_global_max, vel_app, azimuth, xslows_local_max, yslows_local_max):
        self.stations = stations
        self.bimage = bimage
        self.xslow_global_max = xslow_global_max
        self.yslow_global_max = yslow_global_max
        self.vel_app = vel_app
        self.azimuth = azimuth
        self.xslows_local_max = xslows_local_max
        self.yslows_local_max = yslows_local_max

### Time domain beamforming ###
# Compute the 3C beam images of a time window
# This is a wrapper function around get_beam_window
def get_beam_window(stream_in, xslowax, yslowax,
                    normalize = False, 
                    min_distance = 1, local_max_threshold = 0.8,
                    station_dict = None):
    
    # Check if the input is a Stream object
    if not isinstance(stream_in, Stream):
        raise TypeError("Input must be a Stream object!")
    
    # Check if the stations coordinates are set in the input stream_in
    for trace in stream_in:
        if not hasattr(trace.stats, "coordinates"):
            raise ValueError("Station coordinates are not set in the Stream object!")
        
    # Check if the traces have the same length
    if len(set([len(trace.data) for trace in stream_in])) > 1:
        raise ValueError("Traces must have the same length!")

    # Normalize the traces
    if normalize:
        stream_in.normalize()

    ### Perform beamforming for each component
    bimage_obj_dict = {}
    for component in GEO_COMPONENTS:
        stream = stream_in.select(component=component)

        if station_dict is not None:
            stream_bf = Stream()
            stations_bf = station_dict[component]
            for trace in stream:
                if trace.stats.station in stations_bf:
                    stream_bf.append(trace)
        
        print(f"Beamforming {component} component with {len(stream)} stations...")
        bimage_obj = get_beam_image(stream_bf, xslowax, yslowax, 
                                    min_distance = min_distance, local_max_threshold = local_max_threshold)

        bimage_obj_dict[component] = bimage_obj

    # Create a BeamWindow object
    starttime = stream_in[0].stats.starttime
    starttime = utcdatetime_to_timestamp(starttime)
    endtime = stream_in[0].stats.endtime
    endtime = utcdatetime_to_timestamp(endtime)
    beam_window = BeamWindow(starttime, xslowax, yslowax, bimage_obj_dict, endtime = endtime)

    return beam_window

# Compute the beam image for a component
def get_beam_image(stream, xslowax, yslowax, 
                   min_distance = 1, local_max_threshold = 0.8):
    
    # Check if all traces in the stream are of the same component
    if len(set([trace.stats.component for trace in stream])) > 1:
        raise ValueError("All traces must be of the same component!")
    
    # Compute the center of mass of the stations
    x0 = 0.0
    y0 = 0.0
    for trace in stream:
        x0 += trace.stats.coordinates.x
        y0 += trace.stats.coordinates.y

    x0 = x0 / len(stream)
    y0 = y0 / len(stream)

    # Loop over the slowness grid
    bimage = zeros((len(yslowax), len(xslowax)))
    for i, xslow in enumerate(xslowax):
        for j, yslow in enumerate(yslowax):
            slowness = (xslow, yslow)
            bimage[j, i] = get_beam_power(stream, slowness, (x0, y0))

    # Normalize the beam image
    bimage = bimage / amax(bimage)

    # Find the slowness of the maximum beam power
    max_power_idx = bimage.argmax()
    max_i_yslow, max_i_xslow = unravel_index(max_power_idx, bimage.shape)
    xslow_global_max = xslowax[max_i_xslow]
    yslow_global_max = yslowax[max_i_yslow]

    # Convert the slowness of the global maxima to apparent velocity and azimuth
    vel_app, azimuth = slowness2velapp(xslow_global_max, yslow_global_max)

    # Find the local maxima
    peak_inds = peak_local_max(bimage, min_distance=min_distance, threshold_abs=local_max_threshold)

    xslows_local_max = xslowax[peak_inds[:, 1]]
    yslows_local_max = yslowax[peak_inds[:, 0]]

    # Create a BeamImage object
    stations = [trace.stats.station for trace in stream]
    bimage_obj = BeamImage(stations, bimage, xslow_global_max, yslow_global_max, vel_app, azimuth, xslows_local_max, yslows_local_max)

    return bimage_obj

# Get the beam power for a given slowness
def get_beam_power(stream, slowness, refcoord):
    xslow = slowness[0]
    yslow = slowness[1]

    x0 = refcoord[0]
    y0 = refcoord[1]

    numpts = stream[0].stats.npts
    sum_signal = zeros(numpts)
    for trace in stream:
        signal = trace.data
        sampling_int = 1 / trace.stats.sampling_rate
        x = trace.stats.coordinates.x
        y = trace.stats.coordinates.y

        #### Compute the time shift
        time_shift = -xslow * (x - x0) - yslow * (y - y0)
        shift_samples = int(time_shift / sampling_int)

        #### Shift the signal
        shifted_signal = zeros(len(signal))
        if shift_samples > 0:
            shifted_signal[shift_samples:] = signal[:-shift_samples]
        elif shift_samples < 0:
            shifted_signal[:shift_samples] = signal[-shift_samples:]
        else:
            shifted_signal = signal

        sum_signal += shifted_signal
    
    power = norm(sum_signal)

    return power

## Set the station coordinates in the Stream object
def set_station_coords(stream, stacoords):
    for trace in stream:
        station = trace.stats.station
        x = stacoords.loc[station, "east"]
        y = stacoords.loc[station, "north"]

        trace.stats.coordinates = AttribDict({"x":x, "y":y})

    return stream

### Fourier beamforming ###
# Compute the 3C Fourier beam images of a time window
# This is a wrapper function around get_fourier_beam_image
def get_fourier_beam_window(freq_peak, peak_coeff_df, coords_df, xslowax, yslowax, window_length,
                            min_num_stations = 10, 
                            min_distance = 1, local_max_threshold = 0.8):

    # Verify all rows belong to the same time window
    if len(set(peak_coeff_df["time"])) > 1:
        raise ValueError("All rows must belong to the same time window!")

    # Get the start and end times of the time window
    centertime = peak_coeff_df["time"].values[0]
    starttime, endtime = get_start_n_end_from_center(centertime, window_length)

    # Compute the Fourier beam images for each component
    bimage_dict = {}
    for component in GEO_COMPONENTS:
        print(f"Computing the Fourier beam image for {component} component...")
        peak_sta_dict = {}

        for _, row in peak_coeff_df.iterrows():
            station = row["station"]
            x = coords_df.loc[station, "east"]
            y = coords_df.loc[station, "north"]
 
            peak_pha = row[f"phase_{component.lower()}"]
            peak_power = row[f"power_{component.lower()}"]
            peak_coeff = exp(1j * peak_pha / 180 * pi)

            peak_sta_dict[station] = {"x":x, "y":y, "peak_coeff":peak_coeff, "peak_power":peak_power}

        print(f"Number of stations with spectral peaks meeting the criteria: {len(peak_sta_dict)}")

        # Test if the number of stations is above the threshold
        if len(peak_sta_dict) < min_num_stations:
            print(f"Number of stations below threshold for {component} component. The time window will be skipped.")
                
            return None

        # Compute the Fourier beam image for the component
        bimage_obj = get_fourier_beam_image(freq_peak, peak_sta_dict, xslowax, yslowax, 
                                            min_distance = min_distance, local_max_threshold = local_max_threshold)
        
        bimage_dict[component] = bimage_obj


    # Create a FourierBeamWindow object
    beam_window = FourierBeamWindow(freq_peak, starttime, xslowax, yslowax, bimage_dict, endtime = endtime)

    return beam_window

# # Compute the Fourier beam image for a component in parallel
# def get_fourier_beam_image_in_parallel(freq, peak_sta_dict, xslowax, yslowax,
#                                     min_distance = 1, local_max_threshold = 0.8,
#                                     num_processes = 32):
#     print(f"Computing the Fourier beam images for each component in parallel using {num_processes} processes...")

#     # Create a pool of processes
#     pool = Pool(num_processes)

#     # Compute the center of mass of the stations
#     x0 = 0.0
#     y0 = 0.0
#     for sta_info in peak_sta_dict.keys():
#         x0 += sta_info[0]
#         y0 += sta_info[1]

#     x0 = x0 / len(peak_sta_dict)
#     y0 = y0 / len(peak_sta_dict)

#     # Create the arguments for the parallel computation
#     xslow_mesh, yslow_mesh = meshgrid(xslowax, yslowax)
#     xslows = xslow_mesh.flatten()
#     yslows = yslow_mesh.flatten()
#     args = [(peak_sta_dict, freq, (xslow, yslow), (x0, y0)) for xslow, yslow in zip(xslows, yslows)]

#     # Compute the beam power for each slowness grid
#     powers = pool.starmap(get_fourier_beam_power, args)

#     # Reshape the powers to the beam image
#     bimage = zeros((len(yslowax), len(xslowax)))
#     for flat_ind, power in enumerate(powers):
#         i_yslow, i_xslow = unravel_index(flat_ind, bimage.shape)
#         bimage[i_yslow, i_xslow] = power

#     # Normalize the beam image
#     bimage = bimage / amax(bimage)

#     # Find the slowness of the maximum beam power
#     max_power_idx = bimage.argmax()
#     max_i_yslow, max_i_xslow = unravel_index(max_power_idx, bimage.shape)
#     xslow_global_max = xslowax[max_i_xslow]
#     yslow_global_max = yslowax[max_i_yslow]

#     # Convert the slowness of the global maxima to apparent velocity and azimuth
#     vel_app, azimuth = slowness2velapp(xslow_global_max, yslow_global_max)

#     # Find the local maxima
#     peak_inds = peak_local_max(bimage, min_distance=min_distance, threshold_abs=local_max_threshold)

#     xslows_local_max = xslowax[peak_inds[:, 1]]
#     yslows_local_max = yslowax[peak_inds[:, 0]]

#     # Compute the mean peak power of all stations
#     peak_powers = []
#     peak_proms = []
#     for _, peak_dict in peak_sta_dict.items():
#         peak_power = peak_dict["peak_power"]
#         peak_prom = peak_dict["peak_prom"]

#         peak_powers.append(peak_power)
#         peak_proms.append(peak_prom)

#     mean_peak_power = sum(peak_powers) / len(peak_powers)
#     mean_peak_prom = sum(peak_proms) / len(peak_proms)

#     # Create a FourierBeamImage object
#     stations = [sta_info[2] for sta_info in peak_sta_dict.keys()]
#     bimage_obj = FourierBeamImage(stations, mean_peak_power, mean_peak_prom, bimage, xslow_global_max, yslow_global_max, vel_app, azimuth, xslows_local_max, yslows_local_max)
        
#     return bimage_obj
            
# Compute the Fourier beam image for a component
def get_fourier_beam_image(freq, peak_sta_df, xslowax, yslowax,
                           min_distance = 1, local_max_threshold = 0.8):
    print(f"Computing the beam power for each slowness grid...")

    # Loop over the slowness grid
    bimage = zeros((len(yslowax), len(xslowax)))
    for i, xslow in enumerate(xslowax):
        for j, yslow in enumerate(yslowax):
            slowness = (xslow, yslow)
            bimage[j, i] = get_fourier_beam_power(peak_sta_df, freq, slowness)

    # Find the beam coherence
    coherence = amax(bimage) / len(peak_sta_df) ** 2
    print(f"Beam coherence: {coherence:.2f}")

    # Normalize the beam image
    bimage = bimage / amax(bimage)

    # Find the slowness of the maximum beam power
    max_power_idx = bimage.argmax()
    max_i_yslow, max_i_xslow = unravel_index(max_power_idx, bimage.shape)
    xslow_global_max = xslowax[max_i_xslow]
    yslow_global_max = yslowax[max_i_yslow]

    # Convert the slowness of the global maxima to apparent velocity and azimuth
    vel_app, azimuth = slowness2velapp(xslow_global_max, yslow_global_max)

    # Find the local maxima
    peak_inds = peak_local_max(bimage, min_distance=min_distance, threshold_abs=local_max_threshold)

    xslows_local_max = xslowax[peak_inds[:, 1]]
    yslows_local_max = yslowax[peak_inds[:, 0]]

    # Compute the mean peak power of all stations
    mean_peak_power = peak_sta_df["peak_power"].mean()

    # Create a FourierBeamImage object
    stations = peak_sta_df["station"].tolist()
    bimage_obj = FourierBeamImage(stations, mean_peak_power, coherence, bimage, xslow_global_max, yslow_global_max, vel_app, azimuth, xslows_local_max, yslows_local_max)

    return bimage_obj

# Get the beam power for a specific frequency and slowness
# peak_sta_dict is a dictionary with the station coordinates as keys and the Fourier spectra as values
def get_fourier_beam_power(peak_sta_df, freq, slowness):
    xslow = slowness[0]
    yslow = slowness[1]

    spec_sum = 0
    for _, row in peak_sta_df.iterrows():
        x = row["x"]
        y = row["y"]
        coeff = row["peak_coeff"]

        # Compute the time shift
        time_shift = xslow * x + yslow * y

        # Apply the phase shift
        phase_shift = exp(2j * pi * freq * time_shift)
        spec_shifted = coeff * phase_shift

        spec_sum += spec_shifted

    power = abs(spec_sum) ** 2

    return power


### Synthetic array data generation ###
## Function for generating synthetic array data for testing the beamforming function
def get_synthetic_array_data(xslow, yslow, staccords, source="gauss", numpts = 1001, sampling_rate=1000.0, **kwargs):
    stream = Stream()

    timeax = linspace(-(numpts - 1) / 2 / sampling_rate, (numpts - 1) / 2 / sampling_rate, numpts)
    for station in staccords.keys():
        xsta = staccords[station][0]
        ysta = staccords[station][1]
        time_shift = xslow * xsta + yslow * ysta

        if source == "gauss":
            std = kwargs["std"]
            signal = exp(-0.5 * ((timeax - time_shift) / std)**2)
        elif source == "cosine":
            freq = kwargs["freq"]
            signal = cos(2 * pi * freq * (timeax - time_shift))
        else:
            raise ValueError("Invalid source type!")

        trace = Trace()
        trace.data = signal
        trace.stats.sampling_rate = sampling_rate
        trace.stats.npts = numpts
        trace.stats.station = station
        trace.stats.coordinates = AttribDict({"x":xsta, "y":ysta})
        trace.stats.component = "Z"

        stream.append(trace)

    return stream

## Get the coordinates of the stations in the synthetic array
## Stations can form triangles, squares, or hexagons
def get_synthetic_stations(radius, shape="triangle"):
    if shape == "triangle":
        stacoords = {"STA1":(-radius * sqrt(3) / 2, -radius / 2), "STA2":(radius * sqrt(3) / 2, -radius / 2), "STA3":(0, radius)}
    elif shape == "square":
        stacoords = {"STA1":(-radius, -radius), "STA2":(radius, -radius), "STA3":(radius, radius), "STA4":(-radius, radius)}
    elif shape == "hexagon":
        stacoords = {"STA1":(-radius, 0), "STA2":(-radius / 2, -radius * sqrt(3) / 2), "STA3":(radius / 2, -radius * sqrt(3) / 2), "STA4":(radius, 0), "STA5":(radius / 2, radius * sqrt(3) / 2), "STA6":(-radius / 2, radius * sqrt(3) / 2)}
    else:
        raise ValueError("Invalid shape type!")

    return stacoords

# Read a FourierBeamCollection object from an HDF file
def read_fourier_beam_collection(inpath):
    with File(inpath, "r") as file:
        freq = file.attrs["freq"]
        dur = file.attrs["dur"]
        xslowax = file["x_slow_axis"][:]
        yslowax = file["y_slow_axis"][:]

        beam_windows = []
        for starttime_str in file["start_times"]:
            starttime = Timestamp(file[starttime_str].attrs["starttime"])
            bimage_dict = {}
            for component in GEO_COMPONENTS:
                component_group = file[starttime_str][f"beam_image_{component.lower()}"]

                stations = [station.decode("utf-8") for station in component_group["stations"]]
                bimage = component_group["beam_image"][:]
                xslows_local_max = component_group["x_slows_local_max"][:]
                yslows_local_max = component_group["y_slows_local_max"][:]

                mean_peak_power = component_group.attrs["mean_peak_power"]
                coherence = component_group.attrs["coherence"]
                xslow_global_max = component_group.attrs["x_slow_global_max"]
                yslow_global_max = component_group.attrs["y_slow_global_max"]
                vel_app = component_group.attrs["vel_app"]
                azimuth = component_group.attrs["azimuth"]

                bimage_obj = FourierBeamImage(stations, mean_peak_power, coherence, bimage, xslow_global_max, yslow_global_max, vel_app, azimuth, xslows_local_max, yslows_local_max)
                bimage_dict[component] = bimage_obj

            beam_window = FourierBeamWindow(freq, starttime, xslowax, yslowax, bimage_dict, dur = dur)
            beam_windows.append(beam_window)

    beam_collection = FourierBeamCollection(beam_windows)

    return beam_collection

### Utility functions ###
# Get the x and y slownesses of the global maximum of the beam image
def get_global_maxima(xslowax, yslowax, bimage):
    i_yslow_max, i_xslow_max = unravel_index(bimage.argmax(), bimage.shape)
    xslow_global_max = xslowax[i_xslow_max]
    yslow_global_max = yslowax[i_yslow_max]

    return xslow_global_max, yslow_global_max

# Convert x and y slownesses to apparent velocity and azimuth
def slowness2velapp(xslow, yslow):
    slow = sqrt(xslow**2 + yslow**2)

    if slow == 0:
        return inf, 0
    
    velapp = 1 / slow
    azimuth = 180 / pi * arctan2(xslow, yslow)

    return velapp, azimuth

# Get the slowness axes for beamforming
def get_slowness_axes(min_vel_app = 500.0, num_vel_app = 51):
    max_slow = 1 / min_vel_app

    xslowax = linspace(-max_slow, max_slow, num_vel_app)
    yslowax = linspace(-max_slow, max_slow, num_vel_app)

    return xslowax, yslowax

# Select the stations for beamforming
def select_stations(array_selection, array_aperture):
    if array_selection == "A":
        if array_aperture == "small":
            stations = inner_stations_a
        elif array_aperture == "medium":
            stations = inner_stations_a + middle_stations_a
        elif array_aperture == "large":
            stations = inner_stations_a + middle_stations_a + outer_stations_a
        else:
            raise ValueError("Invalid array aperture.")
    elif array_selection == "B":
        if array_aperture == "small":
            stations = inner_stations_b
        elif array_aperture == "medium":
            stations = inner_stations_b + middle_stations_b
        elif array_aperture == "large":
            stations = inner_stations_b + middle_stations_b + outer_stations_b
        else:
            raise ValueError("Invalid array aperture.")
    else:
        raise ValueError("Invalid array selection.")

    if array_aperture == "small":
        min_num_stations = 5
    elif array_aperture == "medium":
        min_num_stations = 10
    elif array_aperture == "large":
        min_num_stations = 15

    return stations, min_num_stations