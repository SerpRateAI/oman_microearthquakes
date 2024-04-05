# Functions and classes for array processing
from obspy import Stream, Trace
from obspy.core.util import AttribDict
from numpy import amax, cos,  exp, linspace, pi, sqrt, zeros
from numpy.linalg import norm

from utils_basic import GEO_COMPONENTS

## Function for computing the images of a seismic array
def get_beam_images(stream_in, minslow=-0.008, maxslow=0.008, numslow=61, components=None):
    ### Check if the input is a Stream object
    if not isinstance(stream_in, Stream):
        raise TypeError("Input must be a Stream object!")
    
    ### Check if the stations coordinates are set in the in put stream_in
    for trace in stream_in:
        if not hasattr(trace.stats, "coordinates"):
            raise ValueError("Station coordinates are not set in the Stream object!")
        
    ### Check if the traces have the same length
    if len(set([len(trace.data) for trace in stream_in])) > 1:
        raise ValueError("Traces must have the same length!")
        
    ### Get the components to work with
    if components is None:
        components = list(set([trace.stats.component for trace in stream_in]))
    else:
        if not isinstance(components, list):
            if isinstance(components, str):
                components = [components]
            else:
                raise TypeError("Invalid components format!")

    ### Get the station coordinates and the reference coordinate
    stacoords = {}
    coords = zeros(2)
    numsta = 0
    for trace in stream_in:
        station = trace.stats.station
        if station not in stacoords:
            stacoords[station] = trace.stats.coordinates
            coords[0] += trace.stats.coordinates.x
            coords[1] += trace.stats.coordinates.y
            numsta += 1

    xref = coords[0] / numsta
    yref = coords[1] / numsta
    refcoord = (xref, yref)

    ### Perform beamforming for each component
    beamdict = {}
    for component in components:
        stream = stream_in.select(component=component)
        if len(stream) < 3:
            raise ValueError("At least 3 stations are required for beamforming!")
        
        #### Define the slowness grid
        bimage = zeros((numslow, numslow))
        xslow = linspace(minslow, maxslow, numslow)
        yslow = linspace(minslow, maxslow, numslow)

        print(f"Beamforming {component} component...")

        #### Loop over the slowness grid
        for i, x in enumerate(xslow):
            for j, y in enumerate(yslow):
                slowness = (x, y)
                bimage[j, i] = get_beam_power(stream, slowness, refcoord)

        #### Normalize the beam image
        bimage = bimage / amax(bimage)

        beamdict[component] = bimage

    return xslow, yslow, beamdict

## Function for computing the beam power for a given slowness
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
        x = stacoords[stacoords["name"] == station]["east"].values[0]
        y = stacoords[stacoords["name"] == station]["north"].values[0]

        trace.stats.coordinates = AttribDict({"x":x, "y":y})

    return stream

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