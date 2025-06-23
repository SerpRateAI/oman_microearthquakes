from h5py import File
from numpy import ndarray
from pandas import Timestamp
from obspy import Trace
from typing import Dict, Union

from utils_basic import GEO_CHANNELS as channels, geo_component2channel

# -----------------------------------------------------------------------------
# Class to store the day-long waveform
# -----------------------------------------------------------------------------
class DayLongWaveform:
    """
    Container for three-component day-long waveform of a station on a specific date.

    Attributes
    ----------
    station : str
        Station code.
    date : pandas.Timestamp
        Date of the waveform (midnight UTC of that day).
    waveform : Dict[str, numpy.ndarray]
        Mapping from component (e.g., 'EHZ', 'EH1', 'EH2') to data array.
    sampling_rate : float
        Sampling rate in Hz.
    num_pts : int
        Number of samples per component.
    starttime : pandas.Timestamp
        Timestamp of the first sample.
    """
    def __init__(self,
                 station: str,
                 date: Timestamp,
                 waveform: Dict[str, ndarray],
                 sampling_rate: float,
                 num_pts: int,
                 starttime: Timestamp):
        self.station = station
        self.date = date
        self.waveform = waveform
        self.sampling_rate = sampling_rate
        self.num_pts = num_pts
        self.starttime = starttime

    def get_component(self, component: str) -> ndarray:
        return self.waveform[geo_component2channel(component)]

# -----------------------------------------------------------------------------
# Save a day-long trace to HDF5
# -----------------------------------------------------------------------------
def save_day_long_trace_to_hdf(trace: Trace,
                               hdf5_path: str,
                               overwrite: bool = False) -> None:
    """
    Save an ObsPy Trace object (day-long time series of one component) to an HDF5 file.

    The HDF5 hierarchy will be structured as:
        /<station>/<YYYY-MM-DD>/<component>

    Each dataset stores the raw trace data and relevant metadata as attributes.

    Parameters
    ----------
    trace : obspy.Trace
        Trace object containing the seismic data for one component.
    hdf5_path : str
        Path to the HDF5 file to write (created if it does not exist).
    overwrite : bool, optional
        If True, overwrite existing dataset for the same channel. If False and
        the channel dataset exists, raise a ValueError. Defaults to False.
    """
    station = trace.stats.station
    # Convert ObsPy UTCDateTime to pandas Timestamp
    starttime_ts = Timestamp(trace.stats.starttime.isoformat())
    date_str = starttime_ts.strftime("%Y-%m-%d")
    channel = trace.stats.channel

    with File(hdf5_path, 'a') as h5f:
        station_grp = h5f.require_group(station)
        day_grp = station_grp.require_group(date_str)
        if channel in day_grp:
            if overwrite:
                del day_grp[channel]
            else:
                raise ValueError(f"Dataset for channel '{channel}' already exists for {station} on {date_str}."
                                 " Use overwrite=True to replace it.")
        # Create dataset without compression
        ds = day_grp.create_dataset(
            name=channel,
            data=trace.data,
            dtype=trace.data.dtype
        )
        # Store metadata: nanoseconds since epoch
        ds.attrs['starttime_ns'] = starttime_ts.value
        ds.attrs['sampling_rate'] = trace.stats.sampling_rate
        ds.attrs['num_pts'] = trace.stats.npts

# -----------------------------------------------------------------------------
# Load a day-long waveform from HDF5
# -----------------------------------------------------------------------------
def load_day_long_waveform_from_hdf(hdf5_path: str,
                                    station: str,
                                    date: Union[str, Timestamp]) -> DayLongWaveform:
    """
    Load a DayLongWaveform from an HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file.
    station : str
        Station code.
    date : str or pandas.Timestamp
        Date to load ("YYYY-MM-DD" or pandas.Timestamp).

    Returns
    -------
    DayLongWaveform
        Instance containing three components of the day-long waveform.

    Raises
    ------
    KeyError
        If the station or date group is not found in the file.
    """
    # Normalize date to pandas.Timestamp at UTC midnight
    if isinstance(date, str):
        date_ts = Timestamp(date)
    else:
        date_ts = date.normalize()
    date_str = date_ts.strftime("%Y-%m-%d")

    with File(hdf5_path, 'r') as h5f:
        if station not in h5f:
            raise KeyError(f"No data found for station {station}!")
        
        station_grp = h5f[station]
        if date_str not in station_grp:
            raise KeyError(f"No data found for station {station} on {date_str}!")

        day_grp = station_grp[date_str]

        waveform: Dict[str, ndarray] = {}
        sampling_rate = None
        num_pts = None
        starttime_ns = None
        for comp in day_grp:
            ds = day_grp[comp]
            waveform[comp] = ds[()]
            if sampling_rate is None:
                sampling_rate = ds.attrs['sampling_rate']
                num_pts = ds.attrs['num_pts']
                # Read nanoseconds since epoch and convert to pandas.Timestamp
                starttime_ns = int(ds.attrs['starttime_ns'])

    starttime_ts = Timestamp(starttime_ns)
    return DayLongWaveform(
        station=station,
        date=date_ts,
        waveform=waveform,
        sampling_rate=float(sampling_rate),
        num_pts=int(num_pts),
        starttime=starttime_ts
    )


# -----------------------------------------------------------------------------
# Load a slice of the day-long waveform
# -----------------------------------------------------------------------------
def load_waveform_slice(hdf5_path: str,
                        station: str,
                        start_time: Union[str, Timestamp],
                        end_time: Union[str, Timestamp]) -> Dict[str, ndarray]:
    """
    Load a slice of the three-component day-long waveform from HDF5.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file.
    station : str
        Station code.
    start_time : str or pandas.Timestamp
        Start of slice (ISO format or pandas.Timestamp).
    end_time : str or pandas.Timestamp
        End of slice (ISO format or pandas.Timestamp).

    Returns
    -------
    waveform_dict : Dict[str, numpy.ndarray]
        Dictionary with keys for each component, containing the sliced data array.
    """
    # Normalize slice times
    start_ts = Timestamp(start_time)
    end_ts = Timestamp(end_time)

    # Get the date string
    date_str = start_ts.strftime("%Y-%m-%d")

    with File(hdf5_path, 'r') as h5f:
        day_grp = h5f[station][date_str]
        # Read metadata from any component
        ds0 = next(iter(day_grp.values()))
        # Nanoseconds since epoch
        orig_start_ns = int(ds0.attrs['starttime_ns'])
        sampling_rate = ds0.attrs['sampling_rate']
        # Convert orig_start to seconds
        orig_start_sec = orig_start_ns / 1e9
        # Compute sample indices
        idx0 = int((start_ts.timestamp() - orig_start_sec) * sampling_rate)
        idx1 = int((end_ts.timestamp() - orig_start_sec) * sampling_rate)
        # Clip to valid range
        idx0 = max(0, idx0)
        idx1 = min(ds0.attrs['num_pts'], idx1)

        waveform_dict: Dict[str, ndarray] = {}
        for comp, ds in day_grp.items():
            waveform_dict[comp] = ds[idx0:idx1]

    return waveform_dict

# Example:
# slice_dict = load_waveform_slice('data.h5', 'ABC', '2025-06-12', '2025-06-12T01:00:00', '2025-06-12T02:00:00')
