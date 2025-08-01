# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Iterator, Tuple
from collections import defaultdict

from numpy import empty, empty_like, ndarray, asarray, float32, arange, amax
from pandas import Timestamp, Timedelta, Series, DataFrame, date_range, cut
from h5py import File
from matplotlib.pyplot import subplots, Figure, Axes

from utils_basic import (
    STARTTIME_GEO as starttime_geo,
    ENDTIME_GEO as endtime_geo,
    SAMPLING_RATE as sampling_rate,  # kept for future use
    GEO_COMPONENTS as components,
    GEO_CHANNELS as channels,
    geo_channel2component
)

from utils_plot import component2label, get_geo_component_color, format_norm_time_lag_xlabels

# -----------------------------------------------------------------------------
# Data‑holding structures (brought inline from utils_cc.py)
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class Template:
    """Three‑component waveform template and accompanying metadata."""

    id: str
    station: str  # network/station code
    starttime: Timestamp
    num_pts: int
    waveform: Dict[str, ndarray[float32]]

    # ---------------------------------------------------------------------
    # Validation & dunder helpers
    # ---------------------------------------------------------------------

    def __post_init__(self) -> None:
        required = set(components)
        if set(self.waveform) != required:
            raise ValueError(f"`waveform` must have keys {required}.")
        for comp, data in self.waveform.items():
            if len(data) != self.num_pts:
                raise ValueError(
                    f"Component {comp!r} length {len(data)} ≠ num_pts {self.num_pts}"
                )

    # readable representation -------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover
        return f"{self.id} – {self.station} – {self.starttime} – {self.num_pts} pts"

    __repr__ = __str__


@dataclass(slots=True)
class Match:
    """Snippet from the data stream that matches the template."""

    starttime: Timestamp
    coeff: float32
    waveform: Dict[str, ndarray[float32]]

    def __post_init__(self) -> None:
        required = set(components)
        if set(self.waveform) != required:
            raise ValueError(f"`waveform` must have keys {required}.")
        lengths = {c: len(a) for c, a in self.waveform.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "All component arrays in `waveform` must share the same length; "
                f"got {lengths}."
            )


class TemplateMatches:
    """Tie a :class:`Template` to all of its :class:`Match` objects."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, template: Template, matches: Iterable[Match] | None = None):
        self.template: Template = template
        self.matches: List[Match] = list(matches) if matches is not None else []

    # ------------------------------------------------------------------
    # Dunder + convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # pragma: no cover
        return len(self.matches)

    def __iter__(self) -> Iterator[Match]:  # pragma: no cover
        return iter(self.matches)

    def __getitem__(self, idx: int) -> Match:  # pragma: no cover
        return self.matches[idx]

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def add_match(self, match: Match) -> None:
        """Append a single :class:`Match`."""
        self.matches.append(match)

    def add_matches(self, matches: Iterable[Match]) -> None:
        """Append many matches in bulk."""
        self.matches.extend(matches)

    def best_match(self) -> Match | None:
        """Return the match with the highest correlation coefficient."""
        return max(self.matches, key=lambda m: m.coeff, default=None)

    def get_match_starttimes(self) -> List[Timestamp]:
        return [m.starttime for m in self.matches]

    # ------------------------------------------------------------------
    # Simple hourly histogram
    # ------------------------------------------------------------------

    def bin_matches_by_hour(
        self,
        starttime_geo: Timestamp = starttime_geo,
        endtime_geo: Timestamp = endtime_geo,
    ) -> DataFrame:
        """Return hourly counts of matches between *starttime_geo* and *endtime_geo*."""
        starttimes = Series(self.get_match_starttimes())
        bin_edges = date_range(starttime_geo, endtime_geo, freq="1h")
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        starttimes_binned = cut(starttimes, bin_edges, labels=bin_centers)
        bin_counts = (
            starttimes_binned.value_counts(sort=False)
            .reindex(bin_centers, fill_value=1)  # avoid log‑scale zeros
        )
        return DataFrame({"bin_center": bin_centers, "bin_count": bin_counts})

    # ------------------------------------------------------------------
    # HDF5 I/O (hierarchy: /<template_id>/<station>/…)
    # ------------------------------------------------------------------

    def to_hdf(
        self,
        path: str | Path,
        *,
        compression: str | None = "gzip",
        overwrite: bool = False,
    ) -> None:
        """Write or append this object to *path* using the layout
        ``/<template_id>/<station>/template`` and ``…/matches``.
        """

        if not self.matches:
            raise ValueError("No matches available to write.")

        path = Path(path)
        n_matches = len(self.matches)
        num_pts = self.template.num_pts
        dtype = self.matches[0].waveform[components[0]].dtype

        # Collect match arrays -------------------------------------------------
        comp_z = empty((n_matches, num_pts), dtype=dtype)
        comp_1 = empty_like(comp_z)
        comp_2 = empty_like(comp_z)
        starttimes = empty(n_matches, dtype="int64")  # pandas nanoseconds
        coeffs = empty(n_matches, dtype="float32")

        for i, m in enumerate(self.matches):
            comp_z[i] = m.waveform[components[0]]
            comp_1[i] = m.waveform[components[1]]
            comp_2[i] = m.waveform[components[2]]
            starttimes[i] = m.starttime.value
            coeffs[i] = m.coeff

        # Write to disk --------------------------------------------------------
        with File(path, "a") as f:
            tpl_grp = f.require_group(self.template.id)
            if self.template.station in tpl_grp:
                if not overwrite:
                    raise ValueError(
                        f"Station {self.template.station!r} already exists under template "
                        f"{self.template.id!r}. Use overwrite=True to replace it."
                    )
                del tpl_grp[self.template.station]
            st_grp = tpl_grp.create_group(self.template.station)

            # --- template metadata & waveform
            gt = st_grp.create_group("template")
            gt.attrs.update(
                {
                    "id": self.template.id,
                    "station": self.template.station,
                    "starttime": self.template.starttime.value,
                    "num_pts": self.template.num_pts,
                }
            )
            for i, component in enumerate(components):
                gt.create_dataset(
                    f"{component}",
                    data=self.template.waveform[component],
                    compression=compression,
                )

            # --- matches
            gm = st_grp.create_group("matches")
            gm.create_dataset("starttime", data=starttimes)
            gm.create_dataset("coeff", data=coeffs)
            for i, component in enumerate(components):
                gm.create_dataset(component, data=comp_z, compression=compression)

    # ------------------------------------------------------------------
    # Reading back
    # ------------------------------------------------------------------

    @classmethod
    def from_hdf(
        cls,
        path: str | Path,
        *,
        id: str | None = None,  # template id
        station: str | None = None,
    ) -> "TemplateMatches":
        """Read one ``TemplateMatches`` from *path*.

        Parameters
        ----------
        id, station
            * If both provided ⇒ single object
            * If only ``id`` ⇒ all stations for that template
        """

        path = Path(path)

        with File(path, "r") as f:
            tpl_grp = f[id]
            stations = [station] if station else list(tpl_grp.keys())
            for st in stations:
                    st_grp = tpl_grp[st]
                    # -------- template part
                    gt = st_grp["template"]
                    tpl_waveforms = {
                        component: asarray(gt[component]) for component in components
                    }
                    tpl = Template(
                        id=str(gt.attrs["id"]),
                        station=str(gt.attrs["station"]),
                        starttime=Timestamp(int(gt.attrs["starttime"]), unit="ns", tz="UTC"),
                        num_pts=int(gt.attrs["num_pts"]),
                        waveform=tpl_waveforms,
                    )

                    # -------- matches part
                    gm = st_grp["matches"]
                    sts = gm["starttime"][...]
                    cs = gm["coeff"][...]
                    for i, component in enumerate(components):
                        cz = gm[component]

                    match_list: List[Match] = []
                    for i in range(sts.shape[0]):
                        match_list.append(
                            Match(
                                starttime=Timestamp(sts[i], unit="ns", tz="UTC"),
                                coeff=cs[i],
                                waveform={
                                    component: asarray(cz[i]) for component in components
                                },
                            )
                        )

            return cls(tpl, match_list)

    
# @dataclass(slots=True)
# class MatchedEvent:
#     """A matched event with its origin time and matched stations."""

#     first_match_time: Timestamp
#     first_match_station: str
#     num_sta: int
#     station: List[str]
#     match_time: List[Timestamp]
#     normalized_time_lag: List[float]

#     def __post_init__(self) -> None:
#         if len(self.station) != len(self.match_time):
#             raise ValueError("The number of stations and match times must be the same.")
#         if len(self.station) != len(self.normalized_time_lag):
#             raise ValueError("The number of stations and normalized time lags must be the same.")
        
# -----------------------------------------------------------------------------
# Associate matched events
# -----------------------------------------------------------------------------

def associate_matched_events(
    tm_dict: Dict[str, "TemplateMatches"],
    min_num_sta: int,
    window_length: float = 0.1,
) -> DataFrame:
    """Identify coincidence events across multiple stations.

    An event is declared when picks from *min_num_sta* distinct stations fall
    within a sliding window of length *window_length* seconds.  All picks that
    participate in that satisfied window are grouped into the same event.
    The search continues until the window can no longer satisfy the criterion,
    ensuring no qualifying pick is missed.
    """

    # ---------------------------------------------------------------------
    # Prepare the picks table
    # ---------------------------------------------------------------------
    window_length = Timedelta(seconds=window_length)

    match_time_dicts = []
    for station, tm in tm_dict.items():
        for starttime in tm.get_match_starttimes():
            match_time_dicts.append({"station": station, "time": starttime})

    match_time_df = (
        DataFrame(match_time_dicts).sort_values("time").reset_index(drop=True)
    )

    first_template_starttime = min(
        tm.template.starttime for tm in tm_dict.values()
    )

    # ---------------------------------------------------------------------
    # Sliding‑window coincidence detection
    # ---------------------------------------------------------------------
    active_counts: dict[str, int] = defaultdict(int)  # active picks per station
    unique_stations = 0                              # # distinct stations now
    i_left = 0                                       # left edge of window

    event_open = False        # True while window satisfies coincidence rule
    event_start_idx = None    # fixed left index of that open window
    last_good_right = None    # last i_right that still satisfied rule

    event_dicts = []

    for i_right, row in match_time_df.iterrows():
        # ----------------------------------------------------------
        # 1. expand: add the new pick at i_right
        # ----------------------------------------------------------
        sta = row["station"]
        if active_counts[sta] == 0:
            unique_stations += 1
        active_counts[sta] += 1

        # ----------------------------------------------------------
        # 2. shrink: move i_left to keep window <= window_length
        # ----------------------------------------------------------
        while (
            match_time_df.loc[i_right, "time"]
            - match_time_df.loc[i_left, "time"]
            > window_length
        ):
            sta_left = match_time_df.loc[i_left, "station"]
            active_counts[sta_left] -= 1
            if active_counts[sta_left] == 0:
                unique_stations -= 1
            i_left += 1

        # ----------------------------------------------------------
        # 3. coincidence logic
        # ----------------------------------------------------------
        if unique_stations >= min_num_sta:
            # window currently satisfies the rule → keep it open / extend
            if not event_open:
                event_open = True
                event_start_idx = i_left  # freeze left edge the first time
            last_good_right = i_right      # extend right edge each iteration
        else:
            # coincidence just broke → close previous event if we had one
            if event_open:
                window = match_time_df.iloc[event_start_idx : last_good_right + 1]
                time_event = window.time.min()
                stations_event = window.station.to_list()

                event_dicts.append(
                    {
                        "first_match_time": time_event,
                        "num_sta": len(set(stations_event)),
                        "station_time": dict(
                            zip(stations_event, window.time.to_list())
                        ),
                    }
                )

                # reset bookkeeping; *re‑include* current pick as fresh start
                event_open = False
                active_counts.clear()
                unique_stations = 0
                i_left = i_right

                # Re‑initialise counts for the current pick
                sta_curr = match_time_df.loc[i_right, "station"]
                active_counts[sta_curr] = 1
                unique_stations = 1

    # --------------------------------------------------------------
    # 4. clean‑up: capture an event that extends to the last pick
    # --------------------------------------------------------------
    if event_open:
        window = match_time_df.iloc[event_start_idx : last_good_right + 1]
        time_event = window.time.min()
        stations_event = window.station.to_list()

        event_dicts.append(
            {
                "first_match_time": time_event,
                "num_sta": len(set(stations_event)),
                "station_time": dict(zip(stations_event, window.time.to_list())),
            }
        )

    # ---------------------------------------------------------------------
    # Build the hierarchical record DataFrame (unchanged logic)
    # ---------------------------------------------------------------------
    record_dicts = []
    for evt in event_dicts:
        first_match_time = evt["first_match_time"]
        num_sta = evt["num_sta"]

        for station, match_time in evt["station_time"].items():
            self_match = abs(
                (first_match_time - first_template_starttime).total_seconds()
            ) < 1 / sampling_rate

            num_pts = tm_dict[station].template.num_pts

            record_dicts.append(
                {
                    "first_match_time": first_match_time,
                    "num_sta": num_sta,
                    "station": station,
                    "num_pts": num_pts,
                    "match_time": match_time,
                    "self_match": self_match,
                }
            )

    record_df = DataFrame(record_dicts)
    record_df.set_index(["first_match_time", "station"], inplace=True)
    record_df.sort_index(inplace=True)

    print(
        f"Number of associated events: {len(record_df.index.unique(level=0))}"
    )

    return record_df

def get_normalized_time_lags(tpl_dict: Dict[str, "Template"],
                             record_df: DataFrame) -> DataFrame:
    """Return a *new* DataFrame with an extra ``lag_time_norm`` column.

    Parameters
    ----------
    tpl_dict : Dict[str, Template]
        Mapping ``station → Template``.  Each Template must expose a
        ``.starttime`` attribute (pandas.Timestamp).
    record_df : DataFrame
        Hierarchical DataFrame indexed by (origin, station) that *must*
        include a ``match_time`` column, holding the detection time for each
        station.

    Returns
    -------
    DataFrame
        A *copy* of ``record_df`` with a float column ``lag_time_norm`` where, for
        every event, the raw lag (match_time − template start) is centred by
        subtracting that event’s mean lag.
    """

    if "match_time" not in record_df.columns:
        raise KeyError("record_df must contain a 'match_time' column")

    # for station in tpl_dict.keys():
    #     print(station)
    #     print(tpl_dict[station].template.starttime)

    print(record_df.head())

    # ------------------------------------------------------------------
    # 1. Compute raw lag per row → seconds (float)
    # ------------------------------------------------------------------
    lag_raw = (
        record_df
        .reset_index()
        .apply(lambda row: (row["match_time"] -
                            tpl_dict[row["station"]].template.starttime).total_seconds(),
               axis=1)
    )

    # Attach raw lag to a *copy* so we don’t mutate caller’s DataFrame
    out_df = record_df.copy()
    out_df["lag_time_raw"] = lag_raw.values
    # print(out_df.head())

    # print(out_df)

    # ------------------------------------------------------------------
    # 2. Normalise inside each event (first‑level index)
    # ------------------------------------------------------------------
    out_df["lag_time_norm"] = (
        out_df.groupby(level=0)["lag_time_raw"].transform(lambda x: x - x.mean())
    )

    # Optionally drop the helper column
    out_df = out_df.drop(columns="lag_time_raw")

    return out_df

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_all_stations_template_waveforms(template_dict: Dict[str, Template], scale: float = 0.3) -> Tuple[Figure, Axes]:
    fig, axs = subplots(1, 3, figsize = (15, 10))

    # Get the min time of all templates
    min_time = min([template.starttime for template in template_dict.values()])
    max_time = 0.0
    for i, station in enumerate(template_dict.keys()):
        template = template_dict[station]

        # Normalize the waveforms by the maximum amplitude of the three components
        max_amp = max(max(template.waveform[component]) for component in components)

        # Get the time axis
        num_pts = template.num_pts
        starttime = template.starttime
        timeax = arange(num_pts) / sampling_rate + (starttime - min_time).total_seconds()
        max_time = max(amax(timeax), max_time)

        for j, component in enumerate(components):
            waveform = template.waveform[component]
            axs[j].plot(timeax, waveform / max_amp * scale + i + 1, color = get_geo_component_color(component))
            axs[j].set_title(f"{component2label(component)}", fontsize = 12, fontweight = "bold")
            axs[j].set_xlabel("Time (s)", fontsize = 12)

            if j == 0:
                axs[j].text(0.0, i + 1.3, f"{station}", fontsize = 12, fontweight = "bold", ha = "left", va = "bottom")

    # Set the x axis limits
    axs[0].set_xlim(0, max_time)
    axs[1].set_xlim(0, max_time)
    axs[2].set_xlim(0, max_time)

    return fig, axs
    


def plot_station_lag_time_histogram(
        ax: Axes,
        record_df: DataFrame,
        station: str,
        min_lag: float = -5e-3,
        max_lag: float = 5e-3,
        bin_width: float = 1e-3,
        color: str = "tab:cyan",
        linewidth: float = 1.0,
) -> Axes:
    """Plot a histogram of *normalised* time lags for one station.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Existing axes to draw the histogram on.
    record_df : pandas.DataFrame
        DataFrame produced by :func:`get_normalized_time_lags`, containing a
        ``lag_norm`` column and indexed by (origin, station).
    station : str
        Station code whose lags to plot.
    min_lag, max_lag : float, optional
        Lower and upper limits (seconds) of the histogram range.  Defaults to
        ±5 ms.
    bin_width : float, optional
        Width of each histogram bin in seconds.  Default is 1 ms.

    Returns
    -------
    matplotlib.axes.Axes
        The same axes, now populated with the histogram.
    """

    if "lag_time_norm" not in record_df.columns:
        raise KeyError("record_df must contain a 'lag_time_norm' column – run get_normalized_time_lags() first")

    # Extract rows for this station -----------------------------------
    try:
        station_df = record_df.xs(station, level="station")
    except KeyError as exc:
        raise KeyError(f"Station {station!r} not found in record_df index") from exc

    lags = station_df["lag_time_norm"].values

    if lags.size == 0:
        raise ValueError(f"No lag data available for station {station!r}")

    bin_edges = arange(min_lag, max_lag + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + bin_width / 2

    ax.hist(lags, bins=bin_centers, color=color, linewidth=linewidth, edgecolor="black")
    ax.set_xlim(min_lag, max_lag)
    
    format_norm_time_lag_xlabels(ax)
    ax.set_ylabel("Count")
    ax.set_title(f"{station}", fontsize=12, fontweight="bold")

    return ax


#------------------------------------------------------------------------------
# 
#------------------------------------------------------------------------------
# Class for storing the template waveforms for a station and component
# ## Class for storing the information of one template, the cc parameters, and all its matched events
# class TemplateMatches:
#     def __init__(self, freqmin, freqmax, mincc, numdet_min, template, matches):
#         self.low_freq = freqmin
#         self.high_freq = freqmax
#         self.min_cc = mincc
#         self.min_num_of_detections = numdet_min

#         if isinstance(template, TemplateEvent):
#             self.template = template
#         else:
#             raise ValueError("Template must be a TemplateEvent object!")
        
#         if isinstance(matches, Matches):
#             self.matches = matches
#         else:
#             raise ValueError("Matches must be a Matches object!")

#         self.num_of_matches = len(matches)

#     def __str__(self):
#         timestr = self.template.first_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
#         return f"{self.template.name}, {timestr}, {self.template.num_of_stations} stations, {len(self.matches)} matches"
    
#     def __repr__(self):
#         return self.__str__()
    
#     ### Save the information to a file
#     def write_to_file(self, outpath):
#         with open(outpath, 'w') as fp:

#             ## Save the template and header information
#             template = self.template
#             timestr = template.first_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

#             fp.write("#\n")
#             fp.write(f"{self.template}\n")
#             fp.write("\n")
#             fp.write("low_freq high_freq min_cc min_num_of_detections\n")

#             if self.high_freq is None:
#                 fp.write(f"{self.low_freq:.1f} None {self.min_cc:.2f} {self.min_num_of_detections:d}\n")
#             else:
#                 fp.write(f"{self.low_freq:.1f} {self.high_freq:.1f} {self.min_cc:.2f} {self.min_num_of_detections:d}\n")

#             fp.write("\n")
#             fp.write("first_start_time duration num_of_stations num_of_matches\n")
#             fp.write(f"{timestr} {template.duration:.3f} {template.num_of_stations:d} {self.num_of_matches:d}\n")
#             fp.write("\n")

#             fp.write("station start_time\n")
#             for i in range(template.num_of_stations):
#                 station = template.stations[i]
#                 starttime = template.start_times[i]
#                 timestr = starttime.strftime("%Y-%m-%dT%H:%M:%S.%f")

#                 fp.write(f"{station} {timestr}\n")

#             fp.write("\n")

#             ## Save the associated matches to a file
#             matches = self.matches
#             matches.append_to_file(fp)

#             print(f"Template and match information is saved to {outpath}")

# ## Class for storing the information of a template event
# class TemplateEvent:
#     def __init__(self, tempname, dur, stations, starttimes):
#         self.name = tempname
#         self.duration = dur
#         self.stations = stations
#         self.start_times = starttimes

#         self.num_of_stations = len(stations)
#         self.first_start_time = amin(starttimes)

#     def __str__(self):
#         timestr = self.first_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
#         return f"{self.name}, {timestr}, {self.num_of_stations} stations"
    
#     def __repr__(self):
#         return self.__str__()
    
#     def get_start_time_for_station(self, station):
        
#         if station not in self.stations:
#             print(f"Warning: {station} is not in the list of stations!")
#             return None

#         i = self.stations.index(station)
#         starttime = self.start_times[i]
                
#         return starttime
    
# ## Class for storing all matched events
# class Matches:
#     def __init__(self):
#         self.events = []

#     def __len__(self):
#         return len(self.events)

#     def __getitem__(self, index):
#         return self.events[index]

#     def __setitem__(self, index, event):
#         if isinstance(event, MatchedEvent):
#             self.events[index] = event
#         else:
#             raise ValueError("Event must be a MatchedEvent object.")

#     def __delitem__(self, index):
#         del self.events[index]

#     def __iter__(self):
#         return iter(self.events)

#     def __str__(self):
#         return str(self.events)

#     def __repr__(self):
#         return repr(self.events)

#     def append(self, event):
#         if isinstance(event, MatchedEvent):
#             self.events.append(event)
#         else:
#             raise ValueError("Event must be a MatchedEvent object.")
        
#     def extend(self, events):
#         if isinstance(events, Matches):
#             self.events.extend(events.events)
#         elif isinstance(events, list):
#             for event in events:
#                 if isinstance(event, MatchedEvent):
#                     self.events.append(event)
#                 else:
#                     raise ValueError("Event must be a MatchedEvent object.")
#         else:
#             raise ValueError("Events must be a Matches or list object.")

#     def insert(self, index, event):
#         if isinstance(event, MatchedEvent):
#             self.events.insert(index, event)
#         else:
#             raise ValueError("Event must be a MatchedEvent object.")

#     def remove(self, event):
#         if isinstance(event, MatchedEvent):
#             self.events.remove(event)
#         else:
#             raise ValueError("Event must be a MatchedEvent object.")

#     def clear(self):
#         self.events.clear()

#     def sort(self, key=None, reverse=False):
#         self.events.sort(key=key, reverse=reverse)

#     def reverse(self):
#         self.events.reverse()

#     def get_match_times(self):
#         times = []
#         for event in self.events:
#             times.append(event.first_start_time)
#         return times
    
#     def get_matches_by_names(self, names):
#         matches = Matches()
#         for event in self.events:
#             if event.name in names:
#                 matches.append(event)
#         return matches
    
#     def append_to_file(self, fp):
#         for event in self.events:
#             event.append_to_file(fp)

#     def get_matches_by_criteria(self, avgcc_min, numsta_min):
#         matches = Matches()
#         for event in self.events:
#             if event.average_cc >= avgcc_min and event.num_of_stations >= numsta_min:
#                 matches.append(event)
#         return matches

# ## Class for storing the cross-correlation result between the template and one matched event
# class MatchedEvent:
#     def __init__(self, matchname, stations, ccvals, starttimes, tshifts, amprats_z, amprats_1, amprats_2):
#         self.name = matchname
#         self.stations = stations
#         self.cc_values = ccvals
#         self.start_times = starttimes
#         self.time_shifts = tshifts
#         self.amplitude_ratios_z = amprats_z
#         self.amplitude_ratios_1 = amprats_1
#         self.amplitude_ratios_2 = amprats_2

#         self.num_of_stations = len(stations)
#         self.average_cc = mean(ccvals)
#         self.first_start_time = amin(starttimes)
#         self.average_amp_rat_z = mean(amprats_z)
#         self.average_amp_rat_1 = mean(amprats_1)
#         self.average_amp_rat_2 = mean(amprats_2)
    

#     def __str__(self):
#         timestr = self.first_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
#         return f"{self.name}, {timestr}, {self.num_of_stations} stations, average cc: {self.average_cc}, average amplitude ratio: {self.average_amp_rat_z}, {self.average_amp_rat_1}, {self.average_amp_rat_2}"
        
#     def __repr__(self):
#         return self.__str__()
    
#     ### Append the information to an already opened file
#     def append_to_file(self, fp):
#         timestr = self.first_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

#         fp.write("##\n")
#         fp.write(f"{self.name}\n")
#         fp.write("\n")
#         fp.write(f"first_start_time num_of_stations average_cc average_amplitude_ratio_z average_amplitude_ratio_1 average_amplitude_ratio_2\n")
#         fp.write(f"{timestr} {self.num_of_stations:d} {self.average_cc:.2f} {self.average_amp_rat_z:.2f} {self.average_amp_rat_1:.2f} {self.average_amp_rat_2:.2f}\n")
#         fp.write("\n")
#         fp.write("station cc_value start_time time_shift amplitude_ratio_z amplitude_ratio_1 amplitude_ratio_2\n")

#         for i in range(self.num_of_stations):
#             station = self.stations[i]
#             ccval = self.cc_values[i]
#             timestr = self.start_times[i].strftime("%Y-%m-%dT%H:%M:%S.%f")
#             tshift = self.time_shifts[i]
#             amprat_z = self.amplitude_ratios_z[i]
#             amprat_1 = self.amplitude_ratios_1[i]
#             amprat_2 = self.amplitude_ratios_2[i]

#             fp.write(f"{station} {ccval:.2f} {timestr} {tshift:.3f} {amprat_z:.2f} {amprat_1:.2f} {amprat_2:.2f}\n")

#         fp.write("\n")


#     ### Get the information for a specific station
#     def get_info_for_station(self, station, entries=None):

#         if station not in self.stations:
#             print(f"Warning: {station} is not in the list of stations!")
#             return None

#         i = self.stations.index(station)

#         ccval = self.cc_values[i]
#         starttime = self.start_times[i]
#         tshift = self.time_shifts[i]
#         amprat_z = self.amplitude_ratios_z[i]
#         amprat_1 = self.amplitude_ratios_1[i]
#         amprat_2 = self.amplitude_ratios_2[i]
        
#         if entries is None:
#             ccval = self.cc_values[i]
#             starttime = self.start_times[i]
#             tshift = self.time_shifts[i]
#             amprat_z = self.amplitude_ratios_z[i]
#             amprat_1 = self.amplitude_ratios_1[i]
#             amprat_2 = self.amplitude_ratios_2[i]

#             output = {"cc_value":ccval, "start_time":starttime, "time_shift":tshift, "amplitude_ratio_z":amprat_z, "amplitude_ratio_1":amprat_1, "amplitude_ratio_2":amprat_2}
                      
#         else:
#             if type(entries) is not list:
#                 if type(entries) is str:
#                     entries = [entries]
#                 else:
#                     print("Error: entries must be a list of strings!")
#                     raise TypeError
        
#             output = {}
#             for entry in entries:
#                 if entry == "cc_value":
#                     output["cc_value"] = self.cc_values[i]
#                 elif entry == "start_time":
#                     output["start_time"] = self.start_times[i]
#                 elif entry == "time_shift":
#                     output["time_shift"] = self.time_shifts[i]
#                 elif entry == "amplitude_ratio_z":
#                     output["amplitude_ratio_z"] = self.amplitude_ratios_z[i]
#                 elif entry == "amplitude_ratio_1":
#                     output["amplitude_ratio_1"] = self.amplitude_ratios_1[i]
#                 elif entry == "amplitude_ratio_2":
#                     output["amplitude_ratio_2"] = self.amplitude_ratios_2[i]
#                 else:
#                     print(f"Error: {entry} is not a valid entry!")
#                     raise ValueError
                
#         return output

# ## Class for storing the waveforms of one template event
# class TemplateEventWaveforms:
#     def __init__(self, tempinfo, stations_active, components_active, stream):
#         numcomp = len(components_active)

#         if len(stations_active) != len(stream) // numcomp:
#             print("Error: The number of active stations does not match the number of traces!")
#             raise ValueError
        
#         self.info = tempinfo
#         self.active_stations = stations_active
#         self.waveforms = stream

#     def __str__(self):
#         timestr = self.temp_info.first_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
#         return f"{self.temp_info.name}, {timestr}, {self.temp_info.num_of_stations} stations, {self.active_stations} active stations"
    
#     def __repr__(self):
#         return self.__str__()

    
# ## Class for storing the waveforms of one matched event
# class MatchedEventWaveforms:
#     def __init__(self, matchinfo, stations_active, components_active, stream):
#         numcomp = len(components_active)

#         if len(stations_active) != len(stream) // numcomp:
#             print("Error: The number of active stations does not match the number of traces!")
#             raise ValueError
        
#         self.info = matchinfo
#         self.active_stations = stations_active
#         self.waveforms = stream

#     def __str__(self):
#         timestr = self.info.first_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
#         return f"{self.info.name}, {timestr}, {self.info.num_of_stations} stations, {self.active_stations} active stations"
        
#     def __repr__(self):
#         return self.__str__()

# # Class for storing the waveforms of all matched events
# class MatchWaveforms:
#     def __init__(self):
#         self.active_stations = []
#         self.events= []

#     def __len__(self):
#         return len(self.events)

#     def __getitem__(self, index):
#         return self.events[index]

#     def __setitem__(self, index, event):
#         self.events[index] = event

#     def __iter__(self):
#         return iter(self.events)

#     def append(self, event):
#         self.events.append(event)
#         for station in event.active_stations:
#             if station not in self.active_stations:
#                 self.active_stations.append(station)

#     def clear(self):
#         self.events.clear()

#     def sort(self, key=None, reverse=False):
#         self.events.sort(key=key, reverse=reverse)

#     def reverse(self):
#         self.events.reverse()

#     def get_match_names(self):
#         names = []
#         for event in self.events:
#             names.append(event.info.name)
#         return names

#     def get_matches_by_names(self, names):
#         events = MatchWaveforms()
#         for event in self.events:
#             if event.info.name in names:
#                 events.append(event)
#         return events
    
#     ## Output the waveforms of each station to an array for plotting
#     def to_arrays(self, component='Z', order='cc', stations=None, normalize=True):
#         if order == 'cc':
#             self.sort(key=lambda x: x.info.average_cc, reverse=True)
#         elif order == 'time':
#             self.sort(key=lambda x: x.info.first_start_time, reverse=False)
#         elif order == 'amplitude_ratio':
#             self.sort(key=lambda x: x.info.average_amp_rat_z, reverse=True)

#         if stations is None:
#             stations = self.active_stations

#         numev = len(self.events)
#         numpts = self.events[0].waveforms[0].stats.npts

#         arraydict = {}
#         for station in stations:
#             datamat = zeros((numev, numpts))

#             for i, event in enumerate(self.events):
#                 try:
#                     trace = event.waveforms.select(station=station, component=component)[0]
#                     data = trace.data

#                     if normalize == True:
#                         data = data / amax(abs(data))

#                 except IndexError:
#                     data = full(numpts, nan)

#                 datamat[i, :] = data
#             arraydict[station] = datamat

#         return arraydict

    
# ## Detect the change in differential travel time using the slope derived from linear regression
# ## Time shifts are in ms!

# def detect_arrival_delta(steps_in, deltas, winlen=10, slopethr_findpk=0.1, slopethr_uncer=0, minshift=25, maxshift=100):    
#     ### Compute the standard deviation of the time shifts in each window
#     slopes = []
#     steps_out = []
#     numpts = len(steps_in)
#     i = 0
#     while i < numpts-winlen:
#         steps_win = steps_in[i:i+winlen]
#         deltas_win = deltas[i:i+winlen]

#         y = deltas_win - mean(deltas_win)
#         x = steps_win - mean(steps_win)

#         slope = linregress(x, y)[0]

#         slopes.append(slope)
#         steps_out.append(steps_win[0]+round(winlen/2))

#         i += 1
#     steps_out = array(steps_out)
#     slopes = array(slopes)

#     ### Detect the peaks in the slope
#     slopes_findpk = slopes[(steps_out > minshift) & (steps_out < maxshift)]
#     steps_findpk = steps_out[(steps_out > minshift) & (steps_out < maxshift)]

#     ipospeaks, posdict = find_peaks(slopes_findpk, height=slopethr_findpk)
#     inegpeaks, negdict = find_peaks(-slopes_findpk, height=slopethr_findpk)

#     posheights = posdict['peak_heights']
#     negheights = negdict['peak_heights']

#     if len(ipospeaks) == 0 and len(inegpeaks) == 0:
#         print('No peaks found')
#         return None, None, steps_out, slopes
#     elif len(ipospeaks) == 0:
#         ipeaks = inegpeaks
#         heights = negheights
#     elif len(inegpeaks) == 0:
#         ipeaks = ipospeaks
#         heights = posheights
#     else:
#         maxpos = amax(posheights)
#         maxneg = amax(negheights)

#         if maxpos > maxneg:
#             ipeaks = ipospeaks
#             heights = posheights
#         else:
#             ipeaks = inegpeaks
#             heights = negheights

#     imax = argmax(heights)
#     ipeak = ipeaks[imax]
#     step_peak = steps_findpk[ipeak]
#     maxslope = slopes_findpk[ipeak]

#     ### Estimate the uncertainty
#     uncer = get_peak_uncer(ipeak, steps_findpk, slopes_findpk, thr=slopethr_uncer)

#     return step_peak, uncer, steps_out, slopes, maxslope

# ## Detect the change in cc values using the slope derived from linear regression
# ## Time shifts are in ms!
# def detect_arrival_ccval(steps_in, ccvals, winlen=10, slopethr_findpk=0, slopethr_uncer=0, minshift=25, maxshift=100):
#     ### Compute the standard deviation of the time shifts in each window
#     slopes = []
#     steps_out = []
#     numpts = len(steps_in)
#     i = 0
#     while i < numpts-winlen:
#         steps_win = steps_in[i:i+winlen]
#         ccvals_win = ccvals[i:i+winlen]

#         y = ccvals_win - mean(ccvals_win)
#         x = steps_win - mean(steps_win)

#         slope = linregress(x, y)[0]

#         slopes.append(slope)
#         steps_out.append(steps_win[0]+round(winlen/2))

#         i += 1
#     steps_out = array(steps_out)
#     slopes = array(slopes)

#     ### Detect the peaks in the slope
#     slopes_findpk = -slopes[(steps_out > minshift) & (steps_out < maxshift)]
#     steps_findpk = steps_out[(steps_out > minshift) & (steps_out < maxshift)]

#     ipeaks, pkdict = find_peaks(slopes_findpk, height=slopethr_findpk)
#     pkheights = pkdict['peak_heights']

#     if len(ipeaks) == 0:
#         print('No peaks found')
#         return None, None, steps_out, slopes

#     imax = argmax(pkheights)
#     ipeak = ipeaks[imax]
#     step_peak = steps_findpk[ipeak]
#     maxslope = slopes_findpk[ipeak]

#     ### Estimate the uncertainty
#     uncer = get_peak_uncer(ipeak, steps_findpk, slopes_findpk, thr=slopethr_uncer)

#     return step_peak, uncer, steps_out, slopes, maxslope

# ## Find the uncertainty of the time shift by finding the width of the peak
# def get_peak_uncer(ipeak, steps, vals, thr=0.0):
#     ### Find the peak width
#     ipeak = int(ipeak)
#     steps = array(steps)
#     vals = array(vals)

#     i = ipeak
#     while i < len(vals)-1:
#         if vals[i] <= thr:
#             break
#         i += 1
#     posuncer = steps[i]-steps[ipeak]

#     i = ipeak
#     while i > 0:
#         if vals[i] <= thr:
#             break
#         i -= 1
#     istart = i
#     neguncer = steps[ipeak]-steps[istart]

#     uncer = amin([posuncer, neguncer])
#     return uncer
    
# ## Find the PS differential time using sliding-window cross-correlation 
# ## All times are in points or ms!
# def get_psdifftime_slidecc(tempstr, targstr, difftime_start, winlen_cc=20, winlen_slope=10, slopethr_findpk=0, slopethr_uncer=0, maxshift_delta=5, maxshift_psdiff=100):
#     ### Compute the sliding-window cc 
#     steps, deltas, ccvals =  get_slidecc(tempstr, targstr, difftime_start, winlen=winlen_cc, maxshift=maxshift_delta)

#     ### Detect the S onset using the cc values
#     step_psdiff, uncer, steps_slope, slopes, maxslope = detect_arrival_ccval(steps, ccvals, winlen=winlen_slope, slopethr_uncer=slopethr_uncer, slopethr_findpk=slopethr_findpk, maxshift=maxshift_psdiff)

#     print(f"P-S differential travel time derived using the cc values: {step_psdiff} +/- {uncer} ms")

#     return step_psdiff, uncer, steps, deltas, ccvals, steps_slope, slopes, maxslope


# ## Compute sliding window cross-correlation between the template and target waveforms
# def get_slidecc(tempstr, targstr, difftime_start, winlen=50, maxshift=5):
#     ### Determine if the templates and the full waveforms are the 3C recored by the same station
#     ### If not, return an error message
#     if len(targstr) != 3 or len(tempstr) != 3:
#         print('Error: full waveform must be 3C')
#         raise ValueError
    
#     if targstr[0].stats.station != tempstr[0].stats.station:
#         print('Error: full waveform and template must be from the same station')
#         raise ValueError  
    
#     ### Select the 3C data for the target stream
#     targtrc_z = targstr.select(channel='GHZ')[0]
#     targtrc_1 = targstr.select(channel='GH1')[0]
#     targtrc_2 = targstr.select(channel='GH2')[0]

#     target_z = targtrc_z.data
#     target_1 = targtrc_1.data
#     target_2 = targtrc_2.data

#     ### Select the 3C data for the template stream

#     temptrc_z = tempstr.select(channel='GHZ')[0]
#     temptrc_1 = tempstr.select(channel='GH1')[0]
#     temptrc_2 = tempstr.select(channel='GH2')[0]

#     template_z = temptrc_z.data
#     template_1 = temptrc_1.data
#     template_2 = temptrc_2.data

#     ### The difference in starting time
#     diffstep_start = abs(round(difftime_start))

#     # print(len(template_z))
#     # print(len(target_z))

#     ### Compute the cross-correlation for each window
#     npts_temp = len(template_z)
#     imax = npts_temp-winlen

#     i = 0
#     deltas = []
#     steps = []
#     ccvals = []
#     while i < imax:
#         tempwin_z = template_z[i:i+winlen]
#         tempwin_1 = template_1[i:i+winlen]
#         tempwin_2 = template_2[i:i+winlen]

#         targwin_z = target_z
#         targwin_1 = target_1
#         targwin_2 = target_2

#         # print(i)
#         # print(len(tempwin_z))
#         # print(len(targwin_z))

#         xcorr_z = correlate_template(targwin_z, tempwin_z)
#         xcorr_1 = correlate_template(targwin_1, tempwin_1)
#         xcorr_2 = correlate_template(targwin_2, tempwin_2)

#         xcorr = (xcorr_z+xcorr_1+xcorr_2)/3

#         if maxshift is not None:
#             xcorr = xcorr[i+diffstep_start-maxshift:i+diffstep_start+maxshift+1]
#             ccind = argmax(xcorr)
#             ccval = xcorr[ccind]

#             delta = ccind-maxshift
#         else:
#             ccind = argmax(xcorr)
#             ccval = xcorr[ccind]

#             delta = ccind-diffstep_start

#         deltas.append(delta)
#         steps.append(i)
#         ccvals.append(ccval)

#         i += 1

#     steps = array(steps)+winlen
#     deltas = array(deltas)
#     ccvals = array(ccvals)
    
#     return steps, deltas, ccvals

# ## Perform 3C template matching for one template
# def template_match_3c(fullstr, tempstr, tempname, mincc=0.5, amplitude_ratio=True):
#     ### Determine if the templates and the full waveforms are the 3C recored by the same station
#     ### If not, return an error message
#     if len(fullstr) != 3 or len(tempstr) != 3:
#         print('Error: full waveform must be 3C')
#         raise ValueError
    
#     if fullstr[0].stats.station != tempstr[0].stats.station:
#         print('Error: full waveform and template must be from the same station')
#         raise ValueError


#     fullstr_z = fullstr.select(channel='GHZ')
#     fullstr_1 = fullstr.select(channel='GH1')
#     fullstr_2 = fullstr.select(channel='GH2')

#     tempstr_z = tempstr.select(channel='GHZ')
#     tempstr_1 = tempstr.select(channel='GH1')
#     tempstr_2 = tempstr.select(channel='GH2')

#     fulltrc_z = fullstr_z[0]
#     data_z = fulltrc_z.data
#     temptrc_z = tempstr_z[0]
#     temp_z = temptrc_z.data
#     dt = temptrc_z.stats.delta

#     xcorr_z = correlate_template(data_z, temp_z)

#     fulltrc_1 = fullstr_1[0]
#     data_1 = fulltrc_1.data
#     temptrc_1 = tempstr_1[0]
#     temp_1 = temptrc_1.data
#     dt = temptrc_1.stats.delta

#     xcorr_1 = correlate_template(data_1, temp_1)

#     fulltrc_2 = fullstr_2[0]
#     data_2= fulltrc_2.data
#     temptrc_2 = tempstr_2[0]
#     temp_2 = temptrc_2.data
#     dt = temptrc_2.stats.delta

#     xcorr_2 = correlate_template(data_2, temp_2)

#     xcorr_sum = (xcorr_z+xcorr_1+xcorr_2)/3

#     indlst, peaks = find_peaks(xcorr_sum, height=mincc)

#     if len(indlst) == 0:
#         print('No detections found.')
#         return None
#     else:
#         npts = temptrc_z.stats.npts
#         amprats_z = []
#         amprats_1 = []
#         amprats_2 = []


#         if amplitude_ratio == True:
#             for ind in indlst:
#                 match_z = data_z[ind:ind+npts]
#                 match_1 = data_1[ind:ind+npts]
#                 match_2 = data_2[ind:ind+npts]

#                 amprat_z = norm(match_z, ord=1)/norm(temp_z, ord=1)
#                 amprat_1 = norm(match_1, ord=1)/norm(temp_1, ord=1)
#                 amprat_2 = norm(match_2, ord=1)/norm(temp_2, ord=1)

#                 amprats_z.append(amprat_z)
#                 amprats_1.append(amprat_1)
#                 amprats_2.append(amprat_2)

#         pktimes = []
#         tshifts = []
#         for ind in indlst:
#             pktime = pd.Timedelta(seconds=ind*dt)+pd.to_datetime(fulltrc_z.stats.starttime.datetime)
#             tshift = pktime-pd.to_datetime(temptrc_z.stats.starttime.datetime)

#             pktimes.append(pktime)
#             tshifts.append(tshift)

#         peaks['time'] = pktimes
#         peaks['time_shift'] = tshifts
#         peaks['duration'] = (npts-1)*dt
#         peaks['amplitude_ratio_z'] = amprats_z
#         peaks['amplitude_ratio_1'] = amprats_1
#         peaks['amplitude_ratio_2'] = amprats_2
#         peakdf = pd.DataFrame(peaks)
#         peakdf.rename(columns={'peak_heights':'cc_value'}, inplace=True)

#         peakdf.insert(0, "template", tempname)
#         peakdf.insert(1, "station", fulltrc_z.stats.station)
        
#         peakdf = peakdf.loc[peakdf['cc_value'] < 0.99] # Remove the self detections and the detections with anomalous high cc values (why doe they exist???)
#         peakdf.sort_values(by='time_shift', ascending=False, inplace=True, ignore_index=True)
#         numdet = len(peakdf)

#         print(f'{numdet} detections found.')

#         return peakdf
    
# ## Find the matched events by associating the detections
# def associate_detections(detdf, numdet_min=4, delta_max=0.1):

#     ### Find all the matched events
#     numdet = len(detdf)
#     matchdfs = []
#     i = 0
#     while i < numdet:
#         station0 = detdf["station"][i]
#         tshift0 = detdf["time_shift"][i]
#         stations = [station0]
#         detdfs_tmp = [detdf.iloc[i]]
#         # print(type(detdfs_tmp[0]))
#         j = i + 1

#         while j < numdet:
#             tshift = detdf["time_shift"][j]
#             station = detdf["station"][j]

#             if tshift - tshift0 > delta_max:
#                 break
#             else:
#                 if station not in stations:
#                     stations.append(station)
#                     detdfs_tmp.append(detdf.iloc[j])
            
#             j += 1

#         # print(numdet_ev)
#         if len(stations) >= numdet_min:
#             #print(type(detdfs_tmp[0]))
            
#             matchdf = pd.DataFrame(detdfs_tmp)
#             matchdf.reset_index(drop=True, inplace=True)
#             # print(matchdf)
#             # print(evdf)
#             matchdfs.append(matchdf)
#             # print(matchdf)
#             #print(matchdf)
#             # detdf.drop(detdf.index[i:j], inplace=True)
#             i = j
#         else:
#             i += 1

#     numev = len(matchdfs)
#     print(f"There are in total {numev} matches.")

#     ### Store the information in a Matches object
#     matches = Matches()
#     for i, evdf in enumerate(matchdfs):
#         #print(evdf)
#         if numev < 10:
#             matchname = "Match"+str(i+1)
#         elif numev < 100:
#             matchname = "Match"+str(i+1).zfill(2)
#         elif numev < 1000:
#             matchname = "Match"+str(i+1).zfill(3)
#         elif numev < 10000:
#             matchname = "Match"+str(i+1).zfill(4)
#         elif numev < 100000:
#             matchname = "Match"+str(i+1).zfill(5)
#         else:
#             print("Too many matches!")
#             raise ValueError
       
#         matchname = matchname
#         stations = evdf["station"].tolist()
#         ccvals = evdf["cc_value"].array
#         starttimes = evdf["time"].tolist()
#         tshifts = evdf["time_shift"].array
#         amprats_z = evdf["amplitude_ratio_z"].array
#         amprats_1 = evdf["amplitude_ratio_1"].array
#         amprats_2 = evdf["amplitude_ratio_2"].array

#         match = MatchedEvent(matchname, stations, ccvals, starttimes, tshifts, amprats_z, amprats_1, amprats_2)
#         matches.append(match)

#     return matches

# ## Read the template and match information from a file
# def read_template_and_match(inpath):

#     with open(inpath, 'r') as fp:

#         ### Read the template information
#         line = fp.readline()
#         if not line[0].startswith("#"):
#             raise ValueError("Error: the format of the template information is incorrect!")

#         line = fp.readline()
#         tempname = line.strip()
        
#         fp.readline()
#         fp.readline()

#         line = fp.readline()
#         fields = line.split()
#         freqmin = float(fields[0])
#         if fields[1] == "None":
#             freqmax = None
#         else:
#             freqmax = float(fields[1])
#         mincc = float(fields[2])
#         numdet_min = int(fields[3])
        
#         fp.readline()
#         fp.readline()

#         line = fp.readline()
#         fields = line.split()
#         dur = float(fields[1])
#         numst = int(fields[2])
#         nummatch = int(fields[3])
        
#         fp.readline()
#         fp.readline()

#         stations = []
#         starttimes = []
#         for _ in range(numst):
#             line = fp.readline()
#             fields = line.split()
#             station = fields[0]
#             starttime = pd.to_datetime(fields[1])
#             stations.append(station)
#             starttimes.append(starttime)

#         fp.readline()

#         template = TemplateEvent(tempname, dur, stations, starttimes)

#         ### Read the match information
#         matches = read_matches(fp, nummatch)
    
#     tempmatch = TemplateMatches(freqmin, freqmax, mincc, numdet_min, template, matches)

#     return tempmatch

# ## Read the match information from an already opened file
# def read_matches(fp, nummatch):
#     matches = Matches()

#     for _ in range(nummatch):
#         line = fp.readline()
#         if not line.startswith("##"):
#             print("Error: the format of the match information is incorrect!")
#             raise ValueError

#         line = fp.readline()
#         matchname = line.strip()

#         fp.readline()
#         fp.readline()

#         line = fp.readline()
#         fields = line.split()
#         numst = int(fields[1])

#         fp.readline()
#         fp.readline()

#         stations = []
#         ccvals = []
#         starttimes = []
#         tshifts = []
#         amprats_z = []
#         amprats_1 = []
#         amprats_2 = []
#         for j in range(numst):
#             line = fp.readline()
#             fields = line.split()
#             station = fields[0]
#             ccval = float(fields[1])
#             starttime = pd.to_datetime(fields[2])
#             tshift = float(fields[3])
#             amprat_z = float(fields[4])
#             amprat_1 = float(fields[5])
#             amprat_2 = float(fields[6])

#             stations.append(station)
#             ccvals.append(ccval)
#             starttimes.append(starttime)
#             tshifts.append(tshift)
#             amprats_z.append(amprat_z)
#             amprats_1.append(amprat_1)
#             amprats_2.append(amprat_2)

#         match = MatchedEvent(matchname, stations, ccvals, starttimes, tshifts, amprats_z, amprats_1, amprats_2)
#         matches.append(match)

#         fp.readline()

#     return matches

# ## Get the time window of the template from its name
# def get_timewin_from_template_name(name):
#     pattern = r"(\d{2})-(\d{2})-(\d{2})-(\d{2})"
#     match = search(pattern, name)

#     if match:
#         daystr = match.group(1)
#         hourstr = match.group(2)
#         day = int(daystr)
#     else:
#         raise ValueError("No time-window information found in template name")

#     if day < 10:
#         timewin = f"2020-02-{daystr}-{hourstr}-00-00"
#     else:
#         timewin = f"2020-01-{daystr}-{hourstr}-00-00"

#     return timewin

# ## Get the subarray from the template name
# def get_subarray_from_template_name(name):
#     fields = split("\d{2}-\d{2}-\d{2}-\d{2}", name)
#     subarray = fields[0][-1]

#     return subarray

# ## Get the frequency band from the suffix
# def get_freqband_from_suffix(suffix):
#     if "bandpass" in suffix:
#         pattern = r"bandpass(\d+)-(\d+)hz"
#         match = search(pattern, suffix)
#         if match:
#             freqmin = float(match.group(1))
#             freqmax = float(match.group(2))

#             return freqmin, freqmax
#         else:
#             raise ValueError("No frequency band information found in the suffix")
#     elif "highpass" in suffix:
#         pattern = r"highpass(\d+)hz"
#         match = search(pattern, suffix)
#         if match:
#             freqmin = float(match.group(1))
#             freqmax = None

#             return freqmin, freqmax
#         else:
#             raise ValueError("No frequency band information found in the suffix")
#     elif "lowpass" in suffix:
#         pattern = r"lowpass(\d+)hz"
#         match = search(pattern, suffix)
#         if match:
#             freqmin = 0
#             freqmax = float(match.group(1))

#             return freqmin, freqmax
#         else:
#             raise ValueError("No frequency band information found in the suffix")
#     else:
#         raise ValueError("No frequency band information found in the suffix")




    

            