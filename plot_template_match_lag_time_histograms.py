"""
Plot the histograms of the zero-mean time lags of all matches of a template
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from numpy import asarray, arange
from pandas import DataFrame
from matplotlib.pyplot import Axes, figure
from matplotlib.gridspec import GridSpec

from utils_basic import GEO_COMPONENTS as components, GEO_CHANNELS as channels, SAMPLING_RATE as sampling_rate, ROOTDIR_GEO as dirpath_geo, PICK_DIR as dirpath_pick, DETECTION_DIR as dirpath_det
from utils_basic import get_geophone_days
from utils_cc import TemplateMatches, Template, associate_matched_events
from utils_cont_waveform import DayLongWaveform, load_day_long_waveform_from_hdf, load_waveform_slice
from utils_snuffler import read_time_windows
from utils_plot import save_figure


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

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
    ax.set_xlabel("Normalised time lag (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"{station}", fontsize=12, fontweight="bold")

    return ax

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--template_id", type=str, help="Template ID")
parser.add_argument("--stations", type=str, nargs="+", default=["A01", "A02", "A03", "A04", "A05", "A06"], help="Stations to plot")

parser.add_argument("--min_num_sta", type=int, default=6, help="Minimum number of stations to consider a matched event")
parser.add_argument("--min_lag", type=float, default=-5e-3, help="Minimum lag time (s)")
parser.add_argument("--max_lag", type=float, default=5e-3, help="Maximum lag time (s)")
parser.add_argument("--bin_width", type=float, default=1e-3, help="Bin width (s)")

parser.add_argument("--subplot_height", type=float, default=3.0, help="Height of each subplot")
parser.add_argument("--subplot_width", type=float, default=4.0, help="Width of each subplot")

parser.add_argument("--min_freq_filter", type=float, help="The low corner frequency for filtering the data", default=20.0)

args = parser.parse_args()
template_id = args.template_id
stations = args.stations
min_num_sta = args.min_num_sta
min_lag = args.min_lag
max_lag = args.max_lag
bin_width = args.bin_width
subplot_height = args.subplot_height
subplot_width = args.subplot_width
min_freq_filter = args.min_freq_filter

# Load the match information
print(f"Loading the data for {template_id}...")
filepath = Path(dirpath_det) / f"template_matches_manual_templates_freq{min_freq_filter:.0f}hz.h5"
tm_dict = {}
for station in stations:
    tm = TemplateMatches.from_hdf(filepath, id=template_id, station=station)
    tm_dict[station] = tm

for station, tm in tm_dict.items():
    print(station)
    print(tm.template.starttime)

# Associate the events
print(f"Associating the events for {template_id}...")
record_df = associate_matched_events(tm_dict, min_num_sta)

# Get the normalized time lags
print(f"Getting the normalized time lags for {template_id}...")
record_df = get_normalized_time_lags(tm_dict, record_df)

# Plot the histograms
# Compute the number of rows
num_rows = len(stations) // 3 + 1

# Generate the subplots
fig = figure(figsize=(subplot_width * 3, subplot_height * num_rows))
gs = GridSpec(num_rows, 3, figure=fig, hspace=0.5)

for i, station in enumerate(stations):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    plot_station_lag_time_histogram(ax, record_df, station, min_lag, max_lag, bin_width)

# Set the subplot titles
fig.suptitle(f"Template {template_id}", fontsize=14, fontweight="bold", y = 0.95)

# Save the figure
save_figure(fig, f"template_match_lag_time_histograms_{template_id}.png")





    
