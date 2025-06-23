"""
Detect similar signals in geophone waveforms and save **all** detections from
*each* template into a single HDF5 file (one file per template).
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from argparse import ArgumentParser
from time import time
from pathlib import Path
from numpy import where, float32

from obspy import Stream
from obspy.signal.cross_correlation import correlate_template
from pandas import Timestamp, Timedelta

from utils_basic import DETECTION_DIR as dirpath, get_geophone_days, SAMPLING_RATE as sampling_rate
from utils_preproc import read_and_process_day_long_geo_waveforms
from utils_sta_lta import Snippet, Snippets
from utils_cc import Template, Match, TemplateMatches

# -----------------------------------------------------------------------------
# Helper: match a single template against one day's data → list[Match]
# -----------------------------------------------------------------------------

def match_template(template: Snippet,
                   data: Stream,
                   cc_threshold: float = 0.85) -> list[Match]:
    """Return a list of Match objects where *template* aligns with *data*."""
    tpl_z, tpl_1, tpl_2 = (
        template.waveform['Z'],
        template.waveform['1'],
        template.waveform['2'],
    )
    tr_z, tr_1, tr_2 = (
        data.select(component="Z")[0],
        data.select(component="1")[0],
        data.select(component="2")[0],
    )

    day_start = Timestamp(tr_z.stats.starttime.datetime, tz="UTC")

    cc = (
        correlate_template(tr_z.data, tpl_z) +
        correlate_template(tr_1.data, tpl_1) +
        correlate_template(tr_2.data, tpl_2)
    ) / 3.0

    indices = where(cc >= cc_threshold)[0]
    num_pts = template.num_pts

    matches: list[Match] = []
    for idx in indices:
        abs_time = day_start + Timedelta(seconds=idx / sampling_rate)
        matches.append(
            Match(
                starttime=abs_time,
                coeff=cc[idx],
                waveform={
                    "Z": tr_z.data[idx : idx + num_pts].astype(float32),
                    "1": tr_1.data[idx : idx + num_pts].astype(float32),
                    "2": tr_2.data[idx : idx + num_pts].astype(float32),
                },
            )
        )
    return matches

# -----------------------------------------------------------------------------
# Main procedure
# -----------------------------------------------------------------------------

def main() -> None:
    # Parse command-line arguments at the start of main
    parser = ArgumentParser(description="Template matching driver: aggregate matches per template")
    parser.add_argument("--station", required=True, help="Station code to process")
    parser.add_argument("--on_threshold", type=float, default=17.0,
                        help="STA/LTA on threshold used in template generation")
    parser.add_argument("--cc_threshold_template", type=float, default=0.95,
                        help="Cross-correlation threshold for loading templates")
    parser.add_argument("--min_pop", type=int, default=200,
                        help="Minimum popularity used in template generation filename")
    parser.add_argument("--cc_threshold_match", type=float, default=0.85,
                        help="Cross-correlation threshold for detecting matches")
    parser.add_argument("--min_freq_filter", type=float, default=10.0,
                        help="Low frequency cutoff for filtering data")
    parser.add_argument("--max_freq_filter", type=float, default=None,
                        help="High frequency cutoff for filtering data")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: only process one day")
    args = parser.parse_args()

    # Unpack arguments
    station = args.station
    on_thr = args.on_threshold
    cc_thr_tpl = args.cc_threshold_template
    min_pop = args.min_pop
    cc_thr_match = args.cc_threshold_match
    fmin = args.min_freq_filter
    fmax = args.max_freq_filter
    test = args.test

    day_test = "2020-01-16"

    # Load template snippets
    tpl_fname = (
        f"template_sta_lta_detections_{station}_"
        f"on{on_thr:.1f}_"
        f"cc{cc_thr_tpl:.2f}_"
        f"pop{min_pop}.h5"
    )
    templates: Snippets = Snippets.from_hdf(Path(dirpath) / tpl_fname)
    print(f"Loaded {len(templates)} templates from {tpl_fname}")

    # Initialize one TemplateMatches per template
    containers: list[TemplateMatches] = []
    for snip in templates:
        tpl = Template(
            id=snip.id,
            starttime=snip.starttime,
            num_pts=snip.num_pts,
            waveform=snip.waveform,
        )
        containers.append(TemplateMatches(tpl))

    # Iterate over days and collect matches
    if test:
        print(f"Running in test mode: only process {day_test}")
        days = [day_test]
    else:
        days = get_geophone_days()

    for day in days:
        print(f"\n=== Processing {day} ===")
        t0 = time()
        stream = read_and_process_day_long_geo_waveforms(
            day,
            stations=[station],
            filter=True,
            filter_type="butter",
            min_freq=fmin,
            max_freq=fmax,
            trim_to_day=False,
        )
        if stream is None:
            print("  No data for this day.")
            continue
        print(f"  Data loaded in {time() - t0:.2f}s")

        t0 = time()
        for snip, tm in zip(templates, containers):
            new_matches = match_template(snip, stream, cc_thr_match)
            tm.add_matches(new_matches)
            print(f"    {snip.id}: +{len(new_matches)} (total {len(tm)})")
        print(f"  Template matching finished in {time() - t0:.2f}s")

    # Write one HDF5 file per template using snippet id
    outdir = Path(dirpath)
    outdir.mkdir(parents=True, exist_ok=True)
    for tm in containers:
        tmpl_id = tm.template.id
        outfile = outdir / f"matches_{station}_template{tmpl_id}.h5"

        if not tm.matches:
            print(f"No matches for {tmpl_id} → {outfile.name}")
            continue

        tm.to_hdf(outfile)
        print(f"Saved {len(tm)} matches for Template {tmpl_id} → {outfile.name}")

    print("All templates processed ✔︎")


if __name__ == "__main__":
    main()

    