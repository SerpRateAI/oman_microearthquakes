"""
Plot the misfit distribution of the template event in a violin plot fo a scale factor of 1 and the best-fit scale factor for the template event.

"""
from argparse import ArgumentParser
from pathlib import Path
from numpy import arange, array
import pandas as pd
from matplotlib.pyplot import subplots, hist

from utils_basic import (
    LOC_DIR as dirpath_loc,
)

from utils_basic import get_freq_limits_string
from utils_loc import load_station_misfits_from_hdf_combined
from utils_plot import save_figure

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--template_id", type=str, required=True)
    parser.add_argument("--min_freq_filter", type=float, required=True)
    parser.add_argument("--max_freq_filter", type=float, required=True)
    parser.add_argument("--arrival_type", type=str, required=True)
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--subarray", type=str, required=True)
    parser.add_argument("--scale_factor_best", type=float, required=True)
    parser.add_argument("--min_misfit_bin", type=float, default=-20.0)
    parser.add_argument("--max_misfit_bin", type=float, default=20.0)
    parser.add_argument("--misfit_bin_width", type=float, default=2.0)
    args = parser.parse_args()

    template_id = args.template_id
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    arrival_type = args.arrival_type
    phase = args.phase
    subarray = args.subarray
    scale_factor_best = args.scale_factor_best
    min_misfit_bin = args.min_misfit_bin
    max_misfit_bin = args.max_misfit_bin
    misfit_bin_width = args.misfit_bin_width

    # Load the station misfits
    freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
    filepath = Path(dirpath_loc) / f"location_info_template_{template_id}_{freq_str}.h5"
    
    ## Read the station misfits for the best-fit scale factor
    misfit_dict_best = load_station_misfits_from_hdf_combined(filepath, arrival_type, phase, subarray, scale_factor_best)
    misfits_best = array(list(misfit_dict_best.values())) * 1000

    ## Read the station misfits for the scale factor of 1
    misfit_dict_one = load_station_misfits_from_hdf_combined(filepath, arrival_type, phase, subarray, 1.0)
    misfits_one = array(list(misfit_dict_one.values())) * 1000
    # Plot the misfit distribution
    fig, ax = subplots(1, 1, figsize=(10, 5))

    bin_edges = arange(min_misfit_bin, max_misfit_bin + misfit_bin_width, misfit_bin_width)
    ax.hist(misfits_one, bins=bin_edges, color="lightgray", edgecolor="black", alpha=0.7, align="mid", label=f"1.0")
    ax.hist(misfits_best, bins=bin_edges, color="lightblue", edgecolor="black", alpha=0.7, align="mid", label=f"{scale_factor_best:.1f}")
    ax.set_xlabel("Station misfit (ms)")
    ax.set_ylabel("Count")

    ax.legend(title = "Scale factor")
    ax.set_title(f"Template {template_id}", fontsize = 14, fontweight = "bold")

    save_figure(fig, f"template_station_misfit_histograms_{template_id}_{freq_str}.png")
