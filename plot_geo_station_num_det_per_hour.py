"""
Plot the number of STA/LTA detections per hour for a station.
"""

from pathlib import Path
from argparse import ArgumentParser
from matplotlib.pyplot import figure, subplots

from utils_basic import DETECTION_DIR as dirpath, STARTTIME_GEO as starttime_bin, ENDTIME_GEO as endtime_bin
from utils_sta_lta import Snippets
from utils_cc import TemplateMatches
from utils_plot import save_figure, format_datetime_xlabels, add_day_night_shading

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def plot_num_det_per_hour(bin_count_df,
                          ax = None, figwidth = 12, figheight = 5, 
                          linewidth = 1, linecolor = "black",
                          axis_label_size = 12, title_size = 14,
                          title = None,
                          label = None):
    
    # Create the figure and axes
    if ax is None:
        fig, ax = subplots(figsize=(figwidth, figheight))
    else:
        fig = ax.get_figure()
    
    # Plot the number of detections per hour
    if label is not None:
        ax.plot(bin_count_df["bin_center"], bin_count_df["bin_count"], linewidth=linewidth, color=linecolor, label=label)
    else:
        ax.plot(bin_count_df["bin_center"], bin_count_df["bin_count"], linewidth=linewidth, color=linecolor)

    # Set the x axis limit
    ax.set_xlim(bin_count_df["bin_center"].min(), bin_count_df["bin_center"].max())
    
    # Set the x and y labels
    format_datetime_xlabels(ax,
                            major_tick_spacing = "1d",
                            num_minor_ticks = 4,
                            date_format = "%Y-%m-%d",
                            va = 'top', ha = 'right',
                            rotation = 30)
    
    ax.set_ylabel("Num. of detections", fontsize = axis_label_size)

    # Set the y-axis to log scale
    ax.set_yscale("log")

    # Set the y-axis limits
    ax.set_ylim(1, 1000)

    # Set the title
    if title is not None:
        ax.set_title(title, fontsize = title_size, fontweight = 'bold')

    return fig, ax


if __name__ == "__main__":
    # Define the input arguments
    parser = ArgumentParser()
    parser.add_argument("--station", type=str, required=True, help="The station to plot")
    parser.add_argument("--on_threshold", type=float, required=True, help="The on threshold")
    parser.add_argument("--template_id", type=int, required=True, help="The template ID")

    args = parser.parse_args()
    station = args.station
    on_threshold = args.on_threshold
    tmpl_id = args.template_id

    # Load the STA/LTA detections
    filename = f"raw_sta_lta_detections_{station}_on{on_threshold:.1f}.h5"
    filepath = Path(dirpath) / filename
    snippets = Snippets.from_hdf(filepath) # This is a list of Snippet objects

    # Load the template matches
    filename = f"matches_{station}_template{tmpl_id}.h5"
    filepath = Path(dirpath) / filename
    template_matches = TemplateMatches.from_hdf(filepath)

    # Bin the STA/LTA detections and template matches by hour
    sta_lta_bin_count_df = snippets.bin_by_hour(starttime_bin, endtime_bin)
    matches_bin_count_df = template_matches.bin_matches_by_hour(starttime_bin, endtime_bin)

    # Plot the STA/LTA detections
    fig, ax = plot_num_det_per_hour(sta_lta_bin_count_df, title = station)
    
    # Add the day and night shading
    add_day_night_shading(ax)

    # Save the figure
    save_figure(fig, f"num_det_per_hour_{station}_on{on_threshold:.1f}_sta_lta.png")

    # Plot the STA/LTA detections and template matches
    fig, ax = plot_num_det_per_hour(sta_lta_bin_count_df,
                                    title = station, linecolor = "gray", label = "STA/LTA")
    
    # Plot the template matches
    plot_num_det_per_hour(matches_bin_count_df,
                          ax = ax,
                          linecolor = "dodgerblue", label = f"Template {tmpl_id}")
    
    # Add the day and night shading
    add_day_night_shading(ax)

    # Add the legend
    ax.legend()

    # Save the figure
    save_figure(fig, f"num_det_per_hour_{station}_on{on_threshold:.1f}_sta_lta_and_template{tmpl_id}.png")