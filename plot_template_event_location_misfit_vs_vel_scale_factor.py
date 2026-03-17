"""
Plot the RMS vs. velocity scale factor for a template event.
"""

#-----------
# Imports
#-----------
from argparse import ArgumentParser
from pathlib import Path
from numpy import loadtxt
from pandas import read_csv
from matplotlib.pyplot import subplots

from utils_basic import LOC_DIR as dirpath, VEL_MODEL_DIR as dirpath_vel_model
from utils_basic import get_freq_limits_string
from utils_plot import save_figure, WAVE_VELOCITY_UNIT as unit

#-----------
# Helper functions
#-----------
def reformat_vel_models(model_vp, model_vs, scale_factor):
    model_vp[:, 0] *= 1000
    model_vs[:, 0] *= 1000
    model_vp[:, 1] *= 1000
    model_vs[:, 1] *= 1000
    model_vp[:, 2] *= 1000
    model_vs[:, 2] *= 1000

def plot_1d_vel_models(ax, model_vp, model_vs, 
                       vp_linewidth = 2.0, vs_linewidth = 2.0, scale_factor = None):

    ax.plot(model_vp[:, 1], model_vp[:, 0], color="tab:blue", linewidth=vp_linewidth, label="$V_p$")
    ax.plot(model_vs[:, 2], model_vs[:, 0], color="tab:orange", linewidth=vs_linewidth, label="$V_s$")

    if scale_factor is not None:
        model_vp_scaled = model_vp.copy()
        model_vp_scaled[:, 1] *= scale_factor

        ax.plot(model_vp_scaled[:, 1], model_vp_scaled[:, 0], color="tab:blue", linewidth=vp_linewidth, linestyle="--", label=f"{scale_factor:.1f} $V_p$")

    ax.set_ylim(0, 30.0)
    ax.set_xlabel(f"Velocity ({unit})")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.legend()

    return ax

#-----------
# Argument parser
#-----------
parser = ArgumentParser()
parser.add_argument("--template_id", type=str, required=True)
parser.add_argument("--subarray", type=str, required=True)
parser.add_argument("--arrival_type", type=str, default="manual_stack")
parser.add_argument("--min_freq_filter", type=float, default=20.0)
parser.add_argument("--max_freq_filter", type=float, default=200.0)
parser.add_argument("--figwidth", type=float, default=15.0)
parser.add_argument("--figheight", type=float, default=10.0)

args = parser.parse_args()
template_id = args.template_id
subarray = args.subarray
arrival_type = args.arrival_type
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
figwidth = args.figwidth
figheight = args.figheight

#-----------
# Main
#-----------

# Read the location information
freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
filename = f"location_info_template_{template_id}_{freq_str}.csv"
filepath = Path(dirpath) / filename
location_df = read_csv(filepath)
location_df = location_df[location_df["arrival_type"] == arrival_type]

scale_factor_min_misfit = location_df["scale_factor"][location_df["misfit"] == location_df["misfit"].min()].values[0]
print(f"Scale factor for the minimum misfit: {scale_factor_min_misfit:.1f}")

# Read the velocity models
filename_vp = f"vp_1d_{subarray.lower()}.nd"
filename_vs = f"vs_1d_{subarray.lower()}.nd"
model_vp = loadtxt(Path(dirpath_vel_model) / filename_vp)
model_vs = loadtxt(Path(dirpath_vel_model) / filename_vs)

reformat_vel_models(model_vp, model_vs, scale_factor_min_misfit)

# Generate the figure
fig, axs = subplots(1, 2, figsize=(figwidth, figheight))

# Plot the RMS vs. velocity scale factor
ax = axs[0]
im =ax.scatter(location_df["scale_factor"], 1000 * location_df["misfit"], s=50.0, c=location_df["depth"], cmap="plasma")
ax.set_xlabel("Scale factor for $V_p$")
ax.set_ylabel("Weighted misfit (ms)")

cax = ax.inset_axes([0.05, 0.7, 0.03, 0.25])
fig.colorbar(im, cax=cax, label="Depth (m)")

# Plot the velocity models
ax = axs[1]
plot_1d_vel_models(ax, model_vp, model_vs, scale_factor = scale_factor_min_misfit)

# Add the super title
fig.suptitle(f"Template event {template_id}, Subarray {subarray}, {min_freq_filter:.0f}-{max_freq_filter:.0f} Hz", fontsize=14, fontweight="bold", y=0.95)

# Save the figure
save_figure(fig, f"template_event_location_misfit_vs_vel_scale_factor_{template_id}_{freq_str}_{arrival_type}.png")