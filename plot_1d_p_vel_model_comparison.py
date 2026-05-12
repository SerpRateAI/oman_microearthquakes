"""
Plot a comparison between:
1) Vp profile in vp_1d.nd
2) Linear Vp model V(z) = v0 + g*z estimated from hammer shots
"""

#-----------
# Imports
#-----------
from argparse import ArgumentParser
from pathlib import Path
from pandas import read_csv
from numpy import loadtxt, linspace
from matplotlib.pyplot import figure

from utils_basic import VEL_MODEL_DIR as dirpath
from utils_plot import save_figure, WAVE_VELOCITY_UNIT as unit

#-----------
# Argument parser
#-----------
parser = ArgumentParser()
parser.add_argument("--filename_vp", type=str, default="vp_1d_a.nd")
parser.add_argument("--filename_vel_params", type=str, default="vel_model_params_from_hammers.csv")

parser.add_argument("--figwidth", type=float, default=8.0)
parser.add_argument("--figheight", type=float, default=8.0)
parser.add_argument("--vp_linewidth", type=float, default=2.5)
parser.add_argument("--vp_linear_linewidth", type=float, default=2.5)
parser.add_argument("--num_depth_points", type=int, default=200)
parser.add_argument("--figname", type=str, default="vp_1d_model_comparison.png")
args = parser.parse_args()

filename_vp = args.filename_vp
filename_vel_params = args.filename_vel_params
figwidth = args.figwidth
figheight = args.figheight
vp_linewidth = args.vp_linewidth
vp_linear_linewidth = args.vp_linear_linewidth
num_depth_points = args.num_depth_points
figname = args.figname

#-----------
# Main
#-----------
# Load reference vp_1d.nd model:
# col 0 = depth (km), col 1 = Vp (km/s)
model_vp = loadtxt(Path(dirpath) / filename_vp)
depth_ref = model_vp[:, 0] * 1000.0
vp_ref = model_vp[:, 1] * 1000.0

# Load hammer-derived linear Vp parameters:
# V(z) = surface_vel + vel_gradient * z, with z in meters
vel_params_df = read_csv(Path(dirpath) / filename_vel_params)
surface_vel = float(vel_params_df.loc[0, "surface_vel"])
vel_gradient = float(vel_params_df.loc[0, "vel_gradient"])

depth_linear = linspace(depth_ref.min(), depth_ref.max(), num_depth_points)
vp_linear = surface_vel + vel_gradient * depth_linear

#-----------
# Plot
#-----------
fig = figure(figsize=(figwidth, figheight))
ax = fig.add_subplot(111)

ax.plot(
    vp_ref,
    depth_ref,
    color="tab:blue",
    linewidth=vp_linewidth,
    linestyle="--",
    label="Report",
)
ax.plot(
    vp_linear,
    depth_linear,
    color="tab:blue",
    linewidth=vp_linear_linewidth,
    label="Data",
)

ax.set_xlabel(f"$V_p$ ({unit})")
ax.set_ylabel("Depth (m)")
ax.set_ylim(0, depth_ref.max())
ax.invert_yaxis()
ax.legend()
ax.set_title("1D velocity model comparison", fontsize=14, fontweight="bold")

save_figure(fig, figname)

