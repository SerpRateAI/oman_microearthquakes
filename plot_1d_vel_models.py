"""
Plot the 1D Vp and Vs velocity models
"""

#-----------
# Imports
#-----------
from argparse import ArgumentParser
from pathlib import Path
from pandas import read_csv
from numpy import loadtxt
from matplotlib.pyplot import figure

from utils_basic import VEL_MODEL_DIR as dirpath
from utils_plot import save_figure, WAVE_VELOCITY_UNIT as unit

#-----------
# Argument parser
#-----------
parser = ArgumentParser()
parser.add_argument("--filename_vp", type=str, default="vp_1d.nd")
parser.add_argument("--filename_vs", type=str, default="vs_1d.nd")

parser.add_argument("--figwidth", type=float, default=10.0)
parser.add_argument("--figheight", type=float, default=10.0)
parser.add_argument("--vp_color", type=str, default="tab:blue")
parser.add_argument("--vs_color", type=str, default="tab:orange")
parser.add_argument("--vp_linewidth", type=float, default=2.0)
parser.add_argument("--vs_linewidth", type=float, default=2.0)
parser.add_argument("--min_depth", type=float, default=0.0)
parser.add_argument("--max_depth", type=float, default=30.0)


args = parser.parse_args()
filename_vp = args.filename_vp
filename_vs = args.filename_vs
min_freq_filter = args.min_freq_filter
max_freq_filter = args.max_freq_filter
figwidth = args.figwidth
figheight = args.figheight
vp_color = args.vp_color
vs_color = args.vs_color
vp_linewidth = args.vp_linewidth
vs_linewidth = args.vs_linewidth
min_depth = args.min_depth
max_depth = args.max_depth

#-----------
# Parameters
#-----------
filename_vp = "vp_1d.nd"
filename_vs = "vs_1d.nd"

#-----------
# Main
#-----------
model_vp = loadtxt(Path(dirpath) / filename_vp)
model_vs = loadtxt(Path(dirpath) / filename_vs)

model_vp[:, 0] *= 1000
model_vs[:, 0] *= 1000

model_vp[:, 1] *= 1000
model_vs[:, 1] *= 1000

model_vp[:, 2] *= 1000
model_vs[:, 2] *= 1000

#-----------
# Plot
#-----------
fig = figure(figsize=(figwidth, figheight))
ax = fig.add_subplot(111)

ax.plot(model_vp[:, 1], model_vp[:, 0], color=vp_color, linewidth=vp_linewidth, label="$V_p$")
ax.plot(model_vs[:, 2], model_vs[:, 0], color=vs_color, linewidth=vs_linewidth, label="$V_s$")

ax.set_ylim(min_depth, max_depth)
ax.set_xlabel(f"Velocity ({unit})")
ax.set_ylabel("Depth (m)")
ax.invert_yaxis()
ax.legend()

save_figure(fig, "vel_models_1d.png")

