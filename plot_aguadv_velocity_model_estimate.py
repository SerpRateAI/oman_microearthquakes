"""
Plot hammer-shot average misfit in velocity-parameter space and the estimated
1D P-velocity model (AGU Advances).

Panel (a) reads ``avg_misfit_vs_vel_model_params.h5`` written by
``test_constrain_vel_model_w_hammers.py``. Panel (b) uses
``linear_vel_model_params_a.csv`` for the best-fit linear model
``v(z) = v0 + alpha * z``.
"""

from argparse import ArgumentParser
from pathlib import Path

from h5py import File
from matplotlib.pyplot import subplots
from matplotlib.ticker import MultipleLocator
from numpy import linspace
from pandas import read_csv

from utils_basic import VEL_MODEL_DIR as dirpath_vel
from utils_plot import WAVE_VELOCITY_UNIT, save_figure

DEFAULT_MISFIT_H5 = Path(dirpath_vel) / "avg_misfit_vs_vel_model_params.h5"
DEFAULT_VEL_PARAMS_CSV = Path(dirpath_vel) / "linear_vel_model_params_a.csv"


def load_avg_misfit(misfit_h5):
    """Load surface-velocity grid, velocity-gradient grid, and average misfit matrix."""
    if not misfit_h5.is_file():
        raise FileNotFoundError(
            f"Average misfit file not found: {misfit_h5}. "
            "Run test_constrain_vel_model_w_hammers.py to produce it."
        )

    with File(misfit_h5, "r") as h5file:
        surface_vel_grid = h5file["surface_vel"][:]
        vel_gradient_grid = h5file["vel_gradient"][:]
        avg_misfit_mat = h5file["avg_misfit"][:]

    return surface_vel_grid, vel_gradient_grid, avg_misfit_mat


def load_vel_model_params(vel_params_csv):
    """Load best-fit surface velocity and velocity gradient from the hammer search."""
    if not vel_params_csv.is_file():
        raise FileNotFoundError(
            f"Velocity-model parameter file not found: {vel_params_csv}. "
            "Run test_constrain_vel_model_w_hammers.py to produce it."
        )

    vel_params_df = read_csv(vel_params_csv)
    surface_vel = float(vel_params_df.loc[0, "surface_vel"])
    vel_gradient = float(vel_params_df.loc[0, "vel_gradient"])
    return surface_vel, vel_gradient


def velocity_profile(surface_vel, vel_gradient, depths):
    """v(z) = v0 + alpha * z with depth positive downward."""
    return surface_vel + vel_gradient * depths


def add_panel_label(ax, label):
    ax.text(
        -0.15,
        1.08,
        label,
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )


def plot_avg_misfit_matrix(
    ax,
    vel_gradient_grid,
    surface_vel_grid,
    avg_misfit_mat,
    vel_gradient_min,
    surface_vel_min,
    misfit_vmax,
):
    """Plot average misfit in (velocity gradient, surface velocity) space."""
    im = ax.pcolormesh(
        vel_gradient_grid,
        surface_vel_grid,
        avg_misfit_mat,
        cmap="binary",
        vmax=misfit_vmax,
    )
    ax.scatter(
        vel_gradient_min,
        surface_vel_min,
        color="crimson",
        marker="*",
        s=200,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    ax.set_xlabel("Velocity gradient (m s$^{-1}$ m$^{-1}$)")
    ax.set_ylabel("Surface velocity (m s$^{-1}$)")
    ax.set_title("Averaged arrival-time misfit", fontsize=14, fontweight="bold")
    ax.set_box_aspect(1)
    return im


def plot_estimated_velocity_model(ax, surface_vel, vel_gradient, max_depth):
    """Plot the hammer-estimated linear 1D P-velocity model."""
    depths = linspace(0.0, max_depth, 200)
    vp = velocity_profile(surface_vel, vel_gradient, depths)

    ax.plot(vp, depths, color="crimson", linewidth=2.0, zorder=2)
    ax.set_xlabel(f"Velocity ({WAVE_VELOCITY_UNIT})", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_ylim(max_depth, 0.0)
    ax.set_title("Estimated 1D P-velocity model", fontsize=14, fontweight="bold")
    ax.text(
        0.05,
        0.05,
        rf"$v_0$ = {surface_vel:.0f} {WAVE_VELOCITY_UNIT}"
        "\n"
        rf"$\alpha$ = {vel_gradient:.0f} m s$^{{-1}}$ m$^{{-1}}$",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
    )
    ax.set_box_aspect(1)


def main():
    parser = ArgumentParser(
        description=(
            "Plot hammer-shot average misfit and the estimated 1D P-velocity "
            "model (AGU Advances)."
        ),
    )
    parser.add_argument(
        "--misfit_h5",
        type=Path,
        default=DEFAULT_MISFIT_H5,
        help="HDF5 with surface_vel, vel_gradient, and avg_misfit",
    )
    parser.add_argument(
        "--vel_params_csv",
        type=Path,
        default=DEFAULT_VEL_PARAMS_CSV,
        help="CSV with best-fit surface_vel and vel_gradient",
    )
    parser.add_argument("--max_depth", type=float, default=30.0, help="Maximum depth (m)")
    parser.add_argument(
        "--misfit_vmax",
        type=float,
        default=3e-3,
        help="Color scale upper limit for average misfit (s)",
    )
    parser.add_argument("--figwidth", type=float, default=12.0, help="Figure width (in)")
    parser.add_argument("--figheight", type=float, default=6.0, help="Figure height (in)")
    parser.add_argument(
        "--figname",
        type=str,
        default="aguadv_velocity_model_estimate.png",
        help="Output figure filename",
    )
    args = parser.parse_args()

    misfit_h5 = args.misfit_h5
    vel_params_csv = args.vel_params_csv
    max_depth = args.max_depth
    misfit_vmax = args.misfit_vmax
    figwidth = args.figwidth
    figheight = args.figheight
    figname = args.figname

    print(f"Loading average misfit from {misfit_h5}...")
    surface_vel_grid, vel_gradient_grid, avg_misfit_mat = load_avg_misfit(misfit_h5)

    print(f"Loading velocity-model parameters from {vel_params_csv}...")
    surface_vel, vel_gradient = load_vel_model_params(vel_params_csv)
    print(f"Surface velocity: {surface_vel:.1f} m/s")
    print(f"Velocity gradient: {vel_gradient:.1f} 1/s")

    fig, axes = subplots(
        1,
        2,
        figsize=(figwidth, figheight),
        gridspec_kw={"wspace": 0.45},
    )
    ax_misfit, ax_vel = axes

    im = plot_avg_misfit_matrix(
        ax_misfit,
        vel_gradient_grid,
        surface_vel_grid,
        avg_misfit_mat,
        vel_gradient,
        surface_vel,
        misfit_vmax,
    )
    cax = ax_misfit.inset_axes([1.04, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Misfit (s)")
    cbar.locator = MultipleLocator(0.001)
    cbar.update_ticks()

    plot_estimated_velocity_model(ax_vel, surface_vel, vel_gradient, max_depth)

    add_panel_label(ax_misfit, "(a)")
    add_panel_label(ax_vel, "(b)")

    print(f"Saving figure to {figname}...")
    save_figure(fig, figname)


if __name__ == "__main__":
    main()
