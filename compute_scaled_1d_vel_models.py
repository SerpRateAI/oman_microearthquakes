"""
Scale the 1D velocity models.
"""

# -----------------------------------------------------------------------------
# Import libraries
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
from numpy import loadtxt, savetxt

from utils_basic import (
    VEL_MODEL_DIR as dirpath_vel
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument("--model_name", type = str, default = "vp_1d", help = "Model name")
parser.add_argument("--scale_factors", type = float, nargs = "+", default = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], help = "Scale factors")

args = parser.parse_args()
scale_factors = args.scale_factors
model_name = args.model_name

# -----------------------------------------------------------------------------
# Load the velocity models
# -----------------------------------------------------------------------------

vel_path = Path(dirpath_vel) / f"{model_name}.nd"
vel_model = loadtxt(vel_path)

# -----------------------------------------------------------------------------
# Sacle the velocity model and save the scaled velocity models
# -----------------------------------------------------------------------------

for scale_factor in scale_factors:
    vel_model_scaled = vel_model.copy()
    vel_model_scaled[:, 1:] = vel_model[:, 1:] * scale_factor

    # Save the scaled velocity model
    filename_scaled = f"{model_name}_scale{scale_factor:.1f}.nd"
    filepath_scaled = vel_path.with_name(filename_scaled)
    savetxt(filepath_scaled, vel_model_scaled, delimiter = " ", fmt = "%.4f")

    print(f"Saved the scaled velocity model for scale factor {scale_factor}.")



