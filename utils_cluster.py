"""
This module contains functions for classifying detections.
"""

import numpy as np
import pandas as pd
from h5py import File
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from pathlib import Path
from scipy.sparse import isspmatrix_csr
from matplotlib.pyplot import figure
from numpy import arange, int8
from scipy.sparse import spmatrix

# -----------------------------------------------------------------------------
# I/O of the similarity matrix
# -----------------------------------------------------------------------------

def save_csr_to_hdf5(matrix: csr_matrix, hdf_path: str | Path, permission: str = "a"):
    """Write a SciPy CSR Boolean matrix to *hdf_path* (HDF5)."""
    """The matrix is by default appended to an existing file storing the waveform snippets and their timing information"""

    if permission not in ["w", "a"]:
        raise ValueError("Permission must be 'w' or 'a'.")

    if not isspmatrix_csr(matrix):
        raise TypeError("Input must be CSR format.")

    with File(hdf_path, permission) as f:
        f.attrs["format"] = "csr_bool_v1"
        f.attrs["shape"] = matrix.shape
        # store components
        f.create_dataset("data",    data=matrix.data.astype("uint8"), compression="gzip")
        f.create_dataset("indices", data=matrix.indices, dtype="int32", compression="gzip")
        f.create_dataset("indptr",  data=matrix.indptr,  dtype="int32", compression="gzip")
    print(f"Sparse matrix written to {hdf_path}")


def load_csr_from_hdf5(hdf_path: str | Path) -> csr_matrix:
    """Load a CSR Boolean matrix saved by `save_csr_to_hdf5`."""
    with File(hdf_path, "r") as f:
        if f.attrs.get("format") != "csr_bool_v1":
            raise ValueError("Unrecognised HDF5 sparse format.")
        shape = tuple(f.attrs["shape"])
        data    = f["data"][:].astype(bool)
        indices = f["indices"][:]
        indptr  = f["indptr"][:]

        matrix = csr_matrix((data, indices, indptr), shape=shape)
    return matrix

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_similarity_matrix(sim_matrix: spmatrix, *, downsample=None, title=None):
    if downsample is not None and downsample > 1:
        idx = arange(0, sim_matrix.shape[0], downsample)
        vis = sim_matrix[idx[:, None], idx]
    else:
        vis = sim_matrix

    img = vis.toarray().astype(int8)
    fig = figure(figsize=(10, 10))
    ax = fig.subplots(1, 1)
    ax.imshow(img, cmap="gray", interpolation="nearest")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Snippet index", fontsize=12)
    ax.set_ylabel("Snippet index", fontsize=12)

    return fig, ax