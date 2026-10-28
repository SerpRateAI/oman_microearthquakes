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
from numpy import arange, argsort, asarray, int8, unique
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


_BAR_EDGEWIDTH = 2.0
_BAR_EDGE_DOTTED = (0, (2, 2))


def _apply_group_bar_outlines(patches, labels_sorted, highlight_labels):
    highlight = set() if highlight_labels is None else set(highlight_labels)
    for patch, label in zip(patches, labels_sorted):
        patch.set_edgecolor("black")
        patch.set_linewidth(_BAR_EDGEWIDTH)
        patch.set_linestyle("-" if label in highlight else _BAR_EDGE_DOTTED)


def plot_label_histogram(
    labels,
    min_count=10,
    subarrays=None,
    highlight_labels=None,
    color_a="tab:blue",
    color_b="tab:orange",
):
    """
    Plot event counts per group with more than ``min_count`` members, sorted by
    count in descending order, with a logarithmic y-axis.

    If ``subarrays`` is given (same length as ``labels``, values ``"A"`` or ``"B"``),
    each bar is stacked: Subarray A in ``color_a``, Subarray B in ``color_b``.

    If ``highlight_labels`` is given, bars for those group labels have a solid black
    outline; all other bars have a dotted black outline.
    """
    labels_unique, counts = unique(labels, return_counts=True)
    mask = counts >= min_count
    labels_unique = labels_unique[mask]
    counts = counts[mask]

    order = argsort(-counts)
    labels_sorted = labels_unique[order]
    counts_sorted = counts[order]

    fig = figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    x = list(range(len(counts_sorted)))
    if len(counts_sorted) > 0:
        if subarrays is not None:
            subarrays = asarray(subarrays)
            counts_a = []
            counts_b = []
            for label in labels_sorted:
                in_group = labels == label
                counts_a.append((subarrays[in_group] == "A").sum())
                counts_b.append((subarrays[in_group] == "B").sum())
            counts_a = asarray(counts_a)
            counts_b = asarray(counts_b)
            bars_a = ax.bar(
                x,
                counts_a,
                align="center",
                color=color_a,
                linewidth=_BAR_EDGEWIDTH,
                edgecolor="black",
                label="Subarray A",
            )
            bars_b = ax.bar(
                x,
                counts_b,
                bottom=counts_a,
                align="center",
                color=color_b,
                linewidth=_BAR_EDGEWIDTH,
                edgecolor="black",
                label="Subarray B",
            )
            _apply_group_bar_outlines(bars_a.patches, labels_sorted, highlight_labels)
            _apply_group_bar_outlines(bars_b.patches, labels_sorted, highlight_labels)
            ax.legend(loc="upper right")
        else:
            bars = ax.bar(
                x,
                counts_sorted,
                align="center",
                color="tab:blue",
                linewidth=_BAR_EDGEWIDTH,
                edgecolor="black",
            )
            _apply_group_bar_outlines(bars.patches, labels_sorted, highlight_labels)
        ax.set_yscale("log")
    else:
        ax.text(
            0.5,
            0.5,
            f"No groups with more than {min_count} events",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

    ax.set_xlabel("Group label", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Number of events per group",
        fontsize=14,
        fontweight="bold",
    )

    if len(counts_sorted) > 0:
        ax.set_xticks(x)
        ax.set_xticklabels(labels_sorted)

        ax.set_xlim(-0.5, len(counts_sorted) - 0.5)
        ax.set_ylim(min_count * 0.9, counts_sorted.max() * 1.1)

    return fig, ax