"""
Cluster the popular detections using the similarity matrix based on the connected components of the graph.

The similarity matrix is a CSR matrix of shape (n_snippets, n_snippets) where each entry is 1 if the two snippets are similar and 0 otherwise.

The output is an array of labels of shape (n_snippets,) where each entry is the label of the cluster to which the snippet belongs.

The labels are integers starting from 0 and increasing by 1 for each new cluster.

"""

from argparse import ArgumentParser
from pathlib import Path
from numpy import abs, int32, full, where, asarray, linspace, amax
from scipy.sparse import csr_matrix, isspmatrix_csr, load_npz
from scipy.sparse.csgraph import connected_components
from matplotlib.pyplot import subplots

from utils_basic import DETECTION_DIR as dirpath, SAMPLING_RATE as sampling_rate, GEO_COMPONENTS as components
from utils_cluster import load_csr_from_hdf5, plot_similarity_matrix
from utils_plot import save_figure
from utils_sta_lta import Snippets
from utils_plot import get_geo_component_color, component2label

# ----------------------------------------------------------------------
# Core routines
# ----------------------------------------------------------------------

"""Filter the similarity matrix by the minimum degree"""
def filter_by_min_degree(sim_matrix: csr_matrix, min_degree: int, *, return_indices=False):
   
    if not isspmatrix_csr(sim_matrix):
        sim_matrix = sim_matrix.tocsr()

    degrees = asarray(sim_matrix.sum(axis=1)).ravel() - 1
    keep_mask = where(degrees >= min_degree)[0]
    if keep_mask.size == 0:
        raise ValueError("No snippets meet the min_degree criterion. Try lowering it.")

    filtered = sim_matrix[keep_mask, :][:, keep_mask].copy()
    print(f"Popular (>={min_degree}) snippets retained: {filtered.shape[0]}")

    if return_indices:
        return (filtered, keep_mask)
    else:
        return filtered

def cluster_snippets(sim_matrix: csr_matrix):
    """Return ``cluster_labels``, ``hub_indices`` and ``hub_degrees``."""

    # 4.1 Remove self‑loops so they don’t inflate degrees
    sim_matrix.setdiag(False)
    sim_matrix.eliminate_zeros()

    # 4.3 Connected‑components clustering (O(n + m))
    num_clusters, cluster_labels = connected_components(sim_matrix, directed=False)
    print(f"Found {num_clusters} clusters")

    # 4.4 Node degrees (number of similarity edges per snippet)
    degrees = sim_matrix.sum(axis=1).A1.astype(int32)   # shape: (n,)

    # 4.5 Pre‑allocate hub containers (faster & cleaner than incremental appends)
    hub_indices = full(num_clusters, -1, dtype=int32)   # −1 means “unset”
    hub_degrees = full(num_clusters,  0, dtype=int32)

    # 4.6 Fill hub info cluster‑by‑cluster
    for cluster_id in range(num_clusters):
        member_mask   = (cluster_labels == cluster_id)
        member_ids    = member_mask.nonzero()[0]
        if member_ids.size == 0:
            continue  # shouldn’t happen, but stay safe

        local_degrees   = degrees[member_ids]
        hub_pos_local   = local_degrees.argmax()      # position within member_ids
        hub_indices[cluster_id] = int32(member_ids[hub_pos_local])
        hub_degrees[cluster_id] = local_degrees[hub_pos_local]

    cluster_labels = cluster_labels.astype(int32)
    hub_indices = hub_indices.astype(int32)
    hub_degrees = hub_degrees.astype(int32)

    return cluster_labels, hub_indices, hub_degrees

def plot_template_waveforms(detections: Snippets,
                            figwidth = 15, figheight = 10,
                            linewidth = 1.0,
                            scale_factor = 0.3,
                            title = None):
    # Create the figure
    fig, axs = subplots(1, 3, figsize = (figwidth, figheight))

    # Plot the 3-component waveforms
    max_time = 0.0

    for i, detection in enumerate(detections):
        waveform_z = detection.waveform['Z']
        waveform_1 = detection.waveform['1']
        waveform_2 = detection.waveform['2']

        num_pts = detection.num_pts

        timeax = linspace(0, num_pts - 1, num_pts) / sampling_rate

        norm_factor = amax([amax(abs(waveform_z)), amax(abs(waveform_1)), amax(abs(waveform_2))])
        
        ax_z = axs[0]
        waveform_plot = waveform_z / norm_factor * scale_factor + i + 1
        color = get_geo_component_color("Z")
        ax_z.plot(timeax, waveform_plot , color = color, linewidth = linewidth)

        ax_1 = axs[1]
        waveform_plot = waveform_1 / norm_factor * scale_factor + i + 1
        color = get_geo_component_color("1")
        ax_1.plot(timeax, waveform_plot , color = color, linewidth = linewidth)

        ax_2 = axs[2]
        waveform_plot = waveform_2 / norm_factor * scale_factor + i + 1
        color = get_geo_component_color("2")
        ax_2.plot(timeax, waveform_plot , color = color, linewidth = linewidth)

        max_time = amax([max_time, amax(timeax)])

    for i, component in enumerate(components):
        # Set the x axis limits
        axs[i].set_xlim(0.0, max_time)

        # Set the y axis ticks and labels
        axs[i].set_yticks([])

        # Set the x axis labels
        axs[i].set_xlabel("Time (s)")

        # Set the subplot titles
        axs[i].set_title(component2label(component), fontsize = 14, fontweight = "bold")

    # Set the figure title
    if title:
        fig.suptitle(title, fontsize = 14, fontweight = "bold", y = 0.93)

    return fig, axs    


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------
if __name__ == "__main__":

    parser = ArgumentParser(description="Cluster the popular detections using the similarity matrix based on the connected components of the graph.")
    parser.add_argument("--station", type=str, help="Station name")
    parser.add_argument("--on_threshold", type=float, help="On threshold")
    parser.add_argument("--cc_threshold", type=float, help="CC threshold")
    parser.add_argument("--min_degree", type=int, help="Minimum degree")
    args = parser.parse_args()

    station = args.station
    on_threshold = args.on_threshold
    cc_threshold = args.cc_threshold
    min_degree = args.min_degree

    # Load the raw detections
    filename = f"raw_sta_lta_detections_{station}_on{on_threshold:.1f}.h5"
    filepath = Path(dirpath) / filename
    detections = Snippets.from_hdf(filepath, trim_pad = True)

    # Load the similarity matrix
    filename = f"similarity_matrix_{station}_on{on_threshold:.1f}_cc{cc_threshold:.2f}.npz"
    filepath = Path(dirpath) / filename
    sim_matrix = load_npz(filepath)

    # Filter the similarity matrix by the minimum degree
    sim_matrix_pop, pop_indices = filter_by_min_degree(sim_matrix, min_degree, return_indices=True)
    detections_pop = detections[pop_indices]

    # Plot the similarity matrix
    title = f"{station}"
    fig, ax = plot_similarity_matrix(sim_matrix_pop, title=title)
    figname = f"similarity_matrix_{station}_on{on_threshold:.1f}_cc{cc_threshold:.2f}_pop{min_degree}.png"
    save_figure(fig, figname)

    # Cluster the popular detections
    labels_pop, hub_indices, hub_degrees = cluster_snippets(sim_matrix_pop)
    print(len(labels_pop))

    # Find the hub detections
    hub_detections = detections_pop[hub_indices]
    hub_detections.set_sequential_ids()
    print(f"Hub detections: {hub_detections}")

    # Optional: quick summary of cluster sizes
    for cluster_id in range(labels_pop.max() + 1):
        size = (labels_pop == cluster_id).sum()
        hub_indice = hub_indices[cluster_id]
        hub_degree = hub_degrees[cluster_id]
        print(f"Cluster {cluster_id:>3}: {size} snippets")
        print(f"Hub index: {hub_indice}")
        print(f"Hub degree: {hub_degree}")

    # Save the hub detections
    filename = f"template_sta_lta_detections_{station}_on{on_threshold:.1f}_cc{cc_threshold:.2f}_pop{min_degree}.h5"
    filepath = Path(dirpath) / filename
    hub_detections.to_hdf(filepath)

    for detection in hub_detections:
        print(detection.num_pts)
        print(len(detection.waveform['Z']))

    # Plot the template waveforms
    fig, axs = plot_template_waveforms(hub_detections, title = f"{station}")
    figname = f"template_waveforms_{station}_on{on_threshold:.1f}_cc{cc_threshold:.2f}_pop{min_degree}.png"
    save_figure(fig, figname)