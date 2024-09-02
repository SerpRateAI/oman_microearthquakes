import sys
import os

import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Command-line arguments
if len(sys.argv) == 7:
    station = 'A01'
else:
    station = sys.argv[7]

window = int(sys.argv[1])
threshold = int(sys.argv[2])
num_epochs = int(sys.argv[3])
weight = int(sys.argv[4])
learning_rate = float(sys.argv[5])
batch_size = int(sys.argv[6])

# Load and reshape latent spaces
latent_spaces_path = f"encoded_latent_spaces/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}.npz"
latent_data = np.load(latent_spaces_path)
latent_spaces = latent_data['all_latent_spaces']
latent_spaces_flattened = latent_spaces.reshape(latent_spaces.shape[0], -1)

# Apply TSNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca')
latent_tsne = tsne.fit_transform(latent_spaces_flattened)

# Collect losses from k = 1 to k = 50 for kmeans
loss_list = []
for i in range(1,51):
    n_clusters = i
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_tsne)
    kmeans_loss = kmeans.inertia_
    loss_list.append(kmeans_loss)

# Plot Loss Graph
plt.figure(figsize=(8, 6))
plt.plot(range(1,51), loss_list, marker='o')
plt.xlabel('k')
plt.ylabel('loss')
plt.title('k means loss')

# Saving the figure
plt.savefig(f"optimal_cluster_plots/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}.png")