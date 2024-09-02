import sys
from os.path import join
import os

from numpy import bool_, linspace, zeros
import numpy as np

from pandas import Timestamp
from pandas import date_range

from tqdm.notebook import tqdm
from time import time
from multiprocessing import Pool
import h5py

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import random

# Specify model to import
window = int(sys.argv[1])
threshold = int(sys.argv[2])
num_epochs = int(sys.argv[3])
weight = int(sys.argv[4])
learning_rate = float(sys.argv[5])
batch_size = int(sys.argv[6])

# Load binary spectrograms and convert to torch object
bin_specs_arr = np.load(f'spectrograms/bin_spec_{window}_{threshold}.npz')['spectrograms']
data = torch.tensor(bin_specs_arr, dtype=torch.float32)
data = data.unsqueeze(1)
full_dataset = TensorDataset(data, data)  # Targets are the same as inputs

# Define model architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 8, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, 2, kernel_size=5, stride=2, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent

# Instantiate the model
model = ConvAutoencoder()

# Load the trained model
model_path = f'models/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Prepare the model and set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Calculate the latent spaces and store to list
all_latent_spaces = []
with torch.no_grad():
    for batch in DataLoader(full_dataset, batch_size=1): 
        inputs, _ = batch
        inputs = inputs.to(device)
        _, latent = model(inputs)
        latent_space = latent.cpu().numpy()
        all_latent_spaces.append(latent_space)

# Convert the list to a NumPy array
all_latent_spaces = np.array(all_latent_spaces)
print(f'Shape of all_latent_spaces: {all_latent_spaces.shape}')

# Save the latent spaces to a .npz file
np.savez(f"encoded_latent_spaces/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}.npz", all_latent_spaces=all_latent_spaces)
print("Latent spaces saved to .npz file.")