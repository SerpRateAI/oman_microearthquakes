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


# Import Internal Modules 
sys.path.append(os.path.abspath('..'))

from utils_basic import SPECTROGRAM_DIR as indir
from utils_plot import plot_geo_total_psd_to_bin_array_spectrogram, save_figure

# Initialize file path constants
window = int(sys.argv[1])
threshold = int(sys.argv[2])

# Set hyperparameters
if len(sys.argv) == 7:
    num_epochs = int(sys.argv[3])
    weight = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    batch_size = int(sys.argv[6])
else:
    num_epochs = 800
    weight = 100
    learning_rate = .0015
    batch_size = 128

# Setting seed for reproducibility
seed = 42 
generator = torch.Generator().manual_seed(seed)

# Load binary spectrograms and convert to torch object
bin_specs_arr = np.load(f'spectrograms/bin_spec_{window}_{threshold}.npz')['spectrograms']
data = torch.tensor(bin_specs_arr, dtype=torch.float32)
data = data.unsqueeze(1)
full_dataset = TensorDataset(data, data)  # Targets are the same as inputs

# Get training data
train_size = .8
val_size = .2
train_data, val_data = random_split(full_dataset, [train_size, val_size]) # Only use train data

# Prepare data for pytorch use
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=16)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=16)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(16, 8, kernel_size=5, stride=2, padding=2), 
            nn.Conv2d(8, 8, kernel_size=5, padding=2), 
            nn.ReLU(True),
            nn.Conv2d(8, 2, kernel_size=5, stride=2, padding=2),
        )
        
        # Decoder
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

# Define biased loss function
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=1.0):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, outputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(outputs, targets)
        
        # Apply weights
        pos_weight = torch.where(targets == 1, self.pos_weight, torch.tensor(1.0, device=outputs.device))
        neg_weight = torch.where(targets == 0, self.neg_weight, torch.tensor(1.0, device=outputs.device))

        # Recalculate loss
        weighted_loss = bce_loss * (pos_weight + neg_weight)
        return weighted_loss.mean()

# Initialize criterion for loss function
pos_weight = weight # penalty weight for false negatives
neg_weight = 1.0   # Penalty weight for false positives
biased_criterion = WeightedBinaryCrossEntropyLoss(pos_weight=pos_weight, neg_weight=neg_weight)
unbiased_criterion = WeightedBinaryCrossEntropyLoss(pos_weight=pos_weight, neg_weight=neg_weight)

# Initialize model and optimizer
model = ConvAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize Loss Lists
train_loss_list = []
val_loss_list = []

# Train the autoencoder and store loss
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs, latent = model(inputs)
        loss = biased_criterion(outputs, targets)

        # Calculate using unbiased criterion for storage
        unbiased_loss = unbiased_criterion(outputs, targets)
        epoch_loss += unbiased_loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate and store the average loss for the epoch
    avg_loss = epoch_loss / len(train_dataloader)
    train_loss_list.append(avg_loss)
print("Training complete.")

# Save Model
model_save_path = f'models/file_{window}_{threshold}_model_{num_epochs}_{pos_weight}_{learning_rate}_{batch_size}.pth'
torch.save(model.state_dict(), model_save_path)

# Find loss for validation data
model.eval()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in val_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, latent = model(inputs)
        unbiased_loss = unbiased_criterion(outputs, targets)
        epoch_loss += unbiased_loss.item()
    avg_loss = epoch_loss / len(val_dataloader)
    val_loss_list.append(avg_loss)
print("Validation Complete")

np.savez(f"model_loss/file_{window}_{threshold}_model_{num_epochs}_{pos_weight}_{learning_rate}_{batch_size}.npz", train_loss=np.array(train_loss_list), val_loss= np.array(val_loss_list))