import os
from scipy.ndimage import convolve
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc
import seaborn as sns
from datetime import datetime, timedelta
import sys
import pandas as pd

# Command Line Arguments
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

# Run model and find loss
all_targets = []
all_outputs = []
with torch.no_grad():
    for batch in DataLoader(full_dataset, batch_size=1):
        inputs, targets = batch
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        all_targets.append(targets.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())

all_targets = np.concatenate(all_targets, axis=0)
all_outputs = np.concatenate(all_outputs, axis=0)

# Calculate AUC
fpr, tpr, _ = roc_curve(all_targets.flatten(), all_outputs.flatten())
roc_auc = auc(fpr, tpr)

# Write auc to csv
new_data = {
    'auc':  [roc_auc],
    'window':  [window],
    'threshold': [threshold],
    'epochs': [num_epochs],
    'weight': [weight],
    'rate': [learning_rate],
    'batch': [batch_size]
}
new_data = pd.DataFrame(new_data)
filename_out = 'diagnostics/auc_vals.csv'

try:
    file_empty = pd.read_csv(filename_out).empty
except FileNotFoundError:
    file_empty = True 
    
new_data.to_csv(filename_out, mode='a', header=file_empty, index=False)

print("New data has been entered")