#!/bin/bash

# Define the remote server and file paths
REMOTE_USER="tianze.liu"
REMOTE_SERVER="poseidon.whoi.edu"
REMOTE_PATH="/vortexfs1/scratch/tianze.liu/Oman_specfem3d/OUTPUT_FILES/*semv"
LOCAL_PATH="/Volumes/OmanData/geophones_no_prefilt/data/specfem_output"

# Copy files from the remote server to the local directory
scp "$REMOTE_USER@$REMOTE_SERVER:$REMOTE_PATH" "$LOCAL_PATH"

echo "Files copied successfully!"