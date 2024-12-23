#!/bin/bash

# Command line variables (model variables)
window=$1
threshold=$2
epochs=$3
rate=$4
batch=$5

# Initial iterable variables
weight=60

for i in {1..20}; do 
    sbatch execute_train_autoencoder.sh "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch"
    weight=$((weight + 20)) 
done
