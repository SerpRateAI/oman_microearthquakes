#!/bin/bash

# Command line variables (model variables)
window=$1
threshold=$2
epochs=$3
weight=$4
rate=$5
batch=$6


for i in {1..8}; do 
    sbatch execute_extract_latent_spaces.sh "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch" "$i"
done
