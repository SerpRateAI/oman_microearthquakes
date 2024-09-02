#!/bin/bash

# Command line variables (model variables)
epochs=$1
weight=$2
rate=$3
batch=$4

# Initial iterable variables
window=35
threshold=165

for i in {1..20}; do 
    sbatch execute_calculate_auc.sh "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch"
    window=$((window + 5)) 
    threshold=$((threshold + 25))
done
