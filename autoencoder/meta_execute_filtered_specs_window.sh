#!/bin/bash

# Initialize the variables
window=35
threshold=165

for i in {1..20}; do 
    sbatch execute_train__autoencoders.sh A01 $window $threshold
    window=$((window + 5)) 
    threshold=$((threshold + 25))
done
