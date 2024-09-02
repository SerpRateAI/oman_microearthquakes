#!/bin/bash
#SBATCH --job-name=plot_all_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --account=ec332
#SBATCH --time=3:00:00
#SBATCH --output=autoencoder_script/%j.log

window=$1
threshold=$2
epochs=$3
weight=$4
rate=$5
batch=$6

echo $window
echo $threshold
echo $epochs
echo $weight
echo $rate
echo $batch

source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

python main_calculate_f1.py "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch"
