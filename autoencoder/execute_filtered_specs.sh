#!/bin/bash
#SBATCH --job-name=plot_all_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --account=ec332
#SBATCH --time=3:00:00
#SBATCH --output=autoencoder_script/%j.log

station=$1
window=$2
threshold=$3

echo $station
echo $window
echo $threshold

source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

python main_filtered_specs.py "$station" "$window" "$threshold"
