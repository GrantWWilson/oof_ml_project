#!/bin/bash

#SBATCH -J toltec_analysis
#SBATCH -o toltec_analysis-%j.out   # Slurm output file
#SBATCH -t 12:00:00                 # Time limit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=toltec-cpu

# This script loops through data/toltec/fg* directories, running
# 04_test_toltec.py for each. Results go into each fg* dir.

# Hardcode your model path or pass via argument
MODEL_PATH="results_model/run_50_batches/models/zernike_predictor_final.keras"

echo "Starting TolTEC analysis job at: $(date)"

# Loop over each subdirectory matching fg*
for config_dir in data/toltec/fg*; do
    if [ -d "$config_dir" ]; then
        echo "--------------------------------------------------"
        echo "Processing config directory: $config_dir"
        echo "Output will go to: $config_dir"

        python scripts/04_test_toltec.py \
            --config-dir "$config_dir" \
            --model-path "$MODEL_PATH" \
            --output-dir "$config_dir"
    fi
done

echo "All directories processed. Job complete at: $(date)"
