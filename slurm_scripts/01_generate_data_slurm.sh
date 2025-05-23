#!/bin/bash

# ----------------------------------------------------------------
# A Slurm job script that takes up to three optional arguments:
#   1) first_batch
#   2) total_batches
#   3) batch_size
# 
# It then redirects output to a file of the form: gw-{first_batch}-{last_batch}.log
# ----------------------------------------------------------------

#SBATCH -J preconditioner
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=toltec-cpu
#SBATCH --parsable

# --- Provide defaults if no arguments are passed:
FIRST_BATCH=${1:-1}
TOTAL_BATCHES=${2:-100}
BATCH_SIZE=${3:-1000}

# Compute the last batch index (assuming inclusive range)
LAST_BATCH=$(( FIRST_BATCH + TOTAL_BATCHES - 1 ))

# Dynamically set the log file name
LOG_FILE="logs/gw-${FIRST_BATCH}-${LAST_BATCH}.log"

# Force all stdout and stderr to go into that file
exec > "${LOG_FILE}" 2>&1

echo "================================================================"
echo "Slurm job started at:  $(date)"
echo "Job Name:              ${SLURM_JOB_NAME}"
echo "Job ID:                ${SLURM_JOB_ID}"
echo "First batch:           ${FIRST_BATCH}"
echo "Total batches:         ${TOTAL_BATCHES}"
echo "Last batch:            ${LAST_BATCH}"
echo "Batch size:            ${BATCH_SIZE}"
echo "Log file:              ${LOG_FILE}"
echo "================================================================"

# Run your Python script, passing the arguments
python scripts/01_generate_data.py \
    --first-batch "${FIRST_BATCH}" \
    --total-batches "${TOTAL_BATCHES}" \
    --batch-size "${BATCH_SIZE}" \
    --output-dir "data/synthetic_45m"

echo "================================================================"
echo "Slurm job finished at: $(date)"
echo "================================================================"
