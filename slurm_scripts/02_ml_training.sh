#!/bin/bash
# -------------------------------------------------------------------------
#  Slurm driver for scripts/02_train_model.py
#  Optional positional args (with sensible defaults):
#    1  DATASET_PATH
#    2  WORKING_DIR
#    3  BAND_FILTER
#    4  BATCH_SIZE
#    5  EPOCHS
#    6  LEARNING_RATE
#    7  SHIFT_PIXELS
#    8  MAX_BATCH_NUM
#    9  VAL_FRACTION
#   10  TEST_FRACTION
#   11  BASIS_OVERSAMPLE
#   12  BASIS_LOSS_WEIGHT
#   13  CURRICULUM_EPOCHS
#
#  Example:
#    sbatch 02_train_model_slurm.sh \
#           data/synthetic_45m/train results_model a2000 64 80 5e‑4 3.0 300 \
#           0.15 0.1 6 8.0 30
# -------------------------------------------------------------------------

#SBATCH -J zernike_train
#SBATCH -o zernike_train-%j.out
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=toltec-cpu

# ----------------------------- ARGUMENTS ----------------------------------
DATASET_PATH=${1:-"data/synthetic_45m/train"}
WORKING_DIR=${2:-"results_model"}
BAND_FILTER=${3:-"a2000"}
BATCH_SIZE=${4:-32}
EPOCHS=${5:-100}
LEARNING_RATE=${6:-0.001}
SHIFT_PIXELS=${7:-2.5}
MAX_BATCH_NUM=${8:-50}
VAL_FRACTION=${9:-0.20}
TEST_FRACTION=${10:-0.10}

BASIS_OVERSAMPLE=${11:-4}
BASIS_LOSS_WEIGHT=${12:-5.0}
CURRICULUM_EPOCHS=${13:-20}

# ----------------------------- LOG HEAD -----------------------------------
echo "================================================================"
echo "Job started : $(date)"
echo "Job name    : $SLURM_JOB_NAME"
echo "Job ID      : $SLURM_JOB_ID"
echo "----------------------------------------------------------------"
echo "Dataset path        : $DATASET_PATH"
echo "Working dir         : $WORKING_DIR"
echo "Band filter         : $BAND_FILTER"
echo "Batch size          : $BATCH_SIZE"
echo "Epochs              : $EPOCHS"
echo "Learning‑rate       : $LEARNING_RATE"
echo "Shift‑pixels        : $SHIFT_PIXELS"
echo "Max batch num       : $MAX_BATCH_NUM"
echo "Val / Test frac     : $VAL_FRACTION  /  $TEST_FRACTION"
echo "Basis oversample    : $BASIS_OVERSAMPLE"
echo "Basis loss‑weight   : $BASIS_LOSS_WEIGHT"
echo "Curriculum epochs   : $CURRICULUM_EPOCHS"
echo "================================================================"
echo

# ----------------------------- TRAIN --------------------------------------
python scripts/02_train_model.py \
  --dataset-path      "$DATASET_PATH" \
  --working-dir       "$WORKING_DIR"  \
  --band-filter       "$BAND_FILTER"  \
  --batch-size        "$BATCH_SIZE"   \
  --epochs            "$EPOCHS"       \
  --learning-rate     "$LEARNING_RATE"\
  --shift-pixels      "$SHIFT_PIXELS" \
  --max-batch-num     "$MAX_BATCH_NUM"\
  --val-fraction      "$VAL_FRACTION" \
  --test-fraction     "$TEST_FRACTION"\
  --basis-oversample  "$BASIS_OVERSAMPLE" \
  --basis-loss-weight "$BASIS_LOSS_WEIGHT"\
  --curriculum-epochs "$CURRICULUM_EPOCHS"

echo
echo "================================================================"
echo "Job finished : $(date)"
echo "================================================================"
