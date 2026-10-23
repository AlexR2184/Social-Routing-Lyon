#!/bin/bash
#SBATCH --job-name=SR_ALL
#SBATCH --output=/scratch/alexroocroft/logs_Scratch/SR_ALL.%A_%a.out
#SBATCH --error=/scratch/alexroocroft/logs_Scratch/SR_ALL.%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=compute
#SBATCH --account=research-tpm-ess
#SBATCH --array=1-117

set -euo pipefail
module load 2025

# Use the existing SymuVia Conda environment directly, without relying on
# shell activation (which can silently leave the wrong python in PATH).
PYTHON=/home/alexroocroft/.conda/envs/env_symupy/bin/python
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Python not found or not executable: $PYTHON" >&2
    exit 1
fi
"$PYTHON" -c 'import numpy; print("[INFO] NumPy", numpy.__version__)'

# ----------------------------
# Scheme dispatch logic: 1–54 Sacrifice, 55–117 CollectiveGood.
# CG tasks 115–117 map to local tasks 61–63 (case 20, repetitions 0–2).
# Run only the extra WP4 experiment with: sbatch --array=115-117 <this file>
# ----------------------------

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "$TASK_ID" -le 54 ]; then
    SCHEME="Sacrifice"
    LOCAL_TASK_ID=$TASK_ID
else
    SCHEME="CollectiveGood"
    LOCAL_TASK_ID=$((TASK_ID - 54))
fi

echo "[INFO] Global task ID: $TASK_ID"
echo "[INFO] Scheme: $SCHEME"
echo "[INFO] Local task ID: $LOCAL_TASK_ID"
echo "[INFO] Cycles: ${SR_NUM_CYCLES:-20}; route-day user detail from cycle: ${SR_ROUTE_DAY_DETAIL_START_CYCLE:-1}"

"$PYTHON" -u SocialRoutingSimulationFramework_Optimised.py \
    "$LOCAL_TASK_ID" \
    "$SCHEME" \
    4 \
    "${SR_NUM_CYCLES:-20}"

