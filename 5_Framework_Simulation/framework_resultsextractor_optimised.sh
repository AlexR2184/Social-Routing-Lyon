#!/bin/bash
#SBATCH --job-name="Extract_simu"
#SBATCH --output=/scratch/alexroocroft/logs_Scratch/Extract.%A_%a.out
#SBATCH --error=/scratch/alexroocroft/logs_Scratch/Extract.%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=compute
#SBATCH --account=research-tpm-ess
#SBATCH --array=1

# Use the exact environment that runs the simulation.
set -euo pipefail
module load 2025
module load openmpi
PYTHON=/home/alexroocroft/.conda/envs/env_symupy/bin/python
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Python not found or not executable: $PYTHON" >&2
    exit 1
fi
"$PYTHON" -c 'import numpy, pandas; print("[INFO] Python environment OK")'

# Example:
#   sbatch --export=ALL,RUN_ID=12345678 framework_resultsextractor_optimised.sh
# Set SR_ROUTE_DAY_AGGREGATE_USERS=1 only if you want a potentially multi-GB
# combined user-level CSV (each run already stores its own compressed CSVs).
: "${RUN_ID:?Please export RUN_ID=<simulation SLURM array job ID> when submitting this extractor}"
RESULTS_FOLDER="/scratch/alexroocroft/Output/Simulations_${RUN_ID}"
SIM_HOURS=4

ARGS=(
    --results-folder "$RESULTS_FOLDER"
    --sim-hours "$SIM_HOURS"
    --uc-prior-dir "/home/alexroocroft/GitHub_Version/3_Complete_SR_and_Baseline_UC"
    --route-day-mode required
    --route-day-expected-cycles "${SR_ROUTE_DAY_EXPECTED_CYCLES:-20}"
    --zip
)
if [ "${SR_ROUTE_DAY_AGGREGATE_USERS:-0}" = "1" ]; then
    ARGS+=(--route-day-include-users)
fi

echo "[INFO] Extracting all cases for SLURM run ID ${RUN_ID}"
srun "$PYTHON" SocialRouting_results_extractor_optimised.py "${ARGS[@]}"

