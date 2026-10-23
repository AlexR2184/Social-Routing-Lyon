#!/bin/bash
#SBATCH --job-name=SR_GRAPHS
#SBATCH --output=/scratch/alexroocroft/logs_Scratch/SR_GRAPHS.%j.out
#SBATCH --error=/scratch/alexroocroft/logs_Scratch/SR_GRAPHS.%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3968M
#SBATCH --partition=compute
#SBATCH --account=research-tpm-ess

# Submit AFTER extracting a completed simulation job:
#   sbatch --export=ALL,RUN_ID=<simulation_job_id> framework_graphgenerate.sh
# Defaults to additional plots for the nine-run subset. To reproduce the
# complete paper figure set from full simulations, export GRAPH_MODE=all.
set -euo pipefail
module load 2025
PYTHON=/home/alexroocroft/.conda/envs/env_symupy/bin/python
if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Missing Python environment: $PYTHON" >&2
    exit 1
fi
: "${RUN_ID:?Set RUN_ID to the *simulation* Slurm array job ID}"
RESULTS_FOLDER="/scratch/alexroocroft/Output/Simulations_${RUN_ID}"
if [ ! -f "${RESULTS_FOLDER}/extracted_output/macro_all_sim4.csv" ]; then
    echo "[ERROR] Extractor output not found. Finish extraction first: $RESULTS_FOLDER" >&2
    exit 1
fi
# The bundled scripts are stored in 6_Graph_Generate. Submit from that folder
# so relative Python filenames resolve correctly on Slurm.
if [ ! -f "generate_all_graphs.py" ]; then
    echo "[ERROR] Submit this script from 6_Graph_Generate or change its working directory." >&2
    exit 1
fi
GRAPH_MODE="${GRAPH_MODE:-additional}"
if [[ "$GRAPH_MODE" != "additional" && "$GRAPH_MODE" != "all" && "$GRAPH_MODE" != "legacy" ]]; then
    echo "[ERROR] GRAPH_MODE must be additional, all, or legacy: $GRAPH_MODE" >&2
    exit 1
fi
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/sr_mpl_${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON" -c 'import numpy, pandas, matplotlib; print("[INFO] NumPy", numpy.__version__, "pandas", pandas.__version__, "matplotlib", matplotlib.__version__)'
"$PYTHON" -u generate_all_graphs.py \
    --results-folder "$RESULTS_FOLDER" \
    --sim-hours 4 \
    --save-folder graphs \
    --mode "$GRAPH_MODE"
