#!/bin/bash
#SBATCH --job-name="Graph_Generate"
#SBATCH --output=../logs_Github/GraphGen.%A.out
#SBATCH --error=../logs_Github/GraphGen.%A.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=3968M
#SBATCH --partition=compute
#SBATCH --account=research-tpm-ess

# # move into this script's folder
# cd "$(dirname "$0")"

# -------------------------
# Load environment
# -------------------------
# module load 2023r1
# module load openmpi
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_symupy

# -------------------------
# User parameters
# -------------------------
# RUN_ID=8957007   # ← replace with the actual array job ID
RUN_ID=9140014

RESULTS_FOLDER="/scratch/alexroocroft/Output/Simulations_${RUN_ID}"
SIM_HOURS=4

echo "Generating graphs for sim_hours=$SIM_HOURS"

SCRIPT_DIR="/home/alexroocroft/GitHub_Version/6_Graph_Generate"

srun python "$SCRIPT_DIR/generate_graphs_from_extractor_optimised_v5.py" \
    --results-folder "$RESULTS_FOLDER" \
    --sim-hours "$SIM_HOURS" \
    --save-folder "$RESULTS_FOLDER/graphs_minmax"

conda deactivate
