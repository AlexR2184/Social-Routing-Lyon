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

# Load modules and environment
module load 2025
module load openmpi
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_symupy

# Run the extractor
# RUN_ID=8957007   # ← replace with the actual array job ID
RUN_ID=9071258

RESULTS_FOLDER="/scratch/alexroocroft/Output/Simulations_${RUN_ID}"
SIM_HOURS=4

echo "Running results extractor..."
srun python SocialRouting_results_extractor_optimised.py \
    --results-folder "$RESULTS_FOLDER" \
    --sim-hours "$SIM_HOURS" \
    --uc-prior-dir "/home/alexroocroft/GitHub_Version/3_Complete_SR_and_Baseline_UC" \
    --zip

conda deactivate
