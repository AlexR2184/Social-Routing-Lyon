#!/bin/bash
#SBATCH --job-name="Extract_simu"
#SBATCH --output=../logs_Github/Extract.%A_%a.out # Output file (A is the job ID, a is the task ID)
#SBATCH --error=../logs_Github/Extract.%A_%a.err  # Error file
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute-p2
#SBATCH --account=research-tpm-ess
#SBATCH --array=1

# Load modules:
module load 2023r1
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate env_symupy
srun python SocialRoutingSimulationFramework_results_extraction_Github.py $SLURM_ARRAY_TASK_ID

conda deactivate