#!/bin/bash
#SBATCH --job-name=SR_Base
#SBATCH --output=../logs_Github/SR_Base.%A_%a.out # Output file (A is the job ID, a is the task ID)
#SBATCH --error=../logs_Github/SR_Base.%A_%a.err  # Error file
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=compute-p2
#SBATCH --account=research-tpm-ess
#SBATCH --array=4

# Load modules:
module load 2023r1
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda environment
conda activate env_symupy


# Run the Python script with the current value of t
srun python 1_Generate_Baselines_parallel.py $SLURM_ARRAY_TASK_ID --unbuffered


# Deactivate conda environment
conda deactivate