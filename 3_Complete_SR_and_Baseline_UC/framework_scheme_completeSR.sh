#!/bin/bash
#SBATCH --job-name=SRschemeSO
#SBATCH --output=../logs_Github/SRschemeSO.%A_%a.out # Output file (A is the job ID, a is the task ID)
#SBATCH --error=../logs_Github/SRschemeSO.%A_%a.err  # Error file
#SBATCH --time=05:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=2G
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

# Activate conda environment
conda activate env_symupy


# Run the Python script with the current value of t
srun python 1_SocialRoutingSimulationFramework_completeSR_modular.py $SLURM_ARRAY_TASK_ID --unbuffered


# Deactivate conda environment
conda deactivate