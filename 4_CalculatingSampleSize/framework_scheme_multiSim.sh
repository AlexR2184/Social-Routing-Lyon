#!/bin/bash
#SBATCH --job-name=MultiSim
#SBATCH --output=../logs_Github/MultiSim.%A_%a.out # Output file (A is the job ID, a is the task ID)
#SBATCH --error=../logs_Github/MultiSim.%A_%a.err  # Error file
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=compute-p2
#SBATCH --account=research-tpm-ess

module load 2025
module load python

source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/alexroocroft/.conda/envs/env_symupy

# Run the Python script with the current value of t
srun python SocialRoutingSimulationFramework_multiple_simulations_modular.py 


# Deactivate conda environment
conda deactivate