#!/bin/bash
#SBATCH --job-name=MultiSimGraphs
#SBATCH --output=../logs_Github/MultiSimGraphs.%A.out
#SBATCH --error=../logs_Github/MultiSimGraphs.%A.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=compute-p2
#SBATCH --account=research-tpm-ess

# Use strict mode, but do not use nounset during conda activation because
# some activate.d scripts, including GDAL, may reference unset variables.
set -eo pipefail

module load 2025
module load python

source ~/miniforge3/etc/profile.d/conda.sh

# Avoid GDAL activate hook failing under nounset.
set +u
conda activate /home/alexroocroft/.conda/envs/env_symupy
set -u

# Use a non-interactive matplotlib backend for HPC plotting.
export MPLBACKEND=Agg

# Make sure the log directory exists relative to the submit directory.
mkdir -p ../logs_Github

RESULTS_DIR="ComputationalReq_multiple_simulations"
PICKLE_FILE="${RESULTS_DIR}/individual_users_output_parallel_multiSim_methodPaper_DTD_run_1.pickle"
SCRIPT_FILE="SocialRoutingSim_Graphgenerate_multiSim_modular_with_histogram.py"

if [[ ! -f "${SCRIPT_FILE}" ]]; then
    echo "ERROR: Cannot find ${SCRIPT_FILE} in the current directory."
    exit 1
fi

if [[ ! -f "${PICKLE_FILE}" ]]; then
    echo "ERROR: Cannot find ${PICKLE_FILE}."
    echo "Run SocialRoutingSimulationFramework_multiple_simulations_modular.py first."
    exit 1
fi

echo "Starting graph/post-processing job at $(date)"
echo "Using script: ${SCRIPT_FILE}"
echo "Using pickle: ${PICKLE_FILE}"

srun python "${SCRIPT_FILE}"

echo "Finished graph/post-processing job at $(date)"

set +u
conda deactivate
