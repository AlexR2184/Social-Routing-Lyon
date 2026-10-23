#!/bin/bash
#SBATCH --job-name=SR_fixedrate130
#SBATCH --output=../logs_Github/SR_fixedrate130_%A_%a.out
#SBATCH --error=../logs_Github/SR_fixedrate130_%A_%a.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --account=research-tpm-ess

set -eo pipefail
module purge
module load 2025
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/alexroocroft/.conda/envs/env_symupy
set -u

PYTHON_BIN="/home/alexroocroft/.conda/envs/env_symupy/bin/python"
cd /home/alexroocroft/GitHub_Version/5_Framework_Simulation

mkdir -p ../logs_Github
mkdir -p /scratch/alexroocroft/ComputationalReq_fixed_rate_validation_130

SCRIPT="SocialRoutingSimulationFramework_fixed_rate_validation_130.py"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN"; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Missing script: $SCRIPT"; exit 1; }
[[ -f ../Funcs/day_iteration_optimised.py ]] || { echo "Missing optimised worker"; exit 1; }
[[ -f ../Funcs/TripsExtraction_SocialRout_module_v4_optimised.py ]] || { echo "Missing extractor"; exit 1; }
[[ -f ../Funcs/RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v5_optimised_gridlockabort.py ]] || { echo "Missing existing gridlock-abort runner"; exit 1; }

case "${SLURM_ARRAY_TASK_ID}" in
  0) RATE=20 ;;
  1) RATE=50 ;;
  2) RATE=80 ;;
  *) echo "Unexpected array index: ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

# Work in small batches so checkpoints are safely written before the
# four-hour job wall-time. Resubmit this job array until all reach 130.
ROUNDS_PER_JOB="${ROUNDS_PER_JOB:-2}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=============================================="
echo "Fixed-rate sampling validation: ${RATE}%"
echo "Target: 130 successful daily simulations"
echo "Batches this job: ${ROUNDS_PER_JOB} (20 attempts per batch)"
echo "Node: $(hostname)  Start: $(date)"
echo "=============================================="

srun "${PYTHON_BIN}" -u "${SCRIPT}" \
  --participation-percent "${RATE}" \
  --total-sample-size 130 \
  --rounds-per-job "${ROUNDS_PER_JOB}" \
  --batch-timeout-minutes 80 \
  --parallel-cores "${SLURM_CPUS_PER_TASK}" \
  --sim-hours 4 \
  --seed 20260904 \
  --participant-seed 20261008 \
  --output-root "ComputationalReq_fixed_rate_validation_130" \
  --scratch-root "/scratch/alexroocroft/ComputationalReq_fixed_rate_validation_130"

echo "Job segment finished: $(date)"
