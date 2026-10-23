#!/bin/bash
#SBATCH --job-name=SR_strict1530
#SBATCH --output=../logs_Github/SR_strict1530_%A_%a.out
#SBATCH --error=../logs_Github/SR_strict1530_%A_%a.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --time=14:00:00
#SBATCH --array=0-11
#SBATCH --account=research-tpm-ess

set -eo pipefail
module purge
module load 2025
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/alexroocroft/.conda/envs/env_symupy
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/alexroocroft/GitHub_Version/5_Framework_Simulation
SCRIPT="SocialRoutingSimulationFramework_sampling_sensitivity_strict.py"
[[ -f "$SCRIPT" ]] || { echo "Missing $SCRIPT"; exit 1; }
[[ -f ../Funcs/day_iteration_optimised.py ]] || { echo 'Missing optimised worker'; exit 1; }
[[ -f ../Funcs/TripsExtraction_SocialRout_module_v4_optimised.py ]] || { echo 'Missing extractor'; exit 1; }

# Array 0-2 Sacrifice/15; 3-5 Sacrifice/30;
#       6-8 CollectiveGood/15; 9-11 CollectiveGood/30.
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
if (( TASK_ID < 6 )); then SCHEME=Sacrifice; LOCAL_ID=$TASK_ID
else SCHEME=CollectiveGood; LOCAL_ID=$((TASK_ID - 6)); fi
if (( LOCAL_ID < 3 )); then DAYS=15; REP=$LOCAL_ID; EXPECTED_CORES=20
else DAYS=30; REP=$((LOCAL_ID - 3)); EXPECTED_CORES=35; fi
SELECTOR=$((REP + 1))
TOTAL_CYCLES=${NUMBER_OF_CYCLES:-15}
CYCLES_PER_JOB=${CYCLES_PER_JOB:-15}

if (( SLURM_CPUS_PER_TASK != EXPECTED_CORES )); then
    echo "ERROR: $DAYS-day run needs --cpus-per-task=$EXPECTED_CORES (requested $SLURM_CPUS_PER_TASK)" >&2
    exit 1
fi
if [[ "$TOTAL_CYCLES" != 15 && "$TOTAL_CYCLES" != 20 ]]; then
    echo 'NUMBER_OF_CYCLES must be 15 or 20' >&2; exit 1
fi

echo "Strict sensitivity: scheme=$SCHEME, rep=$REP, days=$DAYS, total_cycles=$TOTAL_CYCLES, cycles_this_job=$CYCLES_PER_JOB, cores=$SLURM_CPUS_PER_TASK"
srun /home/alexroocroft/.conda/envs/env_symupy/bin/python -u "$SCRIPT" \
    "$SELECTOR" "$SCHEME" 4 "$TOTAL_CYCLES" "$DAYS" "$SLURM_CPUS_PER_TASK" "$CYCLES_PER_JOB"
echo "Job segment finished: $(date)"
