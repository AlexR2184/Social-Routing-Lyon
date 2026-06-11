#!/bin/bash
#SBATCH --job-name=SR_ALL
#SBATCH --output=/scratch/alexroocroft/logs_Scratch/SR_ALL.%A_%a.out
#SBATCH --error=/scratch/alexroocroft/logs_Scratch/SR_ALL.%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=compute
#SBATCH --account=research-tpm-ess
#SBATCH --array=1-114

module load 2025
module load python

source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/alexroocroft/.conda/envs/env_symupy

# ----------------------------
# Scheme dispatch logic
# ----------------------------

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "$TASK_ID" -le 54 ]; then
    SCHEME="Sacrifice"
    LOCAL_TASK_ID=$TASK_ID
else
    SCHEME="CollectiveGood"
    LOCAL_TASK_ID=$((TASK_ID - 54))
fi

echo "[INFO] Global task ID: $TASK_ID"
echo "[INFO] Scheme: $SCHEME"
echo "[INFO] Local task ID: $LOCAL_TASK_ID"

python -u SocialRoutingSimulationFramework_Optimised.py \
    $LOCAL_TASK_ID \
    $SCHEME \
    4 \
    20

