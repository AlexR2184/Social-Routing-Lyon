#!/bin/bash
#SBATCH --job-name=SR_fixedrate_analysis
#SBATCH --output=../logs_Github/SR_fixedrate_analysis_%j.out
#SBATCH --error=../logs_Github/SR_fixedrate_analysis_%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3900MB
#SBATCH --time=00:30:00
#SBATCH --account=research-tpm-ess

set -eo pipefail
module purge
module load 2025
source ~/miniforge3/etc/profile.d/conda.sh
conda activate /home/alexroocroft/.conda/envs/env_symupy
set -u

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg

cd /home/alexroocroft/GitHub_Version/5_Framework_Simulation
mkdir -p ../logs_Github
PYTHON_BIN="/home/alexroocroft/.conda/envs/env_symupy/bin/python"

"${PYTHON_BIN}" -u analyse_fixed_rate_validation_130.py \
  --root ComputationalReq_fixed_rate_validation_130 \
  --reference-days 130 \
  --sample-size 15 \
  --next-size 16 \
  --n-random-samples 1000 \
  --seed 123
