# Social Routing: strict 15-versus-30 daily-sample sensitivity experiment

Independent of the original published/main Social Routing results.

## Files
Upload all THREE files into /home/alexroocroft/GitHub_Version/5_Framework_Simulation:
  SocialRoutingSimulationFramework_sampling_sensitivity_strict.py
  run_sampling_sensitivity_strict.sh
  analyse_sampling_sensitivity_strict.py

Do NOT overwrite SocialRoutingSimulationFramework_Optimised.py or your old results.

## Experiment
- Schemes: Sacrifice and CollectiveGood
- Three paired replications (seeds 12345, 12346, 12347)
- Two sample sizes: STRICTLY 15 or STRICTLY 30 successful daily simulations per cycle
- Initially 15 behavioural cycles; can resume to 20 later
- 12 trajectories total; 6*15*15 + 6*30*15 = 4050 retained simulation days
- 20 cores for the 15-day arm; initially attempt 20, RETAIN ONLY FIRST 15 successes
- 35 cores for the 30-day arm; initially attempt 35, RETAIN ONLY FIRST 30 successes
- Additional batches are attempted on incomplete/gridlocked simulations (up to retry limit)
- Daily SR route allocation follows Bernoulli p=0.4 as in the main framework.
  The daily simulation sample is a Monte Carlo estimate for each behavioural cycle;
  30 daily samples do NOT double the cycle length / weekly SR commitment.
- Matched same-seed initial participants and per-cycle random behavioural draws
  across 15/30 arms (trajectories may diverge by design after travel-time differences).

## Submit from DelftBlue (the 5_Framework_Simulation directory)
mkdir -p ../logs_Github

# Test a single Sacrifice 15-day cycle and check it succeeds and checkpoints:
sbatch --array=0 --cpus-per-task=20 --time=02:00:00 --export=ALL,NUMBER_OF_CYCLES=15,CYCLES_PER_JOB=1 run_sampling_sensitivity_strict.sh
# Then test 30-day scheme similarly, after first test has finished:
sbatch --array=3 --cpus-per-task=35 --time=02:00:00 --export=ALL,NUMBER_OF_CYCLES=15,CYCLES_PER_JOB=1 run_sampling_sensitivity_strict.sh

# Full 15-cycle run; already-completed cycles will resume rather than restart:
sbatch --array=0-2,6-8 --cpus-per-task=20 --time=14:00:00 --export=ALL,NUMBER_OF_CYCLES=15,CYCLES_PER_JOB=15 run_sampling_sensitivity_strict.sh
sbatch --array=3-5,9-11 --cpus-per-task=35 --time=14:00:00 --export=ALL,NUMBER_OF_CYCLES=15,CYCLES_PER_JOB=15 run_sampling_sensitivity_strict.sh

Do not submit duplicate jobs for the same array IDs while previous ones are running.
If the 14-hour job reaches wall time before completing 15 cycles, resubmit the
SAME command after it ends; completed cycles are preserved in checkpoint files.
An interrupted *incomplete* cycle will be repeated, not partially recovered.
Check squeue -u $USER, and logs ../logs_Github/SR_strict1530_<jobid>_<taskid>.out/.err.

## Expand to 20 cycles later (preserves the first 15 cycles)
sbatch --array=0-2,6-8 --cpus-per-task=20 --time=14:00:00 --export=ALL,NUMBER_OF_CYCLES=20,CYCLES_PER_JOB=20 run_sampling_sensitivity_strict.sh
sbatch --array=3-5,9-11 --cpus-per-task=35 --time=14:00:00 --export=ALL,NUMBER_OF_CYCLES=20,CYCLES_PER_JOB=20 run_sampling_sensitivity_strict.sh

## Results paths
Permanent checkpoints (one per experiment arm):
  5_Framework_Simulation/ComputationalReq_sampling_sensitivity_strict/
     scheme_Sacrifice/rep_00/days_15/output.pickle
     scheme_Sacrifice/rep_00/days_30/output.pickle
     ...
Temporary simulation files:
  /scratch/alexroocroft/ComputationalReq_sampling_sensitivity_strict/

## Analysis (only after all 12 arms have completed at least 15 cycles)
python analyse_sampling_sensitivity_strict.py --cycles 15
# After extension to 20:
python analyse_sampling_sensitivity_strict.py --cycles 20

Produces per-cycle, paired-differences, summary CSV and plots in
  ComputationalReq_sampling_sensitivity_strict/analysis_15cycles/
(or analysis_20cycles/).


