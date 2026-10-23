# Computational-Viability and Sample-Size Analysis

This folder contains the simulation and analysis code used to assess the number of daily traffic simulations needed to estimate cycle-level network performance in the Lyon Social Routing case study. It includes the main sample-size analysis and two independent validation experiments.

## Main sample-size analysis

The following scripts evaluate the stability of network-performance estimates as the number of simulated days increases:

- `SocialRoutingSimulationFramework_multiple_simulations_modular.py` — runs the daily simulation samples.
- `SocialRoutingSim_Graphgenerate_multiSim_modular_with_histogram.py` — generates cumulative-mean convergence plots and a daily travel-time histogram.
- `framework_scheme_multiSim.sh` — Slurm simulation wrapper.
- `framework_scheme_multiSim_graphs.sh` — Slurm plotting wrapper.

Supporting code is provided in `Funcs_samplesize/`.

## Fixed-rate validation: `FixedRateValidation130/`

This experiment simulates **20%, 50% and 80% fixed participation rates**, targeting **130 successful daily simulations** for each scenario. Its analysis assesses estimates based on smaller daily samples (including 15 days) against the 130-day reference.

The folder contains the simulation script, analysis script, two Slurm wrappers and `README_fixed_rate_validation_130.md` with execution instructions.

## Daily sample-size sensitivity: `SamplingSensitivity15vs30/`

This experiment compares **15 versus 30 successful daily simulations per behavioural cycle** for both the Sacrifice and Collective Good schemes, with **three paired replications** per scheme. The workflow supports up to **20 behavioural cycles** and provides scripts to analyse differences between the resulting trajectories.

The folder contains the simulation script, analysis script, a Slurm wrapper and `README_sampling_sensitivity_strict.txt` with execution instructions. Its 15-day and 30-day arms require separate Slurm submissions with appropriate CPU allocations.

## Running the code

The two validation experiments are **separate from** the main simulation array in `5_Framework_Simulation/`; running that array does not run these validations.

The supplied validation Slurm scripts use `5_Framework_Simulation/` as their DelftBlue working directory, so that relative paths to network inputs and shared `Funcs/` modules resolve as expected. When executing these experiments, follow their individual READMEs and stage the relevant `.py` and `.sh` files in that working directory. Keep the repository copies organised in the subfolders above, and do not overwrite `SocialRoutingSimulationFramework_Optimised.py`.

Generated results, checkpoints and logs are not stored in this source-code folder or committed to GitHub.
