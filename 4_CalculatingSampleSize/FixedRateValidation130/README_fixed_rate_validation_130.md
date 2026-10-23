# Fixed participation-rate validation — 130 days

This four-file workflow is a separate fixed-state sampling validation,
not a re-run of the cycle-to-cycle behavioural framework.

Files:

- `SocialRoutingSimulationFramework_fixed_rate_validation_130.py`
- `run_fixed_rate_validation_130.sh`
- `analyse_fixed_rate_validation_130.py`
- `run_analyse_fixed_rate_validation_130.sh`

The Slurm script's task indices 0, 1 and 2 correspond to fixed
participation rates **20%, 50%, 80%**. Each scenario targets **130
successful daily simulations**, with intermediate checkpoints and resumption
across multiple four-hour jobs. The analysis compares the 15-day estimator
against a 130-day reference (and considers 16 days), using repeated resampling.

**Working-directory requirement:** the Slurm scripts `cd` into
`/home/alexroocroft/GitHub_Version/5_Framework_Simulation` and expect the two
Python files to be present there. The simulation imports modules from
`../Funcs` and resolves network inputs relative to that working directory.
For execution on DelftBlue, copy these four files to the local
`5_Framework_Simulation` folder, adapt machine-specific paths, and create
`../logs_Github` before submission. Do **not** overwrite the normal
`SocialRoutingSimulationFramework_Optimised.py`.

Example (from the `5_Framework_Simulation` working directory):

```bash
sbatch run_fixed_rate_validation_130.sh
```

Use completed checkpoints before attempting the analysis:

```bash
sbatch run_analyse_fixed_rate_validation_130.sh
```

Outputs go to `ComputationalReq_fixed_rate_validation_130/` in the local
working directory and scratch. **Do not commit these generated outputs.**
Review script settings and logs before treating results as validated; this
source-only package does not include the experimental results.
