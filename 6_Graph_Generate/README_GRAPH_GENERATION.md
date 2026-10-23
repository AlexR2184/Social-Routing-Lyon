# Social Routing — one-command graph generation

Keep these files together in `6_Graph_Generate` (their GitHub location):

- `generate_all_graphs.py` — one master entry point for all graphs
- `generate_graphs_from_extractor_optimised_v5.py` — existing paper figures (baseline-only memory-efficient per-user data loading)
- `generate_additional_graphs.py` — route-day and promise-credibility figures
- `framework_graphgenerate.sh` — DelftBlue Slurm wrapper; defaults to **additional** mode

## Plotting a nine-run subset (two scheme baselines and one CG robustness case)

**Use only the additional graphs.** A nine-run subset includes Sacrifice baseline,
Collective Good baseline, and CG smoothed-promise robustness, but not the full
set of sensitivity experiments. The original v5 paper script therefore cannot
reproduce *all* paper figures from this folder.

Run the extractor first so these exist:

- `extracted_output/macro_all_sim4.csv`
- `extracted_output/route_day_cycle_by_rep_sim4.csv`

Then, from the `6_Graph_Generate` directory:

```bash
sbatch --export=ALL,RUN_ID=<YOUR_SIMULATION_JOB_ID> framework_graphgenerate.sh
```

Or run the exact master command (on DelftBlue, with the correct environment):

```bash
/home/alexroocroft/.conda/envs/env_symupy/bin/python generate_all_graphs.py \
  --results-folder /scratch/alexroocroft/Output/Simulations_<YOUR_SIMULATION_JOB_ID> \
  --sim-hours 4 \
  --mode additional \
  --save-folder graphs
```

The *additional* mode reads the extracted CSVs; it does **not** need any
`output.pickle` files, including those from older sensitivity runs. The
user-level paired-difference distribution can additionally use the per-run
compressed `route_day_outputs/cycle_XX_users.csv.gz` files if available; if
missing, it is skipped with a warning. Append `--no-user-histogram` to skip
that optional check intentionally.

## GitHub / full reproducibility

With the complete original experiment results present (including all needed
`config.json`, `output.pickle` and extractor CSV files), use:

```bash
python generate_all_graphs.py --results-folder /path/to/Simulations_JOB_ID --mode all
```

The two supported subsets are `--mode legacy` (original figures) and
`--mode additional` (additional figures). The Slurm wrapper accepts
`GRAPH_MODE=all` if running against a **complete**, extracted experiment:

```bash
sbatch --export=ALL,RUN_ID=FULL_JOB_ID,GRAPH_MODE=all framework_graphgenerate.sh
```

`--mode all` is the master script default, but it is **not** recommended for
the nine-run output folder if your objective is to regenerate the full set of
sensitivity-analysis plots.

## Output folders

- `graphs/additional/route_days/`
  - `route_day_participant_uc_sr_day_means.pdf`
  - `route_day_paired_sr_minus_uc_by_rep.pdf`
  - `route_day_participant_nonparticipant_all_days.pdf`
  - `route_day_user_paired_difference_distribution.pdf` (optional)
  - replication and summary CSVs
- `graphs/additional/promise_credibility/`
  - `promise_participation_start_of_cycle.pdf`
  - `promise_permanent_withdrawals_end_of_cycle.pdf`
  - `promise_weekly_network_benefit.pdf`
  - replication, paired-difference, terminal-state and summary CSVs

The promise comparison uses start-of-cycle participation (the state generating
that cycle's traffic), end-of-cycle cumulative permanent exclusions (including
cycle 20), and network benefits. Shaded bands show replication min/max, not
confidence intervals. Final-five-cycle statistics are averaged within each
replication first. The route-day comparisons are descriptive, not causal.

## Dependencies

Python 3.9+ with NumPy, pandas and matplotlib. Only the legacy figure stage
loads original `output.pickle` objects. The extractor must be run separately,
before graph generation.
