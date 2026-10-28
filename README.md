# Social Routing Simulation Framework

This repository contains the code used to produce the results for the paper:

**Long-term dynamics of voluntary congestion management on large networks: A cycle-to-cycle behavioural traffic simulation framework**  
Alexander Roocroft, Baiba Pudane, Caspar Chorus, Ludovic Leclercq

Publication details will be added when available.

The repository provides a research-code implementation of a long-horizon simulation framework for analysing voluntary social-routing schemes on large road networks. The framework combines cycle-level participation updating with within-day dynamic traffic simulation using SymuVia/SymuFlow. The case-study application uses a road network based on Lyon, France.

SymuVia/SymuFlow is not developed in this repository. It should be obtained separately from the [LICIT Lab SymuVia repository](https://github.com/licit-lab/symuvia).

---

## Repository structure

The folders are ordered approximately according to the workflow used to reproduce the paper results.

```text
1_SymuVia_Lyon_Inputs/
2_User_and_Moral_Profile/
3_Complete_SR_and_Baseline_UC/
4_CalculatingSampleSize/
5_Framework_Simulation/
6_Graph_Generate/
Funcs/
env_symupy/
```

The sections below describe the purpose of each folder and the main scripts used in the workflow.

---

## Requirements

The code was developed for Python 3.9 and uses SymuVia/SymuFlow for microscopic traffic simulation.

An environment specification is provided in:

```text
env_symupy/env_symupy.yaml
```

A conda environment can be created with:

```bash
conda env create -f env_symupy/env_symupy.yaml
conda activate env_symupy
```

The environment file includes the main simulation dependencies, including:

```yaml
name: env_symupy
channels:
  - conda-forge
  - anaconda
  - licit-lab
dependencies:
  - python=3.9.15
  - symuflow
  - symupy
```

This repository should include the environment specification only. It should not include a complete local virtual environment or machine-specific Python installation.

---

## 1. Lyon network inputs

Folder:

```text
1_SymuVia_Lyon_Inputs/
```

This folder contains the SymuVia network and route-choice inputs for the Lyon case study.

Active subfolders:

```text
1_SymuVia_Lyon_Inputs/2_hour_demand_profile/
1_SymuVia_Lyon_Inputs/4_hour_demand_profile/
```

Each demand-profile folder contains the SymuVia input file and precomputed route-choice files:

```text
L63V_2h.xml or L63V_4h.xml
SR_routes_2h.csv or SR_routes_4h.csv
UC_routes_2h.csv or UC_routes_4h.csv
```

where:

- `L63V_*.xml` is the SymuVia network and demand input file;
- `SR_routes_*.csv` contains the social-route option for each user;
- `UC_routes_*.csv` contains the uncoordinated-route option for each user.

The 2-hour demand profile is useful for faster testing. The 4-hour demand profile is used for the main case-study simulations.

---

## 2. User and moral-profile generation

Folder:

```text
2_User_and_Moral_Profile/
```

This folder contains the processed user-level behavioural inputs used by the social-routing choice model.

Main scripts:

```text
1_giveandtake_extract_moralprofile.py
2_User_profile_generation.py
check_profiles.py
```

Processed input files included in this repository:

```text
moral_profile.csv
moral_profile.pickle
user_profile_2h.pickle
user_profile_4h.pickle
```

The files `user_profile_2h.pickle` and `user_profile_4h.pickle` are the processed user-profile files used by the main Lyon simulations. Therefore, the main simulation pipeline can be run without redistributing the original survey dataset.

The original survey data file used to construct the processed moral-profile inputs is not included in this repository:

```text
Szep_etal_2022_MFQsurvey_data.csv
```

The script `1_giveandtake_extract_moralprofile.py` is included for transparency, but it can only be rerun if the original survey data are obtained separately and placed in this folder. If using the processed profile files included in this repository, users can skip this raw-data extraction step.

To regenerate the user-profile pickle files from the processed moral-profile input, run:

```bash
cd 2_User_and_Moral_Profile
python 2_User_profile_generation.py
```

The generated profiles can be inspected with:

```bash
python check_profiles.py
```

---

## 3. Baseline generation: complete social routing and full uncoordinated routing

Folder:

```text
3_Complete_SR_and_Baseline_UC/
```

This folder generates baseline network-performance outputs for:

- complete social routing, where all users take the social route;
- full uncoordinated routing, where all users take the uncoordinated route.

Active script:

```text
1_Generate_Baselines_parallel.py
```

Example Slurm shell script:

```text
framework_scheme_baselines.sh
```

The generated baseline outputs are saved as pickle files:

```text
macro_output_completeSR_mean_2h.pickle
macro_output_completeSR_mean_4h.pickle
macro_output_fullUC_mean_2h.pickle
macro_output_fullUC_mean_4h.pickle
```

Example command:

```bash
cd 3_Complete_SR_and_Baseline_UC
python 1_Generate_Baselines_parallel.py 4
```

The supplied shell script is configured for a TU Delft DelftBlue-style Slurm environment. It should be adapted before use on another system.

---

## 4. Computational-viability and sample-size analysis

Folder:

```text
4_CalculatingSampleSize/
```

This folder contains the adapted simulation workflow used for the computational-viability analysis in the paper. It runs a larger pool of independently simulated daily realisations to assess how many simulated days are needed to estimate cycle-level network performance with acceptable stability.

Active scripts:

```text
SocialRoutingSimulationFramework_multiple_simulations_modular.py
SocialRoutingSim_Graphgenerate_multiSim_modular_with_histogram.py
```

Example Slurm shell script:

```text
framework_scheme_multiSim.sh
```

Supporting functions are stored in:

```text
4_CalculatingSampleSize/Funcs_samplesize/
```

### Fixed participation-rate validation

`4_CalculatingSampleSize/FixedRateValidation130/` contains a separate sampling-validation experiment with fixed participation rates of 20%, 50% and 80%. It generates up to 130 successful daily simulation outcomes per scenario as a reference for evaluating estimates calculated from smaller samples, including 15 simulated days. The folder contains the simulation and analysis Python scripts, two Slurm scripts and experiment-specific instructions.

### Daily sample-size sensitivity

`4_CalculatingSampleSize/SamplingSensitivity15vs30/` contains a paired sensitivity experiment comparing 15 and 30 successful daily simulations per behavioural cycle. It covers the Sacrifice and Collective Good schemes, with three replications for each sample size and scheme, and supports trajectories of up to 20 cycles. It has its own simulation, analysis and Slurm scripts.

Both validation workflows are independent of the main simulation array described in Section 5. Their supplied Slurm scripts were developed to run from the `5_Framework_Simulation/` working directory, where relative paths to the network inputs and shared `Funcs/` modules resolve. To run them on DelftBlue, follow their folder-specific instructions and stage the necessary scripts in that execution directory; their source files can remain organised under `4_CalculatingSampleSize/` in the repository.

Outputs from this analysis are written to:

```text
4_CalculatingSampleSize/ComputationalReq_multiple_simulations/
```

Example commands:

```bash
cd 4_CalculatingSampleSize
python SocialRoutingSimulationFramework_multiple_simulations_modular.py
python SocialRoutingSim_Graphgenerate_multiSim_modular_with_histogram.py
```

The graph-generation script uses the simulation outputs to produce the cumulative-mean convergence plots and daily travel-time histogram used in the paper.

---

## 5. Main long-horizon simulation framework

Folder:

```text
5_Framework_Simulation/
```

This folder contains the main social-routing simulation framework and the result-extraction script.

Active scripts:

```text
SocialRoutingSimulationFramework_Optimised.py
SocialRouting_results_extractor_optimised.py
```

Example Slurm shell scripts:

```text
framework_scheme_simulation_optimised_ALL.sh
framework_resultsextractor_optimised.sh
```

### Running the main simulation

The main simulation script takes the following command-line arguments:

```text
python SocialRoutingSimulationFramework_Optimised.py <selector> <scheme> <sim_hours> <number_of_cycle>
```

where:

- `<selector>` is the experiment or case selector, typically supplied by a Slurm array task ID;
- `<scheme>` is either `Sacrifice` or `CollectiveGood`;
- `<sim_hours>` is either `2` or `4`;
- `<number_of_cycle>` is the number of behavioural cycles to simulate.

Example:

```bash
cd 5_Framework_Simulation
python SocialRoutingSimulationFramework_Optimised.py 1 Sacrifice 4 20
```

The supplied Slurm script runs **117 tasks** covering the Sacrifice and Collective Good experiment configurations and replications:

```bash
sbatch framework_scheme_simulation_optimised_ALL.sh
```

This includes the behavioural-parameter and promise-credibility sensitivity cases. In the Collective Good `cg_promise_ar1` case (tasks 115–117), promise violations are evaluated using AR(1)-smoothed perceived travel times; the other cases use their defined promise-evaluation settings. To run only that case:

```bash
sbatch --array=115-117 framework_scheme_simulation_optimised_ALL.sh
```

In addition to network-performance and participation metrics, the framework can record experienced travel times on uncoordinated (UC) and social-routing (SR) days. These are descriptive measurements; they do not replace the travel-time information used by the behavioural model. Participation summaries distinguish the state used to simulate a cycle from decisions made at the end of that cycle.

Before use, edit the Slurm account, partition, conda environment, log paths, scratch/output paths, and SymuVia/SymuFlow library paths.

### Extracting simulation results

After the simulations have completed, extract the outputs with:

```bash
python SocialRouting_results_extractor_optimised.py \
    --results-folder /path/to/Output/Simulations_<RUN_ID> \
    --sim-hours 4 \
    --uc-prior-dir ../3_Complete_SR_and_Baseline_UC \
    --route-day-mode auto \
    --zip
```

This creates an `extracted_output/` folder inside the selected results directory. Its main outputs include macro-level and per-user CSV files used for the paper figures. When route-day outputs are present, the extractor also writes:

```text
route_day_cycle_by_rep_sim4.csv
route_day_cycle_mean_across_reps_sim4.csv
route_day_last_cycles_by_case_sim4.csv
```

Use `--route-day-mode required` to enforce route-day data availability for every expected run, or `--route-day-mode off` when extracting only the standard outputs. The Slurm extraction wrapper requires a simulation job ID:

```bash
sbatch --export=ALL,RUN_ID=<SIMULATION_JOB_ID> framework_resultsextractor_optimised.sh
```

The wrapper expects route-day outputs from a complete simulation array; it should not be used unchanged for partial or older runs without those files.

---

## 6. Graph generation

Folder:

```text
6_Graph_Generate/
```

This folder generates the paper figures from the extracted simulation results. The main entry point is:

```text
generate_all_graphs.py
```

It runs the two figure-generation scripts:

```text
generate_graphs_from_extractor_optimised_v5.py
generate_additional_graphs.py
```

The first generates the main network-performance, participation and sensitivity figures. The second produces route-day travel-time comparisons and Collective Good promise-credibility figures. The supporting Slurm wrapper is `framework_graphgenerate.sh`.

Run the results extractor first, then generate the complete set of figures from a **complete** set of extracted simulation outputs:

```bash
cd 6_Graph_Generate
python generate_all_graphs.py \
    --results-folder /path/to/Output/Simulations_<RUN_ID> \
    --sim-hours 4 \
    --mode all \
    --save-folder graphs
```

The available modes are `all` (both figure scripts), `legacy` (the main paper-figure script) and `additional` (route-day and promise-credibility figures). For a subset of runs that does not include the full sensitivity experiment collection, use `--mode additional` rather than attempting to generate the complete figure set.

On a Slurm cluster, submit the wrapper from `6_Graph_Generate/` after extraction, supplying the simulation job ID:

```bash
sbatch --export=ALL,RUN_ID=<SIMULATION_JOB_ID>,GRAPH_MODE=all framework_graphgenerate.sh
```

The wrapper's default `GRAPH_MODE` is `additional`; set `GRAPH_MODE=all` when the necessary full simulation and extracted results are available. Additional graph outputs are organised under `graphs/additional/route_days/` and `graphs/additional/promise_credibility/`.

---

## 7. Shared functions

Folder:

```text
Funcs/
```

This folder contains shared functions used by the baseline-generation and main framework scripts.

Active function files:

```text
day_iteration_optimised.py
Logit_model_v2.py
RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v5_optimised_gridlockabort.py
TripsExtraction_SocialRout_module_v4_optimised.py
```

The file:

```text
RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v5_optimised_gridlockabort.py
```

contains the interface with SymuVia/SymuFlow. The path to the SymuFlow library must be set correctly for the machine on which the code is run.

---

## Configuration and run metadata

There is no root-level `config.json` file required to run the repository.

Some simulation and post-processing scripts create or read run-specific `config.json` files inside individual output folders. These files store run metadata such as the scheme, case label, replication ID, random seed, and case-specific parameter overrides.

Machine-specific paths, local output folders, and cluster-specific settings should be edited in the relevant Python or shell scripts before execution. Local configuration files and generated run outputs are not intended to be committed to version control.

---

## Suggested reproduction workflow

A typical reproduction workflow is:

1. Prepare the SymuVia Lyon inputs in `1_SymuVia_Lyon_Inputs/`.
2. Generate the moral-profile and user-profile files in `2_User_and_Moral_Profile/`.
3. Generate the complete-social-routing and full-uncoordinated-routing baselines using `3_Complete_SR_and_Baseline_UC/1_Generate_Baselines_parallel.py`.
4. Run the computational-viability analysis or independent fixed-rate and 15-versus-30-day validation workflows in `4_CalculatingSampleSize/` as required for the results being reproduced.
5. Run the main long-horizon simulations using `5_Framework_Simulation/SocialRoutingSimulationFramework_Optimised.py` and its Slurm wrapper.
6. Extract simulation results using `5_Framework_Simulation/SocialRouting_results_extractor_optimised.py`.
7. Generate the complete figure set using `6_Graph_Generate/generate_all_graphs.py`, or select a plotting subset using its `--mode` option.

---

## HPC usage

Several `.sh` files are provided as examples for running the workflow on a Slurm-based HPC system. These scripts were written for a TU Delft DelftBlue-style environment and include system-specific paths, partitions, accounts, and conda environment names.

Before using them on another system, update:

- Slurm account and partition names;
- conda or Python environment activation;
- scratch and output directories;
- log-file directories;
- paths to SymuVia/SymuFlow libraries;
- paths to baseline and extracted-output folders.

---

## Data availability and repository contents

This repository is intended to provide the source code and lightweight supporting files needed to understand and reproduce the modelling workflow used in the paper.

Large generated outputs, local Python environments, cached files, and machine-specific files are not included. These files should be generated locally by running the scripts described above.

Some required inputs may be subject to redistribution restrictions, including SymuFlow binary or library files and the original survey data used to construct the moral-profile inputs. The raw survey file `Szep_etal_2022_MFQsurvey_data.csv` is not redistributed in this repository. Processed moral-profile and user-profile files are included so that the main simulation pipeline can be run without the raw survey dataset. Where restricted files are not included, users should obtain them from the original source and place them in the folder structure described in this README.

## Data and extracted outputs

The extracted simulation outputs used to generate the paper figures are not stored directly in this Git repository because of their file size.

The extracted outputs and generated graph files are archived separately on 4TU.ResearchData:

[10.4121/f262a974-407e-4e39-b031-7fd0901529c7](https://doi.org/10.4121/f262a974-407e-4e39-b031-7fd0901529c7)

The 4TU archive contains the extracted simulation-output archive:

```text
extracted_all_sim4.zip
```

This archive contains the extracted CSV outputs produced by `SocialRouting_results_extractor_optimised.py`:

```text
macro_all_sim4.csv
peruser_hist_all_sim4.csv
peruser_all_sim4.csv
```

The 4TU archive also contains the graph outputs generated from these extracted CSV files. These graph outputs are included for convenience, so that the figures can be inspected without rerunning the graph-generation script.

After downloading and unzipping `extracted_all_sim4.zip`, place the extracted CSV files in:

```text
Output/Simulations_<RUN_ID>/extracted_output/
```

The graph outputs can then be regenerated with:

```bash
cd 6_Graph_Generate
python generate_all_graphs.py \
    --results-folder ../Output/Simulations_<RUN_ID> \
    --sim-hours 4 \
    --mode legacy \
    --save-folder graphs
```

For the complete figure set, use `--mode all` with the complete corresponding results and route-day outputs. The deposited archive described above contains the standard extracted outputs; verify that any additional experiment outputs needed for `--mode all` are included in the dataset version you download.

---

## Citation

If you use this code, please cite the associated paper:

```text
Roocroft, A., Pudane, B., Chorus, C., and Leclercq, L. 
Long-term simulation of voluntary congestion management schemes: 
A modelling framework and application for large road networks. 
Publication details to be added.
```

---

## License

This code is released under the MIT License. See the `LICENSE` file for details.

The MIT License applies to the source code in this repository. Simulation outputs, input data, and externally developed software such as SymuVia/SymuFlow may be subject to separate licences or redistribution conditions.