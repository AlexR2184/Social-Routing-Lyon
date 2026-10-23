# Social Routing Simulation Framework

This folder contains the simulation framework for the Lyon Social Routing case study.

The framework simulates voluntary participation in the Sacrifice and Collective Good schemes over successive cycles, accounting for individual participation decisions, travel-time experiences and network performance.

The simulation includes sensitivity analyses of behavioural parameters, participation conditions and promise credibility, including the use of perceived travel times in evaluating promise violations.

The framework also records individual travel-time experiences on uncoordinated (UC) and social-routing (SR) days.

## Files

- `SocialRoutingSimulationFramework_Optimised.py` — main simulation framework.
- `framework_scheme_simulation_optimised_ALL.sh` — runs all 117 simulation tasks on DelftBlue.
- `SocialRouting_results_extractor_optimised.py` — extracts and aggregates simulation results.
- `framework_resultsextractor_optimised.sh` — submits results extraction on DelftBlue.
