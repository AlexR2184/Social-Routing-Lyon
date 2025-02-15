#  **Social-Routing-Lyon**

The code in this repository was used to produce the results in the paper  ‘Long-term simulation of voluntary congestion management schemes: A modelling framework and application for large road networks’ (Alexander Roocroft,  Baiba Pudane, Caspar Chorus, Ludovic Leclercq). This paper is published in TBC and can be found at TBC.

The code can be used to implement the framework described in the paper to simulate the performance and development of a social routing scheme on an example road network in Lyon, France.

The files are grouped into folder ordered by the sequence to reproduce the results of the paper. The SimuVia package used needs to be accessed from its separate repository (https://github.com/licit-lab/symuvia)

1.	1_SymuVia_Lyon_Inputs
2.	2_User_and_Moral_Profile
3.	3_Complete_SR_and_Baseline_UC
4.	4_CalculatingSampleSize
5.	5_Framework_Simulation
6.	6_Graph_Generate
7.	Funcs
8.	logs_Github
9.	env_smyupy
10.	python_environment

### 1_SymuVia_Lyon_Inputs
The files in this folder are used to generate the route options for users on the Lyon network. These are the social route SR and the uncoordinated route UC. The 1 hour demand profile can be used for testing, as it is faster to run the simulations for. The 4 hour profile is used in the case study simulations. The xml file is the input for Lyon demand profile on Symuvia, SR_routes and UC_routes are the respective route choices for each user.

### 2_User_and_Moral_Profile
The folder contains data from the social routing user acceptance study by Szep et al. that is used for the choice model in the framework. The python script ‘1_giveandtake_extract_moralprofile.py’ should be run first to extract the relevant information from this file, ‘Szep_etal_2022_MFQsurvey_data.csv’. Secondly, the parameter to use in the choice models of the Lyon simulations (4hr, 1hr) are generated via ‘2_User_profile_generation.py’ and saved in pickle format.

### 3_Complete_SR_and_Baseline_UC
The files in this folder are to be run to generate the baseline results for network performance for complete social routing (i.e., all users taking SR) and fully uncoordinated (i.e., all users taking UC). The scripts are adapted versions of the main framework script to run on a single day with either 100% UC or SR route choices. The files ‘1_SocialRoutingSimulationFramework_completeSR.py’ and ‘2_SocialRoutingSimulationFramework_UC.py’ are set up to run on a HPC cluster with multiple cores. Two example shell scripts are included for running batch jobs (would need to be adapted for your use). The outputs are saved in pickle format. 

### 4_CalculatingSampleSize
This folder contains an adapted version of the framework for running 130 days of simulation to estimate how many days are needed in a sample of simulations for each cycle. An additional script produces the graph found in the paper.

### 5_Framework_Simulation
This folder contains two python scripts: ‘SocialRoutingSimulationFramework_Github.py’ is the framework for simulating the social routing scheme on the Lyon network over a number of cycles. The folder contains example shell scripts to run the codes on a HPC. The shell script ‘framework_scheme_simulation.sh’ is set up to run the framework code with a different input variable. This is to tell the code which experiment to perform from a combination of scheme type (sacrifice/ collective good) and moral profile (survey / most inclined / least inclined). 

‘SocialRoutingSimulationFramework_results_extraction_Github.py’ is to be run subsequently to extract the results of the simulations.

### 6_Graph_Generate
This final folder takes the output of ‘SocialRoutingSimulationFramework_results_extraction_Github.py’ and produces the analysis graphs found in the paper.

### Funcs
This folder contains functions that are used in the framework. Some relate to the interface with Symuvia, some related to applying the choice model taken from the social routing user acceptance survey by Szep et al..

In the file ‘RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v3.py’, lib_path has to be set correctly for the user to where the Symuvia library is saved.
### logs_Github
This is a folder for saving the outputs of running on the HPC.
### env_symupy
This contains a library for SymuVia used in the function file, ‘RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v3.py’

### python_environment
For compatibility, a txt and yml file contained in this folder list the dependencies of the environment used to run the Python scripts.


