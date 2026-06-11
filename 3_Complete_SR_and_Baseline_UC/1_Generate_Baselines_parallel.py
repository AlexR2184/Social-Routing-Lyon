#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th Novemeber 2023:

Author: Alex Roocroft
"""


from multiprocessing import Pool
import sys
import random
from itertools import product
import pandas as pd
import pickle
import time
import os


# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Funcs import TripsExtraction_SocialRout_module_v4_optimised as Tripextract
from Funcs import day_iteration_optimised as day_iteration

print(f"Running in PID {os.getpid()} — __name__ = {__name__}")

def generate_cycle_social_days(number_of_users, weeks_in_stack, shuffle_participation, select_social_day_list):
    """Generates cycle social days data structure."""
    week_social_day = {}
    for week in range(1, weeks_in_stack + 1):
        if shuffle_participation:
            select_social_day = {}
            for user_id in range(number_of_users):
                random.shuffle(select_social_day_list)
                select_social_day[user_id] = select_social_day_list[:]
            week_social_day[week] = select_social_day
    return week_social_day

def generate_demand(week_social_day, veh_social, veh_init, number_of_users, select_social, days_in_week):
    # This for loop creates the 'demand' input for the Symuvia simulations of the cycle. It assigns the route for each user
    # depending on whether they are using SR or UC on a given day.
    """Generates demand for Symuvia simulations."""
    demand = {}
    counter = 1
    for week in week_social_day:
        select_social_day = week_social_day[week]
        for iteration_day in range(1, days_in_week + 1):
            mask = [False] * number_of_users
            for user_id in range(number_of_users):
                if select_social[user_id]:
                    mask[user_id] = select_social_day[user_id][iteration_day - 1]
            negated_mask = [not m for m in mask]
            selected_rows1 = veh_social[mask]
            selected_rows2 = veh_init[negated_mask]
            demand_day = pd.concat([selected_rows1, selected_rows2], axis=0)
            demand[counter] = demand_day
            counter += 1
    return demand

def run_symuvia_simulation(iteration_cycle, input_data, parallel_cores, max_attempts):
    """Runs Symuvia simulations with error handling and retries."""

    attempts = 0
    while attempts < max_attempts:
        try:
            print(f"Cycle {iteration_cycle}....Attempt {attempts + 1}: Starting multiprocessing pool")

            # Create and run pool
            pool = Pool(processes=parallel_cores)
            pool.map(day_iteration.run_iteration_day, input_data)
            pool.close()
            pool.join()

        except Exception as e:
            print(f"Cycle {iteration_cycle}....Attempt {attempts + 1}: An error occurred - {e}")
            attempts += 1
        else:
            break
    else:
        print(f"Cycle {iteration_cycle}....All SymuVia attempts failed.")

def extract_simulation_output(iteration_cycle, number_stack_simulations, out_folder, temp_folder, symuvia_output, sim_hours, days_in_week, weeks_in_stack):
    """Extracts and processes Symuvia simulation outputs."""
    veh_outputs_week = {}
    veh_outputs = []
    problem_day = []
    problems_in_week = {key: 0 for key in range(1, weeks_in_stack + 1)}
    count = 0
    week_num = 1
    for iteration_day in range(1, number_stack_simulations + 1):
        print(f"Extracting simulation output for Cycle {iteration_cycle}....Day {iteration_day}.")
        count += 1
        symuvia_output_parallel = 'defaultOut' + symuvia_output
        temp_folder_parallel = temp_folder + f'_Cycle_{iteration_cycle}_Day_{iteration_day}/'

        # Use the trip extraction script from function Tripextract.symmvia_output
        veh_output = Tripextract.symmvia_output(out_folder, temp_folder_parallel, symuvia_output_parallel,
                                                iteration_day, iteration_cycle, sim_hours)
        # A check on whether the simulation lead to gridlock and incomplete trips. These are not to be included in the cycle.
        incomplete_trips_check = veh_output['incomplete_trips']

        print(f"Cycle {iteration_cycle}, Day {iteration_day}: Incomplete trips check -> {incomplete_trips_check}")

        veh_outputs.append(veh_output)

        if incomplete_trips_check > 0:
            problem_day.append(iteration_day)
            problems_in_week[week_num] += 1

        if count == days_in_week:
            veh_outputs_week[week_num] = veh_outputs
            week_num += 1
            veh_outputs = []
            count = 0
    return veh_outputs_week, problems_in_week

#--------------------------------------------------------------------------------

def run_baseline(participation_rate, sim_hours, run_num):
    lib_path = r'/home/alexroocroft/.conda/envs/env_symupy/lib/'  # Need to set to the correct path.

    max_rerun_attempts_days=2
    parallel_cores=2
    max_attempts = 5 # if there is an error with Symuvia it will retry up until max_attempts

    #For debugging, if the Symuvia simulations are to be skipped during a rerun, to focus on the decision making between cycles.
    no_sim=0

    social_days_number=1 # all 5 days in a week the users take the SR

    print(f"Initial Participation Rate: {100*participation_rate}%")
    print(f"Days of Commitment to SR: {social_days_number}")


    days_in_week = 1 # 1 for baseline. In main simulations: 5 days in a work week
    weeks_in_cycle = 1 # 1 for baseline. In main simulations: sample size is 20 days, so 4 (5 day) work weeks.

    weeks_in_stack = parallel_cores // days_in_week
    number_stack_simulations=weeks_in_stack*days_in_week

    # If the days that a social route participant changes between weeks, shuffle_participation=True
    shuffle_participation=True
    # Note: this does not matter for the baselines as only one cycle is used.

    # Define the path for the new folder
    cwd = os.getcwd()
    folder_outputs = os.path.join(cwd, f'MethodPaper_simhours_{sim_hours}_initialparticipation_{participation_rate}_id_{run_num}')

    # Create the folder if it doesn't already exist
    os.makedirs(folder_outputs, exist_ok=True)

    #--- Load Inputs ----
    current_dir = os.getcwd()

    # -----SymuVia_Lyon_Inputs---------

    SymuVia_Lyon_Inputs_path = os.path.join(current_dir, os.pardir, '1_SymuVia_Lyon_Inputs')
    SymuVia_Lyon_Inputs_path = os.path.abspath(SymuVia_Lyon_Inputs_path)
    if sim_hours == 4:
        demand_4h_path = os.path.join(SymuVia_Lyon_Inputs_path, '4_hour_demand_profile')
        network_input = os.path.join(demand_4h_path, 'L63V_4h.xml')
        demand_input_social = os.path.join(demand_4h_path, 'SR_routes_4h.csv')
        demand_input_init = os.path.join(demand_4h_path, 'UC_routes_4h.csv')

    if sim_hours == 2:
        demand_1h_path = os.path.join(SymuVia_Lyon_Inputs_path, '2_hour_demand_profile')
        network_input = os.path.join(demand_1h_path, 'L63V_2h.xml')
        demand_input_social = os.path.join(demand_1h_path, 'SR_routes_2h.csv')
        demand_input_init = os.path.join(demand_1h_path, 'UC_routes_2h.csv')

    # Demand and route option loading
    veh_social_1 = pd.read_csv(demand_input_social, sep=";")  # columns: origin;typeofvehicle;creation;path;destination
    veh_init_1 = pd.read_csv(demand_input_init, sep=";")  # columns: origin;typeofvehicle;creation;path;destination

    veh_social = pd.merge(veh_social_1, veh_init_1[['origin', 'destination', 'creation_time']], on=['origin', 'destination', 'creation_time'], how='inner')
    veh_init = pd.merge(veh_init_1, veh_social[['origin', 'destination', 'creation_time']], on=['origin', 'destination', 'creation_time'], how='inner')


    number_of_users = len(veh_init['id'])
    print(f'number_of_users in simulation is {number_of_users}, users in SR is {len(veh_social_1["id"])} and users in UC is {len(veh_init_1["id"])}')


    #--- Simulation Details ----

    start_sim_time='06300'
    if 6+sim_hours>=10:
        finish_sim_time = f'{6 + sim_hours}300'
    else:
        finish_sim_time = f'0{6+sim_hours}300'
    symuvia_output = f"_{start_sim_time}0_{finish_sim_time}0_traf.xml" # creates the file name to save Symuvia output

    select_social=dict()

    for user_id in range(number_of_users):

        if random.random() <= participation_rate:
            select_social[user_id]= True
        else:
            select_social[user_id] = False

    select_social_day_list = [True] * social_days_number + [False] * (days_in_week - social_days_number)

    # Select the routes for the drivers taking part in the social routing
    select_social_day = dict()
    for user_id in range(number_of_users):
        random.shuffle(select_social_day_list)
        select_social_day[user_id] = select_social_day_list[:]



    # ----- Create folders TEMP and OUT for saving SymuVia outputs -------

    if not os.path.exists(folder_outputs):
        # Create the folder if it doesn't exist
        os.makedirs(folder_outputs)

    # Define TEMP and OUT folder paths
    temp_folder_path = os.path.join(folder_outputs, 'TEMP')
    out_folder_path = os.path.join(folder_outputs, 'OUT')

    # Create TEMP and OUT folders if they do not exist
    os.makedirs(temp_folder_path, exist_ok=True)
    os.makedirs(out_folder_path, exist_ok=True)

    # Define network input stem
    network_input_stem = os.path.join(temp_folder_path, 'L63V')

    # Define output locations
    out_folder = out_folder_path + os.sep
    temp_folder = os.path.join(temp_folder_path, 'TEMP')
    temp_folder = temp_folder.rstrip(os.sep)  # Removes any extra slash at the end




    iteration_cycle = 1
    print('-----------------------------------------------')
    print(f'Starting simulation: {run_num}')
    print('---------------------------------------------')

    #--------------- ---------------------------

    # The following adds more weeks to stack if less than 4 (20 days) have no gridlock problems.

    number_stack_simulations_rerun = 0
    rerun_attempts = 0
    weeks_in_stack_rerun = 0
    count_zero_problem_weeks = 0

    while count_zero_problem_weeks < weeks_in_cycle:
        rerun_attempts += 1
        if rerun_attempts == max_rerun_attempts_days:
            print(f'Too many rerun attempts in Cycle: {iteration_cycle}')
            break
        number_stack_simulations_rerun += number_stack_simulations
        weeks_in_stack_rerun += weeks_in_stack

        week_social_day = generate_cycle_social_days(number_of_users, weeks_in_stack_rerun,
                                                     shuffle_participation, select_social_day_list)

        # cycle_social_days[iteration_cycle] = week_social_day

        demand_rerun = generate_demand(week_social_day, veh_social, veh_init, number_of_users, select_social,
                                 days_in_week)
        demand = dict()
        # -----------------------------------------------
        for i in range(1, number_stack_simulations + 1):
            demand[i + (number_stack_simulations_rerun - number_stack_simulations)] = demand_rerun[i]

        countA = 1 + (number_stack_simulations_rerun - number_stack_simulations)
        countB = 1 + number_stack_simulations_rerun

        input_data = [(day, network_input, network_input_stem, demand[day], iteration_cycle, lib_path) for day in range(countA, countB)]

        # The following runs the Symuvia simulations, if no_sim==0. Repeats until max_attempts if Symuvia produces any errors.

        if no_sim == 0:
            print('running simulation')
            run_symuvia_simulation(iteration_cycle, input_data, parallel_cores, max_attempts)

        # ---------------------------------------
        # The following code extracts the results from the Symuvia simulation for the days simulated.
        veh_outputs_week, problems_in_week = extract_simulation_output(iteration_cycle,
                                                                       number_stack_simulations_rerun, out_folder,
                                                                       temp_folder, symuvia_output, sim_hours, days_in_week, weeks_in_stack_rerun)

        # Calculate the number of weeks with at least one day with a gridlock problem.

        count_zero_problem_weeks = sum(1 for value in problems_in_week.values() if value == 0)


        print(f'Number of weeks without a simulation problem in cycle: {iteration_cycle}, is {count_zero_problem_weeks}')

    if count_zero_problem_weeks < weeks_in_cycle:
        print(f'After attempting reruns, number of weeks in stack without a problem remains too high. Exiting simulation...')
        return


#---------------------------------------------------
    sorted_keys_problems_in_week = [key for key, value in
                                    sorted(problems_in_week.items(), key=lambda item: item[1])]
    cycle_key_selection = sorted_keys_problems_in_week[0:weeks_in_cycle] # e.g. choose 4 weeks with no problems from those available
    print(f'Cycle key selection : {cycle_key_selection}')


    print(veh_outputs_week)
    # Reset the week numbers to be 1 to weeks_in_cycle, e.g 1,2,3,4.
    veh_outputs_week = {new_key: veh_outputs_week[old_key] for new_key, old_key in enumerate(cycle_key_selection, start=1)}
    print(veh_outputs_week)

#------------------------------------------------------------------------
    travel_time_init_sim = pd.DataFrame(columns=['travel_time'])

    week = 1  # only one week of one day for the baseline.

    veh_outputs = veh_outputs_week[week]  # select a week of output from the cycle
    print(veh_outputs)
    veh_output = veh_outputs[0]  # choose the first (and only day) for the baseline

    total_travel_time = veh_output[
        'driver_travel_time'].sum()  # sum over all users, each week contains days veh_output, in baseline just one day
    travel_time_init_sim['travel_time'] = veh_output['driver_travel_time']

    # Calculate the additional time per social route day that a driver takes part in

    total_travel_time_hours = total_travel_time / 3600

    print(f'total_travel_time_mean for cycle {iteration_cycle} is {total_travel_time_hours}')

# ------------------------------------------------------------------------

    if run_num == 101:
        file_path = os.path.join(cwd, "macro_output_fullUC_mean_2h.pickle")
    if run_num == 102:
        file_path = os.path.join(cwd, "macro_output_fullUC_mean_4h.pickle")

    if run_num == 103:
        file_path = os.path.join(cwd, "macro_output_completeSR_mean_2h.pickle")
    if run_num == 104:
        file_path = os.path.join(cwd, "macro_output_completeSR_mean_4h.pickle")

    with open(file_path, "wb") as file:
        pickle.dump(
            (total_travel_time_hours,travel_time_init_sim), file)



#%% Simulation Inputs
def main():

    if len(sys.argv) > 1:  # used to load in a counter from the shell script that can select scheme type, sensitivity analysis.
        selector = int(sys.argv[1])

    # Define all scenarios
    scenarios = [
        (0.0, 2, 101),  # UC - 2h
        (0.0, 4, 102),  # UC - 4h
        (1.0, 2, 103),  # SR - 2h
        (1.0, 4, 104),  # SR - 4h
    ]

    run_baseline(*scenarios[selector-1]) #use tuple unpacking to pass the tuple a three seperate args.


if __name__ == '__main__':
    import multiprocessing

    try:
        multiprocessing.set_start_method("fork")  # 'fork' for Linux, 'spawn' if you're on MacOS/Windows
    except RuntimeError:
        pass  # Start method already set (safe fallback)

    print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    print(f"Elapsed time: {elapsed_time:.2f} hours")
