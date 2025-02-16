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

from Funcs import TripsExtraction_SocialRout_module_v3 as Tripextract

from Funcs import day_iteration


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
            with Pool(processes=parallel_cores) as pool:
                pool.map(day_iteration.run_iteration_day, input_data)
        except Exception as e:
            # This block will catch any exception that is a subclass of Exception
            print(f"Cycle {iteration_cycle}....Attempt {attempts + 1}: An error occurred - {e}")

            attempts += 1
        else:
            # If no exception occurred, break out of the loop
            break
    else:
        # This block will execute if all attempts failed
        print(f"Cycle {iteration_cycle}....All SmyuVia attempts failed.")


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

#%% Simulation Inputs
def main():


    parallel_cores=1 # User should set to number of cores available for parallelisation. for base line only one is needed
    # In main simulations: 40 is used to enable twice as many weeks of simulation to be generated at a time for selection of 20 day sample without gridlock issues.

    max_rerun_attempts_days=2

    max_attempts = 5 # if there is an error with Symuvia it will retry up until max_attempts

    # if there is a need to restart the simulation at a specific cycle, last_cycle_restart can be changed
    last_cycle_restart = 1

    #For debugging, if the Symuvia simulations are to be skipped during a rerun, to focus on the decision making between cycles.
    no_sim=0

    sim_hours = 2 # change to 2 for testing 0630 to 0830, 4 for testing 0630 to 1030

    expnum='1' #set to one to indicate sacrifice scheme
    # Note: choice of scheme is not important for the base line Complete SR and UC simulations.

    run_num=102 # run_num ID for the full SO simulation

    social_days_number=1 # all 5 days in a week the users take the SR
    participation_rate1=100 # all users are participating in social routing scheme, all use SR.
    participation_rate=participation_rate1/100

    print(f"Initial Participation Rate: {100*participation_rate}%")
    print(f"Days of Commitment to SR: {social_days_number}")

    if expnum == '1':
        scheme_type = 'Sacrifice'
    else:
        scheme_type='CollectiveGood'

    days_in_week = 1 # 1 for baseline. In main simulations: 5 days in a work week
    weeks_in_cycle = 1 # 1 for baseline. In main simulations: sample size is 20 days, so 4 (5 day) work weeks.
    number_of_cycle = 1 # Only one cycle with full SO routes for all users for evaluating baselines.

    weeks_in_stack=parallel_cores//days_in_week # this will be 8 weeks for 40 cores and 5 days in a week
    number_stack_simulations=weeks_in_stack*days_in_week

    # If the days that a social route participant changes between weeks, shuffle_participation=True
    shuffle_participation=True
    # Note: this does not matter for the baselines as only one cycle is used.

    # Define the path for the new folder
    cwd = os.getcwd()
    folder_outputs = os.path.join(cwd, f'MethodPaper_simhours_{sim_hours}_expnum_{expnum}_initialparticipation_{participation_rate}_SRdays_{social_days_number}_randomInitial_run_{run_num}')

    # Create the folder if it doesn't already exist
    os.makedirs(folder_outputs, exist_ok=True)



    #--- Load Inputs ----
    current_dir = os.getcwd()

    # -----SymuVia_Lyon_Inputs---------
    SymuVia_Lyon_Inputs_path = os.path.join(current_dir, os.pardir, '1_SymuVia_Lyon_Inputs')
    SymuVia_Lyon_Inputs_path = os.path.abspath(SymuVia_Lyon_Inputs_path)

    demand_4h_path = os.path.join(SymuVia_Lyon_Inputs_path, '4_hour_demand_profile')

    network_input = os.path.join(demand_4h_path, 'L63V.xml')
    demand_input_social = os.path.join(demand_4h_path, 'SR_routes.csv')
    demand_input_init = os.path.join(demand_4h_path, 'UC_routes.csv')

    if sim_hours == 2:
        demand_1h_path = os.path.join(SymuVia_Lyon_Inputs_path, '1_hour_demand_profile')

        network_input = os.path.join(demand_1h_path, 'L63V_1h.xml')
        demand_input_social = os.path.join(demand_1h_path, 'social_routes_1h.csv')
        demand_input_init = os.path.join(demand_1h_path, 'init_routes_1h.csv')

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

    #--- Participation Rate ----

    participation_rate_store=dict()
    select_social_store=dict()

    participation_rate_store[0] = participation_rate

    #--- Number of days using social route profile for use in the choice model----

    number_of_days = dict()

    for user_id in range(number_of_users):
        number_of_days[user_id] = (social_days_number / days_in_week) * 10

    select_social=dict()

    for user_id in range(number_of_users):

        if random.random() <= participation_rate:
            select_social[user_id]= True
        else:
            select_social[user_id] = False

    select_social_store[0]=select_social

    select_social_day_list = [True] * social_days_number + [False] * (days_in_week - social_days_number)

    # Select the routes for the drivers taking part in the social routing
    select_social_day = dict()
    for user_id in range(number_of_users):
        random.shuffle(select_social_day_list)
        select_social_day[user_id] = select_social_day_list[:]

    # Load in the choice model parameters for the users.
    # Note: this does not matter for the baselines as only one cycle is used.

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', '2_User_and_Moral_Profile', 'user_profile.pickle')

    with open(file_path, 'rb') as file:
        user_param_store, moral_user = pickle.load(file)

    if sim_hours == 2:
        file_path = os.path.join(script_dir, '..', '2_User_and_Moral_Profile', 'user_profile_1h.pickle')

        with open(file_path, 'rb') as file:
            user_param_store, moral_user = pickle.load(file)

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

    print(f'Scheme Type is set to: {scheme_type}')
    for iteration_cycle in range(last_cycle_restart,number_of_cycle+1):

        print('-----------------------------------------------')
        print(f'Cycle: {iteration_cycle}')
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

            input_data = [(day, network_input, network_input_stem, demand[day], iteration_cycle, temp_folder,
                           symuvia_output, sim_hours) for day in range(countA, countB)]

            # The following runs the Symuvia simulations, if no_sim==0. Repeats until max_attempts if Symuvia produces any errors.

            if no_sim == 0:
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
            break


#---------------------------------------------------
        sorted_keys_problems_in_week = [key for key, value in
                                        sorted(problems_in_week.items(), key=lambda item: item[1])]
        cycle_key_selection = sorted_keys_problems_in_week[0:weeks_in_cycle] # e.g. choose 4 weeks with no problems from those available
        print(f'Cycle key selection : {cycle_key_selection}')

        # Reset the week numbers to be 1 to weeks_in_cycle, e.g 1,2,3,4.
        veh_outputs_week = {new_key: veh_outputs_week[old_key] for new_key, old_key in enumerate(cycle_key_selection, start=1)}

#------------------------------------------------------------------------
        travel_time_init_sim = pd.DataFrame(columns=['travel_time'])

        week = 1  # only one week of one day for the baseline.

        veh_outputs = veh_outputs_week[week]  # select a week of output from the cycle

        veh_output = veh_outputs[0] # choose the first (and only day) for the baseline

        total_travel_time = veh_output[
            'driver_travel_time'].sum()  # sum over all users, each week contains days veh_output, in baseline just one day
        travel_time_init_sim['travel_time'] = veh_output['driver_travel_time']

        # Calculate the additional time per social route day that a driver takes part in

        total_travel_time_hours = total_travel_time / 3600

        print(f'total_travel_time_mean for cycle {iteration_cycle} is {total_travel_time_hours}')

    # ------------------------------------------------------------------------
    file_path = os.path.join(cwd, "macro_output_completeSR_mean.pickle")

    with open(file_path, "wb") as file:
        pickle.dump(
            (total_travel_time_hours,travel_time_init_sim), file)
    return  total_travel_time_hours, travel_time_init_sim

if __name__ == '__main__':
    start_time = time.time()

    total_travel_time_hours,travel_time_init_sim = main()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600

    print(f"Elapsed time: {elapsed_time:.2f} hours")
