#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th Novemeber 2023:

This script varies the number of days of commitment from 1,2,3,4 out of 5. Does not check 0 or 5 days.
Performed for experiment 1 (sacrifice).
"""

from multiprocessing import Pool
import os
import random
from itertools import product
import pandas as pd
import pickle
# For recording the model specification
import zipfile
import sys
import time

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


def extract_simulation_output_samplesize(iteration_cycle, number_stack_simulations, out_folder, temp_folder, symuvia_output, sim_hours, days_in_week, day_count, total_user_experienced_time_day):
    """Extracts and processes Symuvia simulation outputs."""
    problems_in_week = 0
    count = 0
    veh_outputs = []

    for iteration_day in range(1, number_stack_simulations + 1):
        print(f"Extracting simulation output for Round {iteration_cycle} of sample creation.")
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

        if incomplete_trips_check > 0:  # Based on incomplete trips
            problems_in_week = 1
            print('Gridlock issue is detected')
        if count == days_in_week:
            if problems_in_week == 0:
                for veh_output in veh_outputs:
                    total_user_experienced_time = veh_output['driver_travel_time'].sum()
                    day_count += 1
                    total_user_experienced_time_day[day_count] = total_user_experienced_time
                    print(f'day_count is {day_count}')
            veh_outputs = []
            count = 0
            problems_in_week = 0

    return day_count, total_user_experienced_time_day


#%% Simulation Inputs
def main():
    total_sample_size = 130 # 130 was used in the paper.

    parallel_cores=40 # User should set to number of cores available for parallelisation. for base line only one is needed
    # In main simulations: 40 is used to enable twice as many weeks of simulation to be generated at a time for selection of 20 day sample without gridlock issues.

    sim_hours = 4 # change to 2 for testing 0630 to 0830, 4 for testing 0630 to 1030

    #For debugging, if the Symuvia simulations are to be skipped during a rerun, to focus on the decision making between cycles.
    no_sim=0

    max_attempts=5

    run_num=1 # an id number to organise the runs of the simulations. Used for organising the main simulations, not needed here.

    expnum='1' #set to one to indicate sacrifice scheme
    # Note: choice of scheme is not important for the base line Complete SR and UC simulations.

    social_days_number=2 # all 5 days in a week the users take the SR
    participation_rate1=20 # all users are participating in social routing scheme, all use SR.
    participation_rate=participation_rate1/100

    print(f"Initial Participation Rate: {100*participation_rate}%")
    print(f"Days of Commitment to SR: {social_days_number}")

    if expnum == '1':
        scheme_type = 'Sacrifice'
    else:
        scheme_type='CollectiveGood'

    days_in_week = 5 # 1 for baseline. In main simulations: 5 days in a work week

    weeks_in_stack=parallel_cores//days_in_week # this will be 8 weeks for 40 cores and 5 days in a week
    number_stack_simulations=weeks_in_stack*days_in_week

    # If the days that a social route participant changes between weeks, shuffle_participation=True
    shuffle_participation=True
    # Note: this does not matter for the baselines as only one cycle is used.

    # Define the path for the new folder
    cwd = os.getcwd()

    folder_outputs = os.path.join(cwd, "ComputationalReq_multiple_simulations")

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


    iteration_cycle=0
    day_count = 0
    total_user_experienced_time_day = {}
    df_time_output_exp={}

    while day_count < total_sample_size:
        iteration_cycle += 1
    # for iteration_cycle in range(last_cycle_restart,number_of_cycle+1):
        print('-----------------------------------------------')
        print(f'Cycle: {iteration_cycle}')
        print('---------------------------------------------')

        week_social_day = generate_cycle_social_days(number_of_users, weeks_in_stack,
                                                     shuffle_participation, select_social_day_list)

        # cycle_social_days[iteration_cycle] = week_social_day

        demand = generate_demand(week_social_day, veh_social, veh_init, number_of_users, select_social,
                                       days_in_week)


        input_data = [(day, network_input, network_input_stem, demand[day], iteration_cycle, temp_folder,
                       symuvia_output, sim_hours) for day in range(1, 1 + number_stack_simulations)]
        if no_sim == 0:
            run_symuvia_simulation(iteration_cycle, input_data, parallel_cores, max_attempts)

        # ---------------------------------------
        # The following code extracts the results from the Symuvia simulation for the days simulated.
        day_count_new, total_user_experienced_time_day_new = extract_simulation_output_samplesize(iteration_cycle,
                                                                       number_stack_simulations, out_folder,
                                                                       temp_folder, symuvia_output, sim_hours,
                                                                       days_in_week, day_count, total_user_experienced_time_day)
        day_count = day_count_new
        total_user_experienced_time_day = total_user_experienced_time_day_new
        print(f'Current number of sample days in stack: {day_count}')
        print(f'Current sample stack: {total_user_experienced_time_day}')

    df_time_output_exp[(expnum, participation_rate, social_days_number)] = {
        'total_user_experienced_time_day': total_user_experienced_time_day}

    pickle_file = os.path.join(folder_outputs,
                            f"individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.pickle")

    with open(pickle_file, 'wb') as file:
        pickle.dump(df_time_output_exp, file)

    zip_file = os.path.join(folder_outputs,
                            f"individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.zip")

    # Create a zip archive and add the pickle file to it
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(pickle_file, f'individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.pickle')

    print(f'Pickle file {pickle_file} has been zipped to {zip_file}')


if __name__ == '__main__':
    start_time = time.time()

    main()
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600

    # network_para = [user_param_store["Sacrifice"][key]['Network_benefit'] for key in user_param_store["Sacrifice"]]
    print(f"Elapsed time: {elapsed_time:.2f} hours")
#
