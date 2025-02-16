#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th Novemeber 2023:

author: Alex Roocroft

"""


import copy
from multiprocessing import Pool
import os
import sys
import numpy as np
import random
from itertools import product
import pandas as pd
import pickle
import time

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to the Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Funcs import TripsExtraction_SocialRout_module_v3 as Tripextract
from Funcs import Logit_model_v2 as user_choices
from Funcs import day_iteration


def generate_cycle_social_days(number_of_users, weeks_in_stack_total, weeks_in_stack, shuffle_participation, select_social_day_list):
    """Generates cycle social days data structure."""
    week_social_day = {}
    for week in range(weeks_in_stack_total - weeks_in_stack, weeks_in_stack_total): # operates on the weeks in the new part of the stack of weeks.
        if shuffle_participation:
            select_social_day = {}
            for user_id in range(number_of_users):
                random.shuffle(select_social_day_list)
                select_social_day[user_id] = select_social_day_list[:]
            week_social_day[week + 1] = select_social_day #plus one so the first week is one not zero.
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

def calculate_network_benefit(veh_outputs_week, weeks_in_cycle, travel_time_init, Network_benefit_All, days_in_week):
    """Calculates network benefit values for each week."""
    Network_benefit_week = {}
    Network_benefit_unscaled_week = {}

    for week in range(1, weeks_in_cycle + 1):
        veh_outputs = veh_outputs_week[week]
        Network_benefit_user = sum(
            veh_output["driver_travel_time"].sum() - travel_time_init["travel_time"].sum()
            for veh_output in veh_outputs
        )

        # Convert to hours and scale benefit
        Network_benefit = -1 * Network_benefit_user / 3600
        Network_benefit_unscaled = Network_benefit
        Network_benefit = 7.5 * Network_benefit / abs(Network_benefit_All * days_in_week)

        Network_benefit_week[week] = Network_benefit
        Network_benefit_unscaled_week[week] = Network_benefit_unscaled

    return Network_benefit_week, Network_benefit_unscaled_week


def calculate_additional_time(veh_outputs_week, weeks_in_cycle, select_social_day, select_social,
                              number_of_users, days_in_week, social_days_number, travel_time_init):
    """Calculates additional travel time for users on social days and non-participant days."""

    additional_time_week = {}
    promise_additional_time_week = {}
    mean_participant_additional_time_week = {}
    participant_additional_time_socialdays_week = {}

    for week in range(1, weeks_in_cycle + 1):
        veh_outputs = veh_outputs_week[week]
        additional_time_users = pd.Series(0, index=range(number_of_users))
        additional_time_users_fullparticipant = pd.Series(0, index=range(number_of_users))

        # Compute additional time for social days
        for day_output_number, veh_output in enumerate(veh_outputs):
            additional_time_day = pd.Series(0, index=range(number_of_users))
            for user_id in range(number_of_users):
                if select_social_day[user_id][day_output_number]:  # Social route users
                    additional_time_day[user_id] = (
                        veh_output["driver_travel_time"][user_id] - travel_time_init["travel_time"][user_id]
                    )
            additional_time_users += additional_time_day / 60  # Convert to minutes

        # Normalize by number of social days (for social participants)
        additional_time_users_socialdays = additional_time_users / social_days_number

        # Compute additional time for all participants (full week)
        for day_output_number, veh_output in enumerate(veh_outputs):
            additional_time_day = pd.Series(0, index=range(number_of_users))
            for user_id in range(number_of_users):
                if select_social[user_id]:  # All participants (both social & non-social days)
                    additional_time_day[user_id] = (
                        veh_output["driver_travel_time"][user_id] - travel_time_init["travel_time"][user_id]
                    )
            additional_time_users_fullparticipant += additional_time_day / 60  # Convert to minutes

        additional_time_users_fullweek = additional_time_users_fullparticipant / days_in_week

        # Compute mean additional time for participants
        participant_indices = [k for k, v in select_social.items() if v]
        mean_participant_additional_time = additional_time_users_socialdays.loc[participant_indices].mean()
        mean_fullparticipant_additional_time = additional_time_users_fullweek.loc[participant_indices].mean()

        # Assign mean to non-participants
        additional_time_users_socialdays.loc[
            ~additional_time_users_socialdays.index.isin(participant_indices)
        ] = mean_participant_additional_time

        additional_time_users_fullweek.loc[
            ~additional_time_users_fullweek.index.isin(participant_indices)
        ] = mean_fullparticipant_additional_time

        # Store results for this week
        additional_time_week[week] = additional_time_users_socialdays
        promise_additional_time_week[week] = additional_time_users_fullweek


    return (
        additional_time_week,
        promise_additional_time_week,

    )


def store_weekly_data(
    additional_time_week, promise_additional_time_week, Network_benefit_week,
    Network_benefit_unscaled_week, participation_rate, number_of_days, moral_user
):
    """Stores weekly data for decision-making models."""
    df_input_storage_week = {}

    participation_rate_percentage = participation_rate * 10  # Precompute once

    for week in additional_time_week.keys():
        df = pd.DataFrame()
        df2 = df.assign(
            additional_time=additional_time_week[week],
            promise_additional_time=promise_additional_time_week[week],
            Network_benefit=Network_benefit_week[week],
            Network_benefit_unscaled=Network_benefit_unscaled_week[week],
            participation_rate=participation_rate_percentage,
            number_of_days=number_of_days,
        )
        df3 = pd.DataFrame.from_dict(moral_user, orient="index")
        df_input = pd.concat([df2, df3], axis=1)
        df_input_storage_week[week] = df_input

    return df_input_storage_week


def store_cycle_mean_data(
        iteration_cycle, additional_time_week, promise_additional_time_week, Network_benefit_week,
        Network_benefit_unscaled_week, participation_rate, number_of_days, moral_user
):
    """Stores the cycle-mean data for decision-making models."""

    # Compute mean values over all weeks in the cycle
    additional_time_mean = {user_id: np.mean([additional_time_week[week][user_id] for week in additional_time_week])
                            for user_id in additional_time_week[1].index}

    promise_additional_time_mean = {
        user_id: np.mean([promise_additional_time_week[week][user_id] for week in promise_additional_time_week])
        for user_id in promise_additional_time_week[1].index}

    Network_benefit_mean = np.mean(list(Network_benefit_week.values()))
    Network_benefit_unscaled_mean = np.mean(list(Network_benefit_unscaled_week.values()))

    participation_rate_percentage = participation_rate * 10  # Convert decimal to percentage scale

    # Create DataFrame for cycle mean
    df = pd.DataFrame()
    df2 = df.assign(
        additional_time=additional_time_mean,
        promise_additional_time=promise_additional_time_mean,
        Network_benefit=Network_benefit_mean,
        Network_benefit_unscaled=Network_benefit_unscaled_mean,
        participation_rate=participation_rate_percentage,
        number_of_days=number_of_days,
    )

    df3 = pd.DataFrame.from_dict(moral_user, orient="index")
    df_input = pd.concat([df2, df3], axis=1)

    return df_input, promise_additional_time_mean


def compute_user_decisions(df_input, user_choices, user_param_store, number_of_users):
    """Computes user probabilities and choices for scheme participation."""

    df_out = pd.DataFrame()

    # Compute utilities and probabilities for Sacrifice and Collective Good
    utility_S_out = df_input.apply(lambda x: user_choices.utility_SR(x.name, x, user_param_store['Sacrifice']), axis=1)
    prob_S_out = user_choices.prob_SR(utility_S_out)

    utility_CG_out = df_input.apply(lambda x: user_choices.utility_SR(x.name, x, user_param_store['CollectiveGood']),
                                    axis=1)
    prob_CG_out = user_choices.prob_SR(utility_CG_out)

    df_out = df_out.assign(utility_S=utility_S_out, prob_S=prob_S_out, utility_CG=utility_CG_out, prob_CG=prob_CG_out)

    # Simulate random user decisions
    df_random = pd.DataFrame({'value': np.random.rand(number_of_users)})  # Random values between 0 and 1
    df_out['choice_S'] = df_random['value'] <= df_out['prob_S']
    df_out['choice_CG'] = df_random['value'] <= df_out['prob_CG']

    return df_out


def update_participation_status(df_out, scheme_type, select_social,
                                promise_additional_time_mean, additional_time_rel_tol,
                                travel_time_init, lost_user, number_of_users):
    """Updates participation status based on user choices and scheme type."""

    # Identify users who had a bad experience (relative threshold with no forgiveness)
    for user_id in range(number_of_users):
        if select_social[user_id] and promise_additional_time_mean[user_id] > (
                additional_time_rel_tol * travel_time_init['travel_time'][user_id] / 60
        ):
            lost_user[user_id] = True

    # Remove "lost" users from the Collective Good scheme
    df_out.loc[df_out['choice_CG'] & df_out.index.map(lost_user)] = False

    # Update participation rate
    if scheme_type == 'Sacrifice':
        join_scheme_S = df_out['choice_S'].sum()
        participation_rate = join_scheme_S / number_of_users
    else:
        join_scheme_CG = df_out['choice_CG'].sum()
        participation_rate = join_scheme_CG / number_of_users



    # Update social participation status
    if scheme_type == 'Sacrifice':
        select_social = df_out['choice_S'].to_dict()
    else:
        select_social = df_out['choice_CG'].to_dict()

    return participation_rate, select_social, lost_user


#%% Simulation Inputs
def main():
     # ----------------- # ----------------- # ----------------- # ----------------- 

    # ----------------- User inputs - change as needed  ----------------- # ----------------- 

        # ----------------- # ----------------- # ----------------- # ----------------- 

    parallel_cores=40 # standard is 40 for the paper simulation results

    max_rerun_attempts_days=4 # number of reruns of weeks of the simulation to avoid gridlock issues with Symuvia.

    max_attempts = 5 # max number of attempts if there Symuvia returns a run error.

    no_sim=0

    sim_hours = 4 # change to 2 for testing 0630 to 0830, 4 for full results 0630 to 1030

    weeks_in_cycle = 4 # standard is 4 for the paper simulation results

    number_of_cycle = 10 # 10 cycles of participation are to be simulated for results in paper.

    days_in_week = 5 # each week has 5 days (i.e. work week)

    additional_time_rel_tol = 0.2  # Sets the additional participation rule for the Collective Good scheme

    if len(sys.argv) > 1: # used to load in a counter from the shell script that can select scheme type, sensitivity analysis.
        selector = int(sys.argv[1])

    initial_participation_list = [20] # all experiments in paper have 20%, this list could contain alternative tests.
    social_days_number_list = [2] # all experiments in paper have 2 SR days, this list could contain alternative tests.

    # run_list is used to indicate moral_profile used.
    run_list = [1,2,3] # 1 is survey moral profile, 2 is most inclined, 3 is least inclined moral profile

    # expnum_list is used to list the social routing scheme types
    expnum_list = ['1', '2']  # 1 is sacrifice, 2 is collective good

    # ----------------- # ----------------- # ----------------- # ----------------- 
    # ----------------- # ----------------- # ----------------- # ----------------- 


    combinations_1 = list(product(run_list, expnum_list)) # [(1, '1'), (1, '2'), (2, '1'), (2, '2'), (3, '1'), (3, '2')]
    selected_combination_1 = combinations_1[selector-1]  # Adjust for 0-based indexing

    run_num, expnum = selected_combination_1

    combinations = list(product(initial_participation_list, social_days_number_list))
    selected_combination = combinations[0]  # Adjust for 0-based indexing

    participation_rate1, social_days_number = selected_combination
    participation_rate=participation_rate1/100


    print(f"Run_num and expnum: {selected_combination_1}")
    print(f"Initial Participation Rate: {100*participation_rate}%")
    print(f"Days of Commitment to SR: {social_days_number}")

    if expnum == '1':
        scheme_type = 'Sacrifice'
    else:
        scheme_type='CollectiveGood'


    weeks_in_stack=parallel_cores//days_in_week
    number_stack_simulations=weeks_in_stack*days_in_week



    shuffle_participation=True # choice to shuffle which days of the week participants use the social route on each week of cycle.

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

    # Demand loading
    veh_social_1 = pd.read_csv(demand_input_social, sep=";")  # columns: origin;typeofvehicle;creation;path;destination
    veh_init_1 = pd.read_csv(demand_input_init, sep=";")  # columns: origin;typeofvehicle;creation;path;destination

    # merge the rows in the two dataframes so that UC and SR have the same users in each.
    veh_social = pd.merge(veh_social_1, veh_init_1[['origin', 'destination', 'creation_time']], on=['origin', 'destination', 'creation_time'], how='inner')
    veh_init = pd.merge(veh_init_1, veh_social[['origin', 'destination', 'creation_time']], on=['origin', 'destination', 'creation_time'], how='inner')


    number_of_users = len(veh_init['id'])
    print(f'number_of_users in simulation is {number_of_users}')

    #--- Baseline Travel Time ----
    # Construct the path to the pickle file
    pickle_file_path1 = os.path.join(parent_dir, '3_Complete_SR_and_Baseline_UC', 'macro_output_completeSR_mean.pickle')

    # Open and load the pickle file
    with open(pickle_file_path1, 'rb') as file:
        total_travel_time_completeSR, travel_time_completeSR = pickle.load(
            file)

    pickle_file_path1 = os.path.join(parent_dir, '3_Complete_SR_and_Baseline_UC', 'macro_output_fullUC_mean.pickle')

    # Open and load the pickle file
    with open(pickle_file_path1, 'rb') as file:
        total_travel_time_meanUC, travel_time_init  = pickle.load(file)

    Network_benefit_All = (total_travel_time_completeSR - total_travel_time_meanUC) * 5 #values are originally in hours for one day so muliply by 5 for a 5-day week.

    print(f'Simulated_SO_total_travel_time is {total_travel_time_completeSR}')
    print(f'Simulated_UC_total_travel_time is {total_travel_time_meanUC}')

    print(f'Network_benefit_All is {Network_benefit_All}')

    #--- Simulation Details ----

    start_sim_time='06300'
    if 6+sim_hours>=10:
        finish_sim_time = f'{6 + sim_hours}300'
    else:
        finish_sim_time = f'0{6+sim_hours}300'
    symuvia_output = f"_{start_sim_time}0_{finish_sim_time}0_traf.xml"

    #--- Participation Rate ----

    participation_rate_store=dict()
    select_social_store=dict()

    participation_rate_store[0] = participation_rate
    #--- Number of days using social route profile ----
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

    current_dir = os.getcwd()

    # -----SymuVia_Lyon_Inputs---------
    User_and_Moral_Profile_path = os.path.join(current_dir, os.pardir, '2_User_and_Moral_Profile')
    User_and_Moral_Profile_path = os.path.abspath(User_and_Moral_Profile_path)

    user_profile_path = os.path.join(User_and_Moral_Profile_path, 'user_profile.pickle')
    with open(user_profile_path, 'rb') as file:
        user_param_store, moral_user = pickle.load(file)

    if sim_hours==2:
        user_profile_path_1h = os.path.join(User_and_Moral_Profile_path, 'user_profile_1h.pickle')
        with open(user_profile_path_1h, 'rb') as file:
            user_param_store, moral_user = pickle.load(file)

    # the next if statements apply the moral personality sensitivity analysis conditions if needed.
    if run_num == 1:
        print('Survey Moral Profile for all users applied')
    if run_num == 2:

        print('Most inclined Moral Profile for all users applied')
        if expnum == '1':
            print('Sacrifice, most inclined profile : care, fairness, authority ')

            for user_id in range(number_of_users):
                moral_user[user_id] = {'Care': 1, 'Fairness': 1, 'Ingroup': 0, 'Authority': 1, 'Purity': 0}
        if expnum == '2':
            print('Collective Good, most inclined profile : care, fairness, authority and purity ')

            for user_id in range(number_of_users):
                moral_user[user_id] = {'Care': 1, 'Fairness': 1, 'Ingroup': 0, 'Authority': 1, 'Purity': 1}

    if run_num == 3:
        print('Least inclined Moral Profile for all users applied')
        if expnum == '1':
            print('Sacrifice, least inclined profile : ingroup and purity ')

            for user_id in range(number_of_users):
                moral_user[user_id] = {'Care': 0, 'Fairness': 0, 'Ingroup': 1, 'Authority': 0, 'Purity': 1}
        if expnum == '2':
            print('Collective Good, least inclined profile : ingroup')
            for user_id in range(number_of_users):
                moral_user[user_id] = {'Care': 0, 'Fairness': 0, 'Ingroup': 1, 'Authority': 0, 'Purity': 0}


    #%% loop over the days
    df_input_storage_week_cycle=dict()
    df_input_cyclemean_storage = dict()
    df_out_storage=dict()

    current_dir = os.getcwd()

    # Create a folder to save the results:
    results_folder = os.path.join(current_dir, os.pardir, "Output", "Simulations")

    folder_outputs = os.path.join(results_folder, f'MethodPaper_simhours_{sim_hours}_expnum_{expnum}_initialparticipation_{participation_rate*100}_SRdays_{social_days_number}_randomInitial_run_{run_num}')

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


    lost_user = {user_id: False for user_id in range(number_of_users)}

    select_social_inital=select_social

    cycle_social_days={}
    lost_user_count={}
    cycle_key_selection_store={}

    print(f'Scheme Type is set to: {scheme_type}')
    cycle_start_time= time.time()
    for iteration_cycle in range(1,number_of_cycle+1):

        print('-----------------------------------------------')
        print(f'Cycle: {iteration_cycle}')
        print('---------------------------------------------')
        df_input_storage_week = dict()


        number_stack_simulations_rerun = 0
        rerun_attempts = 0
        weeks_in_stack_total = 0
        count_zero_problem_weeks = 0
        week_social_day_store={}

        while count_zero_problem_weeks < weeks_in_cycle:
            rerun_attempts += 1
            if rerun_attempts == max_rerun_attempts_days:
                print(f'Too many rerun attempts in Cycle: {iteration_cycle}')
                break
            number_stack_simulations_rerun += number_stack_simulations
            weeks_in_stack_total += weeks_in_stack

            week_social_day_new = generate_cycle_social_days(number_of_users, weeks_in_stack_total, weeks_in_stack,
                                                         shuffle_participation, select_social_day_list)
            week_social_day_store.update(week_social_day_new)
            demand_rerun = generate_demand(week_social_day_new, veh_social, veh_init, number_of_users, select_social,
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
                                                                           temp_folder, symuvia_output, sim_hours,
                                                                           days_in_week, weeks_in_stack_total)

            # Calculate the number of weeks with at least one day with a gridlock problem.

            count_zero_problem_weeks = sum(1 for value in problems_in_week.values() if value == 0)

            print(
                f'Number of weeks without a simulation problem in cycle: {iteration_cycle}, is {count_zero_problem_weeks}')

        if count_zero_problem_weeks < weeks_in_cycle:
            print(
                f'After attempting reruns, number of weeks in stack without a problem remains too high. Exiting simulation...')
            break

        # ---------------------------------------------------
        sorted_keys_problems_in_week = [key for key, value in
                                        sorted(problems_in_week.items(), key=lambda item: item[1])]
        cycle_key_selection = sorted_keys_problems_in_week[
                              0:weeks_in_cycle]  # e.g. choose 4 weeks with no problems from those available
        print(f'Cycle key selection : {cycle_key_selection}')

        # print(f'veh_outputs_week: {veh_outputs_week}')
        # Reset the week numbers to be 1 to weeks_in_cycle, e.g 1,2,3,4.
        veh_outputs_week = {new_key: veh_outputs_week[old_key] for new_key, old_key in
                            enumerate(cycle_key_selection, start=1)}
        # print(f'week_social_day_store: {week_social_day_store}')

        week_social_day = {new_key: week_social_day_store[old_key] for new_key, old_key in
                            enumerate(cycle_key_selection, start=1)} # save the social day info for the selected weeks from the stack.

        # -------------------------------------
        Network_benefit_week, Network_benefit_unscaled_week = calculate_network_benefit(
            veh_outputs_week, weeks_in_cycle, travel_time_init, Network_benefit_All, days_in_week
        )

        additional_time_week, promise_additional_time_week = calculate_additional_time(
            veh_outputs_week, weeks_in_cycle, select_social_day, select_social, number_of_users, days_in_week,
            social_days_number, travel_time_init)

        df_input_storage_week = store_weekly_data(additional_time_week, promise_additional_time_week, Network_benefit_week,
    Network_benefit_unscaled_week, participation_rate, number_of_days, moral_user
        )

        df_input_cycle_mean, promise_additional_time_mean= store_cycle_mean_data(
            iteration_cycle, additional_time_week, promise_additional_time_week,
            Network_benefit_week, Network_benefit_unscaled_week, participation_rate,
            number_of_days, moral_user
        )

        df_out = compute_user_decisions(df_input_cycle_mean, user_choices, user_param_store, number_of_users)


        participation_rate, select_social, lost_user = update_participation_status(df_out, scheme_type, select_social,
                                    promise_additional_time_mean, additional_time_rel_tol,
                                    travel_time_init, lost_user, number_of_users)

        count_lost = 0
        for value in lost_user.values():
            if value:
                count_lost += 1

        # Store results for this cycle
        cycle_social_days[iteration_cycle] = week_social_day
        cycle_key_selection_store[iteration_cycle]= cycle_key_selection #Weeks selected for the 4 weeks of the cycle
        df_input_cyclemean_storage[iteration_cycle] = df_input_cycle_mean
        df_input_storage_week_cycle[iteration_cycle] = df_input_storage_week
        participation_rate_store[iteration_cycle] = participation_rate
        df_out_storage[iteration_cycle] = df_out
        select_social_store[iteration_cycle] = select_social
        lost_user_count[iteration_cycle] = count_lost

    file_path1 = os.path.join(folder_outputs, 'output.pickle')

    with open(file_path1, 'wb') as file:
        pickle.dump(
            (cycle_key_selection_store,df_input_cyclemean_storage,df_input_storage_week_cycle,participation_rate_store, df_out_storage,folder_outputs,select_social_store,lost_user_count), file)


if __name__ == '__main__':
    start_time = time.time()

    main()
#
#
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600

    # network_para = [user_param_store["Sacrifice"][key]['Network_benefit'] for key in user_param_store["Sacrifice"]]
    print(f"Elapsed time: {elapsed_time:.2f} hours")
#
