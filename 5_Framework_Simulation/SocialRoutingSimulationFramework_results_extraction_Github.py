#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/10/23

Author: Alexander Roocroft
"""

# from multiprocessing import Pool
import zipfile
import multiprocessing
import pandas as pd
import pickle
from itertools import product
import os

def process_experiment_DTD(expnum, initial_participation_rate, result_dict, travel_time_init, number_of_users, sim_hours,days_in_week, days_in_cycle,social_days_number,run_num,results_folder):

    folder_outputs = os.path.join(results_folder, f'MethodPaper_simhours_{sim_hours}_expnum_{expnum}_initialparticipation_{initial_participation_rate}_SRdays_{social_days_number}_randomInitial_run_{run_num}')

    Network_benefit_day={}

    nonparticipant_additional_time_day={}
    fullparticipant_additional_time_day={}
    nonparticipant_experienced_time_day={} # nonparticipant additional time on all days
    fullparticipant_experienced_time_day={}
    allusers_experienced_time_day={}

    day_count=0

    file_path1 = os.path.join(folder_outputs, 'output.pickle')

    with open(file_path1, 'rb') as file:
        (cycle_key_selection_store,df_input_cyclemean_storage,df_input_storage_week_cycle,participation_rate_store, df_out_storage,_,select_social_store,lost_user_count) = pickle.load(
            file)

    for iteration_cycle in cycle_list:
        print(
            f'Running expnum:{expnum},initialparticipation:{initial_participation_rate}, SR days: {social_days_number},cycle:{iteration_cycle}')

        select_social = select_social_store[iteration_cycle - 1]
        df_input_storage_week = dict()

        out_folder = os.path.join(folder_outputs, 'OUT')

        start_sim_time = '06300'
        if 6 + sim_hours >= 10:
            finish_sim_time = f'{6 + sim_hours}300'
        else:
            finish_sim_time = f'0{6 + sim_hours}300'
        symuvia_output = f"_{start_sim_time}0_{finish_sim_time}0_traf.xml"

        weeks_in_cycle=int(days_in_cycle/days_in_week)

        cycle_key_selection = cycle_key_selection_store[iteration_cycle]
        days_of_data_in_cycle = []

        # Loop through each week in cycle_key_selection
        for week in cycle_key_selection:
            # Calculate the starting day of the week
            start_day = (week - 1) * 5 + 1
            # Create a list of the 5 consecutive days for that week
            week_days = list(range(start_day, start_day + 5))
            # Add the days for this week to the main list
            days_of_data_in_cycle.extend(week_days)

        # Print the result
        print(f'Cycle {iteration_cycle} has days_of_data_in_cycle: {days_of_data_in_cycle}')

        veh_outputs_week = {}
        veh_outputs = []
        count = 0
        week_num = 1
        for iteration_day in days_of_data_in_cycle:
            count += 1
            # # Use the trip extraction script
            # veh_output = Tripextract.symmvia_output(out_folder_parallel, symuvia_output_parallel, iteration_day, iteration_cycle, sim_hours)
            file_path = os.path.join(out_folder, f'vehs_Day_{iteration_day}_cycle_{iteration_cycle}.csv')

            # Load the CSV file into a DataFrame
            veh_pd = pd.read_csv(file_path, sep=';')

            veh_output = dict()
            veh_output['driver_travel_time'] = veh_pd['travel_time']

            veh_outputs.append(veh_output)

        for day_output_number, veh_output in enumerate(veh_outputs):
            day_count += 1
            # Network_benefit_user += veh_output['network_stats']['total_time'] - init_total_time
            Network_benefit_user = veh_output['driver_travel_time'].sum() - travel_time_init[
                'travel_time'].sum()

            Network_benefit_day[day_count]=Network_benefit_user

            experienced_travel_time_day = pd.Series(0.0, index=range(number_of_users), name='travel_time')

            for user_id in range(number_of_users):
                experienced_travel_time_day[user_id] = veh_output['driver_travel_time'][user_id]

            experienced_time_users_allusers = (experienced_travel_time_day / 60)


            additional_time_day = pd.Series(0.0, index=range(number_of_users), name='travel_time')
            experienced_travel_time_day = pd.Series(0.0, index=range(number_of_users), name='travel_time')
            for user_id in range(number_of_users):
                user_pariticpating = select_social[user_id]

                if user_pariticpating:
                    additional_time_day[user_id] = (
                            veh_output['driver_travel_time'][user_id] - travel_time_init['travel_time'][
                        user_id])
                    experienced_travel_time_day[user_id] = veh_output['driver_travel_time'][user_id]

            additional_time_users_fullparticipant = (additional_time_day / 60)
            experienced_time_users_fullparticipant = (experienced_travel_time_day / 60)


            additional_time_day = pd.Series(0.0, index=range(number_of_users), name='travel_time')
            experienced_travel_time_day = pd.Series(0.0, index=range(number_of_users), name='travel_time')
            for user_id in range(number_of_users):
                user_pariticpating = select_social[user_id]

                if not user_pariticpating:
                    additional_time_day[user_id] = (
                            veh_output['driver_travel_time'][user_id] - travel_time_init['travel_time'][
                        user_id])
                    experienced_travel_time_day[user_id] = veh_output['driver_travel_time'][user_id]


            # Calculate for each user taking part in the scheme
            additional_time_users_nonparticipant = (additional_time_day / 60)
            experienced_time_users_nonparticipant = (experienced_travel_time_day / 60)

            fullparticipant_additional_time = additional_time_users_fullparticipant.loc[
                additional_time_users_fullparticipant.index.isin(
                    [k for k, v in select_social.items() if v])]
            nonparticipant_additional_time = additional_time_users_nonparticipant.loc[
                additional_time_users_nonparticipant.index.isin(
                    [k for k, v in select_social.items() if not v])]
            fullparticipant_experienced_time = experienced_time_users_fullparticipant.loc[
                experienced_time_users_fullparticipant.index.isin(
                    [k for k, v in select_social.items() if v])]
            nonparticipant_experienced_time = experienced_time_users_nonparticipant.loc[
                experienced_time_users_nonparticipant.index.isin(
                    [k for k, v in select_social.items() if not v])]

            nonparticipant_additional_time_day[day_count] = nonparticipant_additional_time  # nonparticipant additional time on all days
            fullparticipant_additional_time_day[day_count] = fullparticipant_additional_time
            nonparticipant_experienced_time_day[day_count] = nonparticipant_experienced_time  # nonparticipant additional time on all days
            fullparticipant_experienced_time_day[day_count] = fullparticipant_experienced_time
            allusers_experienced_time_day[day_count] = experienced_time_users_allusers

        print(f'Finished expnum:{expnum},initialparticipation:{initial_participation_rate}, SR days: {social_days_number}. cycle:{iteration_cycle}')

    df_time_output_day = {'Network_benefit_day': Network_benefit_day,
                            'additional_time_participants_day': fullparticipant_additional_time_day,
                                           'additional_time_nonparticipants_day': nonparticipant_additional_time_day,
                                           'experienced_time_participants_day': fullparticipant_experienced_time_day,
                                           'experienced_time_nonparticipants_day': nonparticipant_experienced_time_day,
                                           'experienced_time_allusers_day': allusers_experienced_time_day}
   # When you want to store results, you can do so in result_dict
    result_dict[(expnum, initial_participation_rate,days_in_cycle,social_days_number)] = df_time_output_day



def macro_extract(initial_participation_list,sim_hours,days_in_cycle_list,social_days_number_list,combinations, results_folder):

    participation_rate_store_list_exp={}
    Network_benefit_mean_list_exp={}
    lost_user_count_exp={}

    current_dir = os.getcwd()

    # -----SymuVia_Lyon_Inputs---------
    SymuVia_Lyon_Inputs_path = os.path.join(current_dir, os.pardir, '1_SymuVia_Lyon_Inputs')
    SymuVia_Lyon_Inputs_path = os.path.abspath(SymuVia_Lyon_Inputs_path)

    demand_4h_path = os.path.join(SymuVia_Lyon_Inputs_path, '4_hour_demand_profile')

    demand_input_social = os.path.join(demand_4h_path, 'SR_routes.csv')
    demand_input_init = os.path.join(demand_4h_path, 'UC_routes.csv')


    if sim_hours == 2:
        demand_1h_path = os.path.join(SymuVia_Lyon_Inputs_path, '1_hour_demand_profile')

        demand_input_social = os.path.join(demand_1h_path, 'social_routes_1h.csv')
        demand_input_init = os.path.join(demand_1h_path, 'init_routes_1h.csv')

    # Demand loading
    veh_social_1 = pd.read_csv(demand_input_social, sep=";")  # columns: origin;typeofvehicle;creation;path;destination
    veh_init_1 = pd.read_csv(demand_input_init, sep=";")  # columns: origin;typeofvehicle;creation;path;destination

    veh_social = pd.merge(veh_social_1, veh_init_1[['origin', 'destination', 'creation_time']],
                          on=['origin', 'destination', 'creation_time'], how='inner')
    veh_init = pd.merge(veh_init_1, veh_social[['origin', 'destination', 'creation_time']],
                        on=['origin', 'destination', 'creation_time'], how='inner')

    total_stats = {}
    total_stats['total_full_UE'] = veh_init['travel_time'].sum() / 3600
    total_stats['total_full_SO'] = veh_social['travel_time'].sum() / 3600

    total_stats['POA'] = veh_init['travel_time'].sum() / veh_social['travel_time'].sum()
    #
    number_users = len(veh_social['travel_time'])

    for combo in combinations:
        for initial_participation_rate in initial_participation_list:
            for days_in_cycle in days_in_cycle_list:
                for social_days_number in social_days_number_list:

                    run_num, expnum = combo

                    print(f'run_num: {run_num}')
                    print(f'expnum: {expnum}')

                    folder_outputs = os.path.join(results_folder,
                                                  f'MethodPaper_simhours_{sim_hours}_expnum_{expnum}_initialparticipation_{initial_participation_rate}_SRdays_{social_days_number}_randomInitial_run_{run_num}')

                    with open(os.path.join(folder_outputs, 'output.pickle'), 'rb') as file:
                        cycle_key_selection_store, df_input_cyclemean_storage, df_input_storage_week_cycle, participation_rate_store, df_out_storage, folder_outputs1, select_social_store, lost_user_count = pickle.load(
                            file)

                    participation_rate_store_list = []

                    for key in participation_rate_store:
                        participation_rate_store_list.append(participation_rate_store[key])

                    Network_benefit_mean_list = []
                    for key, df in df_input_cyclemean_storage.items():
                        Network_benefit_mean_list.append(df["Network_benefit_unscaled"].mean() / 1000)


                    participation_rate_store_list_exp[(expnum,run_num,initial_participation_rate,days_in_cycle,social_days_number)]=participation_rate_store_list
                    Network_benefit_mean_list_exp[(expnum,run_num,initial_participation_rate,days_in_cycle,social_days_number)]=Network_benefit_mean_list
                    lost_user_count_exp[(expnum,run_num,initial_participation_rate,days_in_cycle,social_days_number)]=[(item / number_users) * 100 for item in lost_user_count.values()]

        # Define the full file path
    file_path = os.path.join(results_folder, "macro_output.pickle")

    # Save the pickle file in the specified folder
    with open(file_path, 'wb') as file:
        pickle.dump(
            (Network_benefit_mean_list_exp,participation_rate_store_list_exp,lost_user_count_exp), file)

if __name__ == '__main__':


         # ----------------- # ----------------- # ----------------- # ----------------- 

    # ----------------- User inputs - make sure matches output from simulations ----------------- # ----------------- 

        # ----------------- # ----------------- # ----------------- # ----------------- 

    sim_hours = 4

    run_list = [1,2,3] # For moral personality sensitivity: 1 is survey moral profile, 2 is most inclined, 3 is least inclined moral profile

    expnum_list = ['1', '2']  # 1 is sacrifice, 2 is collective good no forgiveness
    experiment_list = [int(num) for num in expnum_list] # make expnum_list into integers for use later.

    combinations = list(
        product(run_list, expnum_list))  # [(1, '1'), (1, '2'), (2, '1'), (2, '2'), (3, '1'), (3, '2')]

    
    initial_participation_list = [20]
    social_days_number_list = [2]
    number_of_cycle = 10
    cycle_list=list(range(1,number_of_cycle+1))


    days_in_week = 5

    days_in_cycle_list=[20]

    # ----------------- # ----------------- # ----------------- # ----------------- 
    # ----------------- # ----------------- # ----------------- # ----------------- 



    current_dir = os.getcwd()

    results_folder = os.path.join(current_dir, os.pardir, "Output", "Simulations")
    # Ensure the directory exists
    os.makedirs(results_folder, exist_ok=True)

    macro_extract(initial_participation_list,sim_hours,days_in_cycle_list,social_days_number_list,combinations, results_folder)


    for run_num in run_list:
        current_dir = os.getcwd()

        # -----SymuVia_Lyon_Inputs---------
        SymuVia_Lyon_Inputs_path = os.path.join(current_dir, os.pardir, '1_SymuVia_Lyon_Inputs')
        SymuVia_Lyon_Inputs_path = os.path.abspath(SymuVia_Lyon_Inputs_path)

        demand_4h_path = os.path.join(SymuVia_Lyon_Inputs_path, '4_hour_demand_profile')

        demand_input_social = os.path.join(demand_4h_path, 'SR_routes.csv')
        demand_input_init = os.path.join(demand_4h_path, 'UC_routes.csv')

        if sim_hours == 2:
            demand_1h_path = os.path.join(SymuVia_Lyon_Inputs_path, '1_hour_demand_profile')

            demand_input_social = os.path.join(demand_1h_path, 'social_routes_1h.csv')
            demand_input_init = os.path.join(demand_1h_path, 'init_routes_1h.csv')

        # Demand loading
        veh_social_1 = pd.read_csv(demand_input_social, sep=";")  # columns: origin;typeofvehicle;creation;path;destination
        veh_init_1 = pd.read_csv(demand_input_init, sep=";")  # columns: origin;typeofvehicle;creation;path;destination

        veh_social = pd.merge(veh_social_1, veh_init_1[['origin', 'destination', 'creation_time']],
                              on=['origin', 'destination', 'creation_time'], how='inner')
        veh_init = pd.merge(veh_init_1, veh_social[['origin', 'destination', 'creation_time']],
                            on=['origin', 'destination', 'creation_time'], how='inner')

        number_of_users = len(veh_init['id'])
        travel_time_init = veh_init[['id', 'travel_time']]

        df_time_output_exp = {}

        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        # Create a pool of workers to run the processes in parallel
        pool = multiprocessing.Pool()

        print(f'initial rate is {initial_participation_list}')

        # Use pool.starmap to iterate over both experiment_list and initial_participation_list in parallel
        pool.starmap(process_experiment_DTD,
                     [(expnum, initial_participation_rate, result_dict, travel_time_init, number_of_users, sim_hours,days_in_week,days_in_cycle,social_days_number,run_num,results_folder) for expnum in experiment_list for initial_participation_rate in
                      initial_participation_list for days_in_cycle in days_in_cycle_list for social_days_number in social_days_number_list])


        # Close the pool and wait for all processes to complete
        pool.close()
        pool.join()



        print(f'Completed Extraction...run {run_num}')
        df_time_output_exp={}
        # You can now access the results from the result_dict dictionary
        for (expnum, participation_rate,days_in_cycle,social_days_number), data in result_dict.items():

            df_time_output_exp[(expnum, participation_rate,days_in_cycle,social_days_number)]=data


        pickle_file = f'individual_users_output_parallel_SRdays_randomInitial_methodPaper_DTD_run_{run_num}.pickle'
        zip_file = f'individual_users_output_parallel_SRdays_randomInitial_methodPaper_DTD_run_{run_num}.zip'

        file_path_pickle = os.path.join(results_folder, pickle_file)
        file_path_zip = os.path.join(results_folder, zip_file)

        # Save the pickle file
        with open(file_path_pickle, 'wb') as file:
            pickle.dump(df_time_output_exp, file)

        # Create a zip archive and add the pickle file to it
        with zipfile.ZipFile(file_path_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(file_path_pickle, pickle_file)  # Fix: Use full path


        print(f'Pickle file {pickle_file} has been zipped to {zip_file} .')

