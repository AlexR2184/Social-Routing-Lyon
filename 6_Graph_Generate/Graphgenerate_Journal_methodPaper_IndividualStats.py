#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/4/2023

Author: Alexander Roocroft
"""

from multiprocessing import Pool
import os
import sys

# For vectorized math operations
import numpy as np
# For plotting
import matplotlib.pyplot as plt
import random
import math
import statistics
import datetime

import xml.sax
import pandas as pd
import pickle
# For recording the model specification
from collections import OrderedDict
import csv


import time

# your code here


#%% Simulation Inputs

if __name__ == '__main__':
      # ----------------- # ----------------- # ----------------- # ----------------- 

    # ----------------- User inputs - make sure it matches the simulations  ----------------- # ----------------- 

        # ----------------- # ----------------- # ----------------- # ----------------- 
    run_num=1

    experiment_list = [1,2]
    initial_participation_list = [20]
    social_days_number_list = [2]

    sim_hours = 4
    num_days_in_cycle = 20 # for main simulations should be 20, based on sample size analysis.

    upperbound = 90
    lowerbound = 10


    # ----------------- # ----------------- # ----------------- # ----------------- 
    # ----------------- # ----------------- # ----------------- # ----------------- 


    individual_users_output_execute=0

    start_time = time.time()

    lost_user_count_exp={}
    participation_rate_store_list_exp={}
    Network_benefit_mean_list_exp={}
    Additional_time_mean_list_exp={}
    additional_time_nonparticipants_list_exp = {}
    additional_time_participants_nonsocialdays_list_exp = {}
    additional_time_participants_list_exp={}
    additional_time_participants_plot_exp={}
    additional_time_nonparticipants_plot_exp={}

    current_dir = os.getcwd()

    results_folder = os.path.join(current_dir, os.pardir, "Output", "Simulations")
    results_folder_figs = os.path.join(current_dir, os.pardir, "Output", "Simulations", "figs","")


    file_path_pickle = os.path.join(results_folder, f'individual_users_output_parallel_SRdays_randomInitial_methodPaper_DTD_run_{run_num}.pickle')

    with open(file_path_pickle, 'rb') as f:
        df_time_output_exp_store = pickle.load(f)

    additional_time_participants_plot_exp2 = {}
    additional_time_nonparticipants_plot_exp2 = {}
    initial_participation_rate_index=1
    for exp_num in experiment_list:

        for initial_participation_rate in initial_participation_list:

            
            for social_days_number in social_days_number_list:

                df_exp_store = df_time_output_exp_store[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)]

                additional_time_nonparticipants_plot2 = []
                additional_time_participants_plot2 = []

                values_dict_day = df_exp_store["additional_time_nonparticipants_day"]

                # Create a new dictionary to store the averaged Series
                values_dict = {}

                # Iterate over every 20 keys
                for i in range(0, len(values_dict_day), num_days_in_cycle):
                    # Extract the current 20 keys
                    keys_subset = list(values_dict_day.keys())[i:i + num_days_in_cycle]

                    # Extract the corresponding Series
                    series_subset = [values_dict_day[key] for key in keys_subset]

                    # Concatenate the Series along the columns (axis=1) to form a DataFrame
                    df_subset = pd.concat(series_subset, axis=1)

                    # Calculate the mean along the columns to obtain the averaged Series
                    averaged_series = df_subset.mean(axis=1)

                    # Create a new key for the averaged Series
                    averaged_key = 1+ (i // num_days_in_cycle)  # Adjust the key naming as needed

                    # Store the averaged Series in the new dictionary
                    values_dict[averaged_key] = averaged_series


                for key, values in values_dict.items():

                    max_value = np.max(values)
                    min_value = np.min(values)
                    percentile_75 = np.nanpercentile(values, upperbound)
                    percentile_25 = np.nanpercentile(values, lowerbound)
                    median = np.nanmedian(values)
                    mean = np.nanmean(values)


                    additional_time_nonparticipants_plot2.append({
                        'max': max_value,
                        'min': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'mean': mean,
                        'cycle_nonparticipants' : len(values),
                    })

                values_dict_day = df_exp_store["additional_time_participants_day"]

                # Create a new dictionary to store the averaged Series
                values_dict = {}

                # Iterate over every 20 keys
                for i in range(0, len(values_dict_day), num_days_in_cycle):
                    # Extract the current 20 keys
                    keys_subset = list(values_dict_day.keys())[i:i + num_days_in_cycle]

                    # Extract the corresponding Series
                    series_subset = [values_dict_day[key] for key in keys_subset]

                    # Concatenate the Series along the columns (axis=1) to form a DataFrame
                    df_subset = pd.concat(series_subset, axis=1)

                    # Calculate the mean along the columns to obtain the averaged Series
                    averaged_series = df_subset.mean(axis=1)

                    # Create a new key for the averaged Series
                    averaged_key = 1 + (i // num_days_in_cycle)  # Adjust the key naming as needed

                    # Store the averaged Series in the new dictionary
                    values_dict[averaged_key] = averaged_series

                for key, values in values_dict.items():

                    max_value = np.max(values)
                    min_value = np.min(values)
                    percentile_75 = np.nanpercentile(values, upperbound)
                    percentile_25 = np.nanpercentile(values, lowerbound)
                    median = np.nanmedian(values)
                    mean = np.nanmean(values)

                    additional_time_participants_plot2.append({
                        'max': max_value,
                        'min': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'mean': mean,
                        'cycle_participants': len(values),
                    })


                additional_time_participants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)] = additional_time_participants_plot2
                additional_time_nonparticipants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)] = additional_time_nonparticipants_plot2

    total_additional_time_participants={}
    total_additional_time_nonparticipants={}
    difference_exp={}

    mean_values_exp={}
    mean_values_non_exp={}
    number_of_participants_exp={}
    number_of_nonparticipants_exp={}

    for exp_num in experiment_list:
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))  # 3 rows, 1 column
        for initial_participation_rate in initial_participation_list:

            for social_days_number in social_days_number_list:

                print(f'scheme {exp_num}, initial rate {initial_participation_rate}')
                legend_handles = []

                data = additional_time_participants_plot_exp2[(exp_num,initial_participation_rate,num_days_in_cycle,social_days_number)]
                max_values = [entry['max'] for entry in data]
                min_values = [entry['min'] for entry in data]
                UQ_values = [entry['percentile_75'] for entry in data]
                LQ_values = [entry['percentile_25'] for entry in data]
                median_values = [entry['median'] for entry in data]
                mean_values = [entry['mean'] for entry in data]
                number_of_participants = [entry['cycle_participants'] for entry in data]

                # Create labels for x-axis
                labels = [f'{i + 1}' for i in range(len(data))]

                axs.plot(labels, UQ_values, label='90th percentile', marker='s')
                axs.plot(labels, LQ_values, label='10th percentile', marker='o')
                axs.plot(labels, median_values, label='Median', linewidth=2, marker='d')
                axs.plot(labels, mean_values, label='Mean', linewidth=2, marker='x')

                axs.axhline(0, color='grey', linestyle='--', alpha=0.5)  # You can customize color and linestyle

                axs.set_ylabel('Cycle Mean Additional Time  [min]', fontsize=14)
                axs.set_xlabel('Cycle', fontsize=14)
                axs.axhline(0, color='grey', linestyle='--', alpha=0.5)  # You can customize color and linestyle
                # axs.set_ylim(-30, 20)

                axs.tick_params(axis='both', which='major', labelsize=14)



        plt.subplots_adjust(hspace=.3, wspace=0.3)  # You can adjust the values as needed

        plt.tight_layout()
        plt.savefig(
            results_folder_figs + f'methodPaper_additionaltime_mean_scheme_{exp_num}_participants_SMALL.pdf',
            format='pdf', bbox_inches='tight')

        plt.show()

    for exp_num in experiment_list:
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))  # 3 rows, 1 column
        for initial_participation_rate in initial_participation_list:

            
            for social_days_number in social_days_number_list:
                print(f'scheme {exp_num}, initial rate {initial_participation_rate}')
                legend_handles = []

                data = additional_time_nonparticipants_plot_exp2[
                    (exp_num, initial_participation_rate, num_days_in_cycle, social_days_number)]
                max_values = [entry['max'] for entry in data]
                min_values = [entry['min'] for entry in data]
                UQ_values = [entry['percentile_75'] for entry in data]
                LQ_values = [entry['percentile_25'] for entry in data]
                median_values = [entry['median'] for entry in data]
                mean_values = [entry['mean'] for entry in data]
                number_of_participants = [entry['cycle_nonparticipants'] for entry in data]


                # Create labels for x-axis
                labels = [f'{i + 1}' for i in range(len(data))]

                axs.plot(labels, UQ_values, label='90th percentile', marker='s')
                axs.plot(labels, LQ_values, label='10th percentile', marker='o')
                axs.plot(labels, median_values, label='Median', linewidth=2, marker='d')
                axs.plot(labels, mean_values, label='Mean', linewidth=2, marker='x')

                axs.axhline(0, color='grey', linestyle='--', alpha=0.5)  # You can customize color and linestyle

                axs.set_ylabel('Cycle Mean Additional Time [min]', fontsize=14)
                axs.set_xlabel('Cycle', fontsize=14)
                axs.axhline(0, color='grey', linestyle='--', alpha=0.5)  # You can customize color and linestyle
                # axs.set_ylim(-30, 20)
                axs.tick_params(axis='both', which='major', labelsize=14)


        plt.subplots_adjust(hspace=.3, wspace=0.3)  # You can adjust the values as needed

        plt.tight_layout()
        plt.savefig(
            results_folder_figs + f'methodPaper_additionaltime_mean_scheme_{exp_num}_nonparticipants_SMALL.pdf',
            format='pdf', bbox_inches='tight')

        plt.show()


        #============================Experienced travel time==================================================



    experienced_time_participants_plot_exp2= {}
    experienced_time_nonparticipants_plot_exp2={}
    for exp_num in experiment_list:
        for initial_participation_rate in initial_participation_list:

            for social_days_number in social_days_number_list:

                df_exp_store = df_time_output_exp_store[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)]

                experienced_time_nonparticipants_plot2 = []
                experienced_time_participants_plot2 = []


                values_dict_day = df_exp_store["experienced_time_nonparticipants_day"]

                # Create a new dictionary to store the averaged Series
                values_dict = {}

                # Iterate over every 20 keys
                for i in range(0, len(values_dict_day), num_days_in_cycle):
                    # Extract the current 20 keys
                    keys_subset = list(values_dict_day.keys())[i:i + num_days_in_cycle]

                    # Extract the corresponding Series
                    series_subset = [values_dict_day[key] for key in keys_subset]

                    # Concatenate the Series along the columns (axis=1) to form a DataFrame
                    df_subset = pd.concat(series_subset, axis=1)

                    # Calculate the mean along the columns to obtain the averaged Series
                    averaged_series = df_subset.mean(axis=1)

                    # Create a new key for the averaged Series
                    averaged_key = 1+ (i // num_days_in_cycle)  # Adjust the key naming as needed

                    # Store the averaged Series in the new dictionary
                    values_dict[averaged_key] = averaged_series


                for key, values in values_dict.items():


                    max_value = np.max(values)
                    min_value = np.min(values)
                    percentile_75 = np.nanpercentile(values, upperbound)
                    percentile_25 = np.nanpercentile(values, lowerbound)
                    median = np.nanmedian(values)
                    mean = np.nanmean(values)

                    experienced_time_nonparticipants_plot2.append({
                        'max': max_value,
                        'min': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'mean': mean,
                        'cycle_nonparticipants': len(values),
                    })

                values_dict_day = df_exp_store["experienced_time_participants_day"]

                # Create a new dictionary to store the averaged Series
                values_dict = {}

                # Iterate over every 20 keys
                for i in range(0, len(values_dict_day), num_days_in_cycle):
                    # Extract the current 20 keys
                    keys_subset = list(values_dict_day.keys())[i:i + num_days_in_cycle]

                    # Extract the corresponding Series
                    series_subset = [values_dict_day[key] for key in keys_subset]

                    # Concatenate the Series along the columns (axis=1) to form a DataFrame
                    df_subset = pd.concat(series_subset, axis=1)

                    # Calculate the mean along the columns to obtain the averaged Series
                    averaged_series = df_subset.mean(axis=1)

                    # Create a new key for the averaged Series
                    averaged_key = 1 + (i // num_days_in_cycle)  # Adjust the key naming as needed

                    # Store the averaged Series in the new dictionary
                    values_dict[averaged_key] = averaged_series

                for key, values in values_dict.items():
                    max_value = np.max(values)
                    min_value = np.min(values)
                    percentile_75 = np.nanpercentile(values, upperbound)
                    percentile_25 = np.nanpercentile(values, lowerbound)
                    median = np.nanmedian(values)
                    mean = np.nanmean(values)

                    total_participants = len(values)

                    experienced_time_participants_plot2.append({
                        'max': max_value,
                        'min': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'mean': mean,
                        'cycle_participants': len(values),
                    })
            experienced_time_participants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)] = experienced_time_participants_plot2
            experienced_time_nonparticipants_plot_exp2[
                (exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)] = experienced_time_nonparticipants_plot2



    for exp_num in experiment_list:

        for initial_participation_rate in initial_participation_list:

            for social_days_number in social_days_number_list:

                print(f'scheme {exp_num}, initial rate {initial_participation_rate}')
                legend_handles = []

                data = experienced_time_participants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)]
                max_values = [entry['max'] for entry in data]
                min_values = [entry['min'] for entry in data]
                UQ_values = [entry['percentile_75'] for entry in data]
                LQ_values = [entry['percentile_25'] for entry in data]
                median_values = [entry['median'] for entry in data]
                mean_values = [entry['mean'] for entry in data]
                number_of_participants = [entry['cycle_participants'] for entry in data]


                data = experienced_time_nonparticipants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)]
                max_values = [entry['max'] for entry in data]
                min_values = [entry['min'] for entry in data]
                UQ_values = [entry['percentile_75'] for entry in data]
                LQ_values = [entry['percentile_25'] for entry in data]
                median_values = [entry['median'] for entry in data]
                mean_values = [entry['mean'] for entry in data]
                number_of_nonparticipants = [entry['cycle_nonparticipants'] for entry in data]



    prior_experienced_time_participants_plot_exp2 = {}
    prior_experienced_time_nonparticipants_plot_exp2 = {}
    for exp_num in experiment_list:
        for initial_participation_rate in initial_participation_list:


            for social_days_number in social_days_number_list:

                df_exp_store = df_time_output_exp_store[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)]

                prior_experienced_time_nonparticipants_plot2 = []
                prior_experienced_time_participants_plot2 = []

                values_additional_day = df_exp_store["additional_time_nonparticipants_day"]
                values_experienced_day = df_exp_store["experienced_time_nonparticipants_day"]

                # Create a new dictionary to store the averaged Series
                values_prior_nonparticipant = {}

                # Iterate over every 20 keys
                for i in range(0, len(values_additional_day), num_days_in_cycle):
                    # Extract the current 20 keys
                    keys_subset = list(values_additional_day.keys())[i:i + num_days_in_cycle]

                    # Extract the corresponding Series
                    series_subset_additional = [values_additional_day[key] for key in keys_subset]
                    series_subset_experienced = [values_experienced_day[key] for key in keys_subset]

                    # Concatenate the Series along the columns (axis=1) to form a DataFrame
                    df_subset_additional = pd.concat(series_subset_additional, axis=1)
                    df_subset_experienced = pd.concat(series_subset_experienced, axis=1)

                    # Calculate the mean along the columns to obtain the averaged Series
                    averaged_series_additional = df_subset_additional.mean(axis=1)
                    averaged_series_experienced = df_subset_experienced.mean(axis=1)

                    # Create a new key for the averaged Series
                    averaged_key = 1 + (i // num_days_in_cycle)  # Adjust the key naming as needed

                    # Store the averaged Series in the new dictionary
                    values_prior_nonparticipant[averaged_key] = averaged_series_experienced -averaged_series_additional




                for key, values in values_prior_nonparticipant.items():

                    max_value = np.max(values)
                    min_value = np.min(values)
                    percentile_75 = np.nanpercentile(values, upperbound)
                    percentile_25 = np.nanpercentile(values, lowerbound)
                    median = np.nanmedian(values)
                    mean = np.nanmean(values)

                    prior_experienced_time_nonparticipants_plot2.append({
                        'max': max_value,
                        'min': min_value,
                        'percentile_75': percentile_75,
                        'percentile_25': percentile_25,
                        'median': median,
                        'mean': mean,
                        'cycle_nonparticipants': len(values),
                    })

                values_additional_day = df_exp_store["additional_time_participants_day"]
                values_experienced_day = df_exp_store["experienced_time_participants_day"]

                # Create a new dictionary to store the averaged Series
                values_prior_participant = {}

                # Iterate over every 20 keys
                for i in range(0, len(values_additional_day), num_days_in_cycle):
                    # Extract the current 20 keys
                    keys_subset = list(values_additional_day.keys())[i:i + num_days_in_cycle]

                    # Extract the corresponding Series
                    series_subset_additional = [values_additional_day[key] for key in keys_subset]
                    series_subset_experienced = [values_experienced_day[key] for key in keys_subset]

                    # Concatenate the Series along the columns (axis=1) to form a DataFrame
                    df_subset_additional = pd.concat(series_subset_additional, axis=1)
                    df_subset_experienced = pd.concat(series_subset_experienced, axis=1)

                    # Calculate the mean along the columns to obtain the averaged Series
                    averaged_series_additional = df_subset_additional.mean(axis=1)
                    averaged_series_experienced = df_subset_experienced.mean(axis=1)

                    # Create a new key for the averaged Series
                    averaged_key = 1 + (i // num_days_in_cycle)  # Adjust the key naming as needed

                    # Store the averaged Series in the new dictionary
                    values_prior_participant[averaged_key] = averaged_series_experienced-averaged_series_additional

                for key, values in values_prior_participant.items():
                        max_value = np.max(values)
                        min_value = np.min(values)
                        percentile_75 = np.nanpercentile(values, upperbound)
                        percentile_25 = np.nanpercentile(values, lowerbound)
                        median = np.nanmedian(values)
                        mean = np.nanmean(values)

                        total_participants = len(values)

                        prior_experienced_time_participants_plot2.append({
                            'max': max_value,
                            'min': min_value,
                            'percentile_75': percentile_75,
                            'percentile_25': percentile_25,
                            'median': median,
                            'mean': mean,
                            'cycle_participants': len(values)})
                prior_experienced_time_participants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)] = prior_experienced_time_participants_plot2
                prior_experienced_time_nonparticipants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)] = prior_experienced_time_nonparticipants_plot2






    for exp_num in experiment_list:

        fig, axs = plt.subplots(1, 1, figsize=(4, 4))  # 3 rows, 1 column
        for initial_participation_rate in initial_participation_list:

            for social_days_number in social_days_number_list:

                print(f'scheme {exp_num}, initial rate {initial_participation_rate}')
                legend_handles = []

                data = prior_experienced_time_participants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)]
                max_values = [entry['max'] for entry in data]
                min_values = [entry['min'] for entry in data]
                UQ_values = [entry['percentile_75'] for entry in data]
                LQ_values = [entry['percentile_25'] for entry in data]
                median_values = [entry['median'] for entry in data]
                mean_values = [entry['mean'] for entry in data]
                number_of_participants = [entry['cycle_participants'] for entry in data]

                # Create labels for x-axis
                labelsize = [f'{i + 1}' for i in range(len(data))]

                axs.plot(labels, mean_values, label=f'Mean User Prior', color='#1f77b4', linewidth=2, marker='s')

                axs.set_xlabel('Cycle', fontsize=11)
                axs.axhline(0, color='grey', linestyle='--', alpha=0.5)  # You can customize color and linestyle

                data = experienced_time_participants_plot_exp2[(exp_num, initial_participation_rate,num_days_in_cycle,social_days_number)]
                max_values = [entry['max'] for entry in data]
                min_values = [entry['min'] for entry in data]
                UQ_values = [entry['percentile_75'] for entry in data]
                LQ_values = [entry['percentile_25'] for entry in data]
                median_values = [entry['median'] for entry in data]
                mean_values = [entry['mean'] for entry in data]
                number_of_nonparticipants = [entry['cycle_participants'] for entry in data]

                # Create labels for x-axis
                labels = [f'{i + 1}' for i in range(len(data))]

                axs.plot(labels, mean_values, label=f'Mean User Experienced', color='orange', linewidth=2, marker='o')

                axs.set_ylabel('Cycle Mean Travel Time [min]', fontsize=14)
                axs.set_xlabel('Cycle', fontsize=14)
                axs.axhline(0, color='grey', linestyle='--', alpha=0.5)  # You can customize color and linestyle
                # axs.set_ylim(10, 30)

                # Add handles and labels for the legend

                axs.tick_params(axis='both', which='major', labelsize=14)


        plt.subplots_adjust(hspace=.3, wspace=0.3)  # You can adjust the values as needed
        #
        plt.tight_layout()
        plt.savefig(
            results_folder_figs + f'methodPaper_prior_and_experiencedtime_mean_scheme_{exp_num}_participants_SMALL.pdf',
            format='pdf', bbox_inches='tight')
        plt.show()

    for exp_num in experiment_list:

        fig, axs = plt.subplots(1, 1, figsize=(4,4))  # 3 rows, 1 column
        for initial_participation_rate in initial_participation_list:

            for social_days_number in social_days_number_list:
                print(f'scheme {exp_num}, initial rate {initial_participation_rate}')
                legend_handles = []

                data = prior_experienced_time_nonparticipants_plot_exp2[
                    (exp_num, initial_participation_rate, num_days_in_cycle, social_days_number)]
                max_values = [entry['max'] for entry in data]
                min_values = [entry['min'] for entry in data]
                UQ_values = [entry['percentile_75'] for entry in data]
                LQ_values = [entry['percentile_25'] for entry in data]
                median_values = [entry['median'] for entry in data]
                mean_values = [entry['mean'] for entry in data]
                number_of_participants = [entry['cycle_nonparticipants'] for entry in data]

                # Create labels for x-axis
                labels = [f'{i + 1}' for i in range(len(data))]

                axs.plot(labels, mean_values, label=f'Mean User Prior', color='#1f77b4', linewidth=2, marker='s')

                axs.axhline(0, color='grey', linestyle='--', alpha=0.5)  # You can customize color and linestyle

                data = experienced_time_nonparticipants_plot_exp2[
                    (exp_num, initial_participation_rate, num_days_in_cycle, social_days_number)]
                max_values = [entry['max'] for entry in data]
                min_values = [entry['min'] for entry in data]
                UQ_values = [entry['percentile_75'] for entry in data]
                LQ_values = [entry['percentile_25'] for entry in data]
                median_values = [entry['median'] for entry in data]
                mean_values = [entry['mean'] for entry in data]
                number_of_nonparticipants = [entry['cycle_nonparticipants'] for entry in data]

                # Create labels for x-axis
                labels = [f'{i + 1}' for i in range(len(data))]

                axs.plot(labels, mean_values, label=f'Mean User Experienced', color='orange', linewidth=2, marker='o')

                axs.set_ylabel('Cycle Mean Travel Time [min]', fontsize=14)
                axs.set_xlabel('Cycle', fontsize=14)
                axs.axhline(0, color='grey', linestyle='--', alpha=0.5)  # You can customize color and linestyle
                # axs.set_ylim(10, 30)
                axs.tick_params(axis='both', which='major', labelsize=14)



        plt.subplots_adjust(hspace=.3, wspace=0.3)  # You can adjust the values as needed

        plt.tight_layout()
        plt.savefig(
            results_folder_figs + f'methodPaper_prior_and_experiencedtime_mean_scheme_{exp_num}_nonparticipants_SMALL.pdf',
            format='pdf', bbox_inches='tight')
        plt.show()




