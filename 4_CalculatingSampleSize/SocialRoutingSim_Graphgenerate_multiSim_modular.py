#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20/4/2023

Author: Alexander Roocroft
"""


# For vectorized math operations
import numpy as np
import os
import matplotlib.pyplot as plt

import pandas as pd
import pickle

import time

#%% Simulation Inputs

if __name__ == '__main__':
    results_folder = "ComputationalReq_multiple_simulations" + os.sep
    results_folder_figs = os.path.join("ComputationalReq_multiple_simulations", "figs")
    os.makedirs(results_folder_figs, exist_ok=True)

    run_num_list=[1]
    run_num=1
    exp_num='1'
    initial_participation_rate=.2
    social_days_number=2

    with open(results_folder + f'individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.pickle', 'rb') as f:
        df_time_output_exp_store = pickle.load(f)

    total_user_experienced_time_day_ = df_time_output_exp_store[(exp_num, initial_participation_rate,social_days_number)]['total_user_experienced_time_day']
    total_user_experienced_time_day_1 = list(total_user_experienced_time_day_.values())
    total_user_experienced_time_day = [x / 3600 for x in total_user_experienced_time_day_1]

    # Calculate cumulative means
    cumulative_means = [np.mean(total_user_experienced_time_day[:i + 1]) for i in range(len(total_user_experienced_time_day))]


    # Calculate differences between consecutive elements
    differences = np.diff(cumulative_means)

    # Number of shuffles
    num_shuffles = 2500
    cumulative_means = {}
    cumulative_std={}

    differences = {}
    percent_changes = {}

    for shuffle_num in range(1, num_shuffles + 1):
        # Shuffle the total_user_experienced_time_day
        np.random.shuffle(total_user_experienced_time_day)

        # Calculate cumulative means for the shuffled array
        cumulative_means[shuffle_num] = [np.mean(total_user_experienced_time_day[:i + 1]) for i in range(len(total_user_experienced_time_day))]

        # Calculate differences between consecutive elements for the shuffled array
        differences[shuffle_num] = np.diff(cumulative_means[shuffle_num])
        # Calculate percentage changes for the shuffled array
        percent_changes[shuffle_num] = (differences[shuffle_num]) / cumulative_means[shuffle_num][:-1] * 100


    # Combine all percent_changes arrays into a single 2D array
    all_percent_changes = np.abs(np.vstack(list(percent_changes.values())))

    # Calculate mean, median, and quartiles along axis 0 (across shuffles)
    median_percent_changes = np.median(all_percent_changes, axis=0)
    upper_quartiles = np.percentile(all_percent_changes, 99.9, axis=0)

    # Plot mean absolute differences for the shuffled array
    fig, axs = plt.subplots(1, 1, figsize=(8, 4)) #(10,5)

    # Plot the mean absolute differences for each position in the list
    axs.plot(range(1,len(median_percent_changes)+1), median_percent_changes, marker='.', linestyle='-', label='Median', markersize=6)
    axs.plot(range(1,len(median_percent_changes)+1), upper_quartiles, linestyle='dashdot', label='99.9th Percentile', markersize=5)

    # Shade the area between UQ and x-axis

    axs.fill_between(range(1, len(median_percent_changes) + 1),  # X-values
                     upper_quartiles,  # Y-values (upper boundary)
                     0,  # Y-values (lower boundary, x-axis)
                     color='gray', alpha=0.3)

    # Add a horizontal line at y = 2
    axs.axhline(y=0.5, color='red', linestyle='dotted', label='Threshold at 0.5%', linewidth=2)

    axs.set_xlim(1,130)  # Adjusted to match the length of mean_percent_changes

    axs.set_xlabel('No. of Days', fontsize=12)

    axs.set_ylim(0, 5)  # Adjust the y-axis limit as needed
    axs.set_ylabel(r'$|\Delta\hat{tt}|$ [%]', fontsize=12)

    axs.tick_params(axis='both', which='major', labelsize=12)

    axs.tick_params(axis='both', which='major', labelsize=12)

    # Add legend
    axs.legend(fontsize=10)

    fig.tight_layout()

    plt.savefig(os.path.join(results_folder_figs, 'MultiSim_CumulativeMeans.pdf'), format='pdf')

    plt.show()
