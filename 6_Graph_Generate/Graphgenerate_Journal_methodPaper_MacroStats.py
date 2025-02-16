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
from itertools import product


import time

# your code here


def all_social():

    #--- Load Inputs ----

    #--- Baseline Travel Time ----
    # Construct the path to the pickle file
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    pickle_file_path1 = os.path.join(parent_dir, '3_Complete_SR_and_Baseline_UC', 'macro_output_completeSR_mean.pickle')

    with open(pickle_file_path1, 'rb') as file:
        total_travel_time_completeSR, travel_time_completeSR = pickle.load(
            file)

    pickle_file_path2 = os.path.join(parent_dir, '3_Complete_SR_and_Baseline_UC', 'macro_output_fullUC_mean.pickle')


    # Open and load the pickle file
    with open(pickle_file_path2, 'rb') as file:
        total_travel_time_meanUC, travel_time_init  = pickle.load(file)

    Network_benefit_All = (total_travel_time_completeSR - total_travel_time_meanUC) / 5 #values are originally in hours for 5 day week so divide by 5.

    print(f'total_travel_time_completeSR is {total_travel_time_completeSR}')
    print(f'total_travel_time_meanUC is {total_travel_time_meanUC}')

    print(f'Network_benefit_All is {Network_benefit_All}')

    return Network_benefit_All, total_travel_time_completeSR, travel_time_completeSR,total_travel_time_meanUC, travel_time_init

#%% Simulation Inputs

if __name__ == '__main__':


      # ----------------- # ----------------- # ----------------- # ----------------- 

    # ----------------- User inputs - make sure it matches the simulations  ----------------- # ----------------- 

        # ----------------- # ----------------- # ----------------- # ----------------- 

    experiment_list = [1,2]
    initial_participation_list = [20]
    social_days_number_list = [2]

    sim_hours = 4
    number_of_cycle = 10
    days_of_cycle_list = [20] # for main simulations should be 20, based on sample size analysis. can be reduced for running with fewer resources.

    upperbound = 90
    lowerbound = 10

    # ----------------- # ----------------- # ----------------- # ----------------- 
    # ----------------- # ----------------- # ----------------- # ----------------- 

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
    results_folder_figs = os.path.join(current_dir, os.pardir, "Output", "Simulations", "figs")

    Network_benefit_All, total_travel_time_completeSR, travel_time_completeSR,total_travel_time_meanUC, travel_time_init =all_social()
    print(f'Network Benefit Complete SR routing compared to all uncoord. : {5*Network_benefit_All}')


    # #--------Plot analysis------------------------------------------------------------------------------------------------
    #
        # =================================================================================

    run_num_list=[1]
    experiment_list = ['1','2']
    run_num=1

    Scheme_list=['Sacrifice', 'Collective Good']

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))  # 3 rows, 1 column

    last_three_cycle_TSTT_mean={}

    with open(results_folder + os.sep + f'macro_output.pickle', 'rb') as file:
        Network_benefit_mean_list_exp, participation_rate_store_list_exp, lost_user_count_exp = pickle.load(
            file)

    for exp_num in experiment_list:
        for initial_participation_rate in initial_participation_list:

            for num_days_in_cycle in days_of_cycle_list:
                for social_days_number in social_days_number_list:

                    print(num_days_in_cycle)
                    if exp_num == '1':

                        Network_benefit_mean_list_participation = Network_benefit_mean_list_exp[(exp_num, run_num,initial_participation_rate, num_days_in_cycle,social_days_number)]
                        participation_rate_store_list_participation = participation_rate_store_list_exp[
                            (exp_num,run_num, initial_participation_rate, num_days_in_cycle,social_days_number)]
                        print(f' Sacrifice TSTT :{Network_benefit_mean_list_participation}')
                    if exp_num == '2':
                        Network_benefit_mean_list_participation = Network_benefit_mean_list_exp[
                            (exp_num,run_num, initial_participation_rate, num_days_in_cycle,social_days_number)]
                        participation_rate_store_list_participation = participation_rate_store_list_exp[
                            (exp_num,run_num, initial_participation_rate, num_days_in_cycle,social_days_number)]
                        print(f' Collective-Good TSTT :{Network_benefit_mean_list_participation}')

                    # Plot Network Benefit Mean on the first subplot (axs[0])
                    axs.plot(list(range(1, number_of_cycle + 1)),
                                1000 * np.array(Network_benefit_mean_list_participation),
                                marker='o',
                                linestyle='-', label=Scheme_list[int(exp_num)-1])



                    last_three_cycle_TSTT_mean[exp_num]=1000*np.mean(Network_benefit_mean_list_participation[-3:-1])
                            # Customize both subplots
    axs.axhline(
        y=(total_travel_time_meanUC - total_travel_time_completeSR),
        color='red',
        linestyle='--',
        label="Complete SR"
    )
    axs.set_xlim(0)
    axs.set_xlabel('Cycle', fontsize=16)
    axs.legend(fontsize=12,loc='lower right')

    # axs.set_ylim(0, 10200)
    axs.set_ylabel('Cycle Mean Total Network Time Saving [hr]', fontsize=16)


    denominator = total_travel_time_meanUC - total_travel_time_completeSR
    last_three_cycle_TSTT_mean_proportion_fullSO = {
        key: 100 * value / denominator for key, value in last_three_cycle_TSTT_mean.items()
    }
    # Print the results
    for scheme, proportion in last_three_cycle_TSTT_mean_proportion_fullSO.items():
        print(f"{scheme}: {proportion:.2f}%")



    # Adjust spacing between subplots
    fig.tight_layout()

    # Save the figure
    plt.savefig(results_folder_figs + f'TSTT_scheme1and2_TEST.pdf', format='pdf')

    # Display the figure
    plt.show()

    print(f'Mean of the last three TSTT : {last_three_cycle_TSTT_mean}')

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))  # 3 rows, 1 column

    for exp_num in experiment_list:
        for initial_participation_rate in initial_participation_list:

            for num_days_in_cycle in days_of_cycle_list:
                for social_days_number in social_days_number_list:

                    print(num_days_in_cycle)
                    if exp_num == '1':
                        Network_benefit_mean_list_participation = Network_benefit_mean_list_exp[(exp_num, run_num,initial_participation_rate, num_days_in_cycle,social_days_number)]
                        participation_rate_store_list_participation = participation_rate_store_list_exp[
                            (exp_num,run_num, initial_participation_rate, num_days_in_cycle,social_days_number)]
                    if exp_num == '2':
                        Network_benefit_mean_list_participation = Network_benefit_mean_list_exp[
                            (exp_num, run_num,initial_participation_rate, num_days_in_cycle,social_days_number)]
                        participation_rate_store_list_participation = participation_rate_store_list_exp[
                            (exp_num, run_num,initial_participation_rate, num_days_in_cycle,social_days_number)]

                                       # Plot Participation Rate on the second subplot (axs[1])
                    axs.plot(list(range(len(participation_rate_store_list_participation))),
                                     [x * 100 for x in participation_rate_store_list_participation], marker='o',
                                     linestyle='-', label=Scheme_list[int(exp_num)-1])

                            # Customize both subplots

    axs.set_xlim(0)
    axs.set_xlabel('Cycle', fontsize=16)
    axs.legend(fontsize=12)


    axs.set_ylim(0, 105)
    axs.set_ylabel('Cycle Participation Rate [%]', fontsize=16)

    # Adjust spacing between subplots
    fig.tight_layout()

    # Save the figure
    plt.savefig(results_folder_figs + f'Participation_scheme1and2_TEST.pdf', format='pdf')

    # Display the figure
    plt.show()


    # Figures for the Moral Personality Sensitivity analysis


    run_list = [2, 3]
    SchemeName = ['Sacrifice', 'Collective Good']
    # expnum_list = ['1', '2','3']  # 1 is sacrifice, 2 is collective good no forgiveness, 3 is collective good with forgiveness
    expnum_list = ['1',
                   '2']  # 1 is sacrifice, 2 is collective good no forgiveness, 3 is collective good with forgiveness

    combinations = list(
        product(run_list, expnum_list))

    num_days_in_cycle=days_of_cycle_list[0]
    initial_participation_rate=initial_participation_list[0]
    social_days_number=social_days_number_list[0]

    fig, axs = plt.subplots(1,1, figsize=(8, 6))  # 3 rows, 1 column


    for combo in combinations:
        run_num, expnum = combo
        print(f'run_num: {run_num}')
        print(f'expnum: {expnum}')
        values = participation_rate_store_list_exp[(expnum, run_num, initial_participation_rate, num_days_in_cycle, social_days_number)]
        if run_num == 2:
            axs.plot(list(range(len(values[:]))), [x * 100 for x in values], marker='o',
                        linestyle='-', label=f'{SchemeName[int(expnum)-1]} - Most Inclined')
        if run_num == 3:
            axs.plot(list(range(len(values[:]))), [x * 100 for x in values], marker='o',
                     linestyle='-', label=f'{SchemeName[int(expnum)-1]} - Least Inclined')

        axs.set_xlim(0)
        axs.set_xlabel('Cycle', fontsize=16)
        axs.legend(fontsize=12)

        axs.set_ylim(0, 105)
        axs.set_ylabel('Cycle Participation Rate [%]', fontsize=16)

    # Adjust spacing between subplots
    fig.tight_layout()
    plt.savefig(
        results_folder_figs + f'Participation_scheme_moralSensitivity.pdf',
        format='pdf')
    plt.show()


    fig, axs = plt.subplots(1, 1, figsize=(8, 6))  # 3 rows, 1 column

    for combo in combinations:
        run_num, expnum = combo
        # print(f'run_num: {run_num}')
        # print(f'expnum: {expnum}')
        # (expnum, run_num, initial_participation_rate, days_in_cycle, social_days_number)
        values =  Network_benefit_mean_list_exp[(expnum, run_num, initial_participation_rate, num_days_in_cycle, social_days_number)]
        print(f' expnum {expnum}, run_num {run_num}, TSTT :{values}')

        if run_num == 2:
            axs.plot(list(range(1,len(values[:])+1)), [x * 1000 for x in values], marker='o',
                        linestyle='-', label=f'{SchemeName[int(expnum)-1]} - Most Inclined')
        if run_num == 3:
            axs.plot(list(range(1,len(values[:])+1)), [x * 1000 for x in values], marker='o',
                     linestyle='-', label=f'{SchemeName[int(expnum)-1]} - Least Inclined')

        axs.set_xlim(0)
        axs.set_xlabel('Cycle', fontsize=16)

        axs.set_ylim(0, 10200)
        axs.set_ylabel('Cycle Mean Total Network Time Saving [hr]', fontsize=16)
    # Adjust spacing between subplots
    axs.axhline(
        y=(total_travel_time_meanUC - total_travel_time_completeSR),
        color='red',
        linestyle='--',
        label="Complete SR"
    )
    axs.legend(fontsize=12)

    fig.tight_layout()
    plt.savefig(
        results_folder_figs + f'TSTT_scheme_moralSensitivity.pdf',
        format='pdf')
    plt.show()
