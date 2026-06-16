#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th Novemeber 2023

Updated June 2026:
- Keeps the original 130-day computational-requirement sample-size simulation.
- Adds an optional re-extraction mode. If the 130-day scalar sample already exists
  and the OUT/TEMP folders are still present, the script can skip SymuVia and
  re-extract richer per-user travel-time outputs from the existing simulation files.
"""

from multiprocessing import Pool
import os
import random
import pickle
import zipfile
import sys
import time
import glob
import re

import numpy as np
import pandas as pd

# Add the current script directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.append(script_dir)

from Funcs_samplesize import TripsExtraction_SocialRout_module_v3 as Tripextract
from Funcs_samplesize import day_iteration


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
    """
    Creates the demand input for the SymuVia simulations of the cycle.

    It assigns the route for each user depending on whether the user is using
    SR or UC on a given day.
    """
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
    """Runs SymuVia simulations with error handling and retries."""
    attempts = 0
    while attempts < max_attempts:
        try:
            with Pool(processes=parallel_cores) as pool:
                pool.map(day_iteration.run_iteration_day, input_data)
        except Exception as e:
            print(f"Cycle {iteration_cycle}....Attempt {attempts + 1}: An error occurred - {e}")
            attempts += 1
        else:
            break
    else:
        print(f"Cycle {iteration_cycle}....All SymuVia attempts failed.")


def _as_scalar_incomplete_trips(incomplete_trips_value):
    """
    Converts the incomplete-trips output to a scalar.

    The existing extraction function has previously returned this as a scalar in
    the working code, but this helper also supports Series/arrays defensively.
    """
    if isinstance(incomplete_trips_value, pd.Series):
        if len(incomplete_trips_value) == 0:
            return 0
        return float(incomplete_trips_value.max())

    if isinstance(incomplete_trips_value, (list, tuple, np.ndarray)):
        arr = np.asarray(incomplete_trips_value)
        if arr.size == 0:
            return 0
        return float(np.nanmax(arr))

    return float(incomplete_trips_value)


def _total_driver_travel_time(veh_output):
    """Returns the total driver travel time from a Tripextract.symmvia_output result."""
    driver_tt = veh_output['driver_travel_time']
    if isinstance(driver_tt, pd.Series):
        return float(driver_tt.sum())
    if isinstance(driver_tt, np.ndarray):
        return float(np.nansum(driver_tt))
    if isinstance(driver_tt, list):
        return float(np.nansum(np.asarray(driver_tt, dtype=float)))
    return float(driver_tt)


def _individual_driver_travel_time_table(veh_output):
    """
    Builds a compact per-user/day travel-time table.

    The most important field is driver_travel_time. If the extraction output is
    a DataFrame, useful identifier columns are retained when available. If the
    extraction output is dict-like, the driver_travel_time vector is saved with
    its original index where possible.
    """
    id_like_cols = [
        'id', 'user_id', 'driver_id', 'vehicle_id', 'veh_id',
        'origin', 'destination', 'creation_time', 'creation', 'path'
    ]

    if isinstance(veh_output, pd.DataFrame):
        keep_cols = [col for col in id_like_cols if col in veh_output.columns]
        if 'driver_travel_time' not in veh_output.columns:
            raise KeyError("veh_output DataFrame does not contain 'driver_travel_time'.")
        keep_cols.append('driver_travel_time')
        user_day = veh_output[keep_cols].copy()
        user_day.insert(0, 'row_index', np.arange(len(user_day)))
        return user_day

    driver_tt = veh_output['driver_travel_time']

    if isinstance(driver_tt, pd.Series):
        user_day = driver_tt.rename('driver_travel_time').reset_index()
        if 'index' in user_day.columns:
            user_day = user_day.rename(columns={'index': 'user_index'})
        return user_day

    driver_tt = np.asarray(driver_tt, dtype=float)
    return pd.DataFrame({
        'user_index': np.arange(len(driver_tt)),
        'driver_travel_time': driver_tt
    })


def _extract_one_day(iteration_cycle, iteration_day, out_folder, temp_folder, symuvia_output, sim_hours):
    """Reuses the original trip-extraction function for one cycle/day."""
    symuvia_output_parallel = 'defaultOut' + symuvia_output
    temp_folder_parallel = temp_folder + f'_Cycle_{iteration_cycle}_Day_{iteration_day}/'

    veh_output = Tripextract.symmvia_output(
        out_folder,
        temp_folder_parallel,
        symuvia_output_parallel,
        iteration_day,
        iteration_cycle,
        sim_hours
    )

    incomplete_trips_check = _as_scalar_incomplete_trips(veh_output['incomplete_trips'])
    return veh_output, incomplete_trips_check, temp_folder_parallel


def extract_simulation_output_samplesize(
        iteration_cycle,
        number_stack_simulations,
        out_folder,
        temp_folder,
        symuvia_output,
        sim_hours,
        days_in_week,
        day_count,
        total_user_experienced_time_day,
        user_experienced_time_day,
        successful_day_metadata,
        total_sample_size=None):
    """
    Extracts and processes outputs for a newly simulated batch.

    This keeps the previous total-network scalar output and additionally stores
    the per-user travel-time vector/table for each successful daily realisation.
    """
    count = 0

    for iteration_day in range(1, number_stack_simulations + 1):
        if total_sample_size is not None and day_count >= total_sample_size:
            break

        print(f"Extracting simulation output for Round {iteration_cycle} of sample creation.")
        count += 1

        veh_output, incomplete_trips_check, temp_folder_parallel = _extract_one_day(
            iteration_cycle,
            iteration_day,
            out_folder,
            temp_folder,
            symuvia_output,
            sim_hours
        )

        print(f"Cycle {iteration_cycle}, Day {iteration_day}: Incomplete trips check -> {incomplete_trips_check}")

        if incomplete_trips_check > 0:
            print(f'Gridlock issue detected on Day {iteration_day} - dropping only this day')
        else:
            total_user_experienced_time = _total_driver_travel_time(veh_output)
            user_day_table = _individual_driver_travel_time_table(veh_output)

            day_count += 1
            total_user_experienced_time_day[day_count] = total_user_experienced_time
            user_experienced_time_day[day_count] = user_day_table
            successful_day_metadata[day_count] = {
                'cycle': iteration_cycle,
                'iteration_day': iteration_day,
                'temp_folder_parallel': temp_folder_parallel
            }

            print(f'day_count is {day_count}')

        if count == days_in_week:
            count = 0

    return day_count, total_user_experienced_time_day, user_experienced_time_day, successful_day_metadata


def _existing_scalar_sample_size(pickle_file, key):
    """Returns the number of stored scalar daily outcomes in the existing pickle, if available."""
    if not os.path.exists(pickle_file):
        return 0

    try:
        with open(pickle_file, 'rb') as file:
            existing_data = pickle.load(file)
    except Exception as e:
        print(f"Could not read existing sample pickle {pickle_file}: {e}")
        return 0

    if key not in existing_data:
        print(f"Existing pickle found, but key {key} is not present.")
        return 0

    total_days = existing_data[key].get('total_user_experienced_time_day', {})
    return len(total_days)


def _find_existing_temp_cycle_day_pairs(temp_folder):
    """
    Finds existing TEMP_Cycle_<cycle>_Day_<day> folders.

    temp_folder should be the stem used by the original code, e.g.
    ComputationalReq_multiple_simulations/TEMP/TEMP
    """
    temp_parent = os.path.dirname(temp_folder)
    temp_stem = os.path.basename(temp_folder)
    pattern = os.path.join(temp_parent, f'{temp_stem}_Cycle_*_Day_*')
    candidates = glob.glob(pattern)

    cycle_day_pairs = []
    regex = re.compile(r'_Cycle_(\d+)_Day_(\d+)$')

    for candidate in candidates:
        candidate_clean = candidate.rstrip(os.sep)
        match = regex.search(candidate_clean)
        if match:
            cycle = int(match.group(1))
            day = int(match.group(2))
            cycle_day_pairs.append((cycle, day))

    cycle_day_pairs = sorted(set(cycle_day_pairs))
    return cycle_day_pairs


def reextract_existing_outputs_samplesize(
        total_sample_size,
        out_folder,
        temp_folder,
        symuvia_output,
        sim_hours):
    """
    Re-extracts total and per-user travel times from already existing OUT/TEMP files.

    This is intended for the case where the expensive 130 daily SymuVia
    simulations have already been run, but the original pickle only stored
    total network travel time per day.
    """
    cycle_day_pairs = _find_existing_temp_cycle_day_pairs(temp_folder)

    if len(cycle_day_pairs) == 0:
        raise FileNotFoundError(
            f"No existing temp folders were found using stem: {temp_folder}_Cycle_*_Day_*"
        )

    print(f"Found {len(cycle_day_pairs)} existing cycle/day temp folders.")
    print("Re-extracting successful days until target sample size is reached...")

    day_count = 0
    total_user_experienced_time_day = {}
    user_experienced_time_day = {}
    successful_day_metadata = {}

    for iteration_cycle, iteration_day in cycle_day_pairs:
        if day_count >= total_sample_size:
            break

        try:
            veh_output, incomplete_trips_check, temp_folder_parallel = _extract_one_day(
                iteration_cycle,
                iteration_day,
                out_folder,
                temp_folder,
                symuvia_output,
                sim_hours
            )
        except Exception as e:
            print(f"Skipping Cycle {iteration_cycle}, Day {iteration_day}: extraction failed -> {e}")
            continue

        print(f"Re-extract Cycle {iteration_cycle}, Day {iteration_day}: incomplete trips -> {incomplete_trips_check}")

        if incomplete_trips_check > 0:
            print(f"Skipping Cycle {iteration_cycle}, Day {iteration_day}: incomplete/gridlocked output.")
            continue

        total_user_experienced_time = _total_driver_travel_time(veh_output)
        user_day_table = _individual_driver_travel_time_table(veh_output)

        day_count += 1
        total_user_experienced_time_day[day_count] = total_user_experienced_time
        user_experienced_time_day[day_count] = user_day_table
        successful_day_metadata[day_count] = {
            'cycle': iteration_cycle,
            'iteration_day': iteration_day,
            'temp_folder_parallel': temp_folder_parallel
        }

        print(f"Recovered successful sample day {day_count}/{total_sample_size}")

    if day_count < total_sample_size:
        raise RuntimeError(
            f"Only recovered {day_count} successful days, but total_sample_size={total_sample_size}."
        )

    return day_count, total_user_experienced_time_day, user_experienced_time_day, successful_day_metadata


def save_sample_size_pickle(
        pickle_file,
        zip_file,
        expnum,
        participation_rate,
        social_days_number,
        total_user_experienced_time_day,
        user_experienced_time_day,
        successful_day_metadata,
        select_social,
        number_of_days,
        number_of_users,
        scheme_type,
        sim_hours,
        total_sample_size,
        source_mode):
    """Saves the sample-size outputs in the same outer structure as the original script."""
    key = (expnum, participation_rate, social_days_number)

    df_time_output_exp = {
        key: {
            'total_user_experienced_time_day': total_user_experienced_time_day,
            'user_experienced_time_day': user_experienced_time_day,
            'successful_day_metadata': successful_day_metadata,
            'select_social': select_social,
            'number_of_days': number_of_days,
            'number_of_users': number_of_users,
            'scheme_type': scheme_type,
            'sim_hours': sim_hours,
            'total_sample_size': total_sample_size,
            'source_mode': source_mode
        }
    }

    with open(pickle_file, 'wb') as file:
        pickle.dump(df_time_output_exp, file, protocol=pickle.HIGHEST_PROTOCOL)

    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(pickle_file, os.path.basename(pickle_file))

    print(f'Pickle file saved to: {pickle_file}')
    print(f'Pickle file zipped to: {zip_file}')


#%% Simulation Inputs
def main():
    total_sample_size = 130  # 130 was used in the paper.

    parallel_cores = 20  # User should set to number of cores available for parallelisation.
    sim_hours = 4        # 2 for 0630 to 0830, 4 for 0630 to 1030.

    # ------------------------------------------------------------------
    # Switches
    # ------------------------------------------------------------------
    # If True, and the original 130-day scalar pickle already exists, the
    # script skips SymuVia and re-extracts richer per-user outputs from the
    # existing OUT/TEMP folders.
    reextract_existing_outputs_if_available = True

    # Set True only if you deliberately want to rerun SymuVia even when the
    # 130-day scalar sample already exists.
    force_rerun_simulations = False

    # For debugging only: skips SymuVia in the simulation loop but still tries
    # to extract outputs for the cycle being processed.
    no_sim = 0

    max_attempts = 5

    run_num = 1  # id number used for organising the outputs.

    expnum = '1'  # set to one to indicate sacrifice scheme.
    social_days_number = 2
    participation_rate1 = 20
    participation_rate = participation_rate1 / 100

    print(f"Initial Participation Rate: {100 * participation_rate}%")
    print(f"Days of Commitment to SR: {social_days_number}")

    if expnum == '1':
        scheme_type = 'Sacrifice'
    else:
        scheme_type = 'CollectiveGood'

    days_in_week = 5

    weeks_in_stack = parallel_cores // days_in_week
    number_stack_simulations = weeks_in_stack * days_in_week

    shuffle_participation = True

    cwd = os.getcwd()
    folder_outputs = os.path.join(cwd, "ComputationalReq_multiple_simulations")
    os.makedirs(folder_outputs, exist_ok=True)

    #--- Load Inputs ----
    current_dir = os.getcwd()

    # -----SymuVia_Lyon_Inputs---------
    SymuVia_Lyon_Inputs_path = os.path.join(current_dir, os.pardir, '1_SymuVia_Lyon_Inputs')
    SymuVia_Lyon_Inputs_path = os.path.abspath(SymuVia_Lyon_Inputs_path)

    demand_4h_path = os.path.join(SymuVia_Lyon_Inputs_path, '4_hour_demand_profile')

    network_input = os.path.join(demand_4h_path, 'L63V_4h.xml')
    demand_input_social = os.path.join(demand_4h_path, 'SR_routes_4h.csv')
    demand_input_init = os.path.join(demand_4h_path, 'UC_routes_4h.csv')

    if sim_hours == 2:
        demand_1h_path = os.path.join(SymuVia_Lyon_Inputs_path, '2_hour_demand_profile')

        network_input = os.path.join(demand_1h_path, 'L63V_2h.xml')
        demand_input_social = os.path.join(demand_1h_path, 'social_routes_2h.csv')
        demand_input_init = os.path.join(demand_1h_path, 'init_routes_2h.csv')

    # Demand and route option loading
    veh_social_1 = pd.read_csv(demand_input_social, sep=";")
    veh_init_1 = pd.read_csv(demand_input_init, sep=";")

    veh_social = pd.merge(
        veh_social_1,
        veh_init_1[['origin', 'destination', 'creation_time']],
        on=['origin', 'destination', 'creation_time'],
        how='inner'
    )
    veh_init = pd.merge(
        veh_init_1,
        veh_social[['origin', 'destination', 'creation_time']],
        on=['origin', 'destination', 'creation_time'],
        how='inner'
    )

    number_of_users = len(veh_init['id'])
    print(
        f'number_of_users in simulation is {number_of_users}, '
        f'users in SR is {len(veh_social_1["id"])} and users in UC is {len(veh_init_1["id"])}'
    )

    #--- Simulation Details ----
    start_sim_time = '06300'
    if 6 + sim_hours >= 10:
        finish_sim_time = f'{6 + sim_hours}300'
    else:
        finish_sim_time = f'0{6 + sim_hours}300'
    symuvia_output = f"_{start_sim_time}0_{finish_sim_time}0_traf.xml"

    #--- Participation Rate ----
    participation_rate_store = dict()
    select_social_store = dict()

    participation_rate_store[0] = participation_rate

    #--- Number of days using social route profile for use in the choice model----
    number_of_days = dict()

    for user_id in range(number_of_users):
        number_of_days[user_id] = (social_days_number / days_in_week) * 10

    select_social = dict()

    for user_id in range(number_of_users):
        if random.random() <= participation_rate:
            select_social[user_id] = True
        else:
            select_social[user_id] = False

    select_social_store[0] = select_social

    select_social_day_list = [True] * social_days_number + [False] * (days_in_week - social_days_number)

    # Select the routes for the drivers taking part in the social routing.
    select_social_day = dict()
    for user_id in range(number_of_users):
        random.shuffle(select_social_day_list)
        select_social_day[user_id] = select_social_day_list[:]

    # Load in the choice model parameters for the users.
    # Note: this does not matter for the baselines as only one cycle is used.
    profile_file_path = os.path.join(script_dir, '..', '2_User_and_Moral_Profile', 'user_profile_4h.pickle')

    with open(profile_file_path, 'rb') as file:
        user_param_store, moral_user = pickle.load(file)

    if sim_hours == 2:
        profile_file_path = os.path.join(script_dir, '..', '2_User_and_Moral_Profile', 'user_profile_2h.pickle')

        with open(profile_file_path, 'rb') as file:
            user_param_store, moral_user = pickle.load(file)

    # ----- Create folders TEMP and OUT for saving SymuVia outputs -------
    os.makedirs(folder_outputs, exist_ok=True)

    temp_folder_path = os.path.join(folder_outputs, 'TEMP')
    out_folder_path = os.path.join(folder_outputs, 'OUT')

    os.makedirs(temp_folder_path, exist_ok=True)
    os.makedirs(out_folder_path, exist_ok=True)

    network_input_stem = os.path.join(temp_folder_path, 'L63V')

    out_folder = out_folder_path + os.sep
    temp_folder = os.path.join(temp_folder_path, 'TEMP')
    temp_folder = temp_folder.rstrip(os.sep)

    print(f'Scheme Type is set to: {scheme_type}')

    key = (expnum, participation_rate, social_days_number)

    pickle_file = os.path.join(
        folder_outputs,
        f"individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.pickle"
    )

    zip_file = os.path.join(
        folder_outputs,
        f"individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.zip"
    )

    existing_scalar_days = _existing_scalar_sample_size(pickle_file, key)
    print(f"Existing scalar sample days found in pickle: {existing_scalar_days}")

    if (
            reextract_existing_outputs_if_available
            and not force_rerun_simulations
            and existing_scalar_days >= total_sample_size
    ):
        print("Existing 130-day scalar sample found.")
        print("Skipping SymuVia simulation and re-extracting per-user outputs from existing OUT/TEMP files.")

        (day_count,
         total_user_experienced_time_day,
         user_experienced_time_day,
         successful_day_metadata) = reextract_existing_outputs_samplesize(
            total_sample_size,
            out_folder,
            temp_folder,
            symuvia_output,
            sim_hours
        )

        save_sample_size_pickle(
            pickle_file,
            zip_file,
            expnum,
            participation_rate,
            social_days_number,
            total_user_experienced_time_day,
            user_experienced_time_day,
            successful_day_metadata,
            select_social,
            number_of_days,
            number_of_users,
            scheme_type,
            sim_hours,
            total_sample_size,
            source_mode='reextracted_from_existing_out_temp'
        )

        print(f"Re-extraction complete. Successful days saved: {day_count}")
        return

    print("No complete existing scalar sample found, or force_rerun_simulations=True.")
    print("Running the sample-size SymuVia simulations and saving both aggregate and per-user outputs.")

    iteration_cycle = 0
    day_count = 0
    total_user_experienced_time_day = {}
    user_experienced_time_day = {}
    successful_day_metadata = {}

    while day_count < total_sample_size:
        iteration_cycle += 1

        print('-----------------------------------------------')
        print(f'Cycle: {iteration_cycle}')
        print('---------------------------------------------')

        week_social_day = generate_cycle_social_days(
            number_of_users,
            weeks_in_stack,
            shuffle_participation,
            select_social_day_list
        )

        demand = generate_demand(
            week_social_day,
            veh_social,
            veh_init,
            number_of_users,
            select_social,
            days_in_week
        )

        lib_path = "/home/alexroocroft/.conda/envs/env_symupy/lib/"

        input_data = [
            (
                day,
                network_input,
                network_input_stem,
                demand[day],
                iteration_cycle,
                lib_path,
                symuvia_output,
                sim_hours
            )
            for day in range(1, 1 + number_stack_simulations)
        ]

        if no_sim == 0:
            run_symuvia_simulation(iteration_cycle, input_data, parallel_cores, max_attempts)

        # Extract the results from the SymuVia simulations.
        (day_count,
         total_user_experienced_time_day,
         user_experienced_time_day,
         successful_day_metadata) = extract_simulation_output_samplesize(
            iteration_cycle,
            number_stack_simulations,
            out_folder,
            temp_folder,
            symuvia_output,
            sim_hours,
            days_in_week,
            day_count,
            total_user_experienced_time_day,
            user_experienced_time_day,
            successful_day_metadata,
            total_sample_size=total_sample_size
        )

        print(f'Current number of sample days in stack: {day_count}')
        print(f'Current total network travel-time sample: {total_user_experienced_time_day}')

    save_sample_size_pickle(
        pickle_file,
        zip_file,
        expnum,
        participation_rate,
        social_days_number,
        total_user_experienced_time_day,
        user_experienced_time_day,
        successful_day_metadata,
        select_social,
        number_of_days,
        number_of_users,
        scheme_type,
        sim_hours,
        total_sample_size,
        source_mode='new_simulation_and_extraction'
    )


if __name__ == '__main__':
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600

    print(f"Elapsed time: {elapsed_time:.2f} hours")
