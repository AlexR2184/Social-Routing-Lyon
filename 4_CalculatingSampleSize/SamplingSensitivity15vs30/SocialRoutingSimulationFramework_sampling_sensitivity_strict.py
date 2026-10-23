# STRICT 15/30-DAY SAMPLING SENSITIVITY (15 CYCLES, EXTENDABLE TO 20)
# Main model: SocialRoutingSimulationFramework_Optimised.py (uploaded 2026-10-08)
# Baseline behaviour preserved; only sample size, random stream separation,
# data alignment, timeout/checkpoint/resume and output paths are changed.
# Do NOT overwrite the main framework with this script.

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8th Novemeber 2023:

author: Alex Roocroft

NOTE: Make sure that the hard disk on the HPC has enough space to hold the data. Or the simulation will hang.
"""



import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from multiprocessing import Pool
from pathlib import Path
import sys
import numpy as np
import random
import pandas as pd
import pickle
import time
import traceback
import multiprocessing
import shutil
import json

run_id = os.environ.get(
    "SLURM_ARRAY_JOB_ID",
    os.environ.get("SLURM_JOB_ID", "manual")
)

SMOKE_TEST = bool(int(os.environ.get("SMOKE_TEST", "0")))

# multiprocessing.set_start_method("spawn", force=True)
multiprocessing.set_start_method("forkserver", force=True)

# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to the Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Funcs import TripsExtraction_SocialRout_module_v4_optimised as Tripextract
from Funcs import Logit_model_v2 as user_choices
from Funcs import day_iteration_optimised as day_iteration
# from Funcs.RunSymuFlowWithExternalDemandAndRoutes_SocialRoutes_v6_optimised_gridlockDetect import GridlockAbort

# print(f"Running in PID {os.getpid()} — __name__ = {__name__}")


def utility_batch(df, user_ids, scheme_params, user_choices):
    """Compute utilities in a simple Python loop (faster than df.apply)."""
    out = np.empty(len(user_ids), dtype=float)
    for i, uid in enumerate(user_ids):
        out[i] = user_choices.utility_SR(
            uid,
            df.loc[uid],
            scheme_params
        )
    return out


def generate_demand_day(
    day_id,
    veh_social,
    veh_init,
    select_social_np,
    p_sr,
    temp_folder_path,
    rng
):

    sr_today = select_social_np & (rng.random(select_social_np.size) < p_sr)

    demand_day = veh_init.copy()
    demand_day.loc[sr_today] = veh_social.loc[sr_today]

    filename = f"{temp_folder_path}/demand_day_{day_id}.pickle"
    demand_day.to_pickle(filename, protocol=pickle.HIGHEST_PROTOCOL)

    return filename, sr_today


def run_symuvia_simulation(iteration_cycle, input_data, parallel_cores, max_attempts):
    """Runs SymuVia simulations in parallel and returns per-day results."""

    print(f"Cycle {iteration_cycle}....Starting multiprocessing pool")

    print(f'[INFO] Cycle {iteration_cycle}: starting {len(input_data)} tasks '
          f'with {parallel_cores} workers; 80-minute outer watchdog.', flush=True)
    pool = Pool(processes=parallel_cores, maxtasksperchild=1)
    started = time.monotonic()
    timeout_seconds = 80 * 60
    try:
        futures = [(int(item[0]), pool.apply_async(day_iteration.run_iteration_day, (item,)))
                   for item in input_data]
        while time.monotonic() - started < timeout_seconds:
            if all(f.ready() for _, f in futures):
                break
            time.sleep(5)
        incomplete = any(not f.ready() for _, f in futures)
        results = []
        for day, future in futures:
            if not future.ready():
                results.append({'day': day, 'status': 'batch_timeout',
                                'reason': '80-minute outer watchdog'})
            else:
                try:
                    result = future.get(timeout=0)
                    if not isinstance(result, dict):
                        raise TypeError('Worker returned a non-dictionary result')
                    results.append(result)
                except Exception as exc:
                    results.append({'day': day, 'status': 'worker_exception',
                                    'reason': repr(exc)})
        if incomplete:
            pool.terminate()
        else:
            pool.close()
        pool.join()
        return results
    except BaseException:
        pool.terminate()
        pool.join()
        raise



def align_day_travel_times(extraction_csv, demand_pickle, extracted_tt, expected_users):
    """Match output travel times to input traveller positions using unique OD/departure keys."""
    demand = pd.read_pickle(demand_pickle)
    observed = pd.read_csv(extraction_csv, sep=';')
    keys = ['origin', 'destination', 'creation_time']
    if len(demand) != expected_users or len(observed) != expected_users:
        raise ValueError('Input/output traveller count mismatch')
    raw = np.asarray(extracted_tt, dtype=float)
    if len(raw) != len(observed):
        raise ValueError('Extractor array/CSV length mismatch')
    if not np.allclose(observed['travel_time'].to_numpy(dtype=float), raw,
                       rtol=1e-5, atol=1e-3):
        raise ValueError('Extractor CSV/array travel-time disagreement')
    for frame in (demand, observed):
        frame['creation_time'] = pd.to_numeric(frame['creation_time'], errors='raise').round(3)
        if frame.duplicated(keys).any():
            raise ValueError('Nonunique OD/departure key in demand or output')
    aligned = demand[keys].merge(observed[keys + ['travel_time']],
                                 on=keys, how='left', sort=False,
                                 validate='one_to_one', indicator=True)
    if not (aligned['_merge'] == 'both').all():
        raise ValueError('Output contains unmatched travellers')
    result = aligned['travel_time'].to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite aligned travel times')
    return result


def save_pickle_atomic(path, payload):
    temporary = path + '.tmp'
    with open(temporary, 'wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def extract_simulation_output_updating(
    iteration_cycle,
    start_day,
    end_day,
    out_folder,
    temp_folder,
    symuvia_output,
    sim_hours,
    used_SR_store,
    baseline_tt,
    total_delta_tt,
    total_experienced_tt_all,
    total_experienced_tt_sr,
    total_additional_time_all,
    total_additional_time_sr,
    count_sr_days,
    gridlock_days,   # days without completed simulation
    demand,
    number_of_users,
    successes_needed,
):
    bad_days_local = {}
    good_days_added = 0

    # Always select successful days in ascending day ID, not worker completion order.
    # Simulate up to five extra buffer days, but NEVER use more than the target.
    for day in range(start_day, end_day):
        if good_days_added >= successes_needed:
            break

        if day in gridlock_days:
            continue


        symuvia_output_parallel = "defaultOut" + symuvia_output
        temp_folder_parallel = temp_folder + f"_Cycle_{iteration_cycle}_Day_{day}/"

        veh_output = Tripextract.symmvia_output(
            out_folder,
            temp_folder_parallel,
            symuvia_output_parallel,
            day,
            iteration_cycle,
            sim_hours
        )

        # ---------- EARLY EXIT ----------
        if veh_output["incomplete_trips"] > 0:
            bad_days_local[day] = veh_output["incomplete_trips"]

            continue
        # --------------------------------

        csv_path = os.path.join(out_folder, f'vehs_Day_{day}_cycle_{iteration_cycle}.csv')
        veh_tt = align_day_travel_times(csv_path, demand[day],
                                         veh_output['driver_travel_time'], number_of_users)
        used_SR = used_SR_store[day]

        veh_tt_min = veh_tt / 60.0
        baseline_tt_min = baseline_tt / 60.0

        # --- experienced times (PRIMARY) ---
        total_experienced_tt_all += veh_tt_min
        total_experienced_tt_sr[used_SR] += veh_tt_min[used_SR]

        # --- additional times (DERIVED, legacy) ---
        additional_time = veh_tt_min - baseline_tt_min
        total_additional_time_all += additional_time
        total_additional_time_sr[used_SR] += additional_time[used_SR]

        # --- network totals ---
        total_delta_tt += veh_tt.sum() - baseline_tt.sum()

        count_sr_days[used_SR] += 1


        good_days_added += 1

    return good_days_added, total_delta_tt, bad_days_local

def compute_additional_times(
    *,
    total_additional_time_sr,
    count_sr_days,
    total_additional_time_all,
    good_days_total,
    select_social,
    travel_time_init,
    non_participant_estimate,
):
    """
    Compute additional travel times.

    Semantics:
    - fulldays_additional_time:
        Mean experienced additional time over all simulated days.
        Defined for ALL users. No proxying.
    - SRdays_additional_time:
        Mean additional time on SR days.
        Proxy rules applied ONLY for non-participants.
    """

    if good_days_total == 0:
        raise RuntimeError("good_days_total is zero: cannot compute averages")

    number_of_users = len(select_social)

    participant_mask = np.array(
        [select_social[u] for u in range(number_of_users)],
        dtype=bool
    )
    non_part_mask = ~participant_mask
    any_participants = participant_mask.any()

    baseline_tt_min = travel_time_init["travel_time"].to_numpy() / 60.0

    # ============================================================
    # 1) Full-days additional time (NO PROXYING)
    # ============================================================

    fulldays_additional_time_np = (
        total_additional_time_all / good_days_total
    )

    # ============================================================
    # 2) SR-days additional time (participants only initially)
    # ============================================================

    SRdays_additional_time_np = np.zeros(number_of_users, dtype=float)

    mask_sr = count_sr_days > 0
    SRdays_additional_time_np[mask_sr] = (
        total_additional_time_sr[mask_sr] / count_sr_days[mask_sr]
    )

    # ============================================================
    # 3) Assign non-participant proxies (SR-days ONLY)
    # ============================================================

    if non_participant_estimate == "use_zero":

        SRdays_additional_time_np[non_part_mask] = 0.0

    elif non_participant_estimate == "use_mean":

        if any_participants:
            mean_sr = SRdays_additional_time_np[participant_mask].mean()
        else:
            mean_sr = 0.0

        SRdays_additional_time_np[non_part_mask] = mean_sr

    elif non_participant_estimate == "use_mean_percentage":

        if any_participants:
            base_p = baseline_tt_min[participant_mask] > 0

            if np.any(base_p):
                perc_sr = (
                    SRdays_additional_time_np[participant_mask][base_p] /
                    baseline_tt_min[participant_mask][base_p]
                )
                mean_perc_sr = perc_sr.mean()
            else:
                mean_perc_sr = 0.0
        else:
            mean_perc_sr = 0.0

        SRdays_additional_time_np[non_part_mask] = (
            mean_perc_sr * baseline_tt_min[non_part_mask]
        )

    else:
        raise ValueError(
            f"Unknown non_participant_estimate: {non_participant_estimate}"
        )

    return (
        dict(enumerate(SRdays_additional_time_np)),
        dict(enumerate(fulldays_additional_time_np)),
    )



def compute_choices_apply_decisions(
    df_input,
    df_input_nonweighted,
    user_choices,
    user_param_store,
    number_of_users,
    scheme_type,
    select_social,
    additional_time_rel_tol,
    travel_time_init,
    lost_user,
    reconsideration_counter,
    reconsideration_length,
):
    # ----------------------------
    # Precompute arrays
    # ----------------------------
    user_ids = df_input.index.to_numpy()

    reconsider_np = np.array(
        [reconsideration_counter[u] for u in user_ids],
        dtype=int
    )

    lost_mask = np.array(
        [lost_user[u] for u in user_ids],
        dtype=bool
    )

    # Eligible = not 'cooling-down' AND not permanently lost
    eligible_mask = (reconsider_np == 0) & (~lost_mask)
    eligible_users = user_ids[eligible_mask]

    # ----------------------------
    # Utilities & probabilities
    # ----------------------------
    if scheme_type == "Sacrifice":
        scheme_params = user_param_store["Sacrifice"]
    else:
        scheme_params = user_param_store["CollectiveGood"]

    utility_out = np.full(len(user_ids), np.nan)
    prob_out = np.full(len(user_ids), np.nan)

    if eligible_users.size > 0:
        util_vals = utility_batch(
            df_input,
            eligible_users,
            scheme_params,
            user_choices
        )
        utility_out[eligible_mask] = util_vals
        prob_out[eligible_mask] = user_choices.prob_SR(util_vals)

    # ----------------------------
    # Simulate choices
    # ----------------------------
    rand = np.random.rand(len(user_ids))
    choice = np.zeros(len(user_ids), dtype=bool)
    choice[eligible_mask] = rand[eligible_mask] <= prob_out[eligible_mask]

    # ----------------------------
    # Lost-user logic (CollectiveGood ONLY, PERMANENT)
    # ----------------------------
    lost_now = np.zeros(len(user_ids), dtype=bool)

    if scheme_type == "CollectiveGood":
        baseline_tt = travel_time_init["travel_time"].to_numpy() / 60.0

        promise_tt = (df_input_nonweighted["experienced_tt_fulldays"].to_numpy() - baseline_tt) #does not use memory, it is the performance of this cycle only.

        select_np = np.array(
            [select_social[u] for u in user_ids],
            dtype=bool
        )

        lost_now = (
            select_np &
            (~lost_mask) &
            (promise_tt > additional_time_rel_tol * baseline_tt)
        )

    lost_this_cycle = int(lost_now.sum())

    if lost_this_cycle > 0:
        print(
            f"[DEBUG] Lost users this cycle: {lost_this_cycle} "
            f"(cumulative lost: {sum(lost_user.values()) + lost_this_cycle})"
        )

    # Mark newly lost users as permanently lost
    for u in user_ids[lost_now]:
        lost_user[u] = True

    # Enforce permanence immediately
    choice[lost_now] = False
    choice[lost_mask] = False

    # ----------------------------
    # Apply decisions
    # ----------------------------
    # Enforce loss first
    for uid in user_ids[lost_now]:
        select_social[uid] = False

    # Apply choices for eligible, non-lost users
    for uid in eligible_users:
        select_social[uid] = bool(choice[user_ids == uid])

    participation_rate = np.mean(list(select_social.values())) #this is a decimal number between 0 and 1.

    # ----------------------------
    # Update reconsideration counters - if has selected to join the scheme this cycle.
    # ----------------------------
    for i, uid in enumerate(user_ids):
        if (
            eligible_mask[i]
            and select_social[uid]
            and not df_input.loc[uid, "in_scheme_before"]
        ):
            reconsideration_counter[uid] = reconsideration_length

    # ----------------------------
    # Assemble output DataFrame
    # ----------------------------
    df_out = df_input.copy()
    df_out["utility_scheme"] = utility_out
    df_out["prob_scheme"] = prob_out
    df_out["choice_scheme"] = choice

    return df_out, participation_rate, select_social, lost_user


#%% Simulation Inputs

def run_simulation_framework(
    selector,
    scheme,
    restart_cycle=0,
    sim_hours=4,
    number_of_cycle=15,
    target_good_days=15,
    parallel_cores_override=20,
    cycles_per_job=1,
):


    #Set the path for the Symuvia library:
    lib_path = r'/home/alexroocroft/.conda/envs/env_symupy/lib/'  # Need to set to the correct path.

    # Set the output location:
    # output_dir = '/tudelft.net/staff-umbrella/SocialRoutingSimulation/'
    output_dir = '/scratch/alexroocroft/'


     # ----------------- # ----------------- # ----------------- # ----------------- 

    # ----------------- User inputs - change as needed  ----------------- # ----------------- 

        # ----------------- # ----------------- # ----------------- # -----------------
    no_sim = 0 # set to 1 to stop the simulation in SymuVia but keep the rest of the framework loop

    parallel_cores = parallel_cores_override

    problem_day_limit = 0 # for the non-perfect week rerun logic

    max_rerun_attempts_days=4 # default is 4. number of reruns of weeks of the simulation to avoid gridlock issues with Symuvia.

    max_attempts = 5 # max number of attempts if there Symuvia returns a run error. Default is 5.

    weeks_in_cycle = 3 # standard is 4 for the paper simulation results
    days_in_week = 5 # each week has 5 days (i.e. work week)


    shuffle_participation=True # choice to shuffle which days of the week participants use the social route on each week of cycle.

    # ----------------- # ----------------- # ----------------- # -----------------
    # ----------------- # ----------------- # ----------------- # ----------------- 
    # ============================================================
    # EXPERIMENT CASE DEFINITIONS (WITH LABELS)
    # ============================================================

    base_experiment_cases = [
        {"_label": "baseline"},

        # --- social days ---
        {"social_days_number": 1, "_label": "social_days_1"},
        {"social_days_number": 4, "_label": "social_days_4"},

        # --- moral personality ---
        {"moral_num": 2, "_label": "moral_best"},
        {"moral_num": 3, "_label": "moral_worst"},

        # --- initial participation ---
        {"participation_rate1": 1, "_label": "init_part_1"},
        {"participation_rate1": 60, "_label": "init_part_60"},

        # --- rho - AR(1) memory persistence parameter---
        {"rho": 0.0, "_label": "rho_0p0"},
        {"rho": 0.1, "_label": "rho_0p1"},
        {"rho": 0.8, "_label": "rho_0p8"},

        # --- choice-model profile ---
        {"profile_select": 1, "_label": "profile_1"},
        {"profile_select": 2, "_label": "profile_2"},

        # --- non-participant estimate ---
        {
            "non_participant_estimate": "use_mean_percentage",
            "_label": "non_part_use_mean_percentage",
        },
        {
            "non_participant_estimate": "use_zero",
            "_label": "non_part_use_zero",
        },
        # --- reconsideration length ---
        {"reconsideration_length": 1, "_label": "recon_len_1"},
        {"reconsideration_length": 3, "_label": "recon_len_3"},
        # --- combined extreme reactivity (for CG: forgiving loss threshold) ---
        {
            "rho": 0.0,
            "reconsideration_length": 1,
            "additional_time_rel_tol": 0.2,
            "_label": "rho_0p0_recon_len_1_tol_0p2"
        },
        {
            "moral_num": 3,
            "reconsideration_length": 1,
            "rho": 0.0,
            "additional_time_rel_tol": 0.2,
            "_label": "moral_worst_rho_0p0_recon_len_1_tol_0p2",
        },

    ]

    cg_only_cases = [
        {"additional_time_rel_tol": 0.1, "_label": "cg_tol_0p1"},
        {"additional_time_rel_tol": 0.2, "_label": "cg_tol_0p2"},

    ]

    # ============================================================
    # SELECT CASES BASED ON SCHEME
    # ============================================================

    if scheme == "Sacrifice":
        experiment_cases = base_experiment_cases
    elif scheme == "CollectiveGood":
        experiment_cases = base_experiment_cases + cg_only_cases
    else:
        raise RuntimeError(f"Unknown scheme: {scheme}")

    N_CASES = len(experiment_cases)

    # ============================================================
    # TASK ID → (CASE, REP)
    # ============================================================

    R = 3  # number of repetitions of each case

    task_id = selector
    case_id = (task_id - 1) // R
    rep_id = (task_id - 1) % R

    if case_id >= N_CASES:
        raise ValueError(
            f"Task ID {task_id} exceeds number of experiment cases ({N_CASES})"
        )

    raw_case = experiment_cases[case_id]

    # Extract label (always present)
    case_label = raw_case["_label"]

    # Remove label before applying overrides
    experiment_override = {
        k: v for k, v in raw_case.items() if k != "_label"
    }

    # ============================================================
    # COMMON RANDOM NUMBERS (CRN)
    # ============================================================

    BASE_SEED = 12345
    seed = BASE_SEED + rep_id

    random.seed(seed)
    np.random.seed(seed)

    # ============================================================
    # BASELINE DEFAULT PARAMETERS
    # ============================================================
    scheme_type = scheme # 'Sacrifice'|'CollectiveGood'
    moral_num = 1 # survey based moral distribution
    participation_rate1 = 20
    social_days_number = 2
    additional_time_rel_tol = 0
    rho = 0.4 # AR(1) memory persistence parameter (recommended baseline is 0.85)
    profile_select = 0
    reconsideration_length = 2  # Set the number of cycles that reconsideration


    # set how to calculate the non-participants individual travel time saving. 'use_mean' or 'use_zero'.
    non_participant_estimate = 'use_mean'  # 'use_mean' | 'use_zero' | 'use_mean_percentage'
    # ============================================================
    # APPLY SINGLE-PARAMETER OVERRIDE
    # ============================================================

    for key, value in experiment_override.items():
        if key == "moral_num":
            moral_num = value
        elif key == "participation_rate1":
            participation_rate1 = value
        elif key == "social_days_number":
            social_days_number = value
        elif key == "rho":
            rho = value
        elif key == "profile_select":
            profile_select = value
        elif key == "non_participant_estimate":
            non_participant_estimate = value
        elif key == "reconsideration_length":
            reconsideration_length = value
        elif key == "additional_time_rel_tol":
            additional_time_rel_tol = value


    participation_rate = participation_rate1 / 100

    p_sr = social_days_number / days_in_week #probability of any day being SR for a participant in cycle sample.

    print(f"CASE {case_id} | REP {rep_id}")

    print(f"Simulation length: {sim_hours}")
    print(f"Overrides applied: {experiment_override if experiment_override else 'baseline'}")
    print(f"Initial Participation Rate: {100 * participation_rate:.1f}%")
    print(f"Days of Commitment to SR: {social_days_number}")
    print(f"Reconsideration length: {reconsideration_length}")
    print(f"Rel tol. for CG schemes: {additional_time_rel_tol}")
    print(f"Moral profile type: {moral_num}")
    print(f"Choice-model profile_select: {profile_select}")
    print(f"Non-participant estimate method: {non_participant_estimate}")



    # ============================================================
    # ============================================================

    assert parallel_cores % days_in_week == 0, \
        "parallel_cores must be divisible by days_in_week"

    weeks_in_stack = parallel_cores // days_in_week
    number_stack_simulations = parallel_cores

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
        demand_input_social = os.path.join(demand_1h_path, 'SR_routes_2h.csv')
        demand_input_init = os.path.join(demand_1h_path, 'UC_routes_2h.csv')

    # Demand loading
    veh_social_1 = pd.read_csv(demand_input_social, sep=";")  # columns: origin;typeofvehicle;creation;path;destination
    veh_init_1 = pd.read_csv(demand_input_init, sep=";")  # columns: origin;typeofvehicle;creation;path;destination

    # Canonical UC ordering, explicitly one-to-one matched to SR alternatives.
    keys = ['origin', 'destination', 'creation_time']
    if veh_init_1.duplicated(keys).any() or veh_social_1.duplicated(keys).any():
        raise ValueError('UC/SR OD/departure keys must be unique')
    veh_init = veh_init_1.reset_index(drop=True)
    veh_social = veh_init[keys].merge(veh_social_1, on=keys, how='left',
                                       sort=False, validate='one_to_one', indicator=True)
    if len(veh_init) != len(veh_social) or not (veh_social['_merge'] == 'both').all():
        raise ValueError('UC/SR route sets do not contain identical travellers')
    veh_social = veh_social.drop(columns='_merge')
    if not veh_social[keys].equals(veh_init[keys]):
        raise ValueError('UC/SR row alignment failed')
    veh_social['id'] = veh_init['id'].to_numpy()
    print(f'[INFO] Aligned {len(veh_init)} UC/SR users by OD and departure time', flush=True)


    number_of_users = len(veh_init['id'])
    print(f'number_of_users in simulation is {number_of_users}')

    #--- Baseline Travel Time ----
    if sim_hours == 2:
        # Construct the path to the pickle file
        pickle_file_path1 = os.path.join(parent_dir, '3_Complete_SR_and_Baseline_UC',
                                         'macro_output_completeSR_mean_2h.pickle')

        # Open and load the pickle file
        with open(pickle_file_path1, 'rb') as file:
            total_travel_time_completeSR, travel_time_completeSR = pickle.load(
                file)

        pickle_file_path1 = os.path.join(parent_dir, '3_Complete_SR_and_Baseline_UC', 'macro_output_fullUC_mean_2h.pickle')

        # Open and load the pickle file
        with open(pickle_file_path1, 'rb') as file:
            total_travel_time_meanUC, travel_time_init = pickle.load(file)
    elif sim_hours == 4:
        # Construct the path to the pickle file
        pickle_file_path1 = os.path.join(parent_dir, '3_Complete_SR_and_Baseline_UC', 'macro_output_completeSR_mean_4h.pickle')

        # Open and load the pickle file
        with open(pickle_file_path1, 'rb') as file:
            total_travel_time_completeSR, travel_time_completeSR = pickle.load(
                file)

        pickle_file_path1 = os.path.join(parent_dir, '3_Complete_SR_and_Baseline_UC', 'macro_output_fullUC_mean_4h.pickle')

        # Open and load the pickle file
        with open(pickle_file_path1, 'rb') as file:
            total_travel_time_meanUC, travel_time_init  = pickle.load(file)

    Network_benefit_All = (total_travel_time_completeSR - total_travel_time_meanUC) * 5 #values are originally in hours for one day so muliply by 5 for a 5-day week.

    print(f'Simulated_SO_total_travel_time is {total_travel_time_completeSR}')
    print(f'Simulated_UC_total_travel_time is {total_travel_time_meanUC}')

    print(f'Network_benefit_All is {Network_benefit_All}')

    #--- Simulation save file name definition ----

    start_sim_time='06300'
    if 6+sim_hours>=10:
        finish_sim_time = f'{6 + sim_hours}300'
    else:
        finish_sim_time = f'0{6+sim_hours}300'
    symuvia_output = f"_{start_sim_time}0_{finish_sim_time}0_traf.xml"


    #--- Create store variables for the cycles----

    participation_rate_store=dict()
    select_social_store=dict()
    reconsideration_counter_store= dict()
    participation_rate_store[0] = participation_rate
    df_input_storage_cycle = {}
    df_input_cyclemean_storage = dict()
    df_out_storage=dict()

    perception_memory_store = {} #AR(1) perception memory store
    cycle_social_days={}
    lost_user_count={}

    cycles = []
    participation_pct_list = []
    network_benefit_list = []

    print(f'Scheme Type is set to: {scheme_type}')

    #--- Number of days using social route profile ----
    number_of_days = dict()

    for user_id in range(number_of_users):
        number_of_days[user_id] = (social_days_number / days_in_week) * 10 #this could be heterogenous for users...future work!

    select_social=dict()
    reconsideration_counter= dict()

    for user_id in range(number_of_users):

        if random.random() <= participation_rate:
            select_social[user_id]= True
            reconsideration_counter[user_id] = reconsideration_length
        else:
            select_social[user_id] = False
            reconsideration_counter[user_id] = 0

    select_social_store[0]=select_social.copy()
    reconsideration_counter_store[0]=reconsideration_counter.copy()


    # -----SymuVia_Lyon_Inputs---------
    User_and_Moral_Profile_path = os.path.join(current_dir, os.pardir, '2_User_and_Moral_Profile')
    User_and_Moral_Profile_path = os.path.abspath(User_and_Moral_Profile_path)

    if sim_hours == 4:
        user_profile_path = os.path.join(User_and_Moral_Profile_path, 'user_profile_4h.pickle')
        with open(user_profile_path, 'rb') as file:
            user_param_store_stack,moral_user_stack = pickle.load(file)

    if sim_hours==2:
        user_profile_path_2h = os.path.join(User_and_Moral_Profile_path, 'user_profile_2h.pickle')
        with open(user_profile_path_2h, 'rb') as file:
            user_param_store_stack,moral_user_stack = pickle.load(file)

    # select one profile from the stack of N number of random generated assignments of the user parameter and moral personality distribution.
    user_param_store = user_param_store_stack[profile_select] 
    moral_user = moral_user_stack[profile_select]

    # the next if statements apply the moral personality sensitivity analysis conditions if needed. Overwrites the moral_user import if needed.
    if moral_num == 1:
        print('Survey Moral Profile for all users applied')
    if moral_num == 2:

        print('Most inclined Moral Profile for all users applied') # overwrites the moral personality profile
        if scheme_type == 'Sacrifice':
            print('Sacrifice, most inclined profile : care, fairness, authority ')

            for user_id in range(number_of_users):
                moral_user[user_id] = {'Care': 1, 'Fairness': 1, 'Ingroup': 0, 'Authority': 1, 'Purity': 0}
        if scheme_type == 'CollectiveGood':
            print('Collective Good, most inclined profile : care, fairness, authority and purity ')

            for user_id in range(number_of_users):
                moral_user[user_id] = {'Care': 1, 'Fairness': 1, 'Ingroup': 0, 'Authority': 1, 'Purity': 1}

    if moral_num == 3:
        print('Least inclined Moral Profile for all users applied') # overwrites the moral personality profile
        if scheme_type == 'Sacrifice':
            print('Sacrifice, least inclined profile : ingroup and purity ')

            for user_id in range(number_of_users):
                moral_user[user_id] = {'Care': 0, 'Fairness': 0, 'Ingroup': 1, 'Authority': 0, 'Purity': 1}
        if scheme_type == 'CollectiveGood':
            print('Collective Good, least inclined profile : ingroup')
            for user_id in range(number_of_users):
                moral_user[user_id] = {'Care': 0, 'Fairness': 0, 'Ingroup': 1, 'Authority': 0, 'Purity': 0}

    # -----initialise the lost_user counter.-----
    lost_user = {user_id: False for user_id in range(number_of_users)}
    # ----------------------------------------

    # -----Define folder paths------

    results_folder = os.path.join(
        os.getcwd(), 'ComputationalReq_sampling_sensitivity_strict'
    )
    print(f"[INFO] SLURM_JOB_ID = {run_id}")
    print(f"[INFO] Writing results to: {results_folder}")
    
    folder_outputs = os.path.join(
        results_folder,
        f'scheme_{scheme_type}',
        f'rep_{rep_id:02d}',
        f'days_{target_good_days}',
    )
    os.makedirs(folder_outputs, exist_ok=True)

    config_path = os.path.join(folder_outputs, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(
            {
                "scheme": scheme_type,
                "case_id": case_id,
                "case_label": case_label,
                "rep_id": rep_id,
                "seed": seed,
                "overrides": experiment_override,
                "target_good_days": target_good_days,
                "cycles": number_of_cycle,
                "parallel_cores": parallel_cores,
                "replication": rep_id,
                "sampling_validation": True,
            },
            f,
            indent=2,
        )


    scratch_case = os.path.join(
        '/scratch/alexroocroft/ComputationalReq_sampling_sensitivity_strict',
        f'scheme_{scheme_type}', f'rep_{rep_id:02d}',
        f'days_{target_good_days}',
    )
    temp_folder_path = os.path.join(scratch_case, 'TEMP')
    out_folder_path = os.path.join(scratch_case, 'OUT')

    # Create TEMP and OUT folders if they do not exist
    os.makedirs(temp_folder_path, exist_ok=True)
    os.makedirs(out_folder_path, exist_ok=True)

    # Define network input stem
    network_file_stem = os.path.join(temp_folder_path, 'L63V') #this it the stem of the name of a file in TEMP e.g. 'L63V_Cycle_20_Day_2.xml'

    # Define output locations
    out_folder = out_folder_path + os.sep
    temp_folder = os.path.join(temp_folder_path, 'TEMP') #this it the stem of the name of a folder in TEMP e.g. 'TEMP_Cycle_20_Day_18'
    temp_folder = temp_folder.rstrip(os.sep)  # Removes any extra slash at the end


    # ----------------------------------------


    # ---------------------------------------------------
    # RECOVER STATE FROM output.pickle IF RESTARTING
    # ---------------------------------------------------
    if restart_cycle > 0:
        file_path1 = os.path.join(folder_outputs, 'output.pickle')
        print(f"Restarting from the end of cycle {restart_cycle-1}, loading {file_path1}")

        with open(file_path1, 'rb') as file:
            (df_input_cyclemean_storage,
             participation_rate_store,
             df_out_storage,
             folder_outputs_prev,
             select_social_store,
             lost_user_count,
             reconsideration_counter_store,
             perception_memory_store,
             lost_user) = pickle.load(file)


        # Restore the COMPLETE dynamic state (including permanent CG losses).
        select_social = select_social_store[restart_cycle-1].copy()
        reconsideration_counter = reconsideration_counter_store[restart_cycle-1].copy()
        participation_rate = participation_rate_store[restart_cycle-1]
        with open(file_path1, 'rb') as checkpoint_stream:
            saved_state = pickle.load(checkpoint_stream)
        if len(saved_state) != 9:
            raise RuntimeError('Expected 9-item trajectory checkpoint including lost_user')
        lost_user = saved_state[8].copy()
        start_cycle = restart_cycle
        print(f'[RESUME] Starting cycle {start_cycle} with participation '
              f'{participation_rate:.3%}', flush=True)
    else:
        start_cycle = 1

#==============================================================================================
#========================SIMULATION CYCLE======================================================
#==============================================================================================

    last_cycle = min(number_of_cycle, start_cycle + cycles_per_job - 1)
    print(f'[INFO] This job segment: cycles {start_cycle}..{last_cycle} '
          f'of {number_of_cycle}', flush=True)
    for iteration_cycle in range(start_cycle, last_cycle + 1):
        # Shared per-cycle behavioural choice uniforms across the 15/30-day arms;
        # independent of the number of daily simulations and safe across restarts.
        np.random.seed(seed + 100000 + iteration_cycle)
        if os.path.exists(temp_folder_path):
            shutil.rmtree(temp_folder_path)
        os.makedirs(temp_folder_path, exist_ok=True)


        cycle_start = time.time() # start of internal cycle timer
        print('-----------------------------------------------')
        print(f'Cycle: {iteration_cycle}')
        print('-----------------------------------------------')

# =========================Incremental rerun with minimum good days==============================================

        MIN_GOOD_DAYS = target_good_days
        print(f'[INFO] Cycle {iteration_cycle}: STRICT target {MIN_GOOD_DAYS} successful days',
              flush=True)

        baseline_tt = travel_time_init["travel_time"].to_numpy() #convert to numpy array for speed.

        good_days_total = 0
        total_delta_tt = 0.0

        # --- experienced travel times (minutes) ---
        total_experienced_tt_all = np.zeros(number_of_users, dtype=float)
        total_experienced_tt_sr  = np.zeros(number_of_users, dtype=float)

        # --- additional times (legacy, derived later) ---
        total_additional_time_all = np.zeros(number_of_users, dtype=float)
        total_additional_time_sr  = np.zeros(number_of_users, dtype=float)

        count_sr_days = np.zeros(number_of_users, dtype=int)



        used_SR_store ={}

        select_social_np = np.fromiter(
            select_social.values(),
            dtype=bool,
            count=number_of_users
        )
        assert select_social_np.size == number_of_users

        rerun_attempts = 0
        batch_id = 0

        completed_days = set()
        aborted_days = {}
        bad_days = {}   # day -> incomplete_trips


        while good_days_total < MIN_GOOD_DAYS:

            batch_id += 1
            rerun_attempts += 1

            if rerun_attempts > max_rerun_attempts_days:
                raise RuntimeError(
                    f"[Cycle {iteration_cycle}] "
                    f"Exceeded max rerun attempts "
                    f"({max_rerun_attempts_days}). "
                    f"Aborting entire framework run."
                )

            remaining = MIN_GOOD_DAYS - good_days_total
            # Launch five buffer days: 20 attempted / 15 retained or
            # 35 attempted / 30 retained in the first batch.
            # If some attempts fail, retry only the outstanding successful days
            # plus up to five buffer attempts; never exceed allocated workers.
            batch_size = min(number_stack_simulations, remaining + 5)
            countA = (batch_id - 1) * number_stack_simulations + 1
            countB = countA + batch_size
            print(f'[INFO] Cycle {iteration_cycle}, batch {batch_id}: '
                  f'attempting {batch_size} days; need {remaining} successes '
                  f'(up to {batch_size - remaining} buffer days)', flush=True)

            rng = np.random.default_rng(seed + iteration_cycle + rerun_attempts)

            demand = {}
            for day in range(countA, countB):
                demand[day], used_SR_day = generate_demand_day(
                    day,
                    veh_social,
                    veh_init,
                    select_social_np,
                    p_sr,
                    temp_folder_path,
                    rng
                )

                used_SR_store[day] = used_SR_day

            input_data = [
                (day, network_input, network_file_stem, demand[day], iteration_cycle, lib_path)
                for day in range(countA, countB)
            ]

          
            gridlock_days = set()
            if no_sim == 0:
                results = run_symuvia_simulation(
                    iteration_cycle,
                    input_data,
                    parallel_cores,
                    max_attempts
                )


                for res in results:
                    if res is None:
                        continue
                    if res["status"] == "completed":
                        completed_days.add(res["day"])
                    else:
                        aborted_days[res["day"]] = res.get('reason', res.get('status'))
                        gridlock_days.add(res["day"])
                # A worker returning no status must never be sent to extraction.
                attempted_days = set(range(countA, countB))
                gridlock_days.update(attempted_days - completed_days)

            
            good_days_added, total_delta_tt, bad_days_local = extract_simulation_output_updating(
                iteration_cycle,
                countA,
                countB,
                out_folder,
                temp_folder,
                symuvia_output,
                sim_hours,
                used_SR_store,
                baseline_tt,
                total_delta_tt,
                total_experienced_tt_all,
                total_experienced_tt_sr,
                total_additional_time_all,
                total_additional_time_sr,
                count_sr_days,
                gridlock_days,
                demand,
                number_of_users,
                remaining,
            )

            good_days_total += good_days_added
            bad_days.update(bad_days_local)


        assert good_days_total == MIN_GOOD_DAYS, (
            f'Cycle {iteration_cycle}: expected exactly {MIN_GOOD_DAYS} successful days, '
            f'got {good_days_total}'
        )
        print(
            f"Cycle {iteration_cycle} | "
            f"Simulated days: {len(completed_days)} | "
            f"Aborted: {sorted(aborted_days)} | "
            f"Good days: {good_days_total}",
            flush=True
        )   

        if bad_days:
            print(
                f"Cycle {iteration_cycle} | bad days (incomplete trips): {bad_days}",
                flush=True
            )
        # -------------------------------------
        # --- Network benefit ---
        daily_benefit_unscaled = -total_delta_tt / good_days_total / 3600.0
        Network_benefit_unscaled = daily_benefit_unscaled * days_in_week #in the choice models it is the weekly benefit that is used.
        Network_benefit = 7.5 * Network_benefit_unscaled / abs(Network_benefit_All)

        # ============================================================
        # Experienced travel times (cycle averages)
        # ============================================================

        experienced_tt_fulldays = total_experienced_tt_all / good_days_total

        experienced_tt_SRdays = np.zeros(number_of_users)
        mask_sr = count_sr_days > 0
        experienced_tt_SRdays[mask_sr] = (
            total_experienced_tt_sr[mask_sr] / count_sr_days[mask_sr]
        )



        SRdays_additional_time, fulldays_additional_time = compute_additional_times(
            total_additional_time_sr=total_additional_time_sr,
            count_sr_days=count_sr_days,
            total_additional_time_all=total_additional_time_all,
            good_days_total=good_days_total,
            select_social=select_social,
            travel_time_init=travel_time_init,
            non_participant_estimate=non_participant_estimate,
        )

        df = pd.DataFrame({
            # --- experienced (observed) ---
            "experienced_tt_SRdays": pd.Series(dict(enumerate(experienced_tt_SRdays))),
            "experienced_tt_fulldays": pd.Series(dict(enumerate(experienced_tt_fulldays))),

            # --- additional (policy / derived) ---
            "SRdays_additional_time": pd.Series(SRdays_additional_time),
            "fulldays_additional_time": pd.Series(fulldays_additional_time),

            "Network_benefit": Network_benefit,
            "Network_benefit_unscaled": Network_benefit_unscaled,
            "participation_rate": participation_rate * 10,
            "number_of_days": pd.Series(number_of_days),
        })


        df_moral = pd.DataFrame.from_dict(moral_user, orient="index")

        df_input_cycle_mean =  pd.concat([df, df_moral], axis=1)


        #add df_input_cycle_mean to the results store so that it can be used in the memory calculation.
        df_input_cyclemean_storage[iteration_cycle] = df_input_cycle_mean.copy()

        #make it into numpy for faster calculations.
        df_np = df_input_cycle_mean.to_numpy()
        df_cols = df_input_cycle_mean.columns
        df_index = df_input_cycle_mean.index.to_numpy()

        # -------------------------------
        # AR(1) MEMORY UPDATE
        # -------------------------------

        if iteration_cycle == 1 and restart_cycle == 0:
            # First-ever cycle (no restart): perception = actual experience
            memory_df_out = df_input_cycle_mean.copy()
        else:
            if iteration_cycle == restart_cycle + 1 and restart_cycle > 0:
                # Restart mode: load memory from previous run
                memory_df_out_prev = perception_memory_store[restart_cycle]
            else:
                # Normal continuation: load memory from previous cycle
                memory_df_out_prev = perception_memory_store[iteration_cycle - 1]

            # Apply AR(1): M_t = rho*M_{t-1} + (1-rho)*A_t
            memory_df_out = (
                    rho * memory_df_out_prev +
                    (1 - rho) * df_input_cycle_mean
            )

        # Save updated memory for this cycle
        perception_memory_store[iteration_cycle] = memory_df_out.copy()

        # Decrement counters each cycle
        for uid in reconsideration_counter:
            if reconsideration_counter[uid] > 0:
                reconsideration_counter[uid] -= 1

        # Record previous participation in df_input (memory_df_out)
        memory_df_out['in_scheme_before'] = [select_social[uid] for uid in memory_df_out.index]

        df_out, participation_rate, select_social, lost_user = compute_choices_apply_decisions(memory_df_out, df_input_cycle_mean, user_choices, user_param_store, number_of_users, scheme_type, select_social, additional_time_rel_tol, travel_time_init, lost_user, reconsideration_counter,reconsideration_length)

        count_lost = sum(lost_user.values())

        # Store results for this cycle
    
        participation_rate_store[iteration_cycle] = participation_rate
        df_out_storage[iteration_cycle] = df_out
        select_social_store[iteration_cycle] = select_social.copy()
        lost_user_count[iteration_cycle] = count_lost
        reconsideration_counter_store[iteration_cycle] = reconsideration_counter.copy()

        if scheme_type == "CollectiveGood" and iteration_cycle > 1:
            assert (
                lost_user_count[iteration_cycle]
                >= lost_user_count[iteration_cycle - 1]
            ), (
                f"[Error] Lost users decreased from cycle {iteration_cycle - 1} "
                f"to {iteration_cycle} — lost must be permanent!"
            )


        participation_pct = 100 * participation_rate
        network_benefit_hours = float(df_input_cycle_mean['Network_benefit_unscaled'].iloc[0])

        print(f"Cycle {iteration_cycle}, Network Benefit: {network_benefit_hours:.1f} hours")

        print(f"Cycle {iteration_cycle}, Participation Rate: {participation_pct:.1f} %")

        # 1) What are users actually feeling?
        # Pull per-cycle user features (has 'SRdays_additional_time')
        df_cycle = df_input_cyclemean_storage[iteration_cycle]  # index: user_id

        # Use participation status at the START of this cycle,
        # before the participation decisions are updated.
        sel = np.array([select_social_store[iteration_cycle - 1][u] for u in df_index])

        add_time = df_np[:, df_cols.get_loc("SRdays_additional_time")]

        p_add = add_time[sel]
        n_add = add_time[~sel]

        print(
            f"[Cycle {iteration_cycle}] participants add_time: "
            f"mean={np.mean(p_add):.2f} min, "
            f"median={np.median(p_add):.2f}, "
            f"p90={np.quantile(p_add, 0.9):.2f}"
        )

        # 2) Check decision components (if available)

        print(
            f"[Cycle {iteration_cycle}] mean participation prob: {df_out_storage[iteration_cycle]['prob_scheme'].mean(skipna=True):.3f}")
        print(
            f"[Cycle {iteration_cycle}] realized participation: {df_out_storage[iteration_cycle]['choice_scheme'].mean():.3f}")

        #------------------------------

        cycles.append(iteration_cycle)
        participation_pct_list.append(participation_pct)
        network_benefit_list.append(network_benefit_hours)

        file_path1 = os.path.join(folder_outputs, 'output.pickle')

        # Atomic checkpoint: safe to resubmit after Slurm wall-time cancellation.
        payload = (
                df_input_cyclemean_storage,
                 participation_rate_store,
                 df_out_storage,
                 folder_outputs,
                 select_social_store,
                 lost_user_count,
                 reconsideration_counter_store,
                 perception_memory_store,
                 lost_user)
        save_pickle_atomic(file_path1, payload)
        # Remove heavy per-day output CSVs once aggregate and user-level values
        # have been captured in the cycle checkpoint.
        shutil.rmtree(out_folder_path, ignore_errors=True)
        os.makedirs(out_folder_path, exist_ok=True)
        shutil.rmtree(temp_folder_path, ignore_errors=True)
        os.makedirs(temp_folder_path, exist_ok=True)


        print("\nSummary by cycle:")
        for c, p, b in zip(cycles, participation_pct_list, network_benefit_list):
            print(f"Cycle {c}: Participation Rate = {p:.1f} %, Network Benefit = {b:.1f} hours")

        current_time = time.time()
        current_elapsed_time = (current_time - start_time) / 3600  # hours
        cycle_minutes = (current_time - cycle_start) / 60  # minutes
        eta_hours = current_elapsed_time * (number_of_cycle / iteration_cycle - 1)

        print(
            f"Cycle {iteration_cycle}: "
            f"Cycle runtime = {cycle_minutes:.1f} min | "
            f"Total runtime = {current_elapsed_time:.2f} h ({current_elapsed_time * 60:.0f} min) | "
            f"Estimated remaining time for the simulation = {eta_hours:.2f} h"
        )

# ==============================================================================================
# ==============================================================================================
# ==============================================================================================


def main():
    ## ============================================================
    # COMMAND-LINE ARGUMENTS
    # ============================================================

    # 1) required: selector (Slurm array task ID)
    try:
        selector = int(sys.argv[1])
    except (IndexError, ValueError):
        sys.exit("Error: first argument must be an integer selector.")

    # 2) optional: scheme (Sacrifice | CollectiveGood)
    scheme = "Sacrifice"
    if len(sys.argv) > 2:
        scheme = sys.argv[2]
        if scheme not in {"Sacrifice", "CollectiveGood"}:
            sys.exit("Error: scheme must be 'Sacrifice' or 'CollectiveGood'.")

    # 3) optional: sim_hours (2 or 4)
    sim_hours = 4
    if len(sys.argv) > 3:
        sim_hours = int(sys.argv[3])
        if sim_hours not in {2, 4}:
            sys.exit("Error: sim_hours must be 2 or 4.")

    # 4) optional: total cycles (15 by default, 20 when extending checkpoints)
    number_of_cycle = 15
    if len(sys.argv) > 4:
        number_of_cycle = int(sys.argv[4])
        if number_of_cycle <= 0:
            sys.exit("Error: number_of_cycle must be positive.")

    target_good_days = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    parallel_cores_override = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    cycles_per_job = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    if target_good_days not in (15, 30):
        sys.exit('Error: target_good_days must be 15 or 30')
    if parallel_cores_override < 5 or parallel_cores_override % 5 != 0:
        sys.exit('Error: parallel_cores must be a positive multiple of 5')
    if cycles_per_job < 1:
        sys.exit('Error: cycles_per_job must be >= 1')
    if selector not in (1, 2, 3):
        sys.exit('Error: selector must be 1, 2, or 3 for baseline replications')
    if number_of_cycle not in (15, 20):
        sys.exit('Error: choose 15 cycles initially, or 20 to extend the same checkpoint')

    rep_id = selector - 1
    checkpoint_path = os.path.join(
        os.getcwd(), 'ComputationalReq_sampling_sensitivity_strict',
        f'scheme_{scheme}', f'rep_{rep_id:02d}',
        f'days_{target_good_days}', 'output.pickle')
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, 'rb') as stream:
            state = pickle.load(stream)
        if len(state) != 9:
            sys.exit('Error: invalid checkpoint format; do not overwrite it')
        restart_cycle = max(state[0]) + 1
        if restart_cycle > number_of_cycle:
            print(f'[DONE] {scheme} rep {rep_id}, {target_good_days} days: '
                  f'already completed all {number_of_cycle} cycles', flush=True)
            return
    else:
        restart_cycle = 0

    run_simulation_framework(
        selector,
        scheme,
        restart_cycle,
        sim_hours=sim_hours,
        number_of_cycle=number_of_cycle,
        target_good_days=target_good_days,
        parallel_cores_override=parallel_cores_override,
        cycles_per_job=cycles_per_job,
    )

if __name__ == '__main__':

    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    print(f"Elapsed time: {elapsed_time:.2f} hours")