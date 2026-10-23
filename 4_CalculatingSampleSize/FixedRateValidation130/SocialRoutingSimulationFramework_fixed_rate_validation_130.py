#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimised fixed-state sampling validation for the Transportation major revision.

Purpose
-------
Generate 130 successful daily SymuVia realisations for fixed, randomly
selected participation rates of 20%, 50%, and 80%.
The same random ordering of users is used to create nested populations.
No behavioural participation updates are made during each scenario.

Important implementation choices
--------------------------------
1. Uses the SAME optimised execution stack as the current long-horizon framework:
       Funcs.day_iteration_optimised
       Funcs.TripsExtraction_SocialRout_module_v4_optimised
2. Writes heavy SymuVia TEMP/OUT data to /scratch.
3. Uses the SAME probability-based SR-day mechanism as the current main framework:
   each fixed participant independently uses SR on each simulated day with
   probability p_sr = 2/5 = 0.4.
4. Saves a persistent checkpoint after every completed batch, allowing
   several short DelftBlue submissions to accumulate 130 successful days.
5. Stores attempted-day status information to support the reviewer response on
   failed/gridlocked runs. Retains the existing one-hour gridlock abort and
   adds an outer pool watchdog for workers that fail to return.
"""

import argparse
import json
import multiprocessing
import os
import pickle
import random
import shutil
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

# Match the current optimised long-horizon framework.
multiprocessing.set_start_method("forkserver", force=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Match SocialRoutingSimulationFramework_Optimised.py exactly:
# /home/alexroocroft/GitHub_Version/Funcs sits one level above
# /home/alexroocroft/GitHub_Version/5_Framework_Simulation.
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Funcs import TripsExtraction_SocialRout_module_v4_optimised as Tripextract
from Funcs import day_iteration_optimised as day_iteration


EXPECTED_USERS = 63074
ALLOWED_RATES = (20, 50, 80)
PARTICIPANT_SEED_DEFAULT = 20261008
DAY_SEED_DEFAULT = 20260904


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--participation-percent", type=int, required=True, choices=ALLOWED_RATES,
        help="Fixed participant share, as percent: 20, 50 or 80.",
    )
    parser.add_argument(
        "--total-sample-size", type=int, default=130,
        help="Target number of SUCCESSFUL daily realisations (default 130).",
    )
    parser.add_argument("--parallel-cores", type=int, default=20)
    parser.add_argument("--sim-hours", type=int, choices=[2, 4], default=4)
    parser.add_argument(
        "--seed", type=int, default=DAY_SEED_DEFAULT,
        help="Random seed for daily Bernoulli SR/UC assignments, common to all scenarios.",
    )
    parser.add_argument(
        "--participant-seed", type=int, default=PARTICIPANT_SEED_DEFAULT,
        help="Seed for the fixed random user ordering, common to all scenarios.",
    )
    parser.add_argument(
        "--output-root", default="ComputationalReq_fixed_rate_validation_130",
        help="Persistent results/checkpoints folder under the project folder.",
    )
    parser.add_argument(
        "--scratch-root", default="/scratch/alexroocroft/ComputationalReq_fixed_rate_validation_130",
        help="Scratch root for SymuVia working files.",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=100,
        help="Overall cap on batches, including across resubmissions.",
    )
    parser.add_argument(
        "--rounds-per-job", type=int, default=2,
        help="Number of complete 20-day batches in one DelftBlue job (default 2).",
    )
    parser.add_argument(
        "--max-empty-rounds", type=int, default=2,
        help="Stop and investigate after this many consecutive all-failed batches.",
    )
    parser.add_argument(
        "--batch-timeout-minutes", type=float, default=80.0,
        help="Outer watchdog for a 20-worker batch; saves completed days even if "
             "some workers fail to return after the existing 60-min gridlock abort.",
    )
    return parser.parse_args()


def fixed_participant_mask(number_of_users, participation_percent, seed):
    """Select exactly round(rate*N) users; nested sets for 20/50/80%."""
    rng = np.random.default_rng(seed)
    ordering = rng.permutation(number_of_users)
    n_participants = int(np.floor(number_of_users * participation_percent / 100.0 + 0.5))
    mask = np.zeros(number_of_users, dtype=bool)
    mask[ordering[:n_participants]] = True
    return mask


def ensure_participant_csv(csv_path, mask):
    """Persist and verify the vector; never silently overwrite a different one."""
    if os.path.exists(csv_path):
        prior = pd.read_csv(csv_path)
        if list(prior.columns) != ["user_id", "participant"]:
            raise ValueError(f"Unexpected columns in participant vector: {csv_path}")
        ids = prior["user_id"].to_numpy()
        if not np.array_equal(ids, np.arange(len(mask))):
            raise ValueError("Existing participant CSV is not indexed in user order.")
        prior_mask = parse_bool_series(prior["participant"]).to_numpy()
        if not np.array_equal(prior_mask, mask):
            raise ValueError("Existing fixed participants differ from the requested seed/rate.")
        return
    pd.DataFrame({
        "user_id": np.arange(len(mask), dtype=int),
        "participant": mask.astype(int),
    }).to_csv(csv_path, index=False)


def ensure_manifest(path, settings):
    """Guard against accidentally resuming with incompatible simulation inputs."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != settings:
            raise ValueError(f"Scenario settings changed since first run: {path}\n"
                             f"Previous: {existing}\nRequested: {settings}")
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2, sort_keys=True)


def parse_bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    unknown = sorted(set(lowered) - set(mapping))
    if unknown:
        raise ValueError(
            f"Unrecognised values in participant column: {unknown[:10]}"
        )
    return lowered.map(mapping).astype(bool)


def make_probability_sr_masks(select_social_np, number_of_days, p_sr, rng):
    """
    Return one Boolean SR-use mask per simulated day.

    This mirrors the current optimised main framework:
        sr_today = select_social_np & (rng.random(n_users) < p_sr)

    Therefore:
    - non-participants always use UC;
    - participants independently use SR on each simulated day with probability
      p_sr = social_days_number / days_in_week = 2/5 = 0.4;
    - the expected number of SR days over five days is 2, but it is not forced
      to equal exactly 2 for each individual user.
    """
    n_users = select_social_np.size
    masks = {}

    for day_id in range(1, number_of_days + 1):
        sr_today = select_social_np & (rng.random(n_users) < p_sr)
        masks[day_id] = sr_today

    return masks


def write_demand_pickles(
    masks,
    veh_social,
    veh_init,
    temp_folder_path,
):
    """
    Build daily demand exactly as in the optimised framework:
    start from UC demand and replace rows with SR rows for today's SR users.
    """
    demand_files = {}

    for day, sr_today in masks.items():
        demand_day = veh_init.copy()
        demand_day.loc[sr_today] = veh_social.loc[sr_today]

        demand_file = os.path.join(temp_folder_path, f"demand_day_{day}.pickle")
        demand_day.to_pickle(demand_file, protocol=pickle.HIGHEST_PROTOCOL)
        demand_files[day] = demand_file

    return demand_files


def run_symuvia_optimised(iteration_cycle, input_data, parallel_cores,
                          batch_timeout_minutes=80.0):
    """Use the existing SymuVia worker, but do not wait indefinitely for a hung job.

    Completed outputs are retained even when a different worker fails to return.
    The existing internal 60-minute gridlock abort still handles normal failures.
    """
    print(
        f"[INFO] Round {iteration_cycle}: starting {len(input_data)} day tasks "
        f"with {parallel_cores} workers; outer watchdog {batch_timeout_minutes:.0f} min.",
        flush=True,
    )
    pool = Pool(processes=parallel_cores, maxtasksperchild=1)
    timeout_s = batch_timeout_minutes * 60.0
    start = time.monotonic()
    futures = []
    try:
        futures = [(int(item[0]), pool.apply_async(day_iteration.run_iteration_day, (item,)))
                   for item in input_data]
        while (time.monotonic() - start) < timeout_s:
            if all(future.ready() for _, future in futures):
                break
            time.sleep(min(5.0, max(0.01, timeout_s - (time.monotonic() - start))))

        incomplete = [(day, future) for day, future in futures if not future.ready()]
        results = []
        for day, future in futures:
            if not future.ready():
                results.append({"day": day, "status": "batch_timeout",
                                "reason": "worker did not return before outer watchdog"})
            else:
                try:
                    result = future.get(timeout=0)
                    if not isinstance(result, dict):
                        raise TypeError(f"Worker returned {type(result).__name__}, not dict")
                    results.append(result)
                except Exception as exc:
                    results.append({"day": day, "status": "worker_exception",
                                    "reason": repr(exc)})
        if incomplete:
            print(f"[WARN] Batch watchdog expired; terminating {len(incomplete)} "
                  "unfinished worker(s), retaining completed outputs.", flush=True)
            pool.terminate()
        else:
            pool.close()
        pool.join()
        return results
    except BaseException:
        pool.terminate()
        pool.join()
        raise


def incomplete_trips_scalar(value):
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return 0.0
        return float(value.max())
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size == 0:
            return 0.0
        return float(np.nanmax(arr))
    return float(value)


def user_tt_table(driver_tt):
    if isinstance(driver_tt, pd.Series):
        values = driver_tt.to_numpy(dtype=float)
    else:
        values = np.asarray(driver_tt, dtype=float)

    return pd.DataFrame(
        {
            "user_index": np.arange(len(values), dtype=int),
            "driver_travel_time": values,
        }
    )


def aligned_day_travel_times(extraction_csv, demand_pickle, extracted_tt, expected_users):
    """Align XML-output travel times to canonical UC/SR demand user positions.

    Extractor v4 returns its NumPy travel-time vector in XML vehicle order,
    which is not necessarily demand order. The extractor also writes the
    corresponding vehicle attributes to CSV. Match records by the unique
    OD/departure key and fail rather than silently mislabel individual users.
    """
    demand = pd.read_pickle(demand_pickle)
    observed = pd.read_csv(extraction_csv, sep=";")
    keys = ["origin", "destination", "creation_time"]
    if len(demand) != expected_users or len(observed) != expected_users:
        raise ValueError("Input/output traveller count mismatch for user-level alignment")
    if len(extracted_tt) != len(observed):
        raise ValueError("Extractor array and vehicle CSV have different lengths")
    if not np.allclose(observed["travel_time"].to_numpy(dtype=float),
                       extracted_tt, rtol=1e-5, atol=1e-3):
        raise ValueError("Extractor's CSV and returned travel times differ")
    # Input departure times are specified to 3 decimal places in the 4h files.
    # Rounding also avoids float string-serialization noise in SymuFlow XML.
    for frame in (demand, observed):
        frame["creation_time"] = pd.to_numeric(frame["creation_time"], errors="raise")
        frame["creation_time"] = frame["creation_time"].round(3)
        if frame.duplicated(keys).any():
            raise ValueError("Ambiguous origin/destination/departure match after rounding")
    matched = demand[keys].merge(
        observed[keys + ["travel_time"]], on=keys, how="left",
        sort=False, validate="one_to_one", indicator=True,
    )
    if not (matched["_merge"] == "both").all():
        raise ValueError("Not all simulated vehicles match original demand by OD/departure")
    values = matched["travel_time"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Aligned traveller time contains a non-finite value")
    return values


def save_checkpoint_atomic(path, checkpoint):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def save_final_pickle(
    pickle_file,
    zip_file,
    expnum,
    participation_rate,
    social_days_number,
    total_user_experienced_time_day,
    user_experienced_time_day,
    successful_day_metadata,
    attempted_day_metadata,
    select_social,
    number_of_days,
    number_of_users,
    scheme_type,
    sim_hours,
    total_sample_size,
):
    key = (expnum, participation_rate, social_days_number)
    record = {
        "total_user_experienced_time_day": total_user_experienced_time_day,
        "user_experienced_time_day": user_experienced_time_day,
        "successful_day_metadata": successful_day_metadata,
        "attempted_day_metadata": attempted_day_metadata,
        "select_social": select_social,
        "number_of_days": number_of_days,
        "number_of_users": number_of_users,
        "scheme_type": scheme_type,
        "sim_hours": sim_hours,
        "total_sample_size": total_sample_size,
        "source_mode": "fixed_random_participation_optimised_probability_SR_days",
    }
    outer = {key: record}

    with open(pickle_file, "wb") as f:
        pickle.dump(outer, f, protocol=pickle.HIGHEST_PROTOCOL)

    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pickle_file, os.path.basename(pickle_file))


def main():
    args = parse_args()

    if args.parallel_cores < 5 or args.parallel_cores % 5 != 0:
        raise ValueError("--parallel-cores must be a positive multiple of 5.")
    if args.total_sample_size < 15:
        raise ValueError("--total-sample-size should be at least 15 for validation.")
    if args.rounds_per_job < 1 or args.max_rounds < 1 or args.max_empty_rounds < 1:
        raise ValueError("Round limits must be positive.")
    if args.batch_timeout_minutes <= 0:
        raise ValueError("--batch-timeout-minutes must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    scheme_type = "FixedRandomParticipation"
    # Preserve the original 3-part result key layout for downstream analysis.
    expnum = "1"
    social_days_number = 2
    days_in_week = 5
    p_sr = social_days_number / days_in_week
    weeks_in_batch = args.parallel_cores // days_in_week
    number_stack_simulations = weeks_in_batch * days_in_week

    current_dir = os.getcwd()
    inputs_root = os.path.abspath(
        os.path.join(current_dir, os.pardir, "1_SymuVia_Lyon_Inputs")
    )

    if args.sim_hours == 4:
        demand_path = os.path.join(inputs_root, "4_hour_demand_profile")
        network_input = os.path.join(demand_path, "L63V_4h.xml")
        demand_input_social = os.path.join(demand_path, "SR_routes_4h.csv")
        demand_input_init = os.path.join(demand_path, "UC_routes_4h.csv")
    else:
        demand_path = os.path.join(inputs_root, "2_hour_demand_profile")
        network_input = os.path.join(demand_path, "L63V_2h.xml")
        demand_input_social = os.path.join(demand_path, "social_routes_2h.csv")
        demand_input_init = os.path.join(demand_path, "init_routes_2h.csv")

    veh_social_1 = pd.read_csv(demand_input_social, sep=";")
    veh_init_1 = pd.read_csv(demand_input_init, sep=";")

    veh_social = pd.merge(
        veh_social_1,
        veh_init_1[["origin", "destination", "creation_time"]],
        on=["origin", "destination", "creation_time"],
        how="inner",
    )
    veh_init = pd.merge(
        veh_init_1,
        veh_social[["origin", "destination", "creation_time"]],
        on=["origin", "destination", "creation_time"],
        how="inner",
    )

    # UC/SR data carry their own vehicle IDs, which need not agree. The stable
    # one-to-one traveller key in these Lyon inputs is OD plus departure time.
    # Keep the UC row ordering as the canonical user ordering for the fixed mask.
    veh_init = veh_init.reset_index(drop=True)
    keys = ["origin", "destination", "creation_time"]
    if veh_init.duplicated(keys).any() or veh_social.duplicated(keys).any():
        raise ValueError("UC/SR origin-destination-departure keys are not unique.")
    if len(veh_social) != len(veh_init):
        raise ValueError("UC/SR traveller counts differ after joining.")
    veh_social = veh_init[keys].merge(
        veh_social, on=keys, how="left", sort=False,
        validate="one_to_one", indicator=True,
    )
    if not (veh_social["_merge"] == "both").all():
        raise ValueError("UC traveller without matching SR alternative.")
    veh_social = veh_social.drop(columns="_merge")
    if not veh_social[keys].equals(veh_init[keys]):
        raise ValueError("UC/SR matched travellers are not aligned by key.")
    # A vehicle's 'id' is not a stable identifier across the independently
    # generated UC/SR route CSVs. Keep the canonical UC ID in either demand.
    veh_social["id"] = veh_init["id"].to_numpy()
    print("[INFO] UC/SR matched one-to-one by OD and departure time; "
          "UC vehicle IDs retained as canonical.", flush=True)

    number_of_users = len(veh_init)
    print(
        f"[INFO] Simulation users={number_of_users}; "
        f"SR input users={len(veh_social_1)}; UC input users={len(veh_init_1)}.",
        flush=True,
    )

    if number_of_users != EXPECTED_USERS:
        raise ValueError(
            f"Simulation has {number_of_users} users; expected {EXPECTED_USERS}."
        )
    if len(veh_social) != number_of_users:
        raise ValueError("UC and SR merged demand tables are not aligned in length.")

    state_name = f"fixed_rate_{args.participation_percent}pct"
    folder_outputs = os.path.join(current_dir, args.output_root, state_name)
    os.makedirs(folder_outputs, exist_ok=True)

    settings = {
        "participation_percent": args.participation_percent,
        "participant_seed": args.participant_seed,
        "daily_seed": args.seed,
        "sim_hours": args.sim_hours,
        "parallel_cores": args.parallel_cores,
        "number_of_users": number_of_users,
        "sr_days_per_week_expected": 2,
        "sr_day_probability": p_sr,
        "simulation_stack": "Funcs.day_iteration_optimised / gridlockabort v5",
    }
    ensure_manifest(os.path.join(folder_outputs, "scenario_settings.json"), settings)

    select_social_np = fixed_participant_mask(
        number_of_users, args.participation_percent, args.participant_seed,
    )
    participation_rate = float(select_social_np.mean())
    select_social = {i: bool(v) for i, v in enumerate(select_social_np)}
    ensure_participant_csv(
        os.path.join(folder_outputs, "fixed_participants.csv"), select_social_np,
    )
    number_of_days = {
        user_id: p_sr * 10 for user_id in range(number_of_users)
    }

    run_num = args.participation_percent
    pickle_file = os.path.join(
        folder_outputs,
        f"individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.pickle",
    )
    zip_file = os.path.join(
        folder_outputs,
        f"individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.zip",
    )
    checkpoint_file = os.path.join(folder_outputs, "fixed_rate_checkpoint.pickle")

    if os.path.isfile(pickle_file) and not os.path.isfile(checkpoint_file):
        with open(pickle_file, "rb") as handle:
            stored = pickle.load(handle)
        existing = stored.get((expnum, participation_rate, social_days_number), {})
        completed = len(existing.get("total_user_experienced_time_day", {}))
        if completed >= args.total_sample_size:
            print(f"[DONE] Already have {completed} successful days at "
                  f"{args.participation_percent}%: {pickle_file}", flush=True)
            return
        raise RuntimeError("A partial final pickle exists without a checkpoint. "
                           "Investigate rather than overwriting it.")

    # Fast scratch workspace.
    scratch_state_root = os.path.join(args.scratch_root, state_name)
    temp_folder_path = os.path.join(scratch_state_root, "TEMP")
    out_folder_path = os.path.join(scratch_state_root, "OUT")

    # A fresh run should not inherit stale files from an earlier failed attempt.
    if not os.path.exists(checkpoint_file) and os.path.exists(scratch_state_root):
        print(
            f"[INFO] No checkpoint found: removing stale scratch workspace "
            f"{scratch_state_root}",
            flush=True,
        )
        shutil.rmtree(scratch_state_root)

    os.makedirs(temp_folder_path, exist_ok=True)
    os.makedirs(out_folder_path, exist_ok=True)

    network_file_stem = os.path.join(temp_folder_path, "L63V")
    out_folder = out_folder_path + os.sep
    temp_folder = os.path.join(temp_folder_path, "TEMP").rstrip(os.sep)

    start_sim_time = "06300"
    finish_sim_time = (
        f"{6 + args.sim_hours}300"
        if 6 + args.sim_hours >= 10
        else f"0{6 + args.sim_hours}300"
    )
    symuvia_output = f"_{start_sim_time}0_{finish_sim_time}0_traf.xml"
    lib_path = "/home/alexroocroft/.conda/envs/env_symupy/lib/"

    print(f"[INFO] Scenario: fixed random {args.participation_percent}% participation", flush=True)
    print(f"[INFO] Fixed participants: {int(select_social_np.sum())}/{number_of_users}", flush=True)
    print(f"[INFO] Participant seed: {args.participant_seed}; daily SR assignment seed: {args.seed}", flush=True)
    print(f"[INFO] Fixed participation: {100 * participation_rate:.4f}%", flush=True)
    print(f"[INFO] Target successful days: {args.total_sample_size}", flush=True)
    print(
        f"[INFO] Optimised simulations per round: {number_stack_simulations}",
        flush=True,
    )
    print(
        f"[INFO] SR-day assignment: independent Bernoulli per participant/day, "
        f"p_sr={p_sr:.3f}",
        flush=True,
    )
    print(f"[INFO] Persistent output: {folder_outputs}", flush=True)
    print(f"[INFO] Scratch workspace: {scratch_state_root}", flush=True)

    if os.path.exists(checkpoint_file):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_file}", flush=True)
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)

        iteration_cycle = checkpoint["iteration_cycle"]
        day_count = checkpoint["day_count"]
        total_user_experienced_time_day = checkpoint[
            "total_user_experienced_time_day"
        ]
        user_experienced_time_day = checkpoint["user_experienced_time_day"]
        successful_day_metadata = checkpoint["successful_day_metadata"]
        attempted_day_metadata = checkpoint.get("attempted_day_metadata", [])
        empty_rounds = checkpoint.get("empty_rounds", 0)

        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])

        print(
            f"[INFO] Checkpoint: {day_count}/{args.total_sample_size} successful "
            f"days after {iteration_cycle} rounds.",
            flush=True,
        )
    else:
        iteration_cycle = 0
        day_count = 0
        total_user_experienced_time_day = {}
        user_experienced_time_day = {}
        successful_day_metadata = {}
        attempted_day_metadata = []
        empty_rounds = 0

    batches_this_job = 0
    while day_count < args.total_sample_size and batches_this_job < args.rounds_per_job:
        if iteration_cycle >= args.max_rounds:
            raise RuntimeError(
                f"Reached --max-rounds={args.max_rounds} with only "
                f"{day_count}/{args.total_sample_size} successful days."
            )

        iteration_cycle += 1
        batches_this_job += 1
        round_start = time.perf_counter()

        print("------------------------------------------------", flush=True)
        print(f"Sampling round: {iteration_cycle}", flush=True)
        print("------------------------------------------------", flush=True)

        # Same SR-day mechanism as the current optimised main framework:
        # each fixed participant independently uses SR on a simulated day with
        # probability p_sr = 2/5.
        #
        # In the main framework the batch RNG is seeded from the run seed plus
        # cycle/rerun counters. Here each validation round gets a deterministic
        # independent stream derived from the supplied seed.
        rng = np.random.default_rng(args.seed + iteration_cycle + 1)
        masks = make_probability_sr_masks(
            select_social_np,
            number_stack_simulations,
            p_sr,
            rng,
        )

        demand_start = time.perf_counter()
        demand_files = write_demand_pickles(
            masks,
            veh_social,
            veh_init,
            temp_folder_path,
        )
        print(
            f"[TIMING] Round {iteration_cycle}: demand creation/write "
            f"{time.perf_counter() - demand_start:.1f} s",
            flush=True,
        )

        input_data = [
            (
                day,
                network_input,
                network_file_stem,
                demand_files[day],
                iteration_cycle,
                lib_path,
            )
            for day in range(1, number_stack_simulations + 1)
        ]

        sim_start = time.perf_counter()
        results = run_symuvia_optimised(
            iteration_cycle,
            input_data,
            args.parallel_cores,
            batch_timeout_minutes=args.batch_timeout_minutes,
        )
        print(
            f"[TIMING] Round {iteration_cycle}: optimised SymuVia batch "
            f"{(time.perf_counter() - sim_start) / 60.0:.2f} min",
            flush=True,
        )

        # Interpret the same status dictionary returned by the current optimised
        # long-horizon framework.
        result_by_day = {}
        for res in results:
            if res is None:
                continue
            result_by_day[int(res["day"])] = res

        extract_start = time.perf_counter()
        successes_before = day_count
        extraction_errors = []

        for day in range(1, number_stack_simulations + 1):
            if day_count >= args.total_sample_size:
                break

            res = result_by_day.get(day)
            status = "no_result" if res is None else str(res.get("status", "unknown"))
            reason = None if res is None else res.get("reason")

            attempt_record = {
                "round": iteration_cycle,
                "iteration_day": day,
                "status": status,
                "reason": None if reason is None else str(reason),
            }

            if status != "completed":
                attempted_day_metadata.append(attempt_record)
                print(
                    f"[WARN] Round {iteration_cycle}, day {day}: "
                    f"optimised runner status={status}, reason={reason}",
                    flush=True,
                )
                continue

            symuvia_output_parallel = "defaultOut" + symuvia_output
            temp_folder_parallel = (
                temp_folder + f"_Cycle_{iteration_cycle}_Day_{day}/"
            )

            try:
                veh_output = Tripextract.symmvia_output(
                    out_folder,
                    temp_folder_parallel,
                    symuvia_output_parallel,
                    day,
                    iteration_cycle,
                    args.sim_hours,
                )
            except Exception as exc:
                attempt_record["status"] = "extraction_error"
                attempt_record["reason"] = repr(exc)
                attempted_day_metadata.append(attempt_record)
                extraction_errors.append((day, repr(exc)))
                print(
                    f"[WARN] Round {iteration_cycle}, day {day}: "
                    f"extraction failed -> {exc!r}",
                    flush=True,
                )
                continue

            incomplete = incomplete_trips_scalar(veh_output["incomplete_trips"])
            attempt_record["incomplete_trips"] = incomplete

            if incomplete > 0:
                attempt_record["status"] = "incomplete_trips"
                attempted_day_metadata.append(attempt_record)
                print(
                    f"[WARN] Round {iteration_cycle}, day {day}: "
                    f"incomplete_trips={incomplete}",
                    flush=True,
                )
                continue

            driver_tt = veh_output["driver_travel_time"]
            extracted_tt = (
                driver_tt.to_numpy(dtype=float)
                if isinstance(driver_tt, pd.Series)
                else np.asarray(driver_tt, dtype=float)
            )
            try:
                tt_values = aligned_day_travel_times(
                    os.path.join(out_folder_path,
                                 f"vehs_Day_{day}_cycle_{iteration_cycle}.csv"),
                    demand_files[day], extracted_tt, number_of_users,
                )
            except Exception as exc:
                attempt_record["status"] = "user_alignment_error"
                attempt_record["reason"] = repr(exc)
                attempted_day_metadata.append(attempt_record)
                extraction_errors.append((day, repr(exc)))
                print(f"[WARN] Round {iteration_cycle}, day {day}: "
                      f"user alignment failed -> {exc!r}", flush=True)
                continue

            if not np.all(np.isfinite(tt_values)):
                raise RuntimeError(f"Round {iteration_cycle}, day {day}: non-finite travel time detected.")
            if len(tt_values) != number_of_users:
                raise RuntimeError(
                    f"Round {iteration_cycle}, day {day}: extracted "
                    f"{len(tt_values)} user travel times; expected {number_of_users}."
                )

            day_count += 1
            total_user_experienced_time_day[day_count] = float(
                np.nansum(tt_values)
            )
            user_experienced_time_day[day_count] = user_tt_table(tt_values)
            successful_day_metadata[day_count] = {
                "cycle": iteration_cycle,
                "iteration_day": day,
                "temp_folder_parallel": temp_folder_parallel,
            }

            attempt_record["status"] = "successful_extracted"
            attempted_day_metadata.append(attempt_record)

            print(
                f"[INFO] Successful sample day {day_count}/"
                f"{args.total_sample_size} "
                f"(round {iteration_cycle}, day {day})",
                flush=True,
            )

        successes_this_round = day_count - successes_before

        print(
            f"[TIMING] Round {iteration_cycle}: extraction "
            f"{(time.perf_counter() - extract_start) / 60.0:.2f} min",
            flush=True,
        )
        print(
            f"[INFO] Round {iteration_cycle}: retained {successes_this_round}/"
            f"{number_stack_simulations} attempted days; total "
            f"{day_count}/{args.total_sample_size}.",
            flush=True,
        )

        empty_rounds = empty_rounds + 1 if successes_this_round == 0 else 0
        if extraction_errors:
            print(f"[WARN] Extraction errors in round {iteration_cycle}: "
                  f"{extraction_errors[:3]}", flush=True)

        checkpoint = {
            "iteration_cycle": iteration_cycle,
            "day_count": day_count,
            "total_user_experienced_time_day": total_user_experienced_time_day,
            "user_experienced_time_day": user_experienced_time_day,
            "successful_day_metadata": successful_day_metadata,
            "attempted_day_metadata": attempted_day_metadata,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "scratch_state_root": scratch_state_root,
            "empty_rounds": empty_rounds,
        }
        save_checkpoint_atomic(checkpoint_file, checkpoint)

        print(
            f"[INFO] Checkpoint saved: {checkpoint_file}",
            flush=True,
        )
        print(
            f"[TIMING] Round {iteration_cycle}: total "
            f"{(time.perf_counter() - round_start) / 60.0:.2f} min",
            flush=True,
        )
        if empty_rounds >= args.max_empty_rounds:
            raise RuntimeError(
                f"{empty_rounds} consecutive batches produced zero usable days at "
                f"{args.participation_percent}%. Check gridlock/failure logs before resubmitting. "
                "Progress has been checkpointed."
            )

    if day_count < args.total_sample_size:
        print(f"[PAUSED] {day_count}/{args.total_sample_size} successful days after "
              f"{iteration_cycle} batches. Resubmit the SAME Slurm array to resume.", flush=True)
        return

    save_final_pickle(
        pickle_file,
        zip_file,
        expnum,
        participation_rate,
        social_days_number,
        total_user_experienced_time_day,
        user_experienced_time_day,
        successful_day_metadata,
        attempted_day_metadata,
        select_social,
        number_of_days,
        number_of_users,
        scheme_type,
        args.sim_hours,
        args.total_sample_size,
    )

    # Save compact failure/retention summary to report selection bias.
    statuses = pd.Series([item["status"] for item in attempted_day_metadata])
    status_counts = statuses.value_counts().rename_axis("status").reset_index(name="count")
    status_counts.to_csv(os.path.join(folder_outputs, "attempted_day_status_counts.csv"), index=False)
    pd.DataFrame(attempted_day_metadata).to_csv(
        os.path.join(folder_outputs, "attempted_day_metadata.csv"), index=False,
    )
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    print(f"[DONE] Final pickle: {pickle_file}", flush=True)
    print(f"[DONE] Final zip:    {zip_file}", flush=True)
    print(
        "[INFO] Scratch files have deliberately been retained for failure "
        "diagnostics. Delete them only after checking the outputs.",
        flush=True,
    )


if __name__ == "__main__":
    main()
