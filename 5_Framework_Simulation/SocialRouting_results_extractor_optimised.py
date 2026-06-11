#!/usr/bin/env python3
"""
Social Routing – Results Extractor

Scans all simulation runs organised as:
  Output/Simulations/scheme_*/case_*_rep_*

Extracts:
- macro-level results per cycle
- per-user additional travel time (when UC baseline is available)

Outputs:
  extracted_output/macro_all_sim<hours>.csv
  extracted_output/peruser_all_sim<hours>.csv
  Optional ZIP bundling both.
"""

import argparse
import glob
import os
import pickle
import re
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import json
from pathlib import Path
import time
import numpy as np



HIST_BINS = 120
HIST_RANGE = (-60, 60)  # minutes (fulldays_additional_time is in minutes)

# -----------------------------
# Helpers
# -----------------------------
def make_zip_generic(zip_path: str, files: list[str]):
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=os.path.basename(f))
def safe_open_output(folder_outputs: str):
    """Load the main output.pickle with a clear error if missing."""
    pkl_path = os.path.join(folder_outputs, "output.pickle")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def find_all_run_dirs(results_folder: str) -> list[Path]:
    """
    Return list of run folders:
      <results_folder>/scheme_*/case_*_rep_*
    """
    base = Path(results_folder)
    run_dirs = []
    for scheme_dir in base.glob("scheme_*"):
        for run_dir in scheme_dir.glob("case_*_rep_*"):
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def cycles_from_storage(df_input_cyclemean_storage: Dict) -> List[int]:
    """Infer cycles from the storage dict keys, excluding 0 if present (initial state)."""
    keys = sorted(int(k) for k in df_input_cyclemean_storage.keys())
    return [k for k in keys if k != 0]

def compute_moral_index(moral_profile, user_param_store, scheme, user_id):
    """
    Utility-consistent per-user moral index.
    """
    required = {"Care", "Fairness", "Ingroup", "Authority", "Purity"}
    missing = required - moral_profile.keys()
    if missing:
        raise KeyError(f"Missing moral dimensions: {missing}")

    if scheme not in user_param_store:
        raise KeyError(
            f"Scheme '{scheme}' not found in user_param_store. "
            f"Available: {list(user_param_store.keys())}"
        )

    user_betas = user_param_store[scheme][user_id]

    return (
        user_betas["Care"]       * moral_profile["Care"]
      + user_betas["Fairness"]  * moral_profile["Fairness"]
      + user_betas["Ingroup"]   * moral_profile["Ingroup"]
      + user_betas["Authority"] * moral_profile["Authority"]
      + user_betas["Purity"]    * moral_profile["Purity"]
    )

# -----------------------------
# Core extraction for one run/profile
# -----------------------------
def extract_run(
    run_dir: Path,
    uc_prior_dir: str,
    sim_hours: int,
    limit_cycles=None,
):
    # --- read metadata ---
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        print(f"[WARN] Missing config.json in {run_dir} — skipping")
        return pd.DataFrame(), None, None

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    scheme = cfg.get("scheme", run_dir.parent.name.replace("scheme_", ""))

    # ------------------------------------------------------------
    # Reconstruct moral profiles EXACTLY as in framework
    # ------------------------------------------------------------
    overrides = cfg.get("overrides", {})
    profile_select = overrides.get("profile_select", 0)
    moral_num = overrides.get("moral_num", 1)

    User_and_Moral_Profile_path = Path(
        "/home/alexroocroft/GitHub_Version/2_User_and_Moral_Profile"
    )
    
    if sim_hours == 4:
        profile_pkl = User_and_Moral_Profile_path / "user_profile_4h.pickle"
    elif sim_hours == 2:
        profile_pkl = User_and_Moral_Profile_path / "user_profile_2h.pickle"
    else:
        raise ValueError("sim_hours must be 2 or 4")

    with open(profile_pkl, "rb") as f:
        user_param_store_stack, moral_user_stack = pickle.load(f)

    # Select correct stacks
    user_param_store = user_param_store_stack[profile_select]
    moral_user = moral_user_stack[profile_select].copy()

    # Apply SAME overwrite logic as framework
    number_of_users = len(moral_user)

    if moral_num == 2:  # most inclined
        if scheme == "Sacrifice":
            template = {"Care":1,"Fairness":1,"Ingroup":0,"Authority":1,"Purity":0}
        else:  # CollectiveGood
            template = {"Care":1,"Fairness":1,"Ingroup":0,"Authority":1,"Purity":1}

        for u in range(number_of_users):
            moral_user[u] = template.copy()

    elif moral_num == 3:  # least inclined
        if scheme == "Sacrifice":
            template = {"Care":0,"Fairness":0,"Ingroup":1,"Authority":0,"Purity":1}
        else:  # CollectiveGood
            template = {"Care":0,"Fairness":0,"Ingroup":1,"Authority":0,"Purity":0}

        for u in range(number_of_users):
            moral_user[u] = template.copy()


    case_id = cfg.get("case_id", None)
    case_label = cfg.get("case_label", f"case_{case_id}")
    rep_id = cfg.get("rep_id", None)

    folder_outputs = str(run_dir)

    try:
        (
            df_input_cyclemean_storage,
            participation_rate_store,
            df_out_storage,
            folder_outputs_check,
            select_social_store,
            lost_user_count,
            reconsideration_counter_store,
            perception_memory_store
        ) = safe_open_output(folder_outputs)

    except FileNotFoundError:
        print(f"[WARN] Skipping run rep {rep_id}: no output.pickle in {folder_outputs}")
        return pd.DataFrame(), None, None



    # ------------------------------------------------------------
    # Load UC prior from macro_output_fullUC_mean pickle
    # ------------------------------------------------------------
    uc_macro_pkl = (
        Path(uc_prior_dir)
        / f"macro_output_fullUC_mean_{sim_hours}h.pickle"
    )

    if not uc_macro_pkl.exists():
        raise FileNotFoundError(
            f"Missing UC macro prior: {uc_macro_pkl}"
        )

    with open(uc_macro_pkl, "rb") as f:
        mean_uc_tt, uc_df = pickle.load(f)

    # uc_df: DataFrame with per-user UC travel times
    # index = user_id, column = 'travel_time'
    if "travel_time" not in uc_df.columns:
        raise KeyError(
            f"'travel_time' column not found in {uc_macro_pkl}"
        )

    prior_uc = uc_df["travel_time"].to_dict()

    print(
        f"[INFO] Loaded UC prior from {uc_macro_pkl} "
        f"(mean={mean_uc_tt:.2f}, users={len(prior_uc)})"
    )


    # Determine cycles to process
    all_cycles = cycles_from_storage(df_input_cyclemean_storage)
    cycles = [c for c in all_cycles if (limit_cycles is None or c in limit_cycles)]
    if not cycles:
        return pd.DataFrame(), None, None
    last_cycle = max(cycles)
    # Macro aggregation
    macro_rows = []
    for c in cycles:
        df_cycle_mean = df_input_cyclemean_storage[c]
        # Expect a 'Network_benefit' column (hours). If not present, try fallback.
        # if "Network_benefit" in df_cycle_mean.columns:
        #     nb = float(pd.to_numeric(df_cycle_mean["Network_benefit"], errors="coerce").mean())
        if "Network_benefit_unscaled" in df_cycle_mean.columns:
            nb = float(df_cycle_mean["Network_benefit_unscaled"].iloc[0])
        else:
            nb = float("nan")

        pr = float(participation_rate_store.get(c, float("nan")))
        lost = int(lost_user_count.get(c, 0)) if isinstance(lost_user_count, dict) else 0

        number_of_users = len(prior_uc)

        macro_rows.append({
            "cycle": c,
            "participation_rate": pr,
            "network_benefit_mean": nb,
            "lost_users": lost,
            "number_of_users": number_of_users,
        })

    macro_df = pd.DataFrame(macro_rows).sort_values("cycle").reset_index(drop=True)

    macro_df["scheme"] = scheme
    macro_df["case_id"] = case_id
    macro_df["case_label"] = case_label
    macro_df["rep_id"] = rep_id


    # ------------------------------------------------------------
    # Per-user RAW extraction (all cycles, non-binned)
    # ------------------------------------------------------------
    peruser_rows = []

    for c in cycles:
        df = df_input_cyclemean_storage[c]
        participants = select_social_store.get(c, {})

        if "fulldays_additional_time" not in df.columns:
            raise KeyError(
                f"[Extractor error] Column 'fulldays_additional_time' not found in "
                f"df_input_cyclemean_storage[{c}] for run {run_dir}. "
                f"Available columns: {list(df.columns)}"
            )


        for user_id, row in df.iterrows():
        # --- ALL-DAYS additional time (minutes) ---
            fulldays_additional_tt = row.get("fulldays_additional_time", np.nan)
            if pd.isna(fulldays_additional_tt):
                raise ValueError(
                    f"[Extractor error] fulldays_additional_time is NaN for user_id={user_id} "
                    f"(scheme={scheme}, case_label={case_label}, rep_id={rep_id}, cycle={c})."
                )

            # --- UC baseline prior (seconds in macro_output_fullUC_mean_*h.pickle) ---
            prior_tt_sec = prior_uc.get(user_id, np.nan)
            if pd.isna(prior_tt_sec):
                raise KeyError(
                    f"[Extractor error] UC prior missing for user_id={user_id} "
                    f"(scheme={scheme}, case_label={case_label}, rep_id={rep_id}, cycle={c}). "
                    f"Check UC pickle index alignment with simulation user IDs."
                )

            # convert UC prior to minutes to match fulldays_additional_time (minutes)
            prior_tt = prior_tt_sec / 60.0

            # experienced (all-days cycle mean) in minutes
            experienced_tt = row.get("experienced_tt_fulldays", np.nan)
            if pd.isna(experienced_tt):
                raise ValueError(
                    f"[Extractor error] experienced_tt_fulldays is NaN for "
                    f"user_id={user_id}, cycle={c}, run={run_dir}"
                )



            try:
                moral_profile = moral_user[user_id]
            except KeyError as e:
                raise KeyError(
                    f"[Extractor error] Missing moral profile for user_id={user_id} "
                    f"in run {run_dir}"
                ) from e

            moral_index = compute_moral_index(
                moral_profile,
                user_param_store,
                scheme,
                user_id
            )


            peruser_rows.append({
                "scheme": scheme,
                "case_label": case_label,
                "rep_id": rep_id,
                "cycle": c,
                "user_id": user_id,
                "group": (
                    "participant"
                    if participants.get(user_id, False)
                    else "nonparticipant"
                ),
                "prior_tt": prior_tt,
                "experienced_tt": experienced_tt,
                "fulldays_additional_tt": fulldays_additional_tt, #full days (i.e. all days)
                "moral_index": moral_index,
            })

    peruser_raw_df = pd.DataFrame(peruser_rows)

    if peruser_raw_df["moral_index"].isna().any():
        raise RuntimeError("NaN moral_index detected — extractor failed")

    
    # ------------------------------------------------------------
    # Per-user HISTOGRAM extraction (last cycle only)
    #   We histogram ALL-DAYS additional time (minutes):
    #     fulldays_additional_time = total_additional_time_all / good_days_total
    # ------------------------------------------------------------

    hist_rows = []

    c = last_cycle
    df = df_input_cyclemean_storage[c]

    # ---- hard fail if the metric isn't present ----
    if "fulldays_additional_time" not in df.columns:
        raise KeyError(
            f"[Extractor error] Column 'fulldays_additional_time' not found in "
            f"df_input_cyclemean_storage[{c}] for run {run_dir}. "
            f"Available columns: {list(df.columns)}"
        )

    vals_series = pd.to_numeric(df["fulldays_additional_time"], errors="coerce")
    mask_valid = vals_series.notna()

    if mask_valid.sum() == 0:
        raise ValueError(
            f"[Extractor error] All fulldays_additional_time values are NaN for "
            f"run {run_dir}, last_cycle={c}."
        )

    values = vals_series[mask_valid].to_numpy()

    # Participant status for this cycle (aligned to df.index)
    participants = select_social_store.get(c, {}) if isinstance(select_social_store, dict) else {}
    is_participant = (
        df.index.map(lambda u: bool(participants.get(u, False)))[mask_valid]
        .to_numpy()
    )

    for group_name, mask in [
        ("participant", is_participant),
        ("nonparticipant", ~is_participant),
    ]:
        vals = values[mask]
        if len(vals) == 0:
            continue

        counts, edges = np.histogram(
            vals,
            bins=HIST_BINS,
            range=HIST_RANGE,  # minutes
        )

        for i in range(len(counts)):
            hist_rows.append({
                "scheme": scheme,
                "case_id": case_id,
                "case_label": case_label,
                "rep_id": rep_id,
                "cycle": c,   # last cycle only
                "group": group_name,
                "bin_left": edges[i],
                "bin_right": edges[i + 1],
                "count": int(counts[i]),
            })

    hist_df = pd.DataFrame(hist_rows)




    return macro_df, hist_df, peruser_raw_df


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Social Routing simulation results across all schemes, "
            "experiment cases, and repetitions into unified CSV outputs."
        )
    )

    parser.add_argument(
        "--results-folder",
        required=True,
        help="Root folder containing Output/Simulations/scheme_*/case_*_rep_* directories."
    )

    parser.add_argument(
        "--sim-hours",
        type=int,
        choices=[2, 4],
        required=True,
        help="Simulation duration in hours (used to select UC prior files)."
    )

    parser.add_argument(
        "--cycles",
        default="auto",
        help="Comma-separated list of cycles to extract (e.g., '1,2,3'), "
             "or 'auto' to infer available cycles per run."
    )


    parser.add_argument(
        "--zip",
        action="store_true",
        help="Bundle the extracted macro and per-user CSV files into a ZIP archive."
    )

    parser.add_argument(
        "--uc-prior-dir",
        type=str,
        default="/home/alexroocroft/GitHub_Version/3_Complete_SR_and_prior_UC",
        help="Path to the folder containing full-UC prior pickle files."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    start_time = time.time()
    print("[INFO] Results extractor started.")

    # Create dedicated subfolder for extracted outputs
    save_folder = os.path.join(args.results_folder, "extracted_output")
    os.makedirs(save_folder, exist_ok=True)

    limit_cycles = None if args.cycles == "auto" else [int(x) for x in args.cycles.split(",") if x.strip() != ""]

    # ============================================================
    # Scan all runs and aggregate results
    # ============================================================

    run_dirs = find_all_run_dirs(args.results_folder)

    macro_path = os.path.join(save_folder, f"macro_all_sim{args.sim_hours}.csv")
    peruser_path = os.path.join(save_folder, f"peruser_hist_all_sim{args.sim_hours}.csv")
    peruser_raw_path = os.path.join(
    save_folder,
        f"peruser_all_sim{args.sim_hours}.csv"
    )

    first_macro = True
    first_peruser = True
    first_peruser_raw = True



    for i, run_dir in enumerate(run_dirs, start=1):
        run_start = time.time()
        print(f"[INFO] ({i}/{len(run_dirs)}) Processing {run_dir.name}")

        macro_df, peruser_hist_df, peruser_raw_df = extract_run(
            run_dir=run_dir,
            uc_prior_dir=args.uc_prior_dir,
            sim_hours=args.sim_hours,
            limit_cycles=limit_cycles,
        )

        run_time = time.time() - run_start
        print(f"[INFO] Finished {run_dir.name} in {run_time/60:.1f} minutes")

        elapsed = time.time() - start_time
        avg_per_run = elapsed / i
        remaining = avg_per_run * (len(run_dirs) - i)

        print(
            f"[INFO] Avg/run: {avg_per_run/60:.1f} min | "
            f"ETA: {remaining/3600:.2f} hours"
        )

        if not macro_df.empty:
            macro_df.to_csv(
                macro_path,
                mode="w" if first_macro else "a",
                header=first_macro,
                index=False,
            )
            first_macro = False

        if peruser_hist_df is not None and not peruser_hist_df.empty:
            peruser_hist_df.to_csv(
                peruser_path,
                mode="w" if first_peruser else "a",
                header=first_peruser,
                index=False,
            )
            first_peruser = False

        if peruser_raw_df is not None and not peruser_raw_df.empty:
            peruser_raw_df.to_csv(
                peruser_raw_path,
                mode="w" if first_peruser_raw else "a",
                header=first_peruser_raw,
                index=False,
            )
            first_peruser_raw = False



    # ------------------------------------------------------------
    # Optional ZIP
    # ------------------------------------------------------------
    files_to_zip = [
        os.path.join(save_folder, f"macro_all_sim{args.sim_hours}.csv"),
        os.path.join(save_folder, f"peruser_hist_all_sim{args.sim_hours}.csv"),
        os.path.join(save_folder, f"peruser_all_sim{args.sim_hours}.csv"),
    ]

    # Keep only files that actually exist
    files_to_zip = [f for f in files_to_zip if os.path.exists(f)]


    if args.zip and files_to_zip:
        zip_path = os.path.join(
            save_folder,
            f"extracted_all_sim{args.sim_hours}.zip"
        )
        make_zip_generic(zip_path, files_to_zip)
        print(f"[OK] Zipped outputs to {zip_path}")
    elif args.zip:
        print("[WARN] ZIP requested but no output files found.")
        
    print("\nDone. Unified outputs generated across all schemes, cases, and repetitions.")

    total_time = time.time() - start_time
    print(f"[INFO] Total extractor runtime: {total_time/3600:.2f} hours")

if __name__ == "__main__":
    main()
