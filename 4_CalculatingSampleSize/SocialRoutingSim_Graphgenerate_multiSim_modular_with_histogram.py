#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing for the computational-requirement / multiple-simulation analysis.

This script is intentionally graph/post-processing only. It assumes that
SocialRoutingSimulationFramework_multiple_simulations_modular.py has already
created/re-extracted the simulation outputs and saved a pickle containing at
least:

    total_user_experienced_time_day


The script produces:
    - Histogram of total network travel time across the 130-day pool.
    - Cumulative-mean convergence plot for total network travel time.
    - Confidence-interval half-width summary for random 15-day samples.

Created on 20/4/2023
Updated for revised computational-viability checks.

Author: Alexander Roocroft
"""

import os
import pickle
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


# =============================================================================
# User settings
# =============================================================================

RESULTS_FOLDER = "ComputationalReq_multiple_simulations" + os.sep
RESULTS_FOLDER_FIGS = os.path.join("ComputationalReq_multiple_simulations", "figs")

RUN_NUM = 1
EXP_NUM = "1"
INITIAL_PARTICIPATION_RATE = 0.2
SOCIAL_DAYS_NUMBER = 2
TOTAL_SAMPLE_SIZE = 130

NUM_RANDOM_ORDERINGS = 1000
RANDOM_SEED = 123

# Adopted sample size and one-day extension used in the paper discussion.
J_BASE = 15
J_NEXT = 16

# Total-network convergence threshold used in the paper.
AGGREGATE_CHANGE_THRESHOLD_PERCENT = 0.5


# =============================================================================
# Helper functions
# =============================================================================

def load_results_pickle(results_folder: str, run_num: int) -> Dict[Any, Any]:
    """Load the computational-requirement pickle."""
    pickle_file = os.path.join(
        results_folder,
        f"individual_users_output_parallel_multiSim_methodPaper_DTD_run_{run_num}.pickle",
    )

    if not os.path.exists(pickle_file):
        raise FileNotFoundError(
            f"Could not find expected pickle file:\n{pickle_file}\n"
            "Run the main simulation/re-extraction script first."
        )

    with open(pickle_file, "rb") as f:
        return pickle.load(f)


def get_record(data: Dict[Any, Any], exp_num: str, participation_rate: float, social_days_number: int) -> Dict[str, Any]:
    """Return the stored result record for the requested experiment key."""
    key = (exp_num, participation_rate, social_days_number)
    if key in data:
        return data[key]

    print("Available keys in pickle:")
    for available_key in data.keys():
        print("  ", available_key)

    raise KeyError(
        f"Could not find key {key}. Check EXP_NUM, INITIAL_PARTICIPATION_RATE, "
        "and SOCIAL_DAYS_NUMBER."
    )


def sorted_dict_values_first_n(d: Dict[Any, Any], n: int) -> List[Any]:
    """Return dictionary values sorted by key, keeping the first n."""
    return [d[k] for k in sorted(d.keys())][:n]


def make_random_orderings(n_days: int, n_orderings: int, seed: int) -> np.ndarray:
    """Generate random orderings of the day indices."""
    rng = np.random.default_rng(seed)
    return np.vstack([rng.permutation(n_days) for _ in range(n_orderings)])


def cumulative_percent_changes(days_pool_h: np.ndarray, orderings: np.ndarray) -> np.ndarray:
    """
    Compute absolute percentage changes in cumulative means.

    Returned array has shape (n_orderings, n_days - 1). Column 0 corresponds to
    the change from j=1 to j=2. In plots, this is labelled at x=2.
    """
    n_days = len(days_pool_h)
    denom = np.arange(1, n_days + 1, dtype=float)

    all_changes = []
    for order in orderings:
        ordered_days = days_pool_h[order]
        cumulative_means = np.cumsum(ordered_days) / denom
        diff = np.diff(cumulative_means)

        # Percentage change relative to the previous cumulative estimate.
        pct = 100.0 * diff / cumulative_means[:-1]
        all_changes.append(np.abs(pct))

    return np.vstack(all_changes)


def plot_total_tt_histogram(days_pool_h: np.ndarray, output_folder: str) -> None:
    """Plot histogram of total network travel time across the daily pool."""
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    axs.hist(days_pool_h, bins=15, edgecolor="black", alpha=0.75)

    mean_tt = np.mean(days_pool_h)
    axs.axvline(
        mean_tt,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean = {mean_tt:.0f} h",
    )

    # Keep the previous manual ticks if the values fall in this approximate range.
    if 15000 <= np.min(days_pool_h) and np.max(days_pool_h) <= 18500:
        axs.set_xticks([16000, 16500, 17000, 17500])

    axs.set_xlabel("Total network travel time [h]", fontsize=12)
    axs.set_ylabel("Number of daily simulations", fontsize=12)
    axs.tick_params(axis="both", which="major", labelsize=12)
    axs.legend(fontsize=10, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "MultiSim_TotalTravelTime_130day_histogram.pdf"), format="pdf")
    plt.close(fig)


def plot_cumulative_mean_convergence(all_percent_changes: np.ndarray, output_folder: str) -> Tuple[np.ndarray, np.ndarray]:
    """Plot median and maximum cumulative-mean percentage changes."""
    median_percent_changes = np.median(all_percent_changes, axis=0)
    maximum_percent_changes = np.max(all_percent_changes, axis=0)

    # all_percent_changes[:, 0] is the change from j=1 to j=2.
    x_values = np.arange(2, all_percent_changes.shape[1] + 2)

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    axs.plot(
        x_values,
        median_percent_changes,
        marker=".",
        linestyle="-",
        label="Median",
        markersize=6,
    )

    axs.plot(
        x_values,
        maximum_percent_changes,
        linestyle="dashdot",
        label="Maximum",
        markersize=5,
    )

    axs.fill_between(x_values, maximum_percent_changes, 0, color="gray", alpha=0.3)

    axs.axhline(
        y=AGGREGATE_CHANGE_THRESHOLD_PERCENT,
        color="red",
        linestyle="dotted",
        label="Threshold at 0.5%",
        linewidth=2,
    )

    axs.set_xlim(1, all_percent_changes.shape[1] + 1)
    axs.set_xlabel("No. of days included in cumulative mean", fontsize=12)
    axs.set_ylim(0, 5)
    axs.set_ylabel(r"$|\Delta\hat{TT}^{j}|$ [%]", fontsize=12)
    axs.tick_params(axis="both", which="major", labelsize=12)
    axs.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "MultiSim_CumulativeMeans_abs.pdf"), format="pdf")
    plt.close(fig)

    return median_percent_changes, maximum_percent_changes
    
def confidence_bound_check(days_pool_h: np.ndarray, j: int, n_perm: int, seed: int) -> pd.DataFrame:
    """Approximate relative half-width of 95% confidence intervals for j-day samples."""
    rng = np.random.default_rng(seed)
    rel = []

    for _ in range(n_perm):
        sample = rng.choice(days_pool_h, size=j, replace=False)
        half_width = t.ppf(0.975, j - 1) * np.std(sample, ddof=1) / np.sqrt(j)
        rel.append(100.0 * half_width / np.mean(sample))

    rel = np.asarray(rel)
    summary = pd.DataFrame(
        {
            "metric": ["relative_95ci_half_width_percent"],
            "j": [j],
            "n_random_samples": [n_perm],
            "median": [np.median(rel)],
            "p95": [np.percentile(rel, 95)],
            "p99": [np.percentile(rel, 99)],
            "max": [np.max(rel)],
        }
    )

    return summary


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_FOLDER_FIGS, exist_ok=True)

    df_time_output_exp_store = load_results_pickle(RESULTS_FOLDER, RUN_NUM)
    print("Done loading pickle.")

    record = get_record(
        df_time_output_exp_store,
        EXP_NUM,
        INITIAL_PARTICIPATION_RATE,
        SOCIAL_DAYS_NUMBER,
    )

    total_user_experienced_time_day = record["total_user_experienced_time_day"]
    total_tt_s = np.asarray(sorted_dict_values_first_n(total_user_experienced_time_day, TOTAL_SAMPLE_SIZE), dtype=float)
    total_tt_h = total_tt_s / 3600.0

    if len(total_tt_h) < TOTAL_SAMPLE_SIZE:
        raise ValueError(
            f"Only found {len(total_tt_h)} successful daily simulations, but TOTAL_SAMPLE_SIZE "
            f"is set to {TOTAL_SAMPLE_SIZE}."
        )

    days_pool_h = total_tt_h.copy()
    n_days = len(days_pool_h)

    print(f"Using {n_days} days for aggregate analysis.")
    # Generate a reproducible set of random orderings for the aggregate checks.
    random_orderings = make_random_orderings(n_days, NUM_RANDOM_ORDERINGS, RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Aggregate total-network travel time checks
    # -------------------------------------------------------------------------
    plot_total_tt_histogram(days_pool_h, RESULTS_FOLDER_FIGS)

    all_percent_changes = cumulative_percent_changes(days_pool_h, random_orderings)
    median_percent_changes, maximum_percent_changes = plot_cumulative_mean_convergence(
        all_percent_changes,
        RESULTS_FOLDER_FIGS,
    )

    # Column for change from j=15 to j=16 is j_next - 2, because column 0 is 1->2.
    idx_15_to_16 = J_NEXT - 2
    changes_15_to_16 = all_percent_changes[:, idx_15_to_16]
    n_below_threshold = int(np.sum(changes_15_to_16 < AGGREGATE_CHANGE_THRESHOLD_PERCENT))

    aggregate_summary = pd.DataFrame(
        {
            "metric": ["absolute_percent_change_total_tt_15_to_16"],
            "j_base": [J_BASE],
            "j_next": [J_NEXT],
            "n_random_orderings": [NUM_RANDOM_ORDERINGS],
            "threshold_percent": [AGGREGATE_CHANGE_THRESHOLD_PERCENT],
            "n_below_threshold": [n_below_threshold],
            "share_below_threshold": [n_below_threshold / NUM_RANDOM_ORDERINGS],
            "median": [np.median(changes_15_to_16)],
            "p95": [np.percentile(changes_15_to_16, 95)],
            "p99": [np.percentile(changes_15_to_16, 99)],
            "p999": [np.percentile(changes_15_to_16, 99.9)],
            "max": [np.max(changes_15_to_16)],
        }
    )
    aggregate_summary.to_csv(
        os.path.join(RESULTS_FOLDER_FIGS, "MultiSim_Aggregate_15to16_summary.csv"),
        index=False,
    )

    print("Aggregate 15->16 change summary:")
    print(aggregate_summary.to_string(index=False))

    # Save full aggregate percent-change distribution for reproducibility.
    aggregate_raw = pd.DataFrame(
        all_percent_changes,
        columns=[f"change_to_j_{j}" for j in range(2, n_days + 1)],
    )
    aggregate_raw.insert(0, "ordering_id", np.arange(1, NUM_RANDOM_ORDERINGS + 1))
    aggregate_raw.to_csv(
        os.path.join(RESULTS_FOLDER_FIGS, "MultiSim_Aggregate_all_orderings_percent_changes.csv"),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Confidence-bound check for aggregate daily outcomes
    # -------------------------------------------------------------------------
    confidence_summary = confidence_bound_check(
        days_pool_h,
        j=J_BASE,
        n_perm=NUM_RANDOM_ORDERINGS,
        seed=RANDOM_SEED,
    )
    confidence_summary.to_csv(
        os.path.join(RESULTS_FOLDER_FIGS, "MultiSim_Aggregate_95CI_summary.csv"),
        index=False,
    )

    print("Aggregate 95% CI relative half-width summary:")
    print(confidence_summary.to_string(index=False))

