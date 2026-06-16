#!/usr/bin/env python3
"""
Generate plots from extractor outputs (macro + per-user CSVs)
Usage:
    python generate_graphs_from_extractor_optimised_v4.py \
        --results-folder /scratch/alexroocroft/Output/Simulations_XXXXX \
        --sim-hours 4 \
        --save-folder graphs_v4

v5 compared to v4. This changed summarise_band to make the bands min and max instead of 10/90 percentiles.
"""
import os
import json
import argparse
import gc
import pickle
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FULL_SR_NETWORK_BENEFIT_ALL = -9605.94387500002
FULL_SR_LINE_LEVEL = abs(FULL_SR_NETWORK_BENEFIT_ALL)  # 9605.94 shown on 0–10000 axis

MORAL_COEFFICIENTS = {
    "Sacrifice": {
        "Care": 2.047,
        "Fairness": 0.807,
        "Ingroup": -1.418,
        "Authority": 0.819,
        "Purity": -1.083,
    },
    "CollectiveGood": {
        "Care": 0.479,
        "Fairness": 1.34,
        "Ingroup": -2.845,
        "Authority": 2.184,
        "Purity": 1.486,
    },
}
MORAL_PROFILE_COLS = ["Care", "Fairness", "Ingroup", "Authority", "Purity"]


# Base font sizes
BASE_LABEL_FS = 10
BASE_TICK_FS = 10
BASE_LEGEND_FS = 10

# Requested scale factors
COMBINED_SCALE = 1.5
OTHER_SCALE = 1.85
OTHER_FIG_SCALE = 1.2


def other_figsize(width, height):
    """Slightly enlarge non-combined figures to accommodate scaled text."""
    return (width * OTHER_FIG_SCALE, height * OTHER_FIG_SCALE)


# ============================================================
# Helpers
# ============================================================
def apply_scheme_specific_moral_score(df):
    """Build a scheme-specific net moral score from the logit-model coefficients when
    the moral profile columns are available. Falls back to the existing moral_index
    column otherwise.
    """
    df = df.copy()

    has_profile_cols = all(col in df.columns for col in MORAL_PROFILE_COLS)
    if not has_profile_cols:
        if "moral_index" not in df.columns:
            raise KeyError(
                "Need either 'moral_index' or the five moral profile columns "
                f"{MORAL_PROFILE_COLS} in per-user data."
            )
        df["moral_index"] = pd.to_numeric(df["moral_index"], errors="coerce")
        return df

    for col in MORAL_PROFILE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    def _row_score(row):
        coeffs = MORAL_COEFFICIENTS.get(row["scheme"])
        if coeffs is None:
            return np.nan
        return sum(row[col] * coeffs[col] for col in MORAL_PROFILE_COLS)

    df["moral_index"] = df.apply(_row_score, axis=1)
    return df


def assign_moral_score_group(series):
    """Group net weighted moral scores into interpretable bins."""
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=s.index, dtype=object)
    out[s < 0] = "Negative"
    out[np.isclose(s, 0.0)] = "Zero"
    out[(s > 0) & (s <= 2)] = "Positive up to 2"
    out[s > 2] = "High positive >2"
    return out


def make_plot_dir(base, scheme, metric):
    """
    Create and return:
    <base>/<scheme>/<metric>
    """
    path = os.path.join(base, scheme, metric)
    os.makedirs(path, exist_ok=True)
    return path


def compute_global_network_benefit_ylim(macro_df, pad_ratio=0.05):
    """Return a shared upper y-limit for all network-benefit plots."""
    max_result = pd.to_numeric(macro_df["network_benefit_mean"], errors="coerce").max()
    max_result = 0.0 if pd.isna(max_result) else float(max_result)
    max_val = max(max_result, float(FULL_SR_LINE_LEVEL), 0.0)
    return max_val * (1.0 + pad_ratio) if max_val > 0 else 1.0


def find_all_run_dirs(results_folder: str):
    base = Path(results_folder)
    run_dirs = []
    for scheme_dir in base.glob("scheme_*"):
        for run_dir in scheme_dir.glob("case_*_rep_*"):
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def load_initial_target_participation_from_pickles(results_folder: str):
    """
    Read participation_rate_store[0] from each run's output.pickle.
    Also returns initial_lost_users (=0 at cycle 1 start state).
    """
    rows = []

    for run_dir in find_all_run_dirs(results_folder):
        cfg_path = run_dir / "config.json"
        out_path = run_dir / "output.pickle"

        if (not cfg_path.exists()) or (not out_path.exists()):
            continue

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        with open(out_path, "rb") as f:
            (
                df_input_cyclemean_storage,
                participation_rate_store,
                df_out_storage,
                folder_outputs,
                select_social_store,
                lost_user_count,
                reconsideration_counter_store,
                perception_memory_store,
            ) = pickle.load(f)

        rows.append(
            {
                "scheme": cfg.get("scheme", run_dir.parent.name.replace("scheme_", "")),
                "case_label": cfg.get("case_label", f"case_{cfg.get('case_id', 'unknown')}"),
                "rep_id": cfg.get("rep_id", None),
                "initial_participation_rate": float(participation_rate_store.get(0, np.nan)),
                "initial_lost_users": 0,
            }
        )

    return pd.DataFrame(rows)


def summarise_band(df_in, x_col, y_col):
    return (
        df_in.groupby(x_col)[y_col]
        .agg(
            q10="min",
            mean="mean",
            q90="max",
        )
        .reset_index()
        .sort_values(x_col)
    )


def compute_global_network_benefit_ymin(macro_df, pad_ratio=0.05):
    """Return a shared lower y-limit for all network-benefit plots, revealing negatives when present."""
    min_result = pd.to_numeric(macro_df["network_benefit_mean"], errors="coerce").min()
    if pd.isna(min_result):
        min_result = 0.0
    min_result = float(min_result)
    if min_result >= 0:
        return 0.0
    return min_result * (1.0 + pad_ratio)


def compute_casegroup_network_benefit_ymin(df_in, pad_ratio=0.05):
    """Return a lower y-limit for a specific network-benefit figure."""
    min_result = pd.to_numeric(df_in["network_benefit_mean"], errors="coerce").min()
    if pd.isna(min_result):
        min_result = 0.0
    min_result = float(min_result)
    if min_result >= 0:
        return 0.0
    return min_result * (1.0 + pad_ratio)

def style_axes(ax, label_fs, tick_fs, legend_fs=None):
    """Apply consistent font sizes to one axes."""
    ax.xaxis.label.set_size(label_fs)
    ax.yaxis.label.set_size(label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)

    leg = ax.get_legend()
    if leg is not None:
        if legend_fs is None:
            legend_fs = tick_fs
        for txt in leg.get_texts():
            txt.set_fontsize(legend_fs)


def style_figure(fig, label_fs, tick_fs, legend_fs=None):
    """Apply consistent font sizes to all axes in a figure."""
    for ax in fig.axes:
        style_axes(ax, label_fs=label_fs, tick_fs=tick_fs, legend_fs=legend_fs)


def set_cycle_axis(ax, max_cycle, include_zero=True):
    """Standardise cycle axes and leave a small margin after the final cycle."""
    max_cycle = int(max_cycle)
    right_pad = 0.4 if max_cycle == 20 else 0.0

    if max_cycle == 20:
        if include_zero:
            ax.set_xlim(0, max_cycle + right_pad)
            ax.set_xticks(np.arange(0, max_cycle + 1, 2))
        else:
            ax.set_xlim(1, max_cycle + right_pad)
            ax.set_xticks(np.arange(2, max_cycle + 1, 2))
    else:
        ax.set_xlim(0 if include_zero else 1, max_cycle + right_pad)


def build_aligned_participation_rows(df_case, init_case_df):
    """
    Plot participation as the state USED in each cycle.

    cycle 1 -> initial participation_rate_store[0]
    cycle 2 -> stored participation_rate at cycle 1
    ...
    cycle 20 -> stored participation_rate at cycle 19
    """
    rows = []

    for rep_id, g in df_case.groupby("rep_id"):
        g = g.sort_values("cycle").copy()
        match = init_case_df[init_case_df["rep_id"] == rep_id]
        if match.empty:
            print(f"[WARN] Missing initial participation for rep_id={rep_id}; skipping rep.")
            continue

        init_rate = float(match["initial_participation_rate"].iloc[0])

        # Drop last stored state because it would correspond to cycle 21
        g_trim = g.iloc[:-1].copy()
        g_trim["plot_cycle"] = g_trim["cycle"] + 1
        g_trim["plot_value"] = pd.to_numeric(g_trim["participation_rate"], errors="coerce")

        init_row = pd.DataFrame(
            {
                "rep_id": [rep_id],
                "plot_cycle": [1],
                "plot_value": [init_rate],
            }
        )

        rows.append(
            pd.concat(
                [init_row, g_trim[["rep_id", "plot_cycle", "plot_value"]]],
                ignore_index=True,
            )
        )

    if not rows:
        return pd.DataFrame(columns=["rep_id", "plot_cycle", "plot_value"])

    out = pd.concat(rows, ignore_index=True)
    out["plot_value"] = 100.0 * out["plot_value"]
    return out



def build_aligned_lost_rows(df_case):
    """
    Plot lost users as the state USED in each cycle.

    cycle 1 -> 0
    cycle 2 -> stored lost_users at cycle 1
    ...
    cycle 20 -> stored lost_users at cycle 19
    """
    rows = []

    for rep_id, g in df_case.groupby("rep_id"):
        g = g.sort_values("cycle").copy()
        g["lost_users_pct"] = 100.0 * g["lost_users"] / g["number_of_users"]

        # Drop last stored state because it would correspond to cycle 21
        g_trim = g.iloc[:-1].copy()
        g_trim["plot_cycle"] = g_trim["cycle"] + 1
        g_trim["plot_value"] = pd.to_numeric(g_trim["lost_users_pct"], errors="coerce")

        init_row = pd.DataFrame(
            {
                "rep_id": [rep_id],
                "plot_cycle": [1],
                "plot_value": [0.0],
            }
        )

        rows.append(
            pd.concat(
                [init_row, g_trim[["rep_id", "plot_cycle", "plot_value"]]],
                ignore_index=True,
            )
        )

    if not rows:
        return pd.DataFrame(columns=["rep_id", "plot_cycle", "plot_value"])

    return pd.concat(rows, ignore_index=True)



# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate graphs from Social Routing extractor CSV outputs."
    )
    parser.add_argument(
        "--results-folder",
        required=True,
        help="Folder containing extracted_output (macro_all_simX.csv, peruser_all_simX.csv).",
    )
    parser.add_argument(
        "--sim-hours",
        type=int,
        default=4,
        choices=[2, 4],
        help="Simulation duration (2 or 4 hours).",
    )
    parser.add_argument(
        "--save-folder",
        default="graphs",
        help="Folder to save generated plots.",
    )
    return parser.parse_args()


# ============================================================
# Per-user plots
# ============================================================
def plot_peruser_histograms(hist_csv_path, save_folder):
    print("[INFO] Plotting per-user histograms...")
    os.makedirs(save_folder, exist_ok=True)
    df = pd.read_csv(hist_csv_path)
    for (scheme, case_label), g in df.groupby(["scheme", "case_label"]):
        plt.figure(figsize=other_figsize(8, 5))
        for group, gg in g.groupby("group"):
            centers = 0.5 * (gg["bin_left"] + gg["bin_right"])
            width = gg["bin_right"].iloc[0] - gg["bin_left"].iloc[0]
            density = gg["count"] / (gg["count"].sum() * width)
            label = "Participants" if group == "participant" else "Non-participants"
            plt.bar(
                centers,
                density,
                width=width,
                alpha=0.5,
                label=label,
            )
        plt.xlabel("Mean Additional Travel Time (min)")
        plt.ylabel("Density")
        plt.tight_layout()
        out_dir = make_plot_dir(save_folder, scheme, "peruser_histograms")
        out = os.path.join(
            out_dir,
            f"peruser_distribution_{case_label}.pdf",
        )
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[OK] Saved {out}")
        gc.collect()


def _dominant_group(series):
    participant_count = (series == "participant").sum()
    nonparticipant_count = (series == "nonparticipant").sum()
    if participant_count > nonparticipant_count:
        return "participant"
    if nonparticipant_count > participant_count:
        return "nonparticipant"
    return series.iloc[-1]


def prepare_last_cycle_user_df(peruser_df):
    """Collapse last-cycle per-user rows across replications so each user is counted once per scheme."""
    last_cycle = peruser_df["cycle"].max()
    df = peruser_df[peruser_df["cycle"] == last_cycle].copy()
    needed_cols = [
        "scheme",
        "rep_id",
        "user_id",
        "group",
        "fulldays_additional_tt",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise KeyError(f"prepare_last_cycle_user_df missing required columns: {missing}")
    extra_profile_cols = [c for c in MORAL_PROFILE_COLS if c in df.columns]
    optional_cols = ["moral_index"] if "moral_index" in df.columns else []
    df = df[needed_cols + optional_cols + extra_profile_cols].copy()
    df = apply_scheme_specific_moral_score(df)
    df["fulldays_additional_tt"] = pd.to_numeric(df["fulldays_additional_tt"], errors="coerce")
    df["is_participant"] = df["group"].eq("participant")
    user_df = (
        df.groupby(["scheme", "user_id"], as_index=False)
        .agg(
            moral_index=("moral_index", "mean"),
            participant_rate=("is_participant", "mean"),
            mean_additional_tt=("fulldays_additional_tt", "mean"),
        )
    )
    user_df = user_df.dropna(subset=["moral_index", "mean_additional_tt"]).copy()
    user_df["is_participant"] = user_df["participant_rate"] > 0.5
    user_df["is_saver"] = user_df["mean_additional_tt"] < 0

    def make_group(row):
        if row["is_participant"] and row["is_saver"]:
            return "Participant saver"
        elif row["is_participant"] and (not row["is_saver"]):
            return "Participant non-saver"
        elif (not row["is_participant"]) and row["is_saver"]:
            return "Non-participant saver"
        else:
            return "Non-participant non-saver"

    user_df["group_label"] = user_df.apply(make_group, axis=1)
    user_df["moral_centered"] = user_df.groupby("scheme")["moral_index"].transform(lambda s: s - s.mean())
    user_df["moral_group"] = assign_moral_score_group(user_df["moral_index"])
    return user_df


def plot_moral_index_last_cycles(peruser_hist_df, save_folder, n_last_cycles=4):
    order = [
        "Participant saver",
        "Participant non-saver",
        "Non-participant saver",
        "Non-participant non-saver",
    ]
    outdir_combined = os.path.join(save_folder, "combined", "moral_analysis")
    os.makedirs(outdir_combined, exist_ok=True)
    user_df = prepare_last_cycle_user_df(peruser_hist_df)
    summary = (
        user_df.groupby(["scheme", "group_label"])["moral_index"]
        .agg(
            n_users="count",
            mean="mean",
            std="std",
            p25=lambda x: np.percentile(x, 25) if len(x) else np.nan,
            median="median",
            p75=lambda x: np.percentile(x, 75) if len(x) else np.nan,
        )
        .reset_index()
    )
    summary["group_label"] = pd.Categorical(summary["group_label"], categories=order, ordered=True)
    summary = summary.sort_values(["scheme", "group_label"])
    summary.to_csv(os.path.join(outdir_combined, "moral_index_lastcycle_summary.csv"), index=False)
    print("\n[DEBUG] moral_index_lastcycle_summary")
    print(summary.to_string(index=False))
    print("\n[DEBUG] total unique users per scheme")
    print(user_df.groupby("scheme")["user_id"].nunique())
    schemes = list(user_df["scheme"].dropna().unique())

    for scheme in schemes:
        scheme_df = user_df[user_df["scheme"] == scheme].copy()
        scheme_outdir = os.path.join(save_folder, scheme, "moral_analysis")
        os.makedirs(scheme_outdir, exist_ok=True)
        scheme_summary = (
            scheme_df.groupby("group_label")["moral_index"]
            .agg(
                n_users="count",
                mean="mean",
                std="std",
                p25=lambda x: np.percentile(x, 25) if len(x) else np.nan,
                median="median",
                p75=lambda x: np.percentile(x, 75) if len(x) else np.nan,
            )
            .reindex(order)
            .reset_index()
        )
        scheme_summary.to_csv(os.path.join(scheme_outdir, f"moral_index_lastcycle_{scheme}.csv"), index=False)

        fig, ax = plt.subplots(figsize=other_figsize(7.5, 4.8))
        data_centered = [scheme_df.loc[scheme_df["group_label"] == g, "moral_centered"].dropna().values for g in order]
        ax.boxplot(data_centered, labels=order, showfliers=False)
        ax.set_ylabel("Centered moral utility term")
        ax.set_xticklabels(order, rotation=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(scheme_outdir, "moral_centered_lastcycle.pdf"), bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=other_figsize(7.5, 4.8))
        x = np.arange(len(order))
        means, errs = [], []
        for g in order:
            vals = scheme_df.loc[scheme_df["group_label"] == g, "moral_index"].dropna().values
            if len(vals) == 0:
                means.append(np.nan)
                errs.append(np.nan)
            elif len(vals) == 1:
                means.append(vals[0])
                errs.append(0.0)
            else:
                means.append(np.mean(vals))
                errs.append(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(x, means, yerr=errs, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylabel("Mean moral utility term")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(scheme_outdir, "moral_means_ci_lastcycle.pdf"), bbox_inches="tight")
        plt.close()

    fig, axes = plt.subplots(1, len(schemes), figsize=(14, 4.8), sharey=True)
    if len(schemes) == 1:
        axes = [axes]
    for ax, scheme in zip(axes, schemes):
        scheme_df = user_df[user_df["scheme"] == scheme].copy()
        data_centered = [scheme_df.loc[scheme_df["group_label"] == g, "moral_centered"].dropna().values for g in order]
        ax.boxplot(data_centered, labels=order, showfliers=False)
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylabel("Centered moral utility term")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_combined, "moral_centered_lastcycle.pdf"), bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, len(schemes), figsize=(14, 4.8), sharey=True)
    if len(schemes) == 1:
        axes = [axes]
    for ax, scheme in zip(axes, schemes):
        scheme_df = user_df[user_df["scheme"] == scheme].copy()
        x = np.arange(len(order))
        means, errs = [], []
        for g in order:
            vals = scheme_df.loc[scheme_df["group_label"] == g, "moral_index"].dropna().values
            if len(vals) == 0:
                means.append(np.nan)
                errs.append(np.nan)
            elif len(vals) == 1:
                means.append(vals[0])
                errs.append(0.0)
            else:
                means.append(np.mean(vals))
                errs.append(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ax.errorbar(x, means, yerr=errs, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylabel("Mean moral utility term")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir_combined, "moral_means_ci_lastcycle.pdf"), bbox_inches="tight")
    plt.close()

    print("[OK] Saved centered and mean±CI moral plots for last cycle (combined and per-scheme)")
    gc.collect()


def plot_moral_buoyancy_last_cycle(peruser_df, save_folder, n_bins=18):
    """Continuous last-cycle moral/additional-time plots using scheme-specific
    net moral score groups, plus a coarse grouped-bar summary figure.
    """
    print("[INFO] Plotting last-cycle moral buoyancy figures (score groups + grouped bars)...")

    user_df = prepare_last_cycle_user_df(peruser_df)

    outdir_combined = os.path.join(save_folder, "combined", "moral_analysis")
    os.makedirs(outdir_combined, exist_ok=True)

    moral_group_order = ["Negative", "Zero", "Positive up to 2", "High positive >2"]
    status_order = [(True, "Participants"), (False, "Non-participants")]

    def make_bins(series, n_bins=18):
        lo = float(series.quantile(0.02))
        hi = float(series.quantile(0.98))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(series.min())
            hi = float(series.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        if np.unique(edges).size < 2:
            edges = np.array([lo - 0.5, hi + 0.5])
        return edges

    def _add_ci_band(ax, x, mean, ci, label=None):
        x = np.asarray(x)
        mean = np.asarray(mean)
        ci = np.asarray(ci)
        mask = np.isfinite(x) & np.isfinite(mean) & np.isfinite(ci)
        if mask.sum() == 0:
            return
        line, = ax.plot(x[mask], mean[mask], linewidth=2, marker="o", label=label)
        ax.fill_between(
            x[mask],
            mean[mask] - ci[mask],
            mean[mask] + ci[mask],
            alpha=0.18,
            color=line.get_color(),
        )

    def _coarse_category(series):
        cat = pd.Series(index=series.index, dtype=object)
        cat[series <= -5] = "≤ -5 min"
        cat[(series > -5) & (series <= 0)] = "-5 to 0 min"
        cat[(series > 0) & (series <= 5)] = "0 to 5 min"
        cat[series > 5] = "> 5 min"
        return cat

    def _scheme_plots(df_scheme, out_dir, scheme_name):
        os.makedirs(out_dir, exist_ok=True)

        edges = make_bins(df_scheme["mean_additional_tt"], n_bins=n_bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        df_scheme = df_scheme.copy()
        df_scheme["add_bin"] = pd.cut(
            df_scheme["mean_additional_tt"],
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        )

        # 1) additional time vs centered moral by participation status
        stats_rows = []
        for is_participant, label in status_order:
            sub = df_scheme[df_scheme["is_participant"] == is_participant].copy()
            grouped = sub.groupby("add_bin", observed=False)
            tmp = grouped["moral_centered"].agg(["mean", "std", "count"]).reset_index()
            tmp["x"] = mids[:len(tmp)]
            tmp["ci95"] = 1.96 * tmp["std"] / np.sqrt(tmp["count"].clip(lower=1))
            tmp.loc[tmp["count"] <= 1, "ci95"] = 0.0
            tmp["status"] = label
            stats_rows.append(tmp)

        fig, ax = plt.subplots(figsize=other_figsize(7.8, 4.8))
        stats_df = pd.concat(stats_rows, ignore_index=True)
        for label in [x[1] for x in status_order]:
            tmp = stats_df[stats_df["status"] == label]
            _add_ci_band(ax, tmp["x"], tmp["mean"], tmp["ci95"], label=label)

        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Mean additional travel time (min)")
        ax.set_ylabel("Centered moral utility term")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        style_figure(
            plt.gcf(),
            label_fs=BASE_LABEL_FS * OTHER_SCALE,
            tick_fs=BASE_TICK_FS * OTHER_SCALE,
            legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "additional_time_vs_moral_lastcycle.pdf"), bbox_inches="tight", pad_inches=0.08)
        plt.close()

        stats_df.to_csv(
            os.path.join(out_dir, f"additional_time_vs_moral_lastcycle_{scheme_name}.csv"),
            index=False,
        )

        # 2) participation share vs additional time, split by moral group
        quartile_rows = []
        for moral_group in moral_group_order:
            sub = df_scheme[df_scheme["moral_group"] == moral_group].copy()
            grouped = sub.groupby("add_bin", observed=False)["is_participant"]
            tmp = grouped.agg(participation_rate="mean", n="count").reset_index()
            tmp["x"] = mids[:len(tmp)]
            p = tmp["participation_rate"].fillna(0)
            n = tmp["n"].clip(lower=1)
            tmp["ci95"] = 1.96 * np.sqrt(p * (1 - p) / n)
            tmp["moral_group"] = moral_group
            quartile_rows.append(tmp)

        fig, ax = plt.subplots(figsize=other_figsize(7.8, 4.8))
        quartile_df = pd.concat(quartile_rows, ignore_index=True)
        for moral_group in moral_group_order:
            tmp = quartile_df[quartile_df["moral_group"] == moral_group]
            _add_ci_band(
                ax,
                tmp["x"],
                100.0 * tmp["participation_rate"],
                100.0 * tmp["ci95"],
                label=moral_group,
            )

        ax.set_xlabel("Mean additional travel time (min)")
        ax.set_ylabel("Participant share (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        style_figure(
            plt.gcf(),
            label_fs=BASE_LABEL_FS * OTHER_SCALE,
            tick_fs=BASE_TICK_FS * OTHER_SCALE,
            legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, "participation_vs_additional_time_moral_groups_lastcycle.pdf"),
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close()

        quartile_df.to_csv(
            os.path.join(out_dir, f"participation_vs_additional_time_moral_groups_lastcycle_{scheme_name}.csv"),
            index=False,
        )

        # 3) grouped-bar summary with coarse additional-time categories
        df_bar = df_scheme.copy()
        df_bar["additional_time_category"] = _coarse_category(df_bar["mean_additional_tt"])

        category_order = [
            "≤ -5 min",
            "-5 to 0 min",
            "0 to 5 min",
            "> 5 min",
        ]

        bar_rows = []
        for cat in category_order:
            for moral_group in moral_group_order:
                sub = df_bar[
                    (df_bar["additional_time_category"] == cat)
                    & (df_bar["moral_group"] == moral_group)
                ].copy()

                n = len(sub)
                if n == 0:
                    pr = np.nan
                    ci = np.nan
                else:
                    p = sub["is_participant"].mean()
                    pr = 100.0 * p
                    ci = 100.0 * 1.96 * np.sqrt(p * (1 - p) / n)

                bar_rows.append(
                    {
                        "additional_time_category": cat,
                        "moral_group": moral_group,
                        "participation_rate_pct": pr,
                        "ci95_pct": ci,
                        "n": n,
                    }
                )

        bar_df = pd.DataFrame(bar_rows)
        bar_df.to_csv(
            os.path.join(out_dir, f"participation_bar_moral_groups_lastcycle_{scheme_name}.csv"),
            index=False,
        )

        print(f"\n[DEBUG] grouped-bar counts for {scheme_name}")
        print(
            bar_df.pivot(
                index="additional_time_category",
                columns="moral_group",
                values="n",
            ).reindex(category_order)
        )

        fig, ax = plt.subplots(figsize=other_figsize(10.2, 4.8))
        x = np.arange(len(category_order))
        width = 0.18
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

        for moral_group, off in zip(moral_group_order, offsets):
            tmp = (
                bar_df[bar_df["moral_group"] == moral_group]
                .set_index("additional_time_category")
                .reindex(category_order)
                .reset_index()
            )
            ax.bar(
                x + off,
                tmp["participation_rate_pct"],
                width=width,
                label=moral_group,
                yerr=tmp["ci95_pct"],
                capsize=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(category_order, rotation=15)
        ax.set_ylabel("Participant share (%)")
        ax.set_xlabel("Mean additional travel time (min)")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")
        style_figure(
            plt.gcf(),
            label_fs=BASE_LABEL_FS * OTHER_SCALE,
            tick_fs=BASE_TICK_FS * OTHER_SCALE,
            legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, "participation_bar_moral_groups_lastcycle.pdf"),
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close()

        # 4) appendix-style hexbin
        fig, axes = plt.subplots(1, 2, figsize=other_figsize(12.2, 4.8), sharex=True, sharey=True)
        for ax, (is_participant, label) in zip(axes, status_order):
            sub = df_scheme[df_scheme["is_participant"] == is_participant].copy()
            x = pd.to_numeric(sub["mean_additional_tt"], errors="coerce")
            y = pd.to_numeric(sub["moral_centered"], errors="coerce")
            mask = x.notna() & y.notna()
            x = x[mask]
            y = y[mask]
            if len(x) > 0:
                ax.hexbin(x, y, gridsize=35, mincnt=1)
            ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel("Mean additional travel time (min)")
            ax.set_ylabel("Centered moral utility term")
            ax.grid(True, alpha=0.15)

        style_figure(
            plt.gcf(),
            label_fs=BASE_LABEL_FS * OTHER_SCALE,
            tick_fs=BASE_TICK_FS * OTHER_SCALE,
            legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, "additional_time_vs_moral_hexbin_lastcycle.pdf"),
            bbox_inches="tight",
            pad_inches=0.08,
        )
        plt.close()

    schemes = sorted(user_df["scheme"].dropna().unique())
    for scheme in schemes:
        _scheme_plots(
            user_df[user_df["scheme"] == scheme].copy(),
            os.path.join(save_folder, scheme, "moral_analysis"),
            scheme,
        )

    fig, axes = plt.subplots(1, len(schemes), figsize=(7.8 * len(schemes), 4.8), sharey=True)
    if len(schemes) == 1:
        axes = [axes]

    for ax, scheme in zip(axes, schemes):
        df_scheme = user_df[user_df["scheme"] == scheme].copy()
        edges = make_bins(df_scheme["mean_additional_tt"], n_bins=n_bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        df_scheme["add_bin"] = pd.cut(df_scheme["mean_additional_tt"], bins=edges, include_lowest=True, duplicates="drop")

        for moral_group in moral_group_order:
            sub = df_scheme[df_scheme["moral_group"] == moral_group].copy()
            tmp = sub.groupby("add_bin", observed=False)["is_participant"].agg(
                participation_rate="mean",
                n="count",
            ).reset_index()
            tmp["x"] = mids[:len(tmp)]
            p = tmp["participation_rate"].fillna(0)
            n = tmp["n"].clip(lower=1)
            tmp["ci95"] = 1.96 * np.sqrt(p * (1 - p) / n)

            _add_ci_band(
                ax,
                tmp["x"],
                100.0 * tmp["participation_rate"],
                100.0 * tmp["ci95"],
                label=moral_group,
            )

        ax.set_xlabel("Mean additional travel time (min)")
        ax.set_ylabel("Participant share (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    style_figure(
        plt.gcf(),
        label_fs=BASE_LABEL_FS * COMBINED_SCALE,
        tick_fs=BASE_TICK_FS * COMBINED_SCALE,
        legend_fs=BASE_LEGEND_FS * COMBINED_SCALE,
    )
    plt.savefig(
        os.path.join(outdir_combined, "participation_vs_additional_time_moral_groups_lastcycle.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    print("[OK] Saved moral-score-group buoyancy plots and grouped-bar summaries (combined and per-scheme)")
    gc.collect()


# ============================================================
# Macro plots: single-case reference plots (kept)
# ============================================================
def plot_macro_stats(macro_df, save_folder, results_folder, network_benefit_ylim=None, network_benefit_ymin=0.0):
    """Plot one file per case_label, but with corrected cycle alignment for participation/lost users."""
    print("[INFO] Generating per-case macro-level plots with aligned cycle states...")
    os.makedirs(save_folder, exist_ok=True)

    init_df = load_initial_target_participation_from_pickles(results_folder)

    for scheme in macro_df["scheme"].unique():
        df_scheme = macro_df[macro_df["scheme"] == scheme].copy()
        init_scheme = init_df[init_df["scheme"] == scheme].copy()

        for case_label in df_scheme["case_label"].unique():
            df_case = df_scheme[df_scheme["case_label"] == case_label].copy()
            df_case = df_case.sort_values(["rep_id", "cycle"])
            init_case = init_scheme[init_scheme["case_label"] == case_label].copy()

            # ---------------- Participation ----------------
            plt.figure(figsize=other_figsize(8, 5))
            part = build_aligned_participation_rows(df_case, init_case)
            if not part.empty:
                stats = summarise_band(part, "plot_cycle", "plot_value")
                plt.fill_between(stats["plot_cycle"], stats["q10"], stats["q90"], alpha=0.22)
                plt.plot(stats["plot_cycle"], stats["mean"], linewidth=2, marker="o", label="mean")
            plt.xlabel("Cycle")
            plt.ylabel("Participation Rate (%)")
            ax = plt.gca()
            set_cycle_axis(ax, int(df_case["cycle"].max()), include_zero=True)
            plt.ylim(0, 100)
            legend_loc = "lower right" if scheme == "Sacrifice" else "best"
            plt.legend(loc=legend_loc)
            style_figure(
                plt.gcf(),
                label_fs=BASE_LABEL_FS * OTHER_SCALE,
                tick_fs=BASE_TICK_FS * OTHER_SCALE,
                legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
            )
            plt.tight_layout()
            out_dir = make_plot_dir(save_folder, scheme, "participation_rate")
            path = os.path.join(out_dir, f"participation_{case_label}.pdf")
            plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
            plt.close()
            print(f"[OK] Saved participation plot: {path}")

            # ---------------- Network Benefit ----------------
            plt.figure(figsize=other_figsize(8, 5))
            stats = summarise_band(df_case, "cycle", "network_benefit_mean")
            plt.fill_between(stats["cycle"], stats["q10"], stats["q90"], alpha=0.22)
            plt.plot(stats["cycle"], stats["mean"], linewidth=2, marker="o", label="mean")
            plt.axhline(
                y=FULL_SR_LINE_LEVEL,
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"Full SR ({FULL_SR_LINE_LEVEL:.0f})",
            )
            plt.xlabel("Cycle")
            plt.ylabel("Network Benefit (hrs)")
            ax = plt.gca()
            set_cycle_axis(ax, int(df_case["cycle"].max()), include_zero=True)
            if network_benefit_ylim is not None:
                plt.ylim(network_benefit_ymin, network_benefit_ylim)
            plt.legend(loc="best")
            style_figure(
                plt.gcf(),
                label_fs=BASE_LABEL_FS * OTHER_SCALE,
                tick_fs=BASE_TICK_FS * OTHER_SCALE,
                legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
            )
            plt.tight_layout()
            out_dir = make_plot_dir(save_folder, scheme, "network_benefit")
            path = os.path.join(out_dir, f"network_benefit_{case_label}.pdf")
            plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
            plt.close()
            print(f"[OK] Saved network benefit plot: {path}")

            # ---------------- Lost users (%) ----------------
            plt.figure(figsize=other_figsize(8, 5))
            lost = build_aligned_lost_rows(df_case)
            if not lost.empty:
                stats = summarise_band(lost, "plot_cycle", "plot_value")
                plt.fill_between(stats["plot_cycle"], stats["q10"], stats["q90"], alpha=0.22)
                plt.plot(stats["plot_cycle"], stats["mean"], linewidth=2, marker="o", label="mean")
            plt.xlabel("Cycle")
            plt.ylabel("Lost users (%)")
            ax = plt.gca()
            set_cycle_axis(ax, int(df_case["cycle"].max()), include_zero=True)
            plt.ylim(0, 100)
            plt.legend(loc="best")
            style_figure(
                plt.gcf(),
                label_fs=BASE_LABEL_FS * OTHER_SCALE,
                tick_fs=BASE_TICK_FS * OTHER_SCALE,
                legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
            )
            plt.tight_layout()
            out_dir = make_plot_dir(save_folder, scheme, "lost_users")
            path = os.path.join(out_dir, f"lost_users_pct_{case_label}.pdf")
            plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
            plt.close()
            print(f"[OK] Saved lost users % plot: {path}")


# ============================================================
# Condensed baseline plots
# ============================================================
def plot_baseline_dynamics_condensed(
    macro_df,
    save_folder,
    results_folder,
    baseline_case_label="baseline",
    network_benefit_ylim=None,
    network_benefit_ymin=0.0,
):
    print("[INFO] Plotting condensed baseline dynamics with aligned cycle states...")

    df = macro_df[macro_df["case_label"] == baseline_case_label].copy()
    if df.empty:
        print(f"[WARN] No rows found for baseline case_label={baseline_case_label!r}; skipping condensed baseline figure.")
        return

    init_df = load_initial_target_participation_from_pickles(results_folder)
    init_df = init_df[init_df["case_label"] == baseline_case_label].copy()

    out_dir = os.path.join(save_folder, "combined", "baseline_dynamics")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- Participation ----------------
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for scheme in sorted(df["scheme"].unique()):
        df_s = df[df["scheme"] == scheme].sort_values(["rep_id", "cycle"]).copy()
        init_s = init_df[init_df["scheme"] == scheme].copy()

        part = build_aligned_participation_rows(df_s, init_s)
        if part.empty:
            continue

        stats = summarise_band(part, "plot_cycle", "plot_value")
        line, = ax.plot(stats["plot_cycle"], stats["mean"], linewidth=2, marker="o", label="Collective Good" if scheme == "CollectiveGood" else scheme)
        ax.fill_between(stats["plot_cycle"], stats["q10"], stats["q90"], alpha=0.22, color=line.get_color())

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Participation rate (%)")
    set_cycle_axis(ax, int(df["cycle"].max()), include_zero=True)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    style_figure(
        fig,
        label_fs=BASE_LABEL_FS * COMBINED_SCALE,
        tick_fs=BASE_TICK_FS * COMBINED_SCALE,
        legend_fs=BASE_LEGEND_FS * COMBINED_SCALE,
    )
    fig.savefig(os.path.join(out_dir, "baseline_participation_condensed.pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Network benefit ----------------
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for scheme in sorted(df["scheme"].unique()):
        df_s = df[df["scheme"] == scheme].sort_values(["rep_id", "cycle"]).copy()
        stats = summarise_band(df_s, "cycle", "network_benefit_mean")
        line, = ax.plot(stats["cycle"], stats["mean"], linewidth=2, marker="o", label="Collective Good" if scheme == "CollectiveGood" else scheme)
        ax.fill_between(stats["cycle"], stats["q10"], stats["q90"], alpha=0.22, color=line.get_color())

    ax.axhline(
        y=FULL_SR_LINE_LEVEL,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Full SR ({FULL_SR_LINE_LEVEL:.0f})",
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Network benefit (hrs)")
    set_cycle_axis(ax, int(df["cycle"].max()), include_zero=True)
    if network_benefit_ylim is not None:
        ax.set_ylim(network_benefit_ymin, network_benefit_ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    style_figure(
        fig,
        label_fs=BASE_LABEL_FS * COMBINED_SCALE,
        tick_fs=BASE_TICK_FS * COMBINED_SCALE,
        legend_fs=BASE_LEGEND_FS * COMBINED_SCALE,
    )
    fig.savefig(os.path.join(out_dir, "baseline_network_benefit_condensed.pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Lost users ----------------
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    df_cg = df[df["scheme"] == "CollectiveGood"].copy()
    if not df_cg.empty:
        lost = build_aligned_lost_rows(df_cg)
        if not lost.empty:
            stats = summarise_band(lost, "plot_cycle", "plot_value")
            line, = ax.plot(stats["plot_cycle"], stats["mean"], linewidth=2, marker="o", label="Collective Good")
            ax.fill_between(stats["plot_cycle"], stats["q10"], stats["q90"], alpha=0.22, color=line.get_color())
            ax.legend(loc="lower right")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Lost users (%)")
    set_cycle_axis(ax, int(df["cycle"].max()), include_zero=True)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    style_figure(
        fig,
        label_fs=BASE_LABEL_FS * COMBINED_SCALE,
        tick_fs=BASE_TICK_FS * COMBINED_SCALE,
        legend_fs=BASE_LEGEND_FS * COMBINED_SCALE,
    )
    fig.savefig(os.path.join(out_dir, "baseline_lost_users_condensed.pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved aligned condensed baseline plots")
    gc.collect()


# ============================================================
# Grouped comparison plots
# ============================================================
CASE_GROUPS = {
    "init_part": ["baseline", "init_part_1", "init_part_60"],
    "moral_profile": ["baseline", "moral_best", "moral_worst"],
    "non_participant_estimate": ["baseline", "non_part_use_mean_percentage", "non_part_use_zero"],
    "profile_select": ["baseline", "profile_1", "profile_2"],
    "reconsideration_length": ["baseline", "recon_len_1", "recon_len_3"],
    "rho": ["baseline", "rho_0p0", "rho_0p1", "rho_0p8"],
    "social_days": ["baseline", "social_days_1", "social_days_4"],
    "collective_tolerance": ["baseline", "cg_tol_0p1", "cg_tol_0p2"],
}


def prettify_case_label(label):
    mapping = {
        "baseline": "Baseline",
        "init_part_1": "Init 1%",
        "init_part_60": "Init 60%",
        "moral_best": "Most Inclined",
        "moral_worst": "Least Inclined",
        "non_part_use_mean_percentage": "Use mean %",
        "non_part_use_zero": "Use zero",
        "profile_1": "Profile 1",
        "profile_2": "Profile 2",
        "recon_len_1": r"$H=1$",
        "recon_len_3": r"$H=3$",
        "rho_0p0": r"$\rho$=0.0",
        "rho_0p1": r"$\rho$=0.1",
        "rho_0p8": r"$\rho$=0.8",
        "social_days_1": "1 day",
        "social_days_4": "4 days",
        "cg_tol_0p1": r"$\delta^{CG}=0.1$",
        "cg_tol_0p2": r"$\delta^{CG}=0.2$",
    }
    return mapping.get(label, label)


def plot_grouped_case_comparisons(macro_df, save_folder, results_folder, network_benefit_ylim=None, network_benefit_ymin=0.0):
    print("[INFO] Plotting grouped case comparisons against baseline...")

    init_df = load_initial_target_participation_from_pickles(results_folder)

    base_out = os.path.join(save_folder, "comparisons")
    part_out = os.path.join(base_out, "participation_rate")
    nb_out = os.path.join(base_out, "network_benefit")
    os.makedirs(part_out, exist_ok=True)
    os.makedirs(nb_out, exist_ok=True)

    for scheme in sorted(macro_df["scheme"].unique()):
        df_scheme = macro_df[macro_df["scheme"] == scheme].copy()
        init_scheme = init_df[init_df["scheme"] == scheme].copy()

        scheme_part_out = os.path.join(part_out, scheme)
        scheme_nb_out = os.path.join(nb_out, scheme)
        os.makedirs(scheme_part_out, exist_ok=True)
        os.makedirs(scheme_nb_out, exist_ok=True)

        available_cases = set(df_scheme["case_label"].dropna().unique())

        for group_name, case_list in CASE_GROUPS.items():
            cases = [c for c in case_list if c in available_cases]
            if len(cases) < 2:
                continue

            # ---------- participation comparison ----------
            fig, ax = plt.subplots(figsize=other_figsize(8.5, 4.5))
            for case_label in cases:
                df_case = df_scheme[df_scheme["case_label"] == case_label].sort_values(["rep_id", "cycle"]).copy()
                init_case = init_scheme[init_scheme["case_label"] == case_label].copy()

                part = build_aligned_participation_rows(df_case, init_case)
                if part.empty:
                    continue

                stats = summarise_band(part, "plot_cycle", "plot_value")
                line, = ax.plot(
                    stats["plot_cycle"],
                    stats["mean"],
                    linewidth=2,
                    marker="o",
                    label=prettify_case_label(case_label),
                )
                ax.fill_between(
                    stats["plot_cycle"],
                    stats["q10"],
                    stats["q90"],
                    alpha=0.18,
                    color=line.get_color(),
                )

            ax.set_xlabel("Cycle")
            ax.set_ylabel("Participation rate (%)")
            set_cycle_axis(ax, int(df_scheme["cycle"].max()), include_zero=True)
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.3)
            legend_loc = "lower right" if scheme == "Sacrifice" else "best"
            ax.legend(loc=legend_loc)
            style_figure(
                plt.gcf(),
                label_fs=BASE_LABEL_FS * OTHER_SCALE,
                tick_fs=BASE_TICK_FS * OTHER_SCALE,
                legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(scheme_part_out, f"{group_name}.pdf"), dpi=200, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)

            # ---------- network-benefit comparison ----------
            fig, ax = plt.subplots(figsize=other_figsize(8.5, 4.5))
            for case_label in cases:
                df_case = df_scheme[df_scheme["case_label"] == case_label].sort_values(["rep_id", "cycle"]).copy()
                stats = summarise_band(df_case, "cycle", "network_benefit_mean")

                line, = ax.plot(
                    stats["cycle"],
                    stats["mean"],
                    linewidth=2,
                    marker="o",
                    label=prettify_case_label(case_label),
                )
                ax.fill_between(
                    stats["cycle"],
                    stats["q10"],
                    stats["q90"],
                    alpha=0.18,
                    color=line.get_color(),
                )

            ax.axhline(
                y=FULL_SR_LINE_LEVEL,
                color="red",
                linestyle=":",
                linewidth=2,
                label=f"Full SR ({FULL_SR_LINE_LEVEL:.0f})",
            )
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Network benefit (hrs)")
            set_cycle_axis(ax, int(df_scheme["cycle"].max()), include_zero=True)
            if network_benefit_ylim is not None:
                group_ymin = compute_casegroup_network_benefit_ymin(df_scheme[df_scheme["case_label"].isin(cases)])
                ax.set_ylim(group_ymin, network_benefit_ylim)
            ax.grid(alpha=0.3)
            legend_loc = "lower right" if scheme == "Sacrifice" else "best"
            ax.legend(loc=legend_loc)
            style_figure(
                plt.gcf(),
                label_fs=BASE_LABEL_FS * OTHER_SCALE,
                tick_fs=BASE_TICK_FS * OTHER_SCALE,
                legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(scheme_nb_out, f"{group_name}.pdf"), dpi=200, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)

    print("[OK] Saved grouped comparison plots")
    gc.collect()


# ============================================================
# Other cycle-level micro plots
# ============================================================
def plot_additional_time_cycles(df, save_folder):
    """
    Rep-consistent additional time stats:
      (rep_id, user_id, cycle) -> user-cycle mean fulldays_additional_tt
      then (rep_id, cycle) -> rep-cycle stats
      then average across reps for the plotted line(s)
    """
    print("[INFO] Plotting cycle-wise additional time statistics (rep-consistent)...")

    df_user_cycle = (
        df.groupby(["scheme", "group", "rep_id", "user_id", "cycle"], as_index=False)
        .agg(fulldays_additional_tt=("fulldays_additional_tt", "mean"))
    )

    # ---- NEW: compute shared y-limits across all additional-time panels ----
    all_stats = []
    for (scheme, group), g in df_user_cycle.groupby(["scheme", "group"]):
        rep_cycle_stats = (
            g.groupby(["rep_id", "cycle"])["fulldays_additional_tt"]
            .agg(
                mean="mean",
                median="median",
                p10=lambda x: x.quantile(0.10),
                p90=lambda x: x.quantile(0.90),
            )
            .reset_index()
        )
        stats = (
            rep_cycle_stats.groupby("cycle")
            .agg(
                mean=("mean", "mean"),
                median=("median", "mean"),
                p10=("p10", "mean"),
                p90=("p90", "mean"),
            )
            .reset_index()
            .sort_values("cycle")
        )
        all_stats.append(stats[["mean", "median", "p10", "p90"]].to_numpy().ravel())

    if all_stats:
        vals = np.concatenate(all_stats)
        vals = vals[np.isfinite(vals)]
        y_min = min(float(vals.min()), 0.0)
        y_max = max(float(vals.max()), 0.0)
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.5
        shared_ylim = (y_min - y_pad, y_max + y_pad)
    else:
        shared_ylim = None
    # ----------------------------------------------------------------------

    for (scheme, group), g in df_user_cycle.groupby(["scheme", "group"]):
        rep_cycle_stats = (
            g.groupby(["rep_id", "cycle"])["fulldays_additional_tt"]
            .agg(
                mean="mean",
                median="median",
                p10=lambda x: x.quantile(0.10),
                p90=lambda x: x.quantile(0.90),
                min_val="min",
                max_val="max",
            )
            .reset_index()
        )
        stats = (
            rep_cycle_stats.groupby("cycle")
            .agg(
                mean=("mean", "mean"),
                median=("median", "mean"),
                p10=("p10", "mean"),
                p90=("p90", "mean"),
                min_val=("min_val", "mean"),
                max_val=("max_val", "mean"),
            )
            .reset_index()
            .sort_values("cycle")
        )

        fig = plt.figure(figsize=other_figsize(8.5, 5.5))
        plt.plot(stats["cycle"], stats["mean"], label="Mean", marker="x")
        plt.plot(stats["cycle"], stats["median"], label="Median", marker="d")
        plt.plot(stats["cycle"], stats["p90"], linestyle="--", alpha=0.8, marker="s", label="90th percentile")
        plt.plot(stats["cycle"], stats["p10"], linestyle="--", alpha=0.8, marker="o", label="10th percentile")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.8)
        plt.xlabel("Cycle")
        plt.ylabel("Cycle Mean Additional Time (min)")
        ax = plt.gca()
        set_cycle_axis(ax, int(stats["cycle"].max()), include_zero=True)

        if shared_ylim is not None:
            ax.set_ylim(*shared_ylim)

        style_figure(
            fig,
            label_fs=BASE_LABEL_FS * OTHER_SCALE,
            tick_fs=BASE_TICK_FS * OTHER_SCALE,
            legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
        )
        fig.subplots_adjust(left=0.24, bottom=0.18, right=0.98, top=0.97)

        out_dir = make_plot_dir(save_folder, scheme, "additional_time_cycles")
        out = os.path.join(out_dir, f"additional_time_{group}.pdf")
        plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
        plt.close()
        print(f"[OK] Saved {out}")
        gc.collect()


def plot_prior_vs_experienced(df, save_folder):
    """
    Rep-consistent prior vs experienced:
      (rep_id, user_id, cycle) -> user-cycle means (experienced/additional) + UC prior
      (rep_id, cycle) -> rep-cycle mean of users
      then average across reps for final plotted lines
    """
    print("[INFO] Plotting mean user prior vs experienced (rep-consistent)...")

    # ---- NEW: compute shared y-limits across all prior/experienced panels ----
    all_stats = []
    for scheme in df["scheme"].unique():
        df_s = df[df["scheme"] == scheme]
        for group in ["participant", "nonparticipant"]:
            df_g = df_s[df_s["group"] == group].copy()
            df_user_cycle = (
                df_g.groupby(["rep_id", "user_id", "cycle"], as_index=False)
                .agg(
                    prior_tt=("prior_tt", "first"),
                    fulldays_additional_tt=("fulldays_additional_tt", "mean"),
                    experienced_tt=("experienced_tt", "mean"),
                )
            )
            rep_cycle = (
                df_user_cycle.groupby(["rep_id", "cycle"], as_index=False)
                .agg(
                    experienced_mean=("experienced_tt", "mean"),
                    prior_mean=("prior_tt", "mean"),
                )
                .sort_values(["rep_id", "cycle"])
            )
            stats = (
                rep_cycle.groupby("cycle", as_index=False)
                .agg(
                    experienced_mean=("experienced_mean", "mean"),
                    prior_mean=("prior_mean", "mean"),
                )
                .sort_values("cycle")
            )
            all_stats.append(stats[["prior_mean", "experienced_mean"]].to_numpy().ravel())

    if all_stats:
        vals = np.concatenate(all_stats)
        vals = vals[np.isfinite(vals)]
        y_min = float(vals.min())
        y_max = float(vals.max())
        y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.2
        shared_ylim = (y_min - y_pad, y_max + y_pad)
    else:
        shared_ylim = None
    # -------------------------------------------------------------------------

    for scheme in df["scheme"].unique():
        df_s = df[df["scheme"] == scheme]
        for group in ["participant", "nonparticipant"]:
            df_g = df_s[df_s["group"] == group].copy()
            df_user_cycle = (
                df_g.groupby(["rep_id", "user_id", "cycle"], as_index=False)
                .agg(
                    prior_tt=("prior_tt", "first"),
                    fulldays_additional_tt=("fulldays_additional_tt", "mean"),
                    experienced_tt=("experienced_tt", "mean"),
                )
            )
            if df_user_cycle["experienced_tt"].isna().any():
                raise RuntimeError(
                    "NaN experienced_tt detected — extractor should supply experienced_tt_fulldays explicitly."
                )
            rep_cycle = (
                df_user_cycle.groupby(["rep_id", "cycle"], as_index=False)
                .agg(
                    experienced_mean=("experienced_tt", "mean"),
                    prior_mean=("prior_tt", "mean"),
                )
                .sort_values(["rep_id", "cycle"])
            )
            stats = (
                rep_cycle.groupby("cycle", as_index=False)
                .agg(
                    experienced_mean=("experienced_mean", "mean"),
                    prior_mean=("prior_mean", "mean"),
                )
                .sort_values("cycle")
            )

            plt.figure(figsize=other_figsize(7, 4))
            plt.plot(
                stats["cycle"],
                stats["prior_mean"],
                linestyle="--",
                color="black",
                label="Mean user prior (UC)",
                marker="o",
            )
            plt.plot(
                stats["cycle"],
                stats["experienced_mean"],
                marker="o",
                label="Mean user experienced",
            )
            plt.xlabel("Cycle")
            plt.ylabel("Cycle Mean Travel time (min)")
            ax = plt.gca()
            set_cycle_axis(ax, int(stats["cycle"].max()), include_zero=True)

            # ---- NEW ----
            if shared_ylim is not None:
                ax.set_ylim(*shared_ylim)
            # -------------

            plt.legend(loc="best")
            style_figure(
                plt.gcf(),
                label_fs=BASE_LABEL_FS * OTHER_SCALE,
                tick_fs=BASE_TICK_FS * OTHER_SCALE,
                legend_fs=BASE_LEGEND_FS * OTHER_SCALE,
            )
            plt.tight_layout()
            out_dir = make_plot_dir(save_folder, scheme, "prior_vs_experienced")
            out = os.path.join(out_dir, f"prior_vs_experienced_{group}.pdf")
            plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
            plt.close()
            print(f"[OK] Saved {out}")
            gc.collect()
# ============================================================
# Data loading
# ============================================================
def load_data(results_folder, sim_hours):
    """Load unified CSVs generated by the extractor."""
    macro_path = os.path.join(results_folder, "extracted_output", f"macro_all_sim{sim_hours}.csv")
    if not os.path.exists(macro_path):
        raise FileNotFoundError(f"Macro CSV not found: {macro_path}")
    print(f"[INFO] Loading macro data from: {macro_path}")
    macro_df = pd.read_csv(macro_path)
    return macro_df


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    if os.path.isabs(args.save_folder):
        save_folder = args.save_folder
    else:
        save_folder = os.path.join(args.results_folder, args.save_folder)

    os.makedirs(save_folder, exist_ok=True)

    macro_df = load_data(args.results_folder, args.sim_hours)
    network_benefit_ylim = compute_global_network_benefit_ylim(macro_df)
    network_benefit_ymin = compute_global_network_benefit_ymin(macro_df)
    print(f"[INFO] Shared network benefit y-limits set to: [{network_benefit_ymin:.2f}, {network_benefit_ylim:.2f}]")

    # keep one-file-per-case outputs for reference, but with corrected alignment
    plot_macro_stats(
        macro_df,
        save_folder,
        results_folder=args.results_folder,
        network_benefit_ylim=network_benefit_ylim,
        network_benefit_ymin=network_benefit_ymin,
    )

    # condensed baseline plots for paper
    plot_baseline_dynamics_condensed(
        macro_df,
        save_folder,
        results_folder=args.results_folder,
        network_benefit_ylim=network_benefit_ylim,
    )

    # grouped comparisons against baseline
    plot_grouped_case_comparisons(
        macro_df,
        save_folder,
        results_folder=args.results_folder,
        network_benefit_ylim=network_benefit_ylim,
        network_benefit_ymin=network_benefit_ymin,
    )

    hist_csv = os.path.join(
        args.results_folder,
        "extracted_output",
        f"peruser_hist_all_sim{args.sim_hours}.csv",
    )
    plot_peruser_histograms(hist_csv, save_folder)

    peruser_raw_csv = os.path.join(
        args.results_folder,
        "extracted_output",
        f"peruser_all_sim{args.sim_hours}.csv",
    )
    print("[INFO] Loading per-user raw data once...")
    available_cols = pd.read_csv(peruser_raw_csv, nrows=0).columns.tolist()
    usecols = [
        "scheme",
        "user_id",
        "rep_id",
        "group",
        "cycle",
        "fulldays_additional_tt",
        "prior_tt",
        "experienced_tt",
    ]
    if "moral_index" in available_cols:
        usecols.append("moral_index")
    usecols.extend([c for c in MORAL_PROFILE_COLS if c in available_cols])

    BASELINE_CASE_LABEL = "baseline"
    peruser_df = pd.read_csv(peruser_raw_csv, usecols=usecols + ["case_label"])
    peruser_df = peruser_df[
        peruser_df["case_label"] == BASELINE_CASE_LABEL
    ].copy()

    print(
        peruser_df.groupby(["scheme", "case_label"]).size().sort_index()
    )

    print(
        f"[INFO] Using baseline case only: {peruser_df['case_label'].unique()}"
    )
    plot_additional_time_cycles(peruser_df, save_folder)
    plot_prior_vs_experienced(peruser_df, save_folder)
    plot_moral_index_last_cycles(peruser_df, save_folder, n_last_cycles=4)
    plot_moral_buoyancy_last_cycle(peruser_df, save_folder)

    # Appendix plots: single representative replication
    REP_TO_PLOT = peruser_df["rep_id"].value_counts().idxmax()
    peruser_df_rep = peruser_df[
        peruser_df["rep_id"] == REP_TO_PLOT
    ].copy()
    print(
        peruser_df_rep[
            (peruser_df_rep["scheme"] == "Sacrifice")
            & (peruser_df_rep["case_label"] == "baseline")
        ]
        .groupby(["cycle", "group"])["user_id"]
        .nunique()
        .unstack(fill_value=0)
        .head(10)
    )
    print(f"[INFO] Appendix plots use rep_id = {REP_TO_PLOT}")
    appendix_folder = os.path.join(save_folder, "appendix_replication")
    plot_additional_time_cycles(peruser_df_rep, appendix_folder)
    plot_prior_vs_experienced(peruser_df_rep, appendix_folder)
    plot_moral_index_last_cycles(peruser_df_rep, appendix_folder, n_last_cycles=4)
    plot_moral_buoyancy_last_cycle(peruser_df_rep, appendix_folder)

    print("\n[Done] All graphs saved in:", save_folder)


if __name__ == "__main__":
    main()
