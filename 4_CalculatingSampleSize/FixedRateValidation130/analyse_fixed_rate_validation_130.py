#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-process 130 successful-day fixed-participation validation scenarios.

Input: final pickles written by SocialRoutingSimulationFramework_fixed_rate_validation_130.py
for fixed_rate_20pct, fixed_rate_50pct, fixed_rate_80pct.

No SymuVia reruns or extraction from raw XML are needed: daily aggregate and
canonical user-aligned travel times have already been extracted to the pickles.

Outputs (in <root>/analysis):
* fixed_rate_validation_compact_summary.csv
* fixed_rate_validation_comparison.pdf
* fixed_rate_<rate>pct/{reference_15_vs_130_raw.csv,
    reference_15_vs_130_summary.csv, stability_15_to_16_raw.csv,
    stability_15_to_16_summary.csv, attempt_status_summary.csv,
    attempt_status_raw.csv, MultiSim_TotalTravelTime_130day_histogram.pdf,
    MultiSim_CumulativeMeans_abs.pdf, aggregate_15_vs_130_error.pdf,
    user_p95_15_vs_130_error.pdf, user_15_to_16_change.pdf}

Statistical interpretation: 130 successful daily realisations are an empirical
reference, not the exact stochastic expectation. Samples overlap this reference,
so estimates of absolute sampling error are slightly optimistic versus an
independent large validation pool. All daily assignments are independent
Bernoulli(p=0.4) for participating users; complete-week selection is NOT used.
"""

import argparse
import gc
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RATES = (20, 50, 80)
CI_T_CRITICAL_95_DF14 = 2.1447866879


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='ComputationalReq_fixed_rate_validation_130',
                   help='Folder with fixed_rate_20pct/50pct/80pct scenario outputs')
    p.add_argument('--output-dir', default=None,
                   help='Default: <root>/analysis')
    p.add_argument('--reference-days', type=int, default=130)
    p.add_argument('--sample-size', type=int, default=15)
    p.add_argument('--next-size', type=int, default=16)
    p.add_argument('--n-random-samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()


def get_record(path, rate):
    with path.open('rb') as f:
        outer = pickle.load(f)
    if not isinstance(outer, dict) or len(outer) != 1:
        raise ValueError(f'{path}: expected a dictionary with one experiment')
    key, rec = next(iter(outer.items()))
    if not isinstance(key, tuple) or len(key) != 3 or not isinstance(rec, dict):
        raise ValueError(f'{path}: unexpected record shape/key {key!r}')
    # Stored participation is round(N*rate/100)/N, not necessarily rate/100 exactly.
    if abs(float(key[1]) - rate / 100.0) > 1.0 / float(rec['number_of_users']):
        raise ValueError(f'{path}: key contains participation {key[1]}, expected ~{rate/100}')
    if rec.get('source_mode') != 'fixed_random_participation_optimised_probability_SR_days':
        raise ValueError(f'{path}: wrong source_mode; this analyser is for the NEW fixed-rate runs')
    return rec


def get_user_series(value, expected_users):
    if isinstance(value, pd.DataFrame):
        if 'driver_travel_time' not in value.columns:
            raise KeyError('User-day DataFrame lacks driver_travel_time')
        if 'user_index' not in value.columns:
            raise KeyError('User-day DataFrame lacks user_index (canonical alignment is required)')
        ids = value['user_index'].to_numpy()
        if len(ids) != expected_users or pd.Index(ids).has_duplicates:
            raise ValueError('User-day identifiers are not unique / expected length')
        if not np.array_equal(np.sort(ids), np.arange(expected_users)):
            raise ValueError('User-day identifiers do not cover 0..N-1')
        # Sort explicitly: never infer user alignment from DataFrame row order.
        return value.set_index('user_index')['driver_travel_time'].sort_index().to_numpy(dtype=np.float64)
    raise TypeError(f'Unexpected user-day output type: {type(value)}; expected DataFrame')


def load_day_arrays(rec, n_days):
    totals = rec['total_user_experienced_time_day']
    users = rec['user_experienced_time_day']
    n_users = int(rec['number_of_users'])
    if set(totals) != set(users):
        raise ValueError('Different day keys in aggregate and user-level outputs')
    if len(totals) != n_days:
        raise ValueError(f'Expected exactly {n_days} successful days, found {len(totals)}')
    keys = sorted(totals)
    total_s = np.asarray([totals[k] for k in keys], dtype=np.float64)
    matrix = np.empty((n_days, n_users), dtype=np.float64)
    for i, key in enumerate(keys):
        matrix[i] = get_user_series(users[key], n_users)
    if not np.isfinite(total_s).all() or not np.isfinite(matrix).all():
        raise ValueError('Travel time contains NaN or infinity')
    reconstructed_s = matrix.sum(axis=1)
    max_diff_s = float(np.max(np.abs(reconstructed_s - total_s)))
    if not np.allclose(reconstructed_s, total_s, rtol=1e-10, atol=1e-3):
        raise ValueError(f'User travel time totals disagree with network totals, max = {max_diff_s:.6f}s')
    return total_s / 3600.0, matrix, max_diff_s


def participation_mask(rec, n_users):
    raw = rec['select_social']
    if not isinstance(raw, dict) or set(raw) != set(range(n_users)):
        raise ValueError('Participant mask not aligned with 0..N-1 canonical user indices')
    return np.asarray([bool(raw[i]) for i in range(n_users)], dtype=bool)


def numeric_summary(df, ignored=()):
    rows = []
    for col in df.select_dtypes(include='number').columns:
        if col in ignored:
            continue
        x = df[col].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f'Non-finite statistic: {col}')
        rows.append(dict(metric=col, mean=float(np.mean(x)), median=float(np.median(x)),
                         p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)),
                         p99=float(np.percentile(x, 99)), max=float(np.max(x))))
    return pd.DataFrame(rows)


def analyze_samples(network_h, user_tt_s, mask, sample_size, next_size, n_draws, seed):
    n_days, n_users = user_tt_s.shape
    full_h = float(network_h.mean())
    full_user_s = user_tt_s.mean(axis=0)
    rng = np.random.default_rng(seed)
    reference_rows, stability_rows = [], []
    thresholds = (30.0, 60.0)

    for draw in range(1, n_draws + 1):
        order = rng.permutation(n_days)
        idx_a, idx_b = order[:sample_size], order[:next_size]
        mean_a_h = float(network_h[idx_a].mean())
        mean_b_h = float(network_h[idx_b].mean())
        mean_a_s = user_tt_s[idx_a, :].mean(axis=0)
        mean_b_s = user_tt_s[idx_b, :].mean(axis=0)
        err = np.abs(mean_a_s - full_user_s)
        delta = np.abs(mean_b_s - mean_a_s)
        halfwidth_h = CI_T_CRITICAL_95_DF14 * float(np.std(network_h[idx_a], ddof=1)) / np.sqrt(sample_size) \
            if sample_size == 15 else np.nan

        r = dict(sample_id=draw, sample_days=sample_size, reference_days=n_days,
                 sample_TNTT_h=mean_a_h, reference_TNTT_h=full_h,
                 aggregate_abs_pct_error=100.0 * abs(mean_a_h - full_h) / abs(full_h),
                 aggregate_95ci_relative_halfwidth_pct=100.0 * halfwidth_h / abs(mean_a_h),
                 mean_abs_user_error_s=float(err.mean()),
                 median_abs_user_error_s=float(np.median(err)),
                 p90_abs_user_error_s=float(np.percentile(err, 90)),
                 p95_abs_user_error_s=float(np.percentile(err, 95)),
                 p99_abs_user_error_s=float(np.percentile(err, 99)),
                 max_abs_user_error_s=float(err.max()))
        for tolerance in thresholds:
            r[f'share_users_within_{int(tolerance)}s'] = float(np.mean(err <= tolerance))
        for group_name, group_mask in [('participants', mask), ('nonparticipants', ~mask)]:
            e = err[group_mask]
            if not len(e):
                continue
            r[f'{group_name}_n'] = len(e)
            r[f'{group_name}_median_abs_error_s'] = float(np.median(e))
            r[f'{group_name}_p95_abs_error_s'] = float(np.percentile(e, 95))
            r[f'{group_name}_p99_abs_error_s'] = float(np.percentile(e, 99))
            for tolerance in thresholds:
                r[f'{group_name}_share_within_{int(tolerance)}s'] = float(np.mean(e <= tolerance))
        reference_rows.append(r)

        s = dict(sample_id=draw, j_base=sample_size, j_next=next_size,
                 aggregate_abs_pct_change=100.0 * abs(mean_b_h - mean_a_h) / abs(mean_a_h),
                 mean_abs_user_change_s=float(delta.mean()),
                 median_abs_user_change_s=float(np.median(delta)),
                 p95_abs_user_change_s=float(np.percentile(delta, 95)),
                 p99_abs_user_change_s=float(np.percentile(delta, 99)),
                 max_abs_user_change_s=float(delta.max()))
        stability_rows.append(s)
    return pd.DataFrame(reference_rows), pd.DataFrame(stability_rows)


def plot_hist(x, xlabel, out, title):
    """Histogram for additional sampling-error diagnostics (not a paper-original plot)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.hist(x, bins=22, edgecolor='black', linewidth=0.3)
    ax.set(xlabel=xlabel, ylabel='Count', title=title)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_paper_tntt_histogram(network_h, out):
    """Reproduce the plotting conventions of the original paper's TNTT histogram."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(network_h, bins=15, edgecolor='black', alpha=0.75)
    mean_tt = float(np.mean(network_h))
    ax.axvline(mean_tt, color='red', linestyle='dashed', linewidth=2,
               label=f'Mean = {mean_tt:.0f} h')
    if 15000 <= float(np.min(network_h)) and float(np.max(network_h)) <= 18500:
        ax.set_xticks([16000, 16500, 17000, 17500])
    ax.set_xlabel('Total network travel time [h]', fontsize=12)
    ax.set_ylabel('Number of daily simulations', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(out, format='pdf')
    plt.close(fig)


def plot_convergence(network_h, n_draws, seed, out, title=None):
    """Match the paper's median/MAXIMUM convergence plot (not p95)."""
    rng = np.random.default_rng(seed)
    n_days = len(network_h)
    changes = np.empty((n_draws, n_days - 1), dtype=float)
    for i in range(n_draws):
        perm = network_h[rng.permutation(n_days)]
        cumulative = np.cumsum(perm) / np.arange(1, n_days + 1)
        changes[i] = 100.0 * np.abs(np.diff(cumulative)) / np.abs(cumulative[:-1])
    median_changes = np.median(changes, axis=0)
    max_changes = np.max(changes, axis=0)
    days = np.arange(2, n_days + 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(days, median_changes, marker='.', linestyle='-',
            label='Median', markersize=6)
    ax.plot(days, max_changes, linestyle='dashdot',
            label='Maximum', markersize=5)
    ax.fill_between(days, max_changes, 0, color='gray', alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle='dotted',
               label='Threshold at 0.5%', linewidth=2)
    ax.set_xlim(1, n_days)
    ax.set_ylim(0, 5)
    ax.set_xlabel('No. of days included in cumulative mean', fontsize=12)
    ax.set_ylabel(r'$|\Delta\hat{TT}^{j}|$ [%]', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, format='pdf')
    plt.close(fig)
    return median_changes, max_changes


def attempt_status_tables(rec):
    raw = pd.DataFrame(rec.get('attempted_day_metadata', []))
    if raw.empty or 'status' not in raw.columns:
        raise ValueError('Attempt metadata absent: cannot assess failed-day retention')
    counts = raw['status'].fillna('missing').value_counts().rename_axis('status').reset_index(name='count')
    counts['share'] = counts['count'] / len(raw)
    return raw, counts


def lookup(summary_df, metric, stat):
    matched = summary_df.loc[summary_df.metric == metric, stat]
    if len(matched) != 1:
        raise KeyError(f'Could not find {metric} / {stat}')
    return float(matched.iloc[0])


def analyze_rate(rate, args, root, out_root):
    state = f'fixed_rate_{rate}pct'
    path = root / state / f'individual_users_output_parallel_multiSim_methodPaper_DTD_run_{rate}.pickle'
    print(f'[{rate}%] Loading {path}', flush=True)
    rec = get_record(path, rate)
    network_h, user_tt_s, max_diff_s = load_day_arrays(rec, args.reference_days)
    mask = participation_mask(rec, user_tt_s.shape[1])
    actual_percent = 100 * mask.mean()
    if abs(actual_percent - rate) > 100 / user_tt_s.shape[1]:
        raise ValueError(f'Participant share for {rate}% is {actual_percent:.3f}%')
    raw_attempt, status_counts = attempt_status_tables(rec)
    successes = int((raw_attempt.status == 'successful_extracted').sum())
    if successes != args.reference_days:
        raise ValueError(f'{state}: attempts report {successes} successes, expected {args.reference_days}')
    dest = out_root / state
    dest.mkdir(parents=True, exist_ok=True)
    ref_raw, stab_raw = analyze_samples(network_h, user_tt_s, mask,
                                        args.sample_size, args.next_size,
                                        args.n_random_samples, args.seed)
    ref_summary = numeric_summary(ref_raw, ('sample_id', 'sample_days', 'reference_days',
                                            'participants_n', 'nonparticipants_n'))
    stab_summary = numeric_summary(stab_raw, ('sample_id', 'j_base', 'j_next'))
    ref_raw.to_csv(dest / 'reference_15_vs_130_raw.csv', index=False)
    ref_summary.to_csv(dest / 'reference_15_vs_130_summary.csv', index=False)
    stab_raw.to_csv(dest / 'stability_15_to_16_raw.csv', index=False)
    stab_summary.to_csv(dest / 'stability_15_to_16_summary.csv', index=False)
    raw_attempt.to_csv(dest / 'attempt_status_raw.csv', index=False)
    status_counts.to_csv(dest / 'attempt_status_summary.csv', index=False)
    # Match the TWO original figures used in the paper, including file names.
    # Separate folders per participation rate prevent overwriting.
    plot_paper_tntt_histogram(network_h, dest / 'MultiSim_TotalTravelTime_130day_histogram.pdf')
    plot_convergence(network_h, args.n_random_samples, args.seed,
                     dest / 'MultiSim_CumulativeMeans_abs.pdf')
    plot_hist(ref_raw.aggregate_abs_pct_error,
              'Absolute TNTT error: 15-day sample vs 130-day reference [%]',
              dest / 'aggregate_15_vs_130_error.pdf', f'{rate}% fixed participation')
    plot_hist(ref_raw.p95_abs_user_error_s / 60,
              '95th percentile individual TT error vs 130-day reference [min]',
              dest / 'user_p95_15_vs_130_error.pdf', f'{rate}% fixed participation')
    plot_hist(stab_raw.p95_abs_user_change_s / 60,
              '95th percentile individual TT change: 15 to 16 days [min]',
              dest / 'user_15_to_16_change.pdf', f'{rate}% fixed participation')
    compact = {
        'state': state,
        'participation_rate_percent': actual_percent,
        'number_of_users': user_tt_s.shape[1],
        'successful_reference_days': len(network_h),
        'attempted_days': len(raw_attempt),
        'failed_or_excluded_days': len(raw_attempt) - successes,
        'success_share_percent': 100 * successes / len(raw_attempt),
        'reference_mean_TNTT_h': float(network_h.mean()),
        'max_aggregate_user_consistency_diff_s': max_diff_s,
        'median_15day_TNTT_abs_pct_error': lookup(ref_summary, 'aggregate_abs_pct_error', 'median'),
        'p95_15day_TNTT_abs_pct_error': lookup(ref_summary, 'aggregate_abs_pct_error', 'p95'),
        'median_15day_relative_95ci_halfwidth_pct': lookup(ref_summary, 'aggregate_95ci_relative_halfwidth_pct', 'median'),
        'p95_15day_relative_95ci_halfwidth_pct': lookup(ref_summary, 'aggregate_95ci_relative_halfwidth_pct', 'p95'),
        'median_of_median_user_abs_error_s': lookup(ref_summary, 'median_abs_user_error_s', 'median'),
        'median_of_p95_user_abs_error_s': lookup(ref_summary, 'p95_abs_user_error_s', 'median'),
        'p95_of_p95_user_abs_error_s': lookup(ref_summary, 'p95_abs_user_error_s', 'p95'),
        'median_participant_p95_user_abs_error_s': lookup(ref_summary, 'participants_p95_abs_error_s', 'median'),
        'median_nonparticipant_p95_user_abs_error_s': lookup(ref_summary, 'nonparticipants_p95_abs_error_s', 'median'),
        'median_15to16_TNTT_abs_pct_change': lookup(stab_summary, 'aggregate_abs_pct_change', 'median'),
        'p95_15to16_TNTT_abs_pct_change': lookup(stab_summary, 'aggregate_abs_pct_change', 'p95'),
        'median_15to16_p95_user_abs_change_s': lookup(stab_summary, 'p95_abs_user_change_s', 'median'),
    }
    print(f'[{rate}%] {len(network_h)} days; {len(raw_attempt)} attempts; '
          f'median TNTT error={compact["median_15day_TNTT_abs_pct_error"]:.3f}%; '
          f'median 95th-percentile user error={compact["median_of_p95_user_abs_error_s"]/60:.3f} min',
          flush=True)
    del rec, user_tt_s, ref_raw, stab_raw
    gc.collect()
    return compact


def comparison_figure(summary, outfile):
    rates = summary['participation_rate_percent'].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(rates, summary['median_15day_TNTT_abs_pct_error'], marker='o', label='Median 15-day TNTT error')
    ax.plot(rates, summary['p95_15day_TNTT_abs_pct_error'], marker='s', label='95th percentile 15-day TNTT error')
    ax.set(xlabel='Fixed participation rate [%]',
           ylabel='Absolute error vs 130-day TNTT mean [%]',
           title='Sampling accuracy by fixed participation rate')
    ax.set_xticks(rates)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def main():
    args = parse_args()
    if not (1 < args.sample_size < args.next_size <= args.reference_days):
        raise ValueError('Require 1 < sample-size < next-size <= reference-days')
    if args.sample_size != 15 or args.next_size != 16 or args.reference_days != 130:
        raise ValueError('This WP2 postprocessor is specifically configured for 15, 16, and 130 days')
    if args.n_random_samples <= 0:
        raise ValueError('Number of random samples must be positive')
    root = Path(args.root).resolve()
    out_root = Path(args.output_dir).resolve() if args.output_dir else root / 'analysis'
    # Fail BEFORE doing any work if one scenario is absent/incomplete.
    for rate in RATES:
        path = root / f'fixed_rate_{rate}pct' / f'individual_users_output_parallel_multiSim_methodPaper_DTD_run_{rate}.pickle'
        if not path.is_file():
            raise FileNotFoundError(f'Missing final results for {rate}%: {path}. Resubmit simulation array until complete.')
    out_root.mkdir(parents=True, exist_ok=True)
    compacts = [analyze_rate(rate, args, root, out_root) for rate in RATES]
    summary = pd.DataFrame(compacts)
    summary.to_csv(out_root / 'fixed_rate_validation_compact_summary.csv', index=False)
    # Paper-ready counterpart to the old script's 95% CI half-width summary.
    ci_columns = ['participation_rate_percent',
                  'median_15day_relative_95ci_halfwidth_pct',
                  'p95_15day_relative_95ci_halfwidth_pct']
    ci_summary = summary[ci_columns].copy()
    ci_summary.to_csv(out_root / 'aggregate_95ci_summary.csv', index=False)
    print('\nRelative half-width of estimated 95% CI for 15-day mean TNTT [%]:')
    print(ci_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    comparison_figure(summary, out_root / 'fixed_rate_validation_comparison.pdf')
    print('\n' + summary.to_string(index=False))
    print(f'\n[DONE] Post-processing outputs: {out_root}', flush=True)


if __name__ == '__main__':
    main()
