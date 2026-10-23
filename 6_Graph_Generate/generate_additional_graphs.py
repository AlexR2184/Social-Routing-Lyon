#!/usr/bin/env python3
"""Additional Social Routing revision plots, using the standard extractor outputs.

Route-day analysis: UC/SR-day observations in the Sacrifice/CollectiveGood baselines.
Promise-credibility analysis: CollectiveGood original realised promise vs AR(1)-perceived promise.

No changes to the existing 8-object output.pickle or the v5 figure generator.
The user-level route-day histogram is optional and uses per-run compressed sidecars
NOT included in the extractor's default ZIP.
"""
from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FULL_SR_LINE_LEVEL = 9606

CG_LABELS = {'baseline': 'CG: realised promise', 'cg_promise_ar1': 'CG: AR(1) perception'}
SCHEMES = {'Sacrifice': 'Sacrifice', 'CollectiveGood': 'Collective Good'}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-folder', required=True, type=Path,
                   help='Simulations_JOB_ID directory with extracted_output and optional raw run folders')
    p.add_argument('--sim-hours', type=int, default=4, choices=[2, 4])
    p.add_argument('--save-folder', default='graphs',
                   help='Directory for graphs; defaults to results-folder/graphs')
    p.add_argument('--last-cycles', type=int, default=5,
                   help='Final cycles used for the route-day and promise-credibility comparisons')
    p.add_argument('--expected-cycles', type=int, default=20,
                   help='Require complete 20-cycle trajectories (use 2 for a smoke test)')
    p.add_argument('--expected-reps', type=int, default=3,
                   help='Require 3 replications for each comparison case')
    p.add_argument('--no-user-histogram', action='store_true',
                   help='Skip reading per-run user-level route-day files')
    return p.parse_args()


def validate_columns(df, needed, source):
    missing = sorted(set(needed) - set(df.columns))
    if missing:
        raise ValueError(f'{source}: missing columns {missing}')


def validate_cycles(df, label, expected_cycles, expected_reps):
    """Do not silently draw only completed subsets of the submitted experiment."""
    reps = sorted(df['rep_id'].dropna().unique().tolist())
    if len(reps) != expected_reps:
        raise ValueError(f'{label}: expected {expected_reps} reps, found {reps}')
    for rep, d in df.groupby('rep_id'):
        cycles = sorted(d['cycle'].dropna().unique().astype(int).tolist())
        if cycles != list(range(1, expected_cycles + 1)):
            raise ValueError(f'{label}, rep={rep}: expected cycles 1-{expected_cycles}, found {cycles}')


def make_folder(folder):
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_pdf(fig, path):
    fig.tight_layout()
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {path}', flush=True)


def style(ax):
    ax.grid(alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

def add_reference_line(ax, y=0.0, label=None):
    kwargs = dict(color='red', linestyle='--', linewidth=1.6, alpha=0.85)
    if label is not None:
        kwargs['label'] = label
    ax.axhline(y, **kwargs)


def route_day_summary(route_df, folder, last_cycles, expected_cycles, expected_reps):
    required = {
        'scheme', 'case_label', 'rep_id', 'cycle', 'group_start',
        'mean_ucday_additional_tt_min', 'mean_srday_additional_tt_min',
        'mean_allday_additional_tt_min', 'mean_paired_sr_minus_uc_min',
        'n_paired_users', 'share_paired_sr_slower',
        'n_valid_ucday_users', 'n_valid_srday_users', 'n_users'
    }
    validate_columns(route_df, required, 'route_day_cycle_by_rep')
    out = make_folder(folder / 'additional' / 'route_days')
    base = route_df[(route_df['case_label'] == 'baseline') &
                    (route_df['scheme'].isin(SCHEMES))].copy()
    for scheme in SCHEMES:
        s = base[base['scheme'] == scheme]
        if s.empty:
            raise ValueError(f'Route-day analysis needs the baseline route-day outputs for {scheme}')
        validate_cycles(s, f'Route-day {scheme} baseline', expected_cycles, expected_reps)
        for rep, group in s.groupby('rep_id'):
            n = group.groupby('cycle')['group_start'].agg(['count', 'nunique'])
            if not ((n['count'] == 2) & (n['nunique'] == 2)).all():
                raise ValueError(f'Route-day {scheme} rep={rep}: missing/duplicated group summaries')

    first = expected_cycles - min(last_cycles, expected_cycles) + 1
    recent = base[base['cycle'] >= first].copy()
    fields = {
        'mean_ucday_additional_tt_min': 'uc_day_add_min',
        'mean_srday_additional_tt_min': 'sr_day_add_min',
        'mean_allday_additional_tt_min': 'all_day_add_min',
        'mean_paired_sr_minus_uc_min': 'paired_sr_minus_uc_min',
        'share_paired_sr_slower': 'share_paired_sr_slower',
        'n_paired_users': 'n_paired_users',
        'n_valid_ucday_users': 'n_valid_ucday_users',
        'n_valid_srday_users': 'n_valid_srday_users',
        'n_users': 'n_users'
    }
    # First average over final cycles WITHIN each replication, THEN across reps.
    by_rep = recent.groupby(['scheme', 'group_start', 'rep_id'], as_index=False)[list(fields)].mean()
    by_rep = by_rep.rename(columns=fields)
    by_rep.to_csv(out / 'route_day_last_cycles_by_rep.csv', index=False)
    summary = by_rep.groupby(['scheme','group_start'], as_index=False)[list(fields.values())].mean()
    summary.to_csv(out / 'route_day_last_cycles_means.csv', index=False)

    participant = by_rep[by_rep['group_start'] == 'participant'].copy()
    x = np.arange(len(SCHEMES))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for j, (col, label) in enumerate([('uc_day_add_min','UC-route days'),
                                       ('sr_day_add_min','SR-route days')]):
        g = participant.groupby('scheme')[col].agg(['mean','min','max']).reindex(SCHEMES)
        values = g['mean'].to_numpy(dtype=float)
        errors = np.vstack([values - g['min'].to_numpy(dtype=float),
                            g['max'].to_numpy(dtype=float) - values])
        ax.bar(x + (j-.5)*.32, values, .30, label=label)
        ax.errorbar(x + (j-.5)*.32, values, yerr=errors, fmt='none',
                    ecolor='black', capsize=3, linewidth=1)
    add_reference_line(ax, 0.0)
    ax.set_xticks(x, [SCHEMES[s] for s in SCHEMES])
    ax.set_ylabel('Additional travel time relative to UC prior (min)')
    # ax.set_title(f'Participants on observed UC and SR days (cycles {first}-{expected_cycles})')
    ax.legend(frameon=False)
    style(ax)
    save_pdf(fig, out / 'route_day_participant_uc_sr_day_means.pdf')

    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    stat = participant.groupby('scheme')['paired_sr_minus_uc_min'].agg(['mean','min','max']).reindex(SCHEMES)
    for i, scheme in enumerate(SCHEMES):
        vals = participant[participant['scheme'] == scheme]['paired_sr_minus_uc_min'].to_numpy(float)
        ax.scatter(np.full_like(vals, i, dtype=float), vals, s=30, alpha=.6)
        ax.plot([i-.12,i+.12],[stat.loc[scheme,'mean']]*2, c='black', lw=2.0)
    add_reference_line(ax, 0.0)
    ax.set_xticks(x, [SCHEMES[s] for s in SCHEMES])
    ax.set_ylabel('Mean within-cycle SR minus UC travel time (min)')
    # ax.set_title(f'Paired participants with both route types (cycles {first}-{expected_cycles})')
    style(ax)
    save_pdf(fig, out / 'route_day_paired_sr_minus_uc_by_rep.pdf')

    fig, ax = plt.subplots(figsize=(8,4.5))
    for j, group in enumerate(['participant','nonparticipant']):
        gg = by_rep[by_rep['group_start'] == group].groupby('scheme')['all_day_add_min'].mean().reindex(SCHEMES)
        ax.bar(x + (j-.5)*.32, gg.values, .30, label=group.title())
    add_reference_line(ax, 0.0)
    ax.set_xticks(x, [SCHEMES[s] for s in SCHEMES])
    ax.set_ylabel('Additional travel time, all days (min)')
    # ax.set_title(f'Experienced additional travel time by cycle-start status (cycles {first}-{expected_cycles})')
    ax.legend(frameon=False)
    style(ax)
    save_pdf(fig, out / 'route_day_participant_nonparticipant_all_days.pdf')
    print('[Route days] Route-specific means condition on observing that route type. '
          'Paired contrasts are descriptive, not causal.', flush=True)
    return out


def optional_user_histograms(results_folder, folder, last_cycles, expected_cycles, expected_reps):
    """Use gzipped files for cycle-specific matched UC/SR user observations only."""
    out = make_folder(folder / 'additional' / 'route_days')
    first = expected_cycles - min(last_cycles, expected_cycles) + 1
    files_found = {}
    for scheme in SCHEMES:
        scheme_paths = []
        # Case 00 is baseline in BOTH schemes, rep 00..02.
        for rep in range(expected_reps):
            rd = results_folder / f'scheme_{scheme}' / f'case_00_rep_{rep:02d}' / 'route_day_outputs'
            paths = [rd / f'cycle_{cycle:02d}_users.csv.gz' for cycle in range(first, expected_cycles+1)]
            if not all(p.is_file() for p in paths):
                print(f'[WARN] WP3 user histogram skipped: unavailable {next((str(p) for p in paths if not p.is_file()), rd)}. '
                      'The extractor ZIP does not normally include user-level sidecars.', flush=True)
                return
            scheme_paths.extend((rep, p) for p in paths)
        files_found[scheme] = scheme_paths

    rep_data = []
    raw_user_metrics = []
    for scheme, paths in files_found.items():
        for rep, path in paths:
            d = pd.read_csv(path, usecols=['user_id','group_start','ucday_tt_min','srday_tt_min',
                                           'n_ucdays','n_srdays'])
            d = d[(d['group_start'] == 'participant') & (d['n_ucdays'] > 0) &
                  (d['n_srdays'] > 0)].copy()
            d['paired_delta_min'] = d['srday_tt_min'] - d['ucday_tt_min']
            d = d[np.isfinite(d['paired_delta_min'])]
            d = d[['user_id','paired_delta_min']]
            d['rep_id'] = rep
            rep_data.append((scheme, d))
    for scheme in SCHEMES:
        d = pd.concat([d for s,d in rep_data if s == scheme], ignore_index=True)
        if d.empty:
            warnings.warn(f'Route-day {scheme}: no matched route-day users found')
            continue
        # A user may appear in multiple eligible cycles; average within rep/user
        # before plotting distributions. Replications remain distinct observations.
        d = d.groupby(['rep_id','user_id'],as_index=False)['paired_delta_min'].mean()
        d['scheme'] = scheme
        raw_user_metrics.append(d)
    if not raw_user_metrics:
        return
    data = pd.concat(raw_user_metrics, ignore_index=True)
    values = data['paired_delta_min'].to_numpy(dtype=float)
    lo, hi = np.quantile(values, [.01,.99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        center = float(np.median(values))
        lo, hi = center - 0.5, center + 0.5
    bins = np.linspace(lo, hi, 55)
    fig, ax = plt.subplots(figsize=(8.5,4.7))
    for scheme in SCHEMES:
        series = data.loc[data['scheme']==scheme,'paired_delta_min'].to_numpy(dtype=float)
        in_range = series[(series >= lo) & (series <= hi)]
        if in_range.size:
            ax.hist(in_range, bins=bins, density=True, histtype='step', linewidth=1.8,
                    label=SCHEMES[scheme])
    ax.axvline(0, color='red', linestyle='--', linewidth=1.6, alpha=0.85)
    ax.set_xlabel('Mean within-user-cycle SR minus UC travel time (min)')
    ax.set_ylabel('Density')
    # ax.set_title(f'Participant within-cycle contrasts (cycles {first}-{expected_cycles})')
    ax.legend(frameon=False)
    style(ax)
    save_pdf(fig, out / 'route_day_user_paired_difference_distribution.pdf')
    statistics = data.groupby(['scheme','rep_id'])['paired_delta_min'].agg(
        n_users='size', mean='mean', median='median',
        q10=lambda a: a.quantile(.1), q90=lambda a: a.quantile(.9),
        share_sr_slower=lambda a: (a>0).mean()
    ).reset_index()
    statistics.to_csv(out / 'route_day_user_paired_difference_stats_by_rep.csv', index=False)
    print('[Route days] User histogram: each rep-user is averaged over qualifying cycles; '
          'the same user ID may recur across replications.',flush=True)


def promise_credibility_summary(macro_df, folder, last_cycles, expected_cycles, expected_reps):
    out = make_folder(folder / 'additional' / 'promise_credibility')
    required = {'scheme','case_label','rep_id','cycle','participation_rate_start',
                'participation_rate_end','lost_users_end','number_of_users','network_benefit_mean'}
    validate_columns(macro_df, required, 'macro_all')
    df = macro_df[(macro_df['scheme'] == 'CollectiveGood') &
                  macro_df['case_label'].isin(CG_LABELS)].copy()
    for case in CG_LABELS:
        cdf = df[df['case_label']==case]
        if cdf.empty:
            raise ValueError(f'Promise credibility requires CG case_label={case!r}; verify tasks 55-57 and 115-117 completed')
        validate_cycles(cdf, f'Promise credibility {case}', expected_cycles, expected_reps)
        if cdf.duplicated(['case_label','rep_id','cycle']).any():
            raise ValueError(f'Promise credibility: duplicated rows for {case}')
    df['participation_start_pct'] = 100.0 * df['participation_rate_start']
    df['participation_end_pct'] = 100.0 * df['participation_rate_end']
    df['lost_end_pct'] = 100.0 * df['lost_users_end'] / df['number_of_users']

    def curve(col, ylabel, filename, percent=False):
        fig, ax = plt.subplots(figsize=(8.0,4.7))
        for case, display in CG_LABELS.items():
            subset = df[df['case_label']==case]
            stats = subset.groupby('cycle')[col].agg(['mean','min','max']).reset_index()
            xx = stats['cycle'].to_numpy(dtype=float)
            avg = stats['mean'].to_numpy(dtype=float)
            bottom = stats['min'].to_numpy(dtype=float)
            top = stats['max'].to_numpy(dtype=float)
            (line,) = ax.plot(xx,avg,marker='o',markersize=3,lw=1.9,label=display)
            ax.fill_between(xx,bottom,top,color=line.get_color(),alpha=.15)
        ax.set_xlabel('Cycle')
        ax.set_ylabel(ylabel)
        ax.set_xlim(1,expected_cycles+.2)
        ax.set_xticks(np.arange(2, expected_cycles + 1, 2))
        if 'network_benefit' in filename:
            add_reference_line(ax, FULL_SR_LINE_LEVEL, label=f'Full SR ({FULL_SR_LINE_LEVEL:.0f})')
        if percent:
            ax.set_ylim(0,100)
        ax.legend(frameon=False)
        style(ax)
        save_pdf(fig, out / filename)

    # Align participating populations with the traffic generated in each cycle.
    # For cumulative loss, end-of-cycle is intentional so cycle 20 is included.
    curve('participation_start_pct', 'Participation rate (%)',
          'promise_participation_start_of_cycle.pdf',percent=True)
    curve('lost_users_end','Cumulative permanently excluded users',
          'promise_permanent_withdrawals_end_of_cycle.pdf')
    curve('network_benefit_mean','Network benefit (hrs)',
          'promise_weekly_network_benefit.pdf')

    first = expected_cycles - min(last_cycles,expected_cycles) + 1
    late = df[df['cycle']>=first]
    nums = ['participation_start_pct','participation_end_pct','lost_users_end',
            'lost_end_pct','network_benefit_mean']
    rep = late.groupby(['case_label','rep_id'],as_index=False)[nums].mean()
    rep.to_csv(out / 'promise_last_cycles_by_rep.csv',index=False)
    mean = rep.groupby('case_label',as_index=False)[nums].mean()
    mean.to_csv(out / 'promise_last_cycles_means.csv',index=False)
    paired = rep.pivot(index='rep_id',columns='case_label',values=nums)
    comparison = pd.DataFrame({'rep_id': paired.index})
    for n in nums:
        comparison[n + '_difference_ar1_minus_realised'] = (
            paired[(n,'cg_promise_ar1')] - paired[(n,'baseline')]).to_numpy(dtype=float)
    comparison.to_csv(out / 'promise_paired_replication_differences.csv',index=False)

    # True terminal state: use cycle-20 END-OF-CYCLE withdrawals, not the last-five mean.
    final = df[df['cycle'] == expected_cycles][['case_label','rep_id','lost_users_end',
                                                'participation_rate_end','number_of_users']]
    final.to_csv(out / 'promise_final_cycle_end_states.csv', index=False)
    print(f'[Promise credibility] Results: {out}. Shaded bands are min–max among {expected_reps} runs, not confidence intervals.',
          flush=True)
    return out


def main():
    args = arguments()
    if args.expected_cycles <= 0 or args.last_cycles <= 0 or args.expected_reps <= 0:
        raise ValueError('expected-cycles, expected-reps, last-cycles must be positive')
    root = args.results_folder.expanduser().resolve()
    save = Path(args.save_folder).expanduser()
    if not save.is_absolute():
        save = root / save
    extracted = root / 'extracted_output'
    route_path = extracted / f'route_day_cycle_by_rep_sim{args.sim_hours}.csv'
    macro_path = extracted / f'macro_all_sim{args.sim_hours}.csv'
    for p in (route_path, macro_path):
        if not p.is_file():
            raise FileNotFoundError(f'Expected extractor output missing: {p}')
    print(f'[INFO] Loading {route_path}',flush=True)
    route_day_summary(pd.read_csv(route_path),save,args.last_cycles,args.expected_cycles,args.expected_reps)
    if not args.no_user_histogram:
        optional_user_histograms(root,save,args.last_cycles,args.expected_cycles,args.expected_reps)
    print(f'[INFO] Loading {macro_path}',flush=True)
    promise_credibility_summary(pd.read_csv(macro_path),save,args.last_cycles,args.expected_cycles,args.expected_reps)
    print(f'[DONE] Additional figures and numerical tables: {save}',flush=True)


if __name__ == '__main__':
    main()
