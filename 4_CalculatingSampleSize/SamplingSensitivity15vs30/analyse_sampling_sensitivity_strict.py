#!/usr/bin/env python3
"""Analyse completed 15/20-cycle paired 15/30-day trajectory validation.

Creates a per-cycle CSV, paired-difference CSV, and matched PNG/PDF trajectory figures.
Run in 5_Framework_Simulation after all 12 scenarios reach the selected cycle count.
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path('ComputationalReq_sampling_sensitivity_strict')
SCHEMES = ('Sacrifice', 'CollectiveGood')
REPS = range(3)
DAYS = (15, 30)

# The same Full SR reference as generate_graphs_from_extractor_optimised_v5.py.
FULL_SR_LINE_LEVEL = 9605.94387500002

# Match the paper's baseline scales while showing only simulated cycles.
PARTICIPATION_YLIM = (0, 100)
NETWORK_BENEFIT_YLIM = (-2000, 10200)



def load_scenario(scheme, rep, days, n_cycles):
    path = ROOT / f'scheme_{scheme}' / f'rep_{rep:02d}' / f'days_{days}' / 'output.pickle'
    if not path.exists():
        raise FileNotFoundError(f'Missing trajectory: {path}')
    with path.open('rb') as f:
        state = pickle.load(f)
    if len(state) != 9:
        raise ValueError(f'Unexpected checkpoint format: {path}')
    df_by_cycle, participation_rates = state[0], state[1]
    missing = sorted(set(range(1, n_cycles + 1)).difference(df_by_cycle))
    if missing:
        raise ValueError(f'Incomplete trajectory {path}; missing cycles: {missing}')
    rows = []
    for cycle in range(1, n_cycles + 1):
        frame = df_by_cycle[cycle]
        # Benefit is the NETWORK PERFORMANCE during cycle t.
        # The participation update at the END of cycle t is stored at index t.
        rows.append(dict(scheme=scheme, rep=rep, days=days, cycle=cycle,
                         participation_start_pct=100 * participation_rates[cycle-1],
                         participation_end_pct=100 * participation_rates[cycle],
                         network_benefit_hours=float(frame['Network_benefit_unscaled'].iloc[0])))
    return rows


def plot_trajectories(df, output):
    """Plot 15/30-day sensitivity with paper-style axes and SR benchmark.

    Participation is the state USED during cycle t (the rate at its start),
    matching build_aligned_participation_rows in the paper's plotting script.
    The CSVs preserve both start- and end-of-cycle rates unchanged.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    configurations = [
        ('participation_start_pct', 'participation_trajectories', 'Participation rate (%)'),
        ('network_benefit_hours', 'network_benefit_trajectories', 'Network benefit (hrs)'),
    ]

    max_cycle = int(df['cycle'].max())

    def format_axes(ax, metric):
        # Leave one cycle of visual breathing room after the final observed
        # 15th cycle. Do not show a tick or label for the unobserved cycle 16.
        # For a completed 20-cycle run, retain the paper's small right margin.
        right_limit = 16 if max_cycle == 15 else max_cycle + 0.4
        ax.set_xlim(0, right_limit)
        ax.set_xticks(np.arange(0, max_cycle + 1, 2))
        if metric == 'participation_start_pct':
            ax.set_ylim(*PARTICIPATION_YLIM)
        else:
            ax.set_ylim(*NETWORK_BENEFIT_YLIM)
            # Main-paper reference: Full SR, 9605.94 h per five-day week.
            ax.axhline(FULL_SR_LINE_LEVEL, color='red', linestyle=':',
                       linewidth=2, label=f'Full SR ({FULL_SR_LINE_LEVEL:.0f})')
        ax.set_xlabel('Cycle')
        ax.grid(alpha=0.3)

    def plot_one(ax, scheme, metric):
        sub = df.loc[df['scheme'] == scheme]
        for days in DAYS:
            values = sub.loc[sub['days'] == days].pivot(
                index='cycle', columns='rep', values=metric).sort_index()
            x = values.index.to_numpy(dtype=float)
            center = values.mean(axis=1).to_numpy(dtype=float)
            low = values.min(axis=1).to_numpy(dtype=float)
            high = values.max(axis=1).to_numpy(dtype=float)
            (line,) = ax.plot(x, center, linewidth=2, marker='o',
                              markersize=4, label=f'{days} days')
            # Min/max across the three replications, NOT a confidence interval.
            ax.fill_between(x, low, high, color=line.get_color(), alpha=0.18)
        format_axes(ax, metric)

    for metric, base_filename, ylabel in configurations:
        # Existing two-panel comparison, with matching baseline axes.
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7), sharex=True, sharey=True)
        for ax, scheme in zip(axes, SCHEMES):
            plot_one(ax, scheme, metric)
            ax.set_title('Collective Good' if scheme == 'CollectiveGood' else scheme)
            ax.set_ylabel(ylabel)
            ax.legend(loc='lower right' if metric == 'participation_start_pct' else 'lower left',
                      fontsize=10)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{base_filename}.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Also export single-scheme figures in the format of the main baseline plots.
        for scheme in SCHEMES:
            fig, ax = plt.subplots(figsize=(9.6, 6.0))
            plot_one(ax, scheme, metric)
            ax.set_ylabel(ylabel)
            ax.legend(loc='lower right' if metric == 'participation_start_pct' else 'lower left',
                      fontsize=12)
            ax.tick_params(labelsize=12)
            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            fig.tight_layout()
            for ext in ('png', 'pdf'):
                fig.savefig(output / f'{base_filename}_{scheme}.{ext}',
                            dpi=200, bbox_inches='tight')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare strict 15 vs 30 daily samples')
    parser.add_argument('--cycles', type=int, choices=[15, 20], default=15,
                        help='Number of completed behavioural cycles to analyse')
    parser.add_argument('--plots-only', action='store_true',
                        help='Regenerate figures from the existing trajectory_by_cycle.csv '
                             'without loading the large simulation checkpoints')
    args = parser.parse_args()
    n_cycles = args.cycles
    output = ROOT / f'analysis_{n_cycles}cycles'

    if args.plots_only:
        csv_path = output / 'trajectory_by_cycle.csv'
        if not csv_path.is_file():
            raise FileNotFoundError(f'Run the full analysis first; missing: {csv_path}')
        df = pd.read_csv(csv_path)
    else:
        rows = []
        for scheme in SCHEMES:
            for rep in REPS:
                for days in DAYS:
                    rows.extend(load_scenario(scheme, rep, days, n_cycles))
        df = pd.DataFrame(rows)
        output.mkdir(parents=True, exist_ok=True)
        df.to_csv(output / 'trajectory_by_cycle.csv', index=False)

        p = df.pivot(index=['scheme', 'rep', 'cycle'], columns='days',
                     values=['participation_end_pct', 'network_benefit_hours'])
        p.columns = [f'{name}_{days}' for name, days in p.columns]
        p = p.reset_index()
        for metric in ('participation_end_pct', 'network_benefit_hours'):
            p[f'delta_30_minus_15_{metric}'] = p[f'{metric}_30'] - p[f'{metric}_15']
        p.to_csv(output / 'paired_differences_by_cycle.csv', index=False)

        summary = []
        for (scheme, rep), g in p.groupby(['scheme', 'rep']):
            end = g.loc[g['cycle'] == n_cycles].iloc[0]
            summary.append({
                'scheme': scheme, 'rep': rep,
                'end_cycle_participation_diff_pp': end['delta_30_minus_15_participation_end_pct'],
                'mean_abs_participation_diff_pp': g['delta_30_minus_15_participation_end_pct'].abs().mean(),
                'end_cycle_network_benefit_diff_hours': end['delta_30_minus_15_network_benefit_hours'],
                'mean_abs_network_benefit_diff_hours': g['delta_30_minus_15_network_benefit_hours'].abs().mean(),
            })
        pd.DataFrame(summary).to_csv(output / 'paired_summary.csv', index=False)

    try:
        plot_trajectories(df, output)
    except ImportError:
        print('Matplotlib unavailable; figures skipped')
    print(f'Analysis saved to: {output.resolve()}')


if __name__ == '__main__':
    main()
