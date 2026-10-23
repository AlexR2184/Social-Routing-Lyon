#!/usr/bin/env python3
"""Single entry point for original and additional Social Routing figures.

Runs the unchanged legacy v5 graph producer and the additional route-day and promise-credibility plots
in separate Python processes (to release RAM between the two stages).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-folder', type=Path, required=True,
                   help='Simulations_JOB_ID folder containing extracted_output and raw run folders')
    p.add_argument('--sim-hours', type=int, default=4, choices=(2,4))
    p.add_argument('--save-folder', default='graphs',
                   help='Output relative to results-folder, or absolute path')
    p.add_argument('--mode', default='all', choices=('all','legacy','additional'),
                   help='Generate both, only original figures, or only additional figures')
    p.add_argument('--last-cycles', type=int, default=5)
    p.add_argument('--expected-cycles', type=int, default=20)
    p.add_argument('--expected-reps', type=int, default=3)
    p.add_argument('--no-user-histogram', action='store_true',
                   help='Skip optional per-user route-day histogram')
    args = p.parse_args()

    root = args.results_folder.expanduser().resolve()
    if not root.is_dir():
        p.error(f'Results folder does not exist: {root}')
    output = Path(args.save_folder).expanduser()
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True,exist_ok=True)

    here = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment['MPLBACKEND']='Agg'
    shared = ['--results-folder',str(root),'--sim-hours',str(args.sim_hours),
              '--save-folder',str(output)]
    stages = []
    if args.mode in ('all','legacy'):
        stages.append(('Original v5 paper plots',here/'generate_graphs_from_extractor_optimised_v5.py',shared))
    if args.mode in ('all','additional'):
        extra = shared + ['--last-cycles',str(args.last_cycles),
                          '--expected-cycles',str(args.expected_cycles),
                          '--expected-reps',str(args.expected_reps)]
        if args.no_user_histogram:
            extra.append('--no-user-histogram')
        stages.append(('Additional figures',here/'generate_additional_graphs.py',extra))

    for title,script,opts in stages:
        if not script.is_file():
            raise FileNotFoundError(f'Missing plot generator next to master: {script}')
        cmd = [sys.executable, str(script), *opts]
        print(f'\n=== {title} ===\n$ {" ".join(cmd)}',flush=True)
        subprocess.run(cmd, check=True, env=environment, cwd=str(here))
    print(f'\n[COMPLETE] All requested graphs are in {output}',flush=True)

if __name__=='__main__':
    main()
