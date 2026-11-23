#!/usr/bin/env python
"""
Teacher-only data collection: the teacher plays each level (no student, beta = 1) and every visited decision
is labelled and stored as a replay log in the same format as smw_imitation.py --save-data, so the readout lab,
the CV lab and smw_scaling.py can use it.

    .venv/bin/python scripts/smw_collect.py --levels L2A,L48,... --runs 2 --out runs/smw/teacher_more --workers 4

Run r uses a start delay of 0 idle steps for r = 0 and a random 1..--delay-max otherwise (other enemy phases).
Files are named iter{r}_{level}.npz; episodes already on disk are skipped, so the script can be resumed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from smw_imitation import job, play


def play_run(j: dict) -> dict:
    ep = play(j)
    ep['run'] = j['run']
    return ep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--levels', required=True, help='comma-separated level specs')
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--delay-max', type=int, default=40)
    p.add_argument('--teacher', default='v2')
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--config-from', default='runs/smw/abc_v2/results.json',
                   help='observation / grid config to record with (must match the data it is combined with)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out)
    (out / 'data').mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config_from).read_text())['config']
    if not (out / 'results.json').exists():  # a resumed run keeps the original arguments
        (out / 'results.json').write_text(json.dumps({'config': config, 'args': vars(args)}, indent=1))
    rng = np.random.default_rng(0)
    jobs = []
    for r in range(args.runs):
        for i, lv in enumerate(args.levels.split(',')):
            noop = 0 if r == 0 else int(rng.integers(1, args.delay_max + 1))
            if (out / 'data' / f'iter{r}_{lv}.npz').exists():
                continue
            jobs.append(job(lv, config, 'input', None, beta=1.0, label=True, plan_every=4, q_clip=32.0,
                            seed=1000 * r + i, max_steps=args.max_steps, record=True, teacher=args.teacher,
                            noop=noop, run=r))
    print(f'{len(jobs)} teacher episodes to play', flush=True)
    t0 = time.time()
    summ_path = out / 'collect_summary.json'
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else []  # keep rows of a resumed run
    with mp.get_context('fork').Pool(args.workers, maxtasksperchild=1) as pool:
        for ep in pool.imap_unordered(play_run, jobs):
            rec = ep.pop('record')
            r = ep['run']
            np.savez_compressed(out / 'data' / f"iter{r}_{ep['level']}.npz", **rec)
            row = {'level': ep['level'], 'progress': round(ep['progress'], 3), 'goal': bool(ep.get('reached_goal'))}
            row.update(run=r, moments=len(ep['labels']), minutes=round((time.time() - t0) / 60, 1))
            summary.append(row)
            print(row, flush=True)
            summ_path.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
