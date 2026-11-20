#!/usr/bin/env python
"""
Records the look-ahead planner on training levels for offline readout experiments.

    python scripts/record_teacher.py --config configs/imitation/full.yaml --levels 40 --workers 3 \
        --out runs/teacher/full.npz
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from echomario.envs.make_env import make_env
from echomario.teacher_data import record_levels, save_levels
from echomario.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--levels', type=int, default=40)
    p.add_argument('--seed-start', type=int, default=200_000)
    p.add_argument('--plan-every', type=int, default=2)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--teacher-style', default='speedrunner', help="'speedrunner' (default) or 'collector'")
    p.add_argument('--out', required=True)
    args = p.parse_args()

    config = load_config(args.config)
    seeds = list(range(args.seed_start, args.seed_start + args.levels))
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    start = time.perf_counter()
    with mp.get_context('fork').Pool(args.workers) as pool:
        parts = pool.map(record_levels, [(config, c, args.plan_every, args.teacher_style) for c in chunks])
    levels = sorted((lv for part in parts for lv in part), key=lambda lv: lv['seed'])
    env = make_env(config)
    base = env.unwrapped
    screen_dim = int(base.screen_dim)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_levels(levels, out, screen_dim)
    frames = sum(len(lv['obs']) for lv in levels)
    wins = sum(lv['reached_goal'] for lv in levels)
    print(f'recorded {len(levels)} levels, {frames} frames, teacher finished {wins}/{len(levels)} '
          f'in {time.perf_counter() - start:.0f}s -> {out}')


if __name__ == '__main__':
    main()
