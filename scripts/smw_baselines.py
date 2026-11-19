#!/usr/bin/env python
"""Constant-action floors for Super Mario World: hold right+run, or bunny-hop (run, press jump, release, repeat)."""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
import multiprocessing as mp
import sys

import numpy as np

from echomario.envs.smw_env import ACTION_NAMES, TEST_LEVELS, TRAIN_LEVELS, SMWEnv

RUN, RUN_JUMP = ACTION_NAMES.index('right_run'), ACTION_NAMES.index('right_run_jump')


def play(args):
    level, policy = args
    env = SMWEnv(level, max_steps=2400)
    env.reset()
    step = 0
    while True:
        a = RUN if policy == 'run' else (RUN_JUMP if step % 14 < 12 else RUN)
        _, _, te, tr, info = env.step(a)
        step += 1
        if te or tr:
            break
    env.close()
    return level, policy, round(info['progress'], 3), bool(info['reached_goal'])


if __name__ == '__main__':
    jobs = [(lv, pol) for pol in ('run', 'hop') for lv in TRAIN_LEVELS + TEST_LEVELS]
    with mp.get_context('fork').Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1, maxtasksperchild=1) as pool:
        res = pool.map(play, jobs, chunksize=1)
    out = {}
    for lv, pol, prog, fin in res:
        out.setdefault(pol, {})[lv] = {'progress': prog, 'finished': fin}
    for pol, d in out.items():
        tr = [d[lv]['progress'] for lv in TRAIN_LEVELS]
        te = [d[lv]['progress'] for lv in TEST_LEVELS]
        print(pol, 'train progress', round(float(np.mean(tr)), 3), 'test progress', round(float(np.mean(te)), 3),
              'finished', sum(v['finished'] for v in d.values()), '/', len(d))
    print(json.dumps(out))
