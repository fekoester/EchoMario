#!/usr/bin/env python
"""
Mid-level start states for the 'second half' test split: the v2 teacher plays each level until Mario is
halfway (progress >= 0.5); the emulator state there is saved together with the button sequence that
reaches it (the emulator is deterministic, so the actions alone reproduce it).

    python scripts/smw_midpoints.py --out runs/smw/midpoints --workers 8
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from echomario.envs.smw_env import ALL_LEVELS, SMWEnv
from echomario.experts.smw_planner import SMWPlanner, teacher_config


def midpoint(a: tuple) -> dict:
    level, out, fraction, max_steps, teacher = a
    env = SMWEnv(level, max_steps=max_steps)
    env.reset()
    env.set_render(False)
    planner = SMWPlanner(teacher_config(teacher))
    actions, info, action = [], {'progress': 0.0}, 0
    for step in range(max_steps):
        if step % 4 == 0:
            action = planner.act(env)
        actions.append(action)
        _, _, terminated, truncated, info = env.step(action)
        if info['progress'] >= fraction and not info['died']:
            Path(out, f'{level}.state').write_bytes(env.em.get_state())
            m = env.mario()
            meta = {'level': level, 'fraction': fraction, 'steps': len(actions), 'x': m['x'], 'y': m['y'],
                    'goal_x': env.goal_x, 'teacher': teacher, 'actions': actions}
            Path(out, f'{level}.json').write_text(json.dumps(meta))
            env.close()
            return {'level': level, 'ok': True, 'steps': len(actions), 'x': m['x']}
        if terminated or truncated:
            break
    env.close()
    return {'level': level, 'ok': False, 'progress': round(info['progress'], 3), 'died': bool(info.get('died'))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--levels', default=','.join(ALL_LEVELS))
    p.add_argument('--fraction', type=float, default=0.5)
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--teacher', default='v2')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(lv, args.out, args.fraction, args.max_steps, args.teacher) for lv in args.levels.split(',')]
    with mp.get_context('fork').Pool(args.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(midpoint, jobs):
            print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
