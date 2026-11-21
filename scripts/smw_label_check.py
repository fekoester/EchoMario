#!/usr/bin/env python
"""
How reliable is the teacher's "deadly" label? A button is labelled deadly when none of the planner's few fixed
continuations (run, hop, long jump, wait-and-hop, run-up) survives. This script replays saved episodes to
labelled moments and, for each deadly (moment, button), searches harder: the button for the planner's first
4 steps, then many random sticky action sequences (segments of 2..12 steps of one button) for 3 seconds.
If any of them survives, the label was "deadly for the teacher's search" rather than "deadly".

    PYTHONPATH=scripts .venv/bin/python scripts/smw_label_check.py --pairs 300 --rollouts 150 --workers 3
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from echomario.envs.smw_env import SMWEnv, grid_kwargs
from smw_imitation import parse_spec

FIRST_HOLD = 4
HORIZON = 90  # steps after the first hold (FRAME_SKIP 2 -> 3 seconds)


def check_episode(arg):
    path, positions, config, rollouts, seed = arg
    with np.load(path) as z:
        log = {k: z[k] for k in z.files}
    level, start = parse_spec(str(log['spec']))
    if int(log.get('mutate', -1)) >= 0:
        start['mutate'] = int(log['mutate'])
    env = SMWEnv(level, max_steps=10 ** 9, stagnation_steps=10 ** 9, **grid_kwargs(config))
    env.reset(options=start)
    env.set_render(False)
    for _ in range(int(log['noop'])):
        env.step(0)
    steps, actions = log['steps'], log['actions']
    jumps = {int(steps[p]) for p in range(1, len(steps)) if steps[p] != steps[p - 1] + 1}
    saved = {}
    sample_pos = log['sample_pos'].tolist()
    want = {sample_pos[i]: i for i in positions}
    rng = np.random.default_rng(seed)
    out = []
    for p, (k, a) in enumerate(zip(steps.tolist(), actions.tolist())):
        if p and k != steps[p - 1] + 1:
            env.set_sim_state(saved[k])
        elif k in jumps and k not in saved:
            saved[k] = env.get_sim_state()
        if p in want:
            i = want[p]
            scores = log['scores'][i]
            root = env.get_sim_state()
            for b in np.nonzero(np.isfinite(scores) & (scores <= -1000))[0].tolist():
                survived = 0
                for _ in range(rollouts):
                    env.set_sim_state(root)
                    dead = False
                    for _ in range(FIRST_HOLD):
                        env.emulate(b)
                        _, _, term, _, info = env._after_step(compute_obs=False)
                        if term:
                            dead = info['died']
                            break
                    t = 0
                    while not dead and t < HORIZON:
                        act, seg = int(rng.integers(0, 8)), int(rng.integers(2, 13))
                        for _ in range(seg):
                            env.emulate(act)
                            _, _, term, _, info = env._after_step(compute_obs=False)
                            t += 1
                            if term:
                                dead = info['died']
                                t = HORIZON
                                break
                    if not dead:
                        survived += 1
                        break  # one survivor is enough
                out.append({'file': Path(path).stem, 'moment': i, 'button': b, 'rescued': survived > 0})
            env.set_sim_state(root)
        env.step(a)
    env.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='runs/smw/abc_v2/data')
    p.add_argument('--pairs', type=int, default=300, help='approximate number of deadly (moment, button) pairs')
    p.add_argument('--rollouts', type=int, default=150)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--out', default='runs/smw/label_check.json')
    args = p.parse_args()
    data = Path(args.data)
    config = json.loads((data.parent / 'results.json').read_text())['config']
    files = sorted(data.glob('iter*_*.npz'))
    rng = np.random.default_rng(0)
    pool_pairs = []
    for f in files:
        with np.load(f) as z:
            s = z['scores'].astype(np.float64)
        for i in np.nonzero((np.isfinite(s) & (s <= -1000)).any(1))[0]:
            pool_pairs.append((f, int(i), int((np.isfinite(s[i]) & (s[i] <= -1000)).sum())))
    pick = rng.choice(len(pool_pairs), min(len(pool_pairs), args.pairs // 3 + 1), replace=False)
    by_file: dict[Path, list[int]] = {}
    for j in pick:
        f, i, _ = pool_pairs[j]
        by_file.setdefault(f, []).append(i)
    jobs = [(str(f), sorted(v), config, args.rollouts, n) for n, (f, v) in enumerate(by_file.items())]
    print(f'{len(pool_pairs)} moments with a deadly button; checking {len(pick)} of them in {len(jobs)} episodes',
          flush=True)
    res = []
    with get_context('fork').Pool(args.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(check_episode, jobs):
            res += r
            Path(args.out).write_text(json.dumps(res))
            print(f'{len(res)} pairs checked, rescued {np.mean([x["rescued"] for x in res]):.1%}', flush=True)
    with_rescue = {}
    for x in res:
        with_rescue.setdefault((x['file'], x['moment']), []).append(x['rescued'])
    doomed_all = np.mean([all(v) for v in with_rescue.values()])
    print(f'FINAL: {len(res)} deadly labels checked; {np.mean([x["rescued"] for x in res]):.1%} survivable with a '
          f'random-sticky search; moments where every deadly button was survivable: {doomed_all:.1%}')
    by_button = {}
    for x in res:
        by_button.setdefault(x['button'], []).append(x['rescued'])
    print('by button:', {b: f'{np.mean(v):.0%} of {len(v)}' for b, v in sorted(by_button.items())})


if __name__ == '__main__':
    main()
