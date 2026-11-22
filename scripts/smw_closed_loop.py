#!/usr/bin/env python
"""
Closed-loop test of level transfer: does a readout trained on other levels actually play unseen levels better?

Uses the level folds of smw_scaling.py (variants are grouped with their base level). For each fold, a ridge
readout on the 3000-neuron reservoir is trained on the pool levels' teacher labels (q targets), then plays every
held-out base level alone (no teacher) from a few start delays. Compared with: the constant student (the
pool's best single button, decided at the same 4-step cadence) and bunny-hop (run right, press jump 12 steps,
release 2, repeat).

    PYTHONPATH=scripts .venv/bin/python scripts/smw_closed_loop.py --data runs/smw/v3_all/data --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from echomario.envs.smw_env import ACTION_NAMES, SMWEnv
from echomario.imitation import make_targets
from smw_imitation import job, parse_spec, play
from smw_readout_lab import load_episodes

RUN, RUN_JUMP = ACTION_NAMES.index('right_run'), ACTION_NAMES.index('right_run_jump')


def bunny_hop(arg) -> dict:
    spec, noop, max_steps = arg
    level, start = parse_spec(spec)
    env = SMWEnv(level, max_steps=max_steps)
    env.reset(options=start)
    env.set_render(False)
    for _ in range(noop):
        env.step(0)
    env.track.steps = env.track.since_progress = 0
    step = 0
    while True:
        _, _, te, tr, info = env.step(RUN_JUMP if step % 14 < 12 else RUN)
        step += 1
        if te or tr:
            break
    env.close()
    return {'level': spec, 'progress': float(info['progress']), 'reached_goal': bool(info['reached_goal'] or info['reached_end']),
            'died': bool(info['died']), 'noop': noop}


def ridge(F: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    G = F.T.astype(np.float64) @ F / len(F)
    B = F.T.astype(np.float64) @ Y / len(F)
    r = alpha * np.trace(G) / G.shape[0]
    G[np.diag_indices_from(G)] += r
    G[-1, -1] -= r  # the bias column is not regularized
    return np.linalg.solve(G, B).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='runs/smw/v3_all/data')
    p.add_argument('--alpha', type=float, default=10.0)
    p.add_argument('--starts', default='0,20')
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', default='runs/smw/closed_loop_v3.json')
    args = p.parse_args()
    data = Path(args.data)
    cfg = json.loads((data.parent / 'results.json').read_text())['config']
    eps = load_episodes(data)
    S = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    labels = np.concatenate([e['labels'] for e in eps])
    Y = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(S, labels)])
    levels = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    F = np.load(data.parent / 'feats_reservoir_baseline.npy')
    assert len(F) == len(Y)
    uniq = sorted(set(levels.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % 4) for i, lv in zip(order, uniq)}
    starts = [int(s) for s in args.starts.split(',')]

    jobs, hop_jobs, meta = [], [], []
    for fold in range(4):
        tr = np.nonzero(np.array([fold_of[lv] != fold for lv in levels]))[0]
        W = ridge(F[tr], Y[tr], args.alpha)
        const = int(np.argmax(Y[tr].mean(0)))
        Wc = np.zeros_like(W)
        Wc[-1, const] = 1.0  # bias only: always the pool's best single button
        test = [lv for lv in uniq if fold_of[lv] == fold]
        print(f'fold {fold}: train {len(tr)} moments, constant button {ACTION_NAMES[const]}, test {test}', flush=True)
        for lv in test:
            for n in starts:
                for name, w in (('ridge', W), ('constant', Wc)):
                    jobs.append(job(lv, cfg, 'reservoir', w, max_steps=args.max_steps, noop=n))
                    meta.append((fold, name, lv, n))
                hop_jobs.append((lv, n, args.max_steps))
    with mp.get_context('fork').Pool(args.workers, maxtasksperchild=4) as pool:
        res = pool.map(play, jobs, chunksize=1)
        hops = pool.map(bunny_hop, hop_jobs, chunksize=1)
    rows = [{'fold': f, 'policy': name, 'level': lv, 'noop': n, 'progress': r['progress'],
             'goal': r['reached_goal'], 'died': r['died']} for (f, name, lv, n), r in zip(meta, res)]
    rows += [{'fold': fold_of[h['level']], 'policy': 'bunny_hop', 'level': h['level'], 'noop': h['noop'],
              'progress': h['progress'], 'goal': h['reached_goal'], 'died': h['died']} for h in hops]
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print('\npolicy       mean progress  goal rate  died')
    for name in ('ridge', 'constant', 'bunny_hop'):
        rs = [r for r in rows if r['policy'] == name]
        print(f"{name:12s} {np.mean([r['progress'] for r in rs]):.3f}          {np.mean([r['goal'] for r in rs]):.2f}       "
              f"{np.mean([r['died'] for r in rs]):.2f}")
    by = {}
    for r in rows:
        by.setdefault((r['level'], r['noop']), {})[r['policy']] = r['progress']
    diff = np.array([v['ridge'] - v['bunny_hop'] for v in by.values()])
    print(f'ridge vs bunny-hop per (level, start): better {(diff > 0.02).sum()}, worse {(diff < -0.02).sum()}, '
          f'same {(np.abs(diff) <= 0.02).sum()}')


if __name__ == '__main__':
    main()
