#!/usr/bin/env python
"""
Teaches a frozen reservoir to play real Super Mario World (SNES, via stable-retro): the look-ahead
teacher labels states with emulator save-states, a ridge readout is fitted on reservoir features,
and DAgger rounds let the student drive while the teacher keeps labelling.

    python scripts/smw_imitation.py --out runs/smw/res3000 --iters 4 --workers 3

Needs your own ROM imported once: python -m stable_retro.import <dir with the .sfc>.
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from echomario.envs.smw_env import ACTION_NAMES, OBS_DIM, TEST_LEVELS, TRAIN_LEVELS, SMWEnv
from echomario.experts.smw_planner import SMWPlanner, SMWPlannerConfig
from echomario.imitation import FeatureMap, RidgeAccumulator, make_targets

N_ACT = len(ACTION_NAMES)


def reservoir_config(size: int, seed: int) -> dict:
    return {
        'env': {'name': 'smw'},
        'reservoir': {
            'type': 'sparse_esn', 'size': size, 'spectral_radius': 0.9, 'leak_rate': 0.5, 'input_scale': 1.0,
            'sparsity': 0.01, 'seed': seed, 'state_norm': 'none', 'input_structure': 'local',
            'leak_rate_min': 0.05, 'leak_rate_max': 1.0, 'patch_radius': 2,
        },
    }


def play(args: tuple) -> dict:
    """One episode. beta: probability that the teacher's action is executed at a decision;
    label: whether the teacher labels the visited states."""
    level, config, mode, weights, beta, label, plan_every, q_clip, seed, max_steps = args
    rng = np.random.default_rng(seed)
    env = SMWEnv(level, max_steps=max_steps)
    obs, info = env.reset()
    fm = FeatureMap(config, OBS_DIM, mode)
    fm.reset()
    planner = SMWPlanner(SMWPlannerConfig()) if (label or beta > 0) else None
    feats, targets, labels = [], [], []
    action, step, use_teacher, agree, n_lab = 0, 0, True, 0, 0
    t0 = time.perf_counter()
    while True:
        f = fm.step(obs)
        student_a = int(np.argmax(f @ weights)) if weights is not None else 0
        if step % plan_every == 0:
            if planner is not None and label:
                scores = planner.scores(env)
                expert = planner.choose(scores)
                feats.append(f)
                targets.append(make_targets(scores, expert, 'q', q_clip))
                labels.append(expert)
                agree += int(student_a == expert)
                n_lab += 1
            else:
                expert = None
            use_teacher = weights is None or (expert is not None and rng.random() < beta)
            teacher_a = expert
        action = teacher_a if use_teacher and teacher_a is not None else student_a
        obs, _r, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            break
    env.close()
    out = {
        'level': level, 'reached_goal': bool(info['reached_goal']), 'died': bool(info['died']),
        'progress': float(info['progress']), 'steps': step, 'coins': int(info['coins']),
        'agreement': agree / max(1, n_lab), 'seconds': round(time.perf_counter() - t0, 1),
    }
    if label:
        out['feats'] = np.asarray(feats, np.float32)
        out['targets'] = np.asarray(targets, np.float32)
        out['labels'] = np.asarray(labels, np.int64)
    return out


def summarize(eps: list[dict]) -> dict:
    return {
        'episodes': len(eps),
        'finished': round(float(np.mean([e['reached_goal'] for e in eps])), 3),
        'progress': round(float(np.mean([e['progress'] for e in eps])), 3),
        'died': round(float(np.mean([e['died'] for e in eps])), 3),
        'per_level': {e['level']: round(e['progress'], 2) for e in eps},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--mode', default='reservoir', choices=('reservoir', 'input', 'reservoir+input'))
    p.add_argument('--size', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=4)
    p.add_argument('--beta-decay', type=float, default=0.5)
    p.add_argument('--plan-every', type=int, default=4)
    p.add_argument('--q-clip', type=float, default=32.0, help='score gap in pixels that maps to target 0')
    p.add_argument('--critical-weight', type=float, default=4.0)
    p.add_argument('--alphas', default='1e-3,1e-2,1e-1,1')
    p.add_argument('--train-levels', default=','.join(TRAIN_LEVELS))
    p.add_argument('--test-levels', default=','.join(TEST_LEVELS))
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = reservoir_config(args.size, args.seed)
    train = args.train_levels.split(',')
    test = [lv for lv in args.test_levels.split(',') if lv]
    dim = FeatureMap(config, OBS_DIM, args.mode).dim
    acc = RidgeAccumulator(dim, N_ACT)
    alphas = [float(a) for a in args.alphas.split(',')]
    weights, history, best = None, [], None
    pool = mp.get_context('fork').Pool(args.workers, maxtasksperchild=1)
    start = time.perf_counter()
    for it in range(args.iters):
        beta = args.beta_decay ** it
        jobs = [(lv, config, args.mode, weights, beta, True, args.plan_every, args.q_clip, 1000 * it + i,
                 args.max_steps) for i, lv in enumerate(train)]
        collected = pool.map(play, jobs, chunksize=1)
        for ep in collected:
            if len(ep['labels']):
                acc.add(ep.pop('feats'), ep.pop('labels'), ep.pop('targets'), critical_weight=args.critical_weight)
        # pick alpha by closed-loop progress on the training levels (the test levels stay unseen)
        cands = []
        for alpha in alphas:
            w = acc.solve(alpha)
            ev = pool.map(play, [(lv, config, args.mode, w, 0.0, False, args.plan_every, args.q_clip, 0, args.max_steps)
                                 for lv in train], chunksize=1)
            cands.append((summarize(ev)['progress'], alpha, w, ev))
        prog, alpha, weights, ev_train = max(cands, key=lambda c: c[0])
        ev_test = pool.map(play, [(lv, config, args.mode, weights, 0.0, False, args.plan_every, args.q_clip, 0,
                                   args.max_steps) for lv in test], chunksize=1) if test else []
        rec = {
            'iteration': it, 'beta': beta, 'samples': int(acc.count), 'alpha': alpha,
            'collect': summarize(collected), 'agreement': round(float(np.mean([e['agreement'] for e in collected])), 3),
            'train': summarize(ev_train), 'test': summarize(ev_test) if ev_test else None,
            'label_share': (acc.label_counts / max(1, acc.label_counts.sum())).round(3).tolist(),
            'seconds': round(time.perf_counter() - start),
        }
        history.append(rec)
        print(f"iter {it}: beta={beta:.2f} samples={acc.count} alpha={alpha} teacher/collect finished="
              f"{rec['collect']['finished']} train finished={rec['train']['finished']} prog={rec['train']['progress']} "
              f"test finished={rec['test']['finished'] if ev_test else '-'} prog={rec['test']['progress'] if ev_test else '-'} "
              f"({rec['seconds']}s)", flush=True)
        np.save(out / f'readout_iter{it}.npy', weights)
        if best is None or prog >= best[0]:
            best = (prog, it)
            np.save(out / 'readout.npy', weights)
        (out / 'results.json').write_text(json.dumps({
            'config': config, 'mode': args.mode, 'args': vars(args), 'iterations': history,
            'best_iteration': best[1]}, indent=1))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
