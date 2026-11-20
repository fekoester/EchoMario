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
from collections import deque
from pathlib import Path

import numpy as np

from echomario.envs.smw_env import ACTION_NAMES, TEST_LEVELS, TRAIN_LEVELS, SMWEnv, grid_kwargs, smw_obs_dim
from echomario.experts.smw_planner import SMWPlanner, SMWPlannerConfig
from echomario.imitation import FeatureMap, RidgeAccumulator, make_targets

N_ACT = len(ACTION_NAMES)


def reservoir_config(size: int, seed: int, grid: dict | None = None) -> dict:
    return {
        'env': {'name': 'smw', **({'grid': grid} if grid else {})},
        'reservoir': {
            'type': 'sparse_esn', 'size': size, 'spectral_radius': 0.9, 'leak_rate': 0.5, 'input_scale': 1.0,
            'sparsity': 0.01, 'seed': seed, 'state_norm': 'none', 'input_structure': 'local',
            'leak_rate_min': 0.05, 'leak_rate_max': 1.0, 'patch_radius': 2,
        },
    }


def play(args: tuple) -> dict:
    """One episode. beta: probability that the teacher's action is executed at a decision;
    label: whether the teacher labels the visited states."""
    level, config, mode, weights, beta, label, plan_every, q_clip, seed, max_steps = args[:10]
    rescue = args[10] if len(args) > 10 else 0  # max rewinds after a student death (labelled runs only)
    rewind = 60  # steps (2 s) to rewind before the death; the teacher then drives for rewind + 20
    rng = np.random.default_rng(seed)
    env = SMWEnv(level, max_steps=max_steps, **grid_kwargs(config))
    obs, info = env.reset()
    fm = FeatureMap(config, env.obs_dim, mode)
    fm.reset()
    planner = SMWPlanner(SMWPlannerConfig()) if (label or beta > 0) else None
    feats, targets, labels = [], [], []
    action, step, use_teacher, agree, n_lab, student_a, teacher_a = 0, 0, True, 0, 0, 0, None
    snaps, rescues, forced_until, furthest = deque(maxlen=rewind // plan_every + 2), 0, -1, 0.0
    t0 = time.perf_counter()
    while True:
        if rescue and step % plan_every == 0:
            snaps.append((step, env.get_sim_state(), fm.reservoir.x.copy(), obs.copy()))
        f = fm.step(obs)
        if step % plan_every == 0:
            # the student decides at the teacher's cadence and holds its choice, like the teacher
            student_a = int(np.argmax(f @ weights)) if weights is not None else 0
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
            use_teacher = weights is None or (expert is not None and (step < forced_until or rng.random() < beta))
            teacher_a = expert
        action = teacher_a if use_teacher and teacher_a is not None else student_a
        obs, _r, terminated, truncated, info = env.step(action)
        step += 1
        furthest = max(furthest, float(info['progress']))
        if terminated and info['died'] and label and weights is not None and rescues < rescue and snaps:
            # DAgger rescue: rewind ~2 s, let the teacher steer past the danger, then hand back.
            # Without this the student dies early and every round only adds labels from the
            # first tenth of each level.
            target = step - rewind
            snap = min(snaps, key=lambda s_: abs(s_[0] - target))
            step = snap[0]
            env.set_sim_state(snap[1])
            fm.reservoir.x = snap[2].copy()
            obs = snap[3].copy()
            forced_until = step + rewind + 20
            rescues += 1
            snaps.clear()
            continue
        if terminated or truncated:
            break
    env.close()
    out = {
        'level': level, 'reached_goal': bool(info['reached_goal']), 'died': bool(info['died']),
        'progress': float(info['progress']), 'steps': step, 'coins': int(info['coins']),
        'agreement': agree / max(1, n_lab), 'seconds': round(time.perf_counter() - t0, 1),
        'rescues': rescues, 'furthest': furthest,
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
        **({'rescues': int(sum(e.get('rescues', 0) for e in eps)),
            'furthest': round(float(np.mean([e.get('furthest', e['progress']) for e in eps])), 3)}
           if any(e.get('rescues') for e in eps) else {}),
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
    p.add_argument('--rescues', type=int, default=0, help='rewinds per DAgger episode after a student death')
    p.add_argument('--grid', default='', help="tile window 'rows,cols,rows_above,cols_behind' (default 16,20,10,4)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = dict(zip(('rows', 'cols', 'rows_above', 'cols_behind'), map(int, args.grid.split(',')))) if args.grid else None
    config = reservoir_config(args.size, args.seed, grid)
    train = args.train_levels.split(',')
    test = [lv for lv in args.test_levels.split(',') if lv]
    dim = FeatureMap(config, smw_obs_dim(config), args.mode).dim
    acc = RidgeAccumulator(dim, N_ACT)
    alphas = [float(a) for a in args.alphas.split(',')]
    weights, history, best = None, [], None
    pool = mp.get_context('fork').Pool(args.workers, maxtasksperchild=1)
    start = time.perf_counter()
    for it in range(args.iters):
        beta = args.beta_decay ** it
        jobs = [(lv, config, args.mode, weights, beta, True, args.plan_every, args.q_clip, 1000 * it + i,
                 args.max_steps, args.rescues) for i, lv in enumerate(train)]
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
