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

from echomario.envs.smw_env import (
    ACTION_NAMES,
    BALANCED_TEST_LEVELS,
    BALANCED_TRAIN_LEVELS,
    TEST_LEVELS,
    TRAIN_LEVELS,
    SMWEnv,
    grid_kwargs,
    smw_obs_dim,
)
from echomario.experts.smw_astar import SMWAStar, astar_config
from echomario.experts.smw_planner import SMWPlanner, teacher_config
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


MIDPOINTS = Path('runs/smw/midpoints')


def parse_spec(spec: str) -> tuple[str, dict]:
    """'Level' = the whole level; 'Level@first' = until halfway (the episode ends at 50 %);
    'Level@second' = from the halfway point saved by scripts/smw_midpoints.py, progress over the second half."""
    level, _, part = spec.partition('@')
    level, _, mut = level.partition('#m')  # 'Level#m7': automatic variant 7 of the level (smw_env.mutate_level)
    extra = {'mutate': int(mut)} if mut else {}
    if part == 'first':
        return level, {'end_fraction': 0.5, **extra}
    if part == 'second':
        if extra:
            raise ValueError('mutated second halves are not supported (the midpoint replay assumes the original level)')
        meta = json.loads((MIDPOINTS / f'{level}.json').read_text())
        return level, {'prefix': meta['actions']}
    if part:
        raise ValueError(f'unknown level part {spec!r}')
    return level, extra


def job(level: str, config: dict, mode: str, weights: np.ndarray | None, **kw) -> dict:
    """An episode for play() (level: a spec, see parse_spec). Defaults: the student plays alone, nothing labelled."""
    return {'level': level, 'config': config, 'mode': mode, 'weights': weights, 'beta': 0.0, 'label': False,
            'plan_every': 4, 'q_clip': 32.0, 'seed': 0, 'max_steps': 2400, 'rescue': 0, 'record': False,
            'teacher': 'v1', 'noop': 0, **kw}


def student_values(f: np.ndarray, weights) -> np.ndarray:
    """The student's action values: a linear readout (array) or a small MLP (dict W1, b1, W2, b2, numpy)."""
    if isinstance(weights, dict):
        return np.maximum(f @ weights['W1'] + weights['b1'], 0.0) @ weights['W2'] + weights['b2']
    return f @ weights


def make_teacher(name: str, seed: int = 0):
    """'v1' / 'v2' / 'v2-collector' / 'v3': SMWPlanner; 'astar' / 'astar-div': the A* teacher (smw_astar.py)."""
    if name.startswith('astar'):
        t = SMWAStar(astar_config(name))
        t.reseed(seed)
        return t
    return SMWPlanner(teacher_config(name))


def play(j: dict) -> dict:
    """One episode (see job()). beta: probability that the teacher's action is executed at a decision;
    label: the teacher labels the visited states; rescue: rewinds after a student death in labelled runs;
    noop: idle steps before play (other enemy / platform phases); record: keep a replay log."""
    spec, config, mode, weights, plan_every = j['level'], j['config'], j['mode'], j['weights'], j['plan_every']
    label, beta, rescue, record = j['label'], j['beta'], j['rescue'], j['record']
    level, start = parse_spec(spec)
    rewind = 60  # steps (2 s) to rewind before the death; the teacher then drives for rewind + 20
    rng = np.random.default_rng(j['seed'])
    env = SMWEnv(level, max_steps=j['max_steps'], **grid_kwargs(config))
    obs, info = env.reset(options=start)
    env.set_render(False)  # training never looks at the picture
    for _ in range(j['noop']):
        obs, _r, _te, _tr, info = env.step(0)
    env.track.steps = env.track.since_progress = 0
    fm = FeatureMap(config, env.obs_dim, mode)
    fm.reset()
    planner = make_teacher(j['teacher'], j['seed']) if (label or beta > 0) else None
    feats, targets, labels, tried, raw_scores, sample_pos = [], [], [], [], [], []
    # replay log: the game step and action of every executed step, in order. A rewind shows up as the step
    # number jumping back. The emulator is deterministic, so this reproduces every state (and any
    # observation encoding of it) without the teacher: see echomario.envs.smw_env.replay_observations.
    log_steps, log_actions = [], []
    step, use_teacher, agree, n_lab, student_a, teacher_a = 0, True, 0, 0, 0, None
    snaps, rescues, forced_until, furthest = deque(maxlen=rewind // plan_every + 2), 0, -1, 0.0
    t0 = time.perf_counter()
    while True:
        if rescue and step % plan_every == 0:
            snaps.append((step, env.get_sim_state(), fm.reservoir.x.copy(), obs.copy()))
        f = fm.step(obs)
        if step % plan_every == 0:
            # the student decides at the teacher's cadence and holds its choice, like the teacher
            student_a = int(np.argmax(student_values(f, weights))) if weights is not None else 0
            expert = None
            if planner is not None and label:
                scores = planner.scores(env)
                expert = planner.choose(scores)
                feats.append(f)
                targets.append(make_targets(scores, expert, 'q', j['q_clip']))
                labels.append(expert)
                tried.append(np.isfinite(scores))
                raw_scores.append(scores.astype(np.float32))
                sample_pos.append(len(log_steps))
                agree += int(student_a == expert)
                n_lab += 1
            use_teacher = weights is None or (expert is not None and (step < forced_until or rng.random() < beta))
            teacher_a = expert
        action = teacher_a if use_teacher and teacher_a is not None else student_a
        log_steps.append(step)
        log_actions.append(action)
        obs, _r, terminated, truncated, info = env.step(action)
        step += 1
        furthest = max(furthest, float(info['progress']))
        if terminated and info['died'] and label and weights is not None and rescues < rescue and snaps:
            # DAgger rescue: rewind ~2 s, let the teacher steer past the danger, then hand back.
            # Without this the student dies early and every round only adds labels from the
            # first tenth of each level.
            snap = min(snaps, key=lambda s_: abs(s_[0] - (step - rewind)))
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
        'level': spec, 'reached_goal': bool(info['reached_goal'] or info['reached_end']), 'died': bool(info['died']),
        'progress': float(info['progress']), 'steps': step, 'coins': int(info['coins']),
        'agreement': agree / max(1, n_lab), 'seconds': round(time.perf_counter() - t0, 1),
        'rescues': rescues, 'furthest': furthest, 'noop': j['noop'],
    }
    if label:
        out['feats'] = np.asarray(feats, np.float32)
        out['targets'] = np.asarray(targets, np.float32)
        out['labels'] = np.asarray(labels, np.int64)
        out['tried'] = np.asarray(tried, bool).reshape(len(labels), N_ACT)
    if record:
        out['record'] = {'level': level, 'spec': spec, 'noop': j['noop'], 'prefix': np.asarray(start.get('prefix', []), np.int8),
                         'end_fraction': float(start.get('end_fraction') or 0.0), 'mutate': int(start.get('mutate', -1)),
                         'steps': np.asarray(log_steps, np.int32),
                         'actions': np.asarray(log_actions, np.int8), 'sample_pos': np.asarray(sample_pos, np.int64),
                         'scores': np.asarray(raw_scores, np.float32).reshape(-1, N_ACT),
                         'labels': np.asarray(labels, np.int64)}
    return out


def summarize(eps: list[dict]) -> dict:
    prog = np.array([e['progress'] for e in eps])
    levels = list(dict.fromkeys(e['level'] for e in eps))
    return {
        'episodes': len(eps),
        'finished': round(float(np.mean([e['reached_goal'] for e in eps])), 3),
        'progress': round(float(prog.mean()), 3),
        'se': round(float(prog.std(ddof=1) / np.sqrt(len(prog))), 3) if len(prog) > 1 else 0.0,
        'died': round(float(np.mean([e['died'] for e in eps])), 3),
        'per_level': {lv: round(float(np.mean([e['progress'] for e in eps if e['level'] == lv])), 2) for lv in levels},
        **({'rescues': int(sum(e.get('rescues', 0) for e in eps)),
            'furthest': round(float(np.mean([e.get('furthest', e['progress']) for e in eps])), 3)}
           if any(e.get('rescues') for e in eps) else {}),
    }


def evaluate(pool, levels: list[str], starts: list[int], config: dict, mode: str, w: np.ndarray,
             plan_every: int, max_steps: int) -> list[dict]:
    """The student alone on every level from every start delay (deterministic otherwise)."""
    jobs = [job(lv, config, mode, w, plan_every=plan_every, max_steps=max_steps, noop=n) for lv in levels for n in starts]
    return pool.map(play, jobs, chunksize=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--mode', default='reservoir', choices=('reservoir', 'input', 'reservoir+input'))
    p.add_argument('--size', type=int, default=3000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--leak-min', type=float, default=0.05, help='slowest neuron leak rate (memory)')
    p.add_argument('--iters', type=int, default=4)
    p.add_argument('--beta-decay', type=float, default=0.5)
    p.add_argument('--plan-every', type=int, default=4)
    p.add_argument('--q-clip', type=float, default=32.0, help='score gap in pixels that maps to target 0')
    p.add_argument('--critical-weight', type=float, default=4.0,
                   help='weight 1 + k * (spread of the targets over the actions the teacher tried)')
    p.add_argument('--alphas', default='1e-3,1e-2,1e-1,1')
    p.add_argument('--split', default='original', choices=('original', 'balanced'),
                   help='default level lists; --train-levels / --test-levels override')
    p.add_argument('--train-levels', default='')
    p.add_argument('--test-levels', default='')
    p.add_argument('--split-file', default='',
                   help="JSON {'full': [...], 'half': [...], 'unseen': [...]}: train on full levels and the first half of the "
                        "half levels; test on their second halves and on the unseen levels (overrides the level lists)")
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--rescues', type=int, default=0, help='rewinds per DAgger episode after a student death')
    p.add_argument('--teacher', default='v1', choices=('v1', 'v2', 'v2-collector'),
                   help='v1: 1 s look-ahead, v2: + 3 s safety check, v2-collector: v2 + coins, enemies, power-ups')
    p.add_argument('--obs', default='v1', choices=('v1', 'v2'), help='observation encoding (see smw_env)')
    p.add_argument('--start-delay-max', type=int, default=0,
                   help='labelled episodes start after a random 0..N idle steps (variety beyond one replay per level)')
    p.add_argument('--eval-starts', default='0', help='start delays for the closed-loop evaluations')
    p.add_argument('--save-data', action='store_true',
                   help='save every labelled episode (replay log + teacher scores) to <out>/data for offline tests')
    p.add_argument('--grid', default='', help="tile window 'rows,cols,rows_above,cols_behind' (default 16,20,10,4)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = dict(zip(('rows', 'cols', 'rows_above', 'cols_behind'), map(int, args.grid.split(',')))) if args.grid else None
    config = reservoir_config(args.size, args.seed, grid)
    config['env']['obs'] = args.obs
    config['reservoir']['leak_rate_min'] = args.leak_min
    default_train, default_test = ((BALANCED_TRAIN_LEVELS, BALANCED_TEST_LEVELS) if args.split == 'balanced'
                                   else (TRAIN_LEVELS, TEST_LEVELS))
    train = args.train_levels.split(',') if args.train_levels else list(default_train)
    test = [lv for lv in (args.test_levels.split(',') if args.test_levels else default_test) if lv]
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text())
        train = list(split['full']) + [f'{lv}@first' for lv in split['half']]
        test = [f'{lv}@second' for lv in split['half']] + list(split['unseen'])
    args.train_levels, args.test_levels = ','.join(train), ','.join(test)
    starts = [int(x) for x in args.eval_starts.split(',')]
    dim = FeatureMap(config, smw_obs_dim(config), args.mode).dim
    acc = RidgeAccumulator(dim, N_ACT)
    alphas = [float(a) for a in args.alphas.split(',')]
    rng = np.random.default_rng(args.seed)
    weights, history, best = None, [], None
    pool = mp.get_context('fork').Pool(args.workers, maxtasksperchild=1)
    start = time.perf_counter()
    for it in range(args.iters):
        beta = args.beta_decay ** it
        jobs = [job(lv, config, args.mode, weights, beta=beta, label=True, plan_every=args.plan_every,
                    q_clip=args.q_clip, seed=1000 * it + i, max_steps=args.max_steps, rescue=args.rescues,
                    record=args.save_data, teacher=args.teacher,
                    noop=int(rng.integers(0, args.start_delay_max + 1)) if args.start_delay_max else 0)
                for i, lv in enumerate(train)]
        collected = pool.map(play, jobs, chunksize=1)
        for ep in collected:
            if 'record' in ep:
                (out / 'data').mkdir(exist_ok=True)
                np.savez_compressed(out / 'data' / f"iter{it}_{ep['level']}.npz", **ep.pop('record'))
            if len(ep['labels']):
                acc.add(ep.pop('feats'), ep.pop('labels'), ep.pop('targets'), critical_weight=args.critical_weight,
                        tried=ep.pop('tried'))
        # pick alpha by closed-loop progress on the training levels (the test levels stay unseen)
        cands = []
        for alpha in alphas:
            w = acc.solve(alpha)
            ev = evaluate(pool, train, starts, config, args.mode, w, args.plan_every, args.max_steps)
            cands.append((summarize(ev)['progress'], alpha, w, ev))
        prog, alpha, weights, ev_train = max(cands, key=lambda c: c[0])
        ev_test = evaluate(pool, test, starts, config, args.mode, weights, args.plan_every, args.max_steps) if test else []
        groups = {}
        for e in ev_train + ev_test:
            g = 'first_half' if e['level'].endswith('@first') else 'second_half' if e['level'].endswith('@second') else (
                'full' if e in ev_train else 'unseen')
            groups.setdefault(g, []).append(e)
        rec = {
            'iteration': it, 'beta': beta, 'samples': int(acc.count), 'alpha': alpha,
            'groups': {g: summarize(v) for g, v in groups.items()},
            'collect': summarize(collected), 'agreement': round(float(np.mean([e['agreement'] for e in collected])), 3),
            'train': summarize(ev_train), 'test': summarize(ev_test) if ev_test else None,
            'label_share': (acc.label_counts / max(1, acc.label_counts.sum())).round(3).tolist(),
            'seconds': round(time.perf_counter() - start),
        }
        history.append(rec)
        print(f"iter {it}: beta={beta:.2f} samples={acc.count} alpha={alpha} teacher/collect finished="
              f"{rec['collect']['finished']} prog={rec['collect']['progress']} | train finished={rec['train']['finished']} "
              f"prog={rec['train']['progress']} +- {rec['train']['se']} | test finished="
              f"{rec['test']['finished'] if ev_test else '-'} prog={rec['test']['progress'] if ev_test else '-'} "
              f"+- {rec['test']['se'] if ev_test else '-'} ({rec['seconds']}s)", flush=True)
        if args.split_file:
            print('    groups: ' + '  '.join(f"{g} prog={v['progress']} +- {v['se']} finished={v['finished']}"
                                          for g, v in rec['groups'].items()), flush=True)
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
