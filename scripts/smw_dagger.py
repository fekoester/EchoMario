#!/usr/bin/env python
"""
DAgger on many levels, tested closed loop on unseen levels.

Starts from the teacher-only dataset (e.g. runs/smw/v3_all: replay logs + reservoir features), uses one level
fold of smw_scaling.py / smw_closed_loop.py as the unseen test set, and repeats:

  1. fit the ridge readout on all labels so far, for each alpha of --alphas, and keep the alpha whose student
     plays the training levels best (closed loop, start delay 0); the unseen levels are played with every
     alpha for the record, but never used to choose,
  2. let the student play every training level (base levels and variants of the other folds) while the
     teacher labels every decision; the teacher steers only with probability beta and for a few seconds after
     a rewind (--rescues: after a student death, rewind 2 s and let the teacher drive past the danger),
  3. play the held-out levels alone (no teacher) from the same start delays as smw_closed_loop.py.

Iteration 0 is the offline student (teacher data only). Each iteration's new labels are saved to
<out>/iter{k}.npz (and replay logs to <out>/data), so a stopped run resumes where it was.

    PYTHONPATH=scripts .venv/bin/python scripts/smw_dagger.py --data runs/smw/v3_all/data --fold 0 \
        --out runs/smw/dagger_v3_f0 --iters 4 --workers 7
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

from echomario.envs.smw_env import ACTION_NAMES
from echomario.imitation import make_targets
from smw_closed_loop import ridge
from smw_imitation import job, play, summarize
from smw_readout_lab import load_episodes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='runs/smw/v3_all/data')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--out', default='runs/smw/dagger_v3_f0')
    p.add_argument('--iters', type=int, default=4, help='DAgger collection rounds after the offline student')
    p.add_argument('--alphas', default='0.001,0.01,0.1,1,10', help='ridge alphas (relative to the mean feature variance)')
    p.add_argument('--beta', default='0.3,0,0,0', help='teacher share per collection round')
    p.add_argument('--rescues', type=int, default=3)
    p.add_argument('--delay-max', type=int, default=40, help='collection episodes start after 0..N idle steps')
    p.add_argument('--starts', default='0,20', help='start delays of the closed-loop tests')
    p.add_argument('--teacher', default='v3')
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--workers', type=int, default=7)
    args = p.parse_args()
    out = Path(args.out)
    (out / 'data').mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    cfg = json.loads((data.parent / 'results.json').read_text())['config']
    eps = load_episodes(data)
    S = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    labels = np.concatenate([e['labels'] for e in eps])
    Y0 = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(S, labels)]).astype(np.float32)
    lv0 = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    F0 = np.load(data.parent / 'feats_reservoir_baseline.npy')
    assert len(F0) == len(Y0)
    uniq = sorted(set(lv0.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % 4) for i, lv in zip(order, uniq)}  # identical to smw_scaling.py / smw_closed_loop.py
    test = [lv for lv in uniq if fold_of[lv] == args.fold]
    train_mask = np.array([fold_of[lv] != args.fold for lv in lv0])
    specs = sorted({Path(f).stem.split('_', 1)[1] for f in map(str, data.glob('*.npz'))})
    train_specs = [s for s in specs if fold_of[s.split('#m')[0].split('@')[0]] != args.fold]
    starts = [int(s) for s in args.starts.split(',')]
    alphas = [float(a) for a in args.alphas.split(',')]
    betas = [float(b) for b in args.beta.split(',')]
    print(f'fold {args.fold}: {train_mask.sum()} offline labels, {len(train_specs)} training episodes per round, '
          f'test {test}', flush=True)

    Fs, Ys = [F0[train_mask]], [Y0[train_mask]]
    hist_path = out / 'results.json'
    history = json.loads(hist_path.read_text())['iterations'] if hist_path.exists() else []
    done_eval = {h['iteration'] for h in history}
    rng = np.random.default_rng(1000 + args.fold)
    pool = mp.get_context('fork').Pool(args.workers, maxtasksperchild=4)
    t_start = time.time()
    for it in range(args.iters + 1):
        if it > 0:  # collect round `it` with the student of iteration it-1
            f = out / f'iter{it}.npz'
            noops = rng.integers(0, args.delay_max + 1, len(train_specs))
            if not f.exists():
                beta = betas[min(it - 1, len(betas) - 1)]
                jobs = [job(s, cfg, 'reservoir', W, beta=beta, label=True, seed=1000 * it + i, max_steps=args.max_steps,
                            rescue=args.rescues, record=True, teacher=args.teacher, noop=int(n))
                        for i, (s, n) in enumerate(zip(train_specs, noops))]
                feats, targs, coll = [], [], []
                for ep in pool.imap_unordered(play, jobs, chunksize=1):
                    rec = ep.pop('record')
                    np.savez_compressed(out / 'data' / f"iter{it}_{ep['level']}.npz", **rec)
                    if len(ep['labels']):
                        feats.append(ep.pop('feats'))
                        targs.append(ep.pop('targets'))
                    for k in ('feats', 'targets', 'labels', 'tried'):
                        ep.pop(k, None)
                    coll.append(ep)
                    print(f"  iter {it} collect {len(coll)}/{len(jobs)} {ep['level']}: prog {ep['progress']:.2f} "
                          f"furthest {ep['furthest']:.2f} rescues {ep['rescues']} labels {len(targs[-1]) if targs else 0} "
                          f"agree {ep['agreement']:.2f} | {(time.time() - t_start) / 60:.0f} min", flush=True)
                np.savez(f, feats=np.concatenate(feats), targets=np.concatenate(targs),
                         collect=json.dumps(coll), beta=beta)
            z = np.load(f)
            Fs.append(z['feats'])
            Ys.append(z['targets'])
        F, Y = np.concatenate(Fs), np.concatenate(Ys)
        if it in done_eval:  # resumed: the student of this iteration is on disk
            W = np.load(out / f'readout_iter{it}.npy')
            continue
        cands = []
        for alpha in alphas:
            Wa = ridge(F, Y, alpha)
            et = pool.map(play, [job(s, cfg, 'reservoir', Wa, max_steps=args.max_steps, noop=0) for s in train_specs],
                          chunksize=1)
            ev = pool.map(play, [job(lv, cfg, 'reservoir', Wa, max_steps=args.max_steps, noop=n)
                                 for lv in test for n in starts], chunksize=1)
            qfit = float(Y[np.arange(len(Y)), (F @ Wa).argmax(1)].mean())
            cands.append({'alpha': alpha, 'W': Wa, 'train': summarize(et), 'test': summarize(ev), 'ev': ev, 'fit_q': qfit})
            print(f"  iter {it} alpha {alpha}: fit q {qfit:.3f} | train levels prog {cands[-1]['train']['progress']} "
                  f"| unseen prog {cands[-1]['test']['progress']} (not used to choose)", flush=True)
        best = max(cands, key=lambda c: c['train']['progress'])
        W, ev = best['W'], best['ev']
        np.save(out / f'readout_iter{it}.npy', W)
        rec = {'iteration': it, 'labels': int(len(F)), 'alpha': best['alpha'], 'test': best['test'], 'train': best['train'],
               'test_episodes': [{k: e[k] for k in ('level', 'noop', 'progress', 'reached_goal', 'died')} for e in ev],
               'by_alpha': [{k: c[k] for k in ('alpha', 'fit_q', 'train', 'test')} for c in cands],
               'label_share': np.bincount(Y.argmax(1), minlength=len(ACTION_NAMES)).tolist(),
               'minutes': round((time.time() - t_start) / 60)}
        if it > 0:
            rec['collect'] = summarize(json.loads(str(np.load(out / f'iter{it}.npz')['collect'])))
        history.append(rec)
        hist_path.write_text(json.dumps({'args': vars(args), 'test_levels': test, 'iterations': history}, indent=1))
        print(f"iter {it}: labels {len(F)} alpha {best['alpha']} | unseen prog {rec['test']['progress']} +- {rec['test']['se']} "
              f"died {rec['test']['died']} goal {rec['test']['finished']} | train prog {rec['train']['progress']} "
              f"died {rec['train']['died']} | {rec['minutes']} min", flush=True)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
