#!/usr/bin/env python
"""
Which reservoir generalizes? Level-wise cross-validation on saved DAgger data (smw_imitation.py --save-data).

The levels of the data are split into folds; every configuration is fitted on all but one fold and scored on
the teacher's labels of the held-out levels: how good the chosen action is on the teacher's 0..1 scale
(q_chosen, 1 = as good as the teacher's best), how often it is deadly, and the gap to the levels it was fitted
on. No emulator time except for the final closed-loop check of the best configurations (--closed-loop).

    python scripts/smw_cv_lab.py --data runs/smw/abc_v2/data --out runs/smw/cv_abc
    python scripts/smw_cv_lab.py --data runs/smw/abc_v2/data --out runs/smw/cv_abc --closed-loop 6
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import copy
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from echomario.imitation import make_targets
from smw_readout_lab import attach_streams, load_episodes, other_obs, replay_features

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


def feature_configs(base: dict) -> dict[str, tuple[dict, str]]:
    def res(**kw):
        c = copy.deepcopy(base)
        c['reservoir'].update(kw)
        return c

    return {
        'baseline (3000, leak 0.01-1, radius 0.9, input 1, patch 2)': (base, 'reservoir'),
        'raw input only': (base, 'input'),
        'size 1000': (res(size=1000), 'reservoir'),
        'size 6000': (res(size=6000), 'reservoir'),
        'seed 1': (res(seed=1), 'reservoir'),
        'reservoir + raw input': (base, 'reservoir+input'),
        'size 500': (res(size=500), 'reservoir'),
        'seed 2': (res(seed=2), 'reservoir'),
        'input scale 0.3': (res(input_scale=0.3), 'reservoir'),
        'input scale 3': (res(input_scale=3.0), 'reservoir'),
        'spectral radius 0.5': (res(spectral_radius=0.5), 'reservoir'),
        'spectral radius 0.99': (res(spectral_radius=0.99), 'reservoir'),
        'leak 0.05-1 (shorter memory)': (res(leak_rate_min=0.05), 'reservoir'),
        'leak 0.002-1 (longer memory)': (res(leak_rate_min=0.002), 'reservoir'),
        'leak 0.3-1 (fast neurons only)': (res(leak_rate_min=0.3), 'reservoir'),
        'leak 0.005-0.2 (slow neurons only)': (res(leak_rate_min=0.005, leak_rate_max=0.2), 'reservoir'),
        'patch radius 1': (res(patch_radius=1), 'reservoir'),
        'patch radius 4': (res(patch_radius=4), 'reservoir'),
        'global random input (no patches)': (res(input_structure='dense'), 'reservoir'),
        'old observation (v1)': (other_obs(base), 'reservoir'),
    }


# readout variants on the baseline features: (q_clip, critical weight, level-balanced sample weights)
READOUTS = {
    'readout: q_clip 32, critical 4': (32.0, 4.0, False),
    'readout: level-balanced weights': (32.0, 4.0, True),
    'readout: no critical weighting': (32.0, 0.0, False),
    'readout: critical 16': (32.0, 16.0, False),
    'readout: q_clip 16': (16.0, 4.0, False),
    'readout: q_clip 64': (64.0, 4.0, False),
}


def sample_weights(y: np.ndarray, tried: np.ndarray, levels: np.ndarray, cw: float, balanced: bool) -> np.ndarray:
    yt = np.where(tried, y, np.nan)
    spread = np.nan_to_num(np.nanmax(yt, axis=1) - np.nanmin(yt, axis=1))
    w = 1.0 + cw * spread
    if balanced:  # every level counts the same, however many labelled steps it has
        for lv in np.unique(levels):
            m = levels == lv
            w[m] *= len(w) / (len(np.unique(levels)) * m.sum())
    return w


def cv_scores(F: np.ndarray, y: np.ndarray, scores: np.ndarray, w: np.ndarray, fold: np.ndarray, n_folds: int,
              alphas=ALPHAS) -> dict:
    """Per alpha: held-out and in-sample q_chosen / deadly share / agreement with the teacher's choice."""
    d = F.shape[1]
    dt = np.float64 if d <= 6500 else np.float32
    G, C, N = [], [], []
    for k in range(n_folds):
        m = fold == k
        Fw = F[m] * w[m, None]
        G.append((Fw.T.astype(dt) @ F[m].astype(dt)))
        C.append(Fw.T.astype(np.float64) @ y[m].astype(np.float64))
        N.append(float(w[m].sum()))
    Gt, Ct, Nt = sum(G), sum(C), sum(N)
    best = np.where(np.isfinite(scores), scores, -np.inf).max(axis=1)
    deadly = np.nan_to_num(scores, neginf=-1e9) <= -1000
    out = {}
    preds = {a: np.zeros(len(F), np.int64) for a in alphas}
    preds_in = {a: [] for a in alphas}
    for k in range(n_folds):
        g = (Gt - G[k]).astype(np.float64) / (Nt - N[k])
        c = (Ct - C[k]) / (Nt - N[k])
        tr = float(np.trace(g)) / d
        m = fold == k
        for a in alphas:
            gp = g.copy()
            gp[np.diag_indices(d)] += a * tr
            gp[-1, -1] -= a * tr  # the bias is not penalised
            W = np.linalg.solve(gp, c).astype(np.float32)
            preds[a][m] = np.argmax(F[m] @ W, axis=1)
            mi = ~m
            pi = np.argmax(F[mi] @ W, axis=1)
            preds_in[a].append((np.mean(y[mi][np.arange(mi.sum()), pi]), np.mean(deadly[mi][np.arange(mi.sum()), pi])))
    for a in alphas:
        p = preds[a]
        idx = np.arange(len(p))
        out[str(a)] = {
            'q_chosen': round(float(np.mean(y[idx, p])), 4),
            'deadly': round(float(np.mean(deadly[idx, p])), 4),
            'regret_px': round(float(np.mean(np.minimum(best - np.where(np.isfinite(scores[idx, p]), scores[idx, p], -1e9), 64))), 2),
            'q_in_sample': round(float(np.mean([q for q, _ in preds_in[a]])), 4),
            'deadly_in_sample': round(float(np.mean([dd for _, dd in preds_in[a]])), 4),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--only', default='')
    p.add_argument('--closed-loop', type=int, default=0, help='closed-loop check of the N best configurations')
    p.add_argument('--starts', default='0,13,29')
    args = p.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((data.parent / 'results.json').read_text())
    base = run['config']
    eps = load_episodes(data)
    levels = np.concatenate([np.full(len(e['labels']), str(e['level'])) for e in eps])
    scores = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
    labels = np.concatenate([e['labels'] for e in eps])
    tried = np.isfinite(scores)
    uniq = sorted(set(levels.tolist()))
    order = np.random.default_rng(0).permutation(len(uniq))
    fold_of = {lv: int(i % args.folds) for i, lv in zip(order, uniq)}
    fold = np.array([fold_of[lv] for lv in levels])
    print(f'{len(eps)} episodes, {len(labels)} labelled steps, {len(uniq)} levels in {args.folds} folds: '
          + '; '.join(f'fold {k}: ' + ','.join(lv for lv in uniq if fold_of[lv] == k) for k in range(args.folds)), flush=True)
    path = out / 'cv.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    configs = feature_configs(base)
    names = [n for n in list(configs) + list(READOUTS) if not args.only or n in args.only.split(',')]
    base_feats = None
    for name in names:
        if name in results:
            continue
        t0 = time.perf_counter()
        if name in READOUTS:
            q_clip, cw, balanced = READOUTS[name]
            cfg, mode = configs['baseline (3000, leak 0.01-1, radius 0.9, input 1, patch 2)']
            if base_feats is None:
                attach_streams(eps, cfg, data.parent / 'streams', args.workers)
                base_feats = np.concatenate(replay_features(eps, cfg, mode))
            F = base_feats
        else:
            q_clip, cw, balanced = 32.0, 4.0, False
            cfg, mode = configs[name]
            attach_streams(eps, cfg, data.parent / 'streams', args.workers)
            F = np.concatenate(replay_features(eps, cfg, mode))
            if name.startswith('baseline'):
                base_feats = F
        y = np.stack([make_targets(s, int(a), 'q', q_clip) for s, a in zip(scores, labels)]).astype(np.float32)
        w = sample_weights(y, tried, levels, cw, balanced)
        # fit on targets with this q_clip, but always score with the q_clip-32 scale so variants are comparable
        y32 = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(scores, labels)]).astype(np.float32)
        res = cv_scores_fit_score(F, y, y32, scores, w, fold, args.folds)
        best_a = max(res, key=lambda a: res[a]['q_chosen'])
        results[name] = {'mode': mode, 'reservoir': cfg['reservoir'], 'obs': cfg['env'].get('obs', 'v1'), 'q_clip': q_clip,
                         'critical_weight': cw, 'balanced': balanced, 'per_alpha': res, 'best_alpha': float(best_a),
                         'dim': int(F.shape[1]), 'seconds': round(time.perf_counter() - t0)}
        path.write_text(json.dumps(results, indent=1))
        b = res[best_a]
        print(f"{name:58s} held-out q {b['q_chosen']:.4f} deadly {b['deadly']:.4f} regret {b['regret_px']:5.2f}px | in-sample q "
              f"{b['q_in_sample']:.4f} deadly {b['deadly_in_sample']:.4f} | alpha {best_a} ({results[name]['seconds']}s)", flush=True)
        if name not in READOUTS and not name.startswith('baseline'):
            del F
    if args.closed_loop:
        closed_loop(args, results, configs, eps, scores, labels, levels, tried, run, data, out)


def cv_scores_fit_score(F, y_fit, y_score, scores, w, fold, n_folds):
    """cv_scores, fitting on y_fit but scoring q_chosen on y_score."""
    res = cv_scores(F, y_fit, scores, w, fold, n_folds)
    if y_fit is y_score or np.array_equal(y_fit, y_score):
        return res
    # rescore with the common scale: refit is cheap enough to redo with the scoring targets swapped in afterwards
    d = F.shape[1]
    G, C, N = [], [], []
    for k in range(n_folds):
        m = fold == k
        Fw = F[m] * w[m, None]
        G.append(Fw.T.astype(np.float64) @ F[m].astype(np.float64))
        C.append(Fw.T.astype(np.float64) @ y_fit[m].astype(np.float64))
        N.append(float(w[m].sum()))
    Gt, Ct, Nt = sum(G), sum(C), sum(N)
    deadly = np.nan_to_num(scores, neginf=-1e9) <= -1000
    for a_s in res:
        a = float(a_s)
        p = np.zeros(len(F), np.int64)
        for k in range(n_folds):
            g = (Gt - G[k]) / (Nt - N[k])
            r_ = a * float(np.trace(g)) / d
            g[np.diag_indices(d)] += r_
            g[-1, -1] -= r_
            W = np.linalg.solve(g, (Ct - C[k]) / (Nt - N[k]))
            m = fold == k
            p[m] = np.argmax(F[m] @ W, axis=1)
        res[a_s]['q_chosen'] = round(float(np.mean(y_score[np.arange(len(p)), p])), 4)
        res[a_s]['deadly'] = round(float(np.mean(deadly[np.arange(len(p)), p])), 4)
    return res


def closed_loop(args, results, configs, eps, scores, labels, levels, tried, run, data, out):
    """Fits the N best configurations (by held-out q) on all data and plays groups A/B/C from several starts."""
    from smw_imitation import job, play  # noqa: PLC0415

    split = json.loads(Path('runs/smw/split_abc.json').read_text())
    specs = {'full': split['full'], 'first_half': [f'{lv}@first' for lv in split['half']],
             'second_half': [f'{lv}@second' for lv in split['half']], 'unseen': split['unseen']}
    starts = [int(s) for s in args.starts.split(',')]
    ranked = sorted((n for n in results if n in configs or n in READOUTS),
                    key=lambda n: -results[n]['per_alpha'][str(results[n]['best_alpha'])]['q_chosen'])
    chosen = ranked[:args.closed_loop]
    for must in ('baseline (3000, leak 0.01-1, radius 0.9, input 1, patch 2)', 'raw input only'):
        if must in results and must not in chosen:
            chosen.append(must)
    path = out / 'closed_loop.json'
    cl = json.loads(path.read_text()) if path.exists() else {}
    pool = mp.get_context('fork').Pool(args.workers, maxtasksperchild=1)
    for name in chosen:
        if name in cl:
            continue
        r = results[name]
        if name in READOUTS:
            cfg, mode = configs['baseline (3000, leak 0.01-1, radius 0.9, input 1, patch 2)']
        else:
            cfg, mode = configs[name]
        attach_streams(eps, cfg, data.parent / 'streams', args.workers)
        F = np.concatenate(replay_features(eps, cfg, mode))
        y = np.stack([make_targets(s, int(a), 'q', r['q_clip']) for s, a in zip(scores, labels)])
        w = sample_weights(y, tried, levels, r['critical_weight'], r['balanced'])
        Fw = F * w[:, None]
        g = Fw.T.astype(np.float64) @ F.astype(np.float64) / w.sum()
        c = Fw.T.astype(np.float64) @ y.astype(np.float64) / w.sum()
        r_ = r['best_alpha'] * float(np.trace(g)) / F.shape[1]
        g[np.diag_indices(F.shape[1])] += r_
        g[-1, -1] -= r_
        W = np.linalg.solve(g, c).astype(np.float32)
        del F, Fw
        np.save(out / f"readout_{len(cl)}.npy", W)
        rec = {'readout': f"readout_{len(cl)}.npy", 'alpha': r['best_alpha']}
        for gname, ss in specs.items():
            jobs = [job(s, cfg, mode, W, max_steps=int(run['args']['max_steps']), noop=n) for s in ss for n in starts]
            ep = pool.map(play, jobs, chunksize=1)
            per = {s: round(float(np.mean([e['progress'] for e in ep if e['level'] == s])), 3) for s in ss}
            vals = np.array([e['progress'] for e in ep])
            rec[gname] = {'progress': round(float(vals.mean()), 3), 'se': round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 3),
                          'finished': round(float(np.mean([e['reached_goal'] for e in ep])), 3), 'per_level': per}
        cl[name] = rec
        path.write_text(json.dumps(cl, indent=1))
        print(f'closed loop {name:58s} ' + '  '.join(f"{g} {rec[g]['progress']:.3f}+-{rec[g]['se']:.3f}" for g in specs), flush=True)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
