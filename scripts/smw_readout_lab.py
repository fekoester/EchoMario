#!/usr/bin/env python
"""
Offline readout experiments on Super Mario World DAgger data saved by
``smw_imitation.py --save-data``: replays the recorded input streams through any reservoir / feature
mode, refits the readout with other targets or regularisation, scores it offline against the teacher's
labels, and then plays the fitted readout closed-loop (no teacher needed, so this is fast).

    python scripts/smw_readout_lab.py --data runs/smw/tall_data/data --out runs/smw/lab

The data come from one training run's student, so every variant is judged on the states that
student visited; the closed-loop numbers are the ones that count.
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import copy
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from echomario.envs.smw_env import ACTION_NAMES
from echomario.imitation import FeatureMap, RidgeAccumulator, make_targets

N_ACT = len(ACTION_NAMES)


def load_episodes(data_dir: Path) -> list[dict]:
    """Saved episodes. Old format: the input stream itself ('stream', 'rewinds'). New format: a replay log
    ('steps', 'actions', 'noop'); its observations are rebuilt by attach_streams()."""
    eps = []
    for f in sorted(data_dir.glob('iter*_*.npz')):
        it, level = re.match(r'iter(\d+)_(.+)\.npz', f.name).groups()
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
            # new logs store the base level and the spec ('Level@first'); old ones only the stream
            d['level'] = str(d['level']) if 'level' in d else level
            d['spec'] = str(d.get('spec', level))
            eps.append({'iter': int(it), 'file': f.stem, **d})
    return eps


def _replay_job(a):
    from echomario.envs.smw_env import replay_observations  # noqa: PLC0415

    log, env_kwargs, path = a
    obs = replay_observations(log, **env_kwargs)
    sparse.save_npz(path, sparse.csr_matrix(obs))
    return path


def attach_streams(eps: list[dict], config: dict, cache: Path, workers: int) -> None:
    """Gives every log-format episode the observation stream of ``config``'s encoding (env.obs, env.grid),
    rebuilt by replaying its log in the emulator once and cached as a sparse matrix; plus the rewinds as
    (position, position whose pre-step state is restored)."""
    from echomario.envs.smw_env import grid_kwargs  # noqa: PLC0415

    kw = grid_kwargs(config)
    key = '{obs}_{rows}x{cols}_{rows_above}_{cols_behind}'.format(**kw)
    d = cache / key
    d.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eps if 'steps' in e and not (d / f"{e['file']}.npz").exists()]
    if todo:
        with mp.get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
            pool.map(_replay_job, [({k: e[k] for k in ('level', 'noop', 'steps', 'actions', 'prefix', 'end_fraction') if k in e},
                                    kw, d / f"{e['file']}.npz")
                                   for e in todo], chunksize=1)
    for e in eps:
        if 'steps' not in e:
            continue
        e['stream'] = sparse.load_npz(d / f"{e['file']}.npz").tocsr()
        steps = e['steps'].tolist()
        first: dict[int, int] = {}
        rew = []
        for p, k in enumerate(steps):
            if p and k != steps[p - 1] + 1:
                rew.append((p, first[k]))
            first.setdefault(k, p)
        e['rewinds'] = np.asarray(rew, np.int64).reshape(-1, 2)


def replay_features(eps: list[dict], config: dict, mode: str) -> list[np.ndarray]:
    """Feature vectors at every labelled step, exactly as the student computed them during collection:
    all episodes run as one batch, rewinds restore the reservoir state recorded before that input."""
    obs_dim = eps[0]['stream'].shape[1]
    fm = FeatureMap(config, obs_dim, mode)
    n = len(eps)
    res = fm.reservoir
    if res is not None:
        res.reset(batch_size=n)
    T = max(e['stream'].shape[0] for e in eps)
    targets = [set(e['rewinds'][:, 1].tolist()) for e in eps]
    rewind_at = [dict(map(tuple, e['rewinds'].tolist())) for e in eps]
    sample_at = [dict(zip(e['sample_pos'].tolist(), range(len(e['sample_pos'])))) for e in eps]
    saved = [dict() for _ in eps]
    out = [np.zeros((len(e['sample_pos']), fm.dim), np.float32) for e in eps]
    u = np.zeros((n, obs_dim), np.float32)
    for t in range(T):
        for i, e in enumerate(eps):
            if t < e['stream'].shape[0]:
                st = e['stream']
                u[i] = st[t].toarray()[0] if sparse.issparse(st) else st[t]
                if res is not None:
                    if t in rewind_at[i]:
                        res.x[i] = saved[i][rewind_at[i][t]]
                    if t in targets[i]:
                        saved[i][t] = res.x[i].copy()
            else:
                u[i] = 0.0
        parts = []
        if res is not None:
            parts.append(res.step_batch(u))
        if mode != 'reservoir':
            parts.append(u)
        parts.append(np.ones((n, 1), np.float32))
        f = np.concatenate(parts, axis=1)
        for i in range(n):
            j = sample_at[i].get(t)
            if j is not None:
                out[i][j] = f[i]
    return out


def regret_of(scores: np.ndarray, actions: np.ndarray, cap: float = 2000.0) -> np.ndarray:
    s = np.where(np.isfinite(scores), scores, scores[np.isfinite(scores)].min() if np.isfinite(scores).any() else 0.0)
    best = s.max(axis=1)
    return np.minimum(best - s[np.arange(len(s)), actions], cap)


def offline_metrics(w: np.ndarray, feats: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> dict:
    pred = np.argmax(feats @ w, axis=1)
    reg = regret_of(scores, pred)
    return {'agreement': round(float(np.mean(pred == labels)), 3),
            'regret_px': round(float(np.mean(np.minimum(reg, 64))), 2),
            'fatal': round(float(np.mean(reg >= 1000)), 4)}


def fit_softmax(feats, scores, labels, q_clip: float, cw: float, alphas: list[float]) -> dict[float, np.ndarray]:
    """Multinomial logistic regression (no closed form: L-BFGS). Soft targets: the teacher's q targets
    normalised to a distribution, so equally good actions share the probability. Same sample weights
    and the same relative L2 penalty as the ridge readout; the bias is not penalised."""
    from scipy.optimize import minimize  # noqa: PLC0415

    f = feats.astype(np.float64)
    q = np.stack([make_targets(s, int(a), 'q', q_clip) for s, a in zip(scores, labels)]).astype(np.float64)
    p = q / q.sum(axis=1, keepdims=True)
    sw = 1.0 + cw * (q.max(axis=1) - q.min(axis=1))
    sw /= sw.sum()
    energy = float(np.mean(np.sum(f[:, :-1] ** 2, axis=1))) / (f.shape[1] - 1)
    mask = np.ones((f.shape[1], 1))
    mask[-1] = 0.0
    out, w0 = {}, np.zeros(f.shape[1] * N_ACT)
    for alpha in sorted(alphas, reverse=True):  # strong to weak penalty, warm-started
        lam = alpha * energy

        def loss(wf):
            W = wf.reshape(f.shape[1], N_ACT)
            z = f @ W
            z -= z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
            logp = z - lse
            val = -np.sum(sw[:, None] * p * logp) + 0.5 * lam * np.sum(mask * W ** 2)
            g = f.T @ (sw[:, None] * (np.exp(logp) - p)) + lam * mask * W
            return val, g.ravel()

        r = minimize(loss, w0, jac=True, method='L-BFGS-B', options={'maxiter': 500})
        w0 = r.x
        out[alpha] = r.x.reshape(f.shape[1], N_ACT).astype(np.float32)
    return out


def fit(feats, scores, labels, target: str, q_clip: float, cw: float, alphas: list[float],
        tried_only: bool = False) -> dict[float, np.ndarray]:
    if target == 'softmax':
        return fit_softmax(feats, scores, labels, q_clip, cw, alphas)
    acc = RidgeAccumulator(feats.shape[1], N_ACT)
    y = np.stack([make_targets(s, int(a), target, q_clip) for s, a in zip(scores, labels)])
    tried = np.isfinite(scores) if tried_only else None
    for i in range(0, len(feats), 4096):
        acc.add(feats[i:i + 4096], labels[i:i + 4096], y[i:i + 4096], critical_weight=cw,
                tried=None if tried is None else tried[i:i + 4096])
    return {a: acc.solve(a) for a in alphas}


# feature variants: changes to the reservoir config / feature mode (readout: q targets, q_clip 32, cw 4)
def feature_variants(base: dict) -> dict[str, tuple[dict, str]]:
    def res(**kw):
        c = copy.deepcopy(base)
        c['reservoir'].update(kw)
        return c

    return {
        'baseline: reservoir 3000': (base, 'reservoir'),
        'raw input only (no reservoir)': (base, 'input'),
        'reservoir + raw input': (base, 'reservoir+input'),
        'reservoir seed 1': (res(seed=1), 'reservoir'),
        'reservoir seed 2': (res(seed=2), 'reservoir'),
        'reservoir 6000': (res(size=6000), 'reservoir'),
        # memory sweep: leak rates log-uniform in [min, max]; a smaller min means more slow neurons
        'memory: leaks 0.02-1': (res(leak_rate_min=0.02), 'reservoir'),
        'memory: leaks 0.01-1': (res(leak_rate_min=0.01), 'reservoir'),
        'memory: leaks 0.005-1': (res(leak_rate_min=0.005), 'reservoir'),
        'memory: leaks 0.002-1': (res(leak_rate_min=0.002), 'reservoir'),
        'memory: leaks 0.005-1, radius 0.99': (res(leak_rate_min=0.005, spectral_radius=0.99), 'reservoir'),
        'memory: slow only, leaks 0.005-0.2': (res(leak_rate_min=0.005, leak_rate_max=0.2), 'reservoir'),
        'memory: spectral radius 0.5': (res(spectral_radius=0.5), 'reservoir'),
        'reservoir (leaks 0.005-1) + raw input': (res(leak_rate_min=0.005), 'reservoir+input'),
        # same states and labels, the other observation encoding (only for runs saved as replay logs)
        'other observation encoding': (other_obs(base), 'reservoir'),
    }


def other_obs(base: dict) -> dict:
    c = copy.deepcopy(base)
    c.setdefault('env', {})['obs'] = 'v1' if c['env'].get('obs', 'v1') == 'v2' else 'v2'
    return c


def memory_profile(config: dict, stream: np.ndarray, t0: int = 60, lag: int = 30) -> dict:
    """How long the reservoir remembers one input frame: two copies see the same inputs except at
    step t0, where one sees the frame from `lag` steps earlier. Relative state difference afterwards."""
    fm = FeatureMap(config, stream.shape[1], 'reservoir')
    res = fm.reservoir
    res.reset(batch_size=2)
    diffs = []
    for t in range(min(len(stream), t0 + 121)):
        u = np.stack([stream[t], stream[t]]).astype(np.float32)
        if t == t0:
            u[1] = stream[t0 - lag]
        x = res.step_batch(u)
        if t >= t0:
            diffs.append(float(np.linalg.norm(x[0] - x[1]) / (np.linalg.norm(x[0]) + 1e-9)))
    d = np.array(diffs)
    return {f'{k / 30:.1f}s': round(float(d[k] / d[0]), 4) for k in (15, 30, 60, 90, 120) if k < len(d)}


# readout variants on the baseline features: (target, q_clip, critical weight, student hold)
READOUT_VARIANTS = {
    'q_clip 8': ('q', 8.0, 4.0, 4),
    'q_clip 16': ('q', 16.0, 4.0, 4),
    'q_clip 64': ('q', 64.0, 4.0, 4),
    'q_clip 128': ('q', 128.0, 4.0, 4),
    'no critical weighting': ('q', 32.0, 0.0, 4),
    'critical weight 16': ('q', 32.0, 16.0, 4),
    # the two above measure criticality over all 8 actions; the teacher never tries 2 of them, so every
    # frame got the same weight and the weighting did nothing. These use the tried actions only:
    'critical weight 4, tried actions': ('q', 32.0, 4.0, 4),
    'critical weight 16, tried actions': ('q', 32.0, 16.0, 4),
    'critical weight 64, tried actions': ('q', 32.0, 64.0, 4),
    'one-hot targets': ('onehot', 32.0, 4.0, 4),
    'softmax / logistic readout': ('softmax', 32.0, 4.0, 4),
    'graded targets (death-aware)': ('graded', 32.0, 4.0, 4),
    'student holds 2 steps': ('q', 32.0, 4.0, 2),
    'student holds 8 steps': ('q', 32.0, 4.0, 8),
}


def main() -> None:
    from smw_imitation import job, play  # noqa: PLC0415  (scripts dir is on sys.path)

    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='<run>/data written by smw_imitation.py --save-data')
    p.add_argument('--out', required=True)
    p.add_argument('--alphas', default='1e-4,1e-3,1e-2,1e-1,1')
    p.add_argument('--val-from-iter', type=int, default=-1,
                   help='offline metrics: fit on earlier rounds, score on rounds >= this (default: the last round)')
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--only', default='', help='comma-separated variant names to run (default all)')
    p.add_argument('--critic', action='store_true', help='critic analysis: how well readouts predict the teacher scores')
    p.add_argument('--reeval', action='store_true',
                   help='re-evaluate the saved readouts of lab.json with several start delays per level')
    p.add_argument('--starts', default='0,7,13,19,29', help='--reeval: idle steps before play (one episode each)')
    args = p.parse_args()

    data = Path(args.data)
    run = json.loads((data.parent / 'results.json').read_text())
    base, train_levels, test_levels = run['config'], run['args']['train_levels'].split(','), run['args']['test_levels'].split(',')
    max_steps = int(run['args']['max_steps'])
    alphas = [float(a) for a in args.alphas.split(',')]
    eps = load_episodes(data)
    scores = np.concatenate([e['scores'] for e in eps])
    labels = np.concatenate([e['labels'] for e in eps])
    val_from = args.val_from_iter if args.val_from_iter >= 0 else max(e['iter'] for e in eps)
    val = np.concatenate([np.full(len(e['labels']), e['iter'] >= val_from) for e in eps])
    print(f'{len(eps)} episodes, {len(labels)} labelled steps ({val.sum()} for offline validation)', flush=True)

    variants = {}  # name -> (config, mode, target, q_clip, cw, hold)
    for name, (cfg, mode) in feature_variants(base).items():
        variants[name] = (cfg, mode, 'q', 32.0, 4.0, 4)
    for name, (target, qc, cw, hold) in READOUT_VARIANTS.items():
        variants[name] = (base, 'reservoir', target, qc, cw, hold)
    if args.only:
        keep = args.only.split(',')
        variants = {k: v for k, v in variants.items() if k in keep}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.reeval:
        reeval(args, out, variants, train_levels, test_levels, max_steps, play, job)
        return
    if args.critic:
        results = {}
        for name in [n for n in variants if variants[n][2] == 'q' and variants[n][5] == 4 and 'critical' not in n
                     and not n.startswith('q_clip')]:
            cfg, mode = variants[name][0], variants[name][1]
            attach_streams(eps, cfg, data.parent / 'streams', args.workers)
            feats = np.concatenate(replay_features(eps, cfg, mode))
            results[name] = {str(a): critic_metrics(feats, scores, val, a) for a in alphas}
            best = max(results[name].values(), key=lambda r: r['r2_action_scores'])
            print(f'{name:40s} ' + '  '.join(f'{k} {v}' for k, v in best.items()), flush=True)
            (out / 'critic.json').write_text(json.dumps(results, indent=1))
        return
    results_path = out / 'lab.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    feat_cache: dict[str, np.ndarray] = {}
    pool = mp.get_context('fork').Pool(args.workers, maxtasksperchild=1)
    logs = all('steps' in e for e in eps)
    for name, (cfg, mode, target, qc, cw, hold) in variants.items():
        if name in results:
            print(f'skip {name} (done)', flush=True)
            continue
        if name == 'other observation encoding' and not logs:
            continue  # needs replay logs: old-format data only has the stream of one encoding
        t0 = time.perf_counter()
        key = json.dumps([cfg, mode], sort_keys=True)
        if key not in feat_cache:
            feat_cache.clear()  # keep one feature set in memory
            attach_streams(eps, cfg, data.parent / 'streams', args.workers)
            feat_cache[key] = np.concatenate(replay_features(eps, cfg, mode))
        feats = feat_cache[key]
        tried_only = 'tried actions' in name
        off = fit(feats[~val], scores[~val], labels[~val], target, qc, cw, alphas, tried_only)
        offline = {str(a): offline_metrics(w, feats[val], scores[val], labels[val]) for a, w in off.items()}
        full = fit(feats, scores, labels, target, qc, cw, alphas, tried_only)
        jobs = [job(lv, cfg, mode, w, plan_every=hold, max_steps=max_steps) for a, w in full.items() for lv in train_levels]
        eps_train = pool.map(play, jobs, chunksize=1)
        per_alpha = {}
        for k, a in enumerate(alphas):
            chunk = eps_train[k * len(train_levels):(k + 1) * len(train_levels)]
            per_alpha[str(a)] = {'progress': round(float(np.mean([e['progress'] for e in chunk])), 3),
                                 'finished': int(sum(e['reached_goal'] for e in chunk)),
                                 'per_level': {e['level']: round(e['progress'], 2) for e in chunk}}
        best_a = max(alphas, key=lambda a: per_alpha[str(a)]['progress'])
        ev = pool.map(play, [job(lv, cfg, mode, full[best_a], plan_every=hold, max_steps=max_steps) for lv in test_levels],
                      chunksize=1)
        test = {'progress': round(float(np.mean([e['progress'] for e in ev])), 3),
                'finished': int(sum(e['reached_goal'] for e in ev)),
                'per_level': {e['level']: round(e['progress'], 2) for e in ev}}
        long_ep = max(eps, key=lambda e: e['stream'].shape[0])
        st = long_ep['stream']
        mem = memory_profile(cfg, st.toarray() if sparse.issparse(st) else st) if mode != 'input' else {}
        results[name] = {'mode': mode, 'reservoir': cfg['reservoir'], 'memory': mem, 'target': target, 'q_clip': qc, 'critical_weight': cw,
                         'hold': hold, 'offline': offline, 'train_per_alpha': per_alpha, 'best_alpha': best_a,
                         'train': per_alpha[str(best_a)]['progress'],
                         'train_mean_over_alphas': round(float(np.mean([v['progress'] for v in per_alpha.values()])), 3),
                         'test': test['progress'], 'test_detail': test, 'seconds': round(time.perf_counter() - t0)}
        np.save(out / f"readout_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}.npy", full[best_a])
        results_path.write_text(json.dumps(results, indent=1))
        o = offline[str(best_a)]
        print(f"{name:32s} train {results[name]['train']:.3f} (alpha {best_a:g}, mean over alphas "
              f"{results[name]['train_mean_over_alphas']:.3f})  test {test['progress']:.3f}  offline agree {o['agreement']:.3f} "
              f"regret {o['regret_px']:.1f}px fatal {o['fatal']:.3f}  memory {mem}  ({results[name]['seconds']}s)", flush=True)
    pool.close()
    pool.join()


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a random positive gets a higher score than a random negative (ties count half)."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def critic_metrics(feats, scores, val, alpha: float, death_value: float = -200.0) -> dict:
    """A second readout, fitted to the teacher's raw score of every action it tried (pixels of progress;
    deaths clipped to death_value), one ridge regression per action. Scored on the validation rounds."""
    tried = np.isfinite(scores)
    y = np.where(tried, np.maximum(np.nan_to_num(scores, neginf=death_value), death_value), 0.0)
    f = feats.astype(np.float64)
    pred = np.zeros_like(y)
    for a in range(scores.shape[1]):
        m = tried[:, a] & ~val
        if m.sum() < 10:
            continue
        ft = f[m]
        gram = ft.T @ ft / len(ft)
        reg = alpha * float(np.trace(gram)) / gram.shape[0]
        pen = np.full(gram.shape[0], reg)
        pen[-1] = 0.0
        wa = np.linalg.solve(gram + np.diag(pen), ft.T @ y[m, a] / len(ft))
        pred[:, a] = f @ wa
    t, p_, tv = y[val], pred[val], tried[val]
    yt, pt = t[tv], p_[tv]
    r2 = 1.0 - float(np.mean((yt - pt) ** 2) / np.var(yt))
    deadly = (np.nan_to_num(scores[val], neginf=-1e9) <= -1000)[tv]
    best_true = np.where(tv, t, -np.inf).max(axis=1)
    pick = np.argmax(np.where(tv, p_, -np.inf), axis=1)
    top_ok = t[np.arange(len(t)), pick] >= best_true - 8.0  # the critic's favourite is (within 8 px of) the best
    spear = []
    for i in range(len(t)):
        k = tv[i]
        if k.sum() >= 3 and np.ptp(t[i, k]) > 0:
            spear.append(np.corrcoef(np.argsort(np.argsort(t[i, k])), np.argsort(np.argsort(p_[i, k])))[0, 1])
    v_true, v_pred = best_true, np.where(tv, p_, -np.inf).max(axis=1)
    return {'r2_action_scores': round(r2, 3), 'death_auc': round(_auc(-pt, deadly), 3), 'deadly_share': round(float(deadly.mean()), 3),
            'favourite_is_best': round(float(top_ok.mean()), 3), 'rank_corr': round(float(np.nanmean(spear)), 3),
            'r2_state_value': round(1.0 - float(np.mean((v_true - v_pred) ** 2) / np.var(v_true)), 3)}


def reeval(args, out: Path, variants: dict, train_levels: list[str], test_levels: list[str], max_steps: int, play,
           job) -> None:
    """Each saved readout (best alpha of the first pass) on every level from several start delays."""
    lab = json.loads((out / 'lab.json').read_text())
    starts = [int(x) for x in args.starts.split(',')]
    path = out / 'reeval.json'
    results = json.loads(path.read_text()) if path.exists() else {}
    pool = mp.get_context('fork').Pool(args.workers, maxtasksperchild=1)
    for name, (cfg, mode, target, qc, cw, hold) in variants.items():
        if name not in lab or name in results:
            continue
        t0 = time.perf_counter()
        w = np.load(out / f"readout_{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}.npy")
        rec = {}
        for split, levels in (('train', train_levels), ('test', test_levels)):
            jobs = [job(lv, cfg, mode, w, plan_every=hold, max_steps=max_steps, noop=n) for lv in levels for n in starts]
            eps = pool.map(play, jobs, chunksize=1)
            prog = np.array([e['progress'] for e in eps])
            per_level = {lv: round(float(np.mean([e['progress'] for e in eps if e['level'] == lv])), 3) for lv in levels}
            # standard error over levels of the per-level means (levels are the independent units)
            lv_means = np.array(list(per_level.values()))
            rec[split] = {'progress': round(float(prog.mean()), 3), 'se': round(float(lv_means.std(ddof=1) / np.sqrt(len(lv_means))), 3),
                          'se_episodes': round(float(prog.std(ddof=1) / np.sqrt(len(prog))), 3),
                          'finished': round(float(np.mean([e['reached_goal'] for e in eps])), 3), 'per_level': per_level}
        rec['seconds'] = round(time.perf_counter() - t0)
        results[name] = rec
        path.write_text(json.dumps(results, indent=1))
        print(f"{name:40s} train {rec['train']['progress']:.3f} +- {rec['train']['se_episodes']:.3f}   "
              f"test {rec['test']['progress']:.3f} +- {rec['test']['se_episodes']:.3f}  ({rec['seconds']}s)", flush=True)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
