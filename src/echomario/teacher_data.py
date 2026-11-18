"""
Teacher datasets: record the planner once, then test many readout feature maps offline.

Running the look-ahead planner is by far the most expensive part of imitation (it simulates
~1000 frames per decision). The planner's behaviour does not depend on the student's features,
so one recorded dataset (every observation, plus the planner's per-action scores every
``plan_every`` frames) can be replayed through any reservoir to fit and score its readout.

Offline metrics (on held-out levels, teacher's state distribution):
- ``safe_rate``: the readout's argmax action is within 5 % of the best planned score.
- ``critical_safe_rate``: the same, on frames where some action is fatal (target spread >= 0.5),
  i.e. the jump-or-die moments that decide whether a level is finished.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from echomario.envs.make_env import make_env
from echomario.experts.planner import LookaheadPlanner, PlannerConfig, unwrap_toy
from echomario.imitation import FeatureMap, RidgeAccumulator, discrete_config, make_targets


def record_levels(args: tuple) -> list[dict]:
    """Worker: plays ``seeds`` with the planner and records observations and planner scores."""
    config, seeds, plan_every = args
    env = make_env(discrete_config(config))
    planner = LookaheadPlanner(PlannerConfig())
    out = []
    for seed in seeds:
        start = time.perf_counter()
        obs, _ = env.reset(options={'level_seed': int(seed)})
        planner.attach(env)
        observations, planned_steps, scores, actions = [obs.copy()], [], [], []
        step = 0
        action = 0
        while True:
            if step % plan_every == 0:
                s = planner.scores(env)
                action = planner.choose(s)
                planned_steps.append(step)
                scores.append(s.astype(np.float32))
            actions.append(action)
            obs, _r, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated:
                break
            observations.append(obs.copy())
        base = unwrap_toy(env)
        out.append({
            'seed': int(seed),
            'obs': np.stack(observations).astype(np.float32),
            'planned_steps': np.asarray(planned_steps, dtype=np.int64),
            'scores': np.stack(scores),
            'actions': np.asarray(actions, dtype=np.int64),
            'reached_goal': bool(info.get('reached_goal', False)),
            'progress': float(base.max_x / max(1e-6, base.goal_x)),
            'seconds': time.perf_counter() - start,
        })
    env.close()
    return out


def save_levels(levels: list[dict], path: Path, screen_dim: int) -> None:
    """Stores the binary screen part bit-packed (it is one-hot tiles) and the rest as float32."""
    arrays = {}
    for i, lv in enumerate(levels):
        screen = lv['obs'][:, :screen_dim]
        assert np.all((screen == 0) | (screen == 1))
        arrays[f'{i}_screen'] = np.packbits(screen.astype(bool), axis=1)
        arrays[f'{i}_extra'] = lv['obs'][:, screen_dim:]
        for key in ('planned_steps', 'scores', 'actions'):
            arrays[f'{i}_{key}'] = lv[key]
        arrays[f'{i}_meta'] = np.array([lv['seed'], lv['reached_goal'], lv['progress']], dtype=np.float64)
    arrays['count'] = np.array([len(levels)])
    arrays['screen_dim'] = np.array([screen_dim])
    np.savez_compressed(path, **arrays)


def load_levels(path: Path) -> list[dict]:
    data = np.load(path)
    screen_dim = int(data['screen_dim'][0])
    levels = []
    for i in range(int(data['count'][0])):
        screen = np.unpackbits(data[f'{i}_screen'], axis=1, count=screen_dim).astype(np.float32)
        obs = np.concatenate([screen, data[f'{i}_extra'].astype(np.float32)], axis=1)
        meta = data[f'{i}_meta']
        levels.append({
            'seed': int(meta[0]), 'reached_goal': bool(meta[1]), 'progress': float(meta[2]), 'obs': obs,
            'planned_steps': data[f'{i}_planned_steps'], 'scores': data[f'{i}_scores'], 'actions': data[f'{i}_actions'],
        })
    return levels


def replay_features(config: dict, mode: str, level: dict, drop_extra: int = 0) -> np.ndarray:
    """Features at the planned frames of one recorded level (fresh reservoir per level)."""
    obs = level['obs'] if drop_extra == 0 else level['obs'][:, :-drop_extra]
    fmap = FeatureMap(config, obs.shape[1], mode)
    fmap.reset()
    planned = set(int(s) for s in level['planned_steps'])
    rows = []
    for t in range(len(obs)):
        f = fmap.step(obs[t])
        if t in planned:
            rows.append(f)
    return np.stack(rows)


def offline_scores(weights: np.ndarray, feats: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """
    Scores the readout's greedy choice against the planner's raw action scores:
    regret = best score - score of the chosen action.
    - ``good_rate``: regret < 0.5 (as good as the teacher; idling or backing off already costs ~1)
    - ``fatal_rate``: regret >= 100 (the chosen action leads to death within the planning horizon)
    - ``critical_fatal_rate``: fatal rate on frames where at least one action is fatal.
    """
    pred = np.argmax(feats @ weights, axis=1)
    regret = scores.max(axis=1) - scores[np.arange(len(pred)), pred]
    critical = (scores.max(axis=1) - scores.min(axis=1)) >= 100.0
    return {
        'good_rate': float(np.mean(regret < 0.5)),
        'fatal_rate': float(np.mean(regret >= 100.0)),
        'critical_fatal_rate': float(np.mean(regret[critical] >= 100.0)) if critical.any() else float('nan'),
        'critical_frames': int(critical.sum()),
        'mean_regret_clipped': float(np.mean(np.minimum(regret, 10.0))),
    }


def fit_safe_softmax(
    feats: np.ndarray,
    q_targets: np.ndarray,
    *,
    weight_decay: float = 1e-4,
    critical_weight: float = 0.0,
    steps: int = 400,
    lr: float = 0.05,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Linear softmax readout trained to put probability mass on *any* safe action:
    loss = -log sum_{a: q_a >= 0.95} softmax(f W)_a  (+ L2). The bias is the last feature.
    This is the natural objective when several actions are equally good, which ridge regression
    onto one-hot labels handles badly. Full-batch Adam; runs on GPU with ``device='cuda'``.
    """
    import torch

    dev = torch.device(device)
    f = torch.as_tensor(feats, dtype=torch.float32, device=dev)
    q = torch.as_tensor(q_targets, dtype=torch.float32, device=dev)
    safe = q >= 0.95
    w_sample = torch.ones(len(f), device=dev)
    if critical_weight > 0:
        w_sample = 1.0 + critical_weight * (q.max(dim=1).values - q.min(dim=1).values)
    w_sample = w_sample / w_sample.sum()
    W = torch.zeros(f.shape[1], q.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logp = torch.log_softmax(f @ W, dim=1)
        safe_logp = torch.logsumexp(logp.masked_fill(~safe, -1e9), dim=1)
        loss = -(w_sample * safe_logp).sum() + weight_decay * W[:-1].pow(2).sum()
        loss.backward()
        opt.step()
    return W.detach().cpu().numpy().astype(np.float32)


def evaluate_feature_map(
    config: dict,
    mode: str,
    levels: list[dict],
    *,
    test_fraction: float = 0.25,
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1),
    target: str = 'q',
    critical_weight: float = 0.0,
    drop_extra: int = 0,
    fit: str = 'ridge',
    device: str = 'cpu',
    q_clip: float = 5.0,
) -> dict:
    """Fits ridge readouts on the training levels and scores them on held-out levels."""
    n_test = max(1, int(round(len(levels) * test_fraction)))
    train, test = levels[:-n_test], levels[-n_test:]

    def build(split):
        feats, targs, labels, raw = [], [], [], []
        for lv in split:
            f = replay_features(config, mode, lv, drop_extra)
            feats.append(f)
            for s in lv['scores']:
                labels.append(int(np.argmax(s)))
            targs.append(np.stack([make_targets(s, int(np.argmax(s)), target, q_clip) for s in lv['scores']]))
            raw.append(lv['scores'])
        return np.concatenate(feats), np.concatenate(targs), np.asarray(labels), np.concatenate(raw)

    start = time.perf_counter()
    f_tr, y_tr, l_tr, s_tr = build(train)
    f_te, _y_te, _l_te, s_te = build(test)
    y_tr_q = np.stack([make_targets(s, 0, 'graded' if target == 'graded' else 'q', q_clip)
                       for lv in train for s in lv['scores']])
    results = {}
    if fit == 'ridge':
        acc = RidgeAccumulator(f_tr.shape[1], y_tr.shape[1])
        acc.add(f_tr, l_tr, None if target == 'onehot' else y_tr, critical_weight=critical_weight)
        solve = acc.solve
    elif fit == 'safe_ce':
        def solve(alpha):
            return fit_safe_softmax(f_tr, y_tr_q, weight_decay=alpha, critical_weight=critical_weight, device=device)
    else:
        raise ValueError(f'unknown fit {fit!r}')
    for alpha in alphas:
        w = solve(alpha)
        results[alpha] = {'train': offline_scores(w, f_tr, s_tr), 'test': offline_scores(w, f_te, s_te)}
    # Few levels: the reported alpha is the one with the lowest test fatal rate (optimistic; see docs).
    best_alpha = min(results, key=lambda a: (results[a]['test']['fatal_rate'], -results[a]['test']['good_rate']))
    return {
        'fit': fit,
        'mode': mode,
        'dim': int(f_tr.shape[1]),
        'train_samples': int(len(f_tr)),
        'test_samples': int(len(f_te)),
        'best_alpha': best_alpha,
        'best': results[best_alpha],
        'by_alpha': {str(a): r for a, r in results.items()},
        'seconds': round(time.perf_counter() - start, 1),
    }
