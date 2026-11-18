#!/usr/bin/env python
"""
Fits a reservoir readout by imitating a look-ahead planner (ridge regression + DAgger).

    python scripts/train_imitation.py --config configs/toy_platformer_curriculum_stage1.yaml \
        --mode reservoir --iters 4 --levels 24 --workers 2 --out runs/imitation/stage1_res

Writes ``results.json`` (per-iteration validation/test metrics, teacher ceiling) and, for the
``reservoir`` feature mode, ``best.pt``: a normal ReadoutActorCritic checkpoint with discrete
actions that ``play.py``, ``record_video.py`` and ``train.py --checkpoint`` (PPO fine-tuning) accept.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process: the machine is shared, and the workers already run in parallel.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from echomario.imitation import (
    FEATURE_MODES,
    TARGET_MODES,
    collect_worker,
    discrete_config,
    eval_worker,
    expert_worker,
    summarize,
)
from echomario.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--mode', choices=FEATURE_MODES, default='reservoir')
    p.add_argument('--target', choices=TARGET_MODES, default='q', help='onehot labels or planner-score (q) targets')
    p.add_argument('--q-clip', type=float, default=5.0, help='score gap that maps to target 0 (q targets)')
    p.add_argument('--critical-weight', type=float, default=0.0,
                   help='upweight frames where some action is fatal (weight 1 + k * target spread)')
    p.add_argument('--iters', type=int, default=4, help='DAgger iterations (the first one is pure expert data)')
    p.add_argument('--levels', type=int, default=24, help='training levels per iteration')
    p.add_argument('--beta-decay', type=float, default=0.5, help='expert mixing prob = decay**iteration')
    p.add_argument('--alphas', type=str, default='1e-4,1e-3,1e-2,1e-1,1')
    p.add_argument('--val-levels', type=int, default=20)
    p.add_argument('--test-levels', type=int, default=None, help='default: env.num_eval_seeds')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--skip-teacher', action='store_true')
    p.add_argument('--set', action='append', default=[], help='override a config value, e.g. reservoir.size=2000')
    p.add_argument('--out', required=True)
    return p.parse_args()


def apply_overrides(config: dict, overrides: list[str]) -> None:
    import yaml

    for item in overrides:
        key, value = item.split('=', 1)
        node = config
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)


def chunks(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items)))
    return [items[i::n] for i in range(n)]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.set)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(',')]

    env_cfg = config['env']
    test_start = int(env_cfg.get('eval_seed_start', 10000))
    n_test = args.test_levels or int(env_cfg.get('num_eval_seeds', 20))
    test_seeds = list(range(test_start, test_start + n_test))
    val_seeds = list(range(5000, 5000 + args.val_levels))
    train_seed_base = 100_000 + 1000 * int(config['project'].get('seed', 42))

    ctx = mp.get_context('fork')
    pool = ctx.Pool(args.workers)
    results: dict = {'config': config, 'mode': args.mode, 'args': vars(args), 'iterations': []}

    t0 = time.perf_counter()
    if not args.skip_teacher:
        teacher = sum(pool.map(expert_worker, [(config, c, None) for c in chunks(test_seeds, args.workers)]), [])
        results['teacher_test'] = summarize(teacher)
        print(f"teacher on test levels: {results['teacher_test']}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    acc = None
    weights = None
    best = {'val_success': -1.0}
    for it in range(args.iters):
        beta = args.beta_decay ** it if it > 0 else 1.0
        seeds = [train_seed_base + it * args.levels + i for i in range(args.levels)]
        jobs = [(config, args.mode, c, weights, beta, None, 17 * it + k, args.target, args.critical_weight, args.q_clip)
                for k, c in enumerate(chunks(seeds, args.workers))]
        t_it = time.perf_counter()
        parts = pool.map(collect_worker, jobs)
        collect_stats = []
        for part_acc, part_stats, _secs in parts:
            if acc is None:
                acc = part_acc
            else:
                acc.merge(part_acc)
            collect_stats += part_stats
        collect_seconds = time.perf_counter() - t_it

        # Pick the ridge strength by closed-loop success on validation levels.
        per_alpha = {}
        for alpha in alphas:
            w = acc.solve(alpha)
            val = sum(pool.map(eval_worker, [(config, args.mode, c, w) for c in chunks(val_seeds, args.workers)]), [])
            per_alpha[alpha] = (summarize(val), w)
        best_alpha = max(per_alpha, key=lambda a: (per_alpha[a][0]['success_rate'], per_alpha[a][0]['progress_mean']))
        val_summary, weights = per_alpha[best_alpha]
        test = sum(pool.map(eval_worker, [(config, args.mode, c, weights) for c in chunks(test_seeds, args.workers)]), [])
        test_summary = summarize(test)
        record = {
            'iteration': it,
            'beta': beta,
            'samples': acc.count,
            'label_distribution': (acc.label_counts / max(1, acc.count)).round(4).tolist(),
            'collect': summarize(collect_stats),
            'collect_seconds': round(collect_seconds, 1),
            'alpha': best_alpha,
            'val_by_alpha': {str(a): per_alpha[a][0] for a in alphas},
            'val': val_summary,
            'test': test_summary,
        }
        results['iterations'].append(record)
        print(
            f"iter {it}: beta={beta:.2f} samples={acc.count} alpha={best_alpha:g} "
            f"val={val_summary['success_rate']:.2f}/{val_summary['progress_mean']:.2f} "
            f"test={test_summary['success_rate']:.2f}/{test_summary['progress_mean']:.2f} "
            f"agree={record['collect'].get('expert_agreement', float('nan')):.2f} "
            f"({collect_seconds:.0f}s collect, {time.perf_counter() - t0:.0f}s total)",
            flush=True,
        )
        if val_summary['success_rate'] > best['val_success'] or (
            val_summary['success_rate'] == best['val_success'] and val_summary['progress_mean'] > best.get('val_progress', -1)
        ):
            best = {'val_success': val_summary['success_rate'], 'val_progress': val_summary['progress_mean'],
                    'iteration': it, 'alpha': best_alpha, 'test': test_summary}
            np.save(out_dir / 'readout.npy', weights)
            if args.mode == 'reservoir':
                save_readout_checkpoint(config, weights, out_dir / 'best.pt', test_summary)
        results['best'] = best
        results['seconds'] = round(time.perf_counter() - t0, 1)
        (out_dir / 'results.json').write_text(json.dumps(results, indent=2, default=float))

    pool.close()
    print(f"best (by validation): {best}")


def save_readout_checkpoint(config: dict, weights: np.ndarray, path: Path, test_summary: dict) -> None:
    """Stores the ridge readout as a discrete ReadoutActorCritic so the usual tools can load it."""
    import torch

    from echomario.agents.factory import build_policy
    from echomario.envs.make_env import make_env
    from echomario.reservoirs.esn import make_reservoir
    from echomario.utils.checkpoint import save_checkpoint

    cfg = discrete_config(config)
    env = make_env(cfg)
    reservoir = make_reservoir(cfg, input_dim=int(env.observation_space.shape[0]))
    policy = build_policy(cfg, env, reservoir.cfg.size)
    # Ridge outputs are ~[0, 1]; scale them into logits so the greedy action dominates sampling.
    scale = 20.0
    with torch.no_grad():
        policy.policy.weight.copy_(torch.from_numpy(weights[:-1].T * scale))
        policy.policy.bias.copy_(torch.from_numpy(weights[-1] * scale))
    save_checkpoint(
        path,
        config=cfg,
        policy_state=policy.state_dict(),
        reservoir_state=reservoir.state_dict(),
        global_step=0,
        stats={'kind': 'imitation_ridge', 'logit_scale': scale, 'test': test_summary},
    )
    env.close()


if __name__ == '__main__':
    main()
