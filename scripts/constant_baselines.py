#!/usr/bin/env python
"""
Scores every constant discrete action on a config's evaluation levels: the floor that a learned
agent must beat. On these levels, holding "right + run + jump" (bunny-hopping) finishes a lot of
levels with no perception at all, so success rates must be compared against it.

    python scripts/constant_baselines.py configs/imitation/gaps.yaml configs/toy_platformer_curriculum_stage1.yaml
"""

from __future__ import annotations

import argparse

from echomario.envs.make_env import DiscreteToyActionWrapper, make_env
from echomario.experts.planner import unwrap_toy
from echomario.imitation import EpisodeStats, _death_kind, discrete_config, summarize
from echomario.utils.config import load_config


def constant_action_baselines(config: dict, num_levels: int | None = None) -> dict[str, dict]:
    cfg = discrete_config(config)
    env = make_env(cfg)
    start = int(cfg['env'].get('eval_seed_start', 10000))
    n = num_levels or int(cfg['env'].get('num_eval_seeds', 20))
    out = {}
    for action, name in enumerate(DiscreteToyActionWrapper.ACTION_NAMES):
        stats = []
        for seed in range(start, start + n):
            env.reset(options={'level_seed': seed})
            steps = 0
            while True:
                _o, _r, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            base = unwrap_toy(env)
            stats.append(EpisodeStats(seed, bool(info.get('reached_goal')), base.max_x / base.goal_x, steps,
                                      _death_kind(info, truncated)))
        out[name] = summarize(stats)
    env.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('configs', nargs='+')
    p.add_argument('--levels', type=int, default=None)
    args = p.parse_args()
    for path in args.configs:
        results = constant_action_baselines(load_config(path), args.levels)
        best = max(results, key=lambda k: (results[k]['success_rate'], results[k]['progress_mean']))
        print(f'{path}: best constant action = {best} '
              f"(success {results[best]['success_rate']:.0%}, progress {results[best]['progress_mean']:.2f})")
        for name, r in results.items():
            print(f"    {name:15s} success={r['success_rate']:.2f} progress={r['progress_mean']:.2f}")


if __name__ == '__main__':
    main()
