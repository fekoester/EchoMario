#!/usr/bin/env python
"""
How does an agent play, beyond "finished or not"? Coins, ? blocks, mushrooms, enemies stomped or
jumped over, speed, backtracking and how often it changes buttons (jitter), on a config's
evaluation levels.

    python scripts/analyze_play_style.py --config configs/imitation/full.yaml --agent teacher
    python scripts/analyze_play_style.py --config configs/imitation/gaps.yaml --agent runs/imitation/gaps_res
    python scripts/analyze_play_style.py --config configs/imitation/full.yaml --agent bunnyhop
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np

from echomario.envs.make_env import make_env
from echomario.experts.planner import ACTION_NAMES, LookaheadPlanner, PlannerConfig, unwrap_toy
from echomario.imitation import FeatureMap, discrete_config
from echomario.utils.config import load_config

RIGHT_RUN_JUMP = ACTION_NAMES.index('right_run_jump')
FPS = 30  # the env's render_fps


def play_levels(args: tuple) -> list[dict]:
    config, agent, seeds, style = args
    env = make_env(discrete_config(config))
    planner = LookaheadPlanner(PlannerConfig.named(style)) if agent == 'teacher' else None
    student = None
    if agent not in ('teacher', 'bunnyhop'):
        res = json.loads((Path(agent) / 'results.json').read_text())
        student = (FeatureMap(res['config'], int(env.observation_space.shape[0]), res['mode']),
                   np.load(Path(agent) / 'readout.npy'))
    out = []
    for seed in seeds:
        obs, _ = env.reset(options={'level_seed': seed})
        base = unwrap_toy(env)
        if planner is not None:
            planner.attach(env)
        if student is not None:
            student[0].reset()
        n_enemies = len(base.enemies)
        enemy_xs = [e.x for e in base.enemies]
        actions, xs = [], []
        stomps = jumped_over = 0
        step, action = 0, RIGHT_RUN_JUMP
        while True:
            if student is not None:
                action = int(np.argmax(student[0].step(obs) @ student[1]))
            elif planner is not None and step % 2 == 0:
                action = planner.act(env)
            airborne = not base.player.on_ground
            obs, _r, terminated, truncated, info = env.step(action)
            actions.append(action)
            xs.append(info['x'])
            stomps += int(info['stomped_enemy'])
            if airborne:
                jumped_over += int(info['enemy_pass_count'])
            step += 1
            if terminated or truncated:
                break
        reach = base.max_x
        coins_in_reach = [c for c in base.coins if c.x <= reach + 0.5]
        blocks_in_reach = [b for b in base.question_blocks if b.x <= reach + 0.5]
        mushroom_blocks = [b for b in blocks_in_reach if b.item_kind != 'coin']
        dx = np.diff([2.0] + xs)
        out.append({
            'seed': seed,
            'finished': bool(info['reached_goal']),
            'progress': reach / base.goal_x,
            'frames': step,
            'tiles': reach,
            'coins_taken': sum(c.collected for c in base.coins),
            'coins_in_reach': len(coins_in_reach),
            'blocks_hit': sum(b.used for b in blocks_in_reach),
            'blocks_in_reach': len(blocks_in_reach),
            'mushrooms_spawned': len(base.items),
            'mushroom_blocks_in_reach': len(mushroom_blocks),
            'mushrooms_collected': int(base.player.collected_items),
            'enemies_in_reach': sum(1 for x in enemy_xs if x <= reach + 0.5),
            'enemies_total': n_enemies,
            'stomps': stomps,
            'jumped_over': jumped_over,
            'hit_enemy': bool(info['hit_enemy']),
            'backtrack_tiles': float(-dx[dx < 0].sum()),
            'action_changes': int(np.sum(np.diff(actions) != 0)),
            'actions': Counter(actions),
        })
    env.close()
    return out


def summarize(levels: list[dict]) -> dict:
    tot = lambda k: sum(lv[k] for lv in levels)  # noqa: E731
    acts = Counter()
    for lv in levels:
        acts.update(lv['actions'])
    n_act = sum(acts.values())
    finished = [lv for lv in levels if lv['finished']]
    return {
        'levels': len(levels),
        'finished': round(len(finished) / len(levels), 2),
        'progress': round(float(np.mean([lv['progress'] for lv in levels])), 2),
        'coins_taken_of_reachable': f"{tot('coins_taken')}/{tot('coins_in_reach')} ({tot('coins_taken') / max(1, tot('coins_in_reach')):.0%})",
        'question_blocks_hit': f"{tot('blocks_hit')}/{tot('blocks_in_reach')} ({tot('blocks_hit') / max(1, tot('blocks_in_reach')):.0%})",
        'mushrooms_collected_of_spawned': f"{tot('mushrooms_collected')}/{tot('mushrooms_spawned')} (mushroom blocks reached: {tot('mushroom_blocks_in_reach')})",
        'enemies_reached': tot('enemies_in_reach'),
        'enemies_stomped': tot('stomps'),
        'enemies_jumped_over': tot('jumped_over'),
        'killed_by_enemy': sum(lv['hit_enemy'] for lv in levels),
        'speed_tiles_per_s_finished': round(float(np.mean([lv['tiles'] / lv['frames'] * FPS for lv in finished])), 2) if finished else None,
        'max_run_speed_tiles_per_s': round(0.33 * FPS, 1),
        'backtrack_tiles_per_100': round(100 * tot('backtrack_tiles') / tot('tiles'), 1),
        'button_changes_per_s': round(tot('action_changes') / tot('frames') * FPS, 2),
        'action_share': {ACTION_NAMES[a]: round(c / n_act, 3) for a, c in acts.most_common()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--agent', default='teacher', help="'teacher', 'bunnyhop' or an imitation run dir")
    p.add_argument('--teacher-style', default='speedrunner', help="planner style: 'speedrunner' or 'collector'")
    p.add_argument('--levels', type=int, default=20)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    config = load_config(args.config)
    start = int(config['env'].get('eval_seed_start', 10000))
    seeds = list(range(start, start + args.levels))
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    with mp.get_context('fork').Pool(args.workers) as pool:
        levels = sum(pool.map(play_levels, [(config, args.agent, c, args.teacher_style) for c in chunks]), [])
    summary = summarize(levels)
    print(json.dumps({'config': args.config, 'agent': args.agent, 'teacher_style': args.teacher_style, **summary}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({'summary': summary, 'levels': levels}, indent=1, default=str))


if __name__ == '__main__':
    main()
