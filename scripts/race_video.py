#!/usr/bin/env python
"""
Plays several agents on the same level and stacks their views into one video, with a live
counter of coins, ? blocks, stomps and outcome per agent. Agents: 'teacher:<style>', 'bunnyhop'
or an imitation run directory (optionally 'label=agent').

    python scripts/race_video.py --config configs/imitation/gaps.yaml --seed 10003 \
        --agent "collector teacher=teacher:collector" \
        --agent "student of collector=runs/imitation/gaps_res3000_collector" \
        --agent "student of speedrunner=runs/imitation/gaps_res_local_multi3000_c5_long" \
        --out outputs/race.mp4
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from echomario.envs.make_env import make_env
from echomario.experts.planner import ACTION_NAMES, LookaheadPlanner, PlannerConfig, unwrap_toy
from echomario.imitation import FeatureMap, discrete_config
from echomario.utils.config import load_config

RIGHT_RUN_JUMP = ACTION_NAMES.index('right_run_jump')
BAR = 34


def play(config: dict, agent: str, seed: int, scale: float) -> list[np.ndarray]:
    """Returns the rendered frames (RGB) with a status bar on top."""
    env = make_env(discrete_config(config))
    obs, _ = env.reset(options={'level_seed': seed})
    base = unwrap_toy(env)
    planner = student = None
    if agent.startswith('teacher'):
        planner = LookaheadPlanner(PlannerConfig.named(agent.split(':', 1)[1] if ':' in agent else 'speedrunner'))
        planner.attach(env)
    elif agent != 'bunnyhop':
        res = json.loads((Path(agent) / 'results.json').read_text())
        student = (FeatureMap(res['config'], int(env.observation_space.shape[0]), res['mode']),
                   np.load(Path(agent) / 'readout.npy'))
        student[0].reset()
    frames, step, action, stomps, status = [], 0, RIGHT_RUN_JUMP, 0, ''
    while True:
        if student is not None:
            action = int(np.argmax(student[0].step(obs) @ student[1]))
        elif planner is not None and step % 2 == 0:
            action = planner.act(env)
        frames.append(_with_bar(env.render(), scale, base, stomps, status, ACTION_NAMES[action]))
        obs, _r, terminated, truncated, info = env.step(action)
        stomps += int(info['stomped_enemy'])
        step += 1
        if terminated or truncated:
            if info.get('reached_goal'):
                status = f'FINISHED in {step / 30:.1f}s'
            elif info.get('hit_enemy'):
                status = 'hit an enemy'
            elif truncated or info.get('stagnated'):
                status = 'stuck / out of time'
            else:
                status = 'fell'
            frames.append(_with_bar(env.render(), scale, base, stomps, status, ACTION_NAMES[action]))
            break
    env.close()
    return frames


def _with_bar(frame: np.ndarray, scale: float, base, stomps: int, status: str, action: str) -> np.ndarray:
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bar = np.full((BAR, frame.shape[1], 3), 24, np.uint8)
    coins = sum(c.collected for c in base.coins)
    blocks = sum(b.used for b in base.question_blocks)
    text = f'coins {coins:2d}   ? blocks {blocks}   mushrooms {int(base.player.collected_items)}   progress {base.max_x / base.goal_x:4.0%}'
    cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    right = status or action.replace('_', '+')
    colour = (120, 220, 120) if status.startswith('FIN') else (240, 120, 120) if status else (170, 170, 170)
    (w, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(bar, right, (frame.shape[1] - w - 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return np.concatenate([bar, frame], axis=0)


def label_strip(label: str, width: int) -> np.ndarray:
    strip = np.full((26, width, 3), 48, np.uint8)
    cv2.putText(strip, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 90), 1, cv2.LINE_AA)
    return strip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--agent', action='append', required=True, help="'label=agent' or 'agent'")
    p.add_argument('--scale', type=float, default=0.75)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    config = load_config(args.config)
    runs = []
    for spec in args.agent:
        label, agent = spec.split('=', 1) if '=' in spec else (spec, spec)
        frames = play(config, agent, args.seed, args.scale)
        print(f'{label}: {len(frames)} frames')
        runs.append((label, frames))
    n = max(len(f) for _, f in runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix('.raw.mp4')
    h = sum(26 + f[0].shape[0] for _, f in runs)
    w = runs[0][1][0].shape[1]
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    for t in range(n + args.fps):  # hold the final frame for a second
        panels = []
        for label, frames in runs:
            panels += [label_strip(label, w), frames[min(t, len(frames) - 1)]]
        writer.write(cv2.cvtColor(np.concatenate(panels, axis=0), cv2.COLOR_RGB2BGR))
    writer.release()
    if shutil.which('ffmpeg'):  # re-encode to H.264 so browsers and phones play it
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '23', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', str(out)], check=True)
        raw.unlink()
    else:
        raw.rename(out)
    print(f'wrote {out}: {n} frames, {w}x{h}')


if __name__ == '__main__':
    main()
