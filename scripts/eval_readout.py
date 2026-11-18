#!/usr/bin/env python
"""
Evaluates a readout trained by train_imitation.py, optionally recording a GIF of one level.

    python scripts/eval_readout.py runs/imitation/gaps_res_local_multi --levels 40 --hold 1,2
    python scripts/eval_readout.py runs/imitation/gaps_res_local_multi --gif docs/media/imitation_gaps.gif --gif-seed 10003
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from echomario.envs.make_env import make_env
from echomario.imitation import FeatureMap, discrete_config, eval_worker, summarize


def record_gif(config: dict, mode: str, weights: np.ndarray, seed: int, out: Path, hold: int, fps: int, scale: float):
    import cv2
    from PIL import Image

    env = make_env(discrete_config(config))
    features = FeatureMap(config, int(env.observation_space.shape[0]), mode)
    obs, _ = env.reset(options={'level_seed': seed})
    features.reset()
    frames, step, action = [], 0, 0
    while True:
        f = features.step(obs)
        if step % hold == 0:
            action = int(np.argmax(f @ weights))
        frame = env.render()
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if step % 2 == 0:  # every other frame keeps the GIF small
            frames.append(Image.fromarray(frame))
        obs, _r, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(2000 / fps), loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, reached_goal={info.get('reached_goal')}, x={info.get('x'):.1f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--levels', type=int, default=40)
    p.add_argument('--seed-start', type=int, default=None, help='default: env.eval_seed_start')
    p.add_argument('--hold', default='1', help='comma list of student action-hold lengths to compare')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--gif', default=None)
    p.add_argument('--gif-seed', type=int, default=None)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--scale', type=float, default=0.5)
    args = p.parse_args()

    run = Path(args.run_dir)
    res = json.loads((run / 'results.json').read_text())
    config, mode = res['config'], res['mode']
    weights = np.load(run / 'readout.npy')
    start = args.seed_start if args.seed_start is not None else int(config['env'].get('eval_seed_start', 10000))
    seeds = list(range(start, start + args.levels))

    if args.gif:
        record_gif(config, mode, weights, args.gif_seed if args.gif_seed is not None else start,
                   Path(args.gif), int(args.hold.split(',')[0]), args.fps, args.scale)
        return

    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    with mp.get_context('fork').Pool(args.workers) as pool:
        for hold in (int(h) for h in args.hold.split(',')):
            stats = sum(pool.map(eval_worker, [(config, mode, c, weights, hold) for c in chunks]), [])
            summary = summarize(stats)
            wins = [s.level_seed for s in stats if s.reached_goal]
            print(json.dumps({'hold': hold, **summary, 'finished_seeds': wins[:12]}))


if __name__ == '__main__':
    main()
