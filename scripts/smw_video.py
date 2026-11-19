#!/usr/bin/env python
"""
Video of the reservoir student (or the teacher) playing Super Mario World, with what is going on
inside: the game, the tile grid the reservoir reads from RAM, the 3000 reservoir neurons, a
trace of the most influential neurons and the readout's score for each button combination.

    python scripts/smw_video.py --run runs/smw/res3000 --level YoshiIsland2 --out outputs/smw_YoshiIsland2.mp4
    python scripts/smw_video.py --teacher --level YoshiIsland2 --out outputs/smw_teacher.mp4
"""

from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import json
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from echomario.envs.smw_env import ACTION_NAMES, CHANNEL_NAMES, COLS, COLS_BEHIND, OBS_DIM, ROWS, ROWS_ABOVE, SMWEnv
from echomario.experts.smw_planner import SMWPlanner
from echomario.imitation import FeatureMap

W, H = 1280, 720
GAME_SCALE = 2  # 256x224 -> 512x448
BG = (18, 18, 24)
FG = (230, 230, 230)
DIM = (140, 140, 150)
ACCENT = (90, 200, 255)  # BGR-free: all colours here are RGB, converted at write time
CH_COLOURS = {'solid': (150, 105, 60), 'platform': (205, 165, 90), 'coin': (255, 215, 0),
              'enemy': (235, 70, 70), 'item': (90, 220, 120)}
PRETTY = {'idle': 'nothing', 'right': 'right', 'right_run': 'right + run', 'jump': 'jump',
          'right_jump': 'right + jump', 'right_run_jump': 'right + run + jump', 'left': 'left',
          'left_jump': 'left + jump'}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img, s, xy, scale=0.5, colour=FG, thick=1):
    cv2.putText(img, s, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def diverging(v: np.ndarray) -> np.ndarray:
    """values in [-1, 1] -> RGB, blue negative, dark zero, orange positive."""
    v = np.clip(v, -1, 1)[..., None]
    pos = np.array([255, 150, 40], np.float32)
    neg = np.array([60, 140, 255], np.float32)
    base = np.array([22, 22, 30], np.float32)
    out = np.where(v >= 0, base + v * (pos - base), base + (-v) * (neg - base))
    return out.astype(np.uint8)


def draw_grid(img, grid: np.ndarray, x0: int, y0: int, cell: int):
    for r in range(ROWS):
        for c in range(COLS):
            colour = (34, 34, 44)
            for k, name in enumerate(CHANNEL_NAMES):
                if grid[r, c, k] > 0:
                    colour = CH_COLOURS[name]
            cv2.rectangle(img, (x0 + c * cell, y0 + r * cell), (x0 + (c + 1) * cell - 2, y0 + (r + 1) * cell - 2), colour, -1)
    mx, my = x0 + COLS_BEHIND * cell, y0 + (ROWS_ABOVE - 1) * cell
    cv2.rectangle(img, (mx, my), (mx + cell - 2, my + 2 * cell - 2), (240, 40, 40), 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', help='run dir of scripts/smw_imitation.py')
    p.add_argument('--readout', default='readout.npy')
    p.add_argument('--teacher', action='store_true', help='play the look-ahead teacher instead (reservoir only watches)')
    p.add_argument('--replay', help="actions JSON written by an earlier --teacher video (no re-planning)")
    p.add_argument('--level', default='YoshiIsland2')
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    run = Path(args.run) if args.run else None
    res = json.loads((run / 'results.json').read_text()) if run else None
    if res is None:
        from smw_imitation import reservoir_config  # noqa: PLC0415  (scripts dir is on sys.path)

        config, mode = reservoir_config(3000, 0), 'reservoir'
    else:
        config, mode = res['config'], res['mode']
    weights = np.load(run / args.readout) if run else None
    trained_on = set(res['args']['train_levels'].split(',')) if res else set()

    env = SMWEnv(args.level, max_steps=args.max_steps)
    obs, info = env.reset()
    fm = FeatureMap(config, OBS_DIM, mode)
    fm.reset()
    size = int(config['reservoir']['size'])
    side = int(np.ceil(np.sqrt(size)))
    planner = SMWPlanner() if args.teacher and not args.replay else None
    replay = json.loads(Path(args.replay).read_text()) if args.replay else None
    played = []
    # neurons with the largest readout weights: shown as a scrolling trace
    if weights is not None:
        top = np.argsort(-np.abs(weights[:size]).max(axis=1))[:48]
    else:
        top = np.arange(48)
    trace = deque(maxlen=160)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix('.raw.mp4')
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (W, H))
    who = 'look-ahead teacher (emulator save-states)' if (args.teacher or replay) else 'frozen reservoir + linear readout'
    if run is None:
        seen = 'the reservoir only watches'
    else:
        seen = 'trained on this level' if args.level in trained_on else 'level NOT seen in training'
    step, action, status = 0, 0, ''
    end_hold = 0
    while True:
        f = fm.step(obs)
        state = f[:size]
        q = f @ weights if weights is not None else np.zeros(len(ACTION_NAMES))
        # grab the screen before planning: restoring a save-state blanks the frame buffer
        game = cv2.resize(env.render(), None, fx=GAME_SCALE, fy=GAME_SCALE, interpolation=cv2.INTER_NEAREST)
        if not status:
            if replay is not None:
                action = int(replay[min(step, len(replay) - 1)])
            elif planner is not None:
                if step % 4 == 0:
                    action = planner.act(env)
            else:
                action = int(np.argmax(q))
            played.append(action)

        img = np.full((H, W, 3), BG, np.uint8)
        img[60:60 + game.shape[0], 20:20 + game.shape[1]] = game
        text(img, 'Super Mario World (SNES, real ROM in an emulator)', (20, 28), 0.7, FG, 2)
        text(img, f'player: {who}   |   {args.level}: {seen}', (20, 50), 0.5, DIM)
        # status line under the game
        prog = info['progress']
        cv2.rectangle(img, (20, 520), (532, 534), (50, 50, 60), -1)
        cv2.rectangle(img, (20, 520), (20 + int(512 * prog), 534), (90, 200, 120), -1)
        text(img, f'progress {prog:4.0%}   coins {info["coins"]}   time {step / 30:5.1f}s', (20, 556), 0.55)
        text(img, f'buttons: {PRETTY[ACTION_NAMES[action]]}', (20, 584), 0.7, (255, 215, 90), 2)
        if status:
            text(img, status, (20, 620), 0.8, (120, 230, 120) if 'FINISH' in status else (240, 110, 110), 2)

        # panel 1: what the reservoir reads
        x1 = 560
        text(img, '1. input: tile grid read from RAM', (x1, 80), 0.55, ACCENT)
        draw_grid(img, obs[: ROWS * COLS * len(CHANNEL_NAMES)].reshape(ROWS, COLS, -1), x1, 92, 16)
        for k, name in enumerate(CHANNEL_NAMES):
            cv2.rectangle(img, (x1 + k * 64, 356), (x1 + k * 64 + 10, 366), CH_COLOURS[name], -1)
            text(img, name, (x1 + k * 64 + 14, 366), 0.38, DIM)
        text(img, f'+ speed, on ground, power-up = {OBS_DIM} inputs', (x1, 386), 0.4, DIM)

        # panel 2: reservoir state
        x2 = 900
        text(img, f'2. reservoir: {size} random neurons', (x2, 80), 0.55, ACCENT)
        sq = np.zeros(side * side, np.float32)
        sq[:size] = state / (np.abs(state).max() + 1e-6)
        cell = max(1, 300 // side)
        heat = cv2.resize(diverging(sq.reshape(side, side)), (side * cell, side * cell), interpolation=cv2.INTER_NEAREST)
        img[92:92 + heat.shape[0], x2:x2 + heat.shape[1]] = heat
        text(img, '1 pixel = 1 neuron (orange +, blue -)', (x2, 92 + heat.shape[0] + 18), 0.4, DIM)

        # panel 3: trace of the most influential neurons
        trace.append(state[top] / (np.abs(state[top]).max() + 1e-6))
        y3 = 420
        text(img, '3. 48 key neurons, last 5 s', (x1, y3), 0.55, ACCENT)
        tr = np.zeros((len(top), trace.maxlen), np.float32)
        arr = np.array(trace).T
        tr[:, -arr.shape[1]:] = arr
        tr_img = cv2.resize(diverging(tr), (320, 192), interpolation=cv2.INTER_NEAREST)
        img[y3 + 12:y3 + 12 + 192, x1:x1 + 320] = tr_img

        # panel 4: readout scores
        x4 = 900
        text(img, '4. readout: score per button combo', (x4, y3), 0.55, ACCENT)
        if weights is not None:
            lo, hi = float(q.min()), float(q.max())
            for i, name in enumerate(ACTION_NAMES):
                yy = y3 + 22 + i * 24
                frac = 0.0 if hi - lo < 1e-9 else (q[i] - lo) / (hi - lo)
                colour = (255, 215, 90) if i == action else (110, 110, 130)
                cv2.rectangle(img, (x4 + 130, yy - 12), (x4 + 130 + int(200 * frac), yy + 4), colour, -1)
                text(img, PRETTY[name], (x4, yy), 0.45, FG if i == action else DIM)
        else:
            text(img, '(teacher video: no readout)', (x4, y3 + 30), 0.45, DIM)
        text(img, 'no reward, no backprop: only the readout was fitted (ridge regression on the teacher)',
             (20, 700), 0.45, DIM)

        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if status:
            end_hold += 1
            if end_hold >= 2 * args.fps:
                break
            continue
        obs, _r, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            status = ('FINISHED the level!' if info['reached_goal'] else 'died' if info['died']
                      else 'stuck (no progress for 10 s)' if info['stagnated'] else 'out of time')
    writer.release()
    env.close()
    if planner is not None:
        out.with_suffix('.actions.json').write_text(json.dumps(played))
    if shutil.which('ffmpeg'):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '24', str(out)], check=True)
        raw.unlink()
    else:
        raw.rename(out)
    print(f'wrote {out}: {step} steps, {status}')


if __name__ == '__main__':
    main()
