#!/usr/bin/env python
"""
Video of the reservoir student (or the teacher) playing Super Mario World, with what is going on
inside: the game, the tile grid the reservoir reads from RAM, the 3000 reservoir neurons, a
trace of the most influential neurons and the readout's score for each button combination.

    python scripts/smw_video.py --run runs/smw/res3000 --level YoshiIsland2 --out outputs/smw_YoshiIsland2.mp4
    python scripts/smw_video.py --teacher --level YoshiIsland2 --out outputs/smw_teacher.mp4
    python scripts/smw_video.py --run runs/smw/rescue_tall --levels YoshiIsland2,DonutPlains1 --out outputs/smw_tour.mp4
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

from echomario.envs.smw_env import ACTION_NAMES, CHANNEL_NAMES, SMWEnv, grid_kwargs
from echomario.experts.smw_planner import SMWPlanner
from echomario.imitation import FeatureMap

W, H = 1280, 720
GAME_SCALE = 2  # 256x224 -> 512x448
BG = (18, 18, 24)
FG = (230, 230, 230)
DIM = (140, 140, 150)
ACCENT = (90, 200, 255)  # BGR-free: all colours here are RGB, converted at write time
CH_COLOURS = {'solid': (150, 105, 60), 'platform': (205, 165, 90), 'coin': (255, 215, 0),
              'enemy': (235, 70, 70), 'item': (90, 220, 120),
              # observation v2
              'slope': (120, 160, 60), 'water': (40, 90, 230), 'climb': (60, 180, 60), 'hazard': (255, 0, 255),
              'enemy_stompable': (255, 140, 0), 'enemy_harmful': (235, 30, 30), 'platform_sprite': (60, 200, 255),
              'projectile': (255, 255, 255), 'goal': (255, 255, 120)}
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


def draw_grid(img, grid: np.ndarray, x0: int, y0: int, cell: int, rows_above: int, cols_behind: int,
              names: tuple[str, ...] = CHANNEL_NAMES):
    rows, cols = grid.shape[:2]
    for r in range(rows):
        for c in range(cols):
            colour = (34, 34, 44)
            for k, name in enumerate(names):
                if name in CH_COLOURS and grid[r, c, k] > 0:
                    colour = CH_COLOURS[name]
            cv2.rectangle(img, (x0 + c * cell, y0 + r * cell), (x0 + (c + 1) * cell - 2, y0 + (r + 1) * cell - 2), colour, -1)
    mx, my = x0 + cols_behind * cell, y0 + (rows_above - 1) * cell
    cv2.rectangle(img, (mx, my), (mx + cell - 2, my + 2 * cell - 2), (240, 40, 40), 2)


def title_card(writer, frames: int, lines: list[tuple[str, float, tuple]]) -> None:
    img = np.full((H, W, 3), BG, np.uint8)
    y = H // 2 - 30 * len(lines)
    for s, scale, colour in lines:
        (tw, _), _ = cv2.getTextSize(s, FONT, scale, 2)
        text(img, s, ((W - tw) // 2, y), scale, colour, 2)
        y += int(70 * scale) + 10
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for _ in range(frames):
        writer.write(frame)


def draw_path(img, path: list[tuple[int, int]], goal_x: int, x0: int, y0: int, w: int, h: int, deaths: list) -> None:
    """Mario's route through the level so far: x across the strip, height up/down."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (34, 34, 44), -1)
    cv2.line(img, (x0 + w - 1, y0), (x0 + w - 1, y0 + h), (120, 230, 120), 2)  # goal

    def pt(x, y):
        return (x0 + int(w * min(1.0, max(0.0, x / goal_x))), y0 + int(h * min(1.0, max(0.0, y / 432))))

    if len(path) > 1:
        pts = np.array([pt(x, y) for x, y in path], np.int32)
        cv2.polylines(img, [pts], False, (255, 215, 90), 1, cv2.LINE_AA)
    for x, y in deaths:
        cx, cy = pt(x, y)
        cv2.line(img, (cx - 4, cy - 4), (cx + 4, cy + 4), (240, 80, 80), 2)
        cv2.line(img, (cx - 4, cy + 4), (cx + 4, cy - 4), (240, 80, 80), 2)
    if path:
        cv2.circle(img, pt(*path[-1]), 4, (240, 40, 40), -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--run', help='run dir of scripts/smw_imitation.py')
    p.add_argument('--readout', default='readout.npy')
    p.add_argument('--teacher', action='store_true', help='play the look-ahead teacher instead (reservoir only watches)')
    p.add_argument('--replay', help="actions JSON written by an earlier --teacher video (no re-planning)")
    p.add_argument('--level', default='YoshiIsland2')
    p.add_argument('--levels', default='', help='comma-separated levels played one after another in one video')
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
    hold = int(res['args'].get('plan_every', 4)) if res else 4
    levels = [lv for lv in args.levels.split(',') if lv] or [args.level]
    if args.replay and len(levels) > 1:
        p.error('--replay works with a single level')

    size = int(config['reservoir']['size'])
    side = int(np.ceil(np.sqrt(size)))
    # neurons with the largest readout weights: shown as a scrolling trace
    top = np.argsort(-np.abs(weights[:size]).max(axis=1))[:48] if weights is not None else np.arange(48)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix('.raw.mp4')
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (W, H))
    teacher = args.teacher or args.replay
    who = 'look-ahead teacher (emulator save-states)' if teacher else 'frozen reservoir + linear readout'
    summary = []
    for li, level in enumerate(levels):
        if run is None:
            seen = 'the reservoir only watches'
        else:
            seen = 'trained on this level' if level in trained_on else 'level NOT seen in training'
        if len(levels) > 1:
            title_card(writer, int(1.5 * args.fps), [
                (f'level {li + 1} of {len(levels)}', 0.7, DIM), (level, 1.4, FG),
                (seen, 0.8, (120, 230, 120) if 'trained' in seen else (255, 170, 80))])
        result = play_level(writer, level, args, config, mode, weights, size, side, top, hold, who, seen, len(levels) == 1)
        summary.append((level, *result, 'trained' in seen))
        print(f'{level}: {result[0]} steps, {result[1]}, progress {result[2]:.0%}', flush=True)
    if len(levels) > 1:
        lines = [('summary', 1.0, ACCENT)]
        for level, _steps, status, prog, trained in summary:
            how = 'finished' if 'FINISH' in status else f'{prog:.0%} of the level, {status}'
            lines.append((f'{level} ({"trained on" if trained else "never seen"}): {how}', 0.7, FG))
        title_card(writer, 4 * args.fps, lines)
    writer.release()
    if shutil.which('ffmpeg'):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '24', str(out)], check=True)
        raw.unlink()
    else:
        raw.rename(out)
    print(f'wrote {out}')


def play_level(writer, level, args, config, mode, weights, size, side, top, hold, who, seen, save_actions):
    env = SMWEnv(level, max_steps=args.max_steps, **grid_kwargs(config))
    obs, info = env.reset()
    fm = FeatureMap(config, env.obs_dim, mode)
    fm.reset()
    planner = SMWPlanner() if args.teacher and not args.replay else None
    replay = json.loads(Path(args.replay).read_text()) if args.replay else None
    played, path, deaths = [], [], []
    trace = deque(maxlen=160)
    step, action, status = 0, 0, ''
    end_hold = 0
    while True:
        f = fm.step(obs)
        state = f[:size]
        q = f @ weights if weights is not None else np.zeros(len(ACTION_NAMES))
        m = env.mario()
        if not status:
            path.append((m['x'], m['y']))
        # grab the screen before planning: restoring a save-state blanks the frame buffer
        game = cv2.resize(env.render(), None, fx=GAME_SCALE, fy=GAME_SCALE, interpolation=cv2.INTER_NEAREST)
        if not status:
            if replay is not None:
                action = int(replay[min(step, len(replay) - 1)])
            elif planner is not None:
                if step % 4 == 0:
                    action = planner.act(env)
            elif step % hold == 0:  # the student holds each choice for the teacher's planning interval
                action = int(np.argmax(q))
            played.append(action)

        img = np.full((H, W, 3), BG, np.uint8)
        img[60:60 + game.shape[0], 20:20 + game.shape[1]] = game
        text(img, 'Super Mario World (SNES, real ROM in an emulator)', (20, 28), 0.7, FG, 2)
        text(img, f'player: {who}   |   {level}: {seen}', (20, 50), 0.5, DIM)
        # status line under the game
        prog = info['progress']
        cv2.rectangle(img, (20, 520), (532, 534), (50, 50, 60), -1)
        cv2.rectangle(img, (20, 520), (20 + int(512 * prog), 534), (90, 200, 120), -1)
        text(img, f'progress {prog:4.0%}   coins {info["coins"]}   time {step / 30:5.1f}s', (20, 556), 0.55)
        text(img, f'buttons: {PRETTY[ACTION_NAMES[action]]}', (20, 584), 0.7, (255, 215, 90), 2)
        if status:
            text(img, status, (300, 584), 0.7, (120, 230, 120) if 'FINISH' in status else (240, 110, 110), 2)
        text(img, "Mario's path through the level (side view, goal on the right)", (20, 606), 0.42, DIM)
        draw_path(img, path, env.goal_x, 20, 614, 512, 72, deaths)

        # panel 1: what the reservoir reads
        x1 = 560
        text(img, '1. input: tile grid read from RAM', (x1, 80), 0.55, ACCENT)
        cell_px = max(8, min(16, 320 // env.cols, 256 // env.rows))
        draw_grid(img, obs[: env.screen_dim].reshape(env.rows, env.cols, -1), x1, 92, cell_px, env.rows_above,
                  env.cols_behind, env.channel_names)
        shown = [n for n in env.channel_names if n in CH_COLOURS]
        per_row = 5 if len(shown) <= 5 else 7
        short = {'enemy_stompable': 'stompable', 'enemy_harmful': 'harmful', 'platform_sprite': 'plat. sprite',
                 'projectile': 'projectile'}
        for k, name in enumerate(shown):
            xx, yy = x1 + (k % per_row) * (320 // per_row + 2), 356 + (k // per_row) * 14
            cv2.rectangle(img, (xx, yy - 9), (xx + 8, yy - 1), CH_COLOURS[name], -1)
            text(img, short.get(name, name), (xx + 11, yy), 0.32, DIM)
        extra = 'speeds, Mario state' if env.obs_version == 'v2' else 'speed, on ground, power-up'
        text(img, f'+ {extra} = {env.obs_dim} inputs', (x1, 400), 0.4, DIM)

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
                cv2.rectangle(img, (x4 + 165, yy - 12), (x4 + 165 + int(180 * frac), yy + 4), colour, -1)
                text(img, PRETTY[name], (x4, yy), 0.45, FG if i == action else DIM)
        else:
            text(img, '(teacher video: no readout)', (x4, y3 + 30), 0.45, DIM)
        text(img, 'no reward, no backprop: only the readout was fitted (ridge regression on the teacher)',
             (560, 700), 0.42, DIM)

        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if status:
            end_hold += 1
            if end_hold >= 2 * args.fps:
                break
            continue
        obs, _r, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            if info['died']:
                m = env.mario()
                deaths.append((m['x'], min(m['y'], 432)))
            status = ('FINISHED the level!' if info['reached_goal'] else 'died' if info['died']
                      else 'stuck (no progress for 10 s)' if info['stagnated'] else 'out of time')
    env.close()
    if planner is not None and save_actions:
        Path(args.out).with_suffix('.actions.json').write_text(json.dumps(played))
    return step, status, float(info['progress'])


if __name__ == '__main__':
    main()
