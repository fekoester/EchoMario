#!/usr/bin/env python
"""
Catalogue of every SMW level that can be started by a RAM write, beyond the 24 save states stable-retro ships.

Writing a level value to $0109 and setting the game mode $0100 to 0x0F (load level) from inside any level makes
the game load that level. For each value 0x00..0xFF this script loads the level, waits until Mario can be
controlled, saves a start state (gzip, like stable-retro's .state files) to ~/.cache/echomario/levelstates/
(ROM-derived, never committed) and records properties: a hash of the level's tile map (to find duplicates and
the 24 known levels), screens, vertical/horizontal, water, and how far a constant right+run+jump gets.

    .venv/bin/python scripts/smw_level_catalog.py            # -> runs/smw/level_catalog.json + contact sheet
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from echomario.envs import smw_env  # noqa: E402  (sets the fast core before stable_retro is imported)
from echomario.envs.smw_env import ALL_LEVELS, GAME, WRAM_OFFSET  # noqa: E402

STATE_DIR = Path(os.path.expanduser('~/.cache/echomario/levelstates'))


def wram(env) -> np.ndarray:
    return env.get_ram()[WRAM_OFFSET:]


def layout_hash(w: np.ndarray) -> str:
    """Map16 tile numbers (low bytes at $7EC800, high bytes at $7FC800) of layer 1."""
    return hashlib.md5(bytes(w[0xC800:0xC800 + 0x3800]) + bytes(w[0x1C800:0x1C800 + 0x3800])).hexdigest()[:12]


def ready(w: np.ndarray) -> bool:
    """In-level, nothing frozen, Mario in his normal state."""
    return int(w[0x100]) == 0x14 and int(w[0x71]) == 0 and int(w[0x9D]) == 0 and int(w[0x13FB]) == 0


def settle(env, idle, max_frames=3000, need=60) -> bool:
    run = 0
    for _ in range(max_frames):
        env.step(idle)
        run = run + 1 if ready(wram(env)) else 0
        if run >= need:
            return True
    return False


def probe_run(env, buttons, steps=600) -> dict:
    """Constant right + run (B held) + jump pressed every other 8 frames; how far does Mario get?"""
    names = env.unwrapped.buttons
    hold = np.zeros(len(names), np.uint8)
    for b in ('RIGHT', 'Y'):
        hold[names.index(b)] = 1
    jump = hold.copy()
    jump[names.index('B')] = 1
    w = wram(env)
    x0 = int(w[0x94]) | int(w[0x95]) << 8
    max_x, died, goal, mode_change = x0, False, False, False
    for i in range(steps * 4):
        env.step(jump if (i // 16) % 2 == 0 else hold)
        w = wram(env)
        if int(w[0x100]) != 0x14:
            mode_change = True
            break
        x = int(w[0x94]) | int(w[0x95]) << 8
        max_x = max(max_x, x)
        if int(w[0x71]) == 9 or int(w[0x96]) | int(w[0x97]) << 8 > 27 * 16 + 32:
            died = True
            break
        if int(w[0x1493]) or int(w[0x13C6]):
            goal = True
            break
    return {'x0': x0, 'max_x': max_x, 'died': died, 'goal': goal, 'left_level': mode_change}


def main() -> None:
    import stable_retro as retro
    from PIL import Image

    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/smw/level_catalog.json')
    args = p.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    env = retro.make(GAME, state='YoshiIsland2', render_mode='rgb_array', inttype=retro.data.Integrations.ALL)
    idle = np.zeros(len(env.unwrapped.buttons), np.uint8)
    mem = env.unwrapped.data.memory

    known = {}
    for lv in ALL_LEVELS:
        env.load_state(lv, inttype=retro.data.Integrations.ALL)
        env.reset()
        settle(env, idle, 600, 10)
        known[layout_hash(wram(env))] = lv
    env.load_state('YoshiIsland2', inttype=retro.data.Integrations.ALL)
    env.reset()
    for _ in range(10):
        env.step(idle)
    base = env.unwrapped.em.get_state()

    cat, thumbs = {}, []
    for t in range(256):
        env.unwrapped.em.set_state(base)
        mem.assign(0x7E0109, '|u1', t)
        mem.assign(0x7E0100, '|u1', 0x0F)
        ok = settle(env, idle)
        w = wram(env)
        h = layout_hash(w)
        state = env.unwrapped.em.get_state()
        info = {'value': t, 'name': f'L{t:02X}', 'ready': ok, 'hash': h, 'known_as': known.get(h),
                'screens': int(w[0x5D]), 'vertical': bool(int(w[0x5B]) & 1), 'water': int(w[0x85]),
                'layer2_interacts': bool(int(w[0x5B]) & 0x80)}
        frame = env.render() if hasattr(env, 'render') else None
        if ok:
            (STATE_DIR / f"{info['name']}.state").write_bytes(gzip.compress(state))
            info.update(probe_run(env, None))
        cat[info['name']] = info
        if frame is not None:
            thumbs.append(Image.fromarray(frame).resize((128, 112)))
        print(info, flush=True)

    first = {}
    for name, info in cat.items():
        if info['ready']:
            first.setdefault(info['hash'], name)
    for name, info in cat.items():
        info['duplicate_of'] = first.get(info['hash']) if first.get(info['hash']) != name else None
    Path(args.out).write_text(json.dumps(cat, indent=1))
    cols = 16
    sheet = Image.new('RGB', (128 * cols, 112 * ((len(thumbs) + cols - 1) // cols)))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * 128, (i // cols) * 112))
    sheet.save(Path(args.out).with_suffix('.png'))
    uniq = [n for n, i in cat.items() if i['ready'] and not i['duplicate_of']]
    print(f"{sum(i['ready'] for i in cat.values())} ready, {len(uniq)} distinct layouts, "
          f"{sum(1 for n in uniq if cat[n]['known_as'])} of them among the 24 known levels")


if __name__ == '__main__':
    main()
