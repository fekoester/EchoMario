#!/usr/bin/env python
"""Measures how every Map16 tile (0x000-0x1FF) behaves, by writing it next to Mario in a running level.

    python scripts/smw_tile_probe.py YoshiIsland2 tiles_YoshiIsland2.json

The result for YoshiIsland2 is src/echomario/envs/smw_tiles.py (lava 0x005 and side-solid tiles merged there)."""
import json, sys
import numpy as np
from echomario.envs.smw_env import SMWEnv

def probe_level(level, warmup):
    env = SMWEnv(level, max_steps=10**9, stagnation_steps=10**9); env.reset(); env.set_render(False)
    for a in warmup: env.step(a)
    mem = env.retro.unwrapped.data.memory
    m0 = env.mario(); root = env.get_sim_state(); assert m0['on_ground'], m0
    cx = (m0['x'] + 8) // 16; ground = (m0['y'] + 32) // 16
    up = np.zeros_like(env._masks[0]); up[env.retro.unwrapped.buttons.index('UP')] = 1
    def put(tx, ty, t):
        idx = (tx // 16) * 0x1B0 + ty * 16 + (tx % 16)
        mem.assign(0x7EC800 + idx, '|u1', t & 0xFF); mem.assign(0x7FC800 + idx, '|u1', t >> 8)
    out = {}
    for t in range(0x200):
        r = {}
        env.set_sim_state(root)                      # 1. stand on one row of t
        for dx in (-1, 0, 1): put(cx + dx, ground, t)
        y0, died = env.mario()['y'], False
        for _ in range(20):
            env.emulate(0); died |= env.mario()['anim'] == 9
        dy = env.mario()['y'] - y0
        r['stand'] = 'killed' if died and dy < 16 else 'falls' if dy >= 16 else 'sinks' if dy > 1 else 'holds'
        env.set_sim_state(root)                      # 2. walk into one t in front, at body height
        put(cx + 2, ground - 1, t)
        x0, died = env.mario()['x'], False
        for _ in range(30):
            env.emulate(1); died |= env.mario()['anim'] == 9
        r['walk'] = 'killed' if died else 'blocked' if env.mario()['x'] - x0 < 40 else 'passes'
        env.set_sim_state(root)                      # 3. t inside Mario's body, pressing up
        for dx in (-1, 0, 1):
            for dy_ in (1, 2): put(cx + dx, ground - dy_, t)
        c0 = env.mario()['coins']; water = climb = False
        for _ in range(16):
            env.em.set_button_mask(up, 0); env.em.step(); env.em.step()
            w = env._wram(); water |= bool(w[0x75]); climb |= bool(w[0x74])
        r.update(water=water, climb=climb, coin=env.mario()['coins'] > c0)
        out[t] = r
    env.close()
    return out

def classify(r):
    if r['water']: return 'water'
    if r['coin'] and r['stand'] != 'holds': return 'coin'
    if r['climb']: return 'climb'
    if r['stand'] == 'killed' or r['walk'] == 'killed': return 'hazard'
    if r['stand'] == 'holds' and r['walk'] == 'blocked': return 'solid'
    if r['stand'] == 'holds' and r['walk'] == 'passes': return 'platform'
    if r['stand'] == 'sinks': return 'slope'
    if r['stand'] == 'falls' and r['walk'] == 'blocked': return 'solid_side'
    if r['stand'] == 'falls': return 'empty'
    return 'other:' + json.dumps(r)

level = sys.argv[1]
warm = {'YoshiIsland2': [0] * 30, 'VanillaDome2': [0] * 60, 'Forest1': [0] * 30}[level]
res = probe_level(level, warm)
json.dump({hex(t): {**{k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in r.items()}, 'class': classify(r)}
           for t, r in res.items()}, open(sys.argv[2], 'w'), indent=0)
from collections import Counter
print(level, Counter(classify(r) for r in res.values()))
