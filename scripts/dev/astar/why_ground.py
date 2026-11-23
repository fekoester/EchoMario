"""Replay an episode to step t and explain ground_below(): which tile/sprite counts as ground under Mario.
   PYTHONPATH=scripts .venv/bin/python scripts/dev/astar/why_ground.py runs/smw/astar_all/data/iter0_Forest1.npz 128"""
import json
import sys

import numpy as np
from smw_imitation import grid_kwargs

from echomario.envs.smw_env import PLATFORM_SPRITES, SMWEnv
from echomario.envs.smw_tiles import TILE_CLASS

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
z = np.load(sys.argv[1], allow_pickle=True)
env = SMWEnv(str(z['level']), max_steps=10**9, stagnation_steps=10**9, **grid_kwargs(cfg))
env.reset()
env.set_render(False)
for a in z['actions'][:int(sys.argv[2])]:
    env.step(int(a))
w = env._wram()
m = env.mario(w)
print('mario', m, 'water flag', int(w[0x85]), 'layer2 interactive', bool(int(w[0x5B]) & 0x80))
cx = (m['x'] + 8) // 16
rows = np.arange((m['y'] + 24) // 16, 27)
ty, tx = np.meshgrid(rows, np.arange(cx - 1, cx + 2), indexing='ij')
t = env._tiles(w, ty, tx, False)
table = np.asarray(TILE_CLASS, np.int8)
cls = np.where(t >= 0, table[np.maximum(t, 0)], 0)
print('rows', rows.tolist(), 'cols', (cx - 1, cx, cx + 1))
print('tiles\n', t, '\nclass\n', cls)
print('sprites (slot, status, id):', [(i, int(w[0x14C8 + i]), int(w[0x9E + i]), int(w[0x9E + i]) in PLATFORM_SPRITES)
                                      for i in range(12) if int(w[0x14C8 + i])])
print('ground_below', env.ground_below(w))
