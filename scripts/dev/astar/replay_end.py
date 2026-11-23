"""Replay a recorded episode and show how it ended: last decisions with Mario's state and the teacher scores.
   PYTHONPATH=scripts .venv/bin/python scripts/dev/astar/replay_end.py runs/smw/astar_all/data/iter0_Forest1.npz"""
import json
import sys

import numpy as np
from smw_imitation import grid_kwargs, parse_spec

from echomario.envs.smw_env import ACTION_NAMES, SMWEnv

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
z = np.load(sys.argv[1], allow_pickle=True)
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 8
level = str(z['level'])
env = SMWEnv(level, max_steps=10**9, stagnation_steps=10**9, **grid_kwargs(cfg))
env.reset()
env.set_render(False)
sp = {int(p): k for k, p in enumerate(z['sample_pos'])}
rows = []
info = {}
for t, a in enumerate(z['actions']):
    if t in sp:
        m = env.mario()
        rows.append((t, m['x'], m['y'], m['vx'], m['vy'], m['on_ground'], ACTION_NAMES[int(a)],
                     np.round(z['scores'][sp[t]], 0).tolist()))
    _, _, te, tr, info = env.step(int(a))
    if te or tr:
        break
print(level, 'steps', len(z['actions']), 'progress', round(info.get('progress', 0), 3), 'died', info.get('died'),
      'goal', info.get('reached_goal'), 'final x', env.mario()['x'])
for r in rows[-n_show:]:
    print(*r)
