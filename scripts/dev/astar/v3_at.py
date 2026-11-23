import json
import sys

import numpy as np
from smw_imitation import grid_kwargs

from echomario.envs.smw_env import ACTION_NAMES, SMWEnv

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
level, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
z = np.load(f'runs/smw/v3_all/data/iter0_{level}.npz')
print('noop', z['noop'])
env = SMWEnv(level, max_steps=10**9, stagnation_steps=10**9, **grid_kwargs(cfg)); env.reset(); env.set_render(False)
for _ in range(int(z['noop'])): env.step(0)
prev = None
for k, a in enumerate(z['actions']):
    m = env.mario()
    if lo <= m['x'] <= hi:
        s = (ACTION_NAMES[a], m['x'], m['y'], m['vx'], m['vy'], m['on_ground'], int(env._wram()[0x13E4]))
        print(k, *s)
    env.step(int(a))
    if m['x'] > hi + 60: break
