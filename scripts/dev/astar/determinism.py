"""Does a plan simulated with emulate()+_after_step() from a saved sim state give the same trajectory as the real
env.step() path? Uses teacher v3's recorded episode to reach a state, then replays its next actions both ways.
   PYTHONPATH=scripts .venv/bin/python scripts/dev/astar/determinism.py VanillaDome3 1100 88"""
import json
import sys

import numpy as np
from smw_imitation import grid_kwargs

from echomario.envs.smw_env import SMWEnv

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
level, x_at, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
z = np.load(f'runs/smw/v3_all/data/iter0_{level}.npz')
env = SMWEnv(level, max_steps=10**9, stagnation_steps=10**9, **grid_kwargs(cfg))
env.reset()
for _ in range(int(z['noop'])):
    env.step(0)
acts = [int(a) for a in z['actions']]
k = 0
while env.mario()['x'] < x_at:
    env.step(acts[k])
    k += 1
plan = acts[k:k + n]
s = env.get_sim_state()
real = []
for a in plan:  # the real path (with observations / rendering as during play)
    env.step(a)
    m = env.mario()
    real.append((m['x'], m['y']))
env.set_sim_state(s)
env.set_render(False)
sim = []
for a in plan:
    env.emulate(a)
    env._after_step(compute_obs=False)
    m = env.mario()
    sim.append((m['x'], m['y']))
env.set_sim_state(s)
again = []
for a in plan:
    env.step(a)
    m = env.mario()
    again.append((m['x'], m['y']))
d1 = next((i for i in range(n) if real[i] != sim[i]), None)
d2 = next((i for i in range(n) if real[i] != again[i]), None)
print(level, 'from step', k, 'x', x_at, '| real vs sim first diff at', d1, '| real vs real-again first diff at', d2)
if d1 is not None:
    print(' real', real[d1:d1 + 6], '\n sim ', sim[d1:d1 + 6])
