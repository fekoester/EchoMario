"""Search from one state of a recorded episode with several budgets; show each button's verdict and where
its best plan ends (x, y, on ground, doomed?).
   PYTHONPATH=scripts .venv/bin/python scripts/dev/astar/probe_state.py runs/smw/astar_all/data/iter0_Forest1.npz 40"""
import json
import sys
import time

import numpy as np
from smw_imitation import grid_kwargs

from echomario.envs.smw_env import ACTION_NAMES, SMWEnv
from echomario.experts.smw_astar import AStarConfig, SMWAStar

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
z = np.load(sys.argv[1], allow_pickle=True)
t_stop = int(sys.argv[2])
budgets = [int(b) for b in (sys.argv[3] if len(sys.argv) > 3 else '1500,8000,40000').split(',')]
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}  # more AStarConfig fields, e.g. '{"deep_children": [...]}'
if 'deep_children' in extra:
    extra['deep_children'] = tuple(extra['deep_children'])
env = SMWEnv(str(z['level']), max_steps=10**9, stagnation_steps=10**9, **grid_kwargs(cfg))
env.reset()
env.set_render(False)
for a in z['actions'][:t_stop]:
    env.step(int(a))
m = env.mario()
print('state', t_stop, 'x', m['x'], 'y', m['y'], 'vx', m['vx'], 'ground', m['on_ground'])
orig = env.get_sim_state()
for b in budgets:
    T = SMWAStar(AStarConfig(**{'budget_steps': b, 'verdict_steps': max(400, b // 8), **extra}))
    env.set_sim_state(orig)
    t0 = time.time()
    sc = T.scores(env)
    print(f'budget {b}: {time.time() - t0:.1f}s stats {T.last_stats} stall {T._stall_x}')
    for a in np.argsort(-sc):
        path = T._paths.get(int(a))
        end = ''
        if path:
            env.set_sim_state(orig)
            for p in path:
                for _ in range(T.cfg.seg):
                    env.step(int(p))
            e = env.mario()
            end = f"plan {len(path)} segs -> x {e['x']} y {e['y']} ground {e['on_ground']} doomed {T._pit._doomed(env, m['y'])}"
        print(f'  {ACTION_NAMES[a]:15s} {sc[a]:8.1f} {end}')
    env.set_sim_state(orig)
