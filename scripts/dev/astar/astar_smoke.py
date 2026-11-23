import json
import sys
import time

import numpy as np
from smw_imitation import grid_kwargs

from echomario.envs.smw_env import SMWEnv
from echomario.experts.smw_astar import AStarConfig, SMWAStar

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
level = sys.argv[1] if len(sys.argv) > 1 else 'YoshiIsland2'
env = SMWEnv(level, max_steps=2400, **grid_kwargs(cfg))
env.reset(); env.set_render(False)
t = AStarConfig()
T = SMWAStar(t)
t0 = time.time(); times, steps, proved = [], [], []
maxdx = 0; px = env.mario()['x']
for d in range(int(sys.argv[2]) if len(sys.argv) > 2 else 40):
    s0 = time.time()
    sc = T.scores(env)
    times.append(time.time() - s0); steps.append(T.last_stats['steps']); proved.append(T.last_stats['proved'])
    if d % 10 == 0:
        sc2 = T.scores(env)
        assert np.array_equal(sc, sc2), (sc, sc2)
        print(d, np.round(sc, 1), T.last_stats, f'{times[-1]:.2f}s', flush=True)
    a = T.choose(sc)
    done = False
    for _ in range(4):
        _, _, te, tr, info = env.step(a)
        x = env.mario()['x']; maxdx = max(maxdx, x - px); px = x
        if te or tr: done = True; break
    if done: break
print(level, 'decisions', len(times), 'progress', round(info['progress'], 3), 'died', info['died'], 'goal', info['reached_goal'],
      f'mean {np.mean(times):.2f}s/decision, steps {np.mean(steps):.0f}, proved {np.mean(proved):.2f}, max dx/step {maxdx}')
