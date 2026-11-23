import json
import sys
from collections import deque

import numpy as np
from smw_imitation import grid_kwargs

from echomario.envs.smw_env import ACTION_NAMES, SMWEnv
from echomario.experts.smw_astar import AStarConfig, SMWAStar

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
level = sys.argv[1]
env = SMWEnv(level, max_steps=2400, **grid_kwargs(cfg)); env.reset(); env.set_render(False)
T = SMWAStar(AStarConfig()); import time; t0 = time.time()
hist = deque(maxlen=12)
for d in range(700):
    sc = T.scores(env); a = T.choose(sc); m = env.mario()
    hist.append((d, m['x'], m['y'], m['vx'], m['on_ground'], ACTION_NAMES[a], np.round(sc, 0).tolist(), T.last_stats['verdict_steps']))
    done = False
    for _ in range(4):
        _, _, te, tr, info = env.step(a)
        if te or tr: done = True; break
    if done: break
print(level, 'sec/decision', round((time.time() - t0) / (d + 1), 2), 'rescued', T.rescued, 'stall decisions', T.stall_decisions, 'goal', info['reached_goal'], 'progress', round(info['progress'], 3), 'died', info['died'], 'steps', env.track.steps, 'since_progress', env.track.since_progress)
for h in hist: print(*h)
