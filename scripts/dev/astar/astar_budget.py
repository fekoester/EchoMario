import json
import sys
import time

import numpy as np
from smw_imitation import grid_kwargs

from echomario.envs.smw_env import SMWEnv
from echomario.experts.smw_astar import AStarConfig, SMWAStar
from echomario.imitation import make_targets

cfg = json.load(open('runs/smw/v3_all/results.json'))['config']
level = sys.argv[1]
env = SMWEnv(level, max_steps=2400, **grid_kwargs(cfg)); env.reset(); env.set_render(False)
budgets = [1000, 2000, 4000, 8000]
T = {b: SMWAStar(AStarConfig(budget_steps=b)) for b in budgets}
rows = []
for d in range(400):
    if d % 6 == 0:
        r = {}
        for b in budgets:
            t0 = time.time(); sc = T[b].scores(env)
            r[b] = (sc, time.time() - t0, T[b].last_stats)
        ref = r[8000][0]
        qref = make_targets(ref, 0, 'q', 32.0); dref = ref <= -1000
        row = {'d': d}
        for b in budgets[:-1]:
            q = make_targets(r[b][0], 0, 'q', 32.0)
            row[b] = (float(np.abs(q - qref).mean()), int(((r[b][0] <= -1000) != dref).sum()), round(r[b][1], 1),
                      r[b][2]['verdict_steps'], r[b][2]['proved'])
        row['ref_t'] = round(r[8000][1], 1); row['ref_proved'] = r[8000][2]['proved']; row['ndeadly'] = int(dref.sum())
        rows.append(row); print(level, row, flush=True)
        sc = ref
    else:
        sc = T[2000].scores(env)
    a = T[2000].choose(sc)
    done = False
    for _ in range(4):
        _, _, te, tr, info = env.step(a)
        if te or tr: done = True; break
    if done: break
print('END', level, 'progress', round(info['progress'], 3), 'died', info['died'], 'goal', info['reached_goal'], 'decisions', d + 1, flush=True)
for b in budgets[:-1]:
    print('SUMMARY', level, b, 'mean |dq|', round(np.mean([r[b][0] for r in rows]), 4), 'verdict flips / state',
          round(np.mean([r[b][1] for r in rows]), 3), 'sec', round(np.mean([r[b][2] for r in rows]), 2), flush=True)
