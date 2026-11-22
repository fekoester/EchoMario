#!/usr/bin/env python
"""
How noisy is the v3 teacher's deadly verdict? Replays 90 random critical states (a deadly and a safe button) from
DAgger round 2 episodes without rewinds and rescores each with fresh planners, rescue seeds 0, 0, 1, 2, and
compares with the recorded scores.

    PYTHONPATH=scripts .venv/bin/python scripts/smw_label_noise.py
"""
import dataclasses, json, multiprocessing as mp, numpy as np
from pathlib import Path
from echomario.envs.smw_env import SMWEnv
from echomario.experts.smw_planner import SMWPlanner, teacher_config
from smw_imitation import parse_spec
run = Path('runs/smw/dagger_v3_f0')
def verdict(s):
    s = np.asarray(s, np.float64); fin = np.isfinite(s); return fin & (s > -1000), fin & (s <= -1000)
cands = []
for f in sorted((run / 'data').glob('iter2_*.npz')):
    r = np.load(f)
    steps = r['steps']
    if len(steps) and np.all(np.diff(steps) == 1):  # no rewinds: the log replays straight
        for j, pos in enumerate(r['sample_pos']):
            sf, d = verdict(r['scores'][j])
            if sf.any() and d.any():
                cands.append((str(f), int(j)))
rng = np.random.default_rng(0)
pick = [cands[i] for i in rng.choice(len(cands), min(90, len(cands)), replace=False)]
def work(c):
    f, j = c
    r = np.load(f)
    level, start = parse_spec(str(r['spec']))
    env = SMWEnv(level, max_steps=100000); env.reset(options=start); env.set_render(False)
    for _ in range(int(r['noop'])): env.step(0)
    env.track.steps = env.track.since_progress = 0
    for a in r['actions'][:r['sample_pos'][j]]: env.step(int(a))
    state = env.get_sim_state()
    out = [r['scores'][j].tolist()]
    for seed in (0, 0, 1, 2):
        env.set_sim_state(state)
        p = SMWPlanner(dataclasses.replace(teacher_config('v3'), rescue_seed=seed))
        out.append(p.scores(env).astype(float).tolist())
    env.close()
    return out
with mp.get_context('fork').Pool(3) as pool:
    res = pool.map(work, pick, chunksize=1)
def flips(a, b):
    sa, da = verdict(a); sb, db = verdict(b); both = (sa | da) & (sb | db)
    return int((sa != sb)[both].sum()), int(both.sum())
for name, (i, k) in {'recorded vs seed 0': (0, 1), 'seed 0 vs seed 0': (1, 2), 'seed 0 vs seed 1': (1, 3), 'seed 1 vs seed 2': (3, 4)}.items():
    f = np.array([flips(r[i], r[k]) for r in res]); print(f'{name}: {f[:, 0].sum()}/{f[:, 1].sum()} button verdicts differ ({f[:, 0].sum() / f[:, 1].sum():.3f}); states with any flip {np.mean(f[:, 0] > 0):.2f}')
json.dump(res, open(run / 'noise_check.json', 'w'))
