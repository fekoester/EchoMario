#!/usr/bin/env python
"""
Why do identical observations get different deadly verdicts? Takes 60 random pairs of v3 teacher decisions with
exactly identical input on the same level whose recorded verdicts differ, replays both states (checked against the
stored input), and rescores each with fresh v3 planners (rescue seeds 0 and 1). Result 2026-11-23: copy A vs copy
B under the same seed differ in 1 % of button verdicts, the same state under two seeds in 8.5 %: the recorded
disagreement is the teacher's random rescue search, not hidden game state.

    PYTHONPATH=scripts .venv/bin/python scripts/smw_label_rescore.py
"""
import json, hashlib, multiprocessing as mp, numpy as np
from pathlib import Path
from collections import defaultdict
from echomario.envs.smw_env import SMWEnv, grid_kwargs
from smw_imitation import parse_spec
from smw_readout_lab import attach_streams, load_episodes
data = Path('runs/smw/v3_all/data')
cfg = json.loads((data.parent / 'results.json').read_text())['config']
off = load_episodes(data); attach_streams(off, cfg, data.parent / 'streams', 1)
groups = defaultdict(list)
for e in off:  # teacher-only episodes: no rewinds, the log replays straight
    st = e['stream'].tocsr()
    for j, pos in enumerate(e['sample_pos']):
        row = st[pos]
        h = hashlib.md5(row.indices.tobytes() + row.data.astype(np.float32).tobytes() + str(e['level']).encode()).hexdigest()
        groups[h].append((np.asarray(e['scores'][j], float), e['file'], int(pos)))
def verdict(s):
    fin = np.isfinite(s); return tuple((fin & (s > -1000)).tolist()), fin
pairs_dis, pairs_agr = [], []
for g in groups.values():
    if len(g) < 2: continue
    for a in range(len(g)):
        for b in range(a + 1, len(g)):
            (sa, fa), (sb, fb) = verdict(g[a][0]), verdict(g[b][0])
            both = fa & fb
            crit = any(not x for x, t in zip(sa, both) if t) or any(not x for x, t in zip(sb, both) if t)
            if not crit: continue
            diff = any(x != y for x, y, t in zip(sa, sb, both) if t)
            (pairs_dis if diff else pairs_agr).append((g[a][1:], g[b][1:]))
rng = np.random.default_rng(0)
pick = lambda L, n: [L[i] for i in rng.choice(len(L), min(n, len(L)), replace=False)]
pairs_dis = pick(pairs_dis, 60)
import dataclasses
from echomario.experts.smw_planner import SMWPlanner, teacher_config
def work(pair):
    out = []
    for f, pos in pair:
        r = np.load(data / f'{f}.npz'); level, start = parse_spec(str(r['spec']))
        env = SMWEnv(level, max_steps=10 ** 9, stagnation_steps=10 ** 9, **grid_kwargs(cfg)); env.reset(options=start); env.set_render(False)
        for _ in range(int(r['noop'])): env.step(0)
        env.track.steps = env.track.since_progress = 0
        for a in r['actions'][:pos]: env.step(int(a))
        st = env.get_sim_state(); res = []
        for seed in (0, 1):
            env.set_sim_state(st)
            res.append(SMWPlanner(dataclasses.replace(teacher_config('v3'), rescue_seed=seed)).scores(env).astype(float))
        env.close(); out.append(res)
    return out
with mp.get_context('fork').Pool(3) as pool:
    res = pool.map(work, pairs_dis, chunksize=1)
def v(s): fin = np.isfinite(s); return fin & (s > -1000), fin
def flip(a, b):
    sa, fa = v(a); sb, fb = v(b); both = fa & fb; return int((sa != sb)[both].sum()), int(both.sum())
for name, fn in (('same state, seed 0 vs 1', lambda r: [flip(r[0][0], r[0][1]), flip(r[1][0], r[1][1])]),
                 ('copy A vs copy B, same seed', lambda r: [flip(r[0][0], r[1][0]), flip(r[0][1], r[1][1])]),
                 ('recorded A vs recorded B', None)):
    if fn is None: continue
    f = np.array([x for r in res for x in fn(r)]); print(f'{name}: {f[:, 0].sum()}/{f[:, 1].sum()} = {f[:, 0].sum() / f[:, 1].sum():.3f}')
