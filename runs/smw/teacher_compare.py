"""Teacher v1 vs final v2 on identical episodes (level spec + start delay)."""
import json, sys
import numpy as np
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, 'scripts')
from smw_imitation import job, play, reservoir_config
cfg = reservoir_config(3000, 0)
jobs = []
# 1. old 7 levels, start 0: final v2 (v1 known from runs/smw/tall_data round 0)
for lv in ['YoshiIsland1', 'YoshiIsland2', 'DonutPlains1', 'Forest1', 'DonutPlains5', 'YoshiIsland4', 'DonutPlains4']:
    jobs.append(('old7', 'v2', lv, 0))
# 2. the A/B round-0 episodes of runs/smw/abc_v2 (final v2 already played them): v1 on the same spec + start delay
for f in sorted(Path('runs/smw/abc_v2/data').glob('iter0_*.npz')):
    with np.load(f) as z:
        jobs.append(('abc', 'v1', str(z['spec']), int(z['noop'])))
def run(a):
    group, teacher, spec, noop = a
    e = play(job(spec, cfg, 'reservoir', None, beta=1.0, label=True, teacher=teacher, noop=noop, max_steps=2400))
    return group, teacher, spec, noop, round(e['progress'], 3), e['reached_goal'], e['died']
with get_context('fork').Pool(5, maxtasksperchild=1) as p:
    res = p.map(run, jobs, chunksize=1)
json.dump(res, open('runs/smw/teacher_compare.json', 'w'), indent=0)
v1_old = json.load(open('runs/smw/tall_data/results.json'))['iterations'][0]['collect']['per_level']
v2_old_window = json.load(open('runs/smw/v2_data/results.json'))['iterations'][0]['collect']['per_level']
abc0 = json.load(open('runs/smw/abc_v2/results.json'))['iterations'][0]['collect']['per_level']
print('old 7 levels, start 0:  level  v1 | v2 (first, window check) | v2 final')
for g, t, spec, n, pr, goal, died in res:
    if g == 'old7': print(f'  {spec:16s} {v1_old.get(spec)} | {v2_old_window.get(spec)} | {pr}')
print('A/B round-0 episodes (same spec + start):  spec  v1 | v2 final')
for g, t, spec, n, pr, goal, died in res:
    if g == 'abc': print(f'  {spec:24s} noop {n:2d}: {pr} | {abc0.get(spec)}')
