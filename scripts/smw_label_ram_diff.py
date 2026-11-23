#!/usr/bin/env python
"""
Which RAM bytes differ between v3 teacher decisions with identical input whose deadly verdicts differ, compared
with pairs whose verdicts agree? Replays 300 + 300 pairs and checks every replayed observation against the stored
input. Result 2026-11-23 (after the variant replay fix): the pairs are the same moment (same position, 99 % from
one episode) and their RAM is almost identical; no byte separates disagreeing from agreeing pairs clearly.

    PYTHONPATH=scripts .venv/bin/python scripts/smw_label_ram_diff.py
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
pairs_dis, pairs_agr = pick(pairs_dis, 300), pick(pairs_agr, 300)
need = defaultdict(set)
for p in pairs_dis + pairs_agr:
    for f, pos in p: need[f].add(pos)
print(len(pairs_dis), 'disagreeing pairs,', len(pairs_agr), 'agreeing pairs,', len(need), 'episodes to replay', flush=True)
def work(item):
    f, poss = item
    r = np.load(data / f'{f}.npz'); level, start = parse_spec(str(r['spec']))
    env = SMWEnv(level, max_steps=10 ** 9, stagnation_steps=10 ** 9, **grid_kwargs(cfg)); env.reset(options=start); env.set_render(False)
    for _ in range(int(r['noop'])): env.step(0)
    out, acts, t = {}, r['actions'], 0
    for pos in sorted(poss):
        while t < pos: env.step(int(acts[t])); t += 1
        w = env._wram().copy(); out[pos] = (w, env._obs(w))
    env.close(); return f, out
with mp.get_context('fork').Pool(3) as pool:
    rams = dict(pool.map(work, sorted(need.items()), chunksize=1))
ep_by = {e['file']: e for e in off}
ok = [np.allclose(o, ep_by[f]['stream'][p].toarray()[0], atol=1e-5) for f, d in rams.items() for p, (w, o) in d.items()]
print(f'replayed observations equal to the stored ones: {np.mean(ok):.3f} of {len(ok)}', flush=True)
rams = {f: {p: w for p, (w, o) in d.items()} for f, d in rams.items()}
rd, ra = (np.mean([rams[a[0]][a[1]] != rams[b[0]][b[1]] for a, b in P], axis=0) for P in (pairs_dis, pairs_agr))
order = np.argsort(-(rd - ra))
print('addr   differs(disagree) differs(agree)')
for i in order[:30]: print(f'${i:04X}   {rd[i]:.2f}   {ra[i]:.2f}')
np.savez('runs/smw/dagger_v3_f0/ram_diff.npz', rd=rd, ra=ra)
def rate(pairs):
    return np.mean([rams[a[0]][a[1]] != rams[b[0]][b[1]] for a, b in pairs], axis=0)
import sys
def mx(r): return int(r[0x94]) | (int(r[0x95]) << 8)
for name, P in (('disagree', pairs_dis), ('agree', pairs_agr)):
    dx = np.array([mx(rams[a[0]][a[1]]) - mx(rams[b[0]][b[1]]) for a, b in P])
    vx = np.array([int(rams[a[0]][a[1]][0x7B]) != int(rams[b[0]][b[1]][0x7B]) for a, b in P])
    same_ep = np.array([a[0] == b[0] for a, b in P])
    print(f'{name}: |dx| median {np.median(np.abs(dx)):.0f} px, dx==0 {np.mean(dx == 0):.2f}, dx%16==0 {np.mean(dx % 16 == 0):.2f}, vx differs {vx.mean():.2f}, same episode {same_ep.mean():.2f}')
    print('   |dx| quantiles', np.percentile(np.abs(dx), [10, 25, 50, 75, 90]).round())
# check the replay alignment: re-encode the observation from RAM and compare to the stored stream row
