#!/usr/bin/env python
"""
Is the teacher's "deadly" verdict a function of what the student sees? Groups decisions whose observation (obs v2
tile input, which includes Mario's speed, sub-tile position and the sprites' speeds) is exactly identical on the
same level, in the v3 teacher data and DAgger rounds 1-2 of a smw_dagger.py run, and counts how often the copies
disagree on whether a button is deadly, and where those groups occur. (scripts/smw_label_noise.py measures how
much of that the teacher's own randomness explains.)

    PYTHONPATH=scripts .venv/bin/python scripts/smw_label_duplicates.py
"""
import json, hashlib, numpy as np
from pathlib import Path
from collections import defaultdict
from smw_readout_lab import attach_streams, load_episodes
data = Path('runs/smw/v3_all/data'); run = Path('runs/smw/dagger_v3_f0')
cfg = json.loads((data.parent / 'results.json').read_text())['config']
off = load_episodes(data); attach_streams(off, cfg, data.parent / 'streams', 3)
dag = [e for e in load_episodes(run / 'data') if e['iter'] in (1, 2)]; attach_streams(dag, cfg, run / 'streams', 3)
groups = defaultdict(list)
for e in off + dag:
    st = e['stream'].tocsr()
    for j, pos in enumerate(e['sample_pos']):
        row = st[pos]
        h = hashlib.md5(row.indices.tobytes() + row.data.astype(np.float32).tobytes() + str(e['level']).encode()).hexdigest()
        groups[h].append((e["scores"][j], e["file"], int(pos)))
def verdict(s):
    s = np.asarray(s, np.float64)
    fin = np.isfinite(s)
    return fin & (s > -1000), fin & (s <= -1000)


dis, agr, mixed_btn, tot_btn = [], [], 0, 0
for g in groups.values():
    if len(g) < 2:
        continue
    vs = [verdict(x[0]) for x in g]
    if not any(sf.any() and d.any() for sf, d in vs):
        continue
    S = np.array([sf for sf, _ in vs])
    both = np.array([np.isfinite(np.asarray(x[0], float)) for x in g]).all(0)  # buttons tried in every copy
    mixed = S[:, both].any(0) & ~S[:, both].all(0)
    mixed_btn += int(mixed.sum())
    tot_btn += int(both.sum())
    (dis if mixed.any() else agr).append(g)
print(f'{sum(len(g) for g in groups.values())} states, {sum(1 for g in groups.values() if len(g) > 1)} observations seen more than once')
print(f'repeated observations with a critical copy: {len(dis) + len(agr)}; copies disagree on some button: {len(dis)} '
      f'({len(dis) / max(1, len(dis) + len(agr)):.2f}); button verdicts that differ: {mixed_btn}/{tot_btn} = {mixed_btn / max(1, tot_btn):.3f}')
for name, gs in (('disagree', dis), ('agree', agr)):
    pos = np.array([x[2] for g in gs for x in g])
    print(f'{name}: {len(gs)} groups, {sum(len(g) for g in gs)} copies, median step {np.median(pos):.0f}, '
          f'all copies from one episode {np.mean([len({x[1] for x in g}) == 1 for g in gs]):.2f}')
