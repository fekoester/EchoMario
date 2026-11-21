"""Does training only on 'clean' level parts (teacher competent, its choice rarely deadly) help the reservoir?
Offline: within-level and across-level learnability, fitted on all data vs clean data only.
Closed loop (--closed-loop): readouts fitted on all / clean data play groups A/B/C from 3 start delays."""
import json, sys
import numpy as np
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, 'scripts')
from smw_readout_lab import load_episodes, attach_streams, replay_features
from echomario.imitation import make_targets

CLEAN = ['YoshiIsland2', 'YoshiIsland4', 'Forest4', 'DonutPlains5', 'ChocolateIsland1@first', 'ChocolateIsland3@first',
         'DonutPlains1@first', 'DonutPlains2@first', 'DonutPlains3@first', 'Forest1@first']
data = Path('runs/smw/abc_v2/data')
run = json.loads((data.parent / 'results.json').read_text())
cfg = run['config']
eps = load_episodes(data)
cache = data.parent / 'feats_reservoir_baseline.npy'
if cache.exists():
    F = np.load(cache)
else:
    attach_streams(eps, cfg, data.parent / 'streams', 4)
    F = np.concatenate(replay_features(eps, cfg, 'reservoir'))
    np.save(cache, F)
scores = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
labels = np.concatenate([e['labels'] for e in eps])
y = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(scores, labels)])
deadly = np.nan_to_num(scores, neginf=-1e9) <= -1000
spec = np.concatenate([np.full(len(e['labels']), str(e.get('spec', e['level']))) for e in eps])
level = np.array([s.split('@')[0] for s in spec])
rnd = np.concatenate([np.full(len(e['labels']), e['iter']) for e in eps])
clean = np.isin(spec, CLEAN)
tried = np.isfinite(scores)
yt = np.where(tried, y, np.nan)
w_all = 1.0 + 4.0 * np.nan_to_num(np.nanmax(yt, 1) - np.nanmin(yt, 1))
i = np.arange(len(F))
print(f'{len(F)} samples, clean {clean.sum()} ({clean.mean():.0%})', flush=True)


def fit(mask, alpha):
    Fm, w = F[mask], w_all[mask]
    g = (Fm * w[:, None]).T.astype(np.float64) @ Fm / w.sum()
    r = alpha * np.trace(g) / g.shape[0]
    g[np.diag_indices(g.shape[0])] += r
    g[-1, -1] -= r
    return np.linalg.solve(g, (Fm * w[:, None]).T.astype(np.float64) @ y[mask] / w.sum()).astype(np.float32)


def evaluate(train_mask_fn, test_mask_fn, folds, alphas=(0.01, 0.1, 1.0, 10.0)):
    out = {}
    for a in alphas:
        pred = np.full(len(F), -1)
        for k in folds:
            W = fit(train_mask_fn(k), a)
            te = test_mask_fn(k)
            pred[te] = np.argmax(F[te] @ W, 1)
        m = pred >= 0
        out[a] = (y[i[m], pred[m]].mean(), deadly[i[m], pred[m]].mean(), m.sum())
    return out


const_clean = y[clean].mean(0).argmax()
print(f'constant action on clean parts: q {y[clean, const_clean].mean():.4f} deadly {deadly[clean, const_clean].mean():.4f}')
# 1. within levels (held-out DAgger rounds), scored on the clean parts
for name, src in (('fit on ALL data  ', np.ones(len(F), bool)), ('fit on CLEAN data', clean)):
    res = evaluate(lambda k: src & (rnd % 4 != k), lambda k: clean & (rnd % 4 == k), range(4))
    print(f'within clean levels, {name}: ' + '  '.join(f'a{a}: q {q:.4f} deadly {d:.3f}' for a, (q, d, n) in res.items()), flush=True)
# 2. across levels: each clean level held out in turn (all its parts), scored on it
clean_levels = sorted(set(level[clean]))
for name, src in (('fit on ALL other levels  ', np.ones(len(F), bool)), ('fit on CLEAN other levels', clean)):
    res = evaluate(lambda k: src & (level != clean_levels[k]), lambda k: clean & (level == clean_levels[k]), range(len(clean_levels)))
    print(f'across clean levels, {name}: ' + '  '.join(f'a{a}: q {q:.4f} deadly {d:.3f}' for a, (q, d, n) in res.items()), flush=True)

if '--closed-loop' in sys.argv:
    from smw_imitation import job, play
    split = json.loads(Path('runs/smw/split_abc.json').read_text())
    specs = {'full': split['full'], 'first_half': [f'{lv}@first' for lv in split['half']],
             'second_half': [f'{lv}@second' for lv in split['half']], 'unseen': split['unseen']}
    alpha = float(sys.argv[sys.argv.index('--closed-loop') + 1])
    readouts = {'ALL data': fit(np.ones(len(F), bool), alpha), 'CLEAN data': fit(clean, alpha)}
    with get_context('fork').Pool(6, maxtasksperchild=1) as pool:
        for name, W in readouts.items():
            np.save(data.parent / f"readout_{name.split()[0].lower()}_a{alpha}.npy", W)
            line = []
            for g, ss in specs.items():
                ep = pool.map(play, [job(s, cfg, 'reservoir', W, max_steps=2400, noop=n) for s in ss for n in (0, 13, 29)], chunksize=1)
                v = np.array([e['progress'] for e in ep])
                line.append(f'{g} {v.mean():.3f}+-{v.std(ddof=1) / np.sqrt(len(v)):.3f}')
            print(f'closed loop, readout fitted on {name} (alpha {alpha}): ' + '  '.join(line), flush=True)
