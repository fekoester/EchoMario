"""Same states, same reservoir features: how learnable are teacher v1's labels vs teacher v2's?
Within-level (held-out DAgger rounds) and across-level (held-out level), reservoir + ridge and raw input + ridge."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, 'scripts')
from smw_readout_lab import load_episodes
from echomario.imitation import make_targets

data = Path('runs/smw/abc_v2/data')
eps = load_episodes(data)
F_all = np.load(data.parent / 'feats_reservoir_baseline.npy')
offs = np.cumsum([0] + [len(e['labels']) for e in eps])
idx, s1, s2, spec, rnd = [], [], [], [], []
for e, o in zip(eps, offs):
    f = Path('runs/smw/relabel_v1') / f"{e['file']}.npy"
    if not f.exists():
        continue
    v1 = np.load(f)
    idx.append(np.arange(o, o + len(v1)))
    s1.append(v1)
    s2.append(e['scores'])
    spec += [str(e.get('spec', e['level']))] * len(v1)
    rnd += [e['iter']] * len(v1)
idx = np.concatenate(idx)
F = F_all[idx]
spec, rnd = np.array(spec), np.array(rnd)
level = np.array([s.split('@')[0] for s in spec])
print(f'{len(idx)} relabelled moments in {len(set(spec))} level parts', flush=True)


def targets(S):
    S = S.astype(np.float64)
    lab = np.array([int(np.argmax(np.where(np.isfinite(s), s, -np.inf))) for s in S])
    return np.stack([make_targets(s, a, 'q', 32.0) for s, a in zip(S, lab)]), lab, np.nan_to_num(S, neginf=-1e9) <= -1000


def fit(X, Y, alpha):
    g = X.T.astype(np.float64) @ X / len(X)
    r = alpha * np.trace(g) / g.shape[0]
    g[np.diag_indices(g.shape[0])] += r
    g[-1, -1] -= r
    return np.linalg.solve(g, X.T.astype(np.float64) @ Y / len(X))


S1, S2 = np.concatenate(s1), np.concatenate(s2)
for name, S in (('teacher v2 labels', S2), ('teacher v1 labels', S1)):
    Y, lab, dead = targets(S)
    i = np.arange(len(Y))
    for split, fold, ks in (('within levels', rnd % 4, range(4)),
                            ('across levels', np.array([sorted(set(level)).index(l) for l in level]), range(len(set(level))))):
        const = []
        best = None
        for a in (0.01, 0.1, 1.0, 10.0):
            pred = np.zeros(len(Y), np.int64)
            cpred = np.zeros(len(Y), np.int64)
            for k in ks:
                tr, te = fold != k, fold == k
                W = fit(F[tr], Y[tr], a)
                pred[te] = np.argmax(F[te] @ W, 1)
                cpred[te] = np.argmax(Y[tr].mean(0))
            q = Y[i, pred].mean()
            if best is None or q > best[0]:
                best = (q, a, dead[i, pred].mean())
            const = (Y[i, cpred].mean(), dead[i, cpred].mean())
        print(f'{name:18s} {split:14s} constant q {const[0]:.4f} deadly {const[1]:.3f} | reservoir q {best[0]:.4f} '
              f'(alpha {best[1]}) deadly {best[2]:.3f} | gain {best[0] - const[0]:+.4f}', flush=True)
Y1, lab1, _ = targets(S1)
Y2, lab2, _ = targets(S2)
print(f'teachers agree on the best action in {np.mean(lab1 == lab2):.1%} of moments; '
      f"v1's choice rated by v2: {Y2[np.arange(len(lab1)), lab1].mean():.3f}, v2's choice rated by v1: {Y1[np.arange(len(lab2)), lab2].mean():.3f}")
