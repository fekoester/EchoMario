"""Within-level learnability per level: reservoir + ridge fitted on other DAgger rounds of all levels,
scored per level (gain in q_chosen over the constant action, deadly share)."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, 'scripts')
from smw_readout_lab import load_episodes, attach_streams, replay_features
from echomario.imitation import make_targets
data, alpha = Path(sys.argv[1]), float(sys.argv[2])
cfg = json.loads((data.parent / 'results.json').read_text())['config']
eps = load_episodes(data)
attach_streams(eps, cfg, data.parent / 'streams', 4)
F = np.concatenate(replay_features(eps, cfg, 'reservoir'))
scores = np.concatenate([e['scores'] for e in eps]).astype(np.float64)
labels = np.concatenate([e['labels'] for e in eps])
y = np.stack([make_targets(s, int(a), 'q', 32.0) for s, a in zip(scores, labels)])
deadly = np.nan_to_num(scores, neginf=-1e9) <= -1000
spec = np.concatenate([np.full(len(e['labels']), str(e.get('spec', e['level']))) for e in eps])
fold = np.concatenate([np.full(len(e['labels']), e['iter'] % 4) for e in eps])
pred = np.zeros(len(F), np.int64); const = np.zeros(len(F), np.int64)
for k in range(4):
    tr, te = fold != k, fold == k
    g = F[tr].T.astype(np.float64) @ F[tr] / tr.sum(); r = alpha * np.trace(g) / g.shape[0]
    g[np.diag_indices(g.shape[0])] += r; g[-1, -1] -= r
    W = np.linalg.solve(g, F[tr].T.astype(np.float64) @ y[tr] / tr.sum())
    pred[te] = np.argmax(F[te] @ W, 1); const[te] = np.argmax(y[tr].mean(0))
i = np.arange(len(F))
print(f'{data}: alpha {alpha}')
for s in sorted(set(spec)):
    m = spec == s
    print(f'  {s:24s} n {m.sum():5d}  const q {y[i, const][m].mean():.3f}  reservoir q {y[i, pred][m].mean():.3f}  gain {y[i, pred][m].mean() - y[i, const][m].mean():+.3f}'
          f'  deadly {deadly[i, const][m].mean():.3f} -> {deadly[i, pred][m].mean():.3f}  teacher-chosen-is-deadly {deadly[i, labels][m].mean():.3f}')
